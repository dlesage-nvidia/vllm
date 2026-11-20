# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import numpy as np


@dataclass(frozen=True)
class ResizeAxisTable:
    starts: np.ndarray
    counts: np.ndarray
    coefficients: np.ndarray
    precision: int
    error_bounds: np.ndarray


_CACHE_BYTES = 16 * 1024 * 1024
_table_cache: OrderedDict[tuple[int, int], ResizeAxisTable] = OrderedDict()
_cache_bytes = 0
_cache_lock = Lock()


@dataclass(frozen=True)
class _Interval:
    lower: float
    upper: float

    @classmethod
    def point(cls, value: float) -> "_Interval":
        return cls(float(value), float(value))

    @classmethod
    def enclosure(cls, lower: float, upper: float) -> "_Interval":
        return cls(
            float(np.nextafter(np.float32(lower), np.float32(-np.inf))),
            float(np.nextafter(np.float32(upper), np.float32(np.inf))),
        )

    def add(self, other: "_Interval") -> "_Interval":
        return self.enclosure(self.lower + other.lower, self.upper + other.upper)

    def subtract(self, other: "_Interval") -> "_Interval":
        return self.enclosure(self.lower - other.upper, self.upper - other.lower)

    def multiply(self, other: "_Interval") -> "_Interval":
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return self.enclosure(min(products), max(products))

    def divide(self, other: "_Interval") -> "_Interval":
        if other.lower <= 0 <= other.upper:
            raise ArithmeticError("Resize coefficient denominator includes zero")
        quotients = (
            self.lower / other.lower,
            self.lower / other.upper,
            self.upper / other.lower,
            self.upper / other.upper,
        )
        return self.enclosure(min(quotients), max(quotients))

    def absolute(self) -> "_Interval":
        if self.lower <= 0 <= self.upper:
            return _Interval(0, max(-self.lower, self.upper))
        return _Interval(
            min(abs(self.lower), abs(self.upper)),
            max(abs(self.lower), abs(self.upper)),
        )


def _cubic_weight(distance: float) -> float:
    distance = abs(distance)
    if distance < 1:
        return (1.5 * distance - 2.5) * distance * distance + 1
    if distance < 2:
        return ((-0.5 * distance + 2.5) * distance - 4) * distance + 2
    return 0.0


def _weight_enclosure(distance: _Interval) -> _Interval:
    absolute = distance.absolute()
    pieces = []
    if absolute.lower < 1:
        clipped = _Interval(absolute.lower, min(absolute.upper, 1))
        first = _Interval.point(1.5).multiply(clipped).subtract(_Interval.point(2.5))
        pieces.append(first.multiply(clipped).multiply(clipped).add(_Interval.point(1)))
    if absolute.upper >= 1 and absolute.lower < 2:
        clipped = _Interval(max(absolute.lower, 1), min(absolute.upper, 2))
        first = _Interval.point(-0.5).multiply(clipped).add(_Interval.point(2.5))
        second = first.multiply(clipped).subtract(_Interval.point(4))
        pieces.append(second.multiply(clipped).add(_Interval.point(2)))
    if absolute.upper >= 2:
        pieces.append(_Interval.point(0))
    return _Interval(
        min(piece.lower for piece in pieces), max(piece.upper for piece in pieces)
    )


def _cv_coefficients(source: int, target: int, coordinate: int) -> dict[int, _Interval]:
    scale = np.float32(np.float32(source) / np.float32(target))
    center = np.float32(
        np.float32(np.float32(np.float32(coordinate) + np.float32(0.5)) * scale)
        - np.float32(0.5)
    )
    anchor = math.floor(float(center))
    fraction = np.float32(center - np.float32(anchor))
    if anchor >= source:
        anchor, fraction = source - 1, np.float32(0)
    dilation = max(np.float32(1), scale)
    support = np.float32(np.float32(2) * dilation)
    first = max(0, anchor + math.ceil(float(np.float32(fraction - support))))
    end = min(source, anchor + math.floor(float(np.float32(fraction + support))) + 1)
    intervals = {}
    total = _Interval.point(0)
    for source_index in range(first, end):
        distance = (
            _Interval.point(np.float32(source_index - anchor))
            .subtract(_Interval.point(fraction))
            .divide(_Interval.point(dilation))
        )
        interval = _weight_enclosure(distance)
        intervals[source_index] = interval
        total = total.add(interval)
    return {
        source_index: interval.divide(total)
        for source_index, interval in intervals.items()
    }


def _coefficient_bound(
    source: int,
    target: int,
    coordinate: int,
    start: int,
    coefficients: np.ndarray,
    precision: int,
) -> float:
    try:
        intervals = _cv_coefficients(source, target, coordinate)
    except ArithmeticError:
        return math.inf
    reference = {
        start + tap: float(coefficient) / (1 << precision)
        for tap, coefficient in enumerate(coefficients)
    }
    difference = 0.0
    magnitude = 0.0
    for source_index in set(intervals) | set(reference):
        interval = intervals.get(source_index, _Interval.point(0))
        coefficient = reference.get(source_index, 0.0)
        difference += max(
            abs(interval.lower - coefficient), abs(interval.upper - coefficient)
        )
        magnitude += max(abs(interval.lower), abs(interval.upper))
    operation_roundoff = 2 * len(intervals) * 2**-24
    if operation_roundoff >= 1:
        return math.inf
    bound = 255 * (
        difference + magnitude * operation_roundoff / (1 - operation_roundoff)
    )
    return float(np.nextafter(np.float32(bound), np.float32(np.inf)))


def _make_axis_table(source: int, target: int) -> ResizeAxisTable:
    scale = float(source) / float(target)
    support = 2.0 * scale if scale >= 1.0 else 2.0
    storage_taps = math.ceil(support) * 2 + 1
    inverse_scale = 1.0 / scale if scale >= 1.0 else 1.0
    starts = np.empty(target, dtype=np.int64)
    counts = np.empty(target, dtype=np.int64)
    weights = np.zeros((target, storage_taps), dtype=np.float64)
    maximum_positive_weight = 0.0
    for coordinate in range(target):
        center = scale * (coordinate + 0.5)
        start = max(int(center - support + 0.5), 0)
        count = min(int(center + support + 0.5), source) - start
        count = min(max(count, 0), storage_taps)
        starts[coordinate], counts[coordinate] = start, count
        total = 0.0
        for tap in range(count):
            weight = _cubic_weight((tap + start - center + 0.5) * inverse_scale)
            weights[coordinate, tap] = weight
            total += weight
        if total != 0:
            for tap in range(count):
                weight = float(weights[coordinate, tap]) / total
                weights[coordinate, tap] = weight
                maximum_positive_weight = max(maximum_positive_weight, weight)
    precision = 0
    while precision < 22:
        next_value = int(0.5 + maximum_positive_weight * (1 << (precision + 1)))
        if next_value >= 1 << 15:
            break
        precision += 1
    if precision == 0:
        raise ValueError("Resize coefficient precision must be positive")
    scaled = weights * (1 << precision)
    quantized = np.trunc(np.where(scaled < 0, scaled - 0.5, scaled + 0.5))
    if quantized.min() < -32768 or quantized.max() > 32767:
        raise OverflowError("Resize coefficients exceed signed int16")
    coefficients = quantized.astype(np.int16)
    error_bounds = np.asarray(
        [
            _coefficient_bound(
                source,
                target,
                coordinate,
                int(starts[coordinate]),
                coefficients[coordinate, : counts[coordinate]],
                precision,
            )
            for coordinate in range(target)
        ],
        dtype=np.float32,
    )
    for values in (starts, counts, coefficients, error_bounds):
        values.flags.writeable = False
    return ResizeAxisTable(starts, counts, coefficients, precision, error_bounds)


def resize_axis_table(source: int, target: int) -> ResizeAxisTable:
    """Cache the TorchVision CPU uint8 coefficients and released-CV error bounds.

    Bounds apply to the generic public v0.19 `TvResize` FP32 antialiased cubic
    path with exactly one changed axis. Outward FP32 intervals enclose both
    contracted and uncontracted filter arithmetic. The bound is 255 times the
    coefficient L1 difference plus a conservative FP32 dot-product roundoff
    bound, relative to the unrounded fixed-point reference dot product.

    Coefficients use ordered FP64 construction, signed-int16 quantization and
    the reference's per-axis precision. They are not Pillow's fixed-22 weights.
    The host cache is byte-bounded and contains no GPU tensors or decoded pixels.
    """
    global _cache_bytes
    if any(
        type(extent) is not int or not 1 <= extent <= 2**31 - 1
        for extent in (source, target)
    ):
        raise ValueError("Resize extents must fit positive signed int32")
    key = (source, target)
    with _cache_lock:
        if key in _table_cache:
            _table_cache.move_to_end(key)
            return _table_cache[key]
    table = _make_axis_table(source, target)
    table_bytes = sum(
        values.nbytes
        for values in (
            table.starts,
            table.counts,
            table.coefficients,
            table.error_bounds,
        )
    )
    if table_bytes <= _CACHE_BYTES:
        with _cache_lock:
            if key in _table_cache:
                _table_cache.move_to_end(key)
                return _table_cache[key]
            while _table_cache and _cache_bytes + table_bytes > _CACHE_BYTES:
                _, previous = _table_cache.popitem(last=False)
                _cache_bytes -= sum(
                    values.nbytes
                    for values in (
                        previous.starts,
                        previous.counts,
                        previous.coefficients,
                        previous.error_bounds,
                    )
                )
            _table_cache[key] = table
            _cache_bytes += table_bytes
    return table
