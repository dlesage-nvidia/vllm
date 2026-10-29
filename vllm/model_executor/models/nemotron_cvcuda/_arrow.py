# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read-only, owner-retaining Pillow RGBX views through the Arrow C ABI."""

import ctypes
from dataclasses import dataclass

import numpy as np


class ArrowSchema(ctypes.Structure):
    pass


class ArrowArray(ctypes.Structure):
    pass


ArrowSchema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", ctypes.c_char_p),
    ("flags", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowSchema))),
    ("dictionary", ctypes.POINTER(ArrowSchema)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]
ArrowArray._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowArray))),
    ("dictionary", ctypes.POINTER(ArrowArray)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


@dataclass
class OwnedRGBXView:
    image: object
    schema_capsule: object
    array_capsule: object
    pixels: np.ndarray


class OwnerRetainingPixels(np.ndarray):
    def __array_finalize__(self, parent):
        self.arrow_owners = getattr(parent, "arrow_owners", None)


def capsule_pointer(capsule, name, structure):
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(capsule, name)
    if not pointer:
        raise ValueError("Arrow capsule pointer is null")
    return ctypes.cast(pointer, ctypes.POINTER(structure)).contents


def export_rgbx(image, *, expected_size=(1920, 1080)):
    if image.mode != "RGB" or image.size != expected_size:
        raise ValueError(
            f"Arrow export requires a {expected_size[0]}x{expected_size[1]} RGB image"
        )
    try:
        schema_capsule, array_capsule = image.__arrow_c_array__()
    except ValueError as error:
        if str(error) != "Image rows are not contiguous in memory.":
            raise
        serialized = image.tobytes("raw", "RGBX")
        pixels = (
            np.frombuffer(serialized, dtype=np.uint8)
            .reshape(image.height, image.width, 4)
            .view(OwnerRetainingPixels)
        )
        pixels.arrow_owners = (image, serialized)
        pixels.setflags(write=False)
        return OwnedRGBXView(image, None, None, pixels)
    schema = capsule_pointer(schema_capsule, b"arrow_schema", ArrowSchema)
    array = capsule_pointer(array_capsule, b"arrow_array", ArrowArray)
    if (
        not schema.release
        or schema.format != b"+w:4"
        or schema.n_children != 1
        or not schema.children
        or not schema.children[0]
        or schema.dictionary
        or not array.release
        or array.length != image.width * image.height
        or array.null_count != 0
        or array.offset != 0
        or array.n_children != 1
        or not array.children
        or not array.children[0]
        or array.dictionary
    ):
        raise ValueError("unsupported Pillow RGBX Arrow schema or array")
    child_schema = schema.children[0].contents
    child_array = array.children[0].contents
    expected_bytes = image.width * image.height * 4
    if (
        not child_schema.release
        or child_schema.format != b"C"
        or child_schema.n_children != 0
        or child_schema.dictionary
        or not child_array.release
        or child_array.length != expected_bytes
        or child_array.offset != 0
        or child_array.null_count != 0
        or child_array.n_children != 0
        or child_array.n_buffers != 2
        or not child_array.buffers
        or not child_array.buffers[1]
        or child_array.dictionary
    ):
        raise ValueError("unsupported Pillow RGBX Arrow child buffer")
    buffer = (ctypes.c_uint8 * expected_bytes).from_address(child_array.buffers[1])
    pixels = (
        np.ctypeslib.as_array(buffer)
        .reshape(image.height, image.width, 4)
        .view(OwnerRetainingPixels)
    )
    pixels.arrow_owners = (image, schema_capsule, array_capsule)
    pixels.setflags(write=False)
    return OwnedRGBXView(image, schema_capsule, array_capsule, pixels)
