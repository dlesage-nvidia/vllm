# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential parity: any GPU image backend vs Pillow, on real decoded pixels.

The contract under test is deliberately weak, so it can be pointed at any
implementation:

    decode_batch(datas, image_mode) -> list[np.ndarray | None]

``None`` at a position means "I decline this image"; the caller must then use
Pillow, so declining is always correct. Returning pixels that differ from
Pillow's is never correct. Raising for an input Pillow accepts, or accepting an
input Pillow rejects, is a divergence too.

Crucially the reference is produced by *really decoding* with Pillow. A fake
decoder cannot be substituted here: a fake that returns ``Image.open(data)``
would carry container metadata a real GPU decoder never has, which is exactly
how metadata-transfer bugs survive a green test suite.
"""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

FIXTURES = Path(__file__).parent / "fixtures"

# Same tolerance regime the codebase uses for lossy-codec comparisons: JPEG-like
# codecs may differ by a small amount per channel; lossless ones must be exact.
LOSSY_FORMATS = {"JPEG", "JPEG2000", "WEBP"}
LOSSY_MAX_ABS_DIFF = 2
LOSSY_MAX_MEAN_DIFF = 0.5


def pillow_reference(data: bytes, image_mode: str | None = "RGB",
                     background: tuple[int, int, int] = (255, 255, 255)):
    """Reproduce vLLM's Pillow decode path exactly: open, EXIF, load, convert."""
    image = Image.open(io.BytesIO(data))
    source_format = image.format
    image = ImageOps.exif_transpose(image)
    image.load()
    if image_mode is not None and image.mode != image_mode:
        if image_mode == "RGB" and (
            image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
        ):
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            flat = Image.new("RGB", image.size, background)
            flat.paste(image, mask=image.split()[3])
            image = flat
        else:
            image = image.convert(image_mode)
    return source_format, np.asarray(image)


def load_manifest() -> dict[str, dict]:
    manifest = FIXTURES / "manifest.json"
    if not manifest.exists():
        pytest.skip("parity corpus not generated; run make_parity_corpus.py")
    return json.loads(manifest.read_text())


def assert_matches(name: str, source_format: str | None,
                   reference: np.ndarray, actual: np.ndarray) -> None:
    assert actual.shape == reference.shape, (
        f"{name}: shape {actual.shape} != Pillow {reference.shape}. A transposed "
        "or differently-sized raster means orientation or sub-image selection diverged."
    )
    assert actual.dtype == reference.dtype, f"{name}: dtype {actual.dtype} != {reference.dtype}"
    diff = np.abs(actual.astype(np.int32) - reference.astype(np.int32))
    if source_format in LOSSY_FORMATS:
        assert diff.max() <= LOSSY_MAX_ABS_DIFF, (
            f"{name}: max |diff| {diff.max()} > {LOSSY_MAX_ABS_DIFF}")
        assert diff.mean() <= LOSSY_MAX_MEAN_DIFF, (
            f"{name}: mean |diff| {diff.mean():.3f} > {LOSSY_MAX_MEAN_DIFF}")
    else:
        assert diff.max() == 0, f"{name}: lossless codec differs by up to {diff.max()}"


def check_case(name: str, data: bytes, decode_batch, image_mode="RGB") -> str:
    """Return 'gpu', 'declined', or raise AssertionError. Used by the tests below."""
    try:
        source_format, reference = pillow_reference(data, image_mode)
        pillow_failed = None
    except Exception as exc:  # noqa: BLE001 - the reference is allowed to reject
        source_format, reference, pillow_failed = None, None, exc

    try:
        results = decode_batch([data], image_mode)
    except Exception as exc:  # noqa: BLE001
        if pillow_failed is not None:
            return "declined"  # both reject: consistent
        raise AssertionError(
            f"{name}: backend raised {type(exc).__name__} for an image Pillow decodes fine"
        ) from exc

    assert len(results) == 1, f"{name}: expected 1 result, got {len(results)}"
    actual = results[0]

    if pillow_failed is not None:
        # Pillow rejects this input. Silently producing pixels is the dangerous
        # case: the model would be fed filler instead of the request failing.
        assert actual is None, (
            f"{name}: Pillow rejects this input ({type(pillow_failed).__name__}: "
            f"{pillow_failed}) but the backend returned a {actual.shape} raster. "
            "Malformed input must not be decoded to filler."
        )
        return "declined"

    if actual is None:
        return "declined"
    assert_matches(name, source_format, reference, np.asarray(actual))
    return "gpu"


# --------------------------------------------------------------------------
# The tests. `decode_batch` is supplied by a fixture the implementation wires up.
# --------------------------------------------------------------------------

def test_every_corpus_case_takes_its_expected_route(decode_batch):
    """Pin the route, not just the pixels.

    Matching-or-declining alone is satisfiable by declining everything, so the
    corpus would silently stop testing the GPU path if eligibility narrowed.
    Each fixture's manifest entry states the route it must take.
    """
    manifest = load_manifest()
    wrong = []
    for name in sorted(manifest):
        expected = manifest[name]["expect"]
        data = (FIXTURES / name).read_bytes()
        actual = check_case(name, data, decode_batch)
        if expected == "gpu" and actual != "gpu":
            wrong.append(f"{name}: expected the GPU path, was declined")
        elif expected in ("pillow", "error") and actual != "declined":
            wrong.append(f"{name}: expected Pillow/error, took the GPU path")
    assert not wrong, "\n".join(wrong)


def test_every_corpus_case_matches_or_declines(decode_batch):
    manifest = load_manifest()
    outcomes, failures = {}, []
    for name in sorted(manifest):
        data = (FIXTURES / name).read_bytes()
        try:
            outcomes[name] = check_case(name, data, decode_batch)
        except AssertionError as exc:
            failures.append(str(exc))
    assert not failures, "\n\n".join(failures)
    # The corpus is only meaningful if the happy path actually used the GPU.
    assert outcomes.get("baseline_rgb.jpg") == "gpu", (
        "plain JPEG was declined - the backend is not exercising the GPU path at all")


def test_batch_order_and_positional_fallback(decode_batch):
    """A mixed batch must return results in input order, with declines in place."""
    manifest = load_manifest()
    names = [n for n in ("baseline_rgb.jpg", "animated.webp", "baseline_444.jpg",
                         "truncated.jpg", "grayscale.jpg") if n in manifest]
    datas = [(FIXTURES / n).read_bytes() for n in names]
    results = decode_batch(datas, "RGB")
    assert len(results) == len(datas)
    for name, data, actual in zip(names, datas, results):
        single = decode_batch([data], "RGB")[0]
        if single is None:
            assert actual is None, f"{name}: declined alone but decoded in a batch"
        else:
            assert actual is not None, f"{name}: decoded alone but declined in a batch"
            np.testing.assert_array_equal(
                np.asarray(actual), np.asarray(single),
                err_msg=f"{name}: batch result differs from singleton result")


def test_one_bad_image_does_not_poison_its_batch(decode_batch):
    """A malformed input must not change what its neighbours decode to."""
    good = (FIXTURES / "baseline_rgb.jpg").read_bytes()
    bad = (FIXTURES / "garbage.jpg").read_bytes()
    alone = decode_batch([good], "RGB")[0]
    assert alone is not None
    mixed = decode_batch([good, bad, good], "RGB")
    assert len(mixed) == 3
    assert mixed[1] is None, "malformed input must decline, not produce pixels"
    for index in (0, 2):
        assert mixed[index] is not None, "a neighbour was poisoned by the bad input"
        np.testing.assert_array_equal(np.asarray(mixed[index]), np.asarray(alone))


@pytest.mark.parametrize("tag", [2, 3, 4, 5, 6, 7, 8])
def test_tiff_exif_orientation(decode_batch, tag):
    """TIFF stores Orientation in tag_v2, not info['exif'].

    A backend that re-applies orientation by copying the source ``info`` dict
    onto a freshly constructed image silently drops it here, and for tags 5-8
    also returns a transposed raster of the wrong shape.
    """
    name = f"exif_tiff_orient{tag}.tif"
    check_case(name, (FIXTURES / name).read_bytes(), decode_batch)


def test_truecolor_png_with_transparency_key(decode_batch):
    """Pillow does NOT composite when the source mode already equals the target.

    A backend that decodes to RGBA because "the image has transparency" and then
    composites the background diverges on exactly this input.
    """
    name = "png_truecolor_trns.png"
    check_case(name, (FIXTURES / name).read_bytes(), decode_batch)


@pytest.mark.parametrize("name", ["truncated.jpg", "truncated_no_eoi.jpg",
                                  "garbage.jpg", "empty.jpg"])
def test_malformed_input_is_never_decoded_to_filler(decode_batch, name):
    check_case(name, (FIXTURES / name).read_bytes(), decode_batch)


@pytest.mark.parametrize("image_mode", ["RGB", "RGBA", None])
def test_image_modes(decode_batch, image_mode):
    for name in ("baseline_rgb.jpg", "png_rgba.png", "grayscale.jpg"):
        if not (FIXTURES / name).exists():
            continue
        check_case(f"{name}[{image_mode}]", (FIXTURES / name).read_bytes(),
                   decode_batch, image_mode=image_mode)
