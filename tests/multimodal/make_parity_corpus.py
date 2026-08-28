# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate the differential-parity corpus for image-decode backends.

Every case here is an input class where a GPU decoder and Pillow can plausibly
disagree. A backend is correct only if, for each case, it either matches Pillow
within tolerance or declines the image so Pillow handles it. Cases are generated
rather than checked in so the corpus stays reviewable and license-clean.
"""

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).parent / "fixtures"


def _photo(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Photo-like content: smooth gradients plus texture, so JPEG behaves realistically."""
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        128 + 100 * np.sin(xx / max(w, 1) * 6 + yy / max(h, 1) * 2),
        128 + 100 * np.cos(yy / max(h, 1) * 5),
        128 + 90 * np.sin((xx + yy) / max(w + h, 1) * 9),
    ], -1)
    coarse = (rs.rand(max(h // 16, 1) + 1, max(w // 16, 1) + 1, 3) * 60).astype(np.uint8)
    tex = np.asarray(Image.fromarray(coarse).resize((w, h), Image.BILINEAR), np.float32) - 30
    return np.clip(base + tex, 0, 255).astype(np.uint8)


def _save(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def build() -> dict[str, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    cases: dict[str, dict] = {}

    def add(name, data, expect, why):
        (OUT / name).write_bytes(data)
        cases[name] = {"expect": expect, "why": why, "bytes": len(data)}

    rgb = Image.fromarray(_photo(64, 48))

    # --- baseline: these SHOULD go to the GPU ---
    add("baseline_rgb.jpg", _save(rgb, "JPEG", quality=92, subsampling=2),
        "gpu", "plain 8-bit YCbCr 4:2:0 JPEG, the target workload")
    add("baseline_444.jpg", _save(rgb, "JPEG", quality=92, subsampling=0),
        "gpu", "4:4:4 JPEG must behave the same as 4:2:0")
    add("progressive.jpg", _save(rgb, "JPEG", quality=92, progressive=True),
        "gpu_or_fallback", "progressive JPEG: HW engines cannot do it; must not corrupt")
    add("restart_markers.jpg", _save(rgb, "JPEG", quality=92, restart_marker_blocks=2),
        "gpu_or_fallback", "restart markers change entropy segmentation")

    # --- EXIF orientation: applied exactly once, in the right container ---
    for tag in range(2, 9):
        ex = Image.Exif()
        ex[0x0112] = tag
        add(f"exif_jpeg_orient{tag}.jpg", _save(rgb, "JPEG", quality=95, exif=ex),
            "match_pillow", f"JPEG EXIF orientation {tag} stored in info['exif']")
        # TIFF stores Orientation in tag_v2, NOT info['exif'] -- the class the
        # prior implementation silently dropped.
        add(f"exif_tiff_orient{tag}.tif", _save(rgb, "TIFF", tiffinfo={274: tag}),
            "match_pillow", f"TIFF EXIF orientation {tag} stored in tag_v2, not info['exif']")

    # --- transparency: the composite must match Pillow's, including when Pillow does nothing ---
    keyed = Image.fromarray(_photo(64, 48, 1))
    key_px = keyed.getpixel((0, 0))
    add("png_truecolor_trns.png", _save(keyed, "PNG", transparency=key_px),
        "match_pillow",
        "truecolor PNG with a tRNS colour key: source mode is already RGB, so Pillow's "
        "mode-convert short-circuits and does NOT composite")
    pal = rgb.convert("P", palette=Image.ADAPTIVE, colors=64)
    add("png_palette_trns.png", _save(pal, "PNG", transparency=0),
        "match_pillow", "palette PNG with tRNS index")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray((_photo(64, 48, 2)[:, :, 0]).astype(np.uint8)))
    add("png_rgba.png", _save(rgba, "PNG"), "match_pillow", "true alpha channel")
    add("webp_rgba.webp", _save(rgba, "WEBP", lossless=True), "match_pillow", "WebP with alpha")
    add("png_la.png", _save(rgba.convert("LA"), "PNG"), "match_pillow", "LA (grey+alpha)")

    # --- colour spaces and depth ---
    add("grayscale.jpg", _save(rgb.convert("L"), "JPEG", quality=92),
        "match_pillow", "1-component JPEG promoted to RGB")
    add("cmyk_adobe.jpg", _save(rgb.convert("CMYK"), "JPEG", quality=95),
        "match_pillow", "Adobe CMYK JPEG (transform 0); inverted samples")
    add("png_16bit.png", _save(Image.fromarray((_photo(64, 48, 3).astype(np.uint16) * 257)[:, :, 0], "I;16"), "PNG"),
        "pillow", "16-bit: allow_any_depth is off, must not be silently rescaled")
    add("tiff_multipage.tif",
        _save(rgb, "TIFF", save_all=True, append_images=[Image.fromarray(_photo(64, 48, 4))]),
        "match_pillow", "multi-page TIFF: both must expose the FIRST frame")

    # --- animation: still-image decoders have no compositing semantics ---
    frames = [rgb, Image.fromarray(_photo(64, 48, 5))]
    add("animated.webp", _save(frames[0], "WEBP", save_all=True, append_images=frames[1:], duration=100),
        "pillow", "animated WebP: no still-image compositing semantics")
    add("animated.png", _save(frames[0], "PNG", save_all=True, append_images=frames[1:], duration=100),
        "pillow", "APNG: same")

    # --- malformed / hostile: must ERROR, never silently produce filler pixels ---
    full = _save(Image.fromarray(_photo(256, 256, 6)), "JPEG", quality=90)
    add("truncated.jpg", full[: len(full) // 2], "error",
        "truncated JPEG: Pillow raises; a GPU decoder may return grey filler")
    add("truncated_no_eoi.jpg", full[:-2], "error", "JPEG missing its EOI marker")
    add("garbage.jpg", b"\xff\xd8" + b"\x00" * 512, "error", "JPEG SOI with garbage body")
    add("empty.jpg", b"", "error", "empty payload")

    # --- other containers that must at least not corrupt ---
    add("bmp.bmp", _save(rgb, "BMP"), "match_pillow", "BMP (CPU plugin)")
    add("ppm.ppm", _save(rgb, "PPM"), "match_pillow", "PNM (CPU plugin)")
    add("j2k.j2k", _save(rgb, "JPEG2000", irreversible=False), "match_pillow", "raw J2K codestream")
    add("jp2.jp2", _save(rgb, "JPEG2000", irreversible=True), "match_pillow", "JP2 container")

    (OUT / "manifest.json").write_text(json.dumps(cases, indent=1, sort_keys=True))
    return cases


if __name__ == "__main__":
    built = build()
    print(f"{len(built)} cases -> {OUT}")
    for name, meta in sorted(built.items()):
        print(f"  {meta['expect']:<18} {name:<28} {meta['bytes']:>7}B  {meta['why']}")
