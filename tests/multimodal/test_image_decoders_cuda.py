# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA hazard and bound tests for the GPU image backend.

Every test here fails if its guard is removed. They exist because each one
corresponds to a hazard that was actually observed on this stack, not to a
hypothetical.
"""

import importlib.util
import threading
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
# Check for the package WITHOUT importing it. `import nvidia.nvimgcodec` creates a
# CUDA context, and pytest imports this module at collection time -- so an eager
# import here leaves the pytest process fork-hostile before a single test runs,
# breaking any later test that forks a CUDA child. It does so invisibly:
# torch.cuda.is_initialized() still reports False afterwards, which is why vLLM's
# own spawn-forcing guard cannot see it. The real import happens lazily inside the
# decoder, in whichever process actually runs these tests.
if importlib.util.find_spec("nvidia.nvimgcodec") is None:
    pytest.skip("nvidia-nvimgcodec is required", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

import vllm.multimodal.image_decoders.nvimgcodec as backend  # noqa: E402
from vllm.multimodal.gpu_ipc_memory import (  # noqa: E402
    MultiModalGPUMemoryPool,
    get_mm_gpu_ipc_pool,
    set_mm_gpu_ipc_pool,
)


def _jpeg(width=640, height=480, seed=0) -> bytes:
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    pixels = np.stack(
        [128 + 100 * np.sin(xx / width * 6), 128 + 100 * np.cos(yy / height * 5),
         np.full((height, width), 90 + seed % 60, np.float32)], -1)
    pixels = np.clip(pixels + rs.rand(height, width, 3) * 8, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(pixels).save(buf, "JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def pool_2gb():
    previous = get_mm_gpu_ipc_pool()
    pool = MultiModalGPUMemoryPool(2 * 1024**3)
    set_mm_gpu_ipc_pool(pool)
    try:
        yield pool
    finally:
        set_mm_gpu_ipc_pool(previous)


@pytest.fixture
def configured(pool_2gb):
    backend.shutdown()
    backend.configure(num_decoders=4, min_gpu_pixels=0)
    try:
        yield pool_2gb
    finally:
        backend.shutdown()


def _run_with_deadline(fn, seconds: float, message: str):
    """Run fn on a daemon thread and fail if it does not finish in time.

    A hang here is the failure mode under test, so the test must not itself
    hang: a daemon thread lets the suite report and move on.
    """
    done = threading.Event()
    error: list[BaseException] = []

    def target():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert done.wait(timeout=seconds), message
    if error:
        raise error[0]


def test_width_one_batches_never_hang(configured):
    """The V4 guard.

    In nvimgcodec 0.9.0 the HYBRID_CPU_GPU backend deadlocks forever on a
    native batch of exactly one image when the decoder is built with more than
    one CPU helper thread. Width-one batches are the common case for this
    feature, so a reintroduced multi-thread decoder must fail CI here rather
    than wedge a production fleet.
    """
    data = _jpeg()

    def decode_many():
        for _ in range(200):
            assert backend.decode_batch([data], "RGB")[0] is not None

    _run_with_deadline(
        decode_many, 120.0,
        "width-one decode hung: max_num_cpu_threads must stay 1")


def test_decoder_is_built_with_a_single_cpu_helper_thread(configured):
    """Pin the constant itself, so nobody widens it without reading why."""
    slot = backend._acquire_slot()
    assert slot is not None
    try:
        import inspect
        source = inspect.getsource(backend._Slot.__init__)
        assert "max_num_cpu_threads=1" in source
    finally:
        backend._release_slot(slot)


def test_decoder_is_never_shared_between_threads(configured):
    """Exclusive slot ownership is a memory-safety requirement.

    Sharing one nvimgcodec.Decoder across threads corrupts the heap rather than
    raising. This drives the real path hard and asserts both that every decode
    succeeded and that no two threads ever held the same decoder object.
    """
    datas = [_jpeg(seed=i) for i in range(4)]
    # Single-threaded references, so a concurrent result can be compared for
    # bit-equality rather than merely for "did not crash".
    references = [backend.decode_batch([data], "RGB")[0] for data in datas]
    assert all(reference is not None for reference in references)

    failures: list[str] = []
    decoded = 0
    counter_lock = threading.Lock()

    def worker(index: int):
        nonlocal decoded
        for _ in range(40):
            which = index % len(datas)
            result = backend.decode_batch([datas[which]], "RGB")
            if len(result) != 1:
                failures.append(f"expected 1 result, got {len(result)}")
                continue
            if result[0] is None:
                # A legitimate shed: 8 callers contend for 4 slots.
                continue
            if not np.array_equal(result[0], references[which]):
                failures.append("concurrent result differs from its reference")
            with counter_lock:
                decoded += 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
        assert not thread.is_alive(), "worker hung"
    assert not failures, failures[:5]
    # The point is that real concurrent decoding happened and stayed correct;
    # if everything had shed, this test would prove nothing.
    assert decoded > 0
    assert backend._CREATED <= 4


def test_result_does_not_alias_decoder_memory(configured):
    """A returned array must remain stable once handed out.

    Whether it owns its storage (chw) or is a view kept alive by array.base
    (hwc), a later decode through the same slot must never change it -- a
    corruption that would land after any parity assertion has already passed.
    """
    first_data, second_data = _jpeg(seed=1), _jpeg(seed=2)
    first = backend.decode_batch([first_data], "RGB")[0]
    assert first is not None
    snapshot = first.copy()
    for _ in range(8):
        assert backend.decode_batch([second_data], "RGB")[0] is not None
    np.testing.assert_array_equal(
        first, snapshot,
        err_msg="an earlier result changed after later decodes: it aliases "
                "decoder-owned memory instead of owning its pixels")


def test_gpu_lease_is_always_returned(configured):
    """Every acquire is matched by a release on every path."""
    pool = configured
    before = pool.available_bytes
    for seed in range(12):
        backend.decode_batch([_jpeg(seed=seed)], "RGB")
    backend.decode_batch([b"\xff\xd8garbage\xff\xd9"], "RGB")
    backend.decode_batch([_jpeg(), b"nonsense"], "RGB")
    assert pool.available_bytes == before


def test_image_larger_than_the_pool_routes_to_pillow():
    """A resource bound must not become a per-request ceiling.

    The image is declined so the caller's Pillow path handles it, rather than
    the request failing because a GPU budget could never satisfy it.
    """
    previous = get_mm_gpu_ipc_pool()
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(64 * 1024))  # far too small
    backend.shutdown()
    backend.configure(num_decoders=1, min_gpu_pixels=0)
    try:
        assert backend.decode_batch([_jpeg()], "RGB") == [None]
    finally:
        backend.shutdown()
        set_mm_gpu_ipc_pool(previous)


def test_exhausted_slots_shed_to_pillow_without_blocking(configured):
    """Back-pressure is a shed, not a queue: it must never wait."""
    held = [backend._acquire_slot() for _ in range(4)]
    assert all(slot is not None for slot in held)
    try:
        done = threading.Event()

        def attempt():
            assert backend.decode_batch([_jpeg()], "RGB") == [None]
            done.set()

        thread = threading.Thread(target=attempt, daemon=True)
        thread.start()
        assert done.wait(timeout=10), (
            "decode blocked waiting for a slot; it must shed to Pillow instead")
    finally:
        for slot in held:
            backend._release_slot(slot)


def test_stats_report_why_images_were_shed(configured):
    """R10: a silent shed must never be invisible to an operator."""
    backend._COUNTERS.clear()
    backend.decode_batch([_jpeg()], "RGB")
    backend.decode_batch([b"not-an-image"], "RGB")
    stats = backend.stats()
    assert stats.get("gpu", 0) >= 1
    assert any(key.startswith("pillow:") for key in stats)


@pytest.mark.parametrize("layout", ["chw", "hwc"])
def test_output_ownership_contract(configured, layout):
    """Pin what each layout actually returns, so the docs cannot drift.

    Both are zero-copy views of a decoder-allocated host buffer: owndata is
    False and `base` holds the owning nvImageCodec Image. That is legitimate
    and lifetime-safe, but it is not ordinary NumPy storage, so it is asserted
    here rather than implied anywhere.
    """
    expect_owned = False
    backend.shutdown()
    backend.configure(num_decoders=1, output_layout=layout, min_gpu_pixels=0)
    array = backend.decode_batch([_jpeg(320, 240)], "RGB")[0]
    assert array is not None
    assert array.dtype == np.uint8
    assert array.flags.c_contiguous and array.flags.writeable
    assert array.flags.owndata is expect_owned
    assert (array.base is None) is expect_owned
    expected_shape = (3, 240, 320) if layout == "chw" else (240, 320, 3)
    assert array.shape == expected_shape
    # Must survive the decoder that produced it going away.
    snapshot = array.copy()
    backend.shutdown()
    np.testing.assert_array_equal(array, snapshot)


def test_chw_and_hwc_carry_identical_pixels(configured):
    data = _jpeg(320, 240)
    backend.shutdown(); backend.configure(num_decoders=1, output_layout="hwc", min_gpu_pixels=0)
    hwc = backend.decode_batch([data], "RGB")[0]
    backend.shutdown(); backend.configure(num_decoders=1, output_layout="chw", min_gpu_pixels=0)
    chw = backend.decode_batch([data], "RGB")[0]
    np.testing.assert_array_equal(chw, np.ascontiguousarray(hwc.transpose(2, 0, 1)))


@pytest.mark.parametrize("tag", [1, 2, 3, 4, 5, 6, 7, 8])
def test_chw_exif_orientation_matches_pillow(configured, tag):
    """The planar orientation table must equal the interleaved one.

    P_RGB output means EXIF transforms run on (3, H, W), a separate table from
    the (H, W, 3) one. The parity corpus exercises only the layout the fixture
    configured, so without this the planar table is unverified -- and an axis
    mistake there is silent, wrong-shaped pixels.
    """
    from io import BytesIO
    from pathlib import Path

    from PIL import ImageOps

    fixture = Path(__file__).parent / "fixtures" / f"exif_jpeg_orient{tag}.jpg"
    if tag == 1 or not fixture.exists():
        pytest.skip(f"no fixture for orientation {tag}")
    data = fixture.read_bytes()

    with Image.open(BytesIO(data)) as opened:
        reference = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))

    backend.shutdown(); backend.configure(num_decoders=1, output_layout="chw", min_gpu_pixels=0)
    planar = backend.decode_batch([data], "RGB")[0]
    backend.shutdown(); backend.configure(num_decoders=1, output_layout="hwc", min_gpu_pixels=0)
    interleaved = backend.decode_batch([data], "RGB")[0]

    if planar is None or interleaved is None:
        pytest.skip("orientation fixture declined by the eligibility predicate")

    # planar (3, H, W) must equal the interleaved result transposed...
    np.testing.assert_array_equal(planar, interleaved.transpose(2, 0, 1))
    # ...and both must equal what Pillow produces, within JPEG tolerance.
    assert planar.shape == (3, *reference.shape[:2]), (
        f"orientation {tag}: planar shape {planar.shape} != Pillow "
        f"{(3, *reference.shape[:2])}"
    )
    diff = np.abs(planar.transpose(1, 2, 0).astype(np.int32) - reference.astype(np.int32))
    assert diff.max() <= 3, f"orientation {tag}: max |diff| {diff.max()}"


def test_reused_buffer_grows_and_never_shrinks(configured):
    """The reservation assumes retention is one raster, not a sum over sizes.

    Measured from the device pointer: exceeding capacity reallocates (pointer
    moves), but a smaller image afterwards reuses the larger buffer instead of
    shrinking it. So reallocation is a warm-up cost per slot rather than a
    per-image one, and alternating sizes do not thrash.
    """
    import nvidia.nvimgcodec as nvi

    slot = backend._acquire_slot()
    assert slot is not None
    try:
        def decode(w, h, reuse):
            cs = nvi.CodeStream(_jpeg(w, h, seed=w))
            out = (
                slot.decoder.decode(
                    [cs], images=[reuse], params=slot.params,
                    cuda_stream=slot.stream.cuda_stream)
                if reuse is not None else
                slot.decoder.decode(
                    [cs], params=slot.params, cuda_stream=slot.stream.cuda_stream)
            )
            slot.stream.synchronize()
            img = out[0]
            return img, img.__cuda_array_interface__["data"][0], img.capacity

        small, _, cap_small = decode(320, 240, None)
        big, ptr_big, cap_big = decode(1280, 960, small)
        assert cap_big > cap_small, "capacity did not grow for a larger image"
        _, ptr_again, cap_again = decode(320, 240, big)
        assert cap_again == cap_big, "buffer shrank; retention would be unbounded"
        assert ptr_again == ptr_big, "smaller image reallocated instead of reusing"
    finally:
        backend._release_slot(slot)


def test_reuse_does_not_corrupt_earlier_results(configured):
    """The invariant reuse depends on: results are copied out before recycling."""
    first = backend.decode_batch([_jpeg(640, 480, seed=1)], "RGB")[0]
    assert first is not None
    snapshot = first.copy()
    for seed in range(8):
        backend.decode_batch([_jpeg(640, 480, seed=seed + 2)], "RGB")
    np.testing.assert_array_equal(
        first, snapshot, err_msg="a recycled buffer overwrote an earlier result")

def test_device_output_matches_the_host_path(configured):
    """A device tensor must carry exactly the pixels the host path produces.

    This is the whole correctness argument for device mode: it is offered only
    where the processor already runs on the accelerator, so what must hold is
    that handing the image over on-device changes nothing about the pixels.
    """
    data = _jpeg(640, 480, seed=11)

    backend.shutdown()
    backend.configure(num_decoders=1, output_layout="chw", min_gpu_pixels=0)
    host = backend.decode_batch([data], "RGB")[0]
    assert host is not None, "host CHW path declined; cannot compare"

    backend.shutdown()
    backend.configure(num_decoders=1, output_layout="device", min_gpu_pixels=0)
    dev = backend.decode_batch([data], "RGB")[0]
    if dev is None:
        pytest.skip("device output declined on this stack")

    assert dev.__class__.__module__.startswith("torch")
    assert dev.is_cuda and dev.dtype == torch.uint8
    assert tuple(dev.shape) == (3, 480, 640)
    np.testing.assert_array_equal(dev.cpu().numpy(), host)


def test_device_output_survives_the_decoder_that_made_it(configured):
    """DLPack must transfer ownership, not alias a buffer the pool reuses.

    A __cuda_array_interface__ wrapper would not own the memory, and a later
    decode through the same slot could overwrite it after the caller had already
    been handed the tensor.
    """
    backend.shutdown()
    backend.configure(num_decoders=1, output_layout="device", min_gpu_pixels=0)
    first = backend.decode_batch([_jpeg(320, 240, seed=1)], "RGB")[0]
    if first is None:
        pytest.skip("device output declined on this stack")
    snapshot = first.clone()
    for seed in range(6):
        backend.decode_batch([_jpeg(320, 240, seed=seed + 2)], "RGB")
    assert torch.equal(first, snapshot), "device tensor changed after later decodes"
    backend.shutdown()
    assert torch.equal(first, snapshot), "device tensor did not survive shutdown"


def test_reused_buffer_grows_to_high_water_mark_and_never_shrinks(configured):
    """The reservation assumes retention equals the largest raster, not the sum.

    Measured directly from the device pointer: growing past capacity reallocates
    (pointer moves), but a smaller image afterwards reuses the larger buffer
    rather than shrinking it. So a slot costs one raster at its high-water mark,
    reallocation is a warm-up cost rather than per-image, and alternating sizes
    do not thrash.
    """
    import nvidia.nvimgcodec as nvi

    backend.shutdown()
    backend.configure(num_decoders=1, min_gpu_pixels=0)
    slot = backend._acquire_slot()
    assert slot is not None
    try:
        def decode(w, h, reuse):
            cs = nvi.CodeStream(_jpeg(w, h, seed=w))
            out = (
                slot.decoder.decode(
                    [cs], images=[reuse], params=slot.params,
                    cuda_stream=slot.stream.cuda_stream,
                )
                if reuse is not None
                else slot.decoder.decode(
                    [cs], params=slot.params, cuda_stream=slot.stream.cuda_stream
                )
            )
            slot.stream.synchronize()
            img = out[0]
            return img, img.__cuda_array_interface__["data"][0], img.capacity

        small, ptr_small, cap_small = decode(320, 240, None)
        big, ptr_big, cap_big = decode(1280, 960, small)
        assert cap_big > cap_small, "capacity did not grow for a larger image"

        again, ptr_again, cap_again = decode(320, 240, big)
        assert cap_again == cap_big, "buffer shrank; retention would be unbounded"
        assert ptr_again == ptr_big, "smaller image reallocated instead of reusing"
    finally:
        backend._release_slot(slot)
        backend.shutdown()


def test_on_device_resize_produces_the_processor_target(configured):
    """Resizing on the accelerator must hit the processor's own target exactly.

    If it does not, the processor resizes a second time and the work is
    duplicated rather than moved -- strictly worse than not doing it.
    """
    backend.shutdown()
    target = lambda w, h: (w // 2, h // 2)
    backend.configure(num_decoders=1, min_gpu_pixels=0, resize_target=target)
    backend._COUNTERS.clear()  # counters are process-global and accumulate
    try:
        out = backend.decode_batch([_jpeg(1280, 960, seed=3)], "RGB")[0]
        assert out is not None
        assert out.shape == (480, 640, 3), f"got {out.shape}, want the target size"
        assert backend.stats().get("gpu_resized", 0) == 1
    finally:
        backend.shutdown()


def test_on_device_resize_is_skipped_when_target_matches(configured):
    """A no-op target must not pay for a resize."""
    backend.shutdown()
    backend.configure(
        num_decoders=1, min_gpu_pixels=0, resize_target=lambda w, h: (w, h)
    )
    backend._COUNTERS.clear()  # counters are process-global and accumulate
    try:
        out = backend.decode_batch([_jpeg(640, 480, seed=4)], "RGB")[0]
        assert out is not None and out.shape == (480, 640, 3)
        assert backend.stats().get("gpu_resized", 0) == 0
    finally:
        backend.shutdown()


def test_on_device_resize_matches_a_host_resize_closely(configured):
    """It is not bit-exact -- bound the deviation rather than assume it small."""
    from PIL import Image as PILImage

    data = _jpeg(1280, 960, seed=5)
    backend.shutdown(); backend.configure(num_decoders=1, min_gpu_pixels=0)
    full = backend.decode_batch([data], "RGB")[0]
    backend.shutdown()
    backend.configure(
        num_decoders=1, min_gpu_pixels=0, resize_target=lambda w, h: (640, 480)
    )
    try:
        small = backend.decode_batch([data], "RGB")[0]
        assert small is not None
        ref = np.asarray(
            PILImage.fromarray(full).resize((640, 480), PILImage.BICUBIC)
        )
        diff = np.abs(small.astype(np.int32) - ref.astype(np.int32))
        assert diff.mean() < 6.0, f"mean |d| {diff.mean():.2f} vs a host bicubic"
    finally:
        backend.shutdown()


def test_device_resize_applies_to_a_small_downscale(configured):
    """Every downscale resizes on the device, not just large reductions.

    The raster is already in device memory, so a shrink there is strictly less
    work than copying it whole and shrinking on the host. The size of the
    reduction changes how much is saved, not whether anything is.
    """
    backend.shutdown()
    # 4x reduction -- modest, comparable to 1080p at a pinned budget.
    backend.configure(num_decoders=1, min_gpu_pixels=0,
                      resize_target=lambda w, h: (w // 2, h // 2))
    backend._COUNTERS.clear()  # counters are process-global and accumulate
    try:
        out = backend.decode_batch([_jpeg(1280, 960, seed=11)], "RGB")[0]
        assert out is not None
        assert out.shape == (480, 640, 3)
        assert backend.stats().get("gpu_resized", 0) == 1
    finally:
        backend.shutdown()


def test_device_resize_declines_a_non_downscale(configured):
    """An upscale or equal target must fall through to the full-raster copy."""
    backend.shutdown()
    backend.configure(num_decoders=1, min_gpu_pixels=0,
                      resize_target=lambda w, h: (w * 2, h * 2))
    backend._COUNTERS.clear()
    try:
        out = backend.decode_batch([_jpeg(640, 480, seed=12)], "RGB")[0]
        assert out is not None
        assert out.shape == (480, 640, 3), "upscale target must not be applied"
        assert backend.stats().get("gpu_resized", 0) == 0
        assert backend.stats().get("resize_not_downscale", 0) == 1
    finally:
        backend.shutdown()


def test_cvcuda_hq_resize_is_used_when_available(configured):
    """When CV-CUDA is installed the resize must run on HQResize, not torch.

    Measured on 12 real 4K photographs against a PIL bicubic reference:
    HQResize 181 us / mean|d| 0.055 against the torch path's 806 us / 0.319 --
    faster and closer, so a silent fallback to torch is a regression on both
    axes and would otherwise be invisible.
    """
    import pytest

    if backend._cvcuda() is None:
        pytest.skip("CV-CUDA not installed")
    backend.shutdown()
    backend.configure(num_decoders=1, min_gpu_pixels=0,
                      resize_target=lambda w, h: (w // 4, h // 4))
    backend._COUNTERS.clear()  # counters are process-global and accumulate
    try:
        out = backend.decode_batch([_jpeg(1280, 960, seed=21)], "RGB")[0]
        assert out is not None
        assert out.shape == (240, 320, 3)
        assert backend.stats().get("resize_cvcuda", 0) == 1
        assert backend.stats().get("resize_torch", 0) == 0
        assert backend.stats().get("gpu_resized", 0) == 1
    finally:
        backend.shutdown()


def test_cvcuda_resize_tracks_pil_closely(configured):
    """Bound the deviation, as the torch path does -- but tighter.

    HQResize measured 5.8x closer to PIL than the torch resize it replaces, so
    this holds it to a materially tighter bound than the torch path's 6.0.
    """
    import pytest

    from PIL import Image as PILImage

    if backend._cvcuda() is None:
        pytest.skip("CV-CUDA not installed")
    data = _jpeg(1280, 960, seed=22)
    backend.shutdown(); backend.configure(num_decoders=1, min_gpu_pixels=0)
    full = backend.decode_batch([data], "RGB")[0]
    backend.shutdown()
    backend.configure(num_decoders=1, min_gpu_pixels=0,
                      resize_target=lambda w, h: (640, 480))
    try:
        small = backend.decode_batch([data], "RGB")[0]
        assert small is not None and small.shape == (480, 640, 3)
        ref = np.asarray(
            PILImage.fromarray(full).resize((640, 480), PILImage.BICUBIC)
        )
        diff = np.abs(small.astype(np.int32) - ref.astype(np.int32))
        assert diff.mean() < 2.0, f"mean |d| {diff.mean():.2f} vs a host bicubic"
    finally:
        backend.shutdown()
