# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The plural image path: several images in one request decode in one call.

These exist because a plural entry point that no caller reaches is worth
nothing: the point of the batched API is that a real request uses it, and that
batching it does not change per-image failure semantics.
"""

import asyncio
import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal.media.connector import MediaConnector


def _data_url(width=32, height=24, seed=0) -> str:
    rs = np.random.RandomState(seed)
    array = (rs.rand(height, width, 3) * 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(array).save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def test_plural_fetch_returns_images_in_order():
    urls = [_data_url(32 + 8 * i, 24, seed=i) for i in range(4)]
    images = asyncio.run(MediaConnector().fetch_images_async(urls))
    assert len(images) == 4
    for i, item in enumerate(images):
        assert item.media.size == (32 + 8 * i, 24)


def test_plural_fetch_issues_one_batched_decode(monkeypatch):
    """The whole point: N images must become ONE load_bytes_many call."""
    from vllm.multimodal.media import image as image_mod

    widths: list[int] = []
    original = image_mod.ImageMediaIO.load_bytes_many

    def spy(self, datas):
        widths.append(len(datas))
        return original(self, datas)

    monkeypatch.setattr(image_mod.ImageMediaIO, "load_bytes_many", spy)
    urls = [_data_url(seed=i) for i in range(5)]
    # Batching is only correct for the accelerator backend; the Pillow default
    # deliberately fans out instead (see test_default_pillow_backend_...).
    connector = MediaConnector(
        media_io_kwargs={"image": {"image_backend": "nvimgcodec"}}
    )
    asyncio.run(connector.fetch_images_async(urls))
    assert widths == [5], f"expected one width-5 decode, got {widths}"


def test_one_bad_url_does_not_fail_its_siblings():
    """Scalar fetches failed independently; batching must not change that."""
    urls = [_data_url(seed=0), "data:image/jpeg;base64,bm90YW5pbWFnZQ==", _data_url(seed=1)]
    results = asyncio.run(MediaConnector().fetch_images_settled_async(urls))
    assert len(results) == 3
    assert not isinstance(results[0], BaseException)
    assert isinstance(results[1], BaseException)
    assert not isinstance(results[2], BaseException)


def test_plural_fetch_raises_for_the_failing_url():
    urls = [_data_url(seed=0), "data:image/jpeg;base64,bm90YW5pbWFnZQ=="]
    with pytest.raises(Exception):
        asyncio.run(MediaConnector().fetch_images_async(urls))


def test_empty_request_does_no_work():
    assert asyncio.run(MediaConnector().fetch_images_async([])) == []


def test_rejected_scheme_is_reported_per_image():
    urls = [_data_url(seed=0), "ftp://example.com/x.jpg"]
    results = asyncio.run(MediaConnector().fetch_images_settled_async(urls))
    assert not isinstance(results[0], BaseException)
    assert isinstance(results[1], ValueError)


def test_file_url_rule_still_applies_on_the_bytes_path():
    """The bytes path must not become a way around --allowed-local-media-path."""
    results = asyncio.run(
        MediaConnector().fetch_images_settled_async(["file:///etc/hostname"])
    )
    assert isinstance(results[0], RuntimeError)


def test_parser_accumulates_slots_and_fetches_once():
    """The parser must batch a request's images, not fetch them one by one."""
    from vllm.entrypoints.chat_utils import AsyncMultiModalContentParser

    parser = AsyncMultiModalContentParser.__new__(AsyncMultiModalContentParser)
    parser._image_urls = []
    parser._images_future = None
    calls: list[list[str]] = []

    class FakeConnector:
        async def fetch_images_settled_async(self, urls, *, image_mode="RGB"):
            calls.append(list(urls))
            return [f"img:{u}" for u in urls]

    parser._connector = FakeConnector()
    parser._image_urls = ["a", None, "b"]

    async def drive():
        return await asyncio.gather(
            parser._image_with_uuid_async(0, None),
            parser._image_with_uuid_async(1, None),
            parser._image_with_uuid_async(2, "u2"),
        )

    out = asyncio.run(drive())
    assert calls == [["a", "b"]], f"expected one batched fetch, got {calls}"
    assert out[0][0] == "img:a"
    assert out[1][0] is None  # a None url stays None
    assert out[2] == ("img:b", "u2")


# --- regressions from external review ----------------------------------------


def test_default_pillow_backend_does_not_serialize(monkeypatch):
    """The default backend must keep fanning out, one executor task per image.

    Routing Pillow through the plural entry point would serialise N decodes onto
    a single media-executor thread -- a regression for the backend nearly every
    deployment runs, in exchange for a batch only an accelerator can use.
    """
    from vllm.multimodal.media import image as image_mod

    plural_calls: list[int] = []
    scalar_calls: list[int] = []
    original_many = image_mod.ImageMediaIO.load_bytes_many
    original_one = image_mod.ImageMediaIO.load_bytes

    def spy_many(self, datas):
        plural_calls.append(len(datas))
        return original_many(self, datas)

    def spy_one(self, data):
        scalar_calls.append(1)
        return original_one(self, data)

    monkeypatch.setattr(image_mod.ImageMediaIO, "load_bytes_many", spy_many)
    monkeypatch.setattr(image_mod.ImageMediaIO, "load_bytes", spy_one)

    urls = [_data_url(seed=i) for i in range(4)]
    asyncio.run(MediaConnector().fetch_images_async(urls))
    assert plural_calls == [], "Pillow was routed through the batched path"
    assert len(scalar_calls) == 4


def test_custom_connector_scalar_override_is_not_bypassed():
    """A connector that customises only fetch_image_async must still be used."""
    from vllm.entrypoints.chat_utils import AsyncMultiModalContentParser

    used: list[str] = []

    class LegacyConnector(MediaConnector):
        async def fetch_image_async(self, image_url, *, image_mode="RGB"):
            used.append(image_url)
            return f"custom:{image_url}"

    parser = AsyncMultiModalContentParser.__new__(AsyncMultiModalContentParser)
    parser._image_urls = ["u1", "u2"]
    parser._images_future = None
    parser._connector = LegacyConnector()

    async def drive():
        return await asyncio.gather(
            parser._image_with_uuid_async(0, None),
            parser._image_with_uuid_async(1, None),
        )

    out = asyncio.run(drive())
    assert used == ["u1", "u2"], "the connector's own fetch_image_async was bypassed"
    assert out[0][0] == "custom:u1"


def test_connector_that_implements_batching_is_used_batched():
    from vllm.entrypoints.chat_utils import AsyncMultiModalContentParser

    batched: list[list[str]] = []

    class BatchingConnector(MediaConnector):
        async def fetch_image_async(self, image_url, *, image_mode="RGB"):
            raise AssertionError("scalar path must not be used here")

        async def fetch_images_settled_async(self, image_urls, *, image_mode="RGB"):
            batched.append(list(image_urls))
            return [f"b:{u}" for u in image_urls]

    parser = AsyncMultiModalContentParser.__new__(AsyncMultiModalContentParser)
    parser._image_urls = ["a", "b"]
    parser._images_future = None
    parser._connector = BatchingConnector()

    async def drive():
        return await asyncio.gather(
            parser._image_with_uuid_async(0, None),
            parser._image_with_uuid_async(1, None),
        )

    out = asyncio.run(drive())
    assert batched == [["a", "b"]]
    assert out[1][0] == "b:b"
