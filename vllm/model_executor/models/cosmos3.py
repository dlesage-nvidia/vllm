# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex
import torch
from transformers import Cosmos3OmniConfig

from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig


class Cosmos3CVCUDAProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_processor(self, **kwargs):
        mm_config = self.ctx.get_mm_config()
        merged = mm_config.merge_mm_processor_kwargs(kwargs)
        requested = merged.get("device_resize", False) or any(
            merged.get(scope, {}).get("device_resize", False)
            for scope in ("images_kwargs", "videos_kwargs")
        )
        if not requested:
            return super().get_hf_processor(**kwargs)
        if not (mm_config.mm_processor_kwargs or {}).get("device_resize"):
            raise ValueError(
                "device_resize must be enabled in engine mm_processor_kwargs"
            )
        from importlib.util import find_spec

        from vllm.platforms import current_platform
        from vllm.transformers_utils.processors.cosmos3_cvcuda import (
            Cosmos3CVCUDAProcessor,
        )

        if not mm_config.mm_device_do_normalize:
            raise ValueError(
                "Cosmos3 device_resize requires mm_device_do_normalize=True"
            )
        if not current_platform.is_cuda() or find_spec("cvcuda") is None:
            raise ValueError("device_resize requires CUDA and vllm[cvcuda]")
        from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
            require_cvcuda_release,
        )

        require_cvcuda_release()
        return self.ctx.get_hf_processor(Cosmos3CVCUDAProcessor, **kwargs)


class Cosmos3ProcessingInfo(Cosmos3CVCUDAProcessingInfo):
    def get_hf_config(self) -> Cosmos3OmniConfig:
        return self.ctx.get_hf_config(Cosmos3OmniConfig)


class Cosmos3RawMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            "pixel_values_raw": MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            ),
            "pixel_values_videos_raw": MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            ),
        }


class Cosmos3CVCUDAVisionBase(Qwen3VLForConditionalGeneration):
    _cvcuda_workspace_profiled = False

    def embed_multimodal(self, **kwargs):
        workspace = None
        enabled = (self.multimodal_config.mm_processor_kwargs or {}).get(
            "device_resize"
        )
        if not enabled and not kwargs.keys() & {
            "pixel_values_raw",
            "pixel_values_videos_raw",
        }:
            return super().embed_multimodal(**kwargs)
        device = next(self.visual.parameters()).device
        if enabled and not self._cvcuda_workspace_profiled:
            from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
                reserve_cvcuda_workspace,
            )

            workspace = reserve_cvcuda_workspace(device)
            self._cvcuda_workspace_profiled = True
        processed = dict(kwargs)
        for pixel_key, grid_key, is_video in (
            ("pixel_values", "image_grid_thw", False),
            ("pixel_values_videos", "video_grid_thw", True),
        ):
            frames = processed.pop(f"{pixel_key}_raw", None)
            if frames is None:
                continue
            if not enabled or pixel_key in processed:
                raise ValueError(
                    "Raw pixels require device_resize and no processed pixels"
                )
            from transformers import (
                Qwen2VLImageProcessor,
                Qwen3VLVideoProcessor,
            )

            from vllm.transformers_utils.processors.cosmos3_cvcuda import (
                cvcuda_cosmos3_patchify,
            )

            if isinstance(frames, torch.Tensor):
                frames = list(frames.unbind(0))
            processor_class = (
                Qwen3VLVideoProcessor if is_video else Qwen2VLImageProcessor
            )
            processor_attr = f"_cvcuda_{'video' if is_video else 'image'}_processor"
            processor = getattr(self, processor_attr, None)
            if processor is None:
                processor = processor_class.from_pretrained(
                    self.model_config.model, revision=self.model_config.revision
                )
                setattr(self, processor_attr, processor)
            processed[pixel_key] = cvcuda_cosmos3_patchify(
                frames,
                processed[grid_key],
                device,
                image_processor=processor,
                is_video=is_video,
            )
        embeddings = super().embed_multimodal(**processed)
        del workspace
        return embeddings


@MULTIMODAL_REGISTRY.register_processor(
    Cosmos3RawMultiModalProcessor,
    info=Cosmos3ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Cosmos3ForConditionalGeneration(Cosmos3CVCUDAVisionBase):
    def get_encoder_cudagraph_config(self):
        config = super().get_encoder_cudagraph_config()
        if (self.multimodal_config.mm_processor_kwargs or {}).get("device_resize"):
            config.modalities = []
        return config

    # Cosmos3 unified checkpoints store a Qwen3-VL understanding tower
    # alongside a generation tower in a flat key layout. This mapper drops
    # the generation tower weights and rewrites the understanding tower keys
    # into the nested form expected by Qwen3VLForConditionalGeneration.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={
            regex.compile(
                r"^(layers\.|embed_tokens\.|norm\.)(.+)$"
            ): r"language_model.model.\1\2",
            regex.compile(
                r"^(blocks\.|merger\.|patch_embed\.|pos_embed\.|deepstack_merger_list\.)"
            ): r"visual.\1",
            regex.compile(r"^audio_modality_embed(?:\..*)?$"): None,
            regex.compile(r"^action_modality_embed(?:\..*)?$"): None,
        },
        orig_to_new_substr={
            "_moe_gen": None,
            ".add_q_proj.": None,
            ".add_k_proj.": None,
            ".add_v_proj.": None,
            ".to_add_out.": None,
            ".norm_added_q.": None,
            ".norm_added_k.": None,
            ".to_q.": ".q_proj.",
            ".to_k.": ".k_proj.",
            ".to_v.": ".v_proj.",
            ".to_out.": ".o_proj.",
            ".norm_q.": ".q_norm.",
            ".norm_k.": ".k_norm.",
            # ModelOpt-native dialect (diffusers/transformers read these; vLLM reads
            # weight_scale/input_scale instead), drop so AutoWeightsLoader passes
            ".input_quantizer.": None,
            ".weight_quantizer.": None,
            ".output_quantizer.": None,
        },
        orig_to_new_prefix={
            "proj_in.": None,
            "proj_out.": None,
            "time_embedder.": None,
            "audio_proj_in.": None,
            "audio_proj_out.": None,
            "action_proj_in.": None,
            "action_proj_out.": None,
            "lm_head.": "language_model.lm_head.",
        },
    )

    # Cosmos3 unified Diffusers checkpoints store reasoner weights across
    # transformer/ and vision_encoder/. Match both while excluding VAE and
    # sound_tokenizer weights; the mapper drops generation-only tensors.
    allow_patterns_overrides = ["[tv]*er/*.safetensors"]
