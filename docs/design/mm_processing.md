# Multi-Modal Data Processing

To enable various optimizations in vLLM such as [chunked prefill](../configuration/optimization.md#chunked-prefill) and [prefix caching](../features/automatic_prefix_caching.md), we use [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] to provide the correspondence between placeholder feature tokens (e.g. `<image>`) and multi-modal inputs (e.g. the raw input image) based on the outputs of HF processor.

In vLLM's rendering pipeline (see [BaseRenderer][vllm.renderers.base.BaseRenderer]), tokenization is performed as a separate step before multi-modal processing. Therefore, `BaseMultiModalProcessor` needs to recreate the output of calling HF processor end-to-end, while not being able to see the original text. This is achieved through **Dummy Input Text** and **Prompt Update Detection**.

## Dummy Input Text

Since Transformers 5.10, `ProcessorMixin` now allows multi-modal inputs to be passed by themselves. However, certain subclasses (such as `ChameleonProcessor`) and older out-of-tree implementations may still define their own `__call__` method that assumes the presence of text with corresponding placeholder tokens. This causes a problem as we don't have the original text anymore to pass to these processors.

To work around this, each model defines how to generate dummy text based on the number of multi-modal inputs, via [get_dummy_text][vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_text], which its override of [_get_hf_mm_text][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_text] returns. [_get_hf_mm_inputs][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_inputs] adds this text and `truncation=False` to the normalized HF inputs. The truncation setting is attached only to calls that use dummy text, so unrelated processor calls do not receive a text-specific option.

HF processing is split into four overridable stages:

1. [_get_hf_mm_inputs][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_inputs] extracts and normalizes processor data, kwargs, and passthrough data. It centrally renames vLLM's `audios` field to the standard HF `audio` keyword; processors that require another spelling can adapt it in an override.
2. [_call_hf_processor][vllm.multimodal.processing.BaseMultiModalProcessor._call_hf_processor] invokes the HF processor. Models can override it when they use a different callable or need separate construction and invocation kwargs.
3. [_finalize_hf_mm_data][vllm.multimodal.processing.BaseMultiModalProcessor._finalize_hf_mm_data] merges passthrough fields into the processed output. It lazily creates an empty output when there is no processor data.
4. [_postprocess_hf_mm_data][vllm.multimodal.processing.BaseMultiModalProcessor._postprocess_hf_mm_data] performs model-specific output transformations after passthrough fields have been merged.

This division lets most models customize one stage without reimplementing [_apply_hf_processor_main][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_main]. Implementations that genuinely require multiple processor calls or manually construct their complete output may still override that method, but must use `_get_hf_mm_inputs` and `_finalize_hf_mm_data` to retain the common input and output behavior.

## Prompt Update Detection

One of the main responsibilities of HF processor is to update the prompt with placeholder tokens. For example:

- Insert feature placeholder tokens (e.g. `<image><image>...<image>`, the number of which equals to the feature size) at the start of the string.
- Replace existing input placeholder tokens (e.g. `<image>` for a single image) with feature placeholder tokens (e.g. `<image><image>...<image>`, the number of which equals to the feature size).

The information about which tokens have been updated is key to finding the correspondence between placeholder feature tokens and multi-modal inputs.

Since we call HF processor without the input text, we have to perform this update by ourselves. In vLLM, we represent the necessary information using [PromptUpdate][vllm.multimodal.processing.PromptUpdate] in [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates], and apply them via [_apply_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._apply_prompt_updates].

Some HF processors additionally transform the prompt itself regardless of the multi-modal inputs (such as `ChameleonProcessor` appending a sep token for chat mode). Since the prompt tokens likewise bypass the HF processor, such transformations are replicated via [_postprocess_prompt][vllm.multimodal.processing.BaseMultiModalProcessor._postprocess_prompt] before the prompt updates are located or applied.

## Processor Output Caching

Some HF processors, such as the one for Qwen2-VL, are [very slow](https://github.com/vllm-project/vllm/issues/9238). To alleviate this problem, we cache the multi-modal outputs of HF processor to avoid processing the same multi-modal input (e.g. image) again.

When new data is passed in, we first check which items are in the cache, and which ones are missing. The missing items are passed into the HF processor in a single batch and cached, before being merged with the existing items in the cache.

## Speeding Up Multi‑Modal Data Processing

### Fused Normalisation on the Device

To accelerate the multi‑modal data pipeline (decoding, resizing, normalisation, and rescaling), we offload the **normalisation and rescaling** steps from the CPU to the device and optimises data movement.

#### Fusing Normalisation and Rescaling on the GPU

Traditionally, the CPU would divide pixel values by 255, then subtracts the mean and divides by the standard deviation. We fuse these steps into a single per-channel affine transform on the device.

`FusedMMInputNorm` dispatches to `fused_mm_input_norm_triton` on CUDA, which applies:

    y = x * weight[c] + bias[c]
      = (x * rescale_factor - image_mean[c]) / image_std[c]

with:

    weight[c] = rescale_factor / image_std[c]
    bias[c]   = -image_mean[c] / image_std[c]

The kernel accepts `uint8`, `float16`, `bfloat16` and `float32` inputs. The `uint8` path is the primary fast path: raw bytes travel to the device unprocessed, and the rescale factor is folded into `weight` — no separate divide-by-255 step. Compute is always done in fp32 inside the kernel.

#### Optimized Data Path

    Before: CPU decode → CPU resize → CPU rescale (÷255) → CPU normalize
            → cast to bf16 → PCIe (2 B/elem) → GPU

    After:  CPU decode → CPU resize → uint8 pixel_values
            → PCIe (1 B/elem) → GPU fused affine (fp32 compute) → bf16

The transfer path **Entrypoint (API server / offline LLM) → Engine Core(scheduler + workers) → Device Memory** stays in `uint8`. On the device,`fused_mm_input_norm_triton` computes in fp32 internally and writes the requested `visual_dtype` (a per-call argument to `forward_*`, commonly `bf16`) directly, without materialising a global fp32 intermediate.

#### Toggle: `mm_device_do_normalize`

GPU-side fusion is controlled by `multimodal_config.mm_device_do_normalize`:

- When `True`, normalisation and rescaling are done on the GPU using the `FusedMMInputNorm` layer; when `False`, we fall back to the old CPU‑side path.
- The flag is **enabled by default** for all models that support it.
- Currently, it’s on by default for these architectures:

| name                       | Architecture                               | Example HF Models                                     |
| -------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| `deepseek-v4`              | `DeepseekV4ForConditionalGeneration`       | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`            |
| `deepseek-v4.1`            | `DeepseekV41ForCausalLM`                   | `deepseek-ai/DeepSeek-V4.1-Flash`                     |
| `gemma-3`                  | `Gemma3ForConditionalGeneration`           | `google/gemma-3-4b-it`, etc.                          |
| `gemma-4`                  | `Gemma4ForConditionalGeneration`           | `google/gemma-4-E2B-it`, etc.                         |
| `glm-4v`                   | `Glm4vForConditionalGeneration`            | `zai-org/GLM-4.1V-9B-Thinking`, etc.                  |
| `glm-ocr`                  | `GlmOcrForConditionalGeneration`           | `zai-org/GLM-OCR`                                     |
| `intern-s1-pro`            | `InternS1ProForConditionalGeneration`      | `internlm/Intern-S1-Pro`                              |
| `intern-s2-mobius`         | `InternS2MobiusForConditionalGeneration`   | `internlm/Intern-S2-Mobius`                           |
| `kimi-k2.5`                | `KimiK25ForConditionalGeneration`          | `moonshotai/Kimi-K2.5`, etc.                          |
| `kimi-k3`                  | `KimiK3ForConditionalGeneration`           | `moonshotai/Kimi-K3`                                  |
| `llama-nemotron-vl-embed`  | `LlamaNemotronVLForEmbedding`              | `nvidia/llama-nemotron-embed-vl-1b-v2`                |
| `llama-nemotron-vl-rerank` | `LlamaNemotronVLForSequenceClassification` | `nvidia/llama-nemotron-rerank-vl-1b-v2`               |
| `llava-onevision`          | `LlavaOnevisionForConditionalGeneration`   | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, etc.     |
| `mistral3`                 | `Mistral3ForConditionalGeneration`         | `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, etc. |
| `nano-nemotron-vl`         | `NemotronH_Nano_VL_V2`                     | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`          |
| `nano-nemotron-omni`       | `NemotronH_Nano_Omni_Reasoning_V3`         | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` |
| `nemotron-omni`            | `NemotronH_Omni_Reasoning_V3`              | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` |
| `nemotron-parse`           | `NemotronParseForConditionalGeneration`    | `nvidia/NVIDIA-Nemotron-Parse-2.0`, etc.              |
| `qwen2-vl`                 | `Qwen2VLForConditionalGeneration`          | `Qwen/Qwen2-VL-2B-Instruct`, etc.                     |
| `qwen2.5-omni`             | `Qwen2_5OmniModel`                         | `Qwen/Qwen2.5-Omni-3B`, etc.                          |
| `qwen2.5-vl`               | `Qwen2_5_VLForConditionalGeneration`       | `Qwen/Qwen2.5-VL-3B-Instruct`, etc.                   |
| `qwen3-omni`               | `Qwen3OmniMoeForConditionalGeneration`     | `Qwen/Qwen3-Omni-30B-A3B-Instruct`, etc.              |
| `qwen3-vl`                 | `Qwen3VLForConditionalGeneration`          | `Qwen/Qwen3-VL-4B-Instruct`, etc.                     |
| `qwen3-vl-moe`             | `Qwen3VLMoeForConditionalGeneration`       | `Qwen/Qwen3-VL-30B-A3B-Instruct`, etc.                |
| `qwen3.5`                  | `Qwen3_5ForConditionalGeneration`          | `Qwen/Qwen3.5-0.8B`, etc.                             |
| `qwen3.5-moe`              | `Qwen3_5MoeForConditionalGeneration`       | `Qwen/Qwen3.5-35B-A3B`, etc.                          |
| `qwen4-exp`                | `Qwen4ExpForConditionalGeneration`         | `Qwen/Qwen3.8-Flash-Next`                             |

#### Key Properties and Gains

- **CPU offload & 50% PCIe savings** — normalisation/rescaling leaves the CPU entirely; sending `uint8` (1 byte) instead of `bf16` (2 bytes) halves transfer volume.
- **Single-pass fusion** — `y = x * weight[c] + bias[c]` in one kernel launch via `fused_mm_input_norm_triton`; input is read in its native dtype (`uint8`).
- **float32 compute for free** — arithmetic runs in fp32 inside the kernel regardless of I/O dtypes. Accuracy matches fp32, with no extra bandwidth: intermediates stay in registers, and no global fp32 tensor is materialised.
- **Preallocated output with batch-dim padding** — callers can pass a larger buffer; only the leading `N` rows are written, so buffers are reusable across calls.

#### Resizing on the Device (Nemotron VL)

With `mm_device_do_normalize`, most models still resize images and video frames on the CPU, in `uint8`. Nemotron VL models (`NemotronH_Nano_VL_V2` and the Nemotron Omni architectures above) can resize them on the GPU instead, with [CV-CUDA](https://github.com/CVCUDA/CV-CUDA). This is opt-in and needs CV-CUDA, which the `cvcuda` extra installs (`cvcuda-cu13`, or `cvcuda-cu12` for vLLM built with CUDA 12):

```bash
pip install "vllm[cvcuda]"
vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
    --trust-remote-code --mm-processor-kwargs '{"device_resize": true}'
```

The API server then sends the decoded `uint8` pixels at their source resolution. On the GPU, CV-CUDA resizes them (bicubic, antialiased, `uint8` in and out) and reformats them to NCHW; `FusedMMInputNorm` then casts, rescales and normalizes them exactly as it does for the CPU path. This covers dynamic-resolution images and video frames. The pixels are uploaded on a stream of their own, so the upload does not wait for the steps already queued on the model's stream; the resize kernels are queued on the model's stream after it. CV-CUDA's scratch buffer is capped at 512 MiB and reserved during the startup memory profile.

#### Resizing on the Device (Qwen3.5)

Qwen3.5 dense and MoE vision-language models have an opt-in CV-CUDA preprocessing path under development. The first version uses the public CV-CUDA **v0.19.0 release**, with per-axis floating-point resizing and selective exact integer GPU correction rather than an unreleased resize mode. Full processor, model and serving validation remain unfinished; the examples do not establish deployment acceptance. The path is **off by default**:

```bash
pip install "vllm[cvcuda]"
vllm serve Qwen/Qwen3.5-4B --mm-processor-kwargs '{"device_resize": true}'
```

The same option applies to larger checkpoints. For example, on a host with enough GPU memory for the BF16 weights and runtime buffers:

```bash
vllm serve Qwen/Qwen3.5-122B-A10B --dtype bfloat16 --tensor-parallel-size 4 \
    --max-model-len 16384 --mm-processor-kwargs '{"device_resize": true}'
```

This requires CUDA, the optional CV-CUDA dependency, torchvision preprocessing, and `mm_device_do_normalize=True`. It accepts decoded RGB `uint8` images and videos with bicubic antialiased resizing. It cannot be enabled only on an individual request, because the engine must account for its memory at startup. Requests can disable it with `device_resize: false`.

The API server uses the Hugging Face processor's shape logic on metadata-only tensors, preserving its target grids, frame sampling, per-frame pixel caps, timestamps, and prompt expansion without resizing or packing pixels on the CPU. It sends source-resolution pixels to the model, which resizes and packs Qwen's spatial and temporal patches on the GPU. Odd frame counts repeat the final frame, matching the reference processor.

The `cvcuda` extra and Qwen CI pin the public CV-CUDA v0.19.0 wheel matching the CUDA major version. Development wheels, patched operators and rewritten wheel metadata are not prerequisites. The released `TvResize` performs FP32 horizontal and vertical resizing separately. Coefficient and arithmetic error bounds identify ambiguous rounding outputs, which are corrected with an exact integer GPU dot product before each axis's `uint8` clipping and rounding. Host coefficient/error tables are cached with a 16 MiB byte limit, without caching decoded pixels or GPU tensors. Do not treat historical experimental-build results or standalone resize fixtures as full validation of this path.

The compatibility guard also admits version **0.20.0** for a separate snapshot evaluation lane: CV-CUDA `rel/v0.20.x` at `f4cd27d5d4ed73a4a134973d769aac8fd18e77dc`. This snapshot is distinct from the released v0.19.0 dependency; the default extra and CI pin remain unchanged. Both versions require matching module/distribution versions and the public resize, reformat, tensor, stream and cubic-interpolation APIs. RC, development, beta and local-version suffixes are not admitted. An accepted version/API contract does not attest wheel provenance or establish numerical, memory/stream, model or serving correctness. Genuine snapshot artifact provenance and all unchanged GPU qualification tests remain required; v0.20.0 GPU qualification is pending. The processor retains the same per-axis FP32 resize and exact integer correction, without adopting the snapshot's mixed-dtype resize optimization or changing tolerances.

Qwen resizing must preserve torchvision v2's CPU `uint8` bicubic antialiased preprocessing, including fixed-point coefficients and horizontal-then-vertical `uint8` clipping and rounding. A floating-point resize rounded only once, or a different per-axis coefficient quantization, is not automatically equivalent. Pixel and model-output checks must pass without loosening their thresholds. Rescale and normalization use the existing device input-normalization layer. Public-release actual-weight accuracy, memory/stream correctness and end-to-end serving validation remain pending. The shared helper also changes Nemotron's stream and upload handling, so that path needs regression validation with the released dependency.

Frame chunks prefer at most 256 MiB of source pixels and use an independent resize-consumer stream with two reusable upload slots. The GPU tensor temporary budget is separately bounded to 384 MiB, including those slots, floating intermediates, integer correction and coefficient uploads. Oversized frames use orthogonal row/column tiles that retain the entire filtered axis; when a complete horizontal `uint8` intermediate cannot fit, column strips are recomputed before vertical resizing. Only a minimum complete filtered-axis operand exceeding the actual budgets is rejected. Full output tensors and caller-owned GPU inputs are additional to the temporary budget. Pageable uploads may wait for resize-consumer events, but not unrelated model work. CV-CUDA work is ordered with PyTorch, including when the model uses CUDA's legacy default stream. The startup profile reserves the 512 MiB native scratch and a 384 MiB tensor allowance and warms both floating-point axis/correction variants. Whole-wrapper memory and stream validation remains pending. Encoder CUDA graphs are disabled for this opt-in mode; language-model CUDA graphs are unaffected.
