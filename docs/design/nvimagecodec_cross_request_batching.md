# Cross-Request nvImageCodec Batching

Status: implemented and benchmarked.

Reviewed and updated on 2026-08-26.

## Summary

The pre-service nvImageCodec backend performed one native decode call for
multiple images in the same request. It could not combine the common serving
workload of one JPEG per request, so high request concurrency reached
nvImageCodec as a sequence of native batches of width one.

The process-local request batcher at the media connector boundary combines
compatible singleton JPEG requests into native batches of up to the configured
`batch_size`, defaulting to five. It dispatches a full batch immediately and a
partial batch after a short, bounded coalescing interval. Each API server
process owns its own queue and decoder workers.

The implementation first fixes the regressions and configuration failures found
during review. In particular, it restores parallel Pillow decoding for the
default backend. It does not recursively submit work to `global_thread_pool`
from `load_bytes_many`: that method already runs in the same pool, so nested
submission and waiting can deadlock when all workers are occupied.

The fourth, independently revertible change adds a bounded decode ring. It
copies decoded pixels asynchronously into pinned host buffers, submits another
native batch before materializing the ready Pillow images, and releases each GPU
lease as soon as its copy event completes. It still returns Pillow images;
keeping pixels GPU-resident for downstream preprocessing remains separate work.

## Goals

- Turn concurrent one-image JPEG requests into native nvImageCodec batches near
  width five.
- Preserve request order, exact exception ownership, Pillow fallback semantics,
  image metadata, image mode conversion, EXIF behavior, and cache-key isolation.
- Keep the process-global media executor free while nvImageCodec work waits,
  coalesces, or decodes. Raw URL, file, and cache I/O may still use that pool.
- Leave the default Pillow path no slower than the parent commit for multi-image
  requests.
- Bound decode-queue entries, pending encoded bytes owned by the service,
  decoder concurrency, and GPU memory use.
- Make the achieved native batch widths and queue delay observable.
- Overlap native decode and device-to-host copies with Pillow materialization
  without weakening GPU-memory accounting or positional fallback.

## Non-goals

- Returning GPU-resident images or eliminating device-to-host copies.
- Changing downstream multimodal preprocessing to consume device images.
- Sharing slots or reserved capacity with PyNvVideoCodec.
- Coalescing CPU-plugin codecs across requests in the first implementation.
- Changing the model scheduler, request admission, or inference execution.
- Claiming continuous 100% NVJPG utilization. Native width five is necessary on
  A100, but host conversion can still leave gaps between hardware decode calls.

## Evidence From the Current Branch

The clean inference comparison used one image per request, so it could not expose
the new default-Pillow regression. A local decode diagnostic using eight JPEGs
showed the cost of replacing independent executor jobs with one serial
`load_bytes_many` job:

| Resolution | Serial Pillow batch | Eight scalar worker jobs | Slowdown |
|------------|--------------------:|-------------------------:|---------:|
| 1920x1080  | 70.3 ms             | 17.5 ms                  | 4.02x    |
| 3840x2160  | 276.2 ms            | 52.2 ms                  | 5.29x    |

This is a focused local diagnostic, not an end-to-end serving result. It is
sufficient to make restoration of the old Pillow dispatch a prerequisite.

After restoring scalar Pillow dispatch, a five-pair parent-versus-branch run
used the public asynchronous connector with eight data-URL JPEGs per request.
The parent was `d125b540b7`; the rebased branch was `7e37f1f51a`. Values are
means ± sample standard deviation, and paired changes retain the deliberately
alternated source order:

| Resolution | Parent Pillow | Branch Pillow | Paired change |
| ---------- | ------------: | ------------: | ------------: |
| 1920x1080 | 208.4 ± 21.5 images/s | 207.3 ± 20.0 images/s | -0.3% ± 6.9% |
| 3840x2160 | 49.5 ± 4.7 images/s | 52.0 ± 3.0 images/s | +5.5% ± 5.9% |

The 1080p result is within run noise and 4K is faster, closing the default-path
regression check. The benchmark used the same Python 3.12.13/Pillow 12.3.0
environment and identical input digests for both source trees.

On A100, the existing adapter measurements showed why batch width alone was not
the final optimization: the native-width-five adapter path reached about
0.71 Gpixel/s while a raw native-width-five hardware decode probe reached about
6.22 Gpixel/s. The gap includes device-to-host synchronization, Pillow object
creation, metadata work, and adapter overhead. Cross-request coalescing addresses
the batch-width portion; the bounded ring addresses the serial synchronization
between successive native calls. It does not eliminate the transfer or Pillow
materialization.

The implemented cross-request service was subsequently measured on an A100
with 16 concurrent singleton JPEG requests, two decoder slots, three randomized
10-second repetitions, and no inference. A 0.25 ms coalescing window was the
best point in the measured `0`, `0.1`, `0.25`, `0.5`, and `1.0` ms sweep:

| Resolution | Native batch 1 | Native batch 5 | Improvement | Full-width calls  |
|------------|---------------:|---------------:|------------:|------------------:|
| 1920x1080  | 402.7 images/s | 517.0 images/s | 28.4%       | 3100/3118 (99.4%) |
| 3840x2160  | 106.6 images/s | 154.5 images/s | 44.9%       | 935/938 (99.7%)   |

Both resolutions had zero Pillow fallbacks. CPU time per image fell by 20.5%
at 1080p and 25.9% at 4K. The device-global NVML NVJPG duty value fell when
moving from width one to width five even as throughput increased; it measures
active time, not simultaneous occupancy of the five hardware engines. Native
widths plus throughput scaling are therefore evidence of effective hardware
batching, not continuous saturation.

The production timeout remains zero to avoid intentional low-QPS latency. The
documented 0.25 ms setting is the measured high-QPS throughput tuning point; it
improved throughput over zero timeout by 2.1% at 1080p and 2.8% at 4K in this
sweep.

The final ring-depth comparison used a fresh Python process for every cell,
two decoder workers, native batches of five, concurrency 64, a 0.25 ms
coalescing window, 10 seconds of warmup, and 20 seconds of measurement. Four
counterbalanced repetitions at each resolution all passed source, workload,
fallback, and service-accounting gates. Values are means ± sample standard
deviation:

| Resolution | Depth one | Depth four | Paired throughput change | CPU cores | NVJPG mean |
| ---------- | --------: | ---------: | -----------------------: | --------: | -----------: |
| 1920x1080 | 494.03 ± 4.14 images/s | 545.13 ± 4.23 images/s | +10.35% ± 0.97% | 1.816 → 1.894 | 22.45% → 25.17% |
| 3840x2160 | 153.94 ± 5.48 images/s | 189.54 ± 2.94 images/s | +23.21% ± 3.54% | 1.887 → 1.980 | 17.74% → 22.11% |

Although instantaneous CPU use increased, CPU time per decoded image fell by
5.5% at 1080p and 14.8% at 4K. Depth two showed unstable, bimodal throughput at
1080p, and depth eight could not be kept full by the concurrency-64 workload;
depth four was stable in every paired run and became the default.

A separate contemporaneous three-repetition Pillow campaign measured 396.21 ±
9.63 images/s at 1080p and 93.42 ± 1.03 images/s at 4K, again as means ± sample
standard deviation. CPU use was 5.291 and 4.604 cores, respectively. Relative
to those unpaired baselines, depth-four nvImageCodec delivered 37.6% and 102.9%
more throughput while using 64.2% and 57.0% fewer CPU cores.

The ring does not make the current Pillow-output adapter saturate A100 NVJPG.
Scaling the same depth-four workload to six decoder workers reached 890.9
images/s and 41.2% mean NVJPG at 1080p. Eight workers reached 416.1 images/s
and 48.6% mean NVJPG at 4K. A standalone nvImageCodec probe that reused GPU
outputs reached 2824.1 images/s and 96.8% mean NVJPG at 1080p, and 809.3
images/s and 91.0% at 4K. Increasing the vLLM native batch ceiling from five
to 32 did not improve throughput at either resolution, so larger native calls
alone do not close the gap. Device-to-host transfer, Pillow materialization,
and request feeding remain contributors to the hardware-idle intervals.
Removing that CPU-image contract remains a separate change.

The final end-to-end run used one image per request, output length 128,
concurrency 16, 256 measured requests after 48 warmup requests, and a fresh
Qwen3-VL-2B server for every cell. Three counterbalanced repetitions produced
the following means ± sample standard deviation:

| Resolution | Variant | Requests/s | Mean latency | CPU cores | NVJPG mean |
| ---------- | ------- | ---------: | -----------: | --------: | -----------: |
| 1920x1080 | Pillow | 19.22 ± 0.08 | 831.52 ms | 1.310 | 0.00% |
| 1920x1080 | Depth one | 19.83 ± 0.46 | 803.23 ms | 1.209 | 1.72% |
| 1920x1080 | Depth four | 20.10 ± 0.13 | 794.49 ms | 1.217 | 1.73% |
| 3840x2160 | Pillow | 14.94 ± 0.27 | 1057.56 ms | 2.328 | 0.00% |
| 3840x2160 | Depth one | 15.78 ± 0.08 | 1002.92 ms | 1.993 | 7.92% |
| 3840x2160 | Depth four | 15.71 ± 0.16 | 1007.28 ms | 1.977 | 7.72% |

Depth four improved throughput over Pillow by a paired 4.59% ± 0.89% at
1080p and 5.15% ± 2.90% at 4K. Its incremental paired change over depth one
was 1.37% ± 2.13% and -0.47% ± 1.22%, respectively, so this inference workload
does not demonstrate an additional ring-specific gain. Decode-service claims
averaged only 2.51 images at 1080p and 1.10 at 4K; inference did not build the
sustained backlog that made depth four effective in decode-only testing. Input
ordering and prompt/completion token counts matched across variants. Generated
text hashes were diagnostic rather than equivalent; pixel fidelity is covered
by the dedicated CUDA decoder suite.

## Review Disposition

### Must Be Fixed Before Cross-Request Batching

1. **Restore the default Pillow path.** For an effective Pillow backend, the
   connector must use the prior per-image asynchronous fetch/decode calls and
   `asyncio.gather`. Only the nvImageCodec path should fetch encoded bytes and
   invoke a plural decode. Registered connector overrides retain their scalar
   semantics. Add parent-versus-branch eight-image benchmarks at both target
   resolutions.

2. **Fix Kimi-K3 configuration composition.** Kimi-K3 currently passes an
   already startup-merged `media_io_kwargs` dictionary through request-level
   sanitization a second time. This strips `decoders` and `batch_size`, and it
   normally strips `backend="nvimagecodec"` after GPU memory has been reserved.
   Apply this precedence exactly once:

   `Kimi-K3 model default < engine static configuration < request overrides`

   Protected startup fields (`backend`, `decoders`, `batch_size`, and the new
   timeout) must never be reclassified as request overrides. Test both sync and
   async Kimi renderers and the offline tracker path.

3. **Isolate standalone helpers from the server-only environment default.**
   `vllm.multimodal.utils.fetch_image` constructs a connector without installing
   a GPU IPC pool. It must default explicitly to Pillow, even when
   `VLLM_IMAGE_LOADER_BACKEND=nvimagecodec`. An explicit nvImageCodec request to
   this helper should fail early with a configuration message unless a future API
   supplies a safe reservation owner. It must not configure the process-global
   decoder pool. Cover dataset client-side image encoding as well as the helper.

4. **Make metadata rejection positional and fail-safe.** Exceptions raised by
   the nvImageCodec binding while constructing or reading a `CodeStream` for one
   item should make that item return `None` and use Pillow. Do not blanket-catch
   configuration failures, GPU-budget failures, or systemic CUDA failures.
   Define a public positional batch-error contract so connector and tracker code
   no longer import private exceptions or probe private methods.

5. **Restore the caller's CUDA device as well as its stream.** Capture the
   original device and stream before selecting device zero, then restore both.
   Fix and test the image implementation only in this feature branch. The same
   pattern in PyNvVideoCodec is a separate, independently reviewed change.

6. **Validate YCCK JPEG fidelity.** The generated CMYK JPEG test covers Adobe
   transform zero, not YCCK transform two. Add a redistributable real YCCK
   fixture with attribution and compare nvImageCodec with Pillow on CUDA. If it
   is not within the established tolerance, route YCCK to Pillow.

7. **Complete the positional-error tests.** Add coverage for synchronous
   `fetch_images`, a failure at a non-zero original index after candidate
   filtering, and async tracker interleaving of pending and callable items.
   Assert the original exception object and original request position survive.

8. **Make dependency behavior explicit on Arm.** Version `0.9.0.20` and its
   `[all]` plugin wheels have SBSA `aarch64` artifacts for CUDA 12 and CUDA 13.
   [NVIDIA's installation guide](https://docs.nvidia.com/cuda/nvimagecodec/installation.html)
   documents a different `nvidia-nvimgcodec-tegra-cu12` package for Tegra,
   however, while vLLM's Arm release matrix includes Orin and Thor. A generic
   `aarch64` wheel tag cannot distinguish those platforms. The recommended
   policy is to remove nvImageCodec from unconditional `install_requires` and
   require an explicit platform-appropriate `[all]` installation when the
   backend is enabled. Official images may install it only where their target is
   known. At minimum, never pull the generic SBSA package automatically from a
   generic aarch64 vLLM wheel. Add resolution, import, and decode smoke tests for
   x86-64 and SBSA on both CUDA majors, plus an explicit Tegra image/runtime
   gate. Do not treat a generic Arm container build as Tegra validation.

9. **Test HTJ2K through the production gate.** The CUDA test currently skips
   native HTJ2K when Pillow lacks its JPEG 2000 pixel decoder, even though the
   production nvImageCodec path only needs Pillow to identify the header. Remove
   that skip from native integration tests. Keep it for Pillow reference or
   fixture-generation tests. Prove that J2K and JP2 decode through nvImageCodec
   when Pillow pixel loading is deliberately unavailable.

### Small Stabilization Cleanups

- Replace the DLPack tensor view used only for shape and dtype validation with
  direct nvImageCodec image attributes. Retain validation of the copied host
  array and a real CUDA test.
- Compute the logged per-server decoder reservation as
  `decoder_reserved_bytes // num_api_servers` in all cases.
- Assert `apply_exif_orientation=False` and `allow_any_depth=False` in the fake
  decoder tests.
- Make the GPU-memory test independent of the developer's backend environment.
- Make import plus JPEG, JPEG 2000/HTJ2K, and TIFF decoding mandatory in NVIDIA
  wheel/image smoke tests. Developer and non-NVIDIA lanes may still skip an
  unavailable optional backend.
- Remove the duplicate `candidate_output_modes` declaration and unreachable
  `UnidentifiedImageError` catches while touching those functions.

### Explicitly Deferred or Rejected

| Recommendation | Decision |
| -------------- | -------- |
| Share image and video decoder slots | Reject. The resource types, factories, configuration, invalidation, and retained memory are different. A generic implementation helper can be a later refactor, without sharing slots or capacity. |
| Fall back when one image exceeds the configured GPU pool | Keep the current hard admission error. The operator explicitly chose an insufficient budget; silently using Pillow would hide capacity misconfiguration. Coalescing must isolate the error to that request. |
| Shorten the GPU lease | Implemented for multi-chunk depth-two-or-greater ring execution in the fourth, independently revertible change. An event proves the pinned copy complete before device images and DLPack views are dropped and the lease is released. The depth-one and single-chunk synchronous path remains the exact no-ring baseline and is tracked below. |
| Avoid both Pillow and nvImageCodec header parsing | Defer. Pillow supplies animation, transparency, EXIF, and source metadata that `CodeStream` does not. Measured header work is small relative to raster transfer. |
| Group every native call by exact codec | Benchmark first. It can fragment batches and has no effect on the all-JPEG target workload. Cross-request v1 deliberately queues only JPEG. |
| Merge the GPU and CPU chunk loops | Reject for now. Their memory leases, fallback behavior, and failure handling differ, and combined CPU/GPU decoder use has deadlocked. |
| Consolidate the Pillow and nvImageCodec codec tables | Defer. They encode different semantic and plugin-routing policies. Add consistency tests instead. |
| Optimize no-EXIF Pillow normalization | Worth a standalone Pillow PR. It predates this feature and should be measured independently. |
| Parallelize synchronous URL fetching | Follow-up. It is useful for offline callers but is not required for cross-request server batching. |
| Remove scalar decoder wrappers | Optional cleanup before feature merge; it does not affect this design. |

## Follow-up TODO

This checklist records optimization ideas found during code review and the
review of NVIDIA's public nvImageCodec Python samples. Keep an item here until
it is either implemented with evidence or moved to the decision table above.

### Before Final Review of This Change

- [ ] Finish matched A100 and RTX PRO 6000 decode-only and OSL-128 inference
  measurements at 1080p and 4K. Report throughput, server-plus-MPS CPU use,
  NVJPG utilization, and exact service accounting from fresh-process runs.
- [ ] A/B-test the decoder's `max_num_cpu_threads` setting at one, two, and four
  threads, capped to avoid oversubscribing the API server. NVIDIA's
  [DataLoader sample](https://docs.nvidia.com/cuda/nvimagecodec/samples/torch_dataloader.html)
  gives a batched decoder multiple CPU threads for parsing and dispatch, but
  this change should retain one unless the serving benchmarks show a repeatable
  throughput benefit without a material CPU or memory regression.
- [ ] Benchmark reusable native output images with the bounded ring, following
  NVIDIA's
  [resource-reuse sample](https://docs.nvidia.com/cuda/nvimagecodec/samples/reuse_resources.html).
  Record allocation savings, retained GPU memory, cancellation behavior, and
  mixed-resolution growth before deciding whether it is safe for this change
  or belongs in a follow-up PR.
- [ ] Re-run a multi-wave CUDA ring stress with more than `pipeline_depth`
  back-to-back chunks, mixed resolutions, and cancellation after the final
  tuning choice. Prove that native-image and DLPack lifetimes remain valid and
  that service-owned encoded bytes, raster leases, pinned staging buffers, and
  parked request metadata remain within their separate bounds.

### Separate Optimization PRs

- [ ] Keep decoded pixels on the GPU through resize, normalization, and tensor
  conversion, using the
  [CV-CUDA interop pattern](https://docs.nvidia.com/cuda/nvimagecodec/samples/cvcuda_sample.html).
  This intentionally changes the downstream media contract and is not part of
  the batching/ring change.
- [ ] If native output-image reuse is not included above, add it with explicit
  per-slot retained-memory accounting and a policy for buffers that grew for an
  unusually large image.
- [ ] Revisit duplicate Pillow and `CodeStream` header parsing only after real
  fixtures prove equivalent EXIF, animation, transparency, CMYK/YCCK, and
  malformed-input behavior.
- [ ] Measure grouped-by-shape GPU preprocessing after the GPU-resident contract
  exists; do not add a second grouping policy to the CPU/Pillow return path.
- [ ] Evaluate the pre-existing Pillow no-EXIF copy and synchronous URL-fetch
  opportunities as independent changes with their own baseline measurements.
- [ ] Investigate the history-dependent nvImageCodec throughput seen in both
  long-lived-MPS and fresh direct-context cells. MPS bypass rules out the
  shared server as the sole cause; reproduce the mode change with a dedicated
  endurance harness, collect CUDA/NVJPG timelines, and keep daemon lifecycle
  control outside this feature PR.
- [ ] Benchmark early GPU-lease release for depth one and single-chunk calls.
  They intentionally retain the synchronous no-ring path today; changing them
  to pinned/event drainage must preserve depth-one equivalence and account for
  retained pinned memory before the optimization is generalized.

### Reviewed and Already Accounted For

- [x] Use one native list decode for a batch rather than serial scalar decode
  calls, as shown in NVIDIA's
  [batch sample](https://docs.nvidia.com/cuda/nvimagecodec/samples/batch_sample.html).
- [x] Reuse persistent decoder slots, use explicit CUDA streams and completion
  events, and establish DLPack ownership before releasing native image objects.
- [x] Pass encoded bytes directly; do not stage request images through temporary
  files.
- [x] Bound active encoded/raster work while allowing lightweight request
  waiters to park until capacity is available; do not reject requests merely
  because all decode slots are momentarily busy.
- [x] Reject serial per-image synchronization and large fixed fleets of decoder
  processes as serving defaults. Decoder count, native batch width, and ring
  depth remain independently measured resources.
- [x] Isolate both standalone image and video helpers from the server-only
  nvImageCodec environment default.
- [x] Exercise palette PNG `tRNS` expansion and EXIF orientation through the
  real nvImageCodec library, in addition to fake-backend control-flow tests.
- [x] Restore and measure the default Pillow eight-image path against the exact
  parent; 1080p is within noise and 4K is faster in the five-pair comparison.

## Proposed Architecture

### Placement

Add a process-global `_NvImageCodecDecodeService` in a small image-media module,
not in the model scheduler and not on an individual `MediaConnector` instance.
Trackers and connectors are request-scoped, so an instance queue cannot combine
different requests. The cross-request batcher is one scheduling policy inside
this service; direct nvImageCodec jobs use the same service and accounting.

The service owns:

- one condition-protected FIFO queue per compatibility key;
- one FIFO queue for direct nvImageCodec jobs;
- one lazy dispatcher thread;
- a decode executor with at most `decoders` workers;
- the number of in-flight native jobs;
- immutable process-wide `ImageDecodeServiceConfig`;
- monotonic sequence numbers and deadlines; and
- per-item `concurrent.futures.Future` objects.

One shared condition guards all queues, admission counters, in-flight state,
deadlines, and the stop flag. Per-key conditions cannot atomically select the
oldest ready work across keys.

The dispatcher itself does no CUDA work. A decode worker receives raw encoded
bytes and calls the existing `ImageMediaIO.load_bytes_many` method, which creates
all `CodeStream` objects and borrows a retained decoder slot on that same worker.
No nvImageCodec object crosses a thread boundary. Direct jobs and retries also
run on a claiming service worker; they never resubmit to this executor and wait
on it.

### Connector Dispatch

After asynchronous URL/data-URL fetching:

```text
effective Pillow backend
    -> prior scalar async fetch/decode gather

nvImageCodec, one JPEG, batch_size > 1
    -> submit to process-local decode service's coalescing queue
    -> await wrapped future without occupying global_thread_pool

nvImageCodec, all other inputs
    -> submit one direct plural job to the same decode service
```

The base scalar `fetch_image` and `fetch_image_async` methods delegate to the
same plural path, so public scalar calls receive the same scheduling behavior.
Registered connector overrides retain their historical scalar semantics.

The synchronous connector uses the same service and waits on its future. This
allows coalescing when multiple synchronous caller threads exist. A single
synchronous call pays only the bounded coalescing interval.

Do not submit a service wait back to `global_thread_pool`. That pool is also used
for audio, video, media cache I/O, and ordinary Pillow work.

### Initial Eligibility and Compatibility

Cross-request v1 accepts exactly one encoded JPEG from a request. A cheap JPEG
SOI check selects candidates; normal Pillow and nvImageCodec inspection still
perform authoritative validation in the worker. Corrupt data therefore follows
the normal positional error path.

Configure `decoders`, `batch_size`, `pipeline_depth`, `coalesce_timeout_ms`,
admission limits, and the owning PID once per process. They are not
compatibility-key variants. Any later mismatch is rejected before enqueueing.

Use a frozen, value-based `ImageDecodeSpec` as the compatibility key. It contains
every per-call value that affects output semantics:

- target `image_mode`;
- RGBA background color; and
- an explicit version tag for the decode semantics.

Do not use `ImageMediaIO` object identity as a key. Only entries with identical
specs can share a worker call. If an `ImageMediaIO` contains an unknown extra
keyword, submit it as a direct job until that keyword is classified and added to
the frozen spec. JPEG output is RGB on the hardware path; CMYK/YCCK semantic
checks remain authoritative. CPU-plugin formats, JPEG 2000, HTJ2K, and TIFF
retain the current within-request native batching and fallback behavior. A later
measured extension can add a codec-aware key for them.

### Dispatch Policy

For each compatibility queue:

1. Dispatch immediately when at least `batch_size` entries are waiting.
2. Claim at most `batch_size * pipeline_depth` compatible entries. Do not wait
   for that maximum after one native batch is ready.
3. Otherwise dispatch the oldest partial batch when its coalescing deadline
   expires.
4. Never dispatch more than `decoders` jobs concurrently.
5. Account coalesced and direct jobs against the same `decoders` limit and select
   fairly between the direct queue and the oldest ready compatibility head.
6. Preserve FIFO order within a compatibility key and choose the oldest ready
   head across keys.
7. Skip a cancelled entry that has not been claimed. Finish a claimed native
   batch even if its requester is later cancelled.

Add startup-only `pipeline_depth` and `coalesce_timeout_ms` image options.
Request-level overrides are stripped just like `backend`, `decoders`, and
`batch_size`. Do not choose a non-zero coalescing default from intuition. Sweep
`0`, `0.1`, `0.25`, `0.5`, and `1.0` ms on the A100. A zero timeout still
combines entries already queued when the dispatcher runs and adds no
intentional low-QPS delay.

### Decode Ring and Buffer Ownership

`pipeline_depth` bounds the number of native GPU batches associated with one
decoder worker call. Depth one preserves the pre-ring pageable-copy path. The
default depth four keeps a wider bounded work window: while completed batches
are converted to Pillow images on the CPU, a following batch can decode and
copy on a distinct external CUDA stream. Other depths remain available for
measurement but change the maximum live decoded and pinned-host raster bytes.

Each ring submission follows this ownership sequence:

1. Acquire the decoded-raster byte lease and submit one native batch on its ring
   stream.
2. Create DLPack views, enqueue nonblocking copies into pinned CPU tensors on
   that stream, and record a completion event.
3. Drain the oldest entry by synchronizing its event. Only then drop its device
   images and DLPack views and release its GPU lease.
4. Attempt to refill the freed ring entry before copying the ready host pixels
   into Pillow-owned memory. Plugin misses are retained positionally and use the
   existing CPU-plugin fallback after the GPU ring drains.

Only the first acquisition made with no pending entry may block. Every attempt
made while the worker owns a lease uses atomic nonblocking admission. If the
pool fits only one native batch, the worker drains that entry, releases it, and
then refills the ring; it never waits for bytes that it must itself release.
Exception cleanup synchronizes submitted work, drops every device reference,
and returns every lease before invalidating the decoder slot. Tests inject
metadata, stream-construction, native-decode, pinned-copy, event, and Pillow
conversion failures and verify this order.

Pending work must be bounded. Limit queued entries and encoded bytes using a
single process-wide cap across every compatibility and direct queue:

- `max_pending_items = max(4, pipeline_depth) * decoders * batch_size`; and
- `max_pending_encoded_bytes = max(4, pipeline_depth) * decoders *
  NVIMAGECODEC_MAX_ENCODED_BYTES`.

The first implementation can revise the multiplier only with memory and overload
measurements. Submission must never wait on a `threading.Condition` on the
event-loop thread. The service therefore has a second FIFO admission backlog,
with the same item and byte caps, and promotes it as capacity becomes available.
The asynchronous connector first limits active fetch/decode work to the two
tiers' combined item capacity. Excess requests park on lightweight FIFO tickets
before URL, file, or base64 fetching begins. If exact encoded-byte pressure
still fills both tiers, the already-admitted caller parks on a byte-free FIFO
ticket; the next ticket atomically reserves exact item and byte capacity before
waking. Synchronous callers remain fail-fast with a typed overload error so
shared media-executor threads cannot block on admission. Neither path bypasses
decoder accounting or occupies the shared media executor while waiting.

### Results, Ordering, and Errors

Each pending entry records its request-local position and owns one future. A
worker partitions the positional return from `load_bytes_many` back to those
futures. The internal nvImageCodec `None` result continues to make
`ImageMediaIO` decode only that item with Pillow before the service receives the
final `MediaWithBytes` value. Successful values keep their original encoded
bytes and effective `io_config`, so multimodal hashing remains backend-sensitive.

Define exception types before adding retries:

- `ImageBatchItemError(index, cause)` identifies input, conversion, or capacity
  failure owned by one entry;
- `NvImageCodecIsolatableBatchError` is restricted to documented native errors
  that can be caused by one malformed stream; and
- `NvImageCodecServiceError` covers missing configuration or memory pool,
  process/PID mismatch, decoder construction, result-contract violations, CUDA
  failures, and other systemic errors.

If a combined call reports an indexed item failure, fail that future with the
original exception and retry the remaining entries inline as a batch. Each pass
must remove an offender, so attempts are bounded by the original width. Only an
explicitly typed isolatable batch error permits one inline singleton attempt per
entry. Generic or service exceptions fail the whole claimed batch immediately;
they are never multiplied into singleton retries. An individual over-capacity
image remains a hard error for its owner without poisoning other requests.

At claim time, the worker calls `Future.set_running_or_notify_cancel`. A false
return removes that entry and releases its byte accounting. The async adapter
shields the wrapped concurrent future. On coroutine cancellation it first tries
to cancel unclaimed work; if the work was claimed, it installs a completion
callback that closes `MediaWithBytes.media` instead of orphaning the Pillow
image. Every enqueue/claim/result cancellation race is tested.

All successful images that cannot be delivered must be closed by their current
owner. Every queue, retry, cancellation, and shutdown path must leave queue-byte
accounting, decoder-slot count, and GPU byte lease balanced.

### Lifecycle

Initialize the service lazily only after renderer startup has installed the
per-process GPU memory pool. Record the creating PID in both the service and the
backend/decoder pool, and check it before every native path, including
`batch_size=1`, direct multi-image calls, and test helpers. vLLM uses spawned API
server processes. On a PID mismatch, inspect only lock-free pristine-state flags:
replace never-started state with fresh child objects, but fail closed if a
dispatcher, executor, decoder slot, CUDA stream, or pool lock may have been used.
Never acquire an inherited parent lock while deciding. The error must require
spawn rather than attempting to reuse native state.

Renderer initialization acquires a process-service lease, and
`BaseRenderer.shutdown` releases it. The last lease performs an idempotent,
callable shutdown: stop accepting entries, cancel and remove queued work while
releasing byte accounting, drain claimed work, join the dispatcher, shut down
decode workers, then invalidate retained decoder slots and CUDA streams. A
single `atexit` hook is only a fallback. A fully drained last-lease shutdown may
create a fresh generation later in the same PID; partial shutdown state is not
restartable. Reconfiguration within a live generation remains an error.

## Testing Plan

### CPU and Deterministic Unit Tests

- Enqueue five compatible singleton submissions behind a dispatcher barrier,
  then release it and assert `_decode_native` receives exactly width five. Do
  not assert only the queue-drain width.
- A partial queue flushes at a fake-clock deadline without sleeping in tests.
- `batch_size=1` bypasses intentional coalescing.
- Incompatible image modes and background colors never share a batch.
- Coalesced and direct jobs share one worker limit and make fair progress.
- FIFO mapping is preserved across two simultaneous worker batches.
- One indexed bad image fails only its owner; the remaining entries retry and
  keep their original order.
- A typed isolatable failure performs at most one inline singleton attempt per
  item; configuration, capacity, CUDA, and generic failures are not retried.
- Service-owned queue entries and encoded bytes remain bounded during async
  admission without blocking the event loop. The connector separately caps the
  number of images allowed to fetch or decode; excess online requests park
  before fetching. Exact-byte FIFO tickets handle residual pressure, and both
  layers recover after completion or cancellation.
- Cancellation races at enqueue, claim, and result publication either skip work
  or close an undeliverable `MediaWithBytes.media` exactly once.
- Startup configuration cannot be overridden by request kwargs or reconfigured
  after first use.
- A PID change before initialization creates pristine child state; a PID change
  after any thread, native slot, or CUDA state was used fails before taking an
  inherited lock on every decode entry point.
- Submit-versus-shutdown, claim-versus-shutdown, and completion-versus-shutdown
  races leave no queue entries, admission bytes, active slots, or GPU leases.
- A depth-two ring submits its replacement before Pillow materialization,
  preserves result order across wraparound, and drains rather than blocking when
  the GPU byte pool fits only one native batch.
- Setup, submission, event, pinned-copy, and Pillow-conversion failures
  synchronize submitted work, drop device references before releasing leases,
  close earlier Pillow results, and invalidate the affected decoder slot.
- Last renderer release permits a clean new generation in the same process.
- Pillow backend execution never touches the nvImageCodec decode service.

Use barriers, fake futures, and a fake monotonic clock. Do not make unit tests
depend on thread scheduling or wall-clock sleeps.

### Connector and Renderer Regressions

- Eight Pillow images again run through independent async scalar loads.
- Base connector plural and scalar nvImageCodec methods use the decode service;
  registered connector overrides keep their historical scalar behavior.
- Sync and async errors preserve non-zero positions and exception identity.
- Kimi-K3 retains static backend/decoder/batch/timeout values while applying
  `image_mode=None` in online and offline flows.
- The standalone utility ignores the server-only environment backend.
- The multimodal cache cannot return Pillow pixels for an nvImageCodec key or
  vice versa.

### CUDA Functional Tests

- Compare Pillow and nvImageCodec output for JPEG, CMYK JPEG, licensed YCCK,
  JPEG 2000, HTJ2K, TIFF, PNG, BMP, PNM/PBM, and WebP using the existing codec
  tolerances.
- Assert EXIF orientation is applied exactly once and that native decode params
  disable automatic EXIF and arbitrary-depth conversion.
- Exercise width-one, partial, and width-five batches with one and two decoder
  slots, including at least two native chunks through the real pinned ring.
- Disable Pillow JPEG 2000 pixel decode and prove native J2K/JP2/HTJ2K still
  succeeds through `ImageMediaIO`.
- Inject a plugin miss, decoder construction failure, host-copy failure, and an
  over-capacity image; check fallback/error ownership and resource recovery.
- On a multi-GPU host, start from a non-zero current device and verify both the
  device and stream are restored.

### Performance Validation

Use one A100 setup and fixed inference/model settings for the primary report:

- one image per request;
- output sequence length 128;
- 1920x1080 and 3840x2160 JPEG inputs;
- Pillow baseline, pre-ring nvImageCodec batch five, and nvImageCodec batch
  five with `pipeline_depth=4`;
- a concurrency/QPS sweep from low load through a sustained decode backlog; and
- both real inference and a null-inference/decode-bottleneck harness.

Collect:

- requests/s, images/s, and Gpixel/s;
- end-to-end latency and TTFT p50/p95/p99;
- decode service time and queue delay;
- service claim-width and native decode-width histograms;
- process CPU utilization and CPU-seconds per image;
- NVJPG average, peak, and time distribution; and
- GPU-memory-pool high-water mark and decoder-slot occupancy.

Also rerun decode-only tests and an eight-image Pillow parent-versus-branch
comparison at both resolutions. Use the local RTX 3080 Ti for correctness and
smoke coverage, but use A100 as the reference for five-engine NVJPG claims.

## Acceptance Criteria

- Default Pillow eight-image throughput is within noise of the parent commit and
  does not serialize decoding in one worker.
- With at least five compatible singleton requests queued, one native call has
  width five and each result returns to the correct request.
- Under sustained compatible backlog, full-width calls dominate the batch-width
  histogram; partial batches occur only at startup, drain, timeout, or
  cancellation boundaries.
- At low QPS, added p99 latency is bounded by the configured timeout plus
  scheduling noise.
- At high QPS, decode-only and null-inference throughput improve over the current
  native-width-one path, with lower CPU-seconds per image. Interpret the
  device-global NVJPG duty cycle alongside native batch widths and throughput;
  it is not a per-engine occupancy counter. Record the result even if real
  OSL-128 inference hides the gain.
- Every supported codec and fallback path remains equivalent to Pillow within
  its established tolerance.
- No request-level configuration can change retained decoder resources or queue
  policy.
- Cancellation, malformed input, capacity errors, and shutdown do not leak
  images, decoder slots, threads, futures, or GPU memory leases.

## Delivery Sequence

1. **Stabilization change:** fix Pillow dispatch, Kimi-K3 composition,
   standalone-helper policy, positional errors, image device restoration,
   YCCK/HTJ2K coverage, dependency policy, and the small cleanups above. Re-run
   the current CPU and A100 CUDA suites plus the Pillow regression benchmark.
2. **Cross-request batching change:** add the connector-level queue, dispatcher,
   worker isolation, observability, and deterministic tests with an experimental
   timeout default of zero. The A100 sweep retains zero as the latency-safe
   production default and identifies 0.25 ms as the high-QPS tuning point.
3. **Decode-ring change:** add bounded pinned staging, asynchronous copies, and
   event-proven early lease release as an independently revertible commit on top
   of cross-request batching.
4. **Measured ring tuning:** make decoder-slot provisioning deterministic and
   use `pipeline_depth=4`, selected by the fresh-process A100 depth sweep, as
   the high-backlog default.
5. **GPU-resident preprocessing change:** optionally eliminate the host transfer
   by changing the downstream multimodal contract. This remains a separate PR.
6. **Optional independent cleanups:** generic slot-pool implementation,
   no-EXIF Pillow copy removal, sync URL parallelism, and mixed CPU/GPU image
   prioritization.

The stabilization, cross-request batching, and decode ring are implemented as
successive commits. The instrumented A100 timeout and pipeline-depth sweeps are
complete; their default/tuning decisions are recorded above.
