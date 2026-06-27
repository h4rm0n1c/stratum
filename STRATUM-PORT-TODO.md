# Stratum Port TODO — RoundPipe → Multi-GPU Stratum

**Principle:** Stratum is a heavy derivative of RoundPipe oriented towards
spanned-weights multi-GPU training. The target is feature parity with every
RoundPipe capability that is compatible with, or usefully adaptable to,
Stratum's multi-GPU staged architecture. Do not discard a RoundPipe mechanism
just because the exact implementation is tied to RoundPipe's single-runtime
async scheduler; first decide whether the user-visible capability should be
ported directly, adapted, or explicitly rejected with evidence.

**Port north star:** Stratum should be able to do the useful things
qz-roundpipe/RoundPipe can do, plus the thing qz-roundpipe cannot do by
itself: keep one model's weights spanned across multiple heterogeneous GPUs and
move activations across device boundaries through the llama.cpp/TurboQuant
host-staged fallback when CUDA peer access is unavailable. The port is not a
search for the smallest set of features that makes isolated tests pass. Tests
exist to prove the port. If a qz-roundpipe feature is reachable in Stratum's
architecture, port it or adapt it; do not disable it as a workaround.

**Current validation model:** LFM2.5-8B-A1B is the primary test target until
the port is stable. Qwen35 remains important, but LFM2.5 is the reference model
for proving RoundPipe parity plus Stratum's spanned-weight execution path.

**Reference validation contract:** Stratum's practical target is the same class
of machine that motivated the qz-roundpipe work: a cheap heterogeneous local
rig, not a clean homogeneous cluster. The current reference setup is
container-local `cuda:0` as an RTX 3080-class card plus `cuda:1` as a V100 32
GiB card, with CUDA peer access false in both directions. Host-staged
cross-device transfer is therefore the primary path to keep correct and fast
enough, not an afterthought.

Multi-GPU parity work must stay aligned with that contract:
- Validate boundary transfers on no-P2P devices, preferably with
  `--tensor-split 10 32`.
- Treat `/home/harri/turboquant-work/llama-cpp-turboquant/ggml/src/ggml-cuda/ggml-cuda.cu`
  as the source-algorithm reference for host-staged fallback.
- Treat `/home/harri/qz-roundpipe/` and the extracted RoundPipe package as the
  source references for training/runtime features.
- Validate feature work first on LFM2.5 unless a change is explicitly
  Qwen-specific.
- Use `Dockerfile.refresh` for normal source/docs/script rebuilds; use the full
  `Dockerfile` only when the heavy CUDA/dependency stack changes.
- Keep docs, tests, git metadata, caches, datasets, outputs, and checkpoint
  blobs out of runtime images. Refresh builds must delete any inherited broad
  `/workspace/stratum` copy before installing the minimal runtime payload.

## Status Legend
- [ ] not started
- [~] in progress
- [x] done
- [A] adapt — RoundPipe implementation does not fit directly, but the capability is still valuable for Stratum
- [/] rejected — incompatible or no practical value after adaptation analysis

---

## Current Remaining Work — 2026-06-27 (updated)

Stratum has reached practical full-feature parity with qz-roundpipe on the
reference no-P2P RTX 3080 + V100 machine. Every RoundPipe mechanism that is
compatible with the spanned-weight multi-GPU architecture is now ported,
adapted, or explicitly skipped with justification.

The complete parity ledger is in `HANDBOOK.md` ("Parity status" table).

The remaining open items are a narrow tail:

1. [x] **Deeper scheduler overlap** — All four sub-slices done. Per-layer CUDA
   event timing is now wired end-to-end. `LayerTimingContext` /
   `IterLayerTimer` / `ModelLayerTimer` in `stratum/timing.py` mirror
   RoundPipe's `timer.py` trio. `DeviceStage.forward_range` records per-layer
   fwd/re events on the compute stream (detecting recompute via
   `doing_recompute()`). `run_explicit_group_backward` records backward events
   around `torch.autograd.backward` and attributes them proportionally to
   recompute time per layer. `StratumPipeline.set_layer_timer` attaches a
   `ModelLayerTimer`; `forward()` creates a fresh `IterLayerTimer` per step
   and calls `update_times()` to drain completed events. After warmup
   (`has_estimates()`), `get_training_estimates()` returns bias-corrected
   (fwd_ms, bwd_ms) per layer ready for
   `ModelExecutePlan.auto_from_layer_metrics()`.
   Plan adaptation is also done: `set_layer_timer(timer, adapt_every_n=N)`
   enables automatic plan rebuilding every N steps. `_try_adapt_plan()` calls
   `auto_from_layer_metrics` per-device, builds a symmetric plan via
   `from_stage_ranges`, and calls `_rebuild_from_plan()` when the plan
   changes. `_rebuild_from_plan` atomically updates all derived structures and
   emits a `scheduler_plan_adapted` log event.
   **(c) param_upstream async upload stream:** Each device gets a dedicated
   `param_upstream` CUDA stream. `_upload_group_with_fence` and
   `_upload_first_layer_with_fence` run `ensure_weights` under that stream and
   return a fence `Event`. All H2D copies in `ensure_weights` use
   `non_blocking=True`. Falls back to NF4 prefetch when CUDA is unavailable.

   **(d) Per-layer upload overlap in `forward_range`:** `DeviceStage.forward_range`
   accepts `param_stream` and `first_layer_fence`. When `param_stream` is
   provided, the first layer uses a pre-supplied fence (from cross-group
   pre-upload) or is uploaded at the start; for each subsequent layer, the
   H2D copy is submitted on `param_upstream` BEFORE the previous layer's
   fence-wait, so upload of layer i+1 runs concurrently with layer i's compute.
   `_upload_first_layer_with_fence` handles cross-group overlap. The `_run_group`
   recompute closure also passes `param_stream` so per-layer overlap applies
   during checkpoint recompute. All four sub-slices complete.

   **`set_layer_timer` wired into `train.py` (2026-06-26):** `ModelLayerTimer` is
   now created and attached after `build_pipeline` via `--adapt-plan-every N`
   (default 0 = off). Validated on LFM2.5: 5-step smoke with `--adapt-plan-every 2`
   produced `scheduler_plan_adapted` from 2 coarse groups to 24 per-layer groups
   after 2 warmup steps on no-P2P RTX 3080+V100 hardware.

2. [x] **Non-NF4 layer-copy runtime path** — `stratum.layer_transfer` ports
   chunked upload/download helpers. `prepare_fp16_staged()` now implements the
   per-step FP16 upload lifecycle for `--no-nf4` mode: frozen 2D params above
   `nf4_min_numel` are pinned on CPU and uploaded per-step via
   `copy_tensor_chunked()`; `ensure_weights()` and `free_weights()` handle both
   NF4 and FP16-staged params transparently. `registry.py:build()` calls
   `prepare_fp16_staged()` when `use_nf4=False` and skips staged params in the
   permanent Phase 2 upload. Unit tests prove the lifecycle: staging marks and
   empties params, `free_weights` clears them, and `copy_tensor_chunked` is the
   upload mechanism. Remaining: mutable-buffer snapshot/restore for recompute
   paths that mutate buffers.
3. [x] **Qwen35 stricter parity validation** — Passed 2026-06-26: 5-step batch 2
   / 2 microbatch at 8K (`--tensor-split 9 32`, `--nf4-scope layers`,
   `--cpu-offload-optim`, `--grad-scaler-enabled`, `--prefetch-nf4`,
   `--no-save`). SM86 `flash_attn` on GPU0, SM70 `flash_attn_v100` on GPU1,
   host-staged transfers both directions, all NF4 cache hits, finite loss.
   Deeper sliding-window mask audit against RoundPipe Qwen3 is future work.
 4. [x] **Generic batch and adapter parity** — Llama (`stratum/model/llama.py`),
    Qwen3 (`stratum/model/qwen3.py`), and Qwen3-MoE (`stratum/model/qwen3_moe.py`)
    adapters added 2026-06-26. Each follows the capability-dispatched flash
    attention pattern and recompute bridge. Qwen3-MoE wires the MoE router-logit
    side channel and auxiliary loss. GPT-OSS skipped (no public HF model).

    **Pytree batch API** (added 2026-06-26): `guess_split_spec`, `split_pytree`,
    `merge_pytree`, `split_kwargs_pytree`, and `TokenWeightedReducer` in
    `stratum/batch.py` port RoundPipe's `batch.py` pytree handling. `--pytree-batch`
    flag in `train.py` exercises the pytree microbatch path. `forward_batch()`
    on `StratumPipeline` accepts arbitrary pytree inputs.

    Smoked on heterogeneous no-P2P hardware 2026-06-26:
   - TinyLlama-1.1B: 3 steps, finite loss, SM86+SM70 flash dispatch
   - Qwen3-0.6B: 3 steps, finite loss, SM86+SM70 flash dispatch
   - Qwen3-30B-A3B (`qwen3-moe`): 3 steps, finite loss, aux loss wired, SM86+SM70

   **NF4 fix for 3D stacked expert weights:** `Qwen3MoeExperts` uses 3D
   `nn.Parameter` tensors (`gate_up_proj`, `down_proj`) instead of `nn.Linear`.
   `prepare_nf4` was extended from `ndim != 2` to `ndim < 2` to cover these;
   higher-rank tensors are reshaped to `[-1, last_dim]` for bitsandbytes
   quantization and `payload.shape` retains the original shape for dequant
   reconstruction. See `upload.py`.

    Remaining from this item: `RoundPipePackedData` handoff for chaining
    pipeline outputs across calls (lower priority, eval/debug paths covered).


5. **[x] Sample packing** (added 2026-06-27): `pack_samples`, `pack_collate`,
    `split_packed_batch` in `stratum/packing.py`. `--packing` flag in `train.py`.
    Flash attention dispatches to `flash_attn_varlen_func` on LFM25 and Qwen35
    when the packed cu_seqlens metadata is present. 25 unit tests.
    Smoked on LFM2.5 at batch 1 / 1 microbatch — finite loss, SM86+SM70 flash
    dispatch, host-staged transfers. LFM2.5 ShortConv layers are incompatible
    with 1D packed input; Qwen35 works cleanly.

6. **[x] Host RAM management for LFM2.5** (added 2026-06-27): `release_cached_memory()` in
    `stratum/utils.py` calls `gc.collect()` + glibc `malloc_trim(0)` to return
    cached heap pages to the OS after `prepare_nf4()` frees the FP16 model
    weights. Measurable: RSS drops from ~33 GiB to ~21.7 GiB after pipeline
    build (11+ GiB freed). `--low-rss-nf4-build` now builds the HF module
    skeleton on meta, then `build_pipeline()` streams module tensors from the
    HF safetensors checkpoint during NF4 preparation. Cached NF4 runs attach
    precomputed payloads and stream only remaining non-NF4 tensors such as
    norms/biases. Normal `train.py` remains the default full CPU FP16 path.
    2026-06-27 audit fixed the staged-load trim import, removed the hardcoded
    LFM checkpoint fallback/undefined variable, forwards `--nf4-min-numel` to
    cache loading, keeps the PEFT wrapper for checkpoint save/load, and
    preserved NF4/FP16 payload metadata across fallback parameter replacement.
    Docker LFM2.5 low-RSS NF4 smoke then passed on 2026-06-27 with
    `--host-ram-limit-gib 45`: RSS `6.48 GiB` after pipeline build,
    `7.12 GiB` after dataloader construction, finite loss `11.8045`, GPU peaks
    `3.68 GiB` / `14.36 GiB`, and host-staged transfers both directions.
    A follow-up warm-cache smoke reused
    `/workspace/cache/nf4-lowrss-fp16/LiquidAI_LFM2.5-8B-A1B`, reported
    `nf4: all payloads loaded from cache`, completed pipeline build in
    `12.3s`, and produced the same finite loss without rebuilding NF4 payloads.

7. [~] **Validation depth** — unit coverage exists for the core ported
   infrastructure, but longer LFM2.5/Qwen35 runs, resume/save-load checks under
   CPU optimizer offload, and failure-mode tests remain useful. Added
   2026-06-27 host checks for packed-mode training token accounting and
   optimizer Adam moment round-trip without legacy `device_*.pt` state.
8. [A] **NUMA-aware host coordination** — this is a future Stratum-specific
   optimization, not a current RoundPipe parity blocker. See section 16a.

### Next handover slice

Items 1–6 are complete for the LFM2.5 validation target. `set_layer_timer` is wired into `train.py`
(`--adapt-plan-every N`). The pytree batch API (`guess_split_spec` /
`split_pytree` / `merge_pytree` / `TokenWeightedReducer`) is ported and
wired via `--pytree-batch`. Sample packing (`stratum/packing.py`,
`--packing`) is implemented with flash_attn_varlen_func dispatch on
LFM25 and Qwen35. Host RAM trimming (`release_cached_memory()`) frees
11+ GiB of cached FP16 pages after normal pipeline build. The opt-in
`--low-rss-nf4-build` path passed a real LFM2.5 Docker runtime smoke with
streamed HF safetensors, NF4 cache writes/hits, LoRA meta materialization,
host RSS under 8 GiB after dataloader setup, and finite loss. Next practical
smoke: Qwen35 low-RSS NF4 build, then packed training past the step log
boundary plus an opt-in optimizer-state resume.

---

## 0. RoundPipe Source Audit

Source: `roundpipe` PyPI package v0.1.1 (extracted to `/tmp/roundpipe-dl/roundpipe_src/`)

Key internal modules discovered:
- `roundpipe/models/function.py` — ChunkedCompileLinearCrossEntropy, CompileCrossEntropy (torch.compiled), TOKEN_CHUNK_SIZE
- `roundpipe/transfer.py` — async_d2h/h2d, upload_layers, download_layer, PinnedUpload, RegisterBackwardEvent, create_upload_pair (chunked upload)
- `roundpipe/memory.py` — pin_module_alloc, pin_module_register
- `roundpipe/roundpipe.py` — RoundPipe class (forward_backward, step, synchronize)
- `roundpipe/run.py` — RoundPipeRunContext, forward/backward orchestration with async events
- `roundpipe/scheduler.py` — ModelExecutePlan, chunk_layer_params
- `roundpipe/device.py` — DeviceManager (per-device streams)
- `roundpipe/context.py` — ForwardCtx, RecomputeCtx, save_for_recompute
- `roundpipe/batch.py` — pytree-aware microbatch split/merge, packed data handoff, AvgReducer
- `roundpipe/timer.py` — CUDA-event per-layer fwd/recompute/bwd timing and smoothed estimates
- `roundpipe/optim_stream.py` — background optimizer stream
- `roundpipe/grad_scaler.py` — GradScaler compatible with async optimizer stream
- `roundpipe/models/{llama,qwen3,qwen3_moe,gpt_oss}.py` — additional model adapters and MoE/router-loss patterns

---

## 1. MLP Optimizations — `stratum/model/mlp_opt.py`

Source: `train_lfm25_roundpipe_lora.py:120-298`

- [x] `CheckpointedModule(nn.Module)` — wraps MLP in checkpoint
- [x] `TokenChunkedModule(nn.Module)` — runs MLP in token chunks
- [x] `MemoryFlatFrozenMLPFunction` / `MemoryFlatFrozenMLP` — custom autograd
- [x] `_assert_frozen_mlp(module)` — unified for LFM/Qwen/MoE
- [x] `enable_decoder_mlp_checkpointing()`, `enable_memory_flat_frozen_mlp()`, `enable_decoder_mlp_token_chunking()`
- [x] `apply_mlp_optimizations()` — dispatcher with mutex rules

## 2. Wire MLP Opts Into Architecture Builds

- [x] `LFM25Arch.build()` — calls `apply_mlp_optimizations(core, **kwargs)`
- [x] `Qwen35Arch.build()` — same

## 3. Two-Mode Chunked Loss

- [x] `BlockedPostfixCausalLMLoss` in `stratum/model/blocked_loss.py`
- [x] LFM25 postfix: split into mode 1 (norm full-seq + lm_head chunked) and mode 2 (BlockedPostfix)
- [x] Qwen35 postfix: same split
- [x] `postfix_loss_token_chunk_size` threaded through build_pipeline → build() → build_postfix()

## 4. Telemetry & Debugging

- [x] `stratum/telemetry.py` with:
  - [x] `gpu_memory_snapshot()` enriched (active/inactive/retries/ooms)
  - [x] `mark_model_gpu_phase()` — structured JSON with allocator stats
  - [x] `assert_finite_tensor()` — NaN/Inf detection
  - [x] `enable_operator_telemetry()` — per-operator hooks
  - [x] `parse_int_set()`, `parse_name_list()`
- [x] `memory_telemetry` in prefix, wrapped layer, postfix forward()
- [x] `debug_finite` calls in blocked loss
- [x] `--cuda-memory-summary-on-exception` in train.py

## 5. Memory Watchdog + Phase Tracking

- [x] `stratum/watchdog.py` with:
  - [x] `start_memory_watchdog()` — daemon thread, /proc/self/status
  - [x] `mark_phase()` — elapsed time marker
  - [x] `memory_snapshot()` — RSS/VMS/MemAvailable
  - [x] `mark_memory_phase()` — snapshot + optional abort
  - [x] `mark_gpu_memory_phase()` — GPU allocator snapshot
- [x] Wired into train.py main()

## 6. Capability-Dispatched Flash Attention Patching

- [x] `_patch_lfm25_attention(model, layer_indices=None)` — selective
- [x] `_patch_qwen35_attention(model, layer_indices=None, window_size=None)` — selective + sliding window
- [x] `Qwen35FlashAttention.__init__` accepts `window_size`
- [x] Sliding window passthrough to flash attention
- [x] `--flash-layers` selects a subset when explicitly requested; empty means all full-attention layers
- [x] LFM25 full-attention wrapper dispatches to `flash_attn_v100` on V100 and
  standard `flash_attn` on Ampere+, with hard failure instead of CUDA eager fallback.
- [x] Qwen35 full-attention wrapper dispatches to standard `flash_attn` on
  Ampere+ and `flash_attn_v100` on V100. This is required for heterogeneous
  splits where a full-attention layer lands on either the RTX 3080 or V100.
- [x] LFM25 and Qwen35 `_FlashBackend` NamedTuple pattern — structured backend
  selection with hard failure on flash kernel error.

## 7. Data Loading Flags

- [x] `--longest-first` — sort descending
- [x] `--pad-to-length N` — exact length padding
- [x] `--no-save` — skip final save
- [x] `--dense-attention-masks` — passed to build kwargs
- [x] `--pad-to-multiple` — flash attention defaults 0 to 32

## 8. NF4 Refinements

- [x] `NF4Stats` dataclass
- [x] `prepare_nf4()` returns `NF4Stats`
- [x] `_pin_cpu()` helper (dedicated pinned copy)
- [x] `estimate_module_upload_gib()` — NF4-savvy size estimation
- [x] `NF4Payload` dataclass (was already dataclass from prior fix)
- [x] `--nf4-scope {all,layers}` — keeps the existing Stratum all-module
  behavior for LFM while allowing qz-roundpipe-style decoder-layer-only NF4
  prep for Qwen35, whose embedding/head tensors can kill first-time NF4 cache
  creation on the reference host.

## 9. Checkpoint Format — `stratum/checkpoint.py`

- [x] **Save portable PEFT LoRA adapter with JSON metadata by default**
  - [x] `hf_model.save_pretrained(out_dir)` → `adapter_model.safetensors` + `adapter_config.json`
  - [x] Save `trainer_state.json` for step/format metadata
  - [x] Do not write `device_{id}.pt`, `optim_{id}.pt`, or `meta.pt` by default
  - [x] Keep legacy per-device `.pt` and optimizer `.pt` as explicit opt-in flags only
  - [x] Load via `safetensors.torch.load_file() + hf_model.load_state_dict(strict=False)`, with legacy `.pt` fallback

## 10. `ChunkedCompileLinearCrossEntropy` — torch.compiled Chunked Loss

Source: `roundpipe/models/function.py:64-115`

RoundPipe's chunked loss Function:
- Splits flattened hidden_states into TOKEN_CHUNK_SIZE chunks along dim=0
- For each chunk: `CompileLinearCrossEntropy` (torch.compiled linear + CE)
- Calls `.backward()` per chunk to accumulate lm_head gradients
- Saves accumulated grads, backward restores them
- `TOKEN_CHUNK_SIZE` is a module-level global (default 4096)
- Used by `ChunkedCompileLinearForCausalLMLoss` (user-facing wrapper)

Our current implementation:
- Splits by dim=1 (seq dimension) inside postfix, not by batch*seq
- Uses plain `F.cross_entropy` without torch.compile
- Relies on outer autograd for backward instead of per-chunk backward

**Port status:**
- [x] Add `--torch-compile-loss` flag — compiles CE in postfix `__init__` when set
- [x] Approach: compile a closure in `__init__`, use it in forward's chunked loop
- [x] Keep our current approach as the default (correct, no torch.compile dep)
- [x] Add a Stratum-native `ChunkedLinearCrossEntropyFunction` that flattens
  batch*seq, calls backward per chunk, and returns saved grads in its outer
  backward. This ports RoundPipe's missing memory behavior beyond the earlier
  forward-only logits chunking.

## 11. `PinnedUpload` Autograd Function

Source: `roundpipe/transfer.py:337-373`

Custom autograd that:
- Forward: if tensor not pinned, copies to pinned CPU → `.to(device, non_blocking=True)`
- Backward: copies gradient from GPU to pinned CPU

Makes H2D non-blocking even when caller provides pageable memory.
Backward moves gradients back to pinned CPU for async D2H overlap.

**Parity decision:** ADAPT, not skip.

The exact autograd wrapper is not useful for synchronous NF4 parameter H2D,
but the capability is still useful at two Stratum boundaries:

- [A] Introduce a small transfer/autograd utility for host-staged activation
  transfers where gradients must return to the previous device or CPU with
  pinned-memory guarantees.
- [A] Reuse the pinned fallback for pageable user inputs if Stratum gains a
  CPU-input pipeline API instead of requiring `input_ids.cuda(input_device)`.
- [/] Do not use `PinnedUpload` for NF4 payload upload unless parameter H2D
  becomes async; NF4 payloads are already pinned by `_pin_cpu()`.

## 12. `pin_module_alloc` / `pin_module_register` — `--pin-model`

Source: `roundpipe/memory.py`

Two pinning strategies:
- `pin_module_alloc`: calls `.pin_memory()` on every param and buffer
- `pin_module_register`: uses `cudaHostRegister` for zero-copy pinning
- RoundPipe scripts expose `--pin-model {alloc, register, off}`

Stratum currently pins NF4 payloads but not trainable params or general buffers.

**Port plan:**
- [x] `stratum/memory.py` — both pin_module_alloc and pin_module_register (ported from roundpipe/memory.py)
- [x] `--pin-model {alloc, register, off}` added to train.py
- [x] Called after pipeline build, before training loop

## 13. Chunked Upload (create_upload_pair)

Source: `roundpipe/transfer.py:116-142`

Splits tensors > 256 MB into chunks for async H2D overlap:
```python
CHUNK_UPLOAD_SIZE = 256 * 1024 * 1024
n_chunks = math.ceil(size / CHUNK_UPLOAD_SIZE)
chunk_nelements = math.ceil(src.nelement() / n_chunks)
# ... zip chunked sources with chunked destinations
```

Stratum currently does single-shot `param.data = dequantized.contiguous()`.

**Parity decision:** ADAPT selectively.

NF4 H2D payloads are usually too small to justify RoundPipe's 256 MiB chunking
threshold, but Stratum still has large non-NF4 tensors and dequantized
temporary weights.

- [x] Add size-aware chunked copy for non-NF4 / `--no-nf4` uploads via
  `copy_tensor_chunked` in `ensure_weights()` through the FP16-staged path.
- [A] Add a configurable chunk threshold for `ensure_weights()` only if
  profiling shows large payloads or monolithic dequant/copy stalls.
- [/] Do not blindly chunk every NF4 tensor; that adds overhead to the common
  small-payload path.

## 14. `RegisterBackwardEvent` Autograd Function

Source: `roundpipe/transfer.py:376-409`

Records a CUDA event that backward synchronizes on. Ensures async
uploads finish before backward accesses the tensor.

**Parity decision:** ADAPT if we add async transfers.

There is no current race in synchronous `ensure_weights() -> forward`, but
the event-guarding capability becomes necessary once Stratum overlaps stage
weight upload, activation movement, or optimizer copies.

- [x] Keep a Stratum-local equivalent ready for any async stage prefetch or
  async host-staged activation path.
- [/] Do not add it to the current synchronous weight path; there is no event
  to guard today.

## 15. `async_d2h` / `async_h2d` — Async Host-Device Transfer

Source: `roundpipe/transfer.py:26-113`

Async H2D/D2H with pinned buffers, stream ordering, event-based sync.
Core of RoundPipe's overlapping strategy.

**Parity decision:** ADAPT.

`HostStagingPool` covers the data path but not all RoundPipe semantics:
stream ordering, event return values, reusable async H2D/D2H helpers, and
explicit pinned fallback for pageable tensors.

- [x] Extend `HostStagingPool.transfer()` to synchronize destination stream
  correctness explicitly. The synchronous host-staged backward boundary path
  now copies H2D on the destination current stream, matching RoundPipe's
  upload-to-compute handoff contract and avoiding side-stream-produced grads
  reaching `AccumulateGrad`.
- [x] Implement generic `async_h2d` / `async_d2h` helpers for Stratum internal
  tensors, using pinned fallback and stream/event ordering.
- [x] Wire those helpers into `HostStagingPool` host-staged boundary transfers.
  Boundary activations use explicit `preserve_autograd=True` plus preallocated
  staging/output buffers so the RoundPipe-style async copy layer is shared
  without detaching the forward graph.
- [A] Reuse the same helper layer for future activation offload.

## 16. `upload_layers` / `download_layer` — RoundPipe's Upload Cycle

Source: `roundpipe/transfer.py:145-334`

`upload_layers`: Copies layers onto target device with chunked upload
for each param. Creates module/param/buffer copies. Used by RoundPipe
to bring frozen+trainable params to GPU for each forward.

`download_layer`: Async gradient download from GPU back to CPU.
Used to free GPU memory after backward, grads saved on CPU for optimizer.

**Parity decision:** ADAPT.

`ensure_weights()` / `free_weights()` covers frozen NF4 streaming.
`stratum.layer_transfer` now covers the standalone RoundPipe-style utility
behavior: independent shallow module copies, shared parameter preservation,
chunked tensor copy, optional grad upload, and grad/buffer download. It is not
yet wired into the active runtime.

- [x] Add a Stratum utility for chunked non-NF4 tensor upload (`copy_tensor_chunked`).
- [x] Add standalone layer-copy upload and grad/buffer download helpers.
- [x] Wire `copy_tensor_chunked` into the `--no-nf4` per-step weight-streaming path via `prepare_fp16_staged` / `ensure_weights` / `free_weights`.
- [x] Add optional CPU/offloaded trainable-gradient mode for LoRA params:
  download grads after backward, keep optimizer state on CPU, and re-upload
  updated trainable params before the next forward. This is implemented
  through `PerDeviceOptimizer` and validated with async optimizer + GradScaler
  LFM2.5 smokes.
- [A] Extend that trainable-gradient path into deeper per-layer download
  overlap only if Stratum ports RoundPipe's finer per-layer scheduler phases.
- [A] Add buffer snapshot/restore support for recompute paths that mutate or
  depend on buffers.
- [x] Add an async stage prefetch experiment: upload next stage's NF4 payloads
  while the current stage computes, with event fencing before use.
- [/] Keep permanent in-place `DeviceStage` modules as the default; full
  RoundPipe module-copy semantics are expensive and not required for the
  current staged model layout.

## 16a. NUMA-Aware Host Coordination — Future Stratum Extension

This is not a RoundPipe parity blocker. It is a Stratum-specific future
optimization for dual-socket hosts where frozen NF4 streaming, host-staged
activation transfer, CPU optimizer work, and pinned buffers can be coordinated
more deliberately across CPU sockets.

Current state:
- [x] Host-staged fallback uses reusable pinned buffers and is already fast on
  the no-P2P RTX 3080 + V100 reference path.
- [x] NF4 payloads are pinned on CPU and copied compressed before GPU-side
  dequantization.
- [ ] Stratum does not currently bind staging buffers, prefetch work,
  optimizer work, or allocation first-touch to a NUMA node.
- [ ] Stratum does not currently expose a CPU affinity / NUMA placement map.

Future direction:
- [A] Add startup topology telemetry: CPU sockets, NUMA nodes, GPU PCIe/NUMA
  affinity when available, peer-access matrix, and current process affinity.
- [A] Add optional affinity configuration, for example
  `--cpu-affinity-map 0=0-7,1=8-15`, with environment-variable equivalent for
  Docker runs.
- [A] Allocate per-device/per-boundary pinned staging pools while bound to the
  intended CPU node so first-touch memory placement is predictable.
- [A] Split host-side workers by role/device where useful: NF4 prefetch,
  host-staged boundary transfer, and CPU optimizer sync/step.
- [A] Extend timing telemetry to separate D2H, host fence, H2D, NF4 prefetch,
  NF4 dequant, and optimizer CPU time, so NUMA policy changes can be measured
  instead of guessed.

## 17. `ModelExecutePlan` — Execution Planning

Source: `roundpipe/scheduler.py`

Forward/backward execution plans with per-layer scheduling. Supports
multi-stage pipelining, fused-mode, and memory-budgeted planning.
`--roundpipe-model-memory-limit-gib` controls the memory budget.

**Parity decision:** ADAPT.

`assign_layers_to_devices()` is a placement algorithm, not an execution
planner. RoundPipe's planner provides memory-budgeted stage grouping and
time-informed balancing; Stratum still needs analogous capabilities.

- [x] Add a Stratum stage planner that can split a physical device's assigned
  layer range into sub-stages for upload/free granularity.
- [x] Add `--stratum-stage-memory-limit-gib` as the Stratum replacement for
  `--roundpipe-model-memory-limit-gib`.
- [x] Port RoundPipe's plan/tracker primitives as `stratum.scheduler`:
  `ModelExecutePlan`, `ModelTracker`, `BackwardScheduleSimulator`, and
  `chunk_layer_params`.
- [x] Wire `ModelExecutePlan` into Stratum's active stage runtime so forward,
  recompute, backward, upload, and host-staged boundary transfers can be
  scheduled by explicit layer groups rather than only by static
  device-assigned stages.
- [x] First runtime wiring slice: `StratumPipeline` creates a plan from its
  built `DeviceStage`s, records plan group ranges in timing spans, gates
  forward with `ModelTracker`, and exposes backward group wait/notify via
  tensor hooks.
- [x] Second runtime wiring slice: `StratumPipeline` can run multiple
  scheduler plan groups inside one physical `DeviceStage` via
  `DeviceStage.forward_range()`. Plan groups may split a stage, but cannot
  cross host-staged device boundaries.
- [x] Third runtime wiring slice: stage upload/prefetch is now scoped to the
  active scheduler group, so a split physical `DeviceStage` no longer
  materializes every layer in that stage before the first sub-group runs.
- [x] Fourth runtime wiring slice: scheduler groups now register idempotent
  backward-completion callbacks through tensor hooks. Groups whose input tensor
  has no gradient path are completed by `free_all_weights()` after backward,
  which avoids freeing streamed NF4 weights before PyTorch checkpoint recompute
  has finished.
- [x] Fifth runtime wiring slice: PyTorch checkpoint recompute is now visible
  to Stratum timing as `recompute_save`, `recompute_enter`, and
  `layer_recompute` records with global layer ids, stage device, attention
  type, recompute grain, and saved tensor/byte counts. This adapts
  RoundPipe's explicit `time_fwd("re", layer_id, ...)` recompute accounting
  while Stratum still relies on PyTorch to run the recompute.
- [x] Sixth runtime wiring slice: host-staged boundary transfers now record
  `direction=forward` and `direction=backward` timing entries, and scheduler
  groups now emit `stage_backward` spans from gradient-entry wait hook to
  backward completion/free. Fallback-completed groups are explicitly marked
  with `after_backward_fallback`.
- [x] Seventh runtime wiring slice: `stratum.runtime` now has the
  RoundPipe-style explicit scheduler-group autograd primitive:
  `capture_backward_input()` detaches a group input for recompute, and
  `run_explicit_group_backward()` recomputes the group, runs
  `torch.autograd.backward(outputs, grads)`, records recompute/backward spans,
  and returns input gradients. The primitive is stream-agnostic and tested on
  tensor pytrees. It deliberately does not wrap stage recompute in
  `RecomputeCtx` by default, because LFM/Qwen layer wrappers already use
  `doing_recompute()` for their own checkpoint side-channel data.
- [x] Eighth runtime wiring slice: `anchor_explicit_group_backward()` ports
  RoundPipe's custom autograd anchor shape for scheduler groups. It flattens
  and rebuilds arbitrary pytrees, returns detached output values to cut the
  original forward graph, uses a synthetic grad anchor so param-only groups can
  still launch backward, and preserves nested Stratum side-channel tensors such
  as router logits carried in `kwargs`. Pass-through tensor leaves from the
  group input are deliberately not replay-owned by the current group; this keeps
  earlier router logits on their original graph edge while newly appended router
  logits are still recomputed and backpropped by the producing group.
- [x] Ninth runtime wiring slice: `StratumPipeline` now wraps each active
  scheduler group with `anchor_explicit_group_backward()`. Backward waits still
  attach to group outputs, backward completion/free now fires from explicit
  group backward completion, and the old `free_all_weights()` fallback remains
  only as a safety net for groups that never receive autograd.
- [A] Extend current scheduler-group execution into RoundPipe-style custom
  async upload/recompute/backward streams with finer per-layer stream/event
  fences.
- [x] Wire the `stratum.runtime` autograd anchor around active scheduler
  groups, preserving the full Stratum 7-tuple side channels and host-staged
  boundary transfer fencing.
- [A] Add per-layer/stage timing and feed that into future automatic
  placement beyond raw VRAM ratios. Per-layer `layer_forward` and
  `layer_recompute`, group-level `stage_backward`, and bidirectional
  `boundary_transfer` timing are now recorded; per-layer backward timing and
  timing-fed automatic placement are still pending.
- [/] Do not port RoundPipe fused-mode verbatim; Stratum's prefix/stage/postfix
  topology needs a separate design.

## 18. CLI Flags — still missing from train.py

From RoundPipe scripts that Stratum doesn't have yet:

- [x] `--pin-model {alloc, register, off}` — pinning strategy
- [x] `--torch-compile-loss` — enable torch.compile on CE kernels
- [x] `--lora-target-set {all, attention, attention_input, mlp}` — LoRA module targeting

## 19. Docker Build + Test

- [x] Build refresh image from cached base:
  `docker build -f Dockerfile.refresh --build-arg STRATUM_REFRESH_BASE=stratum:refresh-base -t stratum:latest .`
- [x] Test LFM2.5: 5 steps, batch=2, num_microbatch=2,
  tensor_split=[9,32], max_seq_len=8192, save_every=5. Passed on
  2026-06-24 with finite losses, host-staged RTX 3080→V100 transfers,
  PEFT safetensors checkpoint/final save, and no legacy `.pt` checkpoint blobs.
- [x] Test the same LFM2.5 run with `--prefetch-nf4`. Passed on 2026-06-24
  with finite losses, PEFT safetensors checkpoint/final save, and a small GPU1
  peak increase from ~19.32 GiB to ~19.46 GiB.
- [x] Test LFM2.5 with CPU-offloaded optimizer, `--async-optimizer-step`,
  `--grad-scaler-enabled`, router aux loss, NF4 prefetch, and optimizer-state
  checkpoints. Passed on 2026-06-25 for 2 steps with finite losses `11.3813`
  and `10.8578`, host-staged RTX 3080<->V100 transfers, `flash_attn` on SM86,
  `flash_attn_v100` on SM70, and `optim_0.pt`/`optim_1.pt` written in both
  `checkpoint-2/` and `final/`.
- [x] Test LFM2.5 after recompute-context bridge wiring, without saving
  optimizer state or final checkpoints. Passed on 2026-06-25 for 1 step with
  finite loss `11.3813`, host-staged forward/backward transfers,
  `flash_attn` on SM86, and `flash_attn_v100` on SM70. Smoke output was
  removed after validation.
- [x] Test LFM2.5 after RoundPipe scheduler stage-boundary wiring. Passed on
  2026-06-25 for 1 step with finite loss `11.2807`, scheduler forward and
  backward wait/notify timing events, host-staged RTX 3080<->V100 transfers,
  CPU-offloaded async optimizer, GradScaler, router aux loss, NF4 prefetch,
  `flash_attn` on SM86, and `flash_attn_v100` on SM70. Smoke output was
  removed after validation.
- [x] Test LFM2.5 after per-layer stage timing wiring. Passed on 2026-06-25
  for 1 step with finite loss `11.2807` and `layer_forward` timing records for
  global layer ids `0..23` alongside scheduler and boundary-transfer events.
  Smoke output was removed after validation.
- [x] Test LFM2.5 after scheduler group-scoped upload/prefetch wiring. Passed
  on 2026-06-25 for 1 step with finite loss `11.2807`, host-staged
  RTX 3080<->V100 transfers, SM86 `flash_attn`, SM70 `flash_attn_v100`,
  group-tagged `stage_prefetch` / `stage_upload` timing records, scheduler
  events, and `layer_forward` records for global layer ids `0..23`. Smoke
  output was removed after validation.
- [x] Test LFM2.5 after scheduler group-level backward-completion/free wiring.
  Passed on 2026-06-25 for 1 step with finite loss `11.2807`, host-staged
  RTX 3080<->V100 transfers in both directions, SM86 `flash_attn`, SM70
  `flash_attn_v100`, `scheduler_backward_notify` records for both scheduler
  groups, `stage_group_free` records for the V100 group and the post-backward
  RTX 3080 fallback group, and `layer_forward` records for global layer ids
  `0..23`. Smoke output was removed after validation.
- [x] Test LFM2.5 after recompute lifecycle timing and router side-channel
  cleanup. Passed on 2026-06-25 for 1 step with finite loss `11.2807`,
  host-staged RTX 3080<->V100 transfers in both directions, SM86
  `flash_attn`, SM70 `flash_attn_v100`, 24 `recompute_save`, 24
  `recompute_enter`, and 24 `layer_recompute` records. Recompute payload
  accounting stayed constant at 3 tensors / 2,162,688 bytes per layer,
  proving `_router_logits` is no longer captured in checkpoint saved data.
- [x] Test LFM2.5 after backward boundary/group timing wiring. Passed on
  2026-06-25 for 1 step with finite loss `11.2807`, host-staged
  RTX 3080<->V100 transfers in both directions, SM86 `flash_attn`, SM70
  `flash_attn_v100`, 2 `boundary_transfer` records (`0->1` forward and
  `1->0` backward), and 2 `stage_backward` records for groups `6:24` and
  `0:6`. The RTX 3080 group remained marked `after_backward_fallback=true`,
  preserving the checkpoint-safe free behavior. Smoke output was removed after
  validation.
- [x] Test LFM2.5 router-aux explicit scheduler replay after side-channel
  ownership fix. Passed on 2026-06-26 for 1 step with finite loss `10.6231`,
  `--output-router-logits`, `--router-aux-loss-coef 0.02`, CPU-offloaded async
  optimizer, GradScaler, NF4 prefetch, host-staged RTX 3080<->V100 transfers in
  both directions, SM86 `flash_attn`, and SM70 `flash_attn_v100`. A warning-stack
  wrapper printed no `AccumulateGrad` stream-mismatch stack.
- [x] Test LFM2.5 after wiring `HostStagingPool` through
  `stratum.transfer.async_d2h/async_h2d`. Passed on 2026-06-26 for 1 step with
  finite loss `11.3813`, `--no-save`, CPU-offloaded async optimizer,
  GradScaler, router aux loss, NF4 prefetch, host-staged RTX 3080<->V100
  transfers in both directions, SM86 `flash_attn`, and SM70 `flash_attn_v100`.
  The boundary copies used graph-preserving preallocated staging/output
  buffers rather than the default detached helper mode.
- [x] Test Qwen35 Ampere flash-attention path: focused RTX 3080 probe at
  Qwen shape `(1, 7104, 16, 256)` selected standard `flash_attn` and completed
  forward+backward with ~0.49 GiB peak. Full-attention OOM is fixed.
- [x] Re-test Qwen35 `linear_attn` after CPU-offloaded async optimizer,
  GradScaler, and recompute context porting. First 2026-06-26 retest
  reproduced the old backward recompute OOM on GPU0 inside Transformers'
  `torch_chunk_gated_delta_rule`; the decisive difference from qz-roundpipe was
  that the Stratum image lacked `flash-linear-attention[cuda]`, so Qwen
  `linear_attn` used the quadratic torch fallback. After adding FLA to the
  image and doctor, Qwen35 passed 1 step at batch 1 / 8192 max sequence with
  finite loss `11.4141`, `--tensor-split 9 32`, CPU-offloaded async optimizer,
  GradScaler, NF4 prefetch, host-staged RTX 3080<->V100 transfers in both
  directions, SM86 `flash_attn`, SM70 `flash_attn_v100`, `fla 0.5.1`, GPU0 peak
  9.03 GiB, and GPU1 peak 18.93 GiB. Smoke output was removed after validation.

## 20. Batch API Parity — `roundpipe/batch.py`

RoundPipe supports arbitrary pytree inputs, automatic split spec inference,
custom split/merge functions, `AvgReducer`, and `RoundPipePackedData` for
chaining pipeline outputs across RoundPipe calls.

Current Stratum only handles the fixed training batch shape:
`input_ids`, `attention_mask`, `labels`, with manual slicing in `train.py`.

- [x] Move microbatch splitting into a reusable Stratum helper instead of
  open-coded slicing in `scripts/train.py`.
- [x] Add split/merge hooks or specs for non-standard inputs so Stratum can
  support eval/debug calls and future model wrappers without hardcoding.
  `guess_split_spec`, `split_pytree`, `merge_pytree`, `split_kwargs_pytree`,
  and `TokenWeightedReducer` in `stratum/batch.py` mirror RoundPipe's
  `guess_split_spec` / `AvgReducer` / `Batch` patterns. `--pytree-batch` flag
  in `train.py` exercises the pytree microbatch path.
- [x] Add a reducer abstraction for losses/outputs; default training should
  keep token-weighted loss semantics, not blindly average per-microbatch losses.
- [/] Do not require CPU-only user inputs like RoundPipe; Stratum's current
  explicit input-device placement is acceptable, but the splitter should not
  make that harder to change later.

## 21. Recompute Context + RNG Parity — `roundpipe/context.py`, `run.py`

RoundPipe exposes `save_for_recompute()` / `get_recompute_data()` and preserves
CPU/CUDA RNG state for recomputation. Its model wrappers avoid rebuilding
causal masks/RoPE during recompute by saving those non-grad tensors, avoiding
redundant GPU allocation during `checkpoint_decoder_layer`'s backward recompute.

Stratum uses PyTorch non-reentrant checkpointing in wrapped layers. The
RoundPipe context API is ported and adapted through PyTorch's
`checkpoint(..., context_fn=checkpoint_context_fn)` hook, so wrapped layers can
save non-grad tensors during the original forward and restore them under
`RecomputeCtx` during backward recompute.

The prefix also has the qz-roundpipe fast path, but Stratum's current
per-layer checkpointing does not recompute the prefix. That branch is kept for
future custom scheduler parity rather than counted as current memory savings.

**Port plan — `stratum/context.py`:**

Source: `roundpipe/context.py` (111 lines)

- [x] Port `ForwardCtx(save_for_recompute)` — context manager marking forward
  pass. Provides `save_for_recompute(*data)` to stash non-grad tensors.
- [x] Port `RecomputeCtx(recompute_data)` — context manager marking recompute
  pass. `get_recompute_data()` retrieves stashed tensors.
- [x] Port `doing_recompute()` / `save_for_recompute()` / `get_recompute_data()`
  module-level helpers with thread-local storage.
- [x] Add `checkpoint_context_fn()` for PyTorch non-reentrant checkpoint
  integration. This is the Stratum-native replacement for RoundPipe's custom
  backward recompute context nesting.
- [x] Add recompute lifecycle telemetry via
  `set_recompute_event_recorder()`, with per-layer `recompute_save`,
  `recompute_enter`, and `layer_recompute` timing records.
- [x] Wire prefix forward under `ForwardCtx` in `StratumPipeline.forward()`.
- [x] Wire LFM25/Qwen35 wrapped layers through `checkpoint_context_fn`.
- [x] Keep the MoE `_router_logits` accumulator out of checkpointed layer
  kwargs so recompute payloads contain only the per-layer non-grad data
  needed for replay, not the shared router-logit side channel used by the
  postfix aux loss.

**Wire into model wrappers:**

- [x] LFM25 prefix: call `save_for_recompute(causal_mask, position_embeddings, position_ids)`
  after first computation; on recompute skip rebuild and restore saved data.
- [x] Qwen35 prefix: same change.
- [x] LFM25 wrapped layer: save/restore `attn_mask`, `position_ids`,
  `position_embeddings`, and layer kwargs through `RecomputeCtx`.
- [x] Qwen35 wrapped layer: same change.

Remaining parity gap: RoundPipe's `run.py` also owns a custom recompute
scheduler with explicit RNG preservation, CPU staging of saved tensors, and
per-layer upload/download overlap. Stratum now records the checkpointed
recompute lifecycle, but still delegates RNG preservation to PyTorch
checkpointing and does not yet offload recompute saved tensors.

## 22. Optimizer Stream / CPU Optimizer Parity

RoundPipe has a background optimizer stream, optimizer-owned parameter copies
(fp32 on CPU), async `step()`, `synchronize()`, and a GradScaler designed for
that async optimizer path. This is the single largest VRAM-saving mechanism:
Adam optimizer state (momentum, variance) stays on CPU in fp32, freeing
~2× trainable-param memory on GPU.

Stratum's `PerDeviceOptimizer` now supports the CPU-offloaded path: fp32 CPU
optimizer copies, GPU-grad transfer to those copies, copyback after step, and
safe async deferral through the background optimizer stream. The next parity
gap is deeper RoundPipe-style overlap with per-layer forward fences.

**Port plan — Phase 1: CPU Optimizer Stream**

Source: `roundpipe/optim_stream.py` (85 lines), `roundpipe/attribute.py` (149
lines), `roundpipe/roundpipe.py` RoundPipeBase methods.

- [x] **`stratum/optim_stream.py`** — port `optim_stream.py` verbatim:
  daemon thread, `kernel_queue`, `launch_optim_kernel(fn, *args)`,
  `synchronize_optim()`, `shutdown_optim()` (atexit). No Stratum-specific
  changes needed; it's generic utility code.
- [x] **`stratum/attribute.py`** — port `ParamAttribute` (grad_cpu dict of
  per-layer gradient tensors, fp32 `optim` copy, `optim_grad_buffer`) and
  adapted `LayerAttribute` event-fencing utilities for future deeper overlap.
- [x] **`stratum/optim.py` — extend `PerDeviceOptimizer`**:
  - Add `optim_named_parameters()` — lazily creates fp32 CPU copies of each
    trainable param (like RoundPipe's `RoundPipeBase.optim_named_parameters()`).
  - Add `_move_grad_to_optim()` — gathers GPU gradients into fp32 CPU optim
    copies. Designed to run on the optimizer stream thread.
  - Add `sync_optim_param()` — copies updated fp32 CPU optim params back to
    GPU model params. Runs on optim stream or main thread.
  - Keep backward compatibility: synchronous mode remains the default.

- [A] **Port `RoundPipeBase.step()`** — async orchestrator:
  1. Wait for previous step to complete (`optim_updated` event)
  2. Fence layer events (param_copied, grad_copied)
  3. `launch_optim_kernel(sync_optim_param)`
  4. `launch_optim_kernel(_move_grad_to_optim)`
  5. `launch_optim_kernel(step_fn)` — the actual `optimizer.step()`
  6. Signal completion

- [x] **Wire into `scripts/train.py`**:
  - Add `--async-optimizer-step` flag (default off for now)
  - On each iteration: optionally use async step instead of synchronous
    `optimizer.step()` + `optimizer.scheduler_step()`
  - Call `synchronize_optim()` before metric logging / checkpointing
  - Add `--optim-dtype` flag (default `fp32`) for CPU param precision

## 23. GradScaler — Mixed Precision Support

Source: `roundpipe/grad_scaler.py` (283 lines)

RoundPipe has an async-aware `GradScaler` with a dual-scaler design:
`scale_scaler` (main thread, applies scale to loss) and `main_scaler`
(optimizer stream, unscales and steps). The two synchronize via events.

Stratum trains in fp16 without gradient scaling. This can silently diverge at
longer contexts or higher learning rates due to gradient underflow. Porting the
GradScaler is a prerequisite for stable fp16 training once the optimizer stream
lands (optimizer runs on CPU, needs unscaled gradients from GPU).

- [x] Port `stratum/grad_scaler.py` from `roundpipe/grad_scaler.py`:
  dual-scaler design with `scale()`, `unscale_()`, `step()`, `update()`,
  `get_scale()`, and event-based cross-thread synchronization.
- [x] Wire into `scripts/train.py`:
  - Wrap loss with `scaler.scale()` before `.backward()`
  - Call `scaler.step(optimizer)` instead of `optimizer.step()` directly
  - Call `scaler.update()` after each step
  - Add `--grad-scaler-enabled` flag (default off initially)

## 24. Additional CLI Flag Parity

From RoundPipe scripts that Stratum doesn't have yet:

- [x] `--attn-implementation flash` — Stratum owns flash dispatch for spanned
  heterogeneous runs. HF still loads with `attn_implementation="eager"` so
  Stratum can patch full-attention layers with capability-dispatched flash
  wrappers. There are no separate Stratum modes for GPU-specific flash kernels.
- [x] `--optim-dtype {fp32, fp16}` — controls the precision of CPU optimizer
  parameter copies (default fp32, matches RoundPipe).
- [x] `--nf4-layer-size-floor-gib` — scheduler hint for NF4-adjusted layer
  size estimation. Stratum applies the same floor semantics to its
  memory-budgeted stage splitter.
- [x] `--nf4-min-numel` — minimum elements for NF4 quantization (default 4096,
  matches qz-roundpipe).
- [x] `--recompute-grain {layer, none}` — controls checkpoint granularity
  (default `layer`, matching current behavior).

## 25. Model Adapter Parity

RoundPipe includes adapters for Llama, Qwen3, Qwen3-MoE, and GPT-OSS, including
MoE router logits/loss handling and optimized expert routing paths.

Current Stratum registers only `lfm25-8b-a1b` and `qwen3.5`.

- [ ] Add a generic Llama-family adapter from RoundPipe's `llama.py` as the
  baseline for non-LFM/non-Qwen models.
- [ ] Add Qwen3 adapter parity separate from Qwen3.5 if HF class names and
  attention/mask behavior differ.
- [x] Add MoE/router-loss side-channel support for the current LFM2.5/Qwen35
  adapters: router logits are captured in `kwargs`, preserved through explicit
  scheduler replay, and consumed by postfix aux loss.
- [A] Generalize that support before adding Qwen3-MoE or GPT-OSS adapters,
  since future adapters may need a less fixed pipeline tuple.
- [A] Port RoundPipe's optimized MoE expert token-count recompute pattern for
  any Stratum MoE adapter.

## 26. Correctness / Sharp-Edge Backlog

These came from the audit and are not RoundPipe feature gaps, but they affect
parity quality.

- [x] Fix `stratum/__init__.py` exporting `"upload_to_device"` even though no
  such symbol is imported.
- [x] Validate and fix `HostStagingPool.transfer()` destination-stream
  ordering. It now keeps D2H on a source side stream but enqueues synchronous
  H2D on the destination current stream so boundary backward hooks return
  gradients produced on the stream autograd consumes.
- [x] Pin explicit scheduler-group recompute/backward to the stage's CUDA
  default compute stream when tensors are CUDA, matching RoundPipe
  `run.py`'s `device.compute_stream` discipline instead of inheriting an
  arbitrary autograd engine stream.
- [x] Add explicit CUDA event fencing around CPU-offloaded optimizer gradient
  collection. `PerDeviceOptimizer.step()` records one default-stream event per
  CUDA gradient device after backward and the optimizer thread waits those
  events before `_move_grad_to_optim()` copies grads to CPU optimizer params.
  Live-parameter copyback remains fenced by `_optim_updated` before the next
  forward/checkpoint boundary.
- [x] Remove `copy.deepcopy` from prefix/postfix — now reference original model
  modules directly (aligned with qz-roundpipe).
- [x] Add `debug_finite` checks to LFM25 wrapped layers and postfix.
- [x] Add `dense_attention_masks` support to LFM25 prefix (was Qwen35-only).
- [x] Fix NF4 `ensure_weights()` shared frozen weight handling across devices —
  rematerializes from CPU NF4 payload instead of crashing.
- [ ] Verify Qwen35 prefix/mask behavior against RoundPipe Qwen3: RoundPipe
  can build full/sliding causal masks, while Stratum mostly passes `None` or
  raw `attention_mask`.
- [/] Re-check PEFT save/load after pipeline build: stage layers, prefix, and
  postfix now keep source module references so LoRA adapter save remains tied
  to `hf_model`. Shared frozen NF4 weights can rematerialize per device; any
  future trainable shared params spanning stages still need explicit
  optimizer/autograd handling.
- [x] Add lightweight unit tests for `assign_layers_to_devices`, NF4 payload
  lifecycle/cache metadata, checkpoint metadata, and microbatch loss
  normalization.
- [~] Add broader integration coverage for long-run save/resume, Qwen35
  stricter long-context shapes, and failure-mode behavior. Current deep
  validation still relies mostly on GPU smoke scripts.

## 27. Implementation Order

Recommended order for reaching practical parity with qz-roundpipe/RoundPipe:

1. [x] Fix small correctness issues: stale export, Qwen35 checkpointing,
   transfer stream wait, `deepcopy` removal, `debug_finite` wiring, NF4 shared
   weight handling.
2. [x] Port RoundPipe's custom chunked linear CE autograd behavior.
3. [x] Add Stratum timing instrumentation.
4. [x] Add stage-memory-limit planning and intra-device sub-stages.
 5. **[x] Add generic microbatch split/reduce helpers** — `guess_split_spec`,
    `split_pytree`, `merge_pytree`, `split_kwargs_pytree`, and
    `TokenWeightedReducer` in `stratum/batch.py` port RoundPipe's pytree batch
    API. `--pytree-batch` flag in `train.py` exercises the pytree microbatch
    path alongside the existing fixed-tensor default.
6. **[x] Port CPU optimizer stream** — `optim_stream.py`, `attribute.py`,
   `PerDeviceOptimizer` extensions, async `step()`, `--async-optimizer-step`,
   `--optim-dtype`. Validated on LFM2.5 with CPU offload, async step, and
   optimizer-state checkpoints.
7. **[x] Port recompute context bridge** — `stratum/context.py` plus PyTorch
   non-reentrant checkpoint `context_fn` wiring in LFM25/Qwen35 wrapped layers.
   Full RoundPipe custom scheduler RNG/offload semantics remain future work.
8. **[x] Port GradScaler** — enables stable fp16 training post-optimizer-stream.
9. **[x] Port and stage-wire RoundPipe scheduler primitives** —
   `ModelExecutePlan`, `ModelTracker`, backward tag rotation, and upload chunk
   balancing are now available in `stratum.scheduler`; `StratumPipeline` uses
   the plan/tracker at stage boundaries.
10. **[x] Add CLI flag parity** — `--attn-implementation`, `--optim-dtype`,
   `--nf4-min-numel`, `--nf4-layer-size-floor-gib`, `--recompute-grain`.
11. [ ] Expand model adapters: Llama, Qwen3, then Qwen3-MoE/GPT-OSS after
    generic router/batch side-channel support is clean.
12. [~] Expand integration tests for long-run save/resume, stricter Qwen35,
    and non-NF4 runtime paths.

---

## Summary of decisions

| Section | Decision | Reason |
|---|---|---|
| 9. Checkpoint format | **PORTED** | PEFT-compatible safetensors via hf_model.save_pretrained |
| 10. Torch-compiled loss | **PORTED** | --torch-compile-loss exists and RoundPipe-style per-chunk backward custom autograd is ported |
| 11. PinnedUpload | **ADAPT** | Useful for activation/offload transfer paths, not current sync NF4 upload |
| 12. pin_model | **PORTED** | --pin-model {alloc,register,off}, both strategies in stratum/memory.py |
| 13. Chunked upload | **ADAPT** | Needed for non-NF4 and no-NF4 large tensors, not every NF4 payload |
| 14. RegisterBackwardEvent | **UTILITY PORTED / ADAPT** | Helper exists for future async prefetch/offload; not current sync path |
| 15. async_d2h/h2d | **RUNTIME-WIRED / ADAPT** | Generic helpers are now used by HostStagingPool boundary transfers with graph-preserving preallocated buffers; future activation offload should reuse the same layer |
| 16. upload_layers/download_layer | **PARTIAL / ADAPT** | Layer transfer utilities and experimental NF4 prefetch exist; CPU-offloaded trainable grad handling is ported through PerDeviceOptimizer; deeper per-layer transfer overlap remains |
| 17. ModelExecutePlan | **PARTIAL / ADAPT** | Plan/tracker/tag/chunk primitives, stage memory splitting, stage-boundary runtime wiring, intra-stage scheduler groups, group-scoped upload/prefetch, group-level backward completion/free, per-layer forward/recompute timing, group backward timing, bidirectional boundary-transfer timing, the explicit group recompute/backward primitive, and active autograd anchor wiring are ported; custom async recompute/backward streams, deeper per-layer overlap, and timing-fed placement remain |
| 20. Batch API | **PARTIAL / ADAPT** | Fixed training tensors use token-weighted helpers; generic pytrees remain |
| 21. Recompute context/RNG | **PARTIAL / ADAPT** | ForwardCtx/RecomputeCtx ported, wired through PyTorch checkpoint context, and timed per layer; custom RoundPipe scheduler RNG/offload remains |
| 22. Optimizer stream | **PORTED / ADAPT** | CPU fp32 optim copies + async safe deferral validated on LFM2.5; deeper per-layer overlap remains future work |
| 23. GradScaler | **PORTED** | Async-aware dual scaler wired into training and scheduler skip gating |
| 24. CLI flags | **PORTED** | --attn-implementation, --optim-dtype, --nf4-min-numel, --nf4-layer-size-floor-gib, and --recompute-grain are wired |
| 25. Model adapters | **PENDING** | Llama/Qwen3/MoE/GPT-OSS parity not present in Stratum |
