# Stratum Port TODO — RoundPipe → Multi-GPU Stratum

**Principle:** Stratum is a heavy derivative of RoundPipe oriented towards
spanned-weights multi-GPU training. The target is feature parity with every
RoundPipe capability that is compatible with, or usefully adaptable to,
Stratum's multi-GPU staged architecture. Do not discard a RoundPipe mechanism
just because the exact implementation is tied to RoundPipe's single-runtime
async scheduler; first decide whether the user-visible capability should be
ported directly, adapted, or explicitly rejected with evidence.

## Status Legend
- [ ] not started
- [~] in progress
- [x] done
- [A] adapt — RoundPipe implementation does not fit directly, but the capability is still valuable for Stratum
- [/] rejected — incompatible or no practical value after adaptation analysis

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

## 6. Selective Volta Attention Patching

- [x] `_patch_lfm25_attention(model, layer_indices=None)` — selective
- [x] `_patch_qwen35_attention(model, layer_indices=None, window_size=None)` — selective + sliding window
- [x] `Qwen35VoltaAttention.__init__` accepts `window_size`
- [x] Sliding window passthrough to flash-attn-v100

## 7. Data Loading Flags

- [x] `--longest-first` — sort descending
- [x] `--pad-to-length N` — exact length padding
- [x] `--no-save` — skip final save
- [x] `--dense-attention-masks` — passed to build kwargs
- [x] `--pad-to-multiple` — already existed

## 8. NF4 Refinements

- [x] `NF4Stats` dataclass
- [x] `prepare_nf4()` returns `NF4Stats`
- [x] `_pin_cpu()` helper (dedicated pinned copy)
- [x] `estimate_module_upload_gib()` — NF4-savvy size estimation
- [x] `NF4Payload` dataclass (was already dataclass from prior fix)

## 9. Checkpoint Format — `stratum/checkpoint.py`

- [x] **Save portable PEFT LoRA adapter alongside per-device optim state**
  - [x] `hf_model.save_pretrained(out_dir)` → `adapter_model.safetensors` + `adapter_config.json`
  - [x] Keep per-device `optim_{id}.pt` for optimizer state (resume)
  - [x] Load via `safetensors.torch.load_file() + hf_model.load_state_dict(strict=False)`

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

- [A] Add size-aware chunked copy for non-NF4 permanent uploads in
  `ModelArch.build()` and any future FP16/`--no-nf4` path.
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

- [A] Keep a Stratum-local equivalent ready for any async stage prefetch or
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

- [A] Extend `HostStagingPool.transfer()` to synchronize destination stream
  correctness explicitly or return an event that callers must wait on.
- [A] Implement generic `async_h2d` / `async_d2h` helpers for Stratum internal
  tensors, using pinned fallback and stream/event ordering.
- [A] Use these helpers for boundary hidden-state transfers and future
  activation offload.

## 16. `upload_layers` / `download_layer` — RoundPipe's Upload Cycle

Source: `roundpipe/transfer.py:145-334`

`upload_layers`: Copies layers onto target device with chunked upload
for each param. Creates module/param/buffer copies. Used by RoundPipe
to bring frozen+trainable params to GPU for each forward.

`download_layer`: Async gradient download from GPU back to CPU.
Used to free GPU memory after backward, grads saved on CPU for optimizer.

**Parity decision:** ADAPT.

`ensure_weights()` / `free_weights()` covers frozen NF4 streaming, but does
not cover RoundPipe's full memory behavior: independent module copies,
buffer snapshots for recompute, async gradient download, reusable grad CPU
buffers, or CPU optimizer storage.

- [A] Add optional CPU/offloaded trainable-gradient mode for LoRA params:
  download grads after backward, keep optimizer state on CPU or a selected
  device, and re-upload updated trainable params before the next forward.
- [A] Add buffer snapshot/restore support for recompute paths that mutate or
  depend on buffers.
- [A] Add an async stage prefetch experiment: upload next stage's NF4 payloads
  while the current stage computes, with event fencing before use.
- [/] Keep permanent in-place `DeviceStage` modules as the default; full
  RoundPipe module-copy semantics are expensive and not required for the
  current staged model layout.

## 17. `ModelExecutePlan` — Execution Planning

Source: `roundpipe/scheduler.py`

Forward/backward execution plans with per-layer scheduling. Supports
multi-stage pipelining, fused-mode, and memory-budgeted planning.
`--roundpipe-model-memory-limit-gib` controls the memory budget.

**Parity decision:** ADAPT.

`assign_layers_to_devices()` is a placement algorithm, not an execution
planner. RoundPipe's planner provides memory-budgeted stage grouping and
time-informed balancing; Stratum still needs analogous capabilities.

- [A] Add a Stratum stage planner that can split a physical device's assigned
  layer range into sub-stages for upload/free granularity.
- [A] Add `--stratum-stage-memory-limit-gib` as the Stratum replacement for
  `--roundpipe-model-memory-limit-gib`.
- [A] Add per-layer/stage timing and feed that into future automatic
  placement beyond raw VRAM ratios.
- [/] Do not port RoundPipe fused-mode verbatim; Stratum's prefix/stage/postfix
  topology needs a separate design.

## 18. CLI Flags — still missing from train.py

From RoundPipe scripts that Stratum doesn't have yet:

- [x] `--pin-model {alloc, register, off}` — pinning strategy
- [x] `--torch-compile-loss` — enable torch.compile on CE kernels
- [x] `--lora-target-set {all, attention, attention_input, mlp}` — LoRA module targeting

## 19. Docker Build + Test

- [ ] Build Docker image: `docker build -t stratum:latest .`
- [ ] Test: 5 steps, batch=2, num_microbatch=2, tensor_split=[9,32]

## 20. Batch API Parity — `roundpipe/batch.py`

RoundPipe supports arbitrary pytree inputs, automatic split spec inference,
custom split/merge functions, `AvgReducer`, and `RoundPipePackedData` for
chaining pipeline outputs across RoundPipe calls.

Current Stratum only handles the fixed training batch shape:
`input_ids`, `attention_mask`, `labels`, with manual slicing in `train.py`.

- [A] Move microbatch splitting into a reusable Stratum helper instead of
  open-coded slicing in `scripts/train.py`.
- [A] Add split/merge hooks or specs for non-standard inputs so Stratum can
  support eval/debug calls and future model wrappers without hardcoding.
- [A] Add a reducer abstraction for losses/outputs; default training should
  keep token-weighted loss semantics, not blindly average per-microbatch losses.
- [/] Do not require CPU-only user inputs like RoundPipe; Stratum's current
  explicit input-device placement is acceptable, but the splitter should not
  make that harder to change later.

## 21. Recompute Context + RNG Parity — `roundpipe/context.py`, `run.py`

RoundPipe exposes `save_for_recompute()` / `get_recompute_data()` and preserves
CPU/CUDA RNG state for recomputation. Its model wrappers avoid rebuilding
causal masks/RoPE during recompute by saving those non-grad tensors.

Current Stratum uses PyTorch checkpointing in some wrappers, but lacks a
generic recompute context and Qwen35 does not implement decoder-layer
checkpointing at all.

- [x] Add Qwen35 `checkpoint_decoder_layer` support matching LFM25.
- [A] Add a lightweight Stratum recompute context for non-grad per-layer data
  such as causal masks, RoPE tensors, router token counts, and MoE metadata.
- [A] Preserve RNG state explicitly for any custom recompute path that is not
  delegated to `torch.utils.checkpoint`.
- [A] Audit LFM25/Qwen35 prefixes against RoundPipe's `doing_recompute()` fast
  path; avoid recomputing expensive masks/position embeddings during backward
  when equivalent saved data can be used safely.

## 22. Optimizer Stream / CPU Optimizer Parity

RoundPipe has a background optimizer stream, optimizer-owned parameter copies,
async `step()`, `synchronize()`, and a GradScaler designed for that async
optimizer path.

Current Stratum's `PerDeviceOptimizer` is synchronous and keeps trainable
params, grads, and optimizer state on GPU.

- [A] Add an optional CPU/offloaded optimizer mode for LoRA params. This is
  valuable on low-VRAM devices even if slower.
- [A] Add optional async optimizer stepping only after CPU/offload semantics
  are correct; otherwise it mainly adds race risk.
- [A] Add AMP/GradScaler support if Stratum starts using scaled mixed-precision
  training beyond FP16 model weights.
- [/] Keep the current synchronous per-device AdamW path as the default until
  async/offloaded optimizer behavior is validated.

## 23. Timing / Profiling Parity — `roundpipe/timer.py`, `profile.py`

RoundPipe records CUDA-event timings for forward, recompute, and backward,
then smooths estimates for scheduler decisions.

Current Stratum logs step-level throughput and optional allocator telemetry,
but it does not produce per-layer or per-stage timing estimates.

- [x] Add `TimingRecorder` instrumentation with CUDA events around prefix,
  stages, postfix, NF4 upload/free, and boundary transfers.
- [x] Emit timing to JSONL via `--timing-jsonl` so long runs can tune tensor
  split and stage memory limits empirically.
- [A] Add per-layer wrapped-layer timing if stage-level timings are too coarse
  for planner decisions.
- [A] Feed timing into future automatic placement/stage planning.

## 24. Model Adapter Parity

RoundPipe includes adapters for Llama, Qwen3, Qwen3-MoE, and GPT-OSS, including
MoE router logits/loss handling and optimized expert routing paths.

Current Stratum registers only `lfm25-8b-a1b` and `qwen3.5`.

- [ ] Add a generic Llama-family adapter from RoundPipe's `llama.py` as the
  baseline for non-LFM/non-Qwen models.
- [ ] Add Qwen3 adapter parity separate from Qwen3.5 if HF class names and
  attention/mask behavior differ.
- [A] Add MoE/router-loss tuple support before adding Qwen3-MoE or GPT-OSS
  adapters. The current fixed 7-tuple cannot carry router logits cleanly.
- [A] Port RoundPipe's optimized MoE expert token-count recompute pattern for
  any Stratum MoE adapter.

## 25. Correctness / Sharp-Edge Backlog

These came from the audit and are not RoundPipe feature gaps, but they affect
parity quality.

- [x] Fix `stratum/__init__.py` exporting `"upload_to_device"` even though no
  such symbol is imported.
- [x] Validate `HostStagingPool.transfer()` destination-stream ordering. It
  currently launches H2D on a side stream and returns without an explicit wait
  by the default compute stream.
- [ ] Verify Qwen35 prefix/mask behavior against RoundPipe Qwen3: RoundPipe
  can build full/sliding causal masks, while Stratum mostly passes `None` or
  raw `attention_mask`.
- [ ] Re-check PEFT save/load after pipeline build: stage layers share base
  parameter objects, but prefix/postfix are deep copies. This is intentional
  for frozen weights, but any future trainable prefix/postfix params would need
  explicit handling.
- [~] Add lightweight unit tests for `assign_layers_to_devices`, NF4 payload
  lifecycle, checkpoint metadata, and microbatch loss normalization. Current
  validation is mostly GPU smoke scripts.

## 26. Implementation Order

Recommended order for reaching practical parity:

1. Fix small correctness issues: stale export, Qwen35 checkpointing, transfer
   stream wait, unit tests.
2. [x] Port RoundPipe's custom chunked linear CE autograd behavior, because this
   directly affects long-context VRAM.
3. [x] Add Stratum timing instrumentation so later scheduler work is evidence-based.
4. Add stage-memory-limit planning and intra-device sub-stages.
5. Add generic microbatch split/reduce helpers.
6. Add async transfer helpers and event fencing; then experiment with NF4
   prefetch.
7. Add CPU/offloaded optimizer mode.
8. Expand model adapters: Llama, Qwen3, then MoE/GPT-OSS after tuple/router
   support exists.

---

## Summary of decisions

| Section | Decision | Reason |
|---|---|---|
| 9. Checkpoint format | **PORTED** | PEFT-compatible safetensors via hf_model.save_pretrained |
| 10. Torch-compiled loss | **PORTED** | --torch-compile-loss exists and RoundPipe-style per-chunk backward custom autograd is ported |
| 11. PinnedUpload | **ADAPT** | Useful for activation/offload transfer paths, not current sync NF4 upload |
| 12. pin_model | **PORTED** | --pin-model {alloc,register,off}, both strategies in stratum/memory.py |
| 13. Chunked upload | **ADAPT** | Needed for non-NF4 and no-NF4 large tensors, not every NF4 payload |
| 14. RegisterBackwardEvent | **ADAPT** | Needed when Stratum adds async prefetch/offload; not current sync path |
| 15. async_d2h/h2d | **ADAPT** | HostStagingPool covers part of it but lacks generic helpers and full event semantics |
| 16. upload_layers/download_layer | **ADAPT** | ensure/free covers frozen NF4 only; grad/offload/buffer behavior still missing |
| 17. ModelExecutePlan | **ADAPT** | Stratum needs its own stage-memory/timing planner, not RoundPipe's exact plan |
| 20. Batch API | **ADAPT** | Current fixed train loop lacks RoundPipe's pytree split/merge/reduce flexibility |
| 21. Recompute context/RNG | **ADAPT** | Needed for Qwen checkpointing, MoE metadata, and mask/RoPE recompute avoidance |
| 22. Optimizer stream/scaler | **ADAPT** | Optional CPU/offloaded optimizer has value; async step should come later |
| 23. Timing/profiling | **PARTIAL / ADAPT** | TimingRecorder JSONL is ported; scheduler feedback remains |
| 24. Model adapters | **PENDING** | Llama/Qwen3/MoE/GPT-OSS parity not present in Stratum |
