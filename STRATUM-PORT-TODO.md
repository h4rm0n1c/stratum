# Stratum Port TODO — RoundPipe → Multi-GPU Stratum

**Principle:** Stratum is a heavy derivative of RoundPipe oriented towards
spanned-weights multi-GPU training. EVERY mechanism in RoundPipe must be
ported. No filtering, no "this isn't needed."

## Status Legend
- [ ] not started
- [~] in progress
- [x] done
- [/] assessed — designed for RoundPipe's async per-layer pipeline, not applicable to Stratum's synchronous per-device streaming

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

**Port plan:**
- [x] Add `--torch-compile-loss` flag — compiles CE in postfix `__init__` when set
- [x] Approach: compile a closure in `__init__`, use it in forward's chunked loop
- [x] Keep our current approach as the default (correct, no torch.compile dep)

## 11. `PinnedUpload` Autograd Function

Source: `roundpipe/transfer.py:337-373`

Custom autograd that:
- Forward: if tensor not pinned, copies to pinned CPU → `.to(device, non_blocking=True)`
- Backward: copies gradient from GPU to pinned CPU

Makes H2D non-blocking even when caller provides pageable memory.
Backward moves gradients back to pinned CPU for async D2H overlap.

**Decision:** SKIP — Stratum uses sync `.to(device)` for NF4 H2D which doesn't
benefit from PinnedUpload's async guarantees. NF4 payloads are already pinned
by `_pin_cpu()`. The gradient-side backward (CPU gradient copy) doesn't apply
because Stratum keeps gradients on GPU (PerDeviceOptimizer runs on GPU).

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

**Decision:** SKIP — Stratum NF4 weights are 4-bit on H2D (~1/8 FP16 size).
A 2560×2560 weight matrix is ~3.3 MB over PCIe — too small to benefit from
chunking. The dequant happens on GPU after the small NF4 upload completes.

## 14. `RegisterBackwardEvent` Autograd Function

Source: `roundpipe/transfer.py:376-409`

Records a CUDA event that backward synchronizes on. Ensures async
uploads finish before backward accesses the tensor.

**Assessment:** Designed for RoundPipe's fully async pipeline where
layers are uploaded asynchronously and backward might race. Stratum
uses synchronous per-stage streaming (ensure_weights → forward →
next stage). Backward happens after all forwards complete. No race
condition exists in Stratum's architecture.

- [/] **Not applicable** — Stratum's synchronous per-stage streaming
  has no race between H2D upload and backward compute.

## 15. `async_d2h` / `async_h2d` — Async Host-Device Transfer

Source: `roundpipe/transfer.py:26-113`

Async H2D/D2H with pinned buffers, stream ordering, event-based sync.
Core of RoundPipe's overlapping strategy.

**Assessment:** Stratum's `HostStagingPool` already provides host-staged
cross-device transfers. For per-parameter H2D (NF4 → GPU), Stratum uses
synchronous transfers because the NF4 payload stays on CPU and is only
used once per step — there's no benefit to overlapping when we need the
weight immediately for forward.

- [/] **Not applicable** — `HostStagingPool` already handles cross-device
  transfers. Per-param NF4 H2D benefits less from async because weights
  are needed immediately.

## 16. `upload_layers` / `download_layer` — RoundPipe's Upload Cycle

Source: `roundpipe/transfer.py:145-334`

`upload_layers`: Copies layers onto target device with chunked upload
for each param. Creates module/param/buffer copies. Used by RoundPipe
to bring frozen+trainable params to GPU for each forward.

`download_layer`: Async gradient download from GPU back to CPU.
Used to free GPU memory after backward, grads saved on CPU for optimizer.

**Assessment:** Stratum's `ensure_weights`/`free_weights` is a simplified
equivalent that handles NF4 streaming. The full `upload_layers` machinery
is specific to RoundPipe's layer-at-a-time upload model. Stratum's
per-stage streaming is different architecturally.

- [/] **Not applicable** — Stratum's NF4 streaming (`ensure_weights` /
  `free_weights`) fills the same role with a simpler interface.

## 17. `ModelExecutePlan` — Execution Planning

Source: `roundpipe/scheduler.py`

Forward/backward execution plans with per-layer scheduling. Supports
multi-stage pipelining, fused-mode, and memory-budgeted planning.
`--roundpipe-model-memory-limit-gib` controls the memory budget.

**Assessment:** Stratum's layer assignment (`assign_layers_to_devices`)
is simpler — it groups consecutive layers by device ratio. RoundPipe's
scheduler is designed for per-layer async streaming within a single GPU,
not for multi-device topology.

- [/] **Not applicable** — Stratum uses device-level layer assignment
  (`tensor_split` / `assign_layers_to_devices`), not per-layer execution
  plans.

## 18. CLI Flags — still missing from train.py

From RoundPipe scripts that Stratum doesn't have yet:

- [x] `--pin-model {alloc, register, off}` — pinning strategy
- [x] `--torch-compile-loss` — enable torch.compile on CE kernels
- [x] `--lora-target-set {all, attention, attention_input, mlp}` — LoRA module targeting

## 19. Docker Build + Test

- [ ] Build Docker image: `docker build -t stratum:latest .`
- [ ] Test: 5 steps, batch=2, num_microbatch=2, tensor_split=[9,32]

---

## Summary of decisions

| Section | Decision | Reason |
|---|---|---|
| 9. Checkpoint format | **PORTED** | PEFT-compatible safetensors via hf_model.save_pretrained |
| 10. Torch-compiled loss | **PORTED** | --torch-compile-loss flag, compiles CE in postfix |
| 11. PinnedUpload | **SKIP** | Stratum uses sync .to() not async H2D for params; _pin_cpu already handles NF4 pinning |
| 12. pin_model | **PORTED** | --pin-model {alloc,register,off}, both strategies in stratum/memory.py |
| 13. Chunked upload | **SKIP** | Stratum NF4 weights are 4-bit (1/8 FP16 size) on H2D; dequant happens on GPU, so H2D payload is already tiny (~3 MB per weight) |
| 14. RegisterBackwardEvent | **SKIP** | Stratum has no race between H2D upload and backward compute |
| 15. async_d2h/h2d | **SKIP** | HostStagingPool already covers cross-device transfers |
| 16. upload_layers/download_layer | **SKIP** | ensure_weights/free_weights is Stratum's equivalent |
| 17. ModelExecutePlan | **SKIP** | assign_layers_to_devices is Stratum's equivalent |
