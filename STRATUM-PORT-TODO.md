# Stratum Port TODO — RoundPipe → Multi-GPU Stratum

**Principle:** Stratum is a heavy derivative of RoundPipe oriented towards
spanned-weights multi-GPU training. EVERY mechanism in RoundPipe must be
ported. No filtering, no "this isn't needed."

## Status Legend
- [ ] not started
- [~] in progress
- [x] done

---

## 1. MLP Optimizations — `stratum/model/mlp_opt.py` (NEW FILE)

Source: `train_lfm25_roundpipe_lora.py:120-298`
Same classes also in `train_qwen35_roundpipe_lora.py:117-289`

### Classes to port
- [ ] `CheckpointedModule(nn.Module)` — wraps `layer.mlp` in `torch.utils.checkpoint.checkpoint()`
- [ ] `TokenChunkedModule(nn.Module)` — runs MLP forward in token chunks (positionwise safe)
- [ ] `MemoryFlatFrozenMLPFunction(torch.autograd.Function)` — custom autograd: forward runs MLP in chunks, backward recomputes one chunk at a time via `torch.autograd.grad()`
- [ ] `MemoryFlatFrozenMLP(nn.Module)` — wrapper that dispatches to the Function when hidden_states has gradients

### Validators
- [ ] `_assert_frozen_lfm25_mlp(module)` — checks frozen dense MLP structure (gate/up/down_proj, no bias, LayerNorm OK with nn.Linear)

### Patcher functions
- [ ] `enable_decoder_mlp_checkpointing(model) -> int` — wraps each MLP in CheckpointedModule
- [ ] `enable_memory_flat_frozen_mlp(model, token_chunk_size) -> int` — wraps each MLP in MemoryFlatFrozenMLP
- [ ] `enable_decoder_mlp_token_chunking(model, token_chunk_size) -> int` — wraps each MLP in TokenChunkedModule

### Mutex rules (same as RoundPipe)
- `memory_flat_frozen_mlp` requires `mlp_token_chunk_size > 0`
- `memory_flat_frozen_mlp` conflicts with `checkpoint_mlp`
- `mlp_token_chunk_size` without `memory_flat_frozen_mlp` uses TokenChunkedModule

---

## 2. Wire MLP Opts Into Architecture Builds

### LFM25Arch.build() — `stratum/model/lfm25.py`
- [ ] After `_patch_lfm25_attention(core)`, call MLP patcher based on `checkpoint_mlp` / `memory_flat_frozen_mlp` / `mlp_token_chunk_size` kwargs
- [ ] Pass kwargs through from `build_pipeline()` → `kwargs`

### Qwen35Arch.build() — `stratum/model/qwen35.py`
- [ ] Same as LFM25

---

## 3. Two-Mode Chunked Loss — `--postfix-loss-token-chunk-size`

Source: `train_lfm25_roundpipe_lora.py:591-703` (BlockedPostfixCausalLMLoss)
Source: `train_lfm25_roundpipe_lora.py:964-1077` (LFM25ForCausalLMPostfix with dual paths)

RoundPipe has TWO loss chunking modes:

| Mode | CLI flag | What it chunks |
|---|---|---|
| Standard chunked | `--loss-token-chunk-size N` (always active) | lm_head only; norm runs full sequence |
| Blocked postfix | `--postfix-loss-token-chunk-size N` (default 0=off) | norm + lm_head in blocks, saves/restores grads via CPU |

### Current Stratum state
Our postfix always does "blocked" mode (norm in chunks + lm_head in chunks).
This is MORE aggressive than RoundPipe default. We need BOTH paths.

### Work
- [ ] Add `BlockedPostfixCausalLMLoss` class to `stratum/model/lfm25.py` (port from RoundPipe L591-703)
- [ ] Add `BlockedPostfixCausalLMLoss` class to `stratum/model/qwen35.py` (port adapted for Qwen arch)
- [ ] Split LFM25Postfix into two paths:
  - `postfix_loss_token_chunk_size == 0`: norm full-seq, lm_head chunked by `loss_token_chunk_size`
  - `postfix_loss_token_chunk_size > 0`: BlockedPostfixCausalLMLoss.apply()
- [ ] Same split for Qwen35Postfix

---

## 4. Telemetry & Debugging

Source: `train_lfm25_roundpipe_lora.py:447-530` (enable_operator_telemetry)
Source: `train_lfm25_roundpipe_lora.py:580-588` (assert_finite_tensor)
Source: scattered `mark_model_gpu_phase()` calls

### Memory telemetry at boundaries
- [ ] `memory_telemetry` flag on prefix → `mark_model_gpu_phase("prefix_enter")`, etc.
- [ ] `memory_telemetry` flag on wrapped layer → `mark_model_gpu_phase("layer_enter")`, etc.
- [ ] `memory_telemetry` flag on postfix → `mark_model_gpu_phase("postfix_enter")`, etc.

### Operator telemetry
- [ ] `enable_operator_telemetry(model, layer_indices, module_names)` — registers forward_pre/forward/backward_pre/backward hooks on named submodules
- [ ] `mark_model_gpu_phase()` — prints structured JSON with allocator stats

### NaN/Inf detection
- [ ] `assert_finite_tensor(name, tensor)` — raises `FloatingPointError` on non-finite
- [ ] `debug_finite` flag in prefix/layer/postfix — call assert_finite after norm, loss, layer output

### CUDA memory summary on exception
- [ ] In train.py, catch `RuntimeError` during forward/backward and print `torch.cuda.memory_summary()` when `--cuda-memory-summary-on-exception` is set

---

## 5. Memory Watchdog + Phase Tracking

Source: `train_lfm25_roundpipe_lora.py:315-396` (memory helpers)

- [ ] `start_memory_watchdog(host_ram_limit_gib, interval_sec=1.0)` — daemon thread reading `/proc/self/status VmRSS`, calls `os._exit(137)` when exceeded
- [ ] `mark_phase(name)` — prints `{"phase": name, "elapsed_sec": ...}`
- [ ] `memory_snapshot()` — returns `{"rss_gib", "vms_gib", "mem_available_gib"}`
- [ ] `mark_memory_phase(name, host_ram_limit_gib)` — prints snapshot, aborts if over limit
- [ ] `gpu_memory_snapshot()` — enriched version with `active_bytes`, `inactive_split_bytes`, `alloc_retries`, `cuda_ooms` (port from RoundPipe L354-383)
- [ ] `mark_gpu_memory_phase(name)` — prints GPU snapshot

---

## 6. Selective Volta Attention Patching

Source: `train_lfm25_roundpipe_lora.py:1362-1376` (volta patching with `--volta-layers`)
Source: `train_qwen35_roundpipe_lora.py:1290-1309` (volta patching with `--volta-window-left/right`)

Current Stratum: patches ALL attention layers with VoltaFlash.
RoundPipe: allows selective patching + sliding window.

### Work
- [ ] `_patch_lfm25_attention()` — accept `layer_indices: Optional[set[int]] = None` (None = patch all)
- [ ] `_patch_qwen35_attention()` — accept `layer_indices` + `window_size: Optional[tuple[int,int]] = None`
- [ ] `Qwen35VoltaAttention.forward()` — pass `window_size` to flash_attn_v100 when set
- [ ] Pass `--volta-layers`, `--volta-window-left`, `--volta-window-right` through build kwargs

---

## 7. Data Loading Flags

Source: `train_lfm25_roundpipe_lora.py` various

- [ ] `--longest-first` — sort dataset rows by `len(input_ids)` descending before iterator
- [ ] `--pad-to-length N` — pad each batch to exact N tokens after normal padding
- [ ] `--no-save` — skip final `save_pretrained` call
- [ ] `--dense-attention-masks` — force HF `create_causal_mask` construction (our prefix currently always sets mask to None, which is the `--no-dense-attention-masks` behavior)

---

## 8. NF4 Refinements

Source: `roundpipe_nf4.py`

- [ ] `NF4Stats` dataclass with `tensors`, `source_bytes`, `payload_bytes`, `cache_hits`, `cache_misses`, `cache_writes`, `compression` property
- [ ] Return `NF4Stats` from `prepare_nf4()` instead of ad-hoc prints
- [ ] `estimate_module_upload_gib(module)` — estimate per-module GPU upload footprint respecting NF4
- [ ] `_pin_cpu(tensor)` — dedicated pinned-memory copy helper (RoundPipe's version is more robust — uses `torch.empty_like(t, device='cpu', pin_memory=True).copy_(t)`)

---

## 9. CLI Args — `scripts/train.py`

Add ALL missing RoundPipe arguments (cross-checked against both training scripts):

```
--checkpoint-mlp               (store_true)
--mlp-token-chunk-size N       (int, default 0)
--memory-flat-frozen-mlp       (store_true)
--postfix-loss-token-chunk-size N  (int, default 0)
--memory-telemetry             (store_true)
--operator-telemetry-layers S  (str, default "")
--operator-telemetry-modules S (str, default "input_layernorm,self_attn,post_attention_layernorm,mlp")
--debug-finite                 (store_true)
--cuda-memory-summary-on-exception (store_true)
--host-ram-limit-gib F         (float, default 80.0)
--volta-layers S               (str, default "")
--volta-window-left N          (int, default -1)
--volta-window-right N         (int, default 0)
--longest-first                (store_true)
--pad-to-length N              (int, default 0)
--no-save                      (store_true)
--dense-attention-masks        (store_true)
```

- [ ] Add all args
- [ ] Wire validation logic (memory_flat_frozen_mlp requires mlp_token_chunk_size > 0, conflicts with checkpoint_mlp, etc.)
- [ ] Wire all args through to build_pipeline()
- [ ] Wire watchdog, phase tracking, telemetry in main()

---

## 10. Docker Build + Test

- [ ] `cd /home/harri/stratum && git add -A && git commit -m "stratum: port remaining RoundPipe features"`
- [ ] Build Docker image: `docker build -t stratum:latest .`
- [ ] Test: 5 steps, batch=2, num_microbatch=2, tensor_split=[9,32]
