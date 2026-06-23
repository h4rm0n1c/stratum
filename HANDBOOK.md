# Stratum Handbook

Stratum is a heavy derivative of RoundPipe oriented towards spanned-weights
multi-GPU training. This document covers the architecture, key mechanisms,
design decisions, and known tricky areas for handoff to other agents.

## Quick Reference

| Item | Path |
|---|---|
| Entry point | `scripts/train.py` |
| Training config | CLI args (see `train.py --help`) |
| Dockerfile | `Dockerfile` |
| Port tracking | `STRATUM-PORT-TODO.md` |
| Data | `/home/harri/qz-roundpipe/data/lfm25_fable_merged_48k_train.labels.jsonl` |
| RoundPipe source | extracted at `/tmp/roundpipe-dl/roundpipe_src/roundpipe/` |

## RoundPipe Comparison — What's Ported and What's Different

Stratum is built from the qz-roundpipe training scripts but replaces RoundPipe's
internal runtime with a custom multi-GPU pipeline. Every mechanism that affects
**training correctness** is ported. Some RoundPipe internals are intentionally
not ported due to architectural differences.

### Training mechanisms — ALL PORTED

| Mechanism | RoundPipe | Stratum | Status |
|---|---|---|---|
| NF4 frozen weight streaming | `NF4Linear` JIT-dequant on GPU | CPU→GPU H2D + dequant per step | Different approach, same semantics |
| Chunked lm_head loss | `ChunkedCompileLinearForCausalLMLoss` | Chunked `CE(reduction="sum")` in postfix | Equivalent |
| Blocked postfix loss | `BlockedPostfixCausalLMLoss` (batch=1 only) | Same, ported to `blocked_loss.py` | Identical |
| Microbatching | `num_microbatch` in `RoundPipeRunConfig` | Manual loop in `train.py` | Equivalent |
| Activation checkpointing | `checkpoint(run_layer, ...)` per decoder layer | Same, in `LFM25ForCausalLMWrappedLayer` | Identical |
| MLP checkpointing | `CheckpointedModule` | Same, in `mlp_opt.py` | Identical |
| MLP token chunking | `TokenChunkedModule` | Same, in `mlp_opt.py` | Identical |
| Memory-flat frozen MLP | `MemoryFlatFrozenMLP` custom autograd | Same, in `mlp_opt.py` | Identical |
| Selective volta patching | `--volta-layers`, `--volta-window-*` | Same | Identical |
| LoRA adapter checkpoint | `base.save_pretrained()` (PEFT safetensors) | Same via `hf_model.save_pretrained()` | Identical |

### CLI flags — ALL PORTED

Every flag from `train_lfm25_roundpipe_lora.py` and `train_qwen35_roundpipe_lora.py`
exists in `scripts/train.py`. See CLI reference below.

### RoundPipe internals — intentionally NOT ported

These are RoundPipe's internal runtime components that Stratum replaces:

| RoundPipe internal | What it does | Why not ported | Stratum equivalent |
|---|---|---|---|
| `upload_layers()` | Copies layers to GPU with chunked async upload | Stratum uses per-stage sync streaming | `ensure_weights()` + `free_weights()` |
| `download_layer()` | Async gradient D2H after backward | Stratum keeps grads on GPU for `PerDeviceOptimizer` | Grads stay on GPU |
| `PinnedUpload` autograd | Pins tensors for async H2D, copies grads back | Stratum uses sync `.to(device)`, NF4 payloads already pinned | `_pin_cpu()` in `upload.py` |
| `RegisterBackwardEvent` | CUDA event sync for upload→backward ordering | Stratum syncs per-stage, no race condition | Not needed |
| `ModelExecutePlan` | Per-layer fwd/bwd scheduling with memory budget | Stratum groups layers by device, no per-layer plan | `assign_layers_to_devices()` |
| `DeviceManager` | Per-device stream management (upstream/downstream/compute) | Stratum uses `HostStagingPool` for boundaries, no per-param streams | `HostStagingPool` |
| `ParamAttribute` / `LayerAttribute` | Per-param upload/grad state tracking | Stratum tracks via NF4 payload attr, no upload events needed | `roundpipe_nf4_payload` attr |
| `pin_module_alloc` / `pin_module_register` | CPU memory pinning strategies | **PORTED** to `stratum/memory.py` | Same |
| `async_d2h` / `async_h2d` | Async host-device with event sync | Stratum uses `HostStagingPool` for cross-device, sync for weight H2D | `HostStagingPool.transfer()` |

### Why some things are different

The core architectural difference:

```
RoundPipe: upload layers one-at-a-time → forward one-at-a-time → backward one-at-a-time
           (async, event-driven, maximum overlap, single GPU)

Stratum:   upload all layers per stage → forward all stages sequentially → backward
           (synchronous per-stage, simpler, multi-GPU)
```

RoundPipe's design maximizes per-layer overlap by uploading/downloading
each layer asynchronously while computing the next. Stratum's design groups
layers by GPU device, streams the whole stage's weights at once, and relies
on the simpler synchronous pattern for multi-GPU correctness. The tradeoff
is less opportunity for overlap vs. simpler code that's easier to verify
across device boundaries.

### Unported features (exist only in qz-roundpipe, not needed for Stratum)

| Feature | Reason |
|---|---|
| Operator telemetry hooks | Debugging only, no correctness impact |
| Memory watchdog `/proc/self/status` | Safety guard, **PORTED** to `watchdog.py` |
| `--pin-model` strategies | **PORTED** to `memory.py` |
| `--roundpipe-model-memory-limit-gib` | RoundPipe's scheduler hint, not applicable to Stratum's device groups |
| `--dense-attention-masks` | **PORTED** (passed to build kwargs) |

## Architecture Overview

```
scripts/train.py
  │
  ├── hf_model = AutoModelForCausalLM.from_pretrained(...)
  │   └── peft_model = get_peft_model(hf_model, LoraConfig(...))
  │
  ├── build_pipeline(model_name, hf_model, tensor_split, ...)
  │   └── ModelArch.build()                                     [registry.py]
  │       ├── _patch_{lfm25,qwen35}_attention(core)              [model/*.py]
  │       ├── apply_mlp_optimizations(core, **kwargs)            [mlp_opt.py]
  │       ├── build_prefix(core) → LFM25ForCausalLMPrefix        [model/*.py]
  │       ├── build_wrapped_layer(layer, idx) → LFM25ForCausalLMWrappedLayer
  │       ├── DeviceStage(wrapped_layers, device_id)             [stage.py]
  │       ├── build_postfix(core) → LFM25ForCausalLMPostfix      [model/*.py]
  │       ├── StratumPipeline(prefix, stages, postfix)           [pipeline.py]
  │       ├── prepare_nf4(prefix|stages|postfix)                 [upload.py]
  │       ├── upload non-NF4 params to their devices
  │       └── return pipeline
  │
  └── Training loop
      ├── optimizer.zero_grad()
      ├── for each microbatch:
      │     pipeline(input_ids, ...) → loss → (loss/nmb).backward()
      ├── optimizer.step()
      ├── pipeline.free_all_weights()
      └── save checkpoint
```

## Package Map

### Core pipeline — how training works

| File | Purpose |
|---|---|
| `pipeline.py` | `StratumPipeline` — orchestrates forward through prefix→stages→postfix, boundary transfers, weight streaming |
| `stage.py` | `DeviceStage` — holds a contiguous slice of decoder layers on one GPU |
| `assign.py` | `assign_layers_to_devices()` — llama.cpp's upper_bound algorithm for layer-to-device mapping |
| `host_staging.py` | `HostStagingPool` — pinned CPU buffer for cross-device tensor transfers (P2P or host-staged) |
| `grad_hooks.py` | `make_boundary_hook()` — backward gradient hook that transfers grads across devices |

### Weight streaming (NF4) — the VRAM enabler

| File | Purpose |
|---|---|
| `upload.py` | `prepare_nf4()` → quantizes frozen 2D weights, attaches NF4Payload, drops originals. `ensure_weights()` → uploads NF4→dequant to FP16 per-stage before forward. `free_weights()` → sets param.data=empty(0) after backward. `estimate_module_upload_gib()` → NF4-savvy size estimation. |
| `nf4_linear.py` | `NF4Linear` — frozen Linear with 4-bit weight on GPU, JIT-dequant in forward. **Currently not used** — we use CPU→GPU streaming instead. |

### Model architectures — adding new models

| File | Purpose |
|---|---|
| `model/registry.py` | `ModelArch` base class, `@register("name")` decorator, `build_pipeline()` entry |
| `model/lfm25.py` | LFM2.5-8B-A1B: prefix, wrapped layer, postfix, `Lfm25VoltaAttention` |
| `model/qwen35.py` | Qwen3.5-9B: same structure, `Qwen35VoltaAttention` with sliding window |
| `model/mlp_opt.py` | Three MLP optimizations: CheckpointedModule, TokenChunkedModule, MemoryFlatFrozenMLP |
| `model/blocked_loss.py` | `BlockedPostfixCausalLMLoss` — norm+lm_head in blocks with per-chunk backward + CPU grad save |

### Telemetry and safety

| File | Purpose |
|---|---|
| `telemetry.py` | `mark_model_gpu_phase()`, `enable_operator_telemetry()`, `assert_finite_tensor()`, `parse_int_set/parse_name_list` |
| `watchdog.py` | `start_memory_watchdog()`, `mark_phase()`, `memory_snapshot()`, `mark_memory_phase()` |
| `memory.py` | `pin_module_alloc()` / `pin_module_register()` — CPU memory pinning strategies |

### Optimizer and checkpoint

| File | Purpose |
|---|---|
| `optim.py` | `PerDeviceOptimizer` — one AdamW per device, synchronised LR scheduling |
| `checkpoint.py` | `save_checkpoint()` / `load_checkpoint()` — PEFT safetensors + per-device optim state |
| `utils.py` | device detection, `gpu_memory_snapshot()`, `get_optimal_tensor_split()` |

## Training Mechanics — How a Single Step Works

### Step lifecycle

```
1. optimizer.zero_grad()
2. For each microbatch:
   a. ensure_weights(self.prefix, 0)           # upload embed_tokens NF4→FP16 to GPU 0
   b. self.prefix(input_ids, ...)               # embed_tokens, pos_emb → 7-tuple
   c. For each stage:
        ensure_weights(stage, stage.device_id)  # upload stage's NF4 weights to its GPU
        stage(tuple_data)                       # run all layers in this stage
   d. ensure_weights(self.postfix, last_device) # upload lm_head weights
   e. self.postfix(tuple_data)                  # norm + chunked lm_head → loss
   f. (loss / num_microbatch).backward()        # gradients flow back through all stages
3. optimizer.step()
4. pipeline.free_all_weights()                  # free FP16 data, keep NF4 payload for next step
```

### The 7-tuple

Everything flows through the pipeline as a 7-element tuple:
```
(0) hidden_states
(1) causal_mask_mapping     — dict with "full_attention" and "linear_attention"
(2) position_ids
(3) position_embeddings     — (cos, sin) from RoPE
(4) kwargs                  — extra keyword args (pass-through)
(5) labels                  — for loss computation
(6) logits_to_keep          — for speculative decoding (always 0)
```

### NF4 streaming lifecycle

```
prepare_nf4() — at pipeline build time
  │
  ├── quantize_4bit(weight) → quantized (NF4 uint8) + absmax + code
  ├── pin_memory() on all three tensors
  ├── store as NF4Payload on param.roundpipe_nf4_payload
  └── param.data = empty(0)  ← frees original FP16 weight
  │
  ▼
ensure_weights() — at each step, before stage forward
  │
  ├── quantized_cpu → .to(device)  ← small H2D (4-bit)
  ├── dequantize_4bit() on GPU     ← FP16 on GPU
  └── param.data = dequantized     ← ready for forward
  │
  ▼
  [ forward runs with FP16 weights ]
  │
  ▼
free_weights() — after backward, before next step
  │
  ├── param.data = empty(0)        ← frees FP16 on GPU
  └── NF4Payload kept on CPU       ← ready for next ensure_weights
```

**Crucial detail:** NF4 payloads are on CPU, NOT GPU. This is different from
QLoRA's `Linear4bit` which keeps NF4 weights on GPU. The tradeoff: Stratum
uses PCIe bandwidth every step (~3 MB per weight) but saves GPU VRAM
(no NF4→FP16 overhead residing permanently on GPU).

### Chunked loss — the VRAM-critical path

Both postfixes support two modes:

**Mode 1 (default, `postfix_loss_token_chunk_size=0`):**
- Norm runs full sequence
- lm_head runs in `loss_token_chunk_size`-token chunks
- Each chunk: `lm_head(chunk) → CE(reduction="sum")`
- `loss = sum(all_chunks) / num_items_in_batch`

**Mode 2 (`postfix_loss_token_chunk_size > 0`):**
- `BlockedPostfixCausalLMLoss.apply()` — custom autograd Function
- Both norm AND lm_head run in token blocks
- Each block: `norm → lm_head → CE` with per-block `.backward()` to accumulate lm_head grads
- Hidden state gradient saved to CPU, restored in outer backward
- Requires batch_size=1
- Saves norm activation memory at cost of CPU→GPU grad restore

**Why reduction="sum" and not "mean":**
When a chunk contains only ignored tokens (-100, i.e., padding), `CE(reduction="mean")`
would produce `0/0 = NaN` because num_items_in_chunk = 0. Using `reduction="sum"`
produces 0 for that chunk. The total is divided by `num_items_in_batch` (total
non-ignored tokens across all chunks), so padding chunks contribute 0.

## Device Assignment

`assign_layers_to_devices(n_layers, tensor_split=[9, 32])`:

```
cumulative ratio = [9/41, 41/41] = [0.22, 1.0]
layer i → device = bisect_left([0.22, 1.0], i/41)
```

For 41 layers, `tensor_split=[9, 32]`:
- Layers 0-8 → device 0 (RTX 3080, 9 layers)
- Layers 9-40 → device 1 (V100, 32 layers)

The first device always hosts the prefix (embed_tokens) plus its assigned layers.

## Boundary Transfer

When stages are on different devices, `HostStagingPool` transfers activations
between them:

1. **P2P path** (both GPUs support peer access): `cudaMemcpyPeerAsync` — direct
   GPU→GPU copy through NVLink/NVSwitch. Fastest.
2. **Host-staged path** (no P2P): D2H on source → pinned buffer → H2D on
   destination. Each pool instance handles one boundary.

During backward, `make_boundary_hook()` uses `.to(device)` to move gradients
back to the previous device.

## MLP Optimizations

Three mutually-exclusive MLP strategies, selected by CLI flags:

| Flag | Class | Behavior |
|---|---|---|
| `--checkpoint-mlp` | `CheckpointedModule` | Wraps MLP in `torch.checkpoint` — recomputes in backward |
| `--mlp-token-chunk-size N` | `TokenChunkedModule` | Splits MLP forward into N-token chunks (positionwise safe) |
| `--memory-flat-frozen-mlp` | `MemoryFlatFrozenMLP` | Custom autograd with chunked backward recompute via `torch.autograd.grad()` |

Mutex rules (enforced by `apply_mlp_optimizations()`):
- `memory_flat_frozen_mlp` requires `mlp_token_chunk_size > 0`
- `memory_flat_frozen_mlp` conflicts with `checkpoint_mlp`
- `mlp_token_chunk_size` without `memory_flat_frozen_mlp` → `TokenChunkedModule`
- Skips MoE experts (layers with `gate_exps`)

## Checkpoint Format

Checkpoints are saved with **both** formats:

```
checkpoint-5000/
├── adapter_model.safetensors     ← PEFT-compatible LoRA weights (portable)
├── adapter_config.json            ← PEFT adapter config
├── meta.pt                        ← step number
├── device_0.pt                    ← legacy per-device state (backward compat)
├── device_1.pt
├── optim_0.pt                     ← per-device optimizer state
└── optim_1.pt
```

**Saving:** `hf_model.save_pretrained(out_dir)` produces `adapter_model.safetensors`
+ `adapter_config.json`. This works because PEFT only saves LoRA parameters,
not base model frozen weights. Even though `prepare_nf4()` has set frozen
weights to `empty(0)`, PEFT ignores them.

**Loading (resume):** `safetensors.torch.load_file()` → `hf_model.load_state_dict(strict=False)`.
Prefers PEFT adapter, falls back to legacy `.pt` files.

**Inference on another machine:**
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./checkpoint-5000")
```

## CLI Flags Reference

### Model and data
| Flag | Default | Description |
|---|---|---|
| `--model` | `lfm25-8b-a1b` | Architecture name (registered in registry.py) |
| `--hf-model` | `LiquidAI/LFM2.5-8B-A1B` | HuggingFace model name |
| `--data` | required | Path to pre-tokenized JSONL dataset |
| `--out` | `/runs/stratum-training` | Output directory for checkpoints and logs |
| `--resume` | `""` | Checkpoint path to resume from |
| `--no-save` | False | Skip final checkpoint save |

### Device configuration
| Flag | Default | Description |
|---|---|---|
| `--tensor-split` | None | VRAM ratios per device, e.g. `10 32` |
| `--device-ids` | None | CUDA device IDs to use |
| `--max-seq-len` | 49152 | Truncate sequences longer than this |

### Training hyperparameters
| Flag | Default | Description |
|---|---|---|
| `--steps` | 25000 | Training steps |
| `--batch-size` | 2 | Microbatch not split (actual per-step = batch * num_microbatch) |
| `--lr` | 1e-4 | Learning rate |
| `--lr-scheduler` | `cosine_with_warmup` | LR schedule |
| `--warmup-steps` | 500 | Linear warmup steps |
| `--weight-decay` | 0.1 | AdamW weight decay |
| `--lora-r` | 16 | LoRA rank |
| `--lora-target-set` | `all` | LoRA module targeting: all, attention, attention_input, mlp |
| `--save-every` | 500 | Checkpoint interval |

### VRAM optimization
| Flag | Default | Description |
|---|---|---|
| `--num-microbatch` | 1 | Split batch into N microbatches (gradient accumulation) |
| `--checkpoint-decoder-layer` | True | Activation checkpointing per decoder layer |
| `--no-nf4` | False | Disable NF4 frozen weight compression (FP16 direct upload) |
| `--nf4-cache-dir` | None | Directory to cache NF4 payloads |
| `--pin-model` | `alloc` | CPU pinning: alloc (pin_memory), register (cudaHostRegister), off |
| `--checkpoint-mlp` | False | Wrap MLP in activation checkpointing |
| `--mlp-token-chunk-size` | 0 | Split MLP forward into token chunks (0 = disabled) |
| `--memory-flat-frozen-mlp` | False | Frozen MLP with token-chunked backward recompute |

### Loss chunking
| Flag | Default | Description |
|---|---|---|
| `--loss-token-chunk-size` | 4096 | Token chunk size for lm_head (always active) |
| `--postfix-loss-token-chunk-size` | 0 | Enable blocked postfix loss (mode 2, saves norm memory) |
| `--torch-compile-loss` | False | Use `@torch.compile` on cross_entropy |

### Attention patching
| Flag | Default | Description |
|---|---|---|
| `--volta-layers` | `""` | Comma-separated layer indices for V100 flash attn (empty = all) |
| `--volta-window-left` | -1 | Sliding window left tokens (-1 = full attention) |
| `--volta-window-right` | 0 | Sliding window right tokens (0 = causal) |

### Data loading
| Flag | Default | Description |
|---|---|---|
| `--longest-first` | False | Sort dataset by seq_len descending |
| `--pad-to-multiple` | 0 | Pad batch length to multiple of N (volta_flash needs 32) |
| `--pad-to-length` | 0 | Pad batch to exact length N |
| `--dense-attention-masks` | False | Force HF dense attention mask construction |

### Telemetry and debugging
| Flag | Default | Description |
|---|---|---|
| `--memory-telemetry` | False | Log GPU allocator at prefix/layer/postfix boundaries |
| `--operator-telemetry-layers` | `""` | Comma-separated layer indices for per-operator hooks |
| `--operator-telemetry-modules` | `input_layernorm,self_attn,post_attention_layernorm,mlp` | Module names for operator telemetry |
| `--debug-finite` | False | Assert tensor values are finite |
| `--cuda-memory-summary-on-exception` | False | Print CUDA memory summary on RuntimeError |
| `--host-ram-limit-gib` | 0.0 | Abort when host RSS exceeds this (0 = disabled) |

## Known Quirks and Sharp Edges

### 1. The backward graph — why `.detach()` order matters

In the non-microbatch path, `loss.backward()` MUST be called on the live loss
tensor, not a detached copy. The microbatch path does this correctly:
```python
(mb_out.loss / nmb).backward()  # live graph → correct gradients
if loss is None:
    loss = mb_out.loss.detach()  # detach AFTER backward for logging
```

The non-microbatch path was broken at one point (detach before backward),
already fixed.

### 2. `free_weights()` doesn't free trainable params

`free_weights()` only sets `param.data = empty(0)` for params that have
`roundpipe_nf4_payload` attribute. Trainable LoRA params don't have this
attribute and are NOT freed. This is intentional — LoRA params must persist
for the optimizer.

### 3. PEFT adapter save works despite empty frozen weights

After `prepare_nf4()`, frozen weights in the pipeline modules are `empty(0)`.
But `hf_model.save_pretrained()` calls PEFT's `get_peft_model_state_dict()`
which only returns LoRA parameters (those with `lora_` in the name). The
empty frozen weights are invisible to the save.

For this to work, the pipeline's stage layers MUST share Parameter objects
with `hf_model.get_base_model().model.layers`. This is ensured by
`ModelArch.build()` which uses the original layer objects (not deep copies).

The prefix and postfix use `copy.deepcopy()`, so their NF4-dropped weights
do NOT affect hf_model. Since prefix/postfix contain only frozen weights
(no LoRA), this doesn't matter.

### 4. `BlockedPostfixCausalLMLoss` requires batch_size=1

This is a RoundPipe limitation preserved in the port. The error is raised
at runtime if batch > 1 with `--postfix-loss-token-chunk-size > 0`.

### 5. Qwen3.5 wrapped layers don't support `checkpoint_decoder_layer`

Only LFM25's wrapped layer (`LFM25ForCausalLMWrappedLayer`) uses the
`checkpoint_decoder_layer` flag. Qwen35's `Qwen35ForCausalLMWrappedLayer`
runs the full layer without checkpointing. To add it, apply the same
`torch.checkpoint` pattern as LFM25.

### 6. `upload_stream()` is dead code

The function `upload_stream()` in `upload.py` is exported from `__init__.py`
and used to be the Phase 2 upload function. It was replaced by the
`ensure_weights()`/`free_weights()` pair and the per-device upload loop
in `registry.py:build()`. It remains for backward compatibility but is
never called from the training path.

### 7. NF4 payload format is `bytes` not `int` in `NF4Payload.source_bytes`

The `source_bytes` field is `int`. This is consistent. No issue, just
noting for anyone reading the dataclass.

### 8. Docker build requires careful layer ordering

The Dockerfile (`Dockerfile`) must COPY source files AFTER all CUDA
compilations. The critical order:
```
1. Install system deps
2. Install flash-attn-v100 (sm_70)
3. Install standard flash-attn (sm_86)
4. Install causal_conv1d (sm_70 + sm_86)
5. COPY stratum /workspace/stratum   ← LAST, so source changes don't invalidate CUDA cache
```

If you add new CUDA/C++ extensions, put them before the COPY stratum line.

### 9. Memory pinning with `register` strategy

`pin_module_register()` uses `cudaHostRegister(ptr, size, 0)` which has
specific alignment requirements. The `PAGE_SIZE = 4096` check in the
coalescing loop ensures adjacent allocations within 4KB are merged.
If `cudaHostRegister` fails, check that the storage pointers are properly
aligned to page boundaries.

## How to Add a New Model Architecture

1. Create `stratum/model/{name}.py` with:
   - Prefix class (embeddings + position encoding)
   - WrappedLayer class (one decoder layer with optional checkpointing)
   - Postfix class (norm + lm_head)
   - VoltaAttention class (flash attention variant)
   - Arch class with `@register("{name}")` decorator

2. Register in `stratum/__init__.py`:
   ```python
   import stratum.model.{name}  # noqa: F401
   ```

3. The Arch class must implement:
   - `get_config(model)` → config object with `vocab_size`
   - `get_num_layers(config)` → number of decoder layers
   - `build_prefix(model)` → prefix module
   - `build_wrapped_layer(layer, idx, **kwargs)` → wrapped layer
   - `build_postfix(model, **kwargs)` → postfix module
   - Optionally override `build()` to patch attention modules

## Testing

Build and run:
```bash
docker build -t stratum:latest .
docker run --gpus all --rm stratum:latest python scripts/train.py \
    --model lfm25-8b-a1b \
    --data /data/lfm25_fable_merged_48k_train.labels.jsonl \
    --tensor-split 9 32 \
    --steps 5 \
    --batch-size 2 \
    --num-microbatch 2 \
    --save-every 0 \
    --no-save
```

## Key Files for Agents

When reading the codebase, these are the critical files to understand:

1. **`scripts/train.py`** — the full training loop, CLI args, loading, saving
2. **`stratum/pipeline.py`** — `StratumPipeline.forward()` — the core orchestration
3. **`stratum/upload.py`** — NF4 streaming (prepare/ensure/free)
4. **`stratum/model/registry.py`** — build pipeline assembly
5. **`stratum/model/lfm25.py`** — the most complete example of a model architecture
6. **`stratum/model/mlp_opt.py`** — MLP memory optimizations
7. **`stratum/model/blocked_loss.py`** — BlockedPostfixCausalLMLoss
8. **`stratum/checkpoint.py`** — save/load with PEFT safetensors
