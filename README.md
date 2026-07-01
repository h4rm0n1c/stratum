# Stratum

Multi-GPU layer-parallel training for transformer models on heterogeneous hardware.

Split any transformer model across multiple GPUs — different architectures, different VRAM sizes, no peer access required. Each device permanently owns a stratum of decoder layers. Device boundaries use host-staged PCIe transfers, making mixed-GPU rigs a first-class target rather than an afterthought.

## Why Stratum

Consumer and scraped-together GPU rigs don't have NVLink or peer access. Stratum treats that as the normal case: when `cudaMemcpyPeerAsync` won't work, activations and gradients stage through a reusable pinned host buffer. The model's weights are streamed in NF4 4-bit each step, so only the compressed payload crosses PCIe — not the full FP16 tensor.

The result: you can train a single model split across an RTX 3080 and a V100 (or any other combination) at long context lengths, with LoRA adapters, CPU-offloaded optimizer state, and gradient scaling.

## Quick Start

```bash
# Full build — use only when CUDA/dependency layers change.
docker build -t stratum:latest .

# Fast source refresh — use for normal code and script edits.
docker build -f Dockerfile.refresh -t stratum:latest .

# Verify the container sees CUDA, compiled kernels, and NF4.
scripts/run-unified.sh python scripts/doctor.py

# Train on the container-visible GPUs.
STRATUM_DATA_DIR=/path/to/training_data \
scripts/run-unified.sh python scripts/train.py \
    --model lfm25-8b-a1b \
    --data /workspace/data/training.labels.jsonl \
    --tensor-split 10 32 \
    --steps 25000 \
    --batch-size 2 \
    --out /workspace/out/my-run
```

`scripts/run-unified.sh` is the canonical Docker entry point. It mirrors the RoundPipe launcher pattern: persistent cache mounts, `--ipc=host`, `IPC_LOCK`, Docker memory caps, and explicit GPU visibility controls.

## Supported Models

| Adapter | Model | Notes |
|---|---|---|
| `lfm25-8b-a1b` | LiquidAI/LFM2.5-8B-A1B | MoE, primary validation target |
| `qwen3.5` | Qwen3.5 series | Dense + linear attention, sliding window |
| `llama` | Llama-family models | TinyLlama validated |
| `qwen3` | Qwen3 dense models | Qwen3-0.6B validated |
| `qwen3-moe` | Qwen3-MoE models | Qwen3-30B-A3B validated, router aux loss |

New models register via `@register("name")` in `stratum/model/registry.py`.

## Features

| Capability | Details |
|---|---|
| **NF4 frozen-weight streaming** | 4x compressed upload per step; payloads stay on CPU; cache avoids re-quantization |
| **Host-staged no-P2P transfers** | Pinned buffer pool for cross-device activation/gradient transfer when peer access is unavailable |
| **Capability-dispatched flash attention** | `flash_attn` on SM80+/SM86, `flash_attn_v100` on SM70 — hard failure, no silent eager fallback |
| **MuonClip optimizer** | Hybrid Muon + AdamW with QK-Clip for attention Q/K projections; exact flash-logit stats when patched kernels support them |
| **CPU-offloaded async optimizer** | AdamW or hybrid Muon state in fp32 on CPU; background optimizer stream; frees ~2x trainable-param GPU memory |
| **GradScaler** | Async-safe dual-scaler design for mixed-precision training |
| **Recompute context** | PyTorch checkpoint bridge; non-grad tensors saved/restored to avoid redundant allocation |
| **Scheduler groups** | Explicit layer-group forward/backward with wait/notify semantics |
| **Per-layer timing + plan adaptation** | CUDA-event timing per layer; automatic plan rebuild from EMA estimates |
| **Async H2D upload stream** | Per-device `param_upstream` stream; per-layer upload/compute overlap |
| **Pytree batch API** | `guess_split_spec` / `split_pytree` / `merge_pytree` for arbitrary input shapes |
| **PEFT safetensors checkpoints** | Portable LoRA adapters; optimizer `.pt` is opt-in |
| **MoE router aux loss** | Router-logit side-channel capture; load-balancing loss in postfix |
| **MLP optimizations** | CheckpointedModule, TokenChunkedModule, MemoryFlatFrozenMLP |
| **Chunked cross-entropy loss** | Per-token-chunk lm_head with per-chunk backward; blocked norm+lm_head mode |
| **Memory watchdog** | Host RSS monitoring with optional abort; GPU allocator telemetry |

## How It Works

Stratum uses llama.cpp's `LLAMA_SPLIT_MODE_LAYER` algorithm to assign layers to devices proportional to their VRAM. At build time, frozen weights are quantized to NF4 4-bit and cached. During training:

1. **Upload** — NF4 payloads stream from CPU to GPU and dequantize to FP16 before each stage's forward pass.
2. **Forward** — prefix → stage 0 → (boundary transfer) → stage 1 → postfix runs sequentially.
3. **Backward** — gradients flow back through stages; boundary gradients transfer through the same host-staged pool.
4. **Free** — FP16 weight data is dropped after backward; NF4 payloads remain on CPU for the next step.

When CUDA peer access is available, boundary transfers go direct GPU→GPU. When it isn't (the reference target), each boundary stages through a reusable pinned host buffer — an adaptation of the host-staged fallback pattern from llama.cpp's CUDA backend, extended with reusable pool semantics and graph-preserving async helpers.

## Components

| Module | Purpose |
|---|---|
| `stratum/assign.py` | Layer-to-device assignment via `upper_bound` |
| `stratum/stage.py` | `DeviceStage` — contiguous decoder layers on one GPU |
| `stratum/pipeline.py` | `StratumPipeline` — forward/backward orchestration across devices |
| `stratum/host_staging.py` | `HostStagingPool` — pinned buffer pool for PCIe transfers |
| `stratum/grad_hooks.py` | `make_boundary_hook()` — cross-device gradient hooks |
| `stratum/layer_transfer.py` | Chunked upload/download helpers for standalone layer copies |
| `stratum/transfer.py` | `async_d2h` / `async_h2d` / `PinnedUpload` / `RegisterBackwardEvent` |
| `stratum/upload.py` | NF4 + FP16-staged weight upload; `ensure_weights()` / `free_weights()` / `NF4Prefetch` |
| `stratum/batch.py` | Pytree split/merge (`guess_split_spec`, `split_pytree`, `merge_pytree`); `TokenWeightedReducer` |
| `stratum/scheduler.py` | `ModelExecutePlan`, `ModelTracker`, `BackwardScheduleSimulator` |
| `stratum/runtime.py` | Explicit group backward: `anchor_explicit_group_backward()`, `run_explicit_group_backward()` |
| `stratum/timing.py` | Per-layer CUDA-event timing: `LayerTimingContext`, `IterLayerTimer`, `ModelLayerTimer` |
| `stratum/optim.py` | `PerDeviceOptimizer` — per-device optimizer wrapper with LR scheduling |
| `stratum/muon.py` | Hybrid Muon + AdamW optimizer for trainable adapter params |
| `stratum/qk_clip.py` | QK-Clip stat capture and post-step Q/K scaling |
| `stratum/optim_stream.py` | Background optimizer thread with kernel queue |
| `stratum/attribute.py` | `ParamAttribute` — per-parameter grad/optim state tracking |
| `stratum/grad_scaler.py` | Async-safe GradScaler with dual-scaler design |
| `stratum/context.py` | `ForwardCtx` / `RecomputeCtx` / `checkpoint_context_fn()` |
| `stratum/moe.py` | MoE router-logit capture; `load_balancing_loss_func()` |
| `stratum/planner.py` | Memory-budgeted stage splitting |
| `stratum/memory.py` | `pin_module_alloc()` / `pin_module_register()` |
| `stratum/telemetry.py` | Operator telemetry hooks; NaN/Inf detection |
| `stratum/watchdog.py` | Host RAM watchdog; memory phase markers |
| `stratum/utils.py` | Device detection, `gpu_memory_snapshot()`, `get_optimal_tensor_split()` |
| `stratum/checkpoint.py` | PEFT safetensors save/load; JSON trainer state |
| `stratum/model/registry.py` | Model architecture registry; `build_pipeline()` entry |
| `stratum/model/lfm25.py` | LFM2.5-8B-A1B adapter |
| `stratum/model/qwen35.py` | Qwen3.5 adapter |
| `stratum/model/llama.py` | Llama-family adapter |
| `stratum/model/qwen3.py` | Qwen3 dense adapter |
| `stratum/model/qwen3_moe.py` | Qwen3-MoE adapter |
| `stratum/model/mlp_opt.py` | MLP optimizations: CheckpointedModule, TokenChunkedModule, MemoryFlatFrozenMLP |
| `stratum/model/blocked_loss.py` | `BlockedPostfixCausalLMLoss` — blocked norm+lm_head with per-block backward |

## Configuration

### Model and data

| Flag | Default | Description |
|---|---|---|
| `--model` | `lfm25-8b-a1b` | Registered model name |
| `--hf-model` | `LiquidAI/LFM2.5-8B-A1B` | HuggingFace model ID |
| `--data` | required | Path to pre-tokenized JSONL dataset |
| `--out` | `/runs/stratum-training` | Output directory for checkpoints |
| `--resume` | `""` | Checkpoint path to resume from |
| `--no-save` | false | Skip final checkpoint save |

### Device configuration

| Flag | Default | Description |
|---|---|---|
| `--tensor-split` | auto | VRAM ratios per device, e.g. `10 32` |
| `--device-ids` | auto | CUDA device IDs to use |
| `--max-seq-len` | 49152 | Truncate sequences longer than this |

### Training

| Flag | Default | Description |
|---|---|---|
| `--steps` | 25000 | Training steps |
| `--batch-size` | 2 | Per-step batch size |
| `--num-microbatch` | 1 | Split batch into N microbatches |
| `--lr` | 1e-4 | Learning rate |
| `--lr-scheduler` | cosine_with_warmup | LR schedule |
| `--warmup-steps` | 500 | Linear warmup steps |
| `--weight-decay` | 0.1 | Decoupled optimizer weight decay |
| `--optimizer` | `adamw` | Optimizer: `adamw`, configurable hybrid `muon`, or stable `muonclip` |
| `--muon-momentum` | 0.95 | Muon momentum for matrix trainable tensors |
| `--muon-ns-steps` | 5 | Newton-Schulz iterations for Muon |
| `--muon-update-scale` | 0.2 | Muon update scaling multiplier |
| `--muon-qk-mode` | `clip` | Under `muon`, apply QK-Clip to Q/K params; alternatives: `adamw`, `muon`. `muonclip` always uses `clip` |
| `--muon-qk-clip-threshold` | 100.0 | QK-Clip attention-logit threshold tau |
| `--muon-qk-stat-mode` | `auto` | QK-Clip stat source: patched flash max logits when available, norm bound fallback in `auto`; `exact_flash` requires patched backend |
| `--lora-r` | 16 | LoRA rank |
| `--lora-target-set` | `all` | LoRA targeting: `all`, `attention`, `attention_input`, `mlp` |
| `--save-every` | 500 | Checkpoint interval |

### VRAM optimization

| Flag | Default | Description |
|---|---|---|
| `--no-nf4` | false | Disable NF4 frozen-weight compression |
| `--nf4-scope` | `all` | NF4 prep scope: `all` or `layers` |
| `--nf4-min-numel` | 4096 | Minimum elements to NF4-quantize |
| `--nf4-cache-dir` | `/workspace/cache/nf4-frozen` | NF4 payload cache directory |
| `--prefetch-nf4` | false | Side-stream NF4 payload prefetch |
| `--pin-model` | `alloc` | CPU pinning: `alloc`, `register`, `off` |
| `--stratum-stage-memory-limit-gib` | 0.0 | Split per-device groups into sub-stages |
| `--recompute-grain` | `layer` | Recompute granularity: `stage`, `layer`, or `none` |
| `--offload-stage-inputs` | auto | Host-offload captured stage inputs; defaults on for `--recompute-grain stage` |
| `--checkpoint-decoder-layer` | true | Activation checkpointing per decoder layer |
| `--checkpoint-mlp` | false | Wrap each MLP in activation checkpointing |
| `--mlp-token-chunk-size` | 0 | Split MLPs over token chunks |
| `--memory-flat-frozen-mlp` | false | Token-chunked backward recompute for frozen MLPs |
| `--cpu-offload-optim` | false | Keep optimizer state in fp32 on CPU |
| `--async-optimizer-step` | false | Run optimizer through background stream |
| `--optim-dtype` | `fp32` | CPU optimizer param precision |
| `--grad-scaler-enabled` | false | Enable GradScaler for mixed precision |
| `--loss-token-chunk-size` | 4096 | Token chunk size for chunked lm_head loss |
| `--postfix-loss-token-chunk-size` | 0 | Block norm+lm_head with per-block backward |
| `--torch-compile-loss` | false | Enable torch.compile on CE kernels |

### Attention and data loading

| Flag | Default | Description |
|---|---|---|
| `--attn-implementation` | `flash` | Attention backend (`flash`) |
| `--flash-layers` | `""` | Comma-separated layer indices to patch; empty = all |
| `--flash-window-left` | -1 | Sliding-window left tokens |
| `--flash-window-right` | 0 | Sliding-window right tokens |
| `--longest-first` | false | Sort training data by length descending |
| `--pad-to-multiple` | 32 | Pad sequence length to multiple |
| `--pad-to-length` | 0 | Pad to exact length |
| `--dense-attention-masks` | false | Force HF dense causal mask construction |

### MoE

| Flag | Default | Description |
|---|---|---|
| `--output-router-logits` | false | Capture MoE router logits |
| `--router-aux-loss-coef` | 0.0 | Router auxiliary loss coefficient |

### Telemetry and debugging

| Flag | Default | Description |
|---|---|---|
| `--timing-jsonl` | `""` | Write pipeline timing spans to JSONL |
| `--adapt-plan-every` | 0 | Rebuild scheduler plan from timing every N steps |
| `--host-ram-limit-gib` | 0.0 | Abort when host RSS exceeds this (0 = disabled) |
| `--memory-telemetry` | false | Log GPU allocator at prefix/layer/postfix boundaries |
| `--operator-telemetry-layers` | `""` | Comma-separated layer indices for operator telemetry |
| `--operator-telemetry-modules` | `input_layernorm,self_attn,post_attention_layernorm,mlp` | Submodule names for telemetry |
| `--debug-finite` | false | Assert tensor values are finite after norm/loss |
| `--cuda-memory-summary-on-exception` | false | Print CUDA memory summary on RuntimeError |
| `--pytree-batch` | false | Use pytree-based batch splitting for arbitrary input shapes |

## Cache

Inside the launcher, the cache root is `/workspace/cache`. It persists under `./cache` by default and can be moved with `STRATUM_CACHE_DIR`. NF4 frozen-weight payloads default to `/workspace/cache/nf4-frozen/<hf-model>`, so repeated runs do not re-quantize the same base weights.

```bash
STRATUM_CACHE_DIR=/fast/cache scripts/run-unified.sh python scripts/doctor.py
```

### Launcher environment variables

| Variable | Default | Purpose |
|---|---|---|
| `STRATUM_IMAGE` | `stratum:latest` | Docker image |
| `STRATUM_GPU` | `all` | Docker `--gpus` selector, e.g. `device=0,1` |
| `STRATUM_CUDA_VISIBLE_DEVICES` | unset | Container logical device map |
| `STRATUM_DOCKER_MEMORY` | `88g` | Container RAM limit |
| `STRATUM_DOCKER_MEMORY_SWAP` | `88g` | Container swap limit |
| `STRATUM_CACHE_DIR` | `./cache` | Persistent HF/Torch/Triton/CUDA/NF4 cache |
| `STRATUM_OUT_DIR` | `./out` | Training outputs mounted at `/workspace/out` |
| `STRATUM_DATA_DIR` | `./data` | Optional read-only data mount at `/workspace/data` |

## Docker Build Discipline

`Dockerfile` contains the expensive CUDA, PyTorch, flash-attention, and kernel-build layers. Avoid source-only edits near the top of that file: they invalidate the instruction chain and can force Docker to repeat expensive dependency work. For normal code and runtime script changes, use `Dockerfile.refresh`; it starts from the current `stratum:latest`, deletes any old `/workspace/stratum` tree, installs Stratum from a minimal temporary source copy, and leaves only runtime scripts under `/workspace/stratum/scripts`.

Documentation-only changes do not require an image rebuild. The repository has a `.dockerignore` so build context should stay small. If a build suddenly sends gigabytes of context, stop and check ignored paths before waiting on Docker.

Some hosts print Docker warnings like `Your kernel does not support memory limit capabilities...`. On those machines Docker memory caps may be ignored; use Stratum's `--host-ram-limit-gib` watchdog as the practical safety guard.

## Acknowledgments

Stratum builds on several open-source projects and research. Without the prior work these teams invested, Stratum would not exist.

### Software

| Project | Contribution | Authors |
|---|---|---|
| [RoundPipe](https://github.com/thustorage/RoundPipe) | Staged pipeline training framework. The Prefix/WrappedLayer/Postfix module pattern, NF4 integration, chunked loss, scheduler groups, optimizer stream, and GradScaler are all adapted from RoundPipe's design. | ITcarrot, Tsinghua University |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Multi-GPU layer split algorithm (`LLAMA_SPLIT_MODE_LAYER`). Stratum's host-staged cross-device transfer is an independent innovation built on the same pinned-buffer staging concept, adapted for PyTorch autograd graphs with reusable pool semantics and async graph-preserving helpers. | Georgi Gerganov and contributors |
| [flash-attention-v100](https://github.com/ai-bond/flash-attention-v100) | V100 (SM70) flash attention kernel used by Stratum's capability-dispatched flash attention wrappers. | ai-bond |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | NF4 4-bit quantization and dequantization kernels used for compressed weight upload. | Tim Dettmers, Facebook Research |
| [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | Efficient causal convolution kernels used by LFM2.5 ShortConv layers. | Tri Dao, Dao-AILab |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Model implementations, tokenizers, configuration. | HuggingFace |
| [PEFT](https://github.com/huggingface/peft) | LoRA adapter layer injection and management. | HuggingFace |
| [PyTorch](https://pytorch.org) | Autograd, distributed primitives, CUDA stream management. | Meta AI and contributors |

### Research

- **QLoRA: Efficient Finetuning of Quantized Language Models** — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023). The NF4 quantization scheme used for compressed weight upload. https://arxiv.org/abs/2305.14314
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022). The algorithmic foundation of flash-attention-v100. https://arxiv.org/abs/2205.14135
- **LLaMA: Open and Efficient Foundation Language Models** — Hugo Touvron et al. (2023). The transformer architecture that LFM2.5 and Qwen3.5 both descend from. https://arxiv.org/abs/2302.13971
- **Mixture of Experts Explained** — The MoE routing and grouped GEMM patterns used by LFM2.5-8B-A1B follow the standard sparsely-gated MoE architecture. https://arxiv.org/abs/1701.06538

## License

Apache 2.0 — see [LICENSE](LICENSE) for the full text.
