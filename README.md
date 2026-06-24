# Stratum

Multi-GPU layer-parallel training for transformer models.

Split any transformer model across N GPUs — any architectures, any VRAM sizes.
Each device permanently owns a stratum of decoder layers. Device boundaries
use host-staged PCIe transfers when peer access is unavailable, making
heterogeneous GPU setups a first-class target.

## Quick Start

```bash
# Full build. Use this only when CUDA/dependency layers change.
docker build -t stratum:latest .

# Fast source/runtime-script refresh. Use this for normal Stratum code edits.
docker build -f Dockerfile.refresh -t stratum:latest .

# Verify the container sees CUDA, compiled kernels, caches, and NF4.
scripts/run-unified.sh python scripts/doctor.py

# Run on the container-visible GPUs. Pick physical GPUs with STRATUM_GPU
# and map them to logical cuda:0..N with STRATUM_CUDA_VISIBLE_DEVICES.
STRATUM_DATA_DIR=/path/to/training_data \
scripts/run-unified.sh python scripts/train.py \
    --model lfm25-8b-a1b \
    --data /workspace/data/training.labels.jsonl \
    --tensor-split 10 32 \
    --steps 25000 \
    --batch-size 2 \
    --out /workspace/out/my-run
```

`scripts/run-unified.sh` is the canonical Docker entry point. It mirrors the
qz-roundpipe launcher pattern: persistent cache mounts, `--ipc=host`,
`IPC_LOCK`, Docker memory caps, and explicit GPU visibility controls.

## Reference Runtime

Stratum is designed for heterogeneous, no-P2P multi-GPU machines, not just
clean datacenter boxes. The current reference host is a scraped-together
home/education-style setup: container-local `cuda:0` is an RTX 3080-class card
and `cuda:1` is a V100 32 GiB card, with `peer_access` false in both
directions. On that class of rig, cross-device boundaries must work through the
host-staged pinned-buffer path.

This is intentional. Treat host-staged boundary transfer as the reference path,
not merely as a slow fallback. Multi-GPU changes should be validated with:

```bash
scripts/run-unified.sh python scripts/doctor.py

STRATUM_DATA_DIR=/home/harri/qz-roundpipe/data \
scripts/run-unified.sh python scripts/train.py \
    --model lfm25-8b-a1b \
    --data /workspace/data/lfm25_fable_merged_48k_train.labels.jsonl \
    --tensor-split 10 32 \
    --steps 1 \
    --batch-size 1 \
    --no-save \
    --out /workspace/out/smoke
```

Do not use host Python, host CUDA, or host `nvidia-smi` as evidence that
Stratum training works. The Docker runtime is the product surface.

## Supported Models

- `lfm25-8b-a1b` — LFM2.5-8B-A1B (Liquid AI)
- `qwen3.5` — Qwen3.5 series

New models register via `@register("name")` in `stratum/model/`.

## How It Works

Stratum uses llama.cpp's `LLAMA_SPLIT_MODE_LAYER` algorithm to assign layers
to devices proportional to their VRAM. At startup, frozen weights are uploaded
via NF4 4-bit quantization (4x compression). During training, PCIe is used only
for the hidden-state tensor at device boundaries. When peer access is
unavailable, the fallback stages through a reusable pinned host buffer, adapted
from the local TurboQuant llama.cpp work in
`/home/harri/turboquant-work/llama-cpp-turboquant/ggml/src/ggml-cuda/ggml-cuda.cu`.

## Components

| Module | Purpose |
|--------|---------|
| `stratum/assign.py` | Layer-to-device via `upper_bound` |
| `stratum/stage.py` | `DeviceStage` — contiguous layers on one GPU |
| `stratum/pipeline.py` | `StratumPipeline` — fwd/bwd across devices |
| `stratum/host_staging.py` | Pinned buffer pool for PCIe transfers |
| `stratum/grad_hooks.py` | Cross-device gradient hooks |
| `stratum/upload.py` | Unified NF4 + FP16 weight upload per device |
| `stratum/optim.py` | Per-device optimiser + LR scheduler |
| `stratum/checkpoint.py` | PEFT safetensors + JSON trainer state; legacy `.pt` opt-in |
| `stratum/model/registry.py` | Model architecture registry |

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `lfm25-8b-a1b` | Registered model name |
| `--hf-model` | `LiquidAI/LFM2.5-8B-A1B` | HuggingFace model ID |
| `--tensor-split` | auto | VRAM weights per device |
| `--steps` | 25000 | Training steps |
| `--batch-size` | 2 | Per-step batch size |
| `--lr` | 1e-4 | Learning rate |
| `--lr-scheduler` | cosine_with_warmup | LR schedule |
| `--save-every` | 500 | Checkpoint interval |
| `--no-nf4` | false | Disable NF4 compression |
| `--resume` | "" | Checkpoint to resume from |

Default checkpoints are disk-efficient LoRA/QLoRA artifacts:
`adapter_model.safetensors`, `adapter_config.json`, and `trainer_state.json`.
Stratum does not write giant per-device `.pt` files unless explicitly asked
with `--save-legacy-device-state` or `--save-optimizer-state`.

## Cache

Inside the launcher, the cache root is `/workspace/cache`. It persists under
`./cache` by default and can be moved with `STRATUM_CACHE_DIR`.
NF4 frozen-weight payloads default to `/workspace/cache/nf4-frozen/<hf-model>`,
so repeated runs do not re-quantize the same base weights.

```bash
STRATUM_CACHE_DIR=/fast/cache scripts/run-unified.sh python scripts/doctor.py
```

Useful launcher environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `STRATUM_IMAGE` | `stratum:latest` | Docker image |
| `STRATUM_GPU` | `all` | Docker `--gpus` selector, e.g. `device=0,1` |
| `STRATUM_CUDA_VISIBLE_DEVICES` | unset | Container logical device map |
| `STRATUM_DOCKER_MEMORY` | `88g` | Container RAM limit |
| `STRATUM_DOCKER_MEMORY_SWAP` | `88g` | Container swap limit |
| `STRATUM_CACHE_DIR` | `./cache` | Persistent HF/Torch/Triton/CUDA/NF4 cache |
| `STRATUM_OUT_DIR` | `./out` | Training outputs mounted at `/workspace/out` |
| `STRATUM_DATA_DIR` | `./data` | Optional read-only data mount at `/workspace/data` |

## Docker Build Discipline

`Dockerfile` contains the expensive CUDA, PyTorch, flash-attention, and
kernel-build layers. Avoid source-only edits near the top of that file: they
invalidate the instruction chain and can force Docker to repeat expensive
dependency work. For normal code and runtime script changes, use
`Dockerfile.refresh`; it starts from the current `stratum:latest`, deletes any
old `/workspace/stratum` tree, installs Stratum from a minimal temporary source
copy, and leaves only runtime scripts under `/workspace/stratum/scripts`.

Documentation-only changes do not require an image rebuild. Documentation,
tests, git metadata, caches, datasets, outputs, and local checkpoint/model blobs
are not runtime image contents. The Docker build generates a tiny temporary
README solely to satisfy Python package metadata without baking real docs into
the image.

The repository has a `.dockerignore` so build context should stay small. If a
build suddenly sends gigabytes of context, stop and check ignored paths before
waiting on Docker.

Some hosts print Docker warnings like `Your kernel does not support memory
limit capabilities...`. On those machines Docker memory caps may be ignored;
use Stratum's `--host-ram-limit-gib` watchdog as the practical safety guard.

## Acknowledgments

Stratum builds directly on several open-source projects and research papers.
Without the prior work these teams invested, Stratum would not exist.

### Software

| Project | Contribution | Authors / Maintainers |
|---------|-------------|-----------------------|
| [RoundPipe](https://github.com/thustorage/RoundPipe) | Staged pipeline training framework that Stratum evolved from. The Prefix/WrappedLayer/Postfix module pattern, NF4 integration, and staged uploading are all adapted from RoundPipe's design. | ITcarrot, Tsinghua University |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / TurboQuant local fork | Multi-GPU layer split algorithm (`LLAMA_SPLIT_MODE_LAYER`, `llama-model.cpp:1265-1277`) and Harri's PCIe host-staged cross-device copy fallback from `/home/harri/turboquant-work/llama-cpp-turboquant/ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_copy_across_devices`, `ggml_cuda_copy2d_across_devices`). | Georgi Gerganov and contributors; local TurboQuant work by Harri |
| [flash-attention-v100](https://github.com/ai-bond/flash-attention-v100) | V100 (SM70) flash attention kernel used by `Lfm25VoltaAttention` and `Qwen35VoltaAttention`. | ai-bond |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | NF4 4-bit quantization and dequantization kernels used for compressed weight upload. | Tim Dettmers, Facebook Research |
| [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | Efficient causal convolution kernels used by LFM2.5 ShortConv layers. | Tri Dao, Dao-AILab |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Model implementations for LFM2.5 and Qwen3.5, tokenizers, configuration. | HuggingFace |
| [PEFT](https://github.com/huggingface/peft) | LoRA adapter layer injection and management. | HuggingFace |
| [PyTorch](https://pytorch.org) | The foundation. Autograd, distributed primitives, CUDA stream management. | Meta AI and contributors |

### Research

- **QLoRA: Efficient Finetuning of Quantized Language Models** — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023). The NF4 quantization scheme used for compressed weight upload.
  https://arxiv.org/abs/2305.14314

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022). The algorithmic foundation of flash-attention-v100.
  https://arxiv.org/abs/2205.14135

- **LLaMA: Open and Efficient Foundation Language Models** — Hugo Touvron et al. (2023). The transformer architecture that LFM2.5 and Qwen3.5 both descend from.
  https://arxiv.org/abs/2302.13971

- **Mixture of Experts Explained** — The MoE routing and grouped GEMM patterns used by LFM2.5-8B-A1B follow the standard sparsely-gated MoE architecture.
  https://arxiv.org/abs/1701.06538

## License

Apache 2.0
