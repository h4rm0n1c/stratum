# Stratum

Multi-GPU layer-parallel training for transformer models.

Split any transformer model across N GPUs — any architectures, any VRAM sizes.
Each device permanently owns a stratum of decoder layers. Device boundaries
use host-staged PCIe transfers when peer access is unavailable, making
heterogeneous GPU setups a first-class target.

## Quick Start

```bash
# Build
docker build -t stratum:latest .

# Run on 2 GPUs (10 GB + 32 GB)
docker run --gpus all --ipc=host \
    --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
    -v stratum_cache:/var/cache/stratum \
    -v /path/to/training_data:/data \
    -v /path/to/runs:/runs \
    stratum:latest \
    python scripts/train.py \
        --model lfm25-8b-a1b \
        --data /data/training.labels.jsonl \
        --tensor-split 10 32 \
        --steps 25000 \
        --batch-size 2 \
        --out /runs/my-run
```

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
| `stratum/checkpoint.py` | Save/load across devices |
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

## Cache

Single `$STRATUM_CACHE=/var/cache/stratum` root. Mount a volume or host dir:

```bash
-v stratum_cache:/var/cache/stratum
```

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
