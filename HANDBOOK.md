# Stratum Handbook

Stratum is a heavy derivative of RoundPipe oriented towards spanned-weights
multi-GPU training. This document covers the architecture, key mechanisms,
design decisions, and known tricky areas for handoff to other agents.

## Port Contract

The target is qz-roundpipe/RoundPipe capability parity plus Stratum's own
spanned-weight execution path. Keep the user-visible training/runtime
capabilities from qz-roundpipe wherever they can work in Stratum, adapting the
implementation when RoundPipe's exact scheduler does not fit. The extra value
Stratum must add is one model spanning heterogeneous GPUs, with boundary
activation transfers using the llama.cpp/TurboQuant host-staged fallback when
CUDA peer access is unavailable.

Do not remove, disable, or route around a qz-roundpipe capability just to make
a local test easier. A feature can be rejected only with evidence that it is
incompatible with Stratum's architecture or not useful after adaptation. The
normal path is: find the source reference, port/adapt the behavior, validate it
inside the Docker runtime.

LFM2.5-8B-A1B is the current validation model. Qwen35 remains on the roadmap,
but unless a task is Qwen-specific, use LFM2.5 to prove the port for now.

## Quick Reference

| Item | Path |
|---|---|---|
| **Stratum repo** | `/home/harri/stratum/` |
| Entry point | `scripts/train.py` |
| Container launcher | `scripts/run-unified.sh` |
| Container doctor | `scripts/doctor.py` |
| Training config | CLI args (see `train.py --help`) |
| Dockerfile | `Dockerfile` |
| Port tracking | `STRATUM-PORT-TODO.md` |
| Handbook | `HANDBOOK.md` (this file) |
| Claude handover | `CLAUDE.md` |
| **RoundPipe source** (PyPI 0.1.1) | extracted at `/tmp/roundpipe-dl/roundpipe_src/roundpipe/` |
| RoundPipe NF4 module | `/tmp/roundpipe-dl/roundpipe_src/roundpipe/transfer.py` |
| RoundPipe chunked loss | `/tmp/roundpipe-dl/roundpipe_src/roundpipe/models/function.py` |
| **qz-roundpipe repo** | `/home/harri/qz-roundpipe/` |
| LFM25 training script | `/home/harri/qz-roundpipe/scripts/train_lfm25_roundpipe_lora.py` |
| Qwen35 training script | `/home/harri/qz-roundpipe/scripts/train_qwen35_roundpipe_lora.py` |
| RoundPipe NF4 monkeypatch | `/home/harri/qz-roundpipe/scripts/roundpipe_nf4.py` |
| qz-roundpipe LFM25 flash patch reference | `/home/harri/qz-roundpipe/scripts/patch_lfm25_volta_attention.py` (historical filename) |
| qz-roundpipe Qwen35 flash patch reference | `/home/harri/qz-roundpipe/scripts/patch_volta_attention.py` (historical filename) |
| Design doc | `/home/harri/qz-roundpipe/docs/stratum-design.md` |
| RAMP launch script | `/home/harri/qz-roundpipe/scripts/ramp_long_context.sh` |
| **TurboQuant llama.cpp fork** | `/home/harri/turboquant-work/llama-cpp-turboquant/` |
| Host-staged GPU copy source | `/home/harri/turboquant-work/llama-cpp-turboquant/ggml/src/ggml-cuda/ggml-cuda.cu` |
| **Training data** | `/home/harri/qz-roundpipe/data/lfm25_fable_merged_48k_train.labels.jsonl` |
| Data format | Pre-tokenized JSONL, 25K windows, ~43M supervised tokens |
| Data source | Merged fable_5_distillation_merged_cleaned_25k + WithinUsAI pool |
| **Tested model** | `LiquidAI/LFM2.5-8B-A1B` (registered as `lfm25-8b-a1b`) |
| **Hardware** | RTX 3080 10 GiB (~9.6 GiB visible, GPU 0) + V100 (32 GiB, GPU 1) |
| **Docker image** | `stratum:latest` refreshed from `Dockerfile.refresh`; full dependency base from `Dockerfile` |

## Container Workflow

Stratum training is container-first. Do not treat host Python, host CUDA, or
host `nvidia-smi` as proof that the training runtime is healthy.

Canonical probe:

```bash
scripts/run-unified.sh python scripts/doctor.py
```

Canonical training shape:

```bash
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

GPU selection is two-level:

- `STRATUM_GPU` controls Docker's physical NVIDIA device exposure, e.g.
  `STRATUM_GPU=device=0,1`.
- `STRATUM_CUDA_VISIBLE_DEVICES` optionally controls the logical CUDA IDs
  seen by PyTorch inside the container.

Stratum currently assumes the prefix lives on container-local `cuda:0`.
Therefore physical GPU selection should usually be done by Docker mapping the
desired physical devices into container-local `0..N`, then passing Stratum
`--device-ids 0 1 ...` or relying on auto-detection.

### Reference host contract

The current reference validation host is intentionally heterogeneous and
imperfect:

| Container device | Observed GPU | SM | Observed VRAM | Role |
|---|---|---|---|---|
| `cuda:0` | NVIDIA GeForce RTX 3080 | 8.6 | ~9.6 GiB visible to PyTorch | prefix / smaller stage |
| `cuda:1` | Tesla V100-SXM2-32GB | 7.0 | ~31.7 GiB visible to PyTorch | larger stage / postfix |

`scripts/doctor.py` has reported `peer_access` false for both `0->1` and
`1->0`. That no-peer behavior is not a blocker or a rare fallback case; it is
the target runtime shape. Boundary activation and gradient transfers must be
correct over the host-staged pinned-buffer path, adapted from the local
TurboQuant llama.cpp work.

This matters for design reviews: do not claim multi-GPU support based only on
P2P-capable homogeneous cards. Validate on the no-peer host-staged path with
`--tensor-split 10 32` unless the change is provably unrelated to device
placement or transfer.

### Build discipline

Use two Docker build paths:

```bash
# Expensive dependency build. Use when CUDA/PyTorch/kernel/dependency layers change.
docker build -t stratum:latest .

# Fast source refresh. Use for normal code and runtime script changes.
docker build -f Dockerfile.refresh \
  --build-arg STRATUM_REFRESH_BASE=stratum:refresh-base \
  -t stratum:latest .
```

`Dockerfile.refresh` starts from the stable heavy-layer tag
`stratum:refresh-base`, deletes any inherited `/workspace/stratum` tree,
installs Stratum from a minimal temporary source copy, and force-reinstalls
Stratum without touching the heavy CUDA dependency stack. This is the normal
path after source changes. Do not use `stratum:latest` as the refresh base
while also tagging the output as `stratum:latest`; repeated self-layering can
hit Docker's max-depth limit.

The final image should contain runtime code only:

| Path | Intended contents |
|---|---|
| site-packages | installed `stratum` package |
| `/workspace/stratum/scripts/` | runtime entry scripts such as `train.py` and `doctor.py` |
| `/workspace/cache` | mounted cache at runtime, not baked data |
| `/workspace/out` | mounted output at runtime, not baked data |

Documentation-only changes do not require an image rebuild. Docs, tests, git
metadata, local caches, datasets, generated outputs, and model/checkpoint blobs
do not belong in the runtime image. The Docker build generates a tiny temporary
README only because `pyproject.toml` references a README for package metadata;
the real project documentation is not copied into the image.

Do not casually edit lines above the expensive dependency layers in
`Dockerfile`. Docker's cache keys include prior instruction history; even a
comment or early metadata change can invalidate later apt/CUDA build layers.
There was one aborted full rebuild after such an edit; it missed cache early
and was stopped before the CUDA compile work. Keep source-refresh changes in
`Dockerfile.refresh` unless the dependency stack really changed.

The `.dockerignore` is part of the build contract. Build context should be
small and should exclude docs, tests, git state, caches, datasets, outputs, and
model/checkpoint blobs. If context size jumps, fix ignored paths before
trusting build timings.

### Runtime validation rules

After a refresh build, run:

```bash
scripts/run-unified.sh python scripts/doctor.py
```

The doctor currently verifies:

| Component | Expected state |
|---|---|
| CUDA | visible in the container |
| PyTorch | CUDA build importable |
| bitsandbytes | importable, NF4 dequant works on every visible GPU |
| causal-conv1d | importable |
| flash-linear-attention / `fla` | importable for Qwen3.5 linear-attention fast path |
| flash-attn | importable |
| flash-attn-v100 | importable for SM70 path |
| transformers / peft | importable |
| stratum | importable from `/workspace/stratum` |

If Docker prints `Your kernel does not support memory limit capabilities...`,
do not assume `STRATUM_DOCKER_MEMORY` or `STRATUM_DOCKER_MEMORY_SWAP` protects
the host. Use `--host-ram-limit-gib` and the Stratum watchdog for practical RAM
safety.

The launcher bind-mounts the working tree over `/workspace/stratum` so local
changes are visible during normal development. A refresh image is still useful
because it keeps the baked runtime self-consistent for no-mount usage, but it
must not become a repository archive.

### Handover validation notes

For normal code changes, run the host-side unit suite first:

```bash
python3 -m pytest -q tests
```

The runtime container does not currently include `pytest`. Unless the task is
explicitly changing image dependencies, use stdlib `unittest` for focused
container checks instead of installing extra test tooling into the product
image:

```bash
scripts/run-unified.sh python -m unittest \
  tests.test_transfer \
  tests.test_host_staging \
  tests.test_pipeline_prefetch \
  tests.test_runtime
```

Then run the container doctor:

```bash
scripts/run-unified.sh python scripts/doctor.py
```

For transfer, scheduling, optimizer, or runtime-memory changes, follow those
checks with an LFM2.5 no-save smoke on the heterogeneous no-P2P host. Keep the
smoke aligned with the active parity target: batch 2 / 2 microbatches,
`--tensor-split 9 32`, `--max-seq-len 8192`, `--pin-model alloc`,
CPU-offloaded async optimizer, GradScaler, router auxiliary loss, NF4 prefetch,
and `--host-ram-limit-gib 80`.

Use `--no-save` for disposable validation smokes unless save/resume behavior is
the thing under test. Optimizer state files such as `optim_0.pt` and
`optim_1.pt` are large and are not part of the normal safetensors adapter-save
path; only request them with `--save-optimizer-state` for explicit optimizer
checkpoint/resume validation.

Docker-created smoke outputs may be root-owned on the host. Remove disposable
smoke directories from inside the container:

```bash
scripts/run-unified.sh rm -rf /workspace/out/<smoke-dir>
```

### Current validation baseline

As of 2026-06-24, the LFM2.5 reference validation has passed on the no-P2P
heterogeneous host with:

```bash
scripts/run-unified.sh python scripts/train.py \
  --model lfm25-8b-a1b \
  --data /workspace/data/lfm25_fable_merged_48k_train.labels.jsonl \
  --out /workspace/out/lfm25-stratum-validate-b2-mb2 \
  --steps 5 \
  --batch-size 2 \
  --num-microbatch 2 \
  --tensor-split 9 32 \
  --max-seq-len 8192 \
  --longest-first \
  --pin-model alloc \
  --save-every 5 \
  --host-ram-limit-gib 80 \
  --timing-jsonl /workspace/out/lfm25-stratum-validate-b2-mb2/timing.jsonl
```

Observed result:

| Item | Result |
|---|---|
| GPUs | RTX 3080 `cuda:0` + V100 `cuda:1`, peer access unavailable |
| Attention | 6 LFM2.5 full-attention layers patched with capability-dispatched flash attention |
| Placement | 6 decoder layers on GPU0, 18 decoder layers on GPU1 |
| Transfer | Host-staged boundary path used twice per training step |
| Sequence padding | `--pad-to-multiple` auto-raised to 32 for flash attention |
| Step metrics | 5 finite-loss steps, final logged loss `10.5647` |
| Throughput | Warm steps around 3000 tokens/s for batch 2 / 2 microbatches |
| Peaks | GPU0 ~4.6 GiB, GPU1 ~19.3 GiB |
| Checkpoints | `checkpoint-5/` and `final/` wrote PEFT `adapter_model.safetensors` |
| Legacy blobs | No `device_*.pt`, `optim_*.pt`, or `meta.pt` written by default |

This proves the current LFM2.5 NF4 + LoRA + host-staged multi-GPU path for the
reference setup.

An otherwise identical validation run with `--prefetch-nf4` also passed on
2026-06-24. It produced 5 finite-loss steps, final logged loss `10.5622`, warm
throughput around 3030 tokens/s, and PEFT safetensors checkpoint/final saves.
The observed GPU1 peak rose slightly, from ~19.32 GiB to ~19.46 GiB, consistent
with briefly overlapping prefetched NF4 payloads.

These basic checkpoint runs did not prove Qwen35, CPU/offloaded optimizer mode,
or long-run stability. CPU-offloaded optimizer mode was validated separately in
the run below.

An async optimizer + GradScaler validation passed on 2026-06-25:

```bash
STRATUM_DATA_DIR=/home/harri/qz-roundpipe/data \
STRATUM_OUT_DIR=/home/harri/stratum/out \
scripts/run-unified.sh timeout 1200 python scripts/train.py \
  --model lfm25-8b-a1b \
  --data /workspace/data/lfm25_fable_merged_48k_train.labels.jsonl \
  --out /workspace/out/lfm25-async-gradscaler-smoke \
  --steps 2 \
  --batch-size 2 \
  --num-microbatch 2 \
  --tensor-split 9 32 \
  --max-seq-len 8192 \
  --longest-first \
  --pin-model alloc \
  --save-every 2 \
  --save-optimizer-state \
  --host-ram-limit-gib 80 \
  --timing-jsonl /workspace/out/lfm25-async-gradscaler-smoke/timing.jsonl \
  --cpu-offload-optim \
  --async-optimizer-step \
  --grad-scaler-enabled \
  --output-router-logits \
  --router-aux-loss-coef 0.02 \
  --prefetch-nf4 \
  --cuda-memory-summary-on-exception
```

Observed result: finite losses `11.3813` then `10.8578`, standard
`flash_attn` on the RTX 3080 full-attention layer, `flash_attn_v100` on the
V100 full-attention layers, host-staged activation and gradient boundary
transfers, GradScaler init scale `65536.0`, and both `checkpoint-2/` and
`final/` wrote PEFT adapters plus `optim_0.pt` and `optim_1.pt`.

RoundPipe scheduler stage-boundary wiring was then validated on 2026-06-25
with a 1-step LFM2.5 smoke using CPU-offloaded async optimizer, GradScaler,
router aux loss, NF4 prefetch, and `--no-save`. It produced finite loss
`11.2807`, logged scheduler forward/backward wait/notify timing events, used
host-staged RTX 3080<->V100 transfers, selected standard `flash_attn` on SM86
and `flash_attn_v100` on SM70, recorded `layer_forward` timings for global
layer ids `0..23`, and left no smoke output after cleanup.

The later scheduler group-scoped upload/prefetch wiring was validated on the
same LFM2.5 shape on 2026-06-25. The smoke again produced finite loss
`11.2807`, used host-staged RTX 3080<->V100 transfers, selected the expected
SM86/SM70 flash-attention backends, and emitted group-tagged `stage_prefetch`
and `stage_upload` timing records plus global `layer_forward` ids `0..23`.
The smoke output was removed after validation.

Host-staged boundary transfer helper wiring was validated on 2026-06-26 with
the same LFM2.5 feature set: CPU-offloaded async optimizer, GradScaler, router
aux loss, NF4 prefetch, two-GPU tensor split, and `--no-save`. It produced
finite loss `11.3813`, logged host-staged `0->1` forward and `1->0` backward
boundary transfers, selected standard `flash_attn` on SM86 and
`flash_attn_v100` on SM70, and exercised `HostStagingPool` through the shared
`stratum.transfer.async_d2h/async_h2d` graph-preserving copy layer.

The next scheduler slice wires group-level backward completion/free. Stratum
now registers idempotent completion callbacks through tensor hooks. If a group
input tensor has no grad path, `free_all_weights()` completes and frees that
group after backward so checkpoint recompute never sees prematurely emptied
weights.

Low-RSS NF4 construction was validated on 2026-06-27 in Docker with LFM2.5
using `--low-rss-nf4-build`, `--host-ram-limit-gib 45`, `--tensor-split 9 32`,
batch 1, one microbatch, and a dedicated cache
`/workspace/cache/nf4-lowrss-fp16`. The run built the HF skeleton on meta,
materialized 84 trainable LoRA meta parameters, streamed HF safetensors into
NF4 payloads and remaining FP16 staged tensors, then completed a full
forward/backward step with finite loss `11.8045`. RSS was `6.48 GiB` after
pipeline build and `7.12 GiB` after dataloader construction. GPU peaks were
`3.68 GiB` on the RTX 3080 and `14.36 GiB` on the V100. The run logged
host-staged boundary transfers in both directions and completed with
`--no-save`.

The same cache was then reused for a warm-cache smoke. Startup reported
`nf4: all payloads loaded from cache`, confirming that NF4 quantization and
payload writes were skipped. The pipeline-build phase completed in `12.3s`,
RSS stayed at `6.48 GiB` after pipeline build / `7.13 GiB` after dataloader
construction, and the 1-step training run again completed with finite loss
`11.8045`.

Qwen3.5 status as of 2026-06-26:

| Item | Result |
|---|---|
| NF4 scope | `--nf4-scope layers` matches qz-roundpipe's `layers_only=True` behavior and avoids first-time NF4 cache creation over the huge embedding/head tensors |
| NF4 cache | Reduced Qwen smoke created/restored cached layer NF4 payloads: 186 tensors, 9.67 GiB source -> 2.72 GiB payload |
| Smoke run | 1 step at batch 1 / 4096 max sequence passed with finite loss `10.0315` |
| 8K blocker found | Earlier 5-step batch 2 run failed on GPU0 because Qwen full-attention layer 3 fell back to eager attention on the RTX 3080 |
| Fix direction | Qwen full-attention wrapper dispatches to standard `flash_attn` on Ampere+ and `flash_attn_v100` on V100; silent fallback to quadratic eager is not acceptable for long-context training |

Post-fix validation of the Ampere attention path:

| Item | Result |
|---|---|
| Focused RTX 3080 probe | Qwen3.5 full-attention shape `(batch=1, seq=7104, q_heads=16, kv_heads=4, head_dim=256)` selected `flash_attn` and completed forward+backward |
| Probe peak | ~0.49 GiB allocated on GPU0, confirming the full-attention OOM was removed |
| Full original split | 5-step batch 2 / 2 microbatch Qwen run with `--tensor-split 9 32` advanced past full-attention forward and host-staged boundary transfer |
| Linear-attention blocker | Reproduced on 2026-06-26 when the Stratum image lacked `flash-linear-attention[cuda]`: Transformers printed its fast-path-unavailable warning and OOMed in `torch_chunk_gated_delta_rule` during backward recompute on GPU0 |
| FLA-backed 8K smoke | Passed on 2026-06-26 for 1 step at batch 1 / 8192 max sequence with finite loss `11.4141`, `--tensor-split 9 32`, CPU-offloaded async optimizer, GradScaler, NF4 prefetch, host-staged transfers both directions, SM86 `flash_attn`, SM70 `flash_attn_v100`, and `fla 0.5.1` installed |
| Stricter batch 2 / 2 microbatch 8K smoke | Passed on 2026-06-26 for 5 steps, step 1 loss `11.8784`, `--tensor-split 9 32`, 8 layers on GPU0 / 24 on GPU1, SM86 `flash_attn` (layers 3, 7) and SM70 `flash_attn_v100` (layers 11–31), host-staged transfers both directions (4–52 MiB), NF4 all cache hits, GPU0 peak 5.45 GiB, GPU1 peak 13.35 GiB, CPU-offloaded async optimizer, GradScaler, `--no-save` |

### New model adapters (added 2026-06-26)

Three new architecture adapters were added: `stratum/model/llama.py` (Llama),
`stratum/model/qwen3.py` (Qwen3 dense), and `stratum/model/qwen3_moe.py`
(Qwen3-MoE). All follow the same pattern as LFM25/Qwen35: capability-dispatched
flash attention (`_FlashBackend` NamedTuple), recompute context bridge, and
chunked/blocked loss in the postfix. Qwen3-MoE adds router-logit side-channel
capture via `patch_moe_block_for_router_logits`.

Adapter smokes on heterogeneous no-P2P hardware:

| Model | Architecture | Steps | Smoke result |
|---|---|---|---|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | `llama` | 3 | Passed 2026-06-26, finite loss, SM86+SM70 flash dispatch |
| Qwen/Qwen3-0.6B | `qwen3` | 3 | Passed 2026-06-26, finite loss, SM86+SM70 flash dispatch |
| Qwen/Qwen3-30B-A3B | `qwen3-moe` | 3 | Passed 2026-06-26, finite loss, aux loss wired, SM86 `flash_attn`+SM70 `flash_attn_v100`, 11 layers GPU0 / 37 GPU1, 3D expert weights NF4-staged (see NF4 fix note below) |

**NF4 fix for 3D stacked expert weights:** `Qwen3MoeExperts` stores its
weights as 3D `nn.Parameter` tensors (`gate_up_proj: [N_experts, 2*expert_dim,
hidden]`, `down_proj: [N_experts, hidden, expert_dim]`) instead of individual
`nn.Linear` modules. The original `prepare_nf4` skipped them (`ndim != 2`),
causing them to land in Phase 2 permanent GPU upload and OOM on GPU0.

Fix in `stratum/upload.py`: the `ndim != 2` guard was relaxed to `ndim < 2`.
Higher-rank tensors are reshaped to `[-1, last_dim]` before bitsandbytes
`quantize_4bit` (which requires 2D input); `payload.shape` retains the original
tensor shape so all dequant paths (`upload_stream`, `ensure_weights`,
`NF4Prefetch.finalize`) reconstruct the correct-rank output tensor.

### Recent code-state changes (post-documentation)

The codebase has moved ahead of this document in several areas after the
"Align model wrappers with qz-roundpipe" commit (33b3963) and the
"Add Qwen Ampere flash attention path" commit (4bb332a):

| Change | Effect |
|---|---|
| `copy.deepcopy` removed from prefix/postfix | Prefix and postfix now reference original model modules directly instead of deep-copying. This reduces CPU RAM and ensures PEFT LoRA adapters stay intact. |
| `dense_attention_masks` wired into LFM25 prefix | Matches Qwen35; both architectures now support full causal mask construction. |
| `debug_finite` added to LFM25 wrapped layer + postfix | NaN/Inf detection now available on both architectures. |
| `ensure_weights()` handles shared frozen weights | Frozen weights shared across prefix/stages/postfix rematerialize from CPU NF4 payload per-device instead of crashing. |
| `checkpoint_decoder_layer` defaults to `True` | Activation checkpointing is on by default in `train.py`. |
| Qwen35 `_FlashBackend` NamedTuple pattern | Structured backend selection with hard failure on flash kernel error; no silent OOMs from eager fallback. |
| LFM25 `Lfm25FlashAttention` | Uses the same `_FlashBackend` pattern as Qwen35; CUDA flash backend absence or kernel failure is fatal, not a silent eager fallback. |

### Parity status (updated 2026-06-27)

Stratum has reached practical full-feature parity with qz-roundpipe on the
reference no-P2P RTX 3080 + V100 machine. Every RoundPipe mechanism that is
compatible with the spanned-weight multi-GPU architecture has been ported or
adapted. The table below is the complete parity ledger.

| Capability | RoundPipe source | Status |
|---|---|---|
| NF4 frozen-weight streaming | `roundpipe/transfer.py`, `roundpipe_nf4.py` | **Done** — `prepare_nf4` / `ensure_weights` / `free_weights`; cache hits/misses on every smoke |
| NF4 3D stacked weights | (new — `Qwen3MoeExperts`) | **Done** — `ndim < 2` guard; reshape to 2D for bitsandbytes; `payload.shape` retains original rank |
| NF4 prefetch | `roundpipe_nf4.py` | **Done** — `NF4Prefetch` / `prefetch_weights` / `ensure_prefetched_weights` |
| Flash attention SM86 (Ampere) | `patch_volta_attention.py` | **Done** — `_FlashBackend` pattern; hard failure on kernel error |
| Flash attention SM70 (V100) | `patch_volta_attention.py` | **Done** — `flash_attn_v100` dispatch; same hard-failure policy |
| Host-staged no-P2P transfers | `ggml-cuda.cu` (TurboQuant reference) | **Done** — `HostStagingPool`; `async_d2h`/`async_h2d` on boundary tensors; graph-preserving |
| Chunked cross-entropy loss | `roundpipe/models/function.py` | **Done** — `chunked_linear_cross_entropy` + `BlockedPostfixCausalLMLoss` |
| MLP optimizations | `train_lfm25_roundpipe_lora.py:120–298` | **Done** — `CheckpointedModule`, `TokenChunkedModule`, `MemoryFlatFrozenMLP`, `apply_mlp_optimizations` |
| CPU-offloaded async optimizer | `roundpipe/optim_stream.py` | **Done** — `PerDeviceOptimizer` + `launch_optim_kernel` / `on_optim_stream` / `synchronize_optim` |
| GradScaler (async-safe) | `roundpipe/grad_scaler.py` | **Done** — `stratum/grad_scaler.py`; compatible with CPU-offload path |
| Recompute context bridge | `roundpipe/context.py` | **Done** — `ForwardCtx` / `RecomputeCtx` / `save_for_recompute`; PyTorch checkpoint `context_fn` bridge |
| Scheduler group execution | `roundpipe/scheduler.py` | **Done** — `ModelExecutePlan` / `DeviceStage.forward_range` / `run_explicit_group_backward` |
| Per-layer CUDA event timing | `roundpipe/timer.py` | **Done** — `LayerTimingContext` / `IterLayerTimer` / `ModelLayerTimer`; EMA estimates per layer |
| Timing-fed plan adaptation | `roundpipe/timer.py` + scheduler | **Done** — `set_layer_timer(timer, adapt_every_n=N)`; `_try_adapt_plan` fires every N steps; wired in `train.py` via `--adapt-plan-every N` |
| Async H2D upload stream | `roundpipe/device.py` | **Done** — `param_upstream` stream per device; `_upload_group_with_fence`; `non_blocking=True` H2D copies |
| Per-layer upload/compute overlap | RoundPipe scheduler | **Done** — layer i+1 H2D submitted before layer i fence-wait in `forward_range`; recompute path covered |
| PEFT safetensors checkpoints | `train_lfm25_roundpipe_lora.py` | **Done** — `save_checkpoint` / `load_checkpoint`; optimizer `.pt` is opt-in |
| MoE router aux loss | `train_lfm25_roundpipe_lora.py` | **Done** — `patch_moe_block_for_router_logits` / `pop_router_logits` / `load_balancing_loss_func` |
| LFM2.5 adapter | reference script | **Done** — `stratum/model/lfm25.py`; smoked at batch 2 / 2 microbatch / 8K |
| Qwen3.5 adapter | `train_qwen35_roundpipe_lora.py` | **Done** — `stratum/model/qwen35.py`; smoked at batch 2 / 2 microbatch / 8K |
| Llama adapter | `roundpipe/models/llama.py` | **Done** — `stratum/model/llama.py`; TinyLlama-1.1B smoke passed |
| Qwen3 dense adapter | `roundpipe/models/qwen3.py` | **Done** — `stratum/model/qwen3.py`; Qwen3-0.6B smoke passed |
| Qwen3-MoE adapter | `roundpipe/models/qwen3_moe.py` | **Done** — `stratum/model/qwen3_moe.py`; Qwen3-30B-A3B smoke passed |
| Non-NF4 layer-copy path | `roundpipe/transfer.py` | **Done** — `prepare_fp16_staged` / `copy_tensor_chunked` lifecycle; mutable-buffer snapshot for recompute is an open tail |
| Pytree batch API | `roundpipe/batch.py` | **Done** — `guess_split_spec` / `split_pytree` / `merge_pytree` / `TokenWeightedReducer`; `--pytree-batch` flag in `train.py` |
| Sample packing | `roundpipe/batch.py` | **Done** — `pack_samples` / `pack_collate` / `split_packed_batch` in `stratum/packing.py`; `--packing` flag; flash_attn_varlen_func dispatch on LFM25 and Qwen35; LFM2.5 packing smoke passed 2026-06-27 (batch 2 / 2 microbatch / 8192 ctx, finite loss 7.883, 791 trainable tokens, host-staged boundary transfers, ShortConv seq_idx boundary reset, MoE packed-mode wrapper) |
| Host RAM management | — | **Done for LFM2.5 / partial for Qwen35 depth** — `release_cached_memory()` frees cached FP16 pages after NF4 prep (measurable: 11+ GiB RSS drop on the normal path); `--low-rss-nf4-build` builds the HF skeleton on meta and streams checkpoint tensors during NF4 preparation. LFM2.5 low-RSS NF4 Docker smoke passed on 2026-06-27; Qwen35 low-RSS depth is still future validation |
| GPT-OSS adapter | `roundpipe/models/gpt_oss.py` | **Skipped** — no public HF model available |

**What is not yet done (in priority order):**

1. **Mutable-buffer snapshot/restore** — `--no-nf4` recompute paths that
   mutate buffers (e.g., running stats in norms) could silently corrupt
   recompute. Low priority: the `--no-nf4` path is rarely used and the
   issue only surfaces when a layer mutates a buffer during forward.

2. **Sliding-window mask audit** — Qwen3 sliding-window mask behavior
   against RoundPipe's reference is a future audit item. All public Qwen3
   models have `has_sliding_layers=False`; no correctness risk on current
   targets.

3. **Longer save/resume validation** — multi-step resume under CPU optimizer
   offload, checkpoint failure modes. Unit coverage exists; end-to-end
   integration runs are thin. Host unit coverage now verifies opt-in Adam
   moment round-trip without legacy `device_*.pt` state.

4. **NUMA coordination** — future Stratum optimization (binding staging
   buffers, NF4 prefetch, CPU optimizer, worker affinity to NUMA topology).
   Not a RoundPipe parity item.

## RoundPipe Comparison — Ported, Adapted, Still Missing

Stratum uses qz-roundpipe as a single-GPU reference prototype for model
wrappers and memory mechanisms, then replaces the runtime with a custom
spanned multi-GPU pipeline. The goal is to keep the useful reference-backed
mechanisms while adapting them to Stratum's heterogeneous staged architecture,
including host-staged no-P2P boundary transfer.

See `STRATUM-PORT-TODO.md` for the active parity backlog and implementation
order.

### Training mechanisms

| Mechanism | RoundPipe | Stratum | Status |
|---|---|---|---|
| NF4 frozen weight streaming | `NF4Linear` JIT-dequant on GPU | CPU→GPU H2D + dequant per step | Different approach, same semantics |
| Chunked lm_head loss | `ChunkedCompileLinearForCausalLMLoss` custom autograd | `ChunkedLinearCrossEntropyFunction` in postfix | Equivalent custom per-chunk backward |
| Blocked postfix loss | `BlockedPostfixCausalLMLoss` (batch=1 only) | Same, ported to `blocked_loss.py` | Identical |
| Microbatching | `num_microbatch` plus pytree split/merge hooks | `stratum.batch` fixed-tensor + pytree split/merge (`guess_split_spec`, `split_pytree`, `merge_pytree`, `TokenWeightedReducer`); `--pytree-batch` flag | Token-weighted training path; generic pytrees supported |
| Activation checkpointing | `checkpoint(run_layer, ...)` per decoder layer | LFM25 and Qwen35 | Identical for current adapters |
| MLP checkpointing | `CheckpointedModule` | Same, in `mlp_opt.py` | Identical |
| MLP token chunking | `TokenChunkedModule` | Same, in `mlp_opt.py` | Identical |
| Memory-flat frozen MLP | `MemoryFlatFrozenMLP` custom autograd | Same, in `mlp_opt.py` | Identical |
| Capability-dispatched flash attention | qz-roundpipe flash patch reference scripts | `--flash-layers`, `--flash-window-*`; standard `flash_attn` on SM80+/SM86 and `flash_attn_v100` on SM70 | Stratum-native heterogeneous dispatch |
| LoRA adapter checkpoint | `base.save_pretrained()` (PEFT safetensors) | Same via `hf_model.save_pretrained()` + JSON trainer state | Identical; legacy `.pt` is opt-in only |

### CLI flags

Most training flags from `train_lfm25_roundpipe_lora.py` and
`train_qwen35_roundpipe_lora.py` exist in `scripts/train.py`. RoundPipe runtime
flags that depend on its scheduler need Stratum-native replacements rather than
literal ports. See CLI reference below and `STRATUM-PORT-TODO.md`.

### RoundPipe internals — adaptation backlog

These RoundPipe runtime components should not be copied verbatim, but their
useful capability should be adapted where it helps Stratum:

| RoundPipe internal | What it does | Current Stratum state | Adaptation direction |
|---|---|---|---|
| `upload_layers()` | Copies layers to GPU with chunked async upload | `ensure_weights()` + `free_weights()` handle frozen NF4; `stratum.layer_transfer.upload_layer_copies()` provides standalone module-copy utility | Add optional prefetch/chunked non-NF4 runtime path |
| `download_layer()` | Async gradient D2H after backward | CPU-offloaded optimizer copies now gather trainable grads through `PerDeviceOptimizer`; `stratum.layer_transfer.download_layer_state()` remains a standalone copy helper | Extend event-fenced overlap if Stratum later ports RoundPipe's per-layer scheduler |
| `PinnedUpload` autograd | Pins tensors for async H2D, copies grads back | `stratum.transfer.PinnedUpload` exists; sync NF4 path does not need the autograd wrapper | Use only for pageable CPU-input/offload paths that need autograd H2D; boundary transfers use graph-preserving async helpers directly |
| `RegisterBackwardEvent` | CUDA event sync for upload→backward ordering | `stratum.transfer.RegisterBackwardEvent` exists; no current race in sync weight path | Use when async prefetch/offload introduces upload→backward ordering hazards |
| `ModelExecutePlan` / `ModelTracker` | Per-layer fwd/bwd scheduling with memory budget and semaphore ordering | `stratum.scheduler` ports the plan/tracker/tag/chunk primitives; `StratumPipeline` uses them for stage-boundary and intra-stage scheduler groups, group-scoped upload/prefetch, active explicit autograd-anchor backward, group-level backward completion/free, per-layer forward/recompute timings, group-level backward timings, and bidirectional boundary-transfer timings | Extend to custom async recompute/backward streams and deeper per-layer overlap with host-staged boundary transfers preserved |
| `DeviceManager` | Per-device stream management (upstream/downstream/compute) | `HostStagingPool` covers boundary transfers only | Add explicit stream/event semantics for async paths |
| `ParamAttribute` / `LayerAttribute` | Per-param upload/grad state tracking | `ParamAttribute` backs CPU optimizer copies; layer fencing remains adapted to Stratum stages | Add per-layer fences only if Stratum ports deeper RoundPipe overlap |
| `optim_stream.py` | Background optimizer thread with kernel queue | **PORTED** as `stratum.optim_stream`; async CPU-offloaded optimizer path validated on LFM2.5 | Keep live-parameter copyback fenced at step boundaries |
| `RoundPipeBase.optim_named_parameters()` | Lazy fp32 CPU copies of trainable params | **PORTED** in `PerDeviceOptimizer` for `--cpu-offload-optim` | Same |
| `RoundPipeBase._move_grad_to_optim()` | Gather GPU grads → fp32 CPU optim copies | **PORTED** for CPU-offloaded trainable params | Same |
| `RoundPipeBase.sync_optim_param()` | Copy updated CPU params back to GPU | **PORTED** for CPU-offloaded trainable params | Same |
| `RoundPipeBase.step(is_async)` | Async orchestrator with event fencing | **ADAPTED** via `--async-optimizer-step`; CUDA grad reads are fenced by per-device default-stream events, and synchronization is deferred to the next safe iteration boundary | Deeper per-layer overlap needs explicit Stratum forward fences |
| `context.py` (ForwardCtx/RecomputeCtx) | Thread-local forward/recompute markers + `save_for_recompute()` | **PORTED/ADAPTED** as `stratum.context` plus PyTorch `checkpoint(..., context_fn=...)` bridge in LFM25/Qwen35 wrapped layers; checkpoint recompute now emits per-layer save/enter/exit timing records, and `_router_logits` is kept out of saved recompute payloads | Custom RoundPipe scheduler RNG/offload semantics remain future work |
| `grad_scaler.py` | Dual-scaler GradScaler for async optim stream | **PORTED** as `stratum.grad_scaler`; scheduler is gated on actual non-skipped steps | Same |
| `pin_module_alloc` / `pin_module_register` | CPU memory pinning strategies | **PORTED** to `stratum/memory.py` | Same |
| `async_d2h` / `async_h2d` | Async host-device with event sync | **RUNTIME-WIRED** into `HostStagingPool` host-staged boundary transfers with graph-preserving `out=` buffers; default detached mode remains for optimizer/offload utilities | Extend the same helper layer to future activation offload paths |
| `batch.py` | Pytree microbatch split/merge/reduce | Fixed training tensors use token-weighted helpers | Add generic split specs if future wrappers need them |

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

### qz-roundpipe compatibility notes

| Feature | Reason |
|---|---|
| Operator telemetry hooks | Debugging only, no correctness impact |
| Memory watchdog `/proc/self/status` | Safety guard, **PORTED** to `watchdog.py` |
| `--pin-model` strategies | **PORTED** to `memory.py` |
| `--roundpipe-model-memory-limit-gib` | Replaced by `--stratum-stage-memory-limit-gib` plus `--nf4-layer-size-floor-gib` |
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
| `layer_transfer.py` | RoundPipe-style standalone layer upload/download helpers for chunked module copies and grad/buffer download |
| `grad_hooks.py` | `make_boundary_hook()` — backward gradient hook that transfers grads across devices |
| `runtime.py` | RoundPipe-style explicit group recompute/backward primitive and autograd anchor helper used by active scheduler groups; stream/device orchestration still lives in `pipeline.py` |
| `packing.py` | Sample packing: `pack_samples`, `pack_collate`, `split_packed_batch` for padding-free training |
| `batch.py` | Pytree batch API: `guess_split_spec`, `split_pytree`, `merge_pytree`, `split_kwargs_pytree`, `TokenWeightedReducer` |

### Weight streaming (NF4) — the VRAM enabler

| File | Purpose |
|---|---|
| `upload.py` | `prepare_nf4()` → quantizes frozen 2D weights, attaches NF4Payload, drops originals. `prepare_nf4_from_cache()` → loads precomputed NF4 payloads from disk without re-quantization when used with meta/staged construction. `ensure_weights()` → uploads NF4→dequant to FP16 per-stage before forward. `free_weights()` → sets param.data=empty(0) after backward. `estimate_module_upload_gib()` → NF4-savvy size estimation. `load_module_fp16_from_checkpoint()` → streams individual module weights from the HF checkpoint (safe_open) for staged loading. |
| `nf4_linear.py` | `NF4Linear` — frozen Linear with 4-bit weight on GPU, JIT-dequant in forward. **Currently not used** — we use CPU→GPU streaming instead. |

### Model architectures — adding new models

| File | Purpose |
|---|---|
| `model/registry.py` | `ModelArch` base class, `@register("name")` decorator, `build_pipeline()` entry |
| `model/lfm25.py` | LFM2.5-8B-A1B: prefix, wrapped layer, postfix, `Lfm25FlashAttention`; packed-mode dispatch to `flash_attn_varlen_func` |
| `model/qwen35.py` | Qwen3.5-9B: same structure, `Qwen35FlashAttention` with sliding window; packed-mode dispatch |
| `model/llama.py` | Llama-family adapter (TinyLlama validated) |
| `model/qwen3.py` | Qwen3 dense adapter (Qwen3-0.6B validated) |
| `model/qwen3_moe.py` | Qwen3-MoE adapter (Qwen3-30B-A3B validated); router aux loss |
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
| `checkpoint.py` | `save_checkpoint()` / `load_checkpoint()` — PEFT safetensors + JSON metadata by default; legacy `.pt` opt-in |
| `utils.py` | device detection, `gpu_memory_snapshot()`, `host_rss_gib()`, `release_cached_memory()`, `get_optimal_tensor_split()` |

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
  ├── optional --prefetch-nf4:
  │     prefetch_weights(next stage) copies NF4 payloads on a side stream,
  │     then ensure_prefetched_weights() fences and dequants before use
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

**`--no-nf4` mode:** `prepare_fp16_staged()` implements the same per-step
lifecycle but without compression. Large frozen 2D params (above `nf4_min_numel`)
are pinned on CPU as FP16 payloads (marked `FP16_ATTR`), uploaded to GPU each
step via `copy_tensor_chunked()`, and freed after backward. Small params (biases,
layer norms) and trainable LoRA params are still uploaded permanently at build
time. `ensure_weights()` and `free_weights()` handle both NF4 and FP16-staged
params transparently.

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

The host-staged fallback is adapted from Harri's TurboQuant llama.cpp work:
`/home/harri/turboquant-work/llama-cpp-turboquant/ggml/src/ggml-cuda/ggml-cuda.cu`,
specifically `ggml_cuda_copy_across_devices()` and
`ggml_cuda_copy2d_across_devices()`. The key algorithmic point is a reusable
pinned host staging pool that avoids per-copy `cudaMallocHost` overhead and
falls back cleanly when CUDA peer access is unavailable.

On dual-socket hosts, Stratum currently benefits from pinned memory and CUDA
DMA, but it is not yet NUMA-aware. It does not bind staging buffers, NF4
prefetch work, CPU optimizer work, or allocation first-touch to the CPU socket
nearest a given GPU. That is tracked as a future Stratum extension in
`STRATUM-PORT-TODO.md` section 16a, not as a current qz-roundpipe parity
blocker.

During backward, `make_boundary_hook()` uses the same `HostStagingPool` when
one is supplied, so gradients also take the P2P-or-host-staged path back to the
previous device. The synchronous host-staged backward path copies the
destination H2D leg on the destination current stream, matching RoundPipe's
upload-to-compute handoff and preventing side-stream-produced gradients from
reaching `AccumulateGrad`. It falls back to `.to(device)` only when no pool is
available.

Explicit scheduler-group backward follows the same stream contract:
`StratumPipeline` passes the stage CUDA default stream into
`anchor_explicit_group_backward()`, and `run_explicit_group_backward()` wraps
both recompute and `torch.autograd.backward()` in that stream. This mirrors
RoundPipe `run.py`, where forward, recompute, and backward run under
`device.compute_stream`.

Router-logit side channels need one extra ownership rule in explicit group
replay: tensors already present in a group input are pass-through leaves and
stay on their original graph edge; only tensors newly produced by that group
are anchored for recompute/backward. This avoids later V100 groups replaying
earlier RTX 3080 router logits while preserving router aux loss gradients.

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

Default checkpoints are LoRA/QLoRA-style and portable:

```
checkpoint-5000/
├── adapter_model.safetensors     ← PEFT-compatible LoRA weights (portable)
├── adapter_config.json            ← PEFT adapter config
└── trainer_state.json             ← step number and lightweight metadata
```

**Saving:** `hf_model.save_pretrained(out_dir)` produces `adapter_model.safetensors`
+ `adapter_config.json`. This works because PEFT only saves LoRA parameters,
not base model frozen weights. Even though `prepare_nf4()` has set frozen
weights to `empty(0)`, PEFT ignores them.

Stratum no longer writes `device_{id}.pt`, `optim_{id}.pt`, or `meta.pt` by
default. Those files are same-layout legacy/debug artifacts and are explicitly
opt-in with:

```bash
--save-legacy-device-state
--save-optimizer-state
```

Do not enable those flags for normal LoRA/QLoRA training unless same-layout
optimizer resume is worth the extra disk use. `optim_{id}.pt` files are
PyTorch optimizer `state_dict()` files sharded by Stratum device id; they are
exact-resume state, not portable adapter artifacts. Portable adapter
checkpoints should stay small and safetensors-first.

Disk policy:

- Normal checkpoint: `adapter_model.safetensors`, `adapter_config.json`,
  `README.md`, and `trainer_state.json`.
- Exact-resume checkpoint: opt in with `--save-optimizer-state`; delete after
  the resume path has been validated unless the run must be continued exactly.
- Legacy per-device weights: opt in with `--save-legacy-device-state` only for
  debugging old layouts.
- Smoke tests: do not use `--save-optimizer-state` unless the smoke is
  specifically about optimizer-state save/load.

**Loading (resume):** `safetensors.torch.load_file()` → `hf_model.load_state_dict(strict=False)`.
Prefers PEFT adapter, reads `trainer_state.json`, and falls back to legacy
`.pt` files only for old checkpoints.

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
| `--batch-size` | 2 | Batch size before optional microbatch splitting |
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
| `--pytree-batch` | False | Use pytree-based batch splitting for arbitrary input shapes |
| `--packing` | False | Sample packing: concatenate variable-length samples into 1D sequences with per-segment position IDs, eliminating wasted compute on padding tokens |
| `--checkpoint-decoder-layer` | True | Activation checkpointing per decoder layer |
| `--no-nf4` | False | Disable NF4 frozen weight compression (FP16 direct upload) |
| `--nf4-scope` | `all` | NF4 prep scope: `all` includes prefix/stages/postfix; `layers` matches qz-roundpipe layers-only prep for large Qwen embedding/head tensors |
| `--low-rss-nf4-build` | False | Build the HF module skeleton on meta and stream checkpoint tensors during NF4 preparation instead of loading the full FP16 model into host RAM |
| `--nf4-min-numel` | 4096 | Minimum frozen 2D parameter elements to NF4-quantize |
| `--nf4-layer-size-floor-gib` | 0.0 | qz-roundpipe-style scheduler hint: floor each layer's estimated stage-planning size |
| `--nf4-cache-dir` | `/workspace/cache/nf4-frozen` | Directory to cache NF4 payloads; a model-id subdirectory is added automatically |
| `--pin-model` | `alloc` | CPU pinning: alloc (pin_memory), register (cudaHostRegister), off |
| `--stratum-stage-memory-limit-gib` | 0.0 | Split per-device layer groups into smaller upload/free stages |
| `--prefetch-nf4` | False | Experimental side-stream NF4 payload prefetch for the next stage/postfix |
| `--recompute-grain` | `layer` | `layer` keeps per-layer checkpointing; `none` disables decoder-layer recompute |
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
| `--flash-layers` | `""` | Comma-separated full-attention layer indices to patch; empty = all |
| `--flash-window-left` | -1 | Sliding window left tokens (-1 = full attention) |
| `--flash-window-right` | 0 | Sliding window right tokens (0 = causal) |
| `--attn-implementation` | `flash` | Stratum-owned heterogeneous flash dispatch; HF loads eager, then Stratum patches full-attention layers |

### Data loading
| Flag | Default | Description |
|---|---|---|
| `--longest-first` | False | Sort dataset by seq_len descending |
| `--pad-to-multiple` | 0 | Pad batch length to multiple of N; flash attention automatically raises 0 to 32 |
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
| `--timing-jsonl` | `""` | Write pipeline timing spans to JSONL |
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

For this to work, the pipeline modules MUST share Parameter objects with
`hf_model`. This is ensured by `ModelArch.build()` and the model adapters,
which use the original prefix, layer, and postfix modules rather than deep
copies.

Shared frozen NF4 weights can appear in different stages, for example tied
embedding and LM-head weights split across prefix/postfix on different GPUs.
`ensure_weights()` rematerializes those tensors from the CPU NF4 payload on
the device that is about to run. Trainable shared parameters spanning multiple
Stratum stages are not supported without explicit optimizer/autograd handling;
the upload path raises instead of silently moving them between GPUs.

### 4. `BlockedPostfixCausalLMLoss` requires batch_size=1

This is a RoundPipe limitation preserved in the port. The error is raised
at runtime if batch > 1 with `--postfix-loss-token-chunk-size > 0`.

### 5. Qwen3.5 wrapped layers support `checkpoint_decoder_layer`

Both LFM25 and Qwen35 wrapped layers use the `checkpoint_decoder_layer` flag.
Keep this enabled for long-context runs unless a specific regression is being
isolated.

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
   - FlashAttention wrapper with capability-dispatched backend selection
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

## Testing and Invocation

### Training data

Dataset: `/home/harri/qz-roundpipe/data/lfm25_fable_merged_48k_train.labels.jsonl`

- Pre-tokenized JSONL format: `{"input_ids": [...], "attention_mask": [...], "labels": [...]}`
- 25,000 windows from the fable_5_distillation_merged_cleaned_25k + WithinUsAI merged pool
- ~43 million supervised tokens
- Sequence lengths up to 48K (truncated by `--max-seq-len`)
- Labels use -100 for ignored positions (padding / non-supervised tokens)

### Build Docker image

```bash
cd /home/harri/stratum
docker build -t stratum:latest .
```

The build compiles:
- `flash-attn-v100` (for V100, sm_70) — from source
- `flash-attn` (for RTX 3080, sm_86) — from source, `FLASH_ATTN_CUDA_ARCHS="86"`
- `causal_conv1d` — both sm_70 + sm_86 in one wheel

If CUDA compilation fails, check:
- Docker daemon running: `sudo systemctl start docker`
- Disk space: `df -h /` (need ~20 GiB free for build cache)
- Source changes don't invalidate CUDA cache because `COPY stratum` is after all compilations

### Starting Docker with Stratum

For ad-hoc runs (no volume mounts):
```bash
docker run --gpus all --rm -it \
    -v /home/harri/qz-roundpipe/data:/data \
    stratum:latest \
    python scripts/train.py --data /data/lfm25_fable_merged_48k_train.labels.jsonl ...
```

For development (live code changes visible inside container):
```bash
docker run --gpus all --rm -it \
    -v /home/harri/stratum:/workspace/stratum \
    -v /home/harri/qz-roundpipe/data:/data \
    stratum:latest \
    bash
# Inside container: python scripts/train.py --data /data/... --out /runs/test ...
```

### Smoke test (quick validation)

```bash
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

Expected: 5 steps, prints JSON step logs, no OOM, no NaN, loss should start
around 11-13 (random init for 124K vocab) and decrease.

### Longer test (convergence check)

```bash
docker run --gpus all --rm stratum:latest python scripts/train.py \
    --model lfm25-8b-a1b \
    --data /data/lfm25_fable_merged_48k_train.labels.jsonl \
    --tensor-split 9 32 \
    --steps 500 \
    --batch-size 1 \
    --num-microbatch 1 \
    --lr 1e-4 \
    --warmup-steps 50 \
    --lr-scheduler cosine_with_warmup \
    --save-every 100 \
    --out /runs/long-test
```

### Single-GPU test (for debugging on one card)

```bash
docker run --gpus all --rm stratum:latest python scripts/train.py \
    --model lfm25-8b-a1b \
    --data /data/lfm25_fable_merged_48k_train.labels.jsonl \
    --steps 5 --batch-size 1 --save-every 0 --no-save
```

### Test with MLP optimizations

```bash
docker run --gpus all --rm stratum:latest python scripts/train.py \
    --model lfm25-8b-a1b \
    --data /data/lfm25_fable_merged_48k_train.labels.jsonl \
    --tensor-split 9 32 \
    --steps 5 --batch-size 1 \
    --checkpoint-mlp
```

### Test with torch-compiled loss

```bash
docker run --gpus all --rm stratum:latest python scripts/train.py \
    --model lfm25-8b-a1b \
    --data /data/lfm25_fable_merged_48k_train.labels.jsonl \
    --tensor-split 9 32 \
    --steps 5 --batch-size 1 \
    --torch-compile-loss
```

### Test with blocked postfix loss (batch=1 only)

```bash
docker run --gpus all --rm stratum:latest python scripts/train.py \
    --model lfm25-8b-a1b \
    --data /data/lfm25_fable_merged_48k_train.labels.jsonl \
    --tensor-split 9 32 \
    --steps 5 --batch-size 1 \
    --postfix-loss-token-chunk-size 2048
```

### Restarting Docker daemon

```bash
sudo systemctl start docker   # if stopped
sudo systemctl status docker  # check status
```

### Monitoring during training

- Check GPU memory: `nvidia-smi -l 1`
- Step logs printed as JSON every 10 steps: `{"step": N, "loss": ..., "sec": ..., "tok_s": ..., "gpu0_used": ..., "gpu1_used": ...}`
- Logged to `training.jsonl` in output dir every step
- If loss is NaN: check for ignored-token chunks with `reduction="mean"` (should be `"sum"`)
- If OOM: reduce batch size, increase `--num-microbatch`, enable `--checkpoint-mlp`, or reduce `--loss-token-chunk-size`

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
