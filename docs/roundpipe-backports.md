# What Stratum Can Give Back to RoundPipe

Stratum was built on RoundPipe parity and has since accumulated improvements
that are not specific to multi-GPU training. This document records what is worth
backporting to qz-roundpipe / RoundPipe for single-GPU single-machine use.

---

## Planned contribution PRs

Four improvements are ready to be applied as separate PRs to a local fork of
qz-roundpipe, tested on LFM2.5, and then submitted upstream. Each is
self-contained and can be reviewed independently.

**PR 1 — 3D expert NF4 fix** (`roundpipe_nf4.py`)  
One-file change. Change `param.ndim != 2` to `param.ndim < 2`, add a reshape
to `[-1, weight.shape[-1]]` before `quantize_4bit`, store original shape in the
cache payload, reshape back on dequant. Completes the "first spike" the module
comment deferred. Test: LFM2.5 NF4 prep with and without the fix; verify expert
tensors appear in `nf4 total` count and VRAM use drops on GPU with <10 GiB.

**PR 2 — Low-RSS NF4 build** (`train_lfm25_roundpipe_lora.py`, `roundpipe_nf4.py`)  
Add `--low-rss-nf4-build` flag. Port `materialize_trainable_meta_parameters()`
from `stratum/train.py:98–120` and `load_module_fp16_from_checkpoint()` from
`stratum/upload.py`. Load model on meta device, stream frozen weights from
safetensors during `prepare_nf4_frozen_params()` instead of loading full FP16
first. Also add `release_cached_memory()` (gc + malloc_trim) after NF4 prep.
Test: measure peak RSS before and after on LFM2.5; expect ~26 GiB reduction.

**PR 3 — Token-weighted microbatch loss** (`train_lfm25_roundpipe_lora.py`)  
Replace `AvgReducer` microbatch loss averaging with token-count-weighted sum.
Three lines in the microbatch loop. Test: verify loss value is unchanged for
equal-length microbatches; verify it changes (correctly) for unequal lengths.

**PR 4 — Sample packing** (`train_lfm25_roundpipe_lora.py`, new `roundpipe_packing.py`)  
Add `--packing` flag. Port `stratum/packing.py` (`pack_samples`, `pack_collate`,
`split_packed_batch`) as `roundpipe_packing.py`. Wire into the data collator and
route attention through `flash_attn_varlen_func` when packing is active. Test:
LFM2.5 10-step smoke with and without `--packing`; verify trainable token count
increases and loss is comparable.

Suggested order: PR 1 first (smallest, highest standalone value), then PR 2
(biggest RAM impact), then PR 3 (correctness fix), then PR 4 (most complex).

Local fork: `/home/harri/qz-roundpipe/` — work branches off `master` there,
one branch per PR, test via the existing `scripts/run-unified.sh` pattern.

---

Analysis is based on a direct comparison of `stratum/` and `scripts/train.py`
against `/home/harri/qz-roundpipe/scripts/train_lfm25_roundpipe_lora.py` and
`/tmp/roundpipe-dl/roundpipe_src/roundpipe/`.

---

## Priority 1 — Actual bugs in RoundPipe

### LR schedule broken on resume

RoundPipe restores adapter weights on `--resume` but discards the step counter.
The training loop restarts at `step=1` and the LR scheduler restarts from
warmup. For any run that was resumed mid-training the effective LR history is
wrong from that point forward.

**Fix:** write `trainer_state.json` containing `{"step": N}` alongside
`save_pretrained()`. On resume, read it back and start the loop from
`range(step+1, args.steps+1)`, passing `last_epoch=step` to the scheduler.

Stratum reference: `stratum/checkpoint.py:save_checkpoint()`,
`train.py:650–661`.

---

### MoE expert tensors silently skipped during NF4 prep

`roundpipe_nf4.py:prepare_nf4_frozen_params()` has the guard:

```python
if param.ndim != 2:
    continue
```

This is not an oversight — the `roundpipe_nf4.py` module docstring explicitly
describes itself as *"an intentionally narrow monkeypatch... uploads for frozen
2D weights"* and the function docstring says *"this first spike is about
reducing GPU staging and PCIe traffic."* The `ndim != 2` guard was a stated
scope limit of a prototype that was always meant to be revisited.

The problem is that the prototype became the production path before the 3D case
was handled. MoE expert weight tensors are 3D (`[n_experts, dim_out, dim_in]`).
The guard silently leaves them as full FP16 on GPU permanently with no warning.
For any MoE model (LFM2.5, Qwen MoE, etc.) this causes unexpectedly high VRAM
use and eventual OOM on smaller cards.

**Fix:** change to `ndim < 2`, reshape to `[-1, weight.shape[-1]]` before
`quantize_4bit`, store the original shape in the cache payload for
reconstruction. This is the work the "first spike" comment was deferring.

Stratum reference: `stratum/upload.py` — the `prepare_nf4()` reshape path.

---

### Microbatch loss averaging is mathematically wrong

`AvgReducer` averages per-microbatch losses. If microbatches have different
numbers of label tokens (they always do with variable-length sequences), the
average-of-losses is not equal to the loss over the full batch.

**Fix:** weight each microbatch loss by its fraction of the total non-padding
token count before summing.

```python
total_trainable = sum(mb.trainable_tokens for mb in microbatches)
loss += mb_loss * (mb.trainable_tokens / total_trainable)
```

Stratum reference: `train.py:745`, `stratum/batch.py:microbatch_loss_scale()`.

---

## Priority 2 — Large RAM savings

### `release_cached_memory()` after NF4 prep

After `prepare_nf4_frozen_params()` zeroes out FP16 frozen weights
(`param.data = empty(0)`), glibc holds the freed pages in its heap cache and
does not return them to the OS. RSS stays high even though the memory is
notionally free.

**Fix:** call `gc.collect()` then `ctypes.CDLL("libc.so.6").malloc_trim(0)`
immediately after prep. On the LFM2.5 reference host this drops RSS by 11+ GiB.

Stratum reference: `stratum/utils.py:release_cached_memory()`,
called at `train.py:547`.

---

### Meta-device NF4 build (`--low-rss-nf4-build`)

RoundPipe calls `AutoModelForCausalLM.from_pretrained(..., device_map="cpu")`
which loads the entire FP16 model into RAM (~16 GiB for LFM2.5 8B) before NF4
prep begins. During prep the FP16 weights and NF4 payloads coexist, peaking
at ~32+ GiB.

**Fix:** load the model skeleton on the `meta` device (zero RAM), materialize
only the trainable LoRA parameters on CPU, then during `prepare_nf4_frozen_params()`
stream each frozen tensor directly from the HF safetensors checkpoint file via
`safetensors.safe_open` and quantize it immediately without ever holding the
full FP16 model in RAM. Peak RSS becomes max(single module FP16 size) ≈ 2–3 GiB
for a single transformer stage.

On the LFM2.5 reference host: **~26 GiB RSS saving**.

Stratum reference: `train.py:443–461` (meta load),
`train.py:98–120` (`materialize_trainable_meta_parameters`),
`stratum/upload.py:load_module_fp16_from_checkpoint()` (streaming loader).

---

### `--nf4-scope {all,layers}`

Without scope control, `prepare_nf4_frozen_params()` processes the full PEFT
model uniformly including the embedding table and lm_head. For large-vocabulary
models (Qwen 150k vocab) these tensors are ~700 MiB each and the bitsandbytes
quantize peak is larger than the input tensor. On hosts with <40 GiB RAM this
causes OOM during NF4 prep even though the final compressed model fits easily.

**Fix:** add a `layers_only` flag that limits prep to `model.layers.*` weights,
leaving embeddings and lm_head as FP16 on CPU (they stay cold during training
anyway and are only read once per step during the token embedding and loss
computation).

Stratum reference: `train.py:237`, `build_pipeline(nf4_scope=...)`.

---

## Priority 3 — Throughput

### Sample packing (`--packing`)

RoundPipe always pads variable-length sequences to the longest in the batch.
Padding tokens are masked out for loss but still consume compute in every
attention and MLP layer. At long context (8192+) with short training samples,
padding can account for 80%+ of tokens in a batch.

**Fix:** pack samples by concatenating them into one continuous sequence with
per-segment position IDs and a `cu_seqlens` tensor. Flash attention dispatches
to `flash_attn_varlen_func` which skips padding natively. No model weight
changes required — only the data collator and attention call change.

Stratum reference: `stratum/packing.py` (`pack_samples`, `pack_collate`,
`split_packed_batch`). The module is self-contained with no Stratum-specific
dependencies.

---

## Priority 4 — Observability and operability

### Experiment tracking (MLflow)

RoundPipe has no experiment tracking. Stratum added MLflow with a SQLite
backend: hyperparams logged once at run start, metrics (loss, lr, tok/s,
rss_gib, per-GPU GiB) logged each step. A separate named Docker container
serves the UI, persists after training ends, and survives container restarts.

Stratum reference: `train.py:355–383`, `scripts/run-unified.sh` MLflow block.

---

### Valid JSON step logs

RoundPipe prints step dicts as `print({...}, flush=True)` — Python repr, not
valid JSON. Keys with special characters and float values like `nan`/`inf` break
any downstream parser.

**Fix:** `print(json.dumps(step_dict), flush=True)`.

---

### Per-step LR and host RSS in step log

RoundPipe step logs contain loss, sec, tokens, tok/s, elapsed. They do not
include the current LR (critical for verifying warmup and decay are working) or
host RSS (critical for catching slow memory leaks during long runs).

**Fix:** add `"lr": scheduler.get_last_lr()[0]` and
`"rss_gib": proc_status_kib("VmRSS") / 1024**2` to the step dict.

---

### Dual-channel output (stdout JSON / stderr bar / log file)

RoundPipe mixes structured step dicts, diagnostic prints, tqdm, and warnings
all on stdout. This makes it impossible to pipe step output to a log processor
without filtering noise.

**Fix:** step JSON on stdout only, tqdm bar and diagnostics on stderr, full
audit log to `{out}/training.jsonl`. A `--verbose` flag gates diagnostic prints
so they are off by default.

Stratum reference: `train.py:57–59` (`jprint`), `train.py:665–671` (tqdm),
`train.py:348–350` (log file), `train.py:275` (`--verbose`).

---

### Optimizer state checkpoint

RoundPipe saves adapter weights but not optimizer state. Adam moments are lost
on every resume. The optimizer re-warms from zero which wastes the first several
hundred steps of a resumed run.

**Fix:** `torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")` at
checkpoint. `optimizer.load_state_dict(torch.load(...))` on resume. Guard with
`--save-optimizer-state` flag since the file is large (~2× LoRA param count in
FP32).

---

## Not applicable to single-GPU RoundPipe

| Stratum feature | Reason not applicable |
|---|---|
| `HostStagingPool` / `--tensor-split` / `--device-ids` | Multi-GPU boundary transfer |
| `--prefetch-nf4` side-stream H2D | RoundPipe already does per-layer H2D overlap via `upload_layers()` |
| `--timing-jsonl` boundary spans | No multi-stage boundaries on single GPU |
| `--adapt-plan-every` timing-fed placement | `ModelExecutePlan.auto()` already exists in RoundPipe |
| `--stratum-stage-memory-limit-gib` | Equivalent: `--roundpipe-model-memory-limit-gib` |
| Flash attention V100 / SM86 dispatch | Already in RoundPipe's flash path |
| `HostStagingPool` pin strategies | N/A |
