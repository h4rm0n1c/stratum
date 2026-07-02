# Sample Packing Implementation Plan

## Goal

Replace padding-based batching with sample packing for LLM training in Stratum. **Completed 2026-06-27.**

## Status

| Phase | File(s) | Status |
|---|---|---|
| 1. Packed collation | `stratum/packing.py` — `pack_samples`, `pack_collate`, `compute_cu_seqlens`, `split_packed_batch` | ✅ |
| 2. Flash attention varlen dispatch | `stratum/model/lfm25.py`, `stratum/model/qwen35.py` — `_select_varlen_flash_backend`, packed QKV path, `flash_attn_varlen_func` call | ✅ |
| 3. Prefix packing awareness | `stratum/model/lfm25.py`, `stratum/model/qwen35.py` — `_is_packed` detection, cu_seqlens passthrough | ✅ |
| 4. Train.py wiring | `scripts/train.py` — `--packing` flag, packed collation, batch unpacking, microbatch split | ✅ |
| 5. Tests | `tests/test_packing.py` — 25 tests | ✅ |

## Notes

- LFM2.5's ShortConv (causal_conv1d) layers are incompatible with 1D packed
  input — they require 3D `(batch, seq, hidden)`. Packing works correctly on
  standard transformer architectures (Qwen3.5, Llama, etc.)
- Microbatching with packing splits at cu_seqlens boundaries (each microbatch
  gets whole samples). Token-weighted loss scaling is preserved.
