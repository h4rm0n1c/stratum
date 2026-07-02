# Stratum Reference Map

Stratum is a reference-backed port, not a clean-room rewrite. Keep these local
paths as the first places to check before changing training behavior.

## Primary Sources

| Area | Reference |
| --- | --- |
| qz-roundpipe prototype | `/home/harri/qz-roundpipe/` |
| LFM2.5 RoundPipe training script | `/home/harri/qz-roundpipe/scripts/train_lfm25_roundpipe_lora.py` |
| Qwen3.5 RoundPipe training script | `/home/harri/qz-roundpipe/scripts/train_qwen35_roundpipe_lora.py` |
| qz NF4 RoundPipe monkeypatch | `/home/harri/qz-roundpipe/scripts/roundpipe_nf4.py` |
| qz LFM2.5 Volta flash patch | `/home/harri/qz-roundpipe/scripts/patch_lfm25_volta_attention.py` |
| qz Qwen3.5 Volta flash patch | `/home/harri/qz-roundpipe/scripts/patch_volta_attention.py` |
| Extracted RoundPipe package | `/tmp/roundpipe-dl/roundpipe_src/roundpipe/` |
| RoundPipe optimizer orchestration | `/tmp/roundpipe-dl/roundpipe_src/roundpipe/roundpipe.py` |
| RoundPipe transfer/upload cycle | `/tmp/roundpipe-dl/roundpipe_src/roundpipe/transfer.py` |
| RoundPipe run scheduler | `/tmp/roundpipe-dl/roundpipe_src/roundpipe/run.py` |
| RoundPipe chunked loss | `/tmp/roundpipe-dl/roundpipe_src/roundpipe/models/function.py` |
| TurboQuant llama.cpp fork | `/home/harri/turboquant-work/llama-cpp-turboquant/` |
| Multi-GPU host-staged fallback source | `/home/harri/turboquant-work/llama-cpp-turboquant/ggml/src/ggml-cuda/ggml-cuda.cu` |

For any multi-GPU operation that crosses devices without CUDA peer access,
check the TurboQuant llama.cpp fork before changing Stratum transfer behavior.
The key source functions are `ggml_cuda_copy_across_devices()` and
`ggml_cuda_copy2d_across_devices()`: they are the local reference for reusable
pinned host staging, P2P detection/fallback, and avoiding per-copy pinned
allocation overhead.

## Dependency Source Lookup

Do not vendor huge dependency trees into this repo. For exact installed source,
query the unified development container:

```bash
scripts/run-unified.sh python -c "import importlib.util as u; mods=['torch','transformers','causal_conv1d','flash_attn','flash_attn_v100','bitsandbytes','peft']; print({m:(u.find_spec(m).origin if u.find_spec(m) else None) for m in mods})"
```

Use the returned site-package paths for line-level checks against the versions
Stratum is actually running.
