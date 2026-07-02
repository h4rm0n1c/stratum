# Fable 5 Dataset Audit

Stratum includes an offline-first audit pipeline for building a private,
high-precision Fable 5 training corpus before long-context label generation.

Run a bounded dry audit:

```bash
python3 scripts/fable5_dataset.py --out-dir out/fable5-audit --seed-limit-per-file 100 all
```

Run a full prepared-48K hash scan by omitting `--seed-limit-per-file`.

## Policy

- Treat `lfm25_fable_*_48k` labels as the positive calibration seed.
- Treat newer `qz-fable/reconstructed` pools as priority source candidates.
- Include local Claude VoxTalk/Stratum traces only when Fable 5 provenance is explicit.
- Quarantine generic OpenCode/DeepSeek traces, mirrors, synthetic expansions, and weakly labeled Fable 5 datasets by default.
- Preserve source and window provenance before any final train/validation label build.

## Verified VoxTalk Anchor

The first known local VoxTalk Fable 5 trace starts at:

```text
/home/harri/.claude/projects/-home-harri-voxtalk/900b539d-c88b-4f20-b38b-b0cc8518939d.jsonl:28326
can you do some more RE to doco for me? we'll leave impl for later.
```

The scanner requires the real user anchor row, not only repeated `last-prompt`
metadata. It then checks nearby local model-switch evidence and subsequent
assistant records with `model:"claude-fable-5"`. Tool-call-heavy RE traces keep
their tool-use/tool-result counts and tool names in the manifest.

## Outputs

- `source_inventory.json`: discovered source paths, mtimes, sizes, and default state.
- `trace_manifest.jsonl`: high-confidence candidate trace regions and provenance.
- `48k_seed_hashes.json`: prepared 48K `sha16` hashes for calibration/replay.
- `48k_lineage.json`: recovered build lineage for each prepared 48K dataset.
- `review_samples.json`: bounded source excerpts for manual review.
- `candidate_traces.jsonl`: complete high-confidence traces with tool calls/results preserved.
- `calibration_report.md`: source-state and verified-trace summary.

## Recovered 48K Lineage

The existing prepared 48K files were not self-describing enough to fully
recover raw source rows from the label files alone. The audit now combines
OpenCode command logs, source-pool mtimes, row-count agreement, and prepared
label stats.

Confirmed chain:

- `pool.jsonl` -> `pool_codex.jsonl` via `homogenize_tools_to_codex.py`.
- `pool_codex.jsonl` -> `lfm25_fable_48k` and `lfm25_fable_codex_48k`.
- `pool_codex.jsonl` -> `pool_unified.jsonl` via `rebuild_pool_with_cot.py`.
- `pool_unified.jsonl` -> `pool_unified_codex.jsonl`.
- `pool_unified_codex.jsonl` -> `lfm25_fable_unified_48k`.
- `pool_unified_codex.jsonl` + WithinUsAI rows -> `pool_merged.jsonl`.
- `pool_merged.jsonl` -> `pool_merged_codex.jsonl`.
- `pool_merged_codex.jsonl` -> `lfm25_fable_merged_48k`.

`lfm25_fable_merged_48k` is the main long-context seed. Its source pool has
25,922 rows with thinking: 21,060 `within_us`, 4,665 `kelexine`, 172
`glint_cot`, 22 `glint_raw`, and 3 `747` rows.

Environment overrides use path-separated lists:

```bash
STRATUM_FABLE5_CLAUDE_PROJECTS=/path/a:/path/b \
STRATUM_FABLE5_48K_SEEDS=/path/train.labels.jsonl \
python3 scripts/fable5_dataset.py manifest
```
