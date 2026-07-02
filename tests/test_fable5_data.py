import json
import tempfile
import unittest
from pathlib import Path

from stratum.data.fable5 import (
    ClaudeTraceScanner,
    InventoryConfig,
    VOXTALK_FABLE5_ANCHOR,
    build_48k_lineage,
    build_inventory,
    build_candidate_traces,
    build_manifest,
    load_48k_seed_hashes,
)


class Fable5DataTest(unittest.TestCase):
    def test_claude_scanner_requires_anchor_and_model_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            log = project / "session.jsonl"
            rows = [
                {
                    "type": "user",
                    "message": {"role": "user", "content": "<command-name>/model</command-name>"},
                    "sessionId": "s1",
                    "cwd": "/home/harri/voxtalk",
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": "<local-command-stdout>Set model to Fable 5 and saved</local-command-stdout>",
                    },
                    "sessionId": "s1",
                    "cwd": "/home/harri/voxtalk",
                },
                {
                    "type": "user",
                    "message": {"role": "user", "content": VOXTALK_FABLE5_ANCHOR},
                    "sessionId": "s1",
                    "timestamp": "2026-07-01T21:35:04.531Z",
                    "cwd": "/home/harri/voxtalk",
                    "gitBranch": "master",
                },
                {
                    "type": "assistant",
                    "message": {
                        "model": "claude-fable-5",
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": ""},
                            {"type": "tool_use", "name": "mcp__ghidra-mcp__decompile_function_by_address"},
                        ],
                    },
                    "sessionId": "s1",
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}],
                    },
                    "sessionId": "s1",
                },
            ]
            log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            segments = ClaudeTraceScanner().scan_project_dirs([project])
            traces = build_candidate_traces(
                InventoryConfig(
                    qz_sources=(),
                    seed_48k=(),
                    claude_projects=(project,),
                    codex_dirs=(),
                    hf_candidates=(),
                    hf_quarantine=(),
                )
            )

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.session_id, "s1")
        self.assertEqual(segment.start_line, 3)
        self.assertIn("line:4:message.model=claude-fable-5", segment.model_evidence)
        self.assertEqual(segment.tool_uses, 1)
        self.assertEqual(segment.tool_results, 1)
        self.assertEqual(segment.thinking_blocks, 1)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["events"][0]["line"], 3)
        self.assertEqual(traces[0]["events"][-1]["line"], 5)

    def test_claude_scanner_ignores_last_prompt_anchor_without_user_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "session.jsonl"
            log.write_text(
                json.dumps(
                    {
                        "type": "last-prompt",
                        "lastPrompt": VOXTALK_FABLE5_ANCHOR,
                        "sessionId": "s1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(ClaudeTraceScanner().scan_file(log), [])

    def test_load_48k_seed_hashes_reads_sha16_without_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / "seed.labels.jsonl"
            seed.write_text(
                "\n".join(
                    [
                        json.dumps({"sha16": "aaa", "input_ids": [1], "labels": [1]}),
                        json.dumps({"sha16": "bbb", "input_ids": [2], "labels": [2]}),
                        json.dumps({"sha16": "aaa", "input_ids": [3], "labels": [3]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = load_48k_seed_hashes([seed])

        self.assertEqual(loaded["unique_sha16"], 2)
        self.assertEqual(loaded["files"][0]["rows"], 3)
        self.assertEqual(loaded["files"][0]["hashes"], 3)

    def test_inventory_and_manifest_are_private_default_deny(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "claude"
            project.mkdir()
            config = InventoryConfig(
                qz_sources=(Path(tmp) / "pool_merged_codex.jsonl",),
                seed_48k=(Path(tmp) / "lfm25_fable_merged_48k_train.labels.jsonl",),
                claude_projects=(project,),
                codex_dirs=(),
                hf_candidates=("Glint-Research/Fable-5-traces",),
                hf_quarantine=("TheFusionCube/Fable-5-CoT-Traces",),
            )
            inventory = build_inventory(config)
            manifest = build_manifest(config)

        self.assertEqual(inventory["policy"]["privacy"], "private_training_only")
        states = {row["path"]: row["state"] for row in inventory["sources"]}
        self.assertEqual(states["Glint-Research/Fable-5-traces"], "candidate_priority1")
        self.assertEqual(states["TheFusionCube/Fable-5-CoT-Traces"], "quarantine")
        self.assertEqual(manifest["summary"]["claude_fable5_segments"], 0)

    def test_48k_lineage_maps_prepared_dataset_to_recovered_source_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / "lfm25_fable_merged_48k_train.labels.jsonl"
            seed.write_text(
                json.dumps(
                    {
                        "sha16": "abc",
                        "row_id": "source-1:w0",
                        "seq_len": 49152,
                        "supervised_tokens": 1024,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = InventoryConfig(
                qz_sources=(),
                seed_48k=(seed,),
                claude_projects=(),
                codex_dirs=(),
                opencode_db=Path(tmp) / "missing.db",
                hf_candidates=(),
                hf_quarantine=(),
            )

            lineage = build_48k_lineage(config, limit_per_file=10)

        merged = next(row for row in lineage["datasets"] if row["run_name"] == "lfm25_fable_merged_48k")
        self.assertEqual(merged["source_pool"], "pool_merged_codex")
        self.assertEqual(merged["train_summary"]["rows"], 1)
        self.assertEqual(merged["train_summary"]["full_49152_windows"], 1)
        self.assertEqual(lineage["conclusion"]["primary_long_context_seed"], "lfm25_fable_merged_48k")


if __name__ == "__main__":
    unittest.main()
