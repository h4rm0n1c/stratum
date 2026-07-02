#!/usr/bin/env python3
"""Audit and stage high-precision Fable 5 dataset candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stratum.data.fable5 import (  # noqa: E402
    build_inventory,
    build_48k_lineage,
    build_candidate_traces,
    build_manifest,
    build_review_samples,
    env_config,
    load_48k_seed_hashes,
    summarize_prepared_48k,
    write_json,
    write_jsonl,
    write_markdown_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/fable5-audit"),
        help="Directory for audit artifacts.",
    )
    parser.add_argument(
        "--seed-limit-per-file",
        type=int,
        default=None,
        help="Limit 48K rows read per file for fast dry-runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory", help="Write source_inventory.json.")
    sub.add_parser("manifest", help="Write trace_manifest.jsonl and manifest_summary.json.")
    sub.add_parser("calibrate", help="Write 48k_seed_hashes.json.")
    sub.add_parser("lineage", help="Write recovered 48K lineage JSON.")
    sample_parser = sub.add_parser("sample", help="Write review sample JSON.")
    sample_parser.add_argument("--max-segments", type=int, default=5)
    sub.add_parser("export", help="Write complete high-confidence candidate traces.")
    sub.add_parser("all", help="Run inventory, manifest, calibrate, sample, and markdown report.")

    args = parser.parse_args()
    config = env_config()
    out_dir: Path = args.out_dir

    if args.command in {"inventory", "all"}:
        inventory = build_inventory(config)
        write_json(out_dir / "source_inventory.json", inventory)
        print(json.dumps({"wrote": str(out_dir / "source_inventory.json"), "sources": len(inventory["sources"])}))

    if args.command in {"manifest", "all"}:
        manifest = build_manifest(config, seed_limit_per_file=args.seed_limit_per_file)
        write_jsonl(out_dir / "trace_manifest.jsonl", manifest["manifest_rows"])
        write_json(out_dir / "manifest_summary.json", manifest)
        print(
            json.dumps(
                {
                    "wrote": str(out_dir / "trace_manifest.jsonl"),
                    "segments": manifest["summary"]["claude_fable5_segments"],
                }
            )
        )

    if args.command in {"calibrate", "all"}:
        seed = load_48k_seed_hashes(config.seed_48k, limit_per_file=args.seed_limit_per_file)
        seed["prepared_summary"] = summarize_prepared_48k(config.seed_48k, limit_per_file=args.seed_limit_per_file)
        write_json(out_dir / "48k_seed_hashes.json", seed)
        print(json.dumps({"wrote": str(out_dir / "48k_seed_hashes.json"), "unique_sha16": seed["unique_sha16"]}))

    if args.command in {"lineage", "all"}:
        lineage = build_48k_lineage(config, limit_per_file=args.seed_limit_per_file)
        write_json(out_dir / "48k_lineage.json", lineage)
        print(
            json.dumps(
                {
                    "wrote": str(out_dir / "48k_lineage.json"),
                    "datasets": len(lineage["datasets"]),
                    "primary": lineage["conclusion"]["primary_long_context_seed"],
                }
            )
        )

    if args.command in {"sample", "all"}:
        manifest = build_manifest(config, seed_limit_per_file=args.seed_limit_per_file)
        samples = build_review_samples(manifest["manifest_rows"], max_segments=getattr(args, "max_segments", 5))
        write_json(out_dir / "review_samples.json", samples)
        print(json.dumps({"wrote": str(out_dir / "review_samples.json"), "samples": len(samples)}))

    if args.command in {"export", "all"}:
        traces = build_candidate_traces(config)
        write_jsonl(out_dir / "candidate_traces.jsonl", traces)
        print(json.dumps({"wrote": str(out_dir / "candidate_traces.jsonl"), "traces": len(traces)}))

    if args.command == "all":
        inventory = build_inventory(config)
        manifest = build_manifest(config, seed_limit_per_file=args.seed_limit_per_file)
        lineage = build_48k_lineage(config, limit_per_file=args.seed_limit_per_file)
        write_markdown_report(out_dir / "calibration_report.md", inventory, manifest, lineage)
        print(json.dumps({"wrote": str(out_dir / "calibration_report.md")}))


if __name__ == "__main__":
    main()
