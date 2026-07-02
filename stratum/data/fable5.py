"""Fable 5 dataset inventory and provenance helpers.

This module intentionally does not tokenize or train.  It builds the audit
surface needed before long-context label generation: source inventory, local
Claude Fable 5 provenance detection, 48K seed hash loading, and review samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Iterator


VOXTALK_FABLE5_ANCHOR = "can you do some more RE to doco for me? we'll leave impl for later."

DEFAULT_QZ_SOURCES = [
    Path("/home/harri/qz-fable/reconstructed/pool/pool_merged_codex.jsonl"),
    Path("/home/harri/qz-fable/reconstructed/pool/pool_unified_codex.jsonl"),
    Path("/home/harri/qz-fable/reconstructed/pool/pool_codex.jsonl"),
    Path("/home/harri/qz-fable/real_fable"),
    Path("/home/harri/qz-fable/validation_fable"),
    Path("/home/harri/fable5_data/glint-update/fable5_cot_merged.jsonl"),
    Path("/home/harri/fable5_data/kelexine/data/train-00000-of-00001.parquet"),
    Path("/home/harri/fable5_data/fusioncube"),
]

DEFAULT_48K_DATASETS = [
    Path("data/lfm25_fable_merged_48k_train.labels.jsonl"),
    Path("data/lfm25_fable_merged_48k_validation.labels.jsonl"),
    Path("data/lfm25_fable_codex_48k_train.labels.jsonl"),
    Path("data/lfm25_fable_codex_48k_validation.labels.jsonl"),
    Path("data/lfm25_fable_instruct_48k_train.labels.jsonl"),
    Path("data/lfm25_fable_instruct_48k_validation.labels.jsonl"),
    Path("data/lfm25_fable_unified_48k_train.labels.jsonl"),
    Path("data/lfm25_fable_unified_48k_validation.labels.jsonl"),
    Path("data/lfm25_fable_48k_train.labels.jsonl"),
    Path("data/lfm25_fable_48k_validation.labels.jsonl"),
]

DEFAULT_CLAUDE_PROJECTS = [
    Path("/home/harri/.claude/projects/-home-harri-voxtalk"),
    Path("/home/harri/.claude/projects/-home-harri-stratum"),
]

SHA16_RE = re.compile(r'"sha16"\s*:\s*"([^"]+)"')

POOL_LINEAGE = {
    "pool_codex": {
        "path": Path("/home/harri/qz-fable/reconstructed/pool/pool_codex.jsonl"),
        "input_path": Path("/home/harri/qz-fable/reconstructed/pool/pool.jsonl"),
        "command": (
            "python3 /workspace/qz-roundpipe/scripts/homogenize_tools_to_codex.py "
            "--output /workspace/qz-fable/reconstructed/pool/pool_codex.jsonl"
        ),
        "opencode_time_ms": 1782101721545,
        "purpose": "normalizes mixed Claude/agent tool-call formats into Codex-style traces",
    },
    "pool_unified": {
        "path": Path("/home/harri/qz-fable/reconstructed/pool/pool_unified.jsonl"),
        "input_path": Path("/home/harri/qz-fable/reconstructed/pool/pool_codex.jsonl"),
        "command": (
            "python3 /workspace/qz-roundpipe/scripts/rebuild_pool_with_cot.py "
            "--output /workspace/qz-fable/reconstructed/pool/pool_unified.jsonl"
        ),
        "opencode_time_ms": 1782126670109,
        "purpose": "extracts embedded <think> blocks and collapses the usable pool to rows with thinking",
    },
    "pool_unified_codex": {
        "path": Path("/home/harri/qz-fable/reconstructed/pool/pool_unified_codex.jsonl"),
        "input_path": Path("/home/harri/qz-fable/reconstructed/pool/pool_unified.jsonl"),
        "command": (
            "python3 /workspace/qz-roundpipe/scripts/homogenize_tools_to_codex.py "
            "--input /workspace/qz-fable/reconstructed/pool/pool_unified.jsonl "
            "--output /workspace/qz-fable/reconstructed/pool/pool_unified_codex.jsonl"
        ),
        "opencode_time_ms": 1782126699636,
        "purpose": "Codex-normalized unified thinking pool",
    },
    "pool_merged": {
        "path": Path("/home/harri/qz-fable/reconstructed/pool/pool_merged.jsonl"),
        "input_path": Path("/home/harri/qz-fable/reconstructed/pool/pool_unified_codex.jsonl"),
        "command": "python3 /workspace/qz-roundpipe/scripts/merge_withinus_to_pool.py",
        "opencode_time_ms": 1782153544299,
        "purpose": "adds WithinUsAI reasoning rows to the unified pool",
    },
    "pool_merged_codex": {
        "path": Path("/home/harri/qz-fable/reconstructed/pool/pool_merged_codex.jsonl"),
        "input_path": Path("/home/harri/qz-fable/reconstructed/pool/pool_merged.jsonl"),
        "command": (
            "python3 /workspace/qz-roundpipe/scripts/homogenize_tools_to_codex.py "
            "--input /workspace/qz-fable/reconstructed/pool/pool_merged.jsonl "
            "--output /workspace/qz-fable/reconstructed/pool/pool_merged_codex.jsonl"
        ),
        "opencode_time_ms": 1782153561444,
        "purpose": "Codex-normalized merged pool used by the main long-context 48K dataset",
    },
}

PREPARED_48K_LINEAGE = {
    "lfm25_fable_48k": {
        "source_pool": "pool_codex",
        "source_mode": "glint_cot",
        "command": (
            "python3 scripts/prepare_lfm25_data.py --source glint_cot "
            "--max-len 49152 --run-name lfm25_fable_48k"
        ),
        "opencode_time_ms": 1782095786280,
        "notes": [
            "A prior --source all attempt failed because fable5_cot_merged.jsonl was not mounted at the expected path.",
            "The successful build used Pool B filtering: kelexine + glint_cot rows with non-empty thinking.",
        ],
    },
    "lfm25_fable_instruct_48k": {
        "source_pool": "pool_codex",
        "source_mode": "glint_cot",
        "command": (
            "python3 scripts/prepare_lfm25_data.py --source glint_cot "
            "--max-len 49152 --run-name lfm25_fable_instruct_48k --model LiquidAI/LFM2.5-8B-A1B"
        ),
        "opencode_time_ms": 1782098464251,
        "notes": ["Retokenized Pool B with the instruct tokenizer/model id."],
    },
    "lfm25_fable_codex_48k": {
        "source_pool": "pool_codex",
        "source_mode": "glint_cot",
        "command": (
            "python3 /workspace/qz-roundpipe/scripts/prepare_lfm25_data.py "
            "--run-name lfm25_fable_codex_48k"
        ),
        "opencode_time_ms": 1782101817670,
        "notes": ["Built after prepare_lfm25_data.py was changed to read pool_codex.jsonl by default."],
    },
    "lfm25_fable_unified_48k": {
        "source_pool": "pool_unified_codex",
        "source_mode": "glint_cot",
        "command": (
            "python3 /workspace/qz-roundpipe/scripts/prepare_lfm25_data.py "
            "--run-name lfm25_fable_unified_48k"
        ),
        "opencode_time_ms": 1782126766409,
        "notes": ["Built from the extracted-thinking unified pool."],
    },
    "lfm25_fable_merged_48k": {
        "source_pool": "pool_merged_codex",
        "source_mode": "glint_cot",
        "command": (
            "python3 /workspace/qz-roundpipe/scripts/prepare_lfm25_data.py "
            "--run-name lfm25_fable_merged_48k"
        ),
        "opencode_time_ms": 1782153632844,
        "notes": ["Built from the WithinUsAI-augmented merged pool; this is the main long-context seed."],
    },
}


@dataclass(frozen=True)
class InventoryConfig:
    """Inputs for a Fable 5 dataset audit run."""

    qz_sources: tuple[Path, ...] = tuple(DEFAULT_QZ_SOURCES)
    seed_48k: tuple[Path, ...] = tuple(DEFAULT_48K_DATASETS)
    claude_projects: tuple[Path, ...] = tuple(DEFAULT_CLAUDE_PROJECTS)
    opencode_db: Path = Path("/home/harri/.local/share/opencode/opencode.db")
    codex_dirs: tuple[Path, ...] = (
        Path("/home/harri/.codex/sessions"),
        Path("/home/harri/.qz-codex/codex-home/sessions"),
    )
    hf_candidates: tuple[str, ...] = (
        "Glint-Research/Fable-5-traces",
        "kelexine/fable-5-sft-traces",
    )
    hf_quarantine: tuple[str, ...] = (
        "TheFusionCube/Fable-5-CoT-Traces",
        "HelioAI/Fable-5-Distill-Reasoning-462x",
    )


@dataclass
class ClaudeTraceSegment:
    """High-confidence contiguous Claude trace region."""

    source_file: str
    session_id: str
    start_line: int
    end_line: int
    start_timestamp: str | None
    cwd: str | None
    git_branch: str | None
    anchor: str
    model_evidence: list[str] = field(default_factory=list)
    model_evidence_count: int = 0
    assistant_turns: int = 0
    thinking_blocks: int = 0
    tool_uses: int = 0
    tool_results: int = 0
    tool_names: list[str] = field(default_factory=list)
    prompt_count: int = 0
    byte_estimate: int = 0

    def to_manifest_row(self) -> dict[str, Any]:
        text = "|".join(
            [
                self.source_file,
                self.session_id,
                str(self.start_line),
                str(self.end_line),
                self.anchor,
            ]
        )
        return {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "source_family": "claude_local",
            "source_path": self.source_file,
            "session_id": self.session_id,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_timestamp": self.start_timestamp,
            "cwd": self.cwd,
            "git_branch": self.git_branch,
            "model_evidence": self.model_evidence,
            "model_evidence_count": self.model_evidence_count,
            "tool_names": self.tool_names,
            "tool_uses": self.tool_uses,
            "tool_results": self.tool_results,
            "assistant_turns": self.assistant_turns,
            "thinking_blocks": self.thinking_blocks,
            "prompt_count": self.prompt_count,
            "byte_estimate": self.byte_estimate,
            "calibration": {"seed_48k_overlap": None, "style_match": "pending"},
            "state": "candidate_high_confidence",
            "review_reason": "verified_voxtalk_fable5_anchor",
        }


def utc_iso_from_stat(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def source_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if path.is_dir():
        return "directory"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return "sqlite"
    return suffix.lstrip(".") or "file"


def file_size(path: Path) -> int | None:
    try:
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return path.stat().st_size
    except OSError:
        return None


def classify_source(path: Path) -> tuple[str, str]:
    text = str(path).lower()
    if "fusioncube" in text or "decoy" in text:
        return "quarantine", "known_decoy_or_low_confidence_fable5_label"
    if "opencode" in text:
        return "quarantine", "generic_opencode_requires_model_and_quality_gate"
    if "deepseek" in text:
        return "quarantine", "deepseek_traces_default_quarantine"
    if "pool_merged_codex" in text:
        return "seed_source_priority0", "newest_reconstructed_qz_fable_pool"
    if "pool_unified_codex" in text or "pool_codex" in text:
        return "candidate_priority0", "reconstructed_qz_fable_pool"
    if "real_fable" in text or "validation_fable" in text:
        return "seed_source_priority0", "real_fable_trace_source"
    if "glint" in text or "kelexine" in text:
        return "candidate_priority1", "hf_local_candidate_with_known_provenance"
    if "claude" in text:
        return "candidate_priority1", "local_claude_requires_model_evidence"
    if "codex" in text:
        return "candidate", "local_codex_requires_calibration"
    return "candidate", "unclassified_source"


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_no, json.loads(stripped), line
            except json.JSONDecodeError:
                yield line_no, {"_malformed_json": True, "_raw_prefix": stripped[:256]}, line


def json_get_path(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(parts)
    return ""


def assistant_content_stats(content: Any) -> tuple[int, int, list[str]]:
    thinking = 0
    tool_uses = 0
    tool_names: list[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            typ = item.get("type")
            if typ == "thinking":
                thinking += 1
            elif typ == "tool_use":
                tool_uses += 1
                name = item.get("name")
                if isinstance(name, str):
                    tool_names.append(name)
    return thinking, tool_uses, tool_names


class ClaudeTraceScanner:
    """Find Fable 5 Claude trace regions in local Claude JSONL logs."""

    def __init__(self, anchor: str = VOXTALK_FABLE5_ANCHOR) -> None:
        self.anchor = anchor

    def scan_project_dirs(self, dirs: Iterable[Path]) -> list[ClaudeTraceSegment]:
        segments: list[ClaudeTraceSegment] = []
        for directory in dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.jsonl")):
                segments.extend(self.scan_file(path))
        return segments

    def scan_file(self, path: Path) -> list[ClaudeTraceSegment]:
        rows = list(iter_jsonl(path))
        anchors = [idx for idx, (_, row, _) in enumerate(rows) if self._is_anchor_row(row)]
        if not anchors:
            return []
        segments: list[ClaudeTraceSegment] = []
        for pos, idx in enumerate(anchors):
            next_idx = anchors[pos + 1] if pos + 1 < len(anchors) else len(rows)
            segment_rows = rows[max(0, idx - 3):next_idx]
            segment = self._build_segment(path, rows[idx][0], segment_rows)
            if segment.model_evidence:
                segments.append(segment)
        return segments

    def _is_anchor_row(self, row: dict[str, Any]) -> bool:
        if row.get("type") == "last-prompt" and row.get("lastPrompt") == self.anchor:
            return False
        if json_get_path(row, ("message", "role")) != "user":
            return False
        return content_text(json_get_path(row, ("message", "content"))) == self.anchor

    def _build_segment(
        self,
        path: Path,
        anchor_line: int,
        rows: list[tuple[int, dict[str, Any], str]],
    ) -> ClaudeTraceSegment:
        anchor_row = next(row for line_no, row, _ in rows if line_no == anchor_line)
        session_id = str(anchor_row.get("sessionId") or "")
        tool_names: list[str] = []
        model_evidence: list[str] = []
        model_evidence_count = 0
        assistant_turns = 0
        thinking_blocks = 0
        tool_uses = 0
        tool_results = 0
        prompt_count = 0
        byte_estimate = 0
        end_line = anchor_line

        for line_no, row, raw in rows:
            byte_estimate += len(raw.encode("utf-8", errors="replace"))
            end_line = max(end_line, line_no)
            if row.get("sessionId") and row.get("sessionId") != session_id:
                continue
            model = json_get_path(row, ("message", "model"))
            if model == "claude-fable-5":
                model_evidence_count += 1
                if len(model_evidence) < 12:
                    model_evidence.append(f"line:{line_no}:message.model=claude-fable-5")
            if "Set model to" in content_text(json_get_path(row, ("message", "content"))) and "Fable 5" in content_text(json_get_path(row, ("message", "content"))):
                model_evidence.append(f"line:{line_no}:local_model_switch_fable5")
            if row.get("type") == "assistant":
                assistant_turns += 1
                thinking, uses, names = assistant_content_stats(json_get_path(row, ("message", "content")))
                thinking_blocks += thinking
                tool_uses += uses
                tool_names.extend(names)
            if json_get_path(row, ("message", "role")) == "user":
                prompt_count += 1
                if isinstance(json_get_path(row, ("message", "content")), list):
                    for item in json_get_path(row, ("message", "content")):
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            tool_results += 1

        if model_evidence_count > 12:
            model_evidence.append(f"message.model=claude-fable-5 repeated {model_evidence_count} times")
        unique_tools = sorted(set(tool_names))
        return ClaudeTraceSegment(
            source_file=str(path),
            session_id=session_id,
            start_line=anchor_line,
            end_line=end_line,
            start_timestamp=anchor_row.get("timestamp"),
            cwd=anchor_row.get("cwd"),
            git_branch=anchor_row.get("gitBranch"),
            anchor=self.anchor,
            model_evidence=sorted(set(model_evidence)),
            model_evidence_count=model_evidence_count,
            assistant_turns=assistant_turns,
            thinking_blocks=thinking_blocks,
            tool_uses=tool_uses,
            tool_results=tool_results,
            tool_names=unique_tools,
            prompt_count=prompt_count,
            byte_estimate=byte_estimate,
        )


def build_inventory(config: InventoryConfig) -> dict[str, Any]:
    """Build a source inventory with conservative inclusion states."""

    sources: list[dict[str, Any]] = []
    for group, paths in [
        ("qz", config.qz_sources),
        ("seed_48k", config.seed_48k),
        ("claude", config.claude_projects),
        ("codex", config.codex_dirs),
    ]:
        for path in paths:
            state, reason = classify_source(path)
            exists = path.exists()
            sources.append(
                {
                    "group": group,
                    "path": str(path),
                    "exists": exists,
                    "kind": source_kind(path) if exists else "missing",
                    "size_bytes": file_size(path) if exists else None,
                    "mtime_utc": utc_iso_from_stat(path) if exists else None,
                    "state": "seed_48k" if group == "seed_48k" and exists else state,
                    "reason": "prepared_48k_positive_calibration_seed" if group == "seed_48k" and exists else reason,
                }
            )

    for dataset_id in config.hf_candidates:
        sources.append(
            {
                "group": "huggingface",
                "path": dataset_id,
                "exists": None,
                "kind": "hf_dataset",
                "size_bytes": None,
                "mtime_utc": None,
                "state": "candidate_priority1",
                "reason": "known_hf_fable5_candidate_requires_review",
            }
        )
    for dataset_id in config.hf_quarantine:
        sources.append(
            {
                "group": "huggingface",
                "path": dataset_id,
                "exists": None,
                "kind": "hf_dataset",
                "size_bytes": None,
                "mtime_utc": None,
                "state": "quarantine",
                "reason": "noisy_or_synthetic_fable5_label",
            }
        )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "privacy": "private_training_only",
            "default": "quarantine_until_provenance_and_quality_pass",
            "calibration_seed": "lfm25_fable_*_48k prepared labels",
        },
        "sources": sorted(sources, key=lambda row: (row["group"], row["path"])),
    }


def load_48k_seed_hashes(paths: Iterable[Path], limit_per_file: int | None = None) -> dict[str, Any]:
    """Load prepared 48K sha16/window metadata without reading token arrays into memory."""

    hashes: set[str] = set()
    files: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            files.append({"path": str(path), "exists": False, "rows": 0, "hashes": 0})
            continue
        rows = 0
        file_hashes = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows += 1
                match = SHA16_RE.search(line)
                if match:
                    hashes.add(match.group(1))
                    file_hashes += 1
                if limit_per_file and rows >= limit_per_file:
                    break
        files.append({"path": str(path), "exists": True, "rows": rows, "hashes": file_hashes})
    return {
        "unique_sha16": len(hashes),
        "files": files,
        "sha16": sorted(hashes),
    }


def summarize_prepared_48k(paths: Iterable[Path], limit_per_file: int | None = None) -> dict[str, Any]:
    """Summarize prepared-label rows using lightweight field extraction."""

    seq_re = re.compile(r'"seq_len"\s*:\s*(\d+)')
    sup_re = re.compile(r'"supervised_tokens"\s*:\s*(\d+)')
    row_id_re = re.compile(r'"row_id"\s*:\s*"([^"]+)"')
    files: list[dict[str, Any]] = []
    total_rows = 0
    total_tokens = 0
    total_supervised = 0
    multi_window_sources: set[str] = set()
    for path in paths:
        if not path.exists():
            files.append({"path": str(path), "exists": False})
            continue
        rows = 0
        max_seq = 0
        full_windows = 0
        file_tokens = 0
        file_supervised = 0
        source_windows: dict[str, int] = {}
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows += 1
                seq = int(seq_re.search(line).group(1)) if seq_re.search(line) else 0
                sup = int(sup_re.search(line).group(1)) if sup_re.search(line) else 0
                row_id_match = row_id_re.search(line)
                if row_id_match:
                    source_id = row_id_match.group(1).rsplit(":w", 1)[0]
                    source_windows[source_id] = source_windows.get(source_id, 0) + 1
                max_seq = max(max_seq, seq)
                if seq >= 49152:
                    full_windows += 1
                file_tokens += seq
                file_supervised += sup
                if limit_per_file and rows >= limit_per_file:
                    break
        for source_id, count in source_windows.items():
            if count > 1:
                multi_window_sources.add(f"{path}:{source_id}")
        total_rows += rows
        total_tokens += file_tokens
        total_supervised += file_supervised
        files.append(
            {
                "path": str(path),
                "exists": True,
                "rows": rows,
                "max_seq_len": max_seq,
                "full_49152_windows": full_windows,
                "tokens": file_tokens,
                "supervised_tokens": file_supervised,
            }
        )
    return {
        "files": files,
        "total_rows": total_rows,
        "tokens": total_tokens,
        "supervised_tokens": total_supervised,
        "multi_window_source_count": len(multi_window_sources),
    }


def summarize_pool(path: Path) -> dict[str, Any]:
    """Summarize a raw JSONL pool by row count, thinking coverage, and source tag."""

    if not path.exists():
        return {"path": str(path), "exists": False}
    rows = 0
    thinking_rows = 0
    sources: dict[str, int] = {}
    for _, row, _ in iter_jsonl(path):
        rows += 1
        source = str(row.get("source") or "unknown")
        sources[source] = sources.get(source, 0) + 1
        if str(row.get("thinking") or "").strip():
            thinking_rows += 1
    return {
        "path": str(path),
        "exists": True,
        "mtime_utc": utc_iso_from_stat(path),
        "size_bytes": file_size(path),
        "rows": rows,
        "thinking_rows": thinking_rows,
        "source_counts": dict(sorted(sources.items())),
    }


def _prepared_paths_by_run(paths: Iterable[Path]) -> dict[str, dict[str, Path]]:
    by_run: dict[str, dict[str, Path]] = {}
    for path in paths:
        name = path.name
        match = re.match(r"(.+_48k)_(train|validation)\.labels\.jsonl$", name)
        if not match:
            continue
        by_run.setdefault(match.group(1), {})[match.group(2)] = path
    return by_run


def _sqlite_opencode_evidence(db_path: Path, term: str, limit: int = 6) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT p.session_id, s.directory, s.title, p.time_created, p.data
            FROM part p JOIN session s ON s.id = p.session_id
            WHERE p.data LIKE ?
            ORDER BY p.time_created
            LIMIT ?
            """,
            (f"%{term}%", limit),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    evidence: list[dict[str, Any]] = []
    for row in rows:
        try:
            data = json.loads(row["data"])
        except (TypeError, json.JSONDecodeError):
            continue
        state = data.get("state") if isinstance(data, dict) else None
        state = state if isinstance(state, dict) else {}
        tool_input = state.get("input") if isinstance(state.get("input"), dict) else {}
        command = str(tool_input.get("command") or tool_input.get("filePath") or data.get("text") or "")
        output = str(state.get("output") or "")
        command_lines = [
            line
            for line in command.splitlines()
            if term in line or "prepare_lfm25_data.py" in line or "homogenize_tools_to_codex.py" in line
        ]
        output_lines = [
            line
            for line in output.splitlines()
            if term in line or line.startswith(("Run:", "Pool", "Unified pool:", "Train:", "Processed", "Output:", "Saved to"))
        ]
        evidence.append(
            {
                "session_id": row["session_id"],
                "directory": row["directory"],
                "title": row["title"],
                "time_created_ms": row["time_created"],
                "tool": data.get("tool") if isinstance(data, dict) else None,
                "status": state.get("status"),
                "command_excerpt": "\n".join(command_lines)[:1600],
                "output_excerpt": "\n".join(output_lines)[-1600:],
            }
        )
    return evidence


def build_48k_lineage(config: InventoryConfig, *, limit_per_file: int | None = None) -> dict[str, Any]:
    """Build provenance for the existing 48K prepared labels and source pools."""

    prepared_summary = summarize_prepared_48k(config.seed_48k, limit_per_file=limit_per_file)
    prepared_files_by_path = {row.get("path"): row for row in prepared_summary.get("files", [])}
    prepared_by_run = _prepared_paths_by_run(config.seed_48k)

    pool_rows: dict[str, Any] = {}
    scan_pools = bool(config.qz_sources)
    for name, spec in POOL_LINEAGE.items():
        path = spec["path"]
        summary = summarize_pool(path) if scan_pools else {"path": str(path), "exists": None, "skipped": True}
        pool_rows[name] = {
            "name": name,
            "path": str(path),
            "input_path": str(spec["input_path"]),
            "command": spec["command"],
            "purpose": spec["purpose"],
            "opencode_time_ms": spec["opencode_time_ms"],
            "summary": summary,
            "opencode_evidence": _sqlite_opencode_evidence(config.opencode_db, path.name, limit=3),
        }

    datasets: list[dict[str, Any]] = []
    for run_name, spec in PREPARED_48K_LINEAGE.items():
        split_paths = prepared_by_run.get(run_name, {})
        train = split_paths.get("train")
        validation = split_paths.get("validation")
        train_summary = prepared_files_by_path.get(str(train)) if train else None
        valid_summary = prepared_files_by_path.get(str(validation)) if validation else None
        datasets.append(
            {
                "run_name": run_name,
                "source_pool": spec["source_pool"],
                "source_mode": spec["source_mode"],
                "command": spec["command"],
                "opencode_time_ms": spec["opencode_time_ms"],
                "notes": spec["notes"],
                "train_path": str(train) if train else None,
                "validation_path": str(validation) if validation else None,
                "train_summary": train_summary,
                "validation_summary": valid_summary,
                "opencode_evidence": _sqlite_opencode_evidence(config.opencode_db, run_name, limit=4),
                "confidence": "confirmed_by_opencode_and_filesystem" if train_summary else "historical_opencode_only",
            }
        )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "opencode_db": str(config.opencode_db),
        "prepared_summary": prepared_summary,
        "pools": pool_rows,
        "datasets": datasets,
        "conclusion": {
            "primary_long_context_seed": "lfm25_fable_merged_48k",
            "primary_source_pool": "pool_merged_codex",
            "primary_source_pool_composition": pool_rows.get("pool_merged_codex", {}).get("summary", {}).get("source_counts"),
            "caution": (
                "The prepared-label rows keep row_id and sha16 but not complete raw source metadata; "
                "lineage is recovered from OpenCode command logs, script defaults at the time, filesystem mtimes, "
                "and row-count agreement."
            ),
        },
    }


def row_signature(row: dict[str, Any]) -> str:
    """Stable weak signature for raw source rows when token hashes are unavailable."""

    parts = []
    for key in ("context", "thinking", "cot", "response", "text", "messages"):
        value = row.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            value = json.dumps(value, sort_keys=True, ensure_ascii=False)
        parts.append(value[:4096])
    return hashlib.sha256("\n".join(parts).encode("utf-8", errors="replace")).hexdigest()[:16]


def build_manifest(config: InventoryConfig, *, seed_limit_per_file: int | None = None) -> dict[str, Any]:
    scanner = ClaudeTraceScanner()
    claude_segments = scanner.scan_project_dirs(config.claude_projects)
    seed = load_48k_seed_hashes(config.seed_48k, limit_per_file=seed_limit_per_file)
    prepared = summarize_prepared_48k(config.seed_48k, limit_per_file=seed_limit_per_file)

    rows = [segment.to_manifest_row() for segment in claude_segments]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_48k": {
            "unique_sha16": seed["unique_sha16"],
            "files": seed["files"],
            "prepared_summary": prepared,
            "limited": seed_limit_per_file is not None,
        },
        "manifest_rows": rows,
        "summary": {
            "claude_fable5_segments": len(rows),
            "high_confidence_segments": sum(1 for row in rows if row["state"] == "candidate_high_confidence"),
            "tool_uses": sum(row["tool_uses"] for row in rows),
            "tool_results": sum(row["tool_results"] for row in rows),
        },
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_markdown_report(
    path: Path,
    inventory: dict[str, Any],
    manifest: dict[str, Any],
    lineage: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = inventory.get("sources", [])
    summary = manifest.get("summary", {})
    seed = manifest.get("seed_48k", {})
    prepared = seed.get("prepared_summary", {})
    states: dict[str, int] = {}
    for row in sources:
        states[row["state"]] = states.get(row["state"], 0) + 1

    lines = [
        "# Fable 5 Dataset Audit",
        "",
        f"Created: `{manifest.get('created_at_utc')}`",
        "",
        "## Policy",
        "",
        "- Private training only.",
        "- 48K prepared labels are the positive calibration anchor.",
        "- Sources are quarantined until provenance, quality, and sample review pass.",
        "",
        "## Inventory",
        "",
    ]
    for state, count in sorted(states.items()):
        lines.append(f"- `{state}`: {count}")
    lines.extend(
        [
            "",
            "## 48K Seed",
            "",
            f"- Unique loaded `sha16`: {seed.get('unique_sha16', 0)}",
            f"- Prepared rows scanned: {prepared.get('total_rows', 0)}",
            f"- Prepared tokens scanned: {prepared.get('tokens', 0)}",
            f"- Prepared supervised tokens scanned: {prepared.get('supervised_tokens', 0)}",
            f"- Multi-window source count: {prepared.get('multi_window_source_count', 0)}",
            f"- Limited scan: {seed.get('limited', False)}",
            "",
            "## Verified Claude Fable 5",
            "",
            f"- Segments: {summary.get('claude_fable5_segments', 0)}",
            f"- Tool uses: {summary.get('tool_uses', 0)}",
            f"- Tool results: {summary.get('tool_results', 0)}",
            "",
        ]
    )
    if lineage:
        conclusion = lineage.get("conclusion", {})
        primary_pool = conclusion.get("primary_source_pool")
        pool_summary = (lineage.get("pools", {}).get(primary_pool, {}) or {}).get("summary", {})
        lines.extend(
            [
                "## 48K Lineage",
                "",
                f"- Primary long-context seed: `{conclusion.get('primary_long_context_seed')}`",
                f"- Primary source pool: `{primary_pool}`",
                f"- Primary pool rows: {pool_summary.get('rows', 0)}",
                f"- Primary pool thinking rows: {pool_summary.get('thinking_rows', 0)}",
                f"- Primary pool source counts: `{pool_summary.get('source_counts', {})}`",
                "",
                "| Dataset | Source Pool | Train Rows | Max Seq | Full 48K Windows | Evidence |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for dataset in lineage.get("datasets", []):
            train = dataset.get("train_summary") or {}
            evidence_count = len(dataset.get("opencode_evidence") or [])
            lines.append(
                "| "
                f"`{dataset.get('run_name')}` | "
                f"`{dataset.get('source_pool')}` | "
                f"{train.get('rows', 0)} | "
                f"{train.get('max_seq_len', 0)} | "
                f"{train.get('full_49152_windows', 0)} | "
                f"{evidence_count} OpenCode match(es) |"
            )
        lines.extend(
            [
                "",
                "Lineage is recovered from OpenCode command records, source-pool mtimes, "
                "prepared-label stats, and row-count agreement. Prepared label rows keep "
                "`row_id` and `sha16`, but not complete raw source metadata.",
                "",
            ]
        )
    lines.extend(
        [
            "## Next Gate",
            "",
            "Inspect review samples and use the recovered 48K lineage as the quality baseline for any expanded corpus.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def sample_claude_segment(segment: ClaudeTraceSegment, max_lines: int = 80) -> list[dict[str, Any]]:
    path = Path(segment.source_file)
    rows = []
    for line_no, row, _ in iter_jsonl(path):
        if line_no < segment.start_line:
            continue
        if line_no > segment.end_line or len(rows) >= max_lines:
            break
        rows.append({"line": line_no, "row": row})
    return rows


def export_claude_segment(segment: ClaudeTraceSegment) -> dict[str, Any]:
    """Export a complete verified Claude segment with event structure preserved."""

    path = Path(segment.source_file)
    events = []
    for line_no, row, _ in iter_jsonl(path):
        if line_no < segment.start_line:
            continue
        if line_no > segment.end_line:
            break
        events.append({"line": line_no, "row": row})
    return {
        "id": segment.to_manifest_row()["id"],
        "source_family": "claude_local",
        "source_path": segment.source_file,
        "session_id": segment.session_id,
        "start_line": segment.start_line,
        "end_line": segment.end_line,
        "start_timestamp": segment.start_timestamp,
        "cwd": segment.cwd,
        "git_branch": segment.git_branch,
        "model_evidence": segment.model_evidence,
        "model_evidence_count": segment.model_evidence_count,
        "tool_names": segment.tool_names,
        "tool_uses": segment.tool_uses,
        "tool_results": segment.tool_results,
        "state": "candidate_high_confidence",
        "events": events,
    }


def build_candidate_traces(config: InventoryConfig) -> list[dict[str, Any]]:
    """Export complete high-confidence candidate traces for review/build."""

    scanner = ClaudeTraceScanner()
    return [export_claude_segment(segment) for segment in scanner.scan_project_dirs(config.claude_projects)]


def build_review_samples(manifest_rows: Iterable[dict[str, Any]], max_segments: int = 5) -> list[dict[str, Any]]:
    samples = []
    for row in list(manifest_rows)[:max_segments]:
        segment = ClaudeTraceSegment(
            source_file=row["source_path"],
            session_id=row["session_id"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            start_timestamp=row.get("start_timestamp"),
            cwd=row.get("cwd"),
            git_branch=row.get("git_branch"),
            anchor=VOXTALK_FABLE5_ANCHOR,
            model_evidence=list(row.get("model_evidence") or []),
            model_evidence_count=int(row.get("model_evidence_count") or 0),
        )
        samples.append({"manifest_id": row["id"], "events": sample_claude_segment(segment)})
    return samples


def env_config() -> InventoryConfig:
    """Build config with optional path overrides for tests and ad-hoc runs."""

    def split_paths(name: str, default: tuple[Path, ...]) -> tuple[Path, ...]:
        value = os.environ.get(name)
        if not value:
            return default
        return tuple(Path(part) for part in value.split(os.pathsep) if part)

    return InventoryConfig(
        qz_sources=split_paths("STRATUM_FABLE5_QZ_SOURCES", tuple(DEFAULT_QZ_SOURCES)),
        seed_48k=split_paths("STRATUM_FABLE5_48K_SEEDS", tuple(DEFAULT_48K_DATASETS)),
        claude_projects=split_paths("STRATUM_FABLE5_CLAUDE_PROJECTS", tuple(DEFAULT_CLAUDE_PROJECTS)),
    )
