"""Diagnostic output helpers.

Provides a module-level verbose flag so model adapters and build code can
emit structured diagnostics to stderr without threading a parameter through
every function signature.

Usage:
    from stratum.output import set_verbose, vprint, vwrite

    set_verbose(True)          # call once from train.py when --verbose is set
    vprint({"event": "..."})   # JSON to stderr, no-op when not verbose
    vwrite("Patched 6 layers") # plain text to stderr, no-op when not verbose
"""

from __future__ import annotations

import json
import sys

_verbose: bool = False


def set_verbose(enabled: bool) -> None:
    """Enable or disable verbose diagnostic output."""
    global _verbose
    _verbose = enabled


def vprint(d: dict) -> None:
    """Emit a structured diagnostic as JSON to stderr when verbose."""
    if _verbose:
        print(json.dumps(d), file=sys.stderr, flush=True)


def vwrite(msg: str) -> None:
    """Emit a human-readable diagnostic line to stderr when verbose."""
    if _verbose:
        print(msg, file=sys.stderr, flush=True)
