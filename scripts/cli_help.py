"""Shared CLI help helpers for pipeline entry-point scripts."""

from __future__ import annotations

import sys


def normalize_help_argv(argv: list[str] | None = None) -> list[str]:
    """Map `-help` / `help` to `--help` so either form works with argparse."""
    argv = list(argv if argv is not None else sys.argv)
    if len(argv) > 1 and argv[1] in ("-help", "help"):
        argv[1] = "--help"
    return argv
