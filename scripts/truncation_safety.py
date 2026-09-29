"""Shared cleanup primitive for failed truncation-safe generations."""
from __future__ import annotations

from pathlib import Path


def remove_failed_output(output_path: Path) -> None:
    """Remove artifacts that could make a failed run look scoreable."""
    output_path.unlink(missing_ok=True)
    output_path.with_suffix(".meta.json").unlink(missing_ok=True)
