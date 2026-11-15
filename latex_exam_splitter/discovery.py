"""Find and classify combined exam/solution files in a corpus tree."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from .models import SplitError, SplitPlan
from .parser import analyze_document


LOGGER = logging.getLogger("latex_exam_splitter")
_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".git", ".hg", ".svn", "__pycache__", "separated", "split-output"}
)


def discover_tex_files(root: Path, exclude_paths: Sequence[Path] = ()) -> list[Path]:
    """Recursively list TeX sources, skipping VCS and requested output trees."""

    root = Path(root).resolve()
    excluded_paths = tuple(Path(path).resolve() for path in exclude_paths)
    return sorted(
        path
        for path in root.rglob("*.tex")
        if not any(part in _EXCLUDED_DIRECTORY_NAMES for part in path.relative_to(root).parts)
        and not any(path.is_relative_to(excluded_path) for excluded_path in excluded_paths)
    )


def scan_corpus(
    root: Path, exclude_paths: Sequence[Path] = ()
) -> tuple[list[Path], list[tuple[Path, SplitPlan]]]:
    """Return all readable TeX files and those containing both document parts."""

    files = discover_tex_files(root, exclude_paths=exclude_paths)
    combined: list[tuple[Path, SplitPlan]] = []
    for source in files:
        try:
            plan = analyze_document(source, corpus_root=root)
        except (OSError, UnicodeError, SplitError) as error:
            LOGGER.warning("Could not classify %s: %s", source, error)
            continue
        if plan is not None:
            combined.append((source, plan))
    return files, combined
