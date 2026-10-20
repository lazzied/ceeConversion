"""Helpers for splitting corrected exam Markdown into sections."""

import re
from pathlib import Path


def split_exam(
    source: str | Path = Path("data/input/exam.md"),
    names: tuple[str, ...] | None = None,
    *,
    write: bool = False,
) -> dict[str, str]:
    """Split exam Markdown at level-one or level-two headings.

    ``source`` may be Markdown text or a path. File output is opt-in so merely
    importing this module never reads or writes project files.
    """
    if isinstance(source, Path):
        markdown = source.read_text(encoding="utf-8")
    elif (
        len(source) < 240
        and "\n" not in source
        and not source.lstrip().startswith("#")
        and Path(source).is_file()
    ):
        markdown = Path(source).read_text(encoding="utf-8")
    else:
        markdown = source
    parts = [part.strip() for part in re.split(r"(?m)^(?=#{1,2} )", markdown) if part.strip()]
    if names is None:
        names = ("title", *(f"section{index}" for index in range(1, len(parts))))
    if len(parts) != len(names):
        raise ValueError(f"Expected {len(names)} sections, found {len(parts)}")
    sections = dict(zip(names, parts))
    if write:
        for name, part in sections.items():
            Path(f"{name}.txt").write_text(part + "\n", encoding="utf-8")
    return sections


def strip_fences(text: str) -> str:
    """Remove a single Markdown code fence around an LLM response."""
    lines = text.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text.strip()
