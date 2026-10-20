"""Command-line entry point for Markdown normalization and correction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from structural_fixes.client import process
from structural_fixes.normalization import normalize_markdown
from structural_fixes.prompts import metadata_prompt, numbers_prompt
from structural_fixes.utils import split_exam

log = logging.getLogger(__name__)
METADATA_CONTEXT_CHAR_LIMIT = 2200
METADATA_KEYS = (
    "title", "subtitle", "year", "subject", "paper", "header_left", "header_right"
)


def fix_markdown(markdown: str, *, correction_client: object | None = None) -> str:
    """Normalize an exam and correct its numbered sections in document order."""
    normalized = normalize_markdown(markdown)
    sections = split_exam(normalized)
    corrected: list[str] = []
    for section_name, section_text in sections.items():
        if section_name == "title":
            corrected.append(section_text)
            continue
        log.info("Correcting %s", section_name)
        corrected.append(process(numbers_prompt, section_text, client=correction_client))
    result = "\n\n".join(corrected).strip()
    if not result:
        raise ValueError("Structural correction produced empty Markdown")
    return result + "\n"


def generate_exam_metadata(markdown: str, *, metadata_client: object | None = None) -> dict[str, Any]:
    """Ask the configured LLM for schema metadata and validate its YAML reply."""
    if not markdown.strip():
        raise ValueError("Cannot generate metadata from empty Markdown")
    headings = "\n".join(line for line in markdown.splitlines() if line.lstrip().startswith("#"))
    opening_content = markdown[:METADATA_CONTEXT_CHAR_LIMIT]
    context = f"Headings:\n{headings}\n\nOpening content:\n{opening_content}"
    response = process(metadata_prompt, context, client=metadata_client)
    try:
        parsed = yaml.safe_load(response)
    except yaml.YAMLError as exc:
        raise ValueError(f"Metadata model returned invalid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Metadata model must return a YAML mapping")

    unknown_keys = set(parsed) - set(METADATA_KEYS)
    if unknown_keys:
        raise ValueError(f"Metadata model returned unsupported keys: {sorted(unknown_keys)}")

    metadata: dict[str, Any] = {key: parsed.get(key) for key in METADATA_KEYS}
    for key, value in metadata.items():
        if key == "year":
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError("Metadata year must be an integer or null")
        elif value is not None and not isinstance(value, str):
            raise ValueError(f"Metadata {key} must be a string or null")
        elif isinstance(value, str):
            metadata[key] = value.strip() or None
    if metadata["title"] is None:
        raise ValueError("Metadata model did not provide an exam title")
    return metadata


def main() -> int:
    """Normalize and correct the configured input Markdown file."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("input", nargs="?", type=Path, default=Path("data/input/exam.md"))
    argument_parser.add_argument("-o", "--output", type=Path, default=Path("data/output/exam.md"))
    argument_parser.add_argument("--normalized", type=Path, default=Path("data/output/exam_normalized.md"))
    args = argument_parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        source = args.input.read_text(encoding="utf-8")
        normalized = normalize_markdown(source)
        args.normalized.parent.mkdir(parents=True, exist_ok=True)
        args.normalized.write_text(normalized, encoding="utf-8")
        corrected = fix_markdown(source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(corrected, encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("Structural correction failed: %s", exc)
        return 1
    log.info("Wrote corrected Markdown to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
