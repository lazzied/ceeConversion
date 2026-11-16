"""Freeze OCR Markdown produced by an external OCR system."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKDOWN = PROJECT_ROOT / "data/input/exam.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper/results/ocr_baseline"


def _load_json(path, label):
    if path is None:
        return None
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {path}: {exc}") from exc


def _copy_preserving_source(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return digest


def ingest_ocr_baseline(markdown_path=DEFAULT_MARKDOWN, output_dir=DEFAULT_OUTPUT, *,
                        structured_json=None, metadata_json=None, log_path=None):
    """Save an external OCR transcript and its optional provenance unchanged.

    This function deliberately does not import or run any OCR engine. Metadata
    may include engine/version, settings, elapsed time, and page information.
    Structured JSON is copied verbatim so page numbers and boxes survive.
    """
    markdown_path, output_dir = Path(markdown_path), Path(output_dir)
    if not markdown_path.is_file():
        raise FileNotFoundError(f"External OCR Markdown does not exist: {markdown_path}")
    if not markdown_path.read_bytes().strip():
        raise ValueError(f"External OCR Markdown is empty: {markdown_path}")
    if structured_json is not None and not Path(structured_json).is_file():
        raise FileNotFoundError(f"Structured OCR JSON does not exist: {structured_json}")
    if metadata_json is not None and not Path(metadata_json).is_file():
        raise FileNotFoundError(f"OCR run metadata does not exist: {metadata_json}")
    if log_path is not None and not Path(log_path).is_file():
        raise FileNotFoundError(f"OCR processing log does not exist: {log_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_out = output_dir / "ocr.md"
    markdown_sha256 = _copy_preserving_source(markdown_path, markdown_out)

    structured_out = None
    structured_sha256 = None
    if structured_json is not None:
        _load_json(structured_json, "Structured OCR output")
        structured_out = output_dir / "structured_ocr.json"
        structured_sha256 = _copy_preserving_source(structured_json, structured_out)

    metadata = _load_json(metadata_json, "OCR run metadata") if metadata_json else {}
    metadata = metadata or {}
    if not isinstance(metadata, dict):
        raise ValueError("OCR run metadata must be a JSON object")
    metadata_out = None
    metadata_sha256 = None
    if metadata_json is not None:
        metadata_out = output_dir / "run_metadata.json"
        metadata_sha256 = _copy_preserving_source(metadata_json, metadata_out)

    log_out = None
    log_sha256 = None
    if log_path is not None:
        log_out = output_dir / "ocr_processing.log"
        log_sha256 = _copy_preserving_source(log_path, log_out)

    timing = metadata.get("timing_seconds", {})
    processing_seconds = metadata.get("processing_seconds", metadata.get("elapsed_seconds"))
    if processing_seconds is None and isinstance(timing, dict):
        processing_seconds = timing.get("total")
    manifest = {
        "source": "externally_generated_ocr",
        "ocr_engine": metadata.get("engine", metadata.get("ocr_engine")),
        "ocr_version": metadata.get("version", metadata.get("ocr_version")),
        "configuration": metadata.get("configuration", metadata.get("config")),
        "processing_seconds": processing_seconds,
        "source_markdown": str(markdown_path.resolve()),
        "markdown_file": markdown_out.name,
        "markdown_sha256": markdown_sha256,
        "structured_json_file": structured_out.name if structured_out else None,
        "structured_json_sha256": structured_sha256,
        "metadata_file": metadata_out.name if metadata_out else None,
        "metadata_sha256": metadata_sha256,
        "log_file": log_out.name if log_out else None,
        "log_sha256": log_sha256,
        "page_metadata": "Preserved verbatim in structured_ocr.json when supplied; Markdown is copied without modification.",
    }
    manifest_path = output_dir / "baseline_manifest.json"
    manifest["manifest_file"] = manifest_path.name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(markdown_path=DEFAULT_MARKDOWN, output_dir=DEFAULT_OUTPUT, *,
         structured_json=None, metadata_json=None, log_path=None):
    """Freeze external OCR files from editable function arguments."""
    manifest = ingest_ocr_baseline(
        markdown_path, output_dir,
        structured_json=structured_json,
        metadata_json=metadata_json,
        log_path=log_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
