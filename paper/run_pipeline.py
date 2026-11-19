"""Run OCR/gold normalization, segmentation, alignment, metrics, and audit outputs."""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .alignment_core import SIMILARITY_BACKEND
    from .structure_alignment import write_reports
    from .evaluation_metrics import evaluate_pipeline
    from .gold_normalization import convert_tex
    from .exam_segmentation import parse_exam
    from .content_alignment import run_token_alignment, write_jsonl
    from .unit_extraction import document_units
    from .error_taxonomy import build_taxonomy_report
except ImportError:
    from alignment_core import SIMILARITY_BACKEND
    from structure_alignment import write_reports
    from evaluation_metrics import evaluate_pipeline
    from gold_normalization import convert_tex
    from exam_segmentation import parse_exam
    from content_alignment import run_token_alignment, write_jsonl
    from unit_extraction import document_units
    from error_taxonomy import build_taxonomy_report

from structural_fixes.normalization import normalize_markdown


DEFAULT_OCR = PROJECT_ROOT / "data/input/exam.md"
DEFAULT_GOLD = PROJECT_ROOT / "Gaokao-Math-Problems-Compilation/content/2024/national_paper_a_liberal.tex"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper/results"


def write_text(path: Path, text: str):
    """Write UTF-8 text, creating the destination directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value):
    """Write formatted UTF-8 JSON, creating the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def segment(markdown: str, path: Path):
    """Parse Markdown, save its exam structure, and return that structure."""
    exam = parse_exam(markdown)
    data = asdict(exam)
    write_json(path, data)
    return data


def compile_markdown(markdown: str, label: str, output_dir: Path):
    """Measure Markdown parser/renderer/TeX compile success; leave artifacts for inspection."""
    result = {
        "label": label,
        "parse_success": False,
        "render_success": False,
        "compile_attempted": False,
        "compile_success": False,
        "engine": None,
        "error": None,
    }
    try:
        from json_interface.markdown_parser import MarkdownExamParser
        from latex_converter.latex_renderer import LatexRenderer, RenderOptions

        # Compile checks measure document conversion, not whether referenced
        # image assets happen to be installed on this machine.
        compile_markdown = re.sub(
            r"\\includegraphics(?:\[[^\]]*\])?\{[^{}]*\}", "[FIGURE]", markdown
        )
        exam = MarkdownExamParser().parse(compile_markdown, f"{label}.md")
        result["parse_success"] = True
        tex = LatexRenderer(RenderOptions(figures="placeholder")).render(exam.to_dict())
        result["render_success"] = True
    except Exception as exc:  # A compile baseline records failures instead of aborting evaluation.
        result["error"] = f"Markdown parse/render failed: {type(exc).__name__}: {exc}"
        return result

    check_dir = output_dir / "compile_checks" / label
    check_dir.mkdir(parents=True, exist_ok=True)
    tex_path = check_dir / f"{label}.tex"
    write_text(tex_path, tex)
    engine = shutil.which("xelatex") or shutil.which("lualatex")
    result["engine"] = Path(engine).name if engine else None
    if not engine:
        result["error"] = "XeLaTeX/LuaLaTeX not installed"
        return result

    result["compile_attempted"] = True
    try:
        completed = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
             f"-output-directory={check_dir.resolve()}", str(tex_path.resolve())],
            cwd=check_dir, capture_output=True, text=True, timeout=180, check=False,
        )
        pdf_path = check_dir / f"{label}.pdf"
        result["compile_success"] = completed.returncode == 0 and pdf_path.is_file() and pdf_path.stat().st_size > 0
        if not result["compile_success"]:
            log_path = check_dir / f"{label}.log"
            tex_errors = []
            if log_path.is_file():
                tex_errors = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("!")][:5]
            result["error"] = " | ".join(tex_errors) or (completed.stderr or completed.stdout)[-1200:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = f"TeX compile failed: {type(exc).__name__}: {exc}"
    return result


def _all_unit_lookup(document):
    """Index question, sub-question, equation, and content units by their IDs."""
    _, groups = document_units(document)
    lookup = {}
    for group in groups.values():
        for unit in [group["question"], *group["sub_questions"], *group["display_equations"], *group["content_units"]]:
            lookup[unit["unit_id"]] = unit
    return lookup


def _audit_sample(rows, raw_ocr, raw_gold, size=30, seed=31, previous_path=None):
    """Choose a balanced review sample and carry forward labels from the last run."""
    previous = {}
    if previous_path and previous_path.is_file():
        for line in previous_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (item.get("ocr_unit_id"), item.get("gold_unit_id"), item.get("operation"))
            previous[key] = item
    eligible = list(rows)
    rng = random.Random(seed)
    by_type = {}
    for row in eligible:
        by_type.setdefault(row["unit_type"], []).append(row)
    chosen = []
    types = sorted(by_type)
    # Round-robin strata prevent the sample being dominated by question stems.
    while len(chosen) < min(size, len(eligible)):
        progressed = False
        for kind in types:
            if by_type[kind] and len(chosen) < min(size, len(eligible)):
                index = rng.randrange(len(by_type[kind]))
                chosen.append(by_type[kind].pop(index))
                progressed = True
        if not progressed:
            break
    raw_ocr_units, raw_gold_units = _all_unit_lookup(raw_ocr), _all_unit_lookup(raw_gold)
    result = []
    for row in chosen:
        item = {
            "unit_type": row["unit_type"],
            "ocr_unit_id": row.get("ocr_unit_id"),
            "gold_unit_id": row.get("gold_unit_id"),
            "match_method": row.get("match_method"),
            "comparison_method": row.get("comparison_method"),
            "used_parent_fallback": row.get("used_parent_fallback", False),
            "operation": row.get("operation"),
            "count_in_metrics": row.get("count_in_metrics", True),
            "ocr_raw": (raw_ocr_units.get(row.get("ocr_unit_id")) or {}).get("text"),
            "ocr_normalized": row.get("ocr_text"),
            "gold_raw_tex_to_md": (raw_gold_units.get(row.get("gold_unit_id")) or {}).get("text"),
            "gold_normalized_markdown": row.get("gold_text"),
            "token_errors": row.get("labeled_errors", []),
            "alignment_review_label": None,
            "conversion_noise": None,
            "human_review_label": None,
            "review_notes": None,
        }
        key = (item["ocr_unit_id"], item["gold_unit_id"], item["operation"])
        old = previous.get(key, {})
        for field in ("alignment_review_label", "conversion_noise", "human_review_label", "review_notes"):
            if old.get(field) is not None:
                item[field] = old[field]
        result.append(item)
    return result


def run_pipeline(ocr_path: Path, gold_path: Path, output_dir: Path, *,
                 min_similarity=0.35, low_error_threshold=0.05,
                 audit_size=30, skip_compile=False, ocr_structured_json=None):
    """Run the full evaluation from external OCR Markdown to final reports.

    The stages are gold conversion, shared normalization, segmentation,
    structure checks, raw and normalized alignment, then metrics and review
    files. The OCR path is read directly; this pipeline does not run OCR.
    """
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1")
    if not 0.0 <= low_error_threshold <= 1.0:
        raise ValueError("low_error_threshold must be between 0 and 1")
    if audit_size < 0:
        raise ValueError("audit_size must be nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_dir = output_dir / "documents"
    segmentations_dir = output_dir / "segmentations"
    reports_dir = output_dir / "reports"
    review_dir = output_dir / "review"
    external_ocr_path = Path(ocr_path)
    ocr_raw = external_ocr_path.read_text(encoding="utf-8")
    if gold_path.suffix.lower() == ".tex":
        gold_raw = convert_tex(gold_path.read_text(encoding="utf-8"), normalize=False)
        gold_format = "tex"
    else:
        gold_raw = gold_path.read_text(encoding="utf-8")
        gold_format = "markdown"
    if not ocr_raw.strip() or not gold_raw.strip():
        raise ValueError("OCR and gold documents must both contain text")

    # Deterministic shared style normalization only; no LLM correction is run.
    ocr_normalized = normalize_markdown(ocr_raw)
    gold_normalized = normalize_markdown(gold_raw)
    write_text(documents_dir / "ocr" / "raw.md", ocr_raw)
    write_text(documents_dir / "reference" / "raw.md", gold_raw)
    write_text(documents_dir / "ocr" / "normalized.md", ocr_normalized)
    write_text(documents_dir / "reference" / "normalized.md", gold_normalized)

    raw_ocr_segmentation = segment(ocr_raw, segmentations_dir / "ocr" / "raw.json")
    raw_gold_segmentation = segment(gold_raw, segmentations_dir / "reference" / "raw.json")
    norm_ocr_segmentation_path = segmentations_dir / "ocr" / "normalized.json"
    norm_gold_segmentation_path = segmentations_dir / "reference" / "normalized.json"
    norm_ocr_doc = segment(ocr_normalized, norm_ocr_segmentation_path)
    norm_gold_doc = segment(gold_normalized, norm_gold_segmentation_path)

    # Structure and question similarity report uses the normalized pair.
    _, structure_report = write_reports(
        norm_ocr_segmentation_path, norm_gold_segmentation_path,
        reports_dir / "alignment_results.json", reports_dir / "structure_results.json",
        min_similarity=min_similarity,
    )
    raw_token_report = run_token_alignment(
        raw_ocr_segmentation, raw_gold_segmentation, min_similarity=min_similarity
    )
    raw_rows = raw_token_report["unit_alignments"]
    confidence_data = (
        json.loads(Path(ocr_structured_json).read_text(encoding="utf-8"))
        if ocr_structured_json is not None else None
    )
    taxonomy_report = build_taxonomy_report(
        raw_rows, structure_report, raw_ocr_segmentation, raw_gold_segmentation,
        external_ocr_path.name, confidence_data=confidence_data,
    )
    write_json(reports_dir / "error_taxonomy_results.json", taxonomy_report)
    token_report = run_token_alignment(norm_ocr_doc, norm_gold_doc, min_similarity=min_similarity)
    normalized_rows = token_report["unit_alignments"]
    # Per-unit normalized rows get raw counterparts for the baseline/outcome calculations.
    raw_lookup_ocr = _all_unit_lookup(raw_ocr_segmentation)
    raw_lookup_gold = _all_unit_lookup(raw_gold_segmentation)
    for row in normalized_rows:
        row["raw_ocr_text"] = (raw_lookup_ocr.get(row.get("ocr_unit_id")) or {}).get("text")
        row["raw_gold_text"] = (raw_lookup_gold.get(row.get("gold_unit_id")) or {}).get("text")
    write_json(reports_dir / "token_alignment_results.json", {
        key: value for key, value in token_report.items() if key != "unit_alignments"
    })
    write_jsonl(review_dir / "alignment_pairs.jsonl", normalized_rows)
    audit_path = review_dir / "audit_sample.jsonl"
    audit_rows = _audit_sample(
        normalized_rows, raw_ocr_segmentation, raw_gold_segmentation, size=audit_size,
        previous_path=audit_path,
    )
    write_jsonl(audit_path, audit_rows)

    metrics = evaluate_pipeline(
        raw_rows, normalized_rows,
        raw_ocr_segmentation, raw_gold_segmentation, norm_ocr_doc, norm_gold_doc,
        ocr_raw, gold_raw, ocr_normalized, gold_normalized,
        low_error_threshold=low_error_threshold,
        audit_rows=audit_rows, document_name=external_ocr_path.name,
        min_similarity=min_similarity,
    )
    metrics["structure_error_counts"] = structure_report["error_counts"]
    metrics["structure_error_counts_by_document"] = {
        external_ocr_path.name: structure_report["error_counts"]
    }
    metrics["error_taxonomy_report"] = "reports/error_taxonomy_results.json"
    metrics["top_correction_experiment_candidates"] = taxonomy_report["aggregate"]["top_correction_experiment_candidates"]
    confidence_report = taxonomy_report["documents"][external_ocr_path.name]["confidence_analysis"]
    metrics["error_distribution_raw"]["by_document"][external_ocr_path.name]["confidence"] = confidence_report
    if not skip_compile:
        compile_results = [
            compile_markdown(ocr_raw, "raw_ocr", output_dir),
            compile_markdown(ocr_normalized, "normalized_ocr", output_dir),
            compile_markdown(gold_normalized, "normalized_gold", output_dir),
        ]
        baseline_results = [item for item in compile_results if item["label"] == "raw_ocr"]
        metrics["markdown_to_latex_compile"] = {
            "raw_baseline_successes": sum(item["compile_success"] for item in baseline_results),
            "raw_baseline_attempts": len(baseline_results),
            "raw_baseline_compile_rate": round(sum(item["compile_success"] for item in baseline_results) / len(baseline_results), 4) if baseline_results else None,
            "checks": compile_results,
        }
    else:
        metrics["markdown_to_latex_compile"] = {"skipped": True, "reason": "--skip-compile"}
    metrics["inputs"] = {
        "ocr": str(external_ocr_path.resolve()),
        "gold": str(gold_path.resolve()),
        "gold_format": gold_format, "output_dir": str(output_dir.resolve()),
    }
    metrics["audit_sample"] = {
        "requested": audit_size,
        "written": len(audit_rows),
        "file": "review/audit_sample.jsonl",
    }
    write_json(reports_dir / "metrics_results.json", metrics)

    summary = {
        "ocr_questions": sum(len(section.get("questions", [])) for section in norm_ocr_doc.get("sections", [])),
        "gold_questions": sum(len(section.get("questions", [])) for section in norm_gold_doc.get("sections", [])),
        "similarity_backend": SIMILARITY_BACKEND,
        "outputs": [
            "reports/alignment_results.json",
            "reports/structure_results.json",
            "reports/error_taxonomy_results.json",
            "reports/token_alignment_results.json",
            "reports/metrics_results.json",
            "review/alignment_pairs.jsonl",
            "review/audit_sample.jsonl",
            "reports/run_summary.json",
        ],
        "alignment_methods": token_report["match_method_counts"],
        "comparison_methods": token_report["comparison_method_counts"],
        "parent_fallback_pair_count": sum(bool(row.get("used_parent_fallback")) for row in normalized_rows),
        "excluded_units": token_report["excluded"],
        "syntax_semantic_errors": token_report["error_counts"],
    }
    write_json(reports_dir / "run_summary.json", summary)
    return summary


def main(ocr_path=DEFAULT_OCR, gold_path=DEFAULT_GOLD, output_dir=DEFAULT_OUTPUT, *,
         ocr_structured_json=None, min_similarity=0.35,
         low_error_threshold=0.05, audit_size=30,
         skip_compile=False):
    """Run with editable input-path arguments; no terminal arguments are read."""
    summary = run_pipeline(
        Path(ocr_path), Path(gold_path), Path(output_dir),
        min_similarity=min_similarity,
        low_error_threshold=low_error_threshold,
        audit_size=audit_size,
        skip_compile=skip_compile,
        ocr_structured_json=ocr_structured_json,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
