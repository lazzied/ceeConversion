"""Unit alignment plus syntax/semantic token error labels."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

try:
    from .alignment_core import SIMILARITY_BACKEND, align_questions
    from .unit_matching import align_documents
    from .unit_extraction import document_units
except ImportError:
    from alignment_core import SIMILARITY_BACKEND, align_questions
    from unit_matching import align_documents
    from unit_extraction import document_units


LATEX_SYNTAX_TOKEN = re.compile(r"^\\[A-Za-z]+$|^[{}_^$]$|^\\[\[\]]$")


def classify_token_error(error, unit_type):
    """Rule-based syntax/semantic label for one token-level edit."""
    token_text = " ".join(str(error.get(key) or "") for key in ("gold", "ocr"))
    if unit_type in {"figure", "table", "table_cell"}:
        return "semantic" if unit_type in {"figure", "table_cell"} else "syntactic"
    if error.get("category") == "math_syntax" or any(
        LATEX_SYNTAX_TOKEN.match(token) for token in token_text.split()
    ):
        return "syntactic"
    if error.get("category") == "math":
        return "semantic"
    return "semantic"


def label_unit_errors(rows):
    labeled = []
    counts = Counter()
    for row in rows:
        row_errors = []
        for error in row.get("token_errors", []):
            category = classify_token_error(error, row["unit_type"])
            labeled_error = {
                **error,
                "error_class": category,
                "ocr_unit_id": row.get("ocr_unit_id"),
                "gold_unit_id": row.get("gold_unit_id"),
                "match_method": row.get("match_method"),
                "used_parent_fallback": row.get("used_parent_fallback", False),
                "excluded": row.get("excluded", False),
            }
            row_errors.append(labeled_error)
            labeled.append(labeled_error)
            if row.get("count_in_metrics", True) and row.get("operation") == "match":
                counts[category] += 1
        row["labeled_errors"] = row_errors
    return labeled, dict(sorted(counts.items()))


def excluded_fraction(rows):
    included = [row for row in rows if row.get("count_in_metrics", True)]
    denominator = sum(bool(row.get("ocr_unit_id")) + bool(row.get("gold_unit_id")) for row in included)
    excluded = sum(
        (bool(row.get("ocr_unit_id")) + bool(row.get("gold_unit_id")))
        for row in included if row.get("excluded")
    )
    return {
        "excluded_unit_count": excluded,
        "total_side_unit_count": denominator,
        "fraction": round(excluded / denominator, 6) if denominator else 0.0,
    }


def question_alignment_rows(ocr_document, gold_document, *, min_similarity=0.35):
    ocr_questions, _ = document_units(ocr_document)
    gold_questions, _ = document_units(gold_document)
    rows = []
    for pair in align_questions(ocr_questions, gold_questions, min_similarity=min_similarity):
        ocr = ocr_questions[pair["ocr_index"]] if pair["ocr_index"] is not None else None
        gold = gold_questions[pair["gold_index"]] if pair["gold_index"] is not None else None
        rows.append({
            "ocr_question": ocr.get("number") if ocr else None,
            "gold_question": gold.get("number") if gold else None,
            "ocr_unit_id": ocr.get("_unit_id") if ocr else None,
            "gold_unit_id": gold.get("_unit_id") if gold else None,
            "similarity": round(pair["similarity"], 4),
            "operation": pair["operation"],
            "match_method": pair["match_method"],
            "comparison_method": pair.get("comparison_method", pair["match_method"]),
            "used_parent_fallback": pair["used_parent_fallback"],
            "similarity_backend": SIMILARITY_BACKEND,
        })
    return rows


def run_token_alignment(ocr_document, gold_document, *, min_similarity=0.35):
    rows = align_documents(ocr_document, gold_document, min_similarity=min_similarity)
    errors, error_counts = label_unit_errors(rows)
    excluded = excluded_fraction(rows)
    method_counts = Counter(row["match_method"] for row in rows)
    comparison_counts = Counter(row.get("comparison_method", "unknown") for row in rows)
    return {
        "unit_pair_count": len(rows),
        "similarity_backend": SIMILARITY_BACKEND,
        "excluded": excluded,
        "match_method_counts": dict(sorted(method_counts.items())),
        "comparison_method_counts": dict(sorted(comparison_counts.items())),
        "error_counts": error_counts,
        "errors": errors,
        "unit_alignments": rows,
    }


def write_jsonl(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_report(ocr_document, gold_document, output_path="token_alignment_results.json", *, min_similarity=0.35):
    report = run_token_alignment(ocr_document, gold_document, min_similarity=min_similarity)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
