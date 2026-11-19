"""Compare paired unit contents and label likely syntax or meaning errors."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

try:
    from .alignment_core import SIMILARITY_BACKEND, align_questions
    from .content_tokenizer import character_records, tokenize_content
    from .unit_matching import align_documents
    from .unit_extraction import document_units
except ImportError:
    from alignment_core import SIMILARITY_BACKEND, align_questions
    from content_tokenizer import character_records, tokenize_content
    from unit_matching import align_documents
    from unit_extraction import document_units


LATEX_SYNTAX_TOKEN = re.compile(r"^\\[A-Za-z]+$|^[{}_^$]$|^\\[\[\]]$")


def _tokens(unit):
    """Tokenize one unit, using a single placeholder for a table or figure."""
    if unit is None:
        return []
    kind, text = unit["unit_type"], str(unit.get("text", ""))
    if kind == "figure":
        return [{"token": f"<FIGURE:{text}>", "category": "figure"}] if text else []
    if kind == "table":
        return [{"token": "<table>", "category": "table"}]
    if kind == "display_equation":
        return tokenize_content(f"${text}$")
    return tokenize_content(text)


def _chars(unit):
    """Turn one unit into character records, keeping media as placeholders."""
    if unit is None:
        return []
    kind, text = unit["unit_type"], str(unit.get("text", ""))
    if kind == "figure":
        return [{"token": f"<FIGURE:{text}>", "category": "figure"}] if text else []
    if kind == "table":
        return [{"token": "<table>", "category": "table"}]
    if kind == "display_equation":
        return character_records(f"${text}$")
    return character_records(text)


def edit_alignment(reference, hypothesis, *, include_matches=False):
    """Align two token or character lists and describe every edit.

    ``reference`` is the gold sequence and ``hypothesis`` is the OCR sequence.
    The result includes positions, edit operations, total distance, and the
    number of gold items. Matches can be included for callers that need the
    full alignment path.
    """
    n, m = len(reference), len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    steps = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0], steps[i][0] = i, "deletion"
    for j in range(1, m + 1):
        costs[0][j], steps[0][j] = j, "insertion"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = reference[i - 1]["token"] == hypothesis[j - 1]["token"]
            choices = [
                (costs[i - 1][j - 1] + (0 if same else 1), "match" if same else "substitution", 0),
                (costs[i - 1][j] + 1, "deletion", 1),
                (costs[i][j - 1] + 1, "insertion", 2),
            ]
            costs[i][j], steps[i][j] = min(choices, key=lambda choice: (choice[0], choice[2]))[:2]

    edits = []
    i, j = n, m
    while i or j:
        operation = steps[i][j]
        if operation in {"match", "substitution"}:
            gold_item, ocr_item = reference[i - 1], hypothesis[j - 1]
            if operation != "match" or include_matches:
                edits.append({
                    "operation": operation,
                    "category": gold_item.get("category") or ocr_item.get("category"),
                    "gold": gold_item["token"], "ocr": ocr_item["token"],
                    "gold_index": i - 1, "ocr_index": j - 1,
                })
            i -= 1
            j -= 1
        elif operation == "deletion":
            gold_item = reference[i - 1]
            edits.append({
                "operation": "deletion", "category": gold_item.get("category"),
                "gold": gold_item["token"], "ocr": None,
                "gold_index": i - 1, "ocr_index": None,
            })
            i -= 1
        else:
            ocr_item = hypothesis[j - 1]
            edits.append({
                "operation": "insertion", "category": ocr_item.get("category"),
                "gold": None, "ocr": ocr_item["token"],
                "gold_index": None, "ocr_index": j - 1,
            })
            j -= 1
    return list(reversed(edits)), costs[n][m], len(reference)


def compare_unit_content(rows):
    """Add token and character comparisons to already matched unit rows."""
    for row in rows:
        ocr = (
            {"unit_type": row["unit_type"], "text": row.get("ocr_text")}
            if row.get("ocr_unit_id") else None
        )
        gold = (
            {"unit_type": row["unit_type"], "text": row.get("gold_text")}
            if row.get("gold_unit_id") else None
        )
        token_path, token_distance, gold_token_count = edit_alignment(
            _tokens(gold), _tokens(ocr), include_matches=True
        )
        token_errors = [edit for edit in token_path if edit["operation"] != "match"]
        character_errors, character_distance, gold_character_count = edit_alignment(
            _chars(gold), _chars(ocr)
        )
        row.update({
            "token_alignment": token_path,
            "token_errors": token_errors,
            "unaligned_tokens": [
                edit for edit in token_errors
                if edit["operation"] in {"insertion", "deletion"}
            ],
            "token_edit_distance": token_distance,
            "gold_token_count": gold_token_count,
            "character_errors": character_errors,
            "character_edit_distance": character_distance,
            "gold_character_count": gold_character_count,
        })
    return rows


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
    """Label each edit as syntax or semantic and count included matched edits."""
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
    """Measure how many OCR/gold unit sides were excluded from comparison."""
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
    """Return question-level matches with their similarity and method details."""
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
    """Pair content units, compare their contents, and summarize the edit results."""
    rows = align_documents(ocr_document, gold_document, min_similarity=min_similarity)
    compare_unit_content(rows)
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
    """Write mappings as UTF-8 JSON Lines, creating the parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_report(ocr_document, gold_document,
                 output_path="paper/results/reports/token_alignment_results.json", *, min_similarity=0.35):
    """Run alignment and save its full JSON report to ``output_path``."""
    report = run_token_alignment(ocr_document, gold_document, min_similarity=min_similarity)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
