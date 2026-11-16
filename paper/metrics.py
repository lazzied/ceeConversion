"""Character-level and content-token metrics for aligned exam units."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from token_alignment import (
    align_question_units,
    character_tokens,
    load_segmentation,
    tokenize_math,
    tokenize_text,
)
from block_alignment import align_questions, flatten_questions


def edit_counts(reference, hypothesis):
    """Levenshtein counts, with OCR as hypothesis and reference as gold."""
    n, m = len(reference), len(hypothesis)
    previous = list(range(m + 1))
    trace = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        trace[i][0] = "D"
    for j in range(1, m + 1):
        trace[0][j] = "I"
    for i in range(1, n + 1):
        current = [i] + [0] * m
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                choices = ((previous[j - 1], "M", 0), (previous[j] + 1, "D", 1), (current[j - 1] + 1, "I", 2))
            else:
                choices = ((previous[j - 1] + 1, "S", 0), (previous[j] + 1, "D", 1), (current[j - 1] + 1, "I", 2))
            cost, operation, _ = min(choices, key=lambda item: (item[0], item[2]))
            current[j] = cost
            trace[i][j] = operation
        previous = current
    substitutions = deletions = insertions = 0
    i, j = n, m
    while i or j:
        operation = trace[i][j]
        if operation == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif operation == "D":
            deletions += 1
            i -= 1
        elif operation == "I":
            insertions += 1
            j -= 1
        else:
            i -= 1
            j -= 1
    return substitutions, deletions, insertions


def _tokens(unit, level, force_math=False):
    if unit is None:
        return []
    kind, text = unit["kind"], unit["text"]
    if kind == "figure":
        # Compare the referenced asset by exact identity.
        return [f"<FIGURE:{text}>"] if text else []
    if kind == "table":
        return ["<table>"]
    if level == "character":
        return character_tokens(text)
    return tokenize_math(text) if force_math or kind == "math" else tokenize_text(text)


def _new_total():
    return {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_length": 0}


def _add(total, reference, hypothesis):
    substitutions, deletions, insertions = edit_counts(reference, hypothesis)
    total["substitutions"] += substitutions
    total["deletions"] += deletions
    total["insertions"] += insertions
    total["reference_length"] += len(reference)


def _finish(total):
    errors = total["substitutions"] + total["deletions"] + total["insertions"]
    denominator = total["reference_length"]
    rate = errors / denominator if denominator else None
    return {
        **total,
        "total_errors": errors,
        "rate": round(rate, 6) if rate is not None else None,
        "error_percent": round(rate * 100, 3) if rate is not None else None,
    }


def evaluate_files(ocr_path, gold_path):
    ocr_questions = flatten_questions(load_segmentation(ocr_path))
    gold_questions = flatten_questions(load_segmentation(gold_path))
    character_totals, token_totals = defaultdict(_new_total), defaultdict(_new_total)
    unit_count = 0
    for operation, oi, gi in align_questions(ocr_questions, gold_questions):
        ocr_q = ocr_questions[oi] if oi is not None else {}
        gold_q = gold_questions[gi] if gi is not None else {}
        for pair in align_question_units(ocr_q, gold_q):
            kind = pair["kind"]
            reference = _tokens(pair["reference"], "character")
            hypothesis = _tokens(pair["ocr"], "character")
            reference_tokens = _tokens(pair["reference"], "token", force_math=(kind == "math"))
            hypothesis_tokens = _tokens(pair["ocr"], "token", force_math=(kind == "math"))
            _add(character_totals[kind], reference, hypothesis)
            _add(token_totals[kind], reference_tokens, hypothesis_tokens)
            unit_count += 1

    def aggregate(groups):
        total = _new_total()
        for group in groups.values():
            for key in total:
                total[key] += group[key]
        return _finish(total)

    return {
        "question_pair_count": len(align_questions(ocr_questions, gold_questions)),
        "unit_pair_count": unit_count,
        "character_error_rate": {
            "overall": aggregate(character_totals),
            "by_unit": {kind: _finish(value) for kind, value in sorted(character_totals.items())},
        },
        "content_token_error_rate": {
            "overall": aggregate(token_totals),
            "by_unit": {kind: _finish(value) for kind, value in sorted(token_totals.items())},
        },
        "tokenization": {
            "text": "Chinese characters are individual tokens; Latin words and number groups are tokens; whitespace is ignored.",
            "math": "Each LaTeX command is one token; braces, operators, letters, and digits are separate tokens; whitespace is ignored.",
            "figures": "One identity token; image references must match exactly.",
            "tables": "A <table> placeholder and the ordered cell contents are compared.",
        },
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python metrics.py <ocr_segmentation.json> <gold_segmentation.json>")
        raise SystemExit(2)
    report = evaluate_files(sys.argv[1], sys.argv[2])
    Path("metrics_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
