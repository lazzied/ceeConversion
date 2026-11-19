"""Pair OCR content units with reference units inside aligned questions.

This module chooses correspondences and records how each pair was found.
Token and character comparisons are handled by :mod:`content_alignment`.
"""
from __future__ import annotations

try:
    from .alignment_core import SIMILARITY_BACKEND, align_questions, matching_stage, monotonic_alignment, similarity
    from .unit_extraction import document_units
except ImportError:
    from alignment_core import SIMILARITY_BACKEND, align_questions, matching_stage, monotonic_alignment, similarity
    from unit_extraction import document_units


def _unit_pair(ocr, gold, method, *, parent_fallback=False, matched=True, min_similarity=0.30):
    """Record one unit correspondence and keep its matching provenance."""
    score = similarity(ocr.get("text", "") if ocr else "", gold.get("text", "") if gold else "")
    operation = "match" if matched else ("ocr_only" if ocr else "gold_only")
    if matched and score < min_similarity and method not in {"parent_label", "question_number_fallback"}:
        operation = "excluded_low_similarity"
    return {
        "unit_type": (ocr or gold)["unit_type"],
        "ocr_unit_id": ocr.get("unit_id") if ocr else None,
        "gold_unit_id": gold.get("unit_id") if gold else None,
        "ocr_parent_id": ocr.get("parent_unit_id") if ocr else None,
        "gold_parent_id": gold.get("parent_unit_id") if gold else None,
        "ocr_location": ocr.get("location") if ocr else None,
        "gold_location": gold.get("location") if gold else None,
        "ocr_table_row_widths": ocr.get("table_row_widths") if ocr else None,
        "gold_table_row_widths": gold.get("table_row_widths") if gold else None,
        "ocr_label": ocr.get("label") if ocr else None,
        "gold_label": gold.get("label") if gold else None,
        "ocr_text": ocr.get("text") if ocr else None,
        "gold_text": gold.get("text") if gold else None,
        "similarity": round(score, 4),
        "comparison_method": matching_stage(
            ocr.get("text", "") if ocr else "", gold.get("text", "") if gold else ""
        ) if matched else "unmatched",
        "operation": operation,
        "match_method": method if operation in {"match", "excluded_low_similarity"} else "unmatched",
        "used_parent_fallback": bool(parent_fallback and operation == "match"),
        "similarity_backend": SIMILARITY_BACKEND,
        "count_in_metrics": (ocr or gold)["unit_type"] != "display_equation",
        "excluded": operation != "match",
    }


def _align_lists(ocr_units, gold_units, *, method, score_fn=None, min_similarity=0.30, parent_fallback=False):
    """Align two ordered unit lists and create a report row for every outcome."""
    if score_fn is None:
        score_fn = lambda left, right: similarity(left.get("text", ""), right.get("text", ""))
    ops = monotonic_alignment(ocr_units, gold_units, score_fn, min_similarity=min_similarity)
    pairs = []
    for operation, oi, gi in ops:
        ocr = ocr_units[oi] if oi is not None else None
        gold = gold_units[gi] if gi is not None else None
        if operation == "match":
            pairs.append(_unit_pair(ocr, gold, method, parent_fallback=parent_fallback, min_similarity=min_similarity))
        else:
            pairs.append(_unit_pair(ocr, gold, method, matched=False))
    return pairs


def _subquestion_pairs(ocr_subs, gold_subs, parent_fallback=False):
    """Pair sub-questions by wording and label within their parent question."""
    def score(ocr, gold):
        """Score sub-questions, strongly favoring matching printed labels."""
        text_score = similarity(ocr.get("text", ""), gold.get("text", ""))
        return max(text_score, 0.95 if ocr.get("label") == gold.get("label") else 0.0)

    ops = monotonic_alignment(ocr_subs, gold_subs, score, min_similarity=0.30)
    pairs = []
    for operation, oi, gi in ops:
        ocr = ocr_subs[oi] if oi is not None else None
        gold = gold_subs[gi] if gi is not None else None
        if operation == "match":
            method = "parent_fallback" if parent_fallback else ("parent_label" if ocr.get("label") == gold.get("label") else "parent_fuzzy")
            pairs.append(_unit_pair(ocr, gold, method, parent_fallback=parent_fallback, min_similarity=0.30))
        else:
            pairs.append(_unit_pair(ocr, gold, "unmatched", matched=False))
    return pairs


def _metric_blocks(group):
    """Return content units after question and sub-question records."""
    return [
        unit for unit in group["content_units"]
        if unit["unit_type"] not in {"question", "sub_question"}
    ]


def _block_score(ocr, gold):
    """Score content blocks while keeping figures and table cells in place."""
    same_labeled_kind = (
        ocr["unit_type"] == gold["unit_type"]
        and ocr.get("label") == gold.get("label")
        and ocr.get("label")
    )
    if same_labeled_kind:
        return 1.0
    if ocr["unit_type"] == "figure" or gold["unit_type"] == "figure":
        same_figure = (
            ocr["unit_type"] == gold["unit_type"]
            and ocr.get("text") == gold.get("text")
        )
        return 1.0 if same_figure else 0.0
    if ocr["unit_type"] == "table_cell" or gold["unit_type"] == "table_cell":
        if ocr.get("cell_index") != gold.get("cell_index"):
            return 0.0
    return similarity(ocr.get("text", ""), gold.get("text", ""))


def _align_question_equations(ocr_group, gold_group, sub_map, min_similarity):
    """Align equations inside matched sub-questions, then in question scope."""
    ocr_equations = ocr_group["display_equations"]
    gold_equations = gold_group["display_equations"]
    paired_rows = []
    used_ocr, used_gold = set(), set()

    for ocr_sub_id, gold_sub_id in sub_map.items():
        ocr_subset = [
            eq for eq in ocr_equations
            if eq.get("parent_subquestion_id") == ocr_sub_id
        ]
        gold_subset = [
            eq for eq in gold_equations
            if eq.get("parent_subquestion_id") == gold_sub_id
        ]
        pairs = _align_lists(
            ocr_subset,
            gold_subset,
            method="subquestion_scoped",
            min_similarity=min_similarity,
        )
        paired_rows.extend(pairs)
        used_ocr.update(pair["ocr_unit_id"] for pair in pairs if pair["ocr_unit_id"])
        used_gold.update(pair["gold_unit_id"] for pair in pairs if pair["gold_unit_id"])

    remaining_ocr = [eq for eq in ocr_equations if eq["unit_id"] not in used_ocr]
    remaining_gold = [eq for eq in gold_equations if eq["unit_id"] not in used_gold]
    if remaining_ocr or remaining_gold:
        fallback = any(eq.get("parent_subquestion_id") for eq in remaining_ocr + remaining_gold)
        paired_rows.extend(_align_lists(
            remaining_ocr,
            remaining_gold,
            method="parent_fallback" if fallback else "question_scope",
            min_similarity=min_similarity,
            parent_fallback=fallback,
        ))
    return paired_rows


def _unmatched_question_units(group):
    """Return all descendants of a question whose parent did not match."""
    if group is None:
        return []
    descendants = [
        unit for unit in group["content_units"]
        if unit["unit_type"] not in {"question", "sub_question"}
    ]
    return [group["question"], *group["sub_questions"], *group["display_equations"], *descendants]


def align_documents(ocr_document, gold_document, *, min_similarity=0.35):
    """Pair document units from questions down to text, figures, and tables.

    Questions are paired first. Their sub-questions, equations, and remaining
    content blocks are then aligned within that question's scope. Unmatched
    question descendants stay in the result with an explicit excluded status.
    """
    ocr_questions, ocr_tree = document_units(ocr_document)
    gold_questions, gold_tree = document_units(gold_document)
    question_pairs = align_questions(ocr_questions, gold_questions, min_similarity=min_similarity)
    rows = []

    for parent_pair in question_pairs:
        oi, gi = parent_pair["ocr_index"], parent_pair["gold_index"]
        ocr_q = ocr_questions[oi] if oi is not None else None
        gold_q = gold_questions[gi] if gi is not None else None
        ocr_group = ocr_tree[ocr_q["_unit_id"]] if ocr_q else None
        gold_group = gold_tree[gold_q["_unit_id"]] if gold_q else None
        q_method = parent_pair["match_method"]
        if parent_pair["operation"] == "match":
            rows.append(_unit_pair(
                ocr_group["question"], gold_group["question"], q_method,
                parent_fallback=parent_pair["used_parent_fallback"], min_similarity=min_similarity,
            ))
            sub_pairs = _subquestion_pairs(
                ocr_group["sub_questions"], gold_group["sub_questions"],
                parent_fallback=parent_pair["used_parent_fallback"],
            )
            rows.extend(sub_pairs)
            sub_matches = [p for p in sub_pairs if p["operation"] == "match"]
            sub_map = {p["ocr_unit_id"]: p["gold_unit_id"] for p in sub_matches}

            rows.extend(_align_question_equations(
                ocr_group, gold_group, sub_map, min_similarity
            ))

            # Align remaining content blocks in the question's own scope.
            left = _metric_blocks(ocr_group)
            right = _metric_blocks(gold_group)
            rows.extend(_align_lists(
                left, right, method="parent_block_sequence", score_fn=_block_score,
                min_similarity=min_similarity,
            ))
        else:
            # Preserve unmatched descendants as excluded pairs, rather than
            # silently dropping content from coverage calculations.
            left_units = _unmatched_question_units(ocr_group)
            right_units = _unmatched_question_units(gold_group)
            rows.extend(_unit_pair(unit, None, "unmatched_parent", matched=False) for unit in left_units)
            rows.extend(_unit_pair(None, unit, "unmatched_parent", matched=False) for unit in right_units)

    return rows

