"""Hierarchical unit pairing and token/character edit labels."""

from __future__ import annotations

try:
    from .alignment_core import SIMILARITY_BACKEND, align_questions, matching_stage, monotonic_alignment, similarity
    from .content_tokenizer import character_records, tokenize_content
    from .unit_extraction import content_units, document_units
except ImportError:
    from alignment_core import SIMILARITY_BACKEND, align_questions, matching_stage, monotonic_alignment, similarity
    from content_tokenizer import character_records, tokenize_content
    from unit_extraction import content_units, document_units


def _tokens(unit):
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
    """Return gold-to-OCR Levenshtein operations; whitespace is not tokenized."""
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], trace[i][0] = i, "deletion"
    for j in range(1, m + 1):
        dp[0][j], trace[0][j] = j, "insertion"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = reference[i - 1]["token"] == hypothesis[j - 1]["token"]
            choices = [
                (dp[i - 1][j - 1] + (0 if same else 1), "match" if same else "substitution", 0),
                (dp[i - 1][j] + 1, "deletion", 1),
                (dp[i][j - 1] + 1, "insertion", 2),
            ]
            cost, operation, _ = min(choices, key=lambda item: (item[0], item[2]))
            dp[i][j], trace[i][j] = cost, operation

    edits = []
    i, j = n, m
    while i or j:
        operation = trace[i][j]
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
            edits.append({"operation": "deletion", "category": gold_item.get("category"), "gold": gold_item["token"], "ocr": None, "gold_index": i - 1, "ocr_index": None})
            i -= 1
        else:
            ocr_item = hypothesis[j - 1]
            edits.append({"operation": "insertion", "category": ocr_item.get("category"), "gold": None, "ocr": ocr_item["token"], "gold_index": None, "ocr_index": j - 1})
            j -= 1
    return list(reversed(edits)), dp[n][m], len(reference)


def _unit_pair(ocr, gold, method, *, parent_fallback=False, matched=True, min_similarity=0.30):
    score = similarity(ocr.get("text", "") if ocr else "", gold.get("text", "") if gold else "")
    operation = "match" if matched else ("ocr_only" if ocr else "gold_only")
    if matched and score < min_similarity and method not in {"parent_label", "question_number_fallback"}:
        operation = "excluded_low_similarity"
    token_alignment, token_distance, token_reference_len = edit_alignment(
        _tokens(gold), _tokens(ocr), include_matches=True
    )
    token_edits = [item for item in token_alignment if item["operation"] != "match"]
    char_edits, char_distance, char_reference_len = edit_alignment(_chars(gold), _chars(ocr))
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
        "token_alignment": token_alignment,
        "token_errors": token_edits,
        "unaligned_tokens": [
            item for item in token_edits if item["operation"] in {"insertion", "deletion"}
        ],
        "token_edit_distance": token_distance,
        "gold_token_count": token_reference_len,
        "character_errors": char_edits,
        "character_edit_distance": char_distance,
        "gold_character_count": char_reference_len,
        "count_in_metrics": (ocr or gold)["unit_type"] != "display_equation",
        "excluded": operation != "match",
    }


def _align_lists(ocr_units, gold_units, *, method, score_fn=None, min_similarity=0.30, parent_fallback=False):
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
    def score(ocr, gold):
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


def align_documents(ocr_document, gold_document, *, min_similarity=0.35):
    """Create matched question/subquestion/equation/content-unit rows."""
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

            # Match equations within an aligned subquestion first.
            ocr_eqs, gold_eqs = ocr_group["display_equations"], gold_group["display_equations"]
            used_ocr, used_gold = set(), set()
            for ocr_sub_id, gold_sub_id in sub_map.items():
                left = [eq for eq in ocr_eqs if eq.get("parent_subquestion_id") == ocr_sub_id]
                right = [eq for eq in gold_eqs if eq.get("parent_subquestion_id") == gold_sub_id]
                eq_pairs = _align_lists(left, right, method="subquestion_scoped", min_similarity=min_similarity)
                rows.extend(eq_pairs)
                used_ocr.update(p["ocr_unit_id"] for p in eq_pairs if p["ocr_unit_id"])
                used_gold.update(p["gold_unit_id"] for p in eq_pairs if p["gold_unit_id"])

            left_rest = [eq for eq in ocr_eqs if eq["unit_id"] not in used_ocr]
            right_rest = [eq for eq in gold_eqs if eq["unit_id"] not in used_gold]
            if left_rest or right_rest:
                parent_fallback = any(eq.get("parent_subquestion_id") for eq in left_rest + right_rest)
                method = "parent_fallback" if parent_fallback else "question_scope"
                rows.extend(_align_lists(
                    left_rest, right_rest, method=method,
                    min_similarity=min_similarity, parent_fallback=parent_fallback,
                ))

            # Align remaining content blocks in the question's own scope.
            def metric_blocks(group):
                return [u for u in group["content_units"] if u["unit_type"] not in {"question", "sub_question"}]
            left = metric_blocks(ocr_group)
            right = metric_blocks(gold_group)
            def block_score(a, b):
                if a["unit_type"] == b["unit_type"] and a.get("label") == b.get("label") and a.get("label"):
                    return 1.0
                if a["unit_type"] == "figure" or b["unit_type"] == "figure":
                    return 1.0 if a["unit_type"] == b["unit_type"] and a.get("text") == b.get("text") else 0.0
                if a["unit_type"] == "table_cell" or b["unit_type"] == "table_cell":
                    if a.get("cell_index") != b.get("cell_index"):
                        return 0.0
                return similarity(a.get("text", ""), b.get("text", ""))
            rows.extend(_align_lists(
                left, right, method="parent_block_sequence", score_fn=block_score,
                min_similarity=min_similarity,
            ))
        else:
            # Preserve unmatched descendants as excluded pairs, rather than
            # silently dropping content from coverage calculations.
            left_units = [ocr_group["question"], *ocr_group["sub_questions"], *ocr_group["display_equations"], *[u for u in ocr_group["content_units"] if u["unit_type"] not in {"question", "sub_question"}]] if ocr_group else []
            right_units = [gold_group["question"], *gold_group["sub_questions"], *gold_group["display_equations"], *[u for u in gold_group["content_units"] if u["unit_type"] not in {"question", "sub_question"}]] if gold_group else []
            rows.extend(_unit_pair(unit, None, "unmatched_parent", matched=False) for unit in left_units)
            rows.extend(_unit_pair(None, unit, "unmatched_parent", matched=False) for unit in right_units)

    return rows

