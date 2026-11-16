"""Raw/normalized CER, unit outcomes, character confusions, and normalization harm."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher

try:
    from .alignment_core import SIMILARITY_BACKEND
    from .content_tokenizer import character_records, tokenize_content
    from .unit_matching import align_documents, edit_alignment
    from .unit_extraction import content_units, document_units, flatten_questions
except ImportError:
    from alignment_core import SIMILARITY_BACKEND
    from content_tokenizer import character_records, tokenize_content
    from unit_matching import align_documents, edit_alignment
    from unit_extraction import content_units, document_units, flatten_questions


def edit_counts(reference, hypothesis):
    edits, distance, reference_length = edit_alignment(reference, hypothesis)
    counts = Counter(edit["operation"] for edit in edits)
    return (
        counts["substitution"], counts["deletion"], counts["insertion"],
        distance, reference_length,
    )


def _records(unit):
    if unit is None:
        return []
    kind, text = unit["unit_type"], str(unit.get("text", ""))
    if kind == "figure":
        return [{"token": f"<FIGURE:{text}>", "category": "figure"}] if text else []
    if kind == "table":
        return [{"token": "<table>", "category": "table"}]
    return character_records(text)


def _word_records(unit_type, text):
    text = str(text or "")
    if unit_type == "figure":
        return [{"token": f"<FIGURE:{text}>", "category": "figure"}] if text else []
    if unit_type == "table":
        return [{"token": "<table>", "category": "table"}]
    if unit_type == "display_equation":
        text = f"${text}$"
    return tokenize_content(text)


def _token_metrics(rows):
    edits = Counter()
    references = 0
    exact_units = 0
    matched_units = 0
    categories = defaultdict(Counter)
    category_references = Counter()
    for row in rows:
        if not row.get("count_in_metrics", True) or row.get("operation") != "match":
            continue
        gold = _word_records(row["unit_type"], row.get("gold_text"))
        ocr = _word_records(row["unit_type"], row.get("ocr_text"))
        operations, distance, reference_length = edit_alignment(gold, ocr)
        references += reference_length
        for item in gold:
            category_references[item.get("category") or "other_text"] += 1
        matched_units += 1
        exact_units += distance == 0
        for operation in operations:
            edits[operation["operation"]] += 1
            categories[operation.get("category") or "other_text"][operation["operation"]] += 1
    error_count = edits["substitution"] + edits["insertion"] + edits["deletion"]
    return {
        "word_error_rate": round(error_count / references, 6) if references else None,
        "tokenization": "Latin words, one character per CJK symbol, LaTeX-aware math atoms, and no whitespace tokens; interpret as a mixed-content word/token error rate.",
        "word_edits": {
            "substitutions": edits["substitution"],
            "insertions": edits["insertion"],
            "deletions": edits["deletion"],
            "reference_tokens": references,
            "total_errors": error_count,
        },
        "exact_match": {
            "exact_units": exact_units,
            "matched_units": matched_units,
            "rate": round(exact_units / matched_units, 6) if matched_units else None,
        },
        "word_errors_by_content_class": {
            category: {
                **dict(counts),
                "reference_tokens": category_references[category],
                "word_error_rate": round(
                    sum(counts[key] for key in ("substitution", "insertion", "deletion"))
                    / category_references[category], 6
                ) if category_references[category] else None,
            }
            for category, counts in sorted(
                {key: categories[key] for key in set(categories) | set(category_references)}.items()
            )
        },
    }


def _equation_similarity(rows):
    scores = []
    for row in rows:
        if row.get("unit_type") != "display_equation" or row.get("operation") != "match":
            continue
        reference = _word_records("display_equation", row.get("gold_text"))
        hypothesis = _word_records("display_equation", row.get("ocr_text"))
        _, distance, reference_length = edit_alignment(reference, hypothesis)
        scores.append(max(0.0, 1.0 - distance / reference_length) if reference_length else float(distance == 0))
    return {
        "method": "mean normalized token edit similarity; this is an equation-similarity proxy, not Formula-CDM",
        "matched_equations": len(scores),
        "mean_similarity": round(sum(scores) / len(scores), 6) if scores else None,
        "exact_equation_rate": round(sum(score == 1.0 for score in scores) / len(scores), 6) if scores else None,
    }


def _table_structure(rows):
    gold_tables = [row for row in rows if row.get("unit_type") == "table" and row.get("gold_unit_id")]
    exact = sum(
        row.get("operation") == "match"
        and row.get("ocr_table_row_widths") == row.get("gold_table_row_widths")
        for row in gold_tables
    )
    return {
        "exact_table_layouts": exact,
        "reference_tables": len(gold_tables),
        "accuracy": round(exact / len(gold_tables), 6) if gold_tables else None,
        "comparison": "row count and number of cells in each row; cell text is scored separately",
    }


def _reading_order(ocr_document, gold_document, min_similarity=0.35):
    try:
        from .alignment_core import align_questions
    except ImportError:
        from alignment_core import align_questions
    ocr_questions, gold_questions = flatten_questions(ocr_document), flatten_questions(gold_document)
    question_pairs = align_questions(ocr_questions, gold_questions, min_similarity=min_similarity)
    question_scores = []
    for pair in question_pairs:
        if pair["operation"] != "match":
            continue
        left = ocr_questions[pair["ocr_index"]].get("blocks", [])
        right = gold_questions[pair["gold_index"]].get("blocks", [])
        gold_kinds = [str(block.get("kind", "unknown")) for block in right]
        if not gold_kinds:
            continue
        ocr_kinds = [str(block.get("kind", "unknown")) for block in left]
        matcher = SequenceMatcher(a=ocr_kinds, b=gold_kinds, autojunk=False)
        in_order_gold_blocks = sum(block.size for block in matcher.get_matching_blocks())
        question_scores.append(in_order_gold_blocks / len(gold_kinds))
    return {
        "method": "share of reference block types retained in the same order within matched questions",
        "matched_questions_with_blocks": len(question_scores),
        "accuracy_proxy": round(sum(question_scores) / len(question_scores), 6) if question_scores else None,
    }


def _alignment_coverage(rows):
    eligible = [row for row in rows if row.get("count_in_metrics", True)]
    ocr_sides = sum(bool(row.get("ocr_unit_id")) for row in eligible)
    gold_sides = sum(bool(row.get("gold_unit_id")) for row in eligible)
    matched = sum(row.get("operation") == "match" and not row.get("excluded") for row in eligible)
    unmatched = sum(row.get("operation") != "match" for row in eligible)
    ocr_only = sum(row.get("operation") == "ocr_only" for row in eligible)
    reference_only = sum(row.get("operation") == "gold_only" for row in eligible)
    return {
        "matched_pairs": matched,
        "unmatched_or_excluded_rows": unmatched,
        "ocr_only_units": ocr_only,
        "reference_only_units": reference_only,
        "unmatched_row_fraction": round(unmatched / len(eligible), 6) if eligible else None,
        "ocr_units": ocr_sides,
        "reference_units": gold_sides,
        "ocr_coverage": round(matched / ocr_sides, 6) if ocr_sides else None,
        "reference_coverage": round(matched / gold_sides, 6) if gold_sides else None,
        "combined_coverage": round(2 * matched / (ocr_sides + gold_sides), 6) if ocr_sides + gold_sides else None,
    }


def _manual_alignment_review(audit_rows):
    counts = Counter(row.get("alignment_review_label") for row in audit_rows if row.get("alignment_review_label"))
    true_positive = counts["correct_match"]
    false_positive = counts["incorrect_match"]
    false_negative = counts["missed_match"]
    reviewed_matches = true_positive + false_positive
    reviewed_unmatched = false_negative + counts["no_counterpart"]
    return {
        "labels": dict(sorted(counts.items())),
        "sampled_rows": len(audit_rows),
        "reviewed_rows": sum(counts.values()),
        "alignment_precision": round(true_positive / reviewed_matches, 6) if reviewed_matches else None,
        "alignment_recall": round(true_positive / (true_positive + false_negative), 6) if true_positive + false_negative else None,
        "reviewed_matched_rows": reviewed_matches,
        "reviewed_unmatched_rows": reviewed_unmatched,
        "guidance": "If both unit IDs are present, choose correct_match or incorrect_match. If only one side exists, choose missed_match when its counterpart exists elsewhere, otherwise no_counterpart. Leave unreviewed rows blank.",
    }


def _math_subtype(token):
    value = str(token or "")
    greek_chars = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
    greek_commands = {
        "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
        "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
        "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi",
        "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda",
        "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
    }
    if (value.startswith("\\") and value[1:] in greek_commands) or any(char in greek_chars for char in value):
        return "greek_letters"
    if value.isdigit():
        return "digits"
    if value in {"^", "_"}:
        return "superscript_subscript"
    if value.startswith("\\"):
        return "latex_commands"
    if value.isalpha():
        return "latin_letters"
    if value in set("+-−–×·*/÷=<>≤≥≠≈∼∝±∓"):
        return "operators"
    if value in set("{}[]()|\\$"):
        return "braces_and_delimiters"
    return "other_math"


def _error_distribution(rows):
    by_class, by_unit, by_operation = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    confusions, math_confusions, math_subtypes = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        if row.get("operation") != "match" or not row.get("count_in_metrics", True):
            continue
        for error in row.get("token_errors", []):
            category = error.get("category") or "other_text"
            operation = error.get("operation", "unknown")
            by_class[category][operation] += 1
            by_unit[row.get("unit_type", "unknown")][operation] += 1
            by_operation[operation][category] += 1
            if category in {"math", "math_syntax"}:
                subtype = _math_subtype(error.get("gold") or error.get("ocr"))
                math_subtypes[subtype][operation] += 1
            if operation == "substitution":
                pair = (error.get("gold") or "", error.get("ocr") or "")
                confusions[category][pair] += 1
                if category in {"math", "math_syntax"}:
                    math_confusions[subtype][pair] += 1
    return {
        "by_content_class": {key: dict(value) for key, value in sorted(by_class.items())},
        "by_unit_type": {key: dict(value) for key, value in sorted(by_unit.items())},
        "by_operation": {key: dict(value) for key, value in sorted(by_operation.items())},
        "by_math_subtype": {key: dict(value) for key, value in sorted(math_subtypes.items())},
        "substitution_confusions_by_class": {
            key: [{"gold": gold, "ocr": ocr, "count": count} for (gold, ocr), count in value.most_common(50)]
            for key, value in sorted(confusions.items())
        },
        "math_substitution_confusions": {
            key: [{"gold": gold, "ocr": ocr, "count": count} for (gold, ocr), count in value.most_common(50)]
            for key, value in sorted(math_confusions.items())
        },
        "confidence": {"available": False, "reason": "PaddleOCR-VL result confidence is not currently mapped to aligned text units."},
    }


def _new_counts():
    return {key: 0 for key in ERROR_KEYS}


def _add_edits(totals, reference, hypothesis):
    edits, distance, reference_length = edit_alignment(reference, hypothesis)
    totals["reference_characters"] += reference_length
    totals["overall"]["reference_characters"] += reference_length
    for record in reference:
        category = record.get("category") or "other_text"
        totals["by_category"][category]["reference_characters"] += 1
    for edit in edits:
        operation = edit["operation"]
        category = edit.get("category") or "other_text"
        totals["by_category"][category][operation + "s"] += 1
        totals["overall"][operation + "s"] += 1
    return distance, reference_length


def _metric_result(counts):
    errors = counts["substitutions"] + counts["insertions"] + counts["deletions"]
    reference_length = counts["reference_characters"]
    rate = errors / reference_length if reference_length else None
    return {
        **counts,
        "total_errors": errors,
        "cer": round(rate, 6) if rate is not None else None,
        "error_percent": round(rate * 100, 3) if rate is not None else None,
    }


def _metric_bucket():
    return {"substitutions": 0, "insertions": 0, "deletions": 0, "reference_characters": 0}


def _empty_metric_totals():
    return {"overall": _metric_bucket(), "by_category": defaultdict(_metric_bucket), "reference_characters": 0}


def _unit_lookup(document):
    _, tree = document_units(document)
    lookup = {}
    for group in tree.values():
        for unit in [group["question"], *group["sub_questions"], *group["display_equations"], *group["content_units"]]:
            lookup[unit["unit_id"]] = unit
    return lookup


def _evaluate_pairs(rows, lookup_ocr, lookup_gold, *, count_only_metric_units=True):
    totals = _empty_metric_totals()
    alignable = 0
    for row in rows:
        if count_only_metric_units and not row.get("count_in_metrics", True):
            continue
        if row.get("operation") != "match":
            continue
        ocr = lookup_ocr.get(row.get("ocr_unit_id")) if lookup_ocr else None
        gold = lookup_gold.get(row.get("gold_unit_id")) if lookup_gold else None
        if lookup_ocr and ocr is None or lookup_gold and gold is None:
            continue
        if not lookup_ocr:
            ocr = {"unit_type": row["unit_type"], "text": row.get("ocr_text", "")}
        if not lookup_gold:
            gold = {"unit_type": row["unit_type"], "text": row.get("gold_text", "")}
        ref_records, ocr_records = _records(gold), _records(ocr)
        _add_edits(totals, ref_records, ocr_records)
        alignable += 1
    return {
        "alignable_unit_count": alignable,
        "overall": _metric_result(totals["overall"]),
        "by_content_class": {
            category: _metric_result(counts)
            for category, counts in sorted(totals["by_category"].items())
        },
    }


def _unit_error(row, ocr_unit, gold_unit):
    edits, distance, reference_length = edit_alignment(_records(gold_unit), _records(ocr_unit))
    return distance, reference_length, edits


def _outcome_report(normalized_rows, raw_ocr_units, raw_gold_units):
    outcomes = defaultdict(Counter)
    per_unit = []
    for row in normalized_rows:
        if not row.get("count_in_metrics", True):
            continue
        norm_ocr = {"unit_type": row["unit_type"], "text": row.get("ocr_text") or ""} if row.get("ocr_unit_id") else None
        norm_gold = {"unit_type": row["unit_type"], "text": row.get("gold_text") or ""} if row.get("gold_unit_id") else None
        raw_ocr = raw_ocr_units.get(row.get("ocr_unit_id")) if row.get("ocr_unit_id") else None
        raw_gold = raw_gold_units.get(row.get("gold_unit_id")) if row.get("gold_unit_id") else None
        if row["operation"] != "match" or raw_ocr is None or raw_gold is None:
            status = "excluded"
            raw_distance = norm_distance = None
            raw_rate = norm_rate = None
        else:
            raw_distance, raw_len, _ = _unit_error(row, raw_ocr, raw_gold)
            norm_distance, norm_len, _ = _unit_error(row, norm_ocr, norm_gold)
            raw_rate = raw_distance / raw_len if raw_len else (0.0 if not raw_distance else None)
            norm_rate = norm_distance / norm_len if norm_len else (0.0 if not norm_distance else None)
            if norm_distance == 0:
                status = "perfect"
            elif norm_distance < raw_distance:
                status = "better"
            elif norm_distance > raw_distance:
                status = "worse"
            else:
                status = "same"
        outcomes[row["unit_type"]][status] += 1
        per_unit.append({
            "unit_type": row["unit_type"], "ocr_unit_id": row.get("ocr_unit_id"),
            "gold_unit_id": row.get("gold_unit_id"), "status": status,
            "raw_edit_distance": raw_distance, "normalized_edit_distance": norm_distance,
            "raw_cer": round(raw_rate, 6) if raw_rate is not None else None,
            "normalized_cer": round(norm_rate, 6) if norm_rate is not None else None,
        })
    status_counts = ("better", "worse", "same", "perfect", "excluded")
    return {
        "overall": {key: sum(group[key] for group in outcomes.values()) for key in status_counts},
        "by_unit_type": {
            kind: {key: values[key] for key in status_counts}
            for kind, values in sorted(outcomes.items())
        },
        "units": per_unit,
    }


def _confusion_table(rows, raw_ocr_units, raw_gold_units, limit=50):
    confusion = Counter()
    for row in rows:
        if not row.get("count_in_metrics", True) or row.get("operation") != "match":
            continue
        ocr = raw_ocr_units.get(row.get("ocr_unit_id"))
        gold = raw_gold_units.get(row.get("gold_unit_id"))
        if ocr is None or gold is None:
            continue
        edits, _, _ = edit_alignment(_records(gold), _records(ocr))
        for edit in edits:
            if edit["operation"] == "substitution":
                confusion[(edit.get("gold") or "", edit.get("ocr") or "", edit.get("category") or "other_text")] += 1
    return [
        {"gold": gold, "ocr": ocr, "category": category, "count": count}
        for (gold, ocr, category), count in confusion.most_common(limit)
    ]


def _equal_index_map(left, right):
    mapping = {}
    for block in SequenceMatcher(a=left, b=right, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    return mapping


def normalization_harm(raw_ocr, raw_gold, normalized_ocr, normalized_gold):
    """Among raw OCR characters correct against raw gold, count normalization regressions."""
    raw_ocr = [char for char in raw_ocr if not char.isspace()]
    raw_gold = [char for char in raw_gold if not char.isspace()]
    norm_ocr = [char for char in normalized_ocr if not char.isspace()]
    norm_gold = [char for char in normalized_gold if not char.isspace()]
    raw_correct = []
    for block in SequenceMatcher(a=raw_gold, b=raw_ocr, autojunk=False).get_matching_blocks():
        raw_correct.extend((block.b + offset, block.a + offset) for offset in range(block.size))
    ocr_map = _equal_index_map(raw_ocr, norm_ocr)
    gold_map = _equal_index_map(raw_gold, norm_gold)
    damaged = 0
    for oi, gi in raw_correct:
        ni, ng = ocr_map.get(oi), gold_map.get(gi)
        if ni is None or ng is None or norm_ocr[ni] != norm_gold[ng]:
            damaged += 1
    return {
        "raw_correct_character_count": len(raw_correct),
        "correct_characters_wrongly_changed": damaged,
        "share_wrongly_changed": round(damaged / len(raw_correct), 6) if raw_correct else None,
        "definition": "A raw OCR/gold character match is counted as damaged if normalization does not preserve a matching normalized OCR/gold character.",
    }


def evaluate_pipeline(raw_rows, normalized_rows, raw_ocr_document, raw_gold_document,
                      normalized_ocr_document, normalized_gold_document,
                      raw_ocr_markdown, raw_gold_markdown,
                      normalized_ocr_markdown, normalized_gold_markdown,
                      *, low_error_threshold=0.05, audit_rows=None,
                      document_name="document", min_similarity=0.35):
    raw_ocr_units, raw_gold_units = _unit_lookup(raw_ocr_document), _unit_lookup(raw_gold_document)
    norm_ocr_units, norm_gold_units = _unit_lookup(normalized_ocr_document), _unit_lookup(normalized_gold_document)
    raw_metrics = _evaluate_pairs(raw_rows, raw_ocr_units, raw_gold_units)
    normalized_metrics = _evaluate_pairs(normalized_rows, norm_ocr_units, norm_gold_units)
    raw_token_metrics = _token_metrics(raw_rows)
    normalized_token_metrics = _token_metrics(normalized_rows)
    outcomes = _outcome_report(normalized_rows, raw_ocr_units, raw_gold_units)
    raw_cer = raw_metrics["overall"]["cer"]
    raw_error_distribution = _error_distribution(raw_rows)
    return {
        "alignment_backend": SIMILARITY_BACKEND,
        "raw_do_nothing_baseline": raw_metrics,
        "normalized_ocr_vs_normalized_gold": normalized_metrics,
        "raw_word_error": raw_token_metrics,
        "normalized_word_error": normalized_token_metrics,
        "exact_match_after_normalization": normalized_token_metrics["exact_match"],
        "equation_similarity_after_normalization": _equation_similarity(normalized_rows),
        "table_structure_after_normalization": _table_structure(normalized_rows),
        "reading_order_after_normalization": _reading_order(
            normalized_ocr_document, normalized_gold_document, min_similarity=min_similarity
        ),
        "alignment_quality": {
            "raw_coverage": _alignment_coverage(raw_rows),
            "normalized_coverage": _alignment_coverage(normalized_rows),
            "manual_review": _manual_alignment_review(audit_rows or []),
        },
        "unit_outcomes_after_normalization": outcomes,
        "normalization_correct_character_harm": normalization_harm(
            raw_ocr_markdown, raw_gold_markdown,
            normalized_ocr_markdown, normalized_gold_markdown,
        ),
        "top_50_gold_to_ocr_character_confusions_raw": _confusion_table(raw_rows, raw_ocr_units, raw_gold_units),
        "error_distribution_raw": {
            "document": document_name,
            "by_document": {document_name: raw_error_distribution},
        },
        "excluded_unit_fraction": {
            "raw": _excluded_summary(raw_rows),
            "normalized": _excluded_summary(normalized_rows),
        },
        "gate": {
            "raw_cer_threshold": low_error_threshold,
            "raw_cer_is_low": raw_cer is not None and raw_cer <= low_error_threshold,
            "recommendation": "prioritize_detection_and_guards" if raw_cer is not None and raw_cer <= low_error_threshold else "correction_may_be_valuable",
        },
        "notes": {
            "cer_classes": "CJK text, Latin text, math, figures, tables, and other visible characters are reported separately.",
            "raw_baseline": "OCR input is measured before shared style normalization; gold TeX has only been structurally converted to Markdown.",
            "unit_outcomes": "better/worse/same compare character edit distance before and after normalization; perfect means zero normalized edit distance.",
        },
    }


def _excluded_summary(rows):
    eligible = [row for row in rows if row.get("count_in_metrics", True)]
    total_sides = sum(bool(row.get("ocr_unit_id")) + bool(row.get("gold_unit_id")) for row in eligible)
    excluded_sides = sum(
        bool(row.get("ocr_unit_id")) + bool(row.get("gold_unit_id"))
        for row in eligible if row.get("excluded")
    )
    return {
        "excluded_side_units": excluded_sides,
        "total_side_units": total_sides,
        "fraction": round(excluded_sides / total_sides, 6) if total_sides else 0.0,
    }
