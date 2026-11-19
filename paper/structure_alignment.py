"""Question similarity and structural-error detection for exam OCR."""

from __future__ import annotations

import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from .alignment_core import SIMILARITY_BACKEND, align_questions as align_question_pairs
    from .alignment_core import similarity
except ImportError:
    from alignment_core import SIMILARITY_BACKEND, align_questions as align_question_pairs
    from alignment_core import similarity


def load_segmentation(path):
    """Load a saved segmentation JSON document."""
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def flatten_questions(document):
    """Flatten sections in reading order while retaining section metadata."""
    questions = []
    for section_index, section in enumerate(document.get("sections", [])):
        for question_index, question in enumerate(section.get("questions", [])):
            questions.append({
                **question, "_section_index": section_index,
                "_section_title": section.get("title", ""),
                "_ordinal": len(questions) + 1,
                "_unit_id": f"s{section_index + 1}.q{question_index + 1}",
            })
    return questions


def question_alignment_score(ocr_question, gold_question):
    """Stem-plus-number score used only to identify corresponding questions."""
    stem_score = similarity(ocr_question.get("stem", ""), gold_question.get("stem", ""))
    number_bonus = 0.45 if ocr_question.get("number") == gold_question.get("number") else -0.15
    return stem_score, max(0.0, min(1.0, stem_score + number_bonus / 2))


def align_questions(ocr_questions, gold_questions):
    """Compatibility tuple form for callers of the original helper."""
    return [
        ("match", pair["ocr_index"], pair["gold_index"])
        if pair["operation"] == "match"
        else (("left_only", pair["ocr_index"], None) if pair["operation"] == "ocr_only"
              else ("right_only", None, pair["gold_index"]))
        for pair in align_question_pairs(ocr_questions, gold_questions)
    ]


def _block_signature(block):
    """Return the block kind used to compare document structure."""
    # Labels are compared as their own structural property so a changed
    # subquestion/choice number is reported as a numbering error.
    return block.get("kind", "unknown")


def _error(error_type, location, detail):
    """Create one consistently shaped structural finding."""
    return {"type": error_type, "location": location, "detail": detail}


def detect_structure_errors(ocr_question, gold_question):
    """Compare document organization, labels, block kinds, and figure placement."""
    errors = []
    oq, gq = ocr_question, gold_question
    on, gn = oq.get("number"), gq.get("number")
    if on != gn:
        errors.append(_error("question_number_mismatch", "question", {"ocr": on, "reference": gn}))

    ocr_blocks, gold_blocks = oq.get("blocks", []), gq.get("blocks", [])
    ocr_sig = [_block_signature(block) for block in ocr_blocks]
    gold_sig = [_block_signature(block) for block in gold_blocks]
    matcher = SequenceMatcher(a=ocr_sig, b=gold_sig, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for oi, gi in zip(range(i1, i2), range(j1, j2)):
                ob, gb = ocr_blocks[oi], gold_blocks[gi]
                kind = gb.get("kind", "unknown")
                if kind in {"choice", "statement", "sub_question"} and ob.get("label") != gb.get("label"):
                    errors.append(_error(
                        "block_number_or_label_mismatch", f"block[{gi}]",
                        {"kind": kind, "ocr": ob.get("label"), "reference": gb.get("label")},
                    ))
        elif tag == "replace":
            # Pair replacements at the same local position to distinguish a lost
            # table/figure rendering from a genuinely inserted or deleted block.
            paired = min(i2 - i1, j2 - j1)
            for offset in range(paired):
                oi, gi = i1 + offset, j1 + offset
                ob, gb = ocr_blocks[oi], gold_blocks[gi]
                ok, gk = ob.get("kind", "unknown"), gb.get("kind", "unknown")
                if gk == "table" and ok != "table":
                    errors.append(_error("table_not_rendered", f"block[{gi}]", {"ocr_kind": ok, "reference_kind": "table"}))
                elif gk == "figure" and ok != "figure":
                    errors.append(_error("figure_missing_or_misrendered", f"block[{gi}]", {"ocr_kind": ok, "reference_kind": "figure"}))
                elif ok == "table" and gk != "table":
                    errors.append(_error("unexpected_table", f"block[{oi}]", {"ocr_kind": "table", "reference_kind": gk}))
                elif ok == "figure" and gk != "figure":
                    errors.append(_error("unexpected_figure", f"block[{oi}]", {"ocr_kind": "figure", "reference_kind": gk}))
                else:
                    errors.append(_error("block_type_or_order_mismatch", f"block[{gi}]", {"ocr_kind": ok, "reference_kind": gk}))
                if ok == gk and ok in {"choice", "statement", "sub_question"} and ob.get("label") != gb.get("label"):
                    errors.append(_error(
                        "block_number_or_label_mismatch", f"block[{gi}]",
                        {"kind": gk, "ocr": ob.get("label"), "reference": gb.get("label")},
                    ))
            for oi in range(i1 + paired, i2):
                block = ocr_blocks[oi]
                kind = block.get("kind", "unknown")
                errors.append(_error("extra_block_in_ocr", f"block[{oi}]", {"kind": kind}))
            for gi in range(j1 + paired, j2):
                block = gold_blocks[gi]
                kind = block.get("kind", "unknown")
                name = "table_missing_in_ocr" if kind == "table" else "figure_missing_in_ocr" if kind == "figure" else "block_missing_in_ocr"
                errors.append(_error(name, f"block[{gi}]", {"kind": kind}))
        elif tag == "delete":
            for oi in range(i1, i2):
                block = ocr_blocks[oi]
                kind = block.get("kind", "unknown")
                name = "unexpected_table" if kind == "table" else "unexpected_figure" if kind == "figure" else "extra_block_in_ocr"
                errors.append(_error(name, f"block[{oi}]", {"kind": kind}))
        else:  # insert: block exists in reference but is absent from OCR
            for gi in range(j1, j2):
                block = gold_blocks[gi]
                kind = block.get("kind", "unknown")
                name = "table_missing_in_ocr" if kind == "table" else "figure_missing_in_ocr" if kind == "figure" else "block_missing_in_ocr"
                errors.append(_error(name, f"block[{gi}]", {"kind": kind}))

    # A figure can be present but shifted relative to neighboring block kinds.
    ocr_figures = [i for i, block in enumerate(ocr_blocks) if block.get("kind") == "figure"]
    gold_figures = [i for i, block in enumerate(gold_blocks) if block.get("kind") == "figure"]
    if len(ocr_figures) == len(gold_figures):
        for oi, gi in zip(ocr_figures, gold_figures):
            oprev = ocr_blocks[oi - 1].get("kind") if oi else None
            gprev = gold_blocks[gi - 1].get("kind") if gi else None
            onext = ocr_blocks[oi + 1].get("kind") if oi + 1 < len(ocr_blocks) else None
            gnext = gold_blocks[gi + 1].get("kind") if gi + 1 < len(gold_blocks) else None
            if (oprev, onext) != (gprev, gnext):
                errors.append(_error("figure_misplaced", f"block[{gi}]", {"ocr_index": oi, "reference_index": gi}))

    # Stable de-duplication keeps reports compact when a replacement implies
    # both a type error and a figure/table placement observation.
    unique = []
    seen = set()
    for item in errors:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _question_similarity(ocr_question, gold_question):
    """One fuzzy score over the full question text, with media placeholders."""
    def view(question):
        """Build comparable question text while keeping block kinds and labels."""
        parts = [str(question.get("stem", ""))]
        for block in question.get("blocks", []):
            kind = block.get("kind")
            if kind == "figure" or block.get("figure"):
                content = "<FIGURE>"
            elif kind == "table":
                content = "<TABLE>"
            else:
                content = str(block.get("text", ""))
            parts.append(f"<{kind}:{block.get('label', '')}>{content}")
        return " ".join(parts)
    return round(similarity(view(ocr_question), view(gold_question)), 4)


def _run_reports(ocr_path, gold_path, *, min_similarity=0.35):
    """Build compact similarity rows and separate structural findings."""
    ocr_document = load_segmentation(ocr_path)
    gold_document = load_segmentation(gold_path)
    ocr_questions = flatten_questions(ocr_document)
    gold_questions = flatten_questions(gold_document)
    pair_ops = align_question_pairs(ocr_questions, gold_questions, min_similarity=min_similarity)
    try:
        from .unit_matching import align_documents
    except ImportError:
        from unit_matching import align_documents
    child_rows = align_documents(ocr_document, gold_document, min_similarity=min_similarity)
    similarities, structures = [], []

    for pair in pair_ops:
        operation, oi, gi = pair["operation"], pair["ocr_index"], pair["gold_index"]
        ocr_q = ocr_questions[oi] if oi is not None else None
        gold_q = gold_questions[gi] if gi is not None else None
        child_rows_for_pair = [
            row for row in child_rows
            if (ocr_q and row.get("ocr_unit_id") and row["ocr_unit_id"].startswith(ocr_q["_unit_id"] + "."))
            or (gold_q and row.get("gold_unit_id") and row["gold_unit_id"].startswith(gold_q["_unit_id"] + "."))
        ]
        child_methods = Counter(row["match_method"] for row in child_rows_for_pair)
        child_comparison_methods = Counter(row.get("comparison_method", "unknown") for row in child_rows_for_pair)
        child_fallbacks = [
            {"ocr_unit_id": row.get("ocr_unit_id"), "gold_unit_id": row.get("gold_unit_id"), "method": row["match_method"]}
            for row in child_rows_for_pair if row.get("used_parent_fallback")
        ]
        similarities.append({
            "ocr_question": ocr_q.get("number") if ocr_q else None,
            "reference_question": gold_q.get("number") if gold_q else None,
            "ocr_ordinal": ocr_q.get("_ordinal") if ocr_q else None,
            "reference_ordinal": gold_q.get("_ordinal") if gold_q else None,
            "similarity": _question_similarity(ocr_q, gold_q) if ocr_q and gold_q else 0.0,
            "match_method": pair["match_method"],
            "comparison_method": pair.get("comparison_method", pair["match_method"]),
            "unit_match_methods": dict(sorted(child_methods.items())),
            "unit_comparison_methods": dict(sorted(child_comparison_methods.items())),
            "used_parent_fallback": pair["used_parent_fallback"] or bool(child_fallbacks),
            "parent_fallback_units": child_fallbacks,
            "similarity_backend": SIMILARITY_BACKEND,
        })
        if operation == "ocr_only":
            errors = [_error("extra_question_in_ocr", "question", {"ocr_number": ocr_q.get("number")})]
        elif operation == "gold_only":
            errors = [_error("question_missing_in_ocr", "question", {"reference_number": gold_q.get("number")})]
        else:
            errors = detect_structure_errors(ocr_q, gold_q)
        if errors:
            structures.append({
                "ocr_question": ocr_q.get("number") if ocr_q else None,
                "reference_question": gold_q.get("number") if gold_q else None,
                "ocr_ordinal": ocr_q.get("_ordinal") if ocr_q else None,
                "reference_ordinal": gold_q.get("_ordinal") if gold_q else None,
                "match_method": pair["match_method"],
                "comparison_method": pair.get("comparison_method", pair["match_method"]),
                "used_parent_fallback": pair["used_parent_fallback"],
                "errors": errors,
            })
    return similarities, structures


def run_alignment(ocr_path, gold_path, *, min_similarity=0.35):
    """Compatibility entry point returning the compact similarity rows."""
    return _run_reports(ocr_path, gold_path, min_similarity=min_similarity)[0]


def write_reports(ocr_path, gold_path,
                 alignment_path=PROJECT_ROOT / "paper/results/reports/alignment_results.json",
                 structure_path=PROJECT_ROOT / "paper/results/reports/structure_results.json", *,
                 min_similarity=0.35):
    """Write question similarity and structural findings for two documents."""
    similarities, structures = _run_reports(ocr_path, gold_path, min_similarity=min_similarity)
    alignment_path, structure_path = Path(alignment_path), Path(structure_path)
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    structure_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_path.write_text(json.dumps(similarities, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter(error["type"] for item in structures for error in item["errors"])
    report = {"question_pair_count": len(similarities), "error_counts": dict(sorted(counts.items())), "questions": structures}
    Path(structure_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return similarities, report


def main(ocr_segmentation_path=PROJECT_ROOT / "paper/results/segmentations/ocr/normalized.json",
         gold_segmentation_path=PROJECT_ROOT / "paper/results/segmentations/reference/normalized.json",
         alignment_output_path=PROJECT_ROOT / "paper/results/reports/alignment_results.json",
         structure_output_path=PROJECT_ROOT / "paper/results/reports/structure_results.json", *, min_similarity=0.35):
    """Write structure reports from editable function arguments."""
    similarities, report = write_reports(
        ocr_segmentation_path, gold_segmentation_path,
        alignment_output_path, structure_output_path,
        min_similarity=min_similarity,
    )
    print(f"Wrote {len(similarities)} similarity rows and {sum(report['error_counts'].values())} structural findings")
    return similarities, report


if __name__ == "__main__":
    main()
