"""Question similarity and structural-error detection for exam OCR."""

from __future__ import annotations

import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path



def load_segmentation(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def flatten_questions(document):
    """Flatten sections in reading order while retaining section metadata."""
    return [
        {**question, "_section_index": section_index, "_section_title": section["title"]}
        for section_index, section in enumerate(document.get("sections", []))
        for question in section.get("questions", [])
    ]


def similarity(text_a, text_b):
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    return SequenceMatcher(a=str(text_a), b=str(text_b), autojunk=False).ratio()


def monotonic_alignment(left, right, similarity_fn, gap_penalty=-0.35):
    """Needleman-Wunsch sequence alignment. Return match/left_only/right_only ops."""
    n, m = len(left), len(right)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], trace[i][0] = i * gap_penalty, "left_only"
    for j in range(1, m + 1):
        dp[0][j], trace[0][j] = j * gap_penalty, "right_only"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + 2 * similarity_fn(left[i - 1], right[j - 1]) - 1
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty
            best = max(match, delete, insert)
            dp[i][j] = best
            trace[i][j] = "match" if best == match else ("left_only" if best == delete else "right_only")

    result = []
    i, j = n, m
    while i or j:
        op = trace[i][j]
        if op == "match":
            result.append((op, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif op == "left_only":
            result.append((op, i - 1, None))
            i -= 1
        else:
            result.append((op, None, j - 1))
            j -= 1
    return list(reversed(result))


def question_alignment_score(ocr_question, gold_question):
    """Stem-plus-number score used only to identify corresponding questions."""
    stem_score = similarity(ocr_question.get("stem", ""), gold_question.get("stem", ""))
    number_bonus = 0.45 if ocr_question.get("number") == gold_question.get("number") else -0.15
    return stem_score, max(0.0, min(1.0, stem_score + number_bonus / 2))


def align_questions(ocr_questions, gold_questions):
    return monotonic_alignment(
        ocr_questions,
        gold_questions,
        lambda ocr, gold: question_alignment_score(ocr, gold)[1],
    )


def _block_signature(block):
    # Labels are compared as their own structural property so a changed
    # subquestion/choice number is reported as a numbering error.
    return block.get("kind", "unknown")


def _error(error_type, location, detail):
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


def _run_reports(ocr_path, gold_path):
    """Build compact similarity rows and separate structural findings."""
    ocr_questions = flatten_questions(load_segmentation(ocr_path))
    gold_questions = flatten_questions(load_segmentation(gold_path))
    pair_ops = align_questions(ocr_questions, gold_questions)
    similarities, structures = [], []

    for operation, oi, gi in pair_ops:
        ocr_q = ocr_questions[oi] if oi is not None else None
        gold_q = gold_questions[gi] if gi is not None else None
        similarities.append({
            "ocr_question": ocr_q.get("number") if ocr_q else None,
            "reference_question": gold_q.get("number") if gold_q else None,
            "similarity": _question_similarity(ocr_q, gold_q) if ocr_q and gold_q else 0.0,
        })
        if operation == "left_only":
            errors = [_error("extra_question_in_ocr", "question", {"ocr_number": ocr_q.get("number")})]
        elif operation == "right_only":
            errors = [_error("question_missing_in_ocr", "question", {"reference_number": gold_q.get("number")})]
        else:
            errors = detect_structure_errors(ocr_q, gold_q)
        if errors:
            structures.append({
                "ocr_question": ocr_q.get("number") if ocr_q else None,
                "reference_question": gold_q.get("number") if gold_q else None,
                "errors": errors,
            })
    return similarities, structures


def run_alignment(ocr_path, gold_path):
    """Compatibility entry point returning the compact similarity rows."""
    return _run_reports(ocr_path, gold_path)[0]


def write_reports(ocr_path, gold_path, alignment_path="alignment_results.json", structure_path="structure_results.json"):
    similarities, structures = _run_reports(ocr_path, gold_path)
    Path(alignment_path).write_text(json.dumps(similarities, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter(error["type"] for item in structures for error in item["errors"])
    report = {"question_pair_count": len(similarities), "error_counts": dict(sorted(counts.items())), "questions": structures}
    Path(structure_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return similarities, report


if __name__ == "__main__":
    similarities, report = write_reports(
        "paper/ocr_exam_segmented.json", "paper/exam_segmented.json"
    )
    print(f"Wrote {len(similarities)} similarity rows and {sum(report['error_counts'].values())} structural findings")
