"""Content-unit alignment and semantic/syntactic OCR error detection."""

from __future__ import annotations

import html
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from block_alignment import align_questions, flatten_questions, load_segmentation, monotonic_alignment, similarity


MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|\$(?!\$)([^$]+?)\$", re.S)
CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.I | re.S)
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
MATH_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|[A-Za-z]|[0-9]|[^\s]")
TEXT_TOKEN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|"
    r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?|[0-9]+(?:[.,][0-9]+)?|[^\s]"
)


def tokenize_text(text):
    """Chinese characters are individual tokens; whitespace is discarded."""
    return TEXT_TOKEN.findall(str(text or ""))


def tokenize_math(text):
    """LaTeX commands are atomic; braces, operators, letters, and digits split."""
    return MATH_TOKEN.findall(str(text or ""))


def character_tokens(text):
    """Character-level units with whitespace omitted."""
    return [char for char in str(text or "") if not char.isspace()]


def _text_and_math_units(text, location):
    result, cursor = [], 0
    def add_tokens(kind, value, formula=None):
        pattern = MATH_TOKEN if kind == "math" else TEXT_TOKEN
        result.extend(
            {"kind": kind, "text": token, "location": location, **({"formula": formula} if formula is not None else {})}
            for token in pattern.findall(value)
        )

    for match in MATH_SPAN.finditer(text or ""):
        before = (text or "")[cursor:match.start()]
        add_tokens("text", before)
        formula = match.group(1) if match.group(1) is not None else match.group(2)
        add_tokens("math", formula, formula=formula)
        cursor = match.end()
    tail = (text or "")[cursor:]
    add_tokens("text", tail)
    return result


def _table_cells(markup):
    return [html.unescape(re.sub(r"<[^>]+>", "", cell)).strip() for cell in CELL.findall(markup or "")]


def question_units(question):
    """Extract text, math, table-cell, and figure-reference comparison units."""
    units = _text_and_math_units(str(question.get("stem", "")), "stem")
    for index, block in enumerate(question.get("blocks", [])):
        kind = block.get("kind", "unknown")
        location = f"block[{index}]"
        if kind == "figure" or block.get("figure"):
            value = block.get("figure")
            path = value.get("path", "") if isinstance(value, dict) else (value or "")
            units.append({"kind": "figure", "text": str(path), "location": location})
            if kind != "figure" and block.get("text"):
                units.extend(_text_and_math_units(str(block.get("text", "")), location))
        elif kind == "table":
            cells = _table_cells(block.get("text", ""))
            units.append({"kind": "table", "text": "<table>", "location": location})
            for cell_index, cell in enumerate(cells):
                units.extend(_text_and_math_units(cell, f"{location}.cell[{cell_index}]"))
        else:
            label = str(block.get("label", ""))
            units.extend(_text_and_math_units(f"{label} {block.get('text', '')}".strip(), location))
    return units


def align_question_units(ocr_question, gold_question):
    """Align ordered units while matching only comparable unit types."""
    ocr_units, gold_units = question_units(ocr_question), question_units(gold_question)
    aligned = []
    def unit_similarity(left, right):
        left_cell = re.search(r"\.cell\[(\d+)\]", left["location"])
        right_cell = re.search(r"\.cell\[(\d+)\]", right["location"])
        if bool(left_cell) != bool(right_cell) or (left_cell and left_cell.group(1) != right_cell.group(1)):
            return 0.0
        if left["kind"] != right["kind"]:
            # Pair same content when OCR lost only its math delimiters. Token
            # units prevent a whole prose run from swallowing a formula.
            return similarity(left["text"], right["text"])
        if left["kind"] == "figure":
            return 1.0 if left["text"] == right["text"] else 0.0
        if left["kind"] == "table":
            return 1.0
        return similarity(left["text"], right["text"])

    for _op, oi, gi in monotonic_alignment(ocr_units, gold_units, unit_similarity):
        ocr_unit = ocr_units[oi] if oi is not None else None
        reference_unit = gold_units[gi] if gi is not None else None
        unit_type_mismatch = bool(
            ocr_unit and reference_unit and ocr_unit["kind"] != reference_unit["kind"]
        )
        aligned.append({
            "kind": "math" if any(unit and unit["kind"] == "math" for unit in (ocr_unit, reference_unit)) else (ocr_unit or reference_unit)["kind"],
            "unit_type_mismatch": unit_type_mismatch,
            "ocr": ocr_unit,
            "reference": reference_unit,
        })
    return aligned


def _tokens(unit, side, force_math=False):
    if unit is None:
        return []
    if unit["kind"] == "figure":
        # Image identity is exact: the reference path must match.
        return [f"<FIGURE:{unit['text']}>" ] if unit["text"] else []
    if unit["kind"] == "table":
        return ["<table>"]
    return tokenize_math(unit["text"]) if force_math or unit["kind"] == "math" else tokenize_text(unit["text"])


def _balanced_braces(text):
    depth = 0
    for index, char in enumerate(text):
        if char == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif char == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _classify_difference(kind, ocr, reference, unit_type_mismatch=False):
    if unit_type_mismatch:
        return "syntactic", "math_delimiter_or_unit_type_difference"
    if kind in {"text", "table_cell", "figure"}:
        return "semantic", "recognized_content_difference"
    ocr_text = (ocr.get("formula") or ocr["text"]) if ocr else ""
    ref_text = (reference.get("formula") or reference["text"]) if reference else ""
    if not _balanced_braces(ocr_text):
        return "syntactic", "unbalanced_latex_braces"
    if not _balanced_braces(ref_text):
        return "syntactic", "reference_latex_braces_unbalanced"
    ocr_tokens = tokenize_math(ocr_text)
    ref_tokens = tokenize_math(ref_text)
    diff_tokens = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=ocr_tokens, b=ref_tokens, autojunk=False).get_opcodes():
        if tag != "equal":
            diff_tokens.extend(ocr_tokens[i1:i2])
            diff_tokens.extend(ref_tokens[j1:j2])
    latex_syntax = re.compile(r"^\\[A-Za-z]+$|^[{}_^]$")
    if any(latex_syntax.match(token) for token in diff_tokens):
        return "syntactic", "latex_command_or_structure_difference"
    return "semantic", "formula_content_difference"


def _error_spans(kind, ocr, reference, unit_type_mismatch=False):
    ocr_text = ocr["text"] if ocr else ""
    ref_text = reference["text"] if reference else ""
    left = _tokens(ocr, "ocr", force_math=(kind == "math"))
    right = _tokens(reference, "reference", force_math=(kind == "math"))
    if kind == "figure":
        equal = bool(ocr_text) and ocr_text == ref_text
        if equal:
            return []
        return [{
            "category": "semantic",
            "type": "figure_reference_mismatch",
            "ocr": ocr_text or None,
            "reference": ref_text or None,
        }]
    if kind == "table":
        return [] if (ocr is not None) == (reference is not None) else [{
            "category": "syntactic", "type": "table_placeholder_missing",
            "ocr": ocr_text or None, "reference": ref_text or None,
        }]
    if left == right and not unit_type_mismatch:
        return []
    if unit_type_mismatch:
        category, error_type = "syntactic", "math_delimiter_or_unit_type_difference"
    else:
        category, error_type = _classify_difference(kind, ocr, reference)
    return [{
        "category": category,
        "type": error_type,
        "ocr": ocr_text or None,
        "reference": ref_text or None,
        "location": (ocr or reference or {}).get("location"),
    }]


def compare_question(ocr_question, gold_question):
    aligned = align_question_units(ocr_question, gold_question)
    errors = []
    unit_results = []
    for pair in aligned:
        kind, ocr, reference = pair["kind"], pair["ocr"], pair["reference"]
        type_mismatch = pair["unit_type_mismatch"]
        error_spans = _error_spans(kind, ocr, reference, type_mismatch)
        errors.extend({"unit": kind, **error} for error in error_spans)
        ocr_text = ocr["text"] if ocr else ""
        ref_text = reference["text"] if reference else ""
        unit_results.append({
            "unit": kind,
            "location": (ocr or reference or {}).get("location"),
            "character_match": character_tokens(ocr_text) == character_tokens(ref_text),
            "token_match": _tokens(ocr, "ocr", force_math=(kind == "math")) == _tokens(reference, "reference", force_math=(kind == "math")),
            "unit_type_mismatch": type_mismatch,
            "errors": error_spans,
        })
    return unit_results, errors


def run_token_alignment(ocr_path, gold_path):
    ocr_questions = flatten_questions(load_segmentation(ocr_path))
    gold_questions = flatten_questions(load_segmentation(gold_path))
    results, all_errors = [], []
    for op, oi, gi in align_questions(ocr_questions, gold_questions):
        ocr_q = ocr_questions[oi] if oi is not None else None
        gold_q = gold_questions[gi] if gi is not None else None
        units, errors = compare_question(ocr_q or {}, gold_q or {})
        result = {
            "ocr_question": ocr_q.get("number") if ocr_q else None,
            "reference_question": gold_q.get("number") if gold_q else None,
            "unit_alignments": units,
            "errors": errors,
        }
        results.append(result)
        all_errors.extend({
            "ocr_question": result["ocr_question"],
            "reference_question": result["reference_question"],
            **error,
        } for error in errors)
    counts = {
        category: sum(error["category"] == category for error in all_errors)
        for category in ("syntactic", "semantic")
    }
    return {"question_pair_count": len(results), "error_counts": counts, "errors": all_errors, "questions": results}


def write_report(ocr_path, gold_path, output_path="token_alignment_results.json"):
    report = run_token_alignment(ocr_path, gold_path)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = write_report("paper/ocr_exam_segmented.json", "paper/exam_segmented.json")
    print(f"Wrote token-alignment report: {report['error_counts']}")
