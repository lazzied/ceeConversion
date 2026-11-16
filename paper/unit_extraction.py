"""Hierarchical question, sub-question, and display-equation extraction."""

from __future__ import annotations

import re
import html


DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.S)
STANDALONE_MATH = re.compile(r"^\$(?!\$)(.+)\$$", re.S)
TABLE_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.I | re.S)
TABLE_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.I | re.S)


def flatten_questions(document):
    result = []
    for section_index, section in enumerate(document.get("sections", [])):
        for question_index, question in enumerate(section.get("questions", [])):
            result.append({
                **question,
                "_section_index": section_index,
                "_section_title": section.get("title", ""),
                "_ordinal": len(result) + 1,
                "_unit_id": f"s{section_index + 1}.q{question_index + 1}",
            })
    return result


def _subquestion_records(question):
    occurrences = {}
    records = []
    for block_index, block in enumerate(question.get("blocks", [])):
        if block.get("kind") != "sub_question":
            continue
        label = str(block.get("label", ""))
        occurrences[label] = occurrences.get(label, 0) + 1
        unit_id = f"{question['_unit_id']}.sub{label}.{occurrences[label]}"
        records.append({
            "unit_type": "sub_question",
            "unit_id": unit_id,
            "parent_unit_id": question["_unit_id"],
            "label": label,
            "text": str(block.get("text", "")),
            "block_index": block_index,
            "block": block,
        })
    return records


def _equation_records(question, subquestions):
    labels = {record["label"]: record["unit_id"] for record in subquestions}
    records = []
    sources = [("stem", question.get("stem", ""), None)]
    for block_index, block in enumerate(question.get("blocks", [])):
        text = str(block.get("text", ""))
        if not text:
            continue
        label = str(block.get("follows_sub_question") or "")
        sources.append((f"block[{block_index}]", text, labels.get(label)))

    equation_index = 0
    for location, text, parent_id in sources:
        matches = list(DISPLAY_MATH.finditer(text))
        found_spans = [(m.start(), m.end(), m.group(1) if m.group(1) is not None else m.group(2)) for m in matches]
        standalone = STANDALONE_MATH.match(text.strip())
        if not found_spans and standalone:
            found_spans = [(0, len(text), standalone.group(1))]
        for start, end, equation in found_spans:
            equation_index += 1
            records.append({
                "unit_type": "display_equation",
                "unit_id": f"{question['_unit_id']}.eq{equation_index}",
                "parent_unit_id": parent_id or question["_unit_id"],
                "parent_subquestion_id": parent_id,
                "text": equation.strip(),
                "location": location,
                "char_start": start,
                "char_end": end,
            })
    return records


def question_record(question):
    return {
        "unit_type": "question",
        "unit_id": question["_unit_id"],
        "parent_unit_id": None,
        "section": question.get("_section_title", ""),
        "number": question.get("number"),
        "ordinal": question.get("_ordinal"),
        "text": str(question.get("stem", "")),
    }


def child_units(question):
    subquestions = _subquestion_records(question)
    equations = _equation_records(question, subquestions)
    return subquestions, equations


def content_units(question):
    """Leaf units used for CER: stem, subquestions, blocks, figures, table cells."""
    subquestions, equations = child_units(question)
    units = [question_record(question)]
    units.extend(subquestions)
    subquestion_indexes = {unit["block_index"] for unit in subquestions}
    for block_index, block in enumerate(question.get("blocks", [])):
        if block_index in subquestion_indexes:
            continue
        kind = block.get("kind", "unknown")
        location = f"block[{block_index}]"
        if kind == "figure" or block.get("figure"):
            figure = block.get("figure")
            path = figure.get("path", "") if isinstance(figure, dict) else (figure or "")
            units.append({
                "unit_type": "figure", "unit_id": f"{question['_unit_id']}.figure{block_index + 1}",
                "parent_unit_id": question["_unit_id"], "text": str(path), "location": location,
                "block_kind": kind,
            })
            if kind == "figure":
                continue
        if kind == "table":
            table_text = str(block.get("text", ""))
            row_widths = [
                len(TABLE_CELL.findall(row)) for row in TABLE_ROW.findall(table_text)
            ]
            units.append({
                "unit_type": "table", "unit_id": f"{question['_unit_id']}.table{block_index + 1}",
                "parent_unit_id": question["_unit_id"], "text": "<table>", "location": location,
                "table_row_count": len(row_widths), "table_row_widths": row_widths,
            })
            for cell_index, raw_cell in enumerate(TABLE_CELL.findall(table_text)):
                cell = html.unescape(re.sub(r"<[^>]+>", "", raw_cell)).strip()
                units.append({
                    "unit_type": "table_cell", "unit_id": f"{question['_unit_id']}.table{block_index + 1}.cell{cell_index + 1}",
                    "parent_unit_id": f"{question['_unit_id']}.table{block_index + 1}", "text": cell,
                    "location": f"{location}.cell[{cell_index}]", "cell_index": cell_index,
                })
            continue
        label = str(block.get("label", ""))
        text = " ".join(part for part in (label, str(block.get("text", ""))) if part).strip()
        units.append({
            "unit_type": kind, "unit_id": f"{question['_unit_id']}.block{block_index + 1}",
            "parent_unit_id": question["_unit_id"], "label": label,
            "text": text, "location": location,
        })
    return units, equations


def document_units(document):
    """Return question records and a question-id keyed hierarchy of child units."""
    questions = flatten_questions(document)
    units = {}
    for question in questions:
        subquestions, equations = child_units(question)
        content, _ = content_units(question)
        units[question["_unit_id"]] = {
            "question": question_record(question),
            "sub_questions": subquestions,
            "display_equations": equations,
            "content_units": content,
            "source_question": question,
        }
    return questions, units
