"""Data-derived OCR error taxonomy and multi-exam aggregation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .alignment_core import align_questions, similarity
    from .unit_extraction import document_units, flatten_questions
except ImportError:
    from alignment_core import align_questions, similarity
    from unit_extraction import document_units, flatten_questions


GLYPH_PAIRS = {
    frozenset(pair) for pair in (
        ("0", "O"), ("0", "o"), ("1", "l"), ("1", "I"), ("5", "S"),
        ("2", "Z"), ("8", "B"), ("6", "G"), ("rn", "m"),
    )
}
OPERATOR_PAIRS = {
    frozenset(pair) for pair in (
        ("x", "×"), ("*", "×"), ("-", "−"), ("-", "–"), ("−", "–"),
        ("/", "÷"), (".", "·"), ("<=", "≤"), (">=", "≥"), ("!=", "≠"),
    )
}
GREEK_COMMANDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
    "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi",
    "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi",
    "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
}
GREEK_CHARS = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
MATH_OPERATORS = set("+-−–×·*/÷=<>≤≥≠≈∼∝±∓")
DELIMITERS = set("{}[]()|\\$")


def _math_group(token):
    token = str(token or "")
    bare = token[1:] if token.startswith("\\") else token
    if (token.startswith("\\") and bare in GREEK_COMMANDS) or any(char in GREEK_CHARS for char in token):
        return "greek_letters"
    if token in {"^", "_"}:
        return "superscripts_subscripts"
    if token.isdigit():
        return "digits"
    if token in MATH_OPERATORS or token.startswith("\\") and bare in {"times", "cdot", "div", "le", "leq", "ge", "geq", "neq", "approx", "pm", "mp"}:
        return "operators"
    if token in DELIMITERS:
        return "braces_delimiters"
    if token.startswith("\\"):
        return "latex_commands"
    if token.isalpha():
        return "latin_letters"
    return "other_math_symbols"


def _primary_character_category(error, content_class):
    operation = error.get("operation")
    gold, ocr = str(error.get("gold") or ""), str(error.get("ocr") or "")
    if operation == "insertion":
        return "character_insertion"
    if operation == "deletion":
        return "character_deletion"
    if operation != "substitution":
        return "other_character_error"
    if frozenset((gold, ocr)) in GLYPH_PAIRS:
        return "glyph_confusion"
    if frozenset((gold, ocr)) in OPERATOR_PAIRS:
        return "operator_confusion"
    if content_class == "math":
        return "mathematical_symbol_confusion"
    return "character_substitution"


def _increment_confusion(table, category, gold, ocr):
    if gold is not None and ocr is not None and gold != ocr:
        table[category][(str(gold), str(ocr))] += 1


def _counter_report(counter):
    return dict(sorted(counter.items()))


def _confusions_report(confusions):
    return {
        category: [
            {"gold": gold, "ocr": ocr, "count": count}
            for (gold, ocr), count in pairs.most_common()
        ]
        for category, pairs in sorted(confusions.items())
    }


def _block_text(block):
    figure = block.get("figure")
    if isinstance(figure, dict):
        return str(figure.get("path", ""))
    return str(block.get("text", ""))


def _confidence_spans(data):
    """Find text spans with numeric confidence values in common OCR JSON shapes."""
    spans = []
    text_keys = {"text", "block_content", "content", "transcription", "ocr_text"}
    score_keys = {"confidence", "score", "probability", "rec_score", "text_score"}
    if isinstance(data, dict):
        text = next((data[key] for key in text_keys if isinstance(data.get(key), str) and data[key].strip()), None)
        score = next((data[key] for key in score_keys if isinstance(data.get(key), (int, float))), None)
        if text is not None and score is not None:
            score = float(score)
            if score > 1 and score <= 100:
                score /= 100
            if 0 <= score <= 1:
                spans.append((text, score))
        for value in data.values():
            spans.extend(_confidence_spans(value))
    elif isinstance(data, list):
        for value in data:
            spans.extend(_confidence_spans(value))
    unique = {}
    for text, score in spans:
        key = (re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold(), score)
        unique[key] = (text, score)
    return list(unique.values())


def _confidence_error_report(rows, confidence_data):
    spans = _confidence_spans(confidence_data)
    if not spans:
        return {
            "available": False,
            "reason": "OCR confidence is absent from the structured output or is not attached to text spans.",
        }
    by_band = defaultdict(Counter)
    mapped_errors = unmapped_errors = 0
    bands = ((0.0, 0.5, "low"), (0.5, 0.7, "medium_low"), (0.7, 0.85, "medium_high"), (0.85, 1.01, "high"))
    normalized_spans = [
        (re.sub(r"\s+", "", unicodedata.normalize("NFKC", span_text)).casefold(), confidence)
        for span_text, confidence in spans
    ]
    for row in rows:
        if row.get("operation") != "match" or not row.get("count_in_metrics", True):
            continue
        text = unicodedata.normalize("NFKC", str(row.get("ocr_text") or ""))
        text_key = re.sub(r"\s+", "", text).casefold()
        if not text_key:
            continue
        candidates = []
        for span_key, confidence in normalized_spans:
            score = 1.0 if text_key in span_key or span_key in text_key else similarity(text_key, span_key)
            candidates.append((score, confidence))
        score, confidence = max(candidates, default=(0.0, 0.0))
        if score < 0.65:
            unmapped_errors += len(row.get("character_errors", []))
            continue
        band = next(name for low, high, name in bands if low <= confidence < high)
        for error in row.get("character_errors", []):
            by_band[band][f"{error.get('category') or 'other_text'}:{error.get('operation') or 'unknown'}"] += 1
            mapped_errors += 1
    return {
        "available": True,
        "confidence_spans": len(spans),
        "mapped_character_errors": mapped_errors,
        "unmapped_character_errors": unmapped_errors,
        "by_confidence_band": {band: _counter_report(counts) for band, counts in sorted(by_band.items())},
        "mapping_note": "Unit text is matched to the closest OCR JSON span; inspect matches before treating these bands as calibrated token confidence.",
    }


def _reading_order_events(ocr_document, gold_document):
    ocr_questions = flatten_questions(ocr_document)
    gold_questions = flatten_questions(gold_document)
    pairs = align_questions(ocr_questions, gold_questions)
    events = []
    for pair in pairs:
        if pair["operation"] != "match":
            continue
        ocr_q = ocr_questions[pair["ocr_index"]]
        gold_q = gold_questions[pair["gold_index"]]
        ocr_blocks, gold_blocks = ocr_q.get("blocks", []), gold_q.get("blocks", [])
        used_gold, mapping = set(), []
        for oi, ob in enumerate(ocr_blocks):
            candidates = []
            for gi, gb in enumerate(gold_blocks):
                if gi in used_gold or ob.get("kind") != gb.get("kind"):
                    continue
                o_label, g_label = str(ob.get("label", "")), str(gb.get("label", ""))
                score = similarity(_block_text(ob), _block_text(gb))
                if o_label or g_label:
                    score = max(score, 1.0 if o_label == g_label else 0.0)
                if score >= 0.65:
                    candidates.append((score, gi))
            if candidates:
                _, gi = max(candidates, key=lambda item: (item[0], -item[1]))
                used_gold.add(gi)
                mapping.append((oi, gi))
        inversions = [
            {"ocr_blocks": [left[0], right[0]], "reference_blocks": [left[1], right[1]]}
            for index, left in enumerate(mapping)
            for right in mapping[index + 1:]
            if left[1] > right[1]
        ]
        if inversions:
            events.append({
                "category": "reading_order_error",
                "ocr_question": ocr_q.get("number"),
                "reference_question": gold_q.get("number"),
                "inversions": inversions,
                "evidence": "Similar same-kind blocks appear in a different order.",
            })
    return events


def _equation_segmentation_events(ocr_document, gold_document, rows):
    ocr_questions, ocr_tree = document_units(ocr_document)
    gold_questions, gold_tree = document_units(gold_document)
    pairs = align_questions(ocr_questions, gold_questions)
    events = []
    rows_by_unit_id = defaultdict(list)
    for row in rows:
        if row.get("unit_type") == "display_equation":
            if row.get("ocr_unit_id"):
                rows_by_unit_id[row["ocr_unit_id"]].append(row)
            if row.get("gold_unit_id"):
                rows_by_unit_id[row["gold_unit_id"]].append(row)
    for pair in pairs:
        if pair["operation"] != "match":
            continue
        oq = ocr_questions[pair["ocr_index"]]
        gq = gold_questions[pair["gold_index"]]
        ocr_eq = ocr_tree[oq["_unit_id"]]["display_equations"]
        gold_eq = gold_tree[gq["_unit_id"]]["display_equations"]
        if len(ocr_eq) == len(gold_eq):
            continue
        ocr_ids = {unit["unit_id"] for unit in ocr_eq}
        gold_ids = {unit["unit_id"] for unit in gold_eq}
        equation_rows = []
        seen_rows = set()
        for unit_id in ocr_ids | gold_ids:
            for row in rows_by_unit_id.get(unit_id, []):
                identity = (row.get("ocr_unit_id"), row.get("gold_unit_id"), row.get("operation"))
                if identity not in seen_rows:
                    seen_rows.add(identity)
                    equation_rows.append(row)
        events.append({
            "category": "equation_segmentation_candidate",
            "candidate_type": "possible_split_or_extra_equation" if len(ocr_eq) > len(gold_eq) else "possible_merge_or_missing_equation",
            "ocr_question": oq.get("number"),
            "reference_question": gq.get("number"),
            "ocr_equation_count": len(ocr_eq),
            "reference_equation_count": len(gold_eq),
            "ocr_equation_ids": sorted(ocr_ids),
            "reference_equation_ids": sorted(gold_ids),
            "alignment_rows": [
                {"ocr_unit_id": row.get("ocr_unit_id"), "gold_unit_id": row.get("gold_unit_id"), "operation": row.get("operation")}
                for row in equation_rows
            ],
            "requires_review": True,
        })
    return events


def analyze_document(rows, structure_report, ocr_document, gold_document, document_name, confidence_data=None):
    """Classify raw OCR differences into measured, reviewable categories."""
    primary = Counter()
    operations = Counter()
    by_language = defaultdict(Counter)
    by_content_type = defaultdict(Counter)
    char_confusions = defaultdict(Counter)
    math_confusions = defaultdict(Counter)
    math_token_confusions = defaultdict(Counter)
    math_error_counts = defaultdict(Counter)
    command_errors = Counter()
    command_confusions = Counter()
    eligible_rows = [
        row for row in rows
        if row.get("count_in_metrics", True)
        and row.get("operation") == "match"
        and not row.get("excluded", False)
    ]

    for row in eligible_rows:
        unit_type = row.get("unit_type", "unknown")
        for error in row.get("character_errors", []):
            operation = error.get("operation", "unknown")
            content_class = error.get("category") or "other_text"
            category = _primary_character_category(error, content_class)
            primary[category] += 1
            operations[operation] += 1
            by_language[content_class][operation] += 1
            by_content_type[unit_type][operation] += 1
            if operation == "substitution":
                _increment_confusion(char_confusions, content_class, error.get("gold"), error.get("ocr"))
                if content_class == "math":
                    group = _math_group(error.get("gold") or error.get("ocr"))
                    _increment_confusion(math_confusions, group, error.get("gold"), error.get("ocr"))
        for error in row.get("token_errors", []):
            if error.get("category") not in {"math", "math_syntax"}:
                continue
            operation = error.get("operation", "unknown")
            value = error.get("gold") or error.get("ocr") or ""
            group = _math_group(value)
            math_error_counts[group][operation] += 1
            gold, ocr = error.get("gold"), error.get("ocr")
            if isinstance(gold, str) and gold.startswith("\\") or isinstance(ocr, str) and ocr.startswith("\\"):
                command_errors[operation] += 1
                primary["latex_command_error"] += 1
                if operation == "substitution":
                    command_confusions[(str(gold or ""), str(ocr or ""))] += 1
            if operation == "substitution":
                _increment_confusion(math_token_confusions, group, gold, ocr)

    structure_counts = Counter(structure_report.get("error_counts", {}))
    reading_events = _reading_order_events(ocr_document, gold_document)
    equation_events = _equation_segmentation_events(ocr_document, gold_document, rows)
    equation_event_counts = Counter(event["candidate_type"] for event in equation_events)
    reading_structure_count = sum(
        count for error_type, count in structure_counts.items()
        if error_type in {"figure_misplaced", "block_type_or_order_mismatch"}
    )
    primary["structural_error"] += sum(structure_counts.values()) - reading_structure_count
    primary["reading_order_error"] += reading_structure_count + len(reading_events)
    primary["equation_segmentation_candidate"] += len(equation_events)

    confusions = _confusions_report(char_confusions)
    math_confusions_report = _confusions_report(math_confusions)
    documents = {
        document_name: {
            "character_error_categories": _counter_report(primary),
            "character_operations": _counter_report(operations),
            "unaligned_unit_rows": sum(
                row.get("count_in_metrics", True) and row.get("operation") != "match"
                for row in rows
            ),
            "errors_by_language_or_content_class": {key: _counter_report(value) for key, value in sorted(by_language.items())},
            "errors_by_content_type": {key: _counter_report(value) for key, value in sorted(by_content_type.items())},
            "chinese_confusions": confusions.get("cjk_text", []),
            "character_confusions_by_class": confusions,
            "mathematical_confusions_by_symbol_type": math_confusions_report,
            "math_token_confusions_by_symbol_type": _confusions_report(math_token_confusions),
            "math_token_errors_by_symbol_type": {key: _counter_report(value) for key, value in sorted(math_error_counts.items())},
            "latex_command_errors": {
                "operation_counts": _counter_report(command_errors),
                "confusions": [
                    {"gold": gold, "ocr": ocr, "count": count}
                    for (gold, ocr), count in command_confusions.most_common()
                ],
            },
            "structural_error_counts": _counter_report(structure_counts),
            "reading_order_events": reading_events,
            "equation_segmentation_candidates": equation_events,
            "equation_segmentation_counts": _counter_report(equation_event_counts),
            "confidence_analysis": _confidence_error_report(eligible_rows, confidence_data),
        }
    }
    return aggregate_documents(documents)


IMPACT_WEIGHTS = {
    "character_substitution": 1.0,
    "character_deletion": 1.0,
    "character_insertion": 1.0,
    "glyph_confusion": 1.2,
    "operator_confusion": 2.0,
    "mathematical_symbol_confusion": 1.8,
    "latex_command_error": 1.5,
    "structural_error": 2.0,
    "reading_order_error": 2.0,
    "equation_segmentation_candidate": 2.0,
}


def aggregate_documents(documents):
    """Aggregate document reports while retaining the per-document breakdown."""
    category_totals, operation_totals = Counter(), Counter()
    language_totals, content_totals, structure_totals = defaultdict(Counter), defaultdict(Counter), Counter()
    class_confusions, math_confusions, math_token_confusions, math_tokens = (
        defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    )
    command_operations, command_pairs = Counter(), Counter()
    reading_events, equation_events = [], []
    equation_event_totals = Counter()
    unaligned_rows = 0
    confidence_bands = defaultdict(Counter)
    confidence_spans = 0
    confidence_mapped = confidence_unmapped = 0
    confidence_available = False
    for document, report in documents.items():
        category_totals.update(report.get("character_error_categories", {}))
        operation_totals.update(report.get("character_operations", {}))
        structure_totals.update(report.get("structural_error_counts", {}))
        unaligned_rows += report.get("unaligned_unit_rows", 0)
        confidence = report.get("confidence_analysis", {})
        confidence_available = confidence_available or confidence.get("available", False)
        confidence_spans += confidence.get("confidence_spans", 0)
        confidence_mapped += confidence.get("mapped_character_errors", 0)
        confidence_unmapped += confidence.get("unmapped_character_errors", 0)
        for band, counts in confidence.get("by_confidence_band", {}).items():
            confidence_bands[band].update(counts)
        for key, counts in report.get("errors_by_language_or_content_class", {}).items():
            language_totals[key].update(counts)
        for key, counts in report.get("errors_by_content_type", {}).items():
            content_totals[key].update(counts)
        for key, values in report.get("character_confusions_by_class", {}).items():
            for item in values:
                class_confusions[key][(item["gold"], item["ocr"])] += item["count"]
        for key, values in report.get("mathematical_confusions_by_symbol_type", {}).items():
            for item in values:
                math_confusions[key][(item["gold"], item["ocr"])] += item["count"]
        for key, values in report.get("math_token_confusions_by_symbol_type", {}).items():
            for item in values:
                math_token_confusions[key][(item["gold"], item["ocr"])] += item["count"]
        for key, counts in report.get("math_token_errors_by_symbol_type", {}).items():
            math_tokens[key].update(counts)
        command = report.get("latex_command_errors", {})
        command_operations.update(command.get("operation_counts", {}))
        for item in command.get("confusions", []):
            command_pairs[(item["gold"], item["ocr"])] += item["count"]
        reading_events.extend({"document": document, **event} for event in report.get("reading_order_events", []))
        equation_events.extend({"document": document, **event} for event in report.get("equation_segmentation_candidates", []))
        equation_event_totals.update(report.get("equation_segmentation_counts", {}))

    # Keep counts data-driven; explicit impact weights adjust review priority.
    candidates = []
    for category, count in category_totals.items():
        if not count:
            continue
        weight = IMPACT_WEIGHTS.get(category, 1.0)
        candidates.append({
            "category": category, "count": count, "impact_weight": weight,
            "priority_score": round(count * weight, 3),
        })
    candidates.sort(key=lambda item: (-item["priority_score"], -item["count"], item["category"]))
    aggregate = {
        "document_count": len(documents),
        "character_error_categories": _counter_report(category_totals),
        "character_operations": _counter_report(operation_totals),
        "unaligned_unit_rows": unaligned_rows,
        "errors_by_language_or_content_class": {key: _counter_report(value) for key, value in sorted(language_totals.items())},
        "errors_by_content_type": {key: _counter_report(value) for key, value in sorted(content_totals.items())},
        "character_confusions_by_class": _confusions_report(class_confusions),
        "chinese_confusions": _confusions_report(class_confusions).get("cjk_text", []),
        "mathematical_confusions_by_symbol_type": _confusions_report(math_confusions),
        "math_token_confusions_by_symbol_type": _confusions_report(math_token_confusions),
        "math_token_errors_by_symbol_type": {key: _counter_report(value) for key, value in sorted(math_tokens.items())},
        "latex_command_errors": {
            "operation_counts": _counter_report(command_operations),
            "confusions": [
                {"gold": gold, "ocr": ocr, "count": count}
                for (gold, ocr), count in command_pairs.most_common()
            ],
        },
        "structural_error_counts": _counter_report(structure_totals),
        "reading_order_events": reading_events,
        "equation_segmentation_candidates": equation_events,
        "equation_segmentation_counts": _counter_report(equation_event_totals),
        "top_correction_experiment_candidates": candidates[:5],
        "priority_note": "Counts are measured from the paired data. Impact weights are explicit heuristics, not learned probabilities. Specialized findings such as LaTeX command errors can overlap with character-level counts.",
        "classification_note": "Glyph and operator labels use explicit visual/symbol equivalence rules; their frequencies are measured from this alignment, not assumed in advance.",
        "confidence_analysis": {
            "available": confidence_available,
            "confidence_spans": confidence_spans,
            "mapped_character_errors": confidence_mapped,
            "unmapped_character_errors": confidence_unmapped,
            "by_confidence_band": {band: _counter_report(counts) for band, counts in sorted(confidence_bands.items())},
            "note": "Confidence matching uses the closest OCR text span and should be audited for calibration.",
        },
    }
    return {"schema_version": 1, "documents": documents, "aggregate": aggregate}


def build_taxonomy_report(rows, structure_report, ocr_document, gold_document, document_name, confidence_data=None):
    return analyze_document(rows, structure_report, ocr_document, gold_document, document_name, confidence_data)


def aggregate_report_files(paths, output_path):
    documents = {}
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for name, result in report.get("documents", {}).items():
            if name in documents:
                base_name = f"{name}#{Path(path).parent.name}"
                name, suffix = base_name, 2
                while name in documents:
                    name, suffix = f"{base_name}_{suffix}", suffix + 1
            documents[name] = result
    combined = aggregate_documents(documents)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return combined


def main(report_paths=(Path("paper/results/error_taxonomy_results.json"),),
         output_path=Path("paper/results/error_taxonomy_corpus.json")):
    """Aggregate reports named in the function arguments; no CLI parsing."""
    result = aggregate_report_files(report_paths, output_path)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
