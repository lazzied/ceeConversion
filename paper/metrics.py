"""
metrics.py

Evaluation metrics for OCR-vs-Gold exam segmentation.

This module reuses the alignment pipeline from alignment.py:
    - load_segmentation()
    - flatten_questions()
    - run_alignment()
    - block_text()

Metrics:
    - CER: Character Error Rate (suitable for OCR text, including math text)
    - WER: Word Error Rate (reported only for Latin-script prose)
    - TER: Token Error Rate using a multilingual / LaTeX-aware tokenizer

Run:
    python metrics.py paper/ocr_exam_segmented.json paper/exam_segmented.json

Or import:
    from metrics import evaluate_files
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

# Reuse the existing alignment implementation. Keep metrics.py in the
# same directory as alignment.py, or make the project directory a package.
from block_alignment import run_alignment


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?|\d+(?:[.,]\d+)?"
)

# Order matters: LaTeX commands are captured before individual symbols.
_LATEX_TOKEN_RE = re.compile(
    r"""\\[a-zA-Z]+|\\.|[A-Za-zÀ-ÖØ-öø-ÿ]+|[0-9]+(?:[.,][0-9]+)?|"""
    r"""[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|"""
    r"""[{}()\[\]]|[^\s]"""
)


def _normalise_text(text: object) -> str:
    """Convert a value to comparable text without changing its content."""
    if text is None:
        return ""
    return str(text)


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> dict:
    """
    Return Levenshtein edit counts transforming reference (Gold)
    into hypothesis (OCR).

    S = substitutions, D = deletions from Gold, I = insertions in OCR.
    """
    n = len(reference)
    m = len(hypothesis)

    # Each cell stores (cost, operation). Operations are:
    # M=match, S=substitution, D=deletion, I=insertion.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        trace[i][0] = "D"

    for j in range(1, m + 1):
        dp[0][j] = j
        trace[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                candidates = [
                    (dp[i - 1][j - 1], "M", 0),
                    (dp[i - 1][j] + 1, "D", 1),
                    (dp[i][j - 1] + 1, "I", 2),
                ]
            else:
                candidates = [
                    (dp[i - 1][j - 1] + 1, "S", 0),
                    (dp[i - 1][j] + 1, "D", 1),
                    (dp[i][j - 1] + 1, "I", 2),
                ]

            # Stable tie-breaking: diagonal, deletion, insertion.
            cost, operation, _ = min(candidates, key=lambda item: (item[0], item[2]))
            dp[i][j] = cost
            trace[i][j] = operation

    substitutions = deletions = insertions = 0
    i, j = n, m

    while i > 0 or j > 0:
        operation = trace[i][j]

        if operation == "M":
            i -= 1
            j -= 1
        elif operation == "S":
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
            raise RuntimeError("Invalid edit-distance traceback state.")

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": substitutions + deletions + insertions,
        "edit_distance": dp[n][m],
        "reference_length": n,
    }


def _aggregate_error_rate(pairs: Iterable[tuple[str, str]], tokenizer) -> dict:
    """Aggregate edit counts over pairs; each pair is (OCR, Gold)."""
    totals = defaultdict(int)

    for ocr_text, gold_text in pairs:
        hypothesis = tokenizer(_normalise_text(ocr_text))
        reference = tokenizer(_normalise_text(gold_text))
        counts = edit_counts(reference, hypothesis)

        for key in (
            "substitutions",
            "deletions",
            "insertions",
            "errors",
            "edit_distance",
            "reference_length",
        ):
            totals[key] += counts[key]

    denominator = totals["reference_length"]
    rate = totals["errors"] / denominator if denominator else None

    return {
        "rate": round(rate, 6) if rate is not None else None,
        "error_percent": round(rate * 100, 3) if rate is not None else None,
        "substitutions": totals["substitutions"],
        "deletions": totals["deletions"],
        "insertions": totals["insertions"],
        "total_errors": totals["errors"],
        "reference_length": denominator,
    }


def character_tokens(text: str) -> list[str]:
    """One Unicode code point per token (the unit used by this CER)."""
    return list(text)


def latex_aware_tokens(text: str) -> list[str]:
    """
    Tokenize mixed multilingual text and LaTeX.

    Examples:
        \\frac -> one token
        { } ^ + = -> individual tokens
        Latin words -> one token per word
        Chinese/Japanese/Korean -> one character per token
        numbers -> one token per numeric sequence
    """
    return _LATEX_TOKEN_RE.findall(text)


def latin_words(text: str) -> list[str]:
    """Simple word tokenizer for Latin-script prose; punctuation is ignored."""
    return _LATIN_WORD_RE.findall(text.lower())


def _is_latin_prose(text: str) -> bool:
    """
    WER is only included for text that appears to be Latin-script prose.
    CJK text and LaTeX/math-heavy text are excluded from WER aggregation.
    """
    if not text.strip() or _CJK_RE.search(text):
        return False

    if "\\" in text:
        return False

    math_symbols = sum(1 for char in text if char in "{}^_=<>∑∫√≤≥±×÷")
    visible = sum(1 for char in text if not char.isspace())

    return visible > 0 and math_symbols / visible < 0.10


def _collect_metric_pairs(alignment_results: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """
    Extract aligned OCR/Gold text pairs from alignment.py's result format.

    Stems are evaluated separately from blocks. Block pairs include
    unmatched blocks as OCR-vs-empty or empty-vs-Gold so omissions and
    extra blocks contribute insertion/deletion errors.
    """
    stem_pairs = []
    block_pairs = []

    for question in alignment_results:
        ocr_stem = _normalise_text(question.get("ocr_stem"))
        gold_stem = _normalise_text(question.get("gold_stem"))

        if ocr_stem or gold_stem:
            stem_pairs.append((ocr_stem, gold_stem))

        block_result = question.get("block_alignment")
        if not block_result:
            continue

        for block in block_result.get("alignment", []):
            ocr_text = _normalise_text(block.get("ocr_text"))
            gold_text = _normalise_text(block.get("gold_text"))

            if ocr_text or gold_text:
                block_pairs.append((ocr_text, gold_text))

    return {
        "stems": stem_pairs,
        "blocks": block_pairs,
        "all": stem_pairs + block_pairs,
    }


def _metric_group(pairs: list[tuple[str, str]]) -> dict:
    cer = _aggregate_error_rate(pairs, character_tokens)
    token_error_rate = _aggregate_error_rate(pairs, latex_aware_tokens)

    # WER is deliberately limited to Latin-script prose. It is not a
    # suitable metric for Chinese text or raw mathematical LaTeX.
    latin_pairs = [
        (ocr, gold)
        for ocr, gold in pairs
        if _is_latin_prose(ocr) or _is_latin_prose(gold)
    ]
    wer = _aggregate_error_rate(latin_pairs, latin_words)

    return {
        "cer": cer,
        "token_error_rate": token_error_rate,
        "wer_latin_prose_only": wer,
        "pair_count": len(pairs),
        "wer_pair_count": len(latin_pairs),
    }


def evaluate_alignment(alignment_results: list[dict]) -> dict:
    """
    Calculate CER, TER and applicable WER from run_alignment() results.
    """
    pairs = _collect_metric_pairs(alignment_results)

    return {
        "metric_definitions": {
            "CER": "(character substitutions + deletions + insertions) / Gold characters",
            "Token Error Rate": "(token substitutions + deletions + insertions) / Gold tokens, using this module multilingual/LaTeX tokenizer",
            "WER": "(word substitutions + deletions + insertions) / Gold words; Latin prose only",
            "direction": "OCR is the hypothesis; Gold is the reference.",
            "note": (
                "CER/token error rate are aggregated over stems and aligned blocks. "
                "Unmatched blocks are compared with an empty string. "
                "WER excludes CJK and LaTeX/math-heavy pairs."
            ),
        },
        "overall": _metric_group(pairs["all"]),
        "question_stems": _metric_group(pairs["stems"]),
        "content_blocks": _metric_group(pairs["blocks"]),
    }


def evaluate_files(ocr_path: str | Path, gold_path: str | Path) -> dict:
    """
    Run the existing alignment first, then calculate metrics from its
    matched OCR/Gold stems and blocks.
    """
    alignment_results = run_alignment(str(ocr_path), str(gold_path))
    metrics = evaluate_alignment(alignment_results)

    return {
        "alignment_result_count": len(alignment_results),
        "metrics": metrics,
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python metrics.py "
            "<ocr_segmentation.json> <gold_segmentation.json>"
        )
        raise SystemExit(2)

    report = evaluate_files(sys.argv[1], sys.argv[2])

    output_path = Path("metrics_results.json")
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Alignment pairs evaluated: {report['alignment_result_count']}")
    print("\nOVERALL")
    print(json.dumps(report["metrics"]["overall"], indent=2))
    print("\nQUESTION STEMS")
    print(json.dumps(report["metrics"]["question_stems"], indent=2))
    print("\nCONTENT BLOCKS")
    print(json.dumps(report["metrics"]["content_blocks"], indent=2))
    print(f"\nFull metrics report saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
