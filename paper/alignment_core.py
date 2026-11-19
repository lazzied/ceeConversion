"""Shared fuzzy sequence matching and explicit fallback metadata."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata

try:
    from rapidfuzz import fuzz
except ImportError:  # The dependency is declared; keep reports usable in lean envs.
    fuzz = None


SIMILARITY_BACKEND = "rapidfuzz.fuzz.ratio" if fuzz is not None else "difflib.SequenceMatcher"


def matching_stage(left, right):
    """Say whether two strings match exactly, after simple cleanup, or fuzzily."""
    left, right = str(left or ""), str(right or "")
    if left == right:
        return "exact_text"
    key = lambda text: re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    if key(left) == key(right):
        return "normalized_text"
    return "similarity"


def similarity(left, right):
    """Give two strings a similarity score from zero (different) to one (same)."""
    left, right = str(left or ""), str(right or "")
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if fuzz is not None:
        return fuzz.ratio(left, right) / 100.0
    return SequenceMatcher(a=left, b=right, autojunk=False).ratio()


def monotonic_alignment(left, right, score_fn, *, gap_penalty=-0.35, min_similarity=0.0):
    """Line up two ordered lists while allowing items to be missing or extra.

    ``score_fn`` compares a possible pair. Pairs below ``min_similarity`` are
    treated as poor matches, while ``gap_penalty`` controls how willing the
    aligner is to leave an item unmatched. The returned indexes follow the
    original order of both lists.
    """
    n, m = len(left), len(right)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], trace[i][0] = i * gap_penalty, "left_only"
    for j in range(1, m + 1):
        dp[0][j], trace[0][j] = j * gap_penalty, "right_only"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = score_fn(left[i - 1], right[j - 1])
            match = dp[i - 1][j - 1] + (2 * score - 1 if score >= min_similarity else -1.0)
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty
            best = max(match, delete, insert)
            dp[i][j] = best
            trace[i][j] = "match" if best == match else ("left_only" if best == delete else "right_only")

    operations = []
    i, j = n, m
    while i or j:
        operation = trace[i][j]
        if operation == "match":
            operations.append((operation, i - 1, j - 1))
            i -= 1
            j -= 1
        elif operation == "left_only":
            operations.append((operation, i - 1, None))
            i -= 1
        else:
            operations.append((operation, None, j - 1))
            j -= 1
    return list(reversed(operations))


def align_questions(ocr_questions, gold_questions, *, min_similarity=0.35):
    """Pair OCR and gold questions in reading order.

    Pairing uses question text and number. A skipped pair may be recovered by
    printed number, and the recovery method is kept on the result row.
    """
    def score(ocr, gold):
        """Score two question stems, giving a small bonus for equal numbers."""
        stage = matching_stage(ocr.get("stem", ""), gold.get("stem", ""))
        text_score = 1.0 if stage in {"exact_text", "normalized_text"} else similarity(ocr.get("stem", ""), gold.get("stem", ""))
        if ocr.get("number") == gold.get("number"):
            return min(1.0, text_score + 0.12)
        return text_score

    raw_ops = monotonic_alignment(
        ocr_questions, gold_questions, score, min_similarity=min_similarity
    )
    pairs = []
    unmatched_ocr, unmatched_gold = [], []
    for operation, oi, gi in raw_ops:
        if operation == "match":
            ocr, gold = ocr_questions[oi], gold_questions[gi]
            pairs.append({
                "operation": "match", "ocr_index": oi, "gold_index": gi,
                "similarity": similarity(ocr.get("stem", ""), gold.get("stem", "")),
                "match_method": "sequence_fuzzy", "used_parent_fallback": False,
                "comparison_method": matching_stage(ocr.get("stem", ""), gold.get("stem", "")),
            })
        elif operation == "left_only":
            unmatched_ocr.append(oi)
        else:
            unmatched_gold.append(gi)

    # Recover a skipped question only when its printed number agrees and the
    # stems have a minimum amount of evidence. It remains visibly a fallback.
    for oi in list(unmatched_ocr):
        candidates = [
            gi for gi in unmatched_gold
            if ocr_questions[oi].get("number") == gold_questions[gi].get("number")
            and similarity(ocr_questions[oi].get("stem", ""), gold_questions[gi].get("stem", "")) >= min_similarity
        ]
        if not candidates:
            continue
        gi = max(candidates, key=lambda index: similarity(
            ocr_questions[oi].get("stem", ""), gold_questions[index].get("stem", "")
        ))
        pairs.append({
            "operation": "match", "ocr_index": oi, "gold_index": gi,
            "similarity": similarity(ocr_questions[oi].get("stem", ""), gold_questions[gi].get("stem", "")),
            "match_method": "question_number_fallback", "used_parent_fallback": False,
            "comparison_method": matching_stage(ocr_questions[oi].get("stem", ""), gold_questions[gi].get("stem", "")),
        })
        unmatched_ocr.remove(oi)
        unmatched_gold.remove(gi)

    for oi in unmatched_ocr:
        pairs.append({
            "operation": "ocr_only", "ocr_index": oi, "gold_index": None,
            "similarity": 0.0, "match_method": "unmatched", "used_parent_fallback": False,
            "comparison_method": "unmatched",
        })
    for gi in unmatched_gold:
        pairs.append({
            "operation": "gold_only", "ocr_index": None, "gold_index": gi,
            "similarity": 0.0, "match_method": "unmatched", "used_parent_fallback": False,
            "comparison_method": "unmatched",
        })

    # Restore reading order after fallback pairs were appended.
    pairs.sort(key=lambda pair: (
        min(pair["ocr_index"] if pair["ocr_index"] is not None else 10**9,
            pair["gold_index"] if pair["gold_index"] is not None else 10**9),
        pair["ocr_index"] if pair["ocr_index"] is not None else 10**9,
    ))
    return pairs
