
import json
import re
import unicodedata
from collections import Counter
from rapidfuzz import fuzz


# ---------------------------------------------------------
# 1. Load and flatten the segmentations
# ---------------------------------------------------------

def load_segmentation(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_questions(document):
    """
    Flatten all sections while preserving reading order.
    Section boundaries are deliberately ignored.
    """
    questions = []

    for section_index, section in enumerate(document["sections"]):
        for question in section["questions"]:
            questions.append({
                **question,
                "_section_index": section_index,
                "_section_title": section["title"],
            })

    return questions



def similarity(text_a, text_b):
    a = text_a
    b = text_b

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


# ---------------------------------------------------------
# 3. Generic monotonic sequence alignment
# ---------------------------------------------------------

def monotonic_alignment(
    left,
    right,
    similarity_fn,
    gap_penalty=-0.35,
):
    """
    Needleman-Wunsch-style global sequence alignment.

    Returns:
        [
            ("match", left_index, right_index),
            ("left_only", left_index, None),
            ("right_only", None, right_index)
        ]
    """

    n = len(left)
    m = len(right)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * gap_penalty
        trace[i][0] = "left_only"

    for j in range(1, m + 1):
        dp[0][j] = j * gap_penalty
        trace[0][j] = "right_only"

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            sim = similarity_fn(left[i - 1], right[j - 1])

            # Similarity score, adjusted by the cost of matching.
            match_score = 2 * sim - 1.0

            match = dp[i - 1][j - 1] + match_score
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty

            best = max(match, delete, insert)

            dp[i][j] = best

            if best == match:
                trace[i][j] = "match"
            elif best == delete:
                trace[i][j] = "left_only"
            else:
                trace[i][j] = "right_only"

    # Backtracking.
    alignment = []
    i, j = n, m

    while i > 0 or j > 0:
        operation = trace[i][j]

        if operation == "match":
            alignment.append(("match", i - 1, j - 1))
            i -= 1
            j -= 1

        elif operation == "left_only":
            alignment.append(("left_only", i - 1, None))
            i -= 1

        elif operation == "right_only":
            alignment.append(("right_only", None, j - 1))
            j -= 1

    return list(reversed(alignment))


# ---------------------------------------------------------
# 4. Question-level similarity and alignment
# ---------------------------------------------------------

def question_similarity(ocr_question, gold_question):
    stem_score = similarity(
        ocr_question.get("stem", ""),
        gold_question.get("stem", ""),
    )

    # Question numbers are strong anchors, but not absolute.
    ocr_number = ocr_question.get("number")
    gold_number = gold_question.get("number")

    if ocr_number == gold_number:
        number_bonus = 0.45
    else:
        number_bonus = -0.15

    return stem_score, number_bonus


def align_questions(ocr_questions, gold_questions):
    def score(q1, q2):
        stem_score, number_bonus = question_similarity(q1, q2)
        return max(0.0, min(1.0, stem_score + number_bonus / 2))

    alignment = monotonic_alignment(
        ocr_questions,
        gold_questions,
        score,
    )

    results = []

    for operation, oi, gi in alignment:

        if operation == "left_only":
            results.append({
                "status": "unaligned",
                "ocr_question": ocr_questions[oi],
                "gold_question": None,
                "similarity": 0.0,
            })
            continue

        if operation == "right_only":
            results.append({
                "status": "unaligned",
                "ocr_question": None,
                "gold_question": gold_questions[gi],
                "similarity": 0.0,
            })
            continue

        ocr_q = ocr_questions[oi]
        gold_q = gold_questions[gi]

        sim, _ = question_similarity(ocr_q, gold_q)

        if sim >= 0.60:
            status = "aligned"
        elif sim >= 0.30:
            status = "weak"
        else:
            status = "unaligned"

        results.append({
            "status": status,
            "ocr_question": ocr_q,
            "gold_question": gold_q,
            "similarity": round(sim, 3),
        })

    return results


# ---------------------------------------------------------
# 5. Convert blocks into comparable text
# ---------------------------------------------------------

def block_text(block):
    parts = []

    label = block.get("label", "")
    text = block.get("text", "")
    figure = block.get("figure")

    if label:
        parts.append(label)

    if text:
        parts.append(text)

    if figure:
        # Do not compare image filenames:
        # the OCR and gold pipelines may generate different paths.
        parts.append("<FIGURE>")

    return " ".join(parts)


def block_similarity(block_a, block_b):
    return similarity(
        block_text(block_a),
        block_text(block_b),
    )


# ---------------------------------------------------------
# 6. Detect structural incompatibility
# ---------------------------------------------------------

def block_structure_compatible(ocr_question, gold_question):
    ocr_blocks = ocr_question.get("blocks", [])
    gold_blocks = gold_question.get("blocks", [])

    ocr_types = Counter(
        block.get("kind", "unknown")
        for block in ocr_blocks
    )

    gold_types = Counter(
        block.get("kind", "unknown")
        for block in gold_blocks
    )

    return ocr_types == gold_types


# ---------------------------------------------------------
# 7. Align blocks inside a matched question
# ---------------------------------------------------------

def align_blocks(ocr_question, gold_question):
    ocr_blocks = ocr_question.get("blocks", [])
    gold_blocks = gold_question.get("blocks", [])

    # If the segmentation structures differ, do not force
    # a one-to-one block alignment.
    if not block_structure_compatible(ocr_question, gold_question):
        return {
            "mode": "parent_fallback",
            "reason": "Different block-type distributions",
            "alignment": [],
        }

    alignment = monotonic_alignment(
        ocr_blocks,
        gold_blocks,
        block_similarity,
    )

    results = []

    for operation, oi, gi in alignment:

        if operation == "left_only":
            results.append({
                "operation": "ocr_only",
                "ocr_block": ocr_blocks[oi],
                "gold_block": None,
                "similarity": 0.0,
            })

        elif operation == "right_only":
            results.append({
                "operation": "gold_only",
                "ocr_block": None,
                "gold_block": gold_blocks[gi],
                "similarity": 0.0,
            })

        else:
            sim = block_similarity(
                ocr_blocks[oi],
                gold_blocks[gi],
            )

            if sim >= 0.60:
                status = "aligned"
            elif sim >= 0.30:
                status = "weak"
            else:
                status = "unaligned"

            results.append({
                "operation": status,
                "ocr_block": ocr_blocks[oi],
                "gold_block": gold_blocks[gi],
                "similarity": round(sim, 3),
            })

    return {
        "mode": "block_alignment",
        "reason": None,
        "alignment": results,
    }


# ---------------------------------------------------------
# 8. Run the complete alignment
# ---------------------------------------------------------

def run_alignment(ocr_path, gold_path):

    ocr_document = load_segmentation(ocr_path)
    gold_document = load_segmentation(gold_path)

    ocr_questions = flatten_questions(ocr_document)
    gold_questions = flatten_questions(gold_document)

    question_pairs = align_questions(
        ocr_questions,
        gold_questions,
    )

    final_results = []

    for pair in question_pairs:

        ocr_q = pair["ocr_question"]
        gold_q = pair["gold_question"]

        result = {
            "question_status": pair["status"],
            "ocr_number": (
                ocr_q["number"] if ocr_q else None
            ),
            "gold_number": (
                gold_q["number"] if gold_q else None
            ),
            "question_similarity": pair["similarity"],
            "block_alignment": None,
        }

        if (
            ocr_q is not None
            and gold_q is not None
            and pair["status"] != "unaligned"
        ):
            result["block_alignment"] = align_blocks(
                ocr_q,
                gold_q,
            )

        final_results.append(result)

    return final_results


# ---------------------------------------------------------
# 9. Display a readable report
# ---------------------------------------------------------

def print_report(results):

    for result in results:

        print(
            f"OCR Q{result['ocr_number']} "
            f"<-> GOLD Q{result['gold_number']} | "
            f"{result['question_status']} | "
            f"similarity={result['question_similarity']}"
        )

        block_result = result["block_alignment"]

        if block_result is None:
            continue

        if block_result["mode"] == "parent_fallback":
            print(
                "    PARENT FALLBACK:",
                block_result["reason"],
            )
            continue

        for block in block_result["alignment"]:
            print(
                "    ",
                block["operation"],
                f"similarity={block['similarity']}",
            )


if __name__ == "__main__":

    results = run_alignment(
        "paper/ocr_exam_segmented.json",
        "paper/exam_segmented.json",

    )

    print_report(results)

    with open(
        "alignment_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )
