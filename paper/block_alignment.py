import json
from collections import Counter, defaultdict

from rapidfuzz import fuzz


# ---------------------------------------------------------
# 1. Load and flatten the segmentations
# ---------------------------------------------------------

def load_segmentation(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_questions(document):
    """Flatten all sections while preserving reading order."""
    questions = []

    for section_index, section in enumerate(document["sections"]):
        for question in section["questions"]:
            questions.append({
                **question,
                "_section_index": section_index,
                "_section_title": section["title"],
            })

    return questions


# ---------------------------------------------------------
# 2. Generic text similarity
# ---------------------------------------------------------

def similarity(text_a, text_b):
    if not text_a or not text_b:
        return 0.0

    return fuzz.ratio(str(text_a), str(text_b)) / 100.0


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
        ("match", left_index, right_index)
        ("left_only", left_index, None)
        ("right_only", None, right_index)
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
# 4. Question-level matching
# ---------------------------------------------------------

def question_alignment_score(ocr_question, gold_question):
    """
    Used only to decide which OCR question corresponds to which
    gold question. The number bonus is NOT part of the reported
    stem or combined similarity.
    """
    stem_score = similarity(
        ocr_question.get("stem", ""),
        gold_question.get("stem", ""),
    )

    ocr_number = ocr_question.get("number")
    gold_number = gold_question.get("number")

    number_bonus = 0.45 if ocr_number == gold_number else -0.15

    alignment_score = max(
        0.0,
        min(1.0, stem_score + number_bonus / 2),
    )

    return stem_score, alignment_score


def align_questions(ocr_questions, gold_questions):
    def score(q1, q2):
        _, alignment_score = question_alignment_score(q1, q2)
        return alignment_score

    return monotonic_alignment(
        ocr_questions,
        gold_questions,
        score,
    )


# ---------------------------------------------------------
# 5. Convert blocks into comparable text
# ---------------------------------------------------------

def block_text(block):
    parts = []

    label = block.get("label", "")
    text = block.get("text", "")
    figure = block.get("figure")

    if label:
        parts.append(str(label))

    if text:
        parts.append(str(text))

    # Do not compare image filenames: OCR and gold pipelines
    # may generate different paths.
    if figure:
        parts.append("<FIGURE>")

    return " ".join(parts)


def block_similarity(block_a, block_b):
    return similarity(
        block_text(block_a),
        block_text(block_b),
    )


# ---------------------------------------------------------
# 6. Align blocks by kind and calculate per-kind similarity
# ---------------------------------------------------------

def align_blocks(ocr_question, gold_question):
    """
    Align each block kind independently.

    This allows choices, statements, subquestions, and ordinary
    text blocks to be compared without requiring identical block
    type distributions.
    """
    ocr_blocks = ocr_question.get("blocks", [])
    gold_blocks = gold_question.get("blocks", [])

    ocr_by_kind = defaultdict(list)
    gold_by_kind = defaultdict(list)

    for block in ocr_blocks:
        ocr_by_kind[block.get("kind", "unknown")].append(block)

    for block in gold_blocks:
        gold_by_kind[block.get("kind", "unknown")].append(block)

    all_kinds = list(dict.fromkeys(
        [block.get("kind", "unknown") for block in ocr_blocks]
        + [block.get("kind", "unknown") for block in gold_blocks]
    ))

    all_alignments = []
    per_kind_similarity = {}

    for kind in all_kinds:
        left = ocr_by_kind[kind]
        right = gold_by_kind[kind]

        alignment = monotonic_alignment(
            left,
            right,
            block_similarity,
        )

        kind_results = []
        similarity_total = 0.0

        for operation, oi, gi in alignment:
            if operation == "left_only":
                ocr_block = left[oi]
                gold_block = None
                sim = 0.0
                status = "ocr_only"

            elif operation == "right_only":
                ocr_block = None
                gold_block = right[gi]
                sim = 0.0
                status = "gold_only"

            else:
                ocr_block = left[oi]
                gold_block = right[gi]
                sim = block_similarity(ocr_block, gold_block)

                if sim >= 0.60:
                    status = "aligned"
                elif sim >= 0.30:
                    status = "weak"
                else:
                    status = "unaligned"

            similarity_total += sim

            item = {
                "kind": kind,
                "operation": status,
                "ocr_text": block_text(ocr_block) if ocr_block else None,
                "gold_text": block_text(gold_block) if gold_block else None,
                "similarity": round(sim, 3),
                "ocr_block": ocr_block,
                "gold_block": gold_block,
            }

            kind_results.append(item)
            all_alignments.append(item)

        # Divide by the larger sequence length so missing OCR/gold
        # blocks reduce the score instead of disappearing.
        denominator = max(len(left), len(right))
        kind_score = (
            similarity_total / denominator
            if denominator
            else None
        )

        if kind_score is not None:
            per_kind_similarity[kind] = round(kind_score, 3)

    available_kind_scores = list(per_kind_similarity.values())

    content_similarity = (
        sum(available_kind_scores) / len(available_kind_scores)
        if available_kind_scores
        else None
    )

    return {
        "mode": "block_alignment_by_kind",
        "per_kind_similarity": per_kind_similarity,
        "content_similarity": (
            round(content_similarity, 3)
            if content_similarity is not None
            else None
        ),
        "alignment": all_alignments,
    }


# ---------------------------------------------------------
# 7. Combine stem and content similarities
# ---------------------------------------------------------

def combined_similarity(stem_score, content_score):
    """
    Default weighting:
        stem/content = 60% / 40%

    If the question has no comparable blocks, use stem similarity
    alone rather than treating missing content as a zero.
    """
    if content_score is None:
        return stem_score

    return 0.60 * stem_score + 0.40 * content_score


def similarity_status(score):
    if score >= 0.60:
        return "aligned"
    elif score >= 0.30:
        return "weak"
    return "unaligned"


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

    for operation, oi, gi in question_pairs:

        if operation == "left_only":
            ocr_q = ocr_questions[oi]
            final_results.append({
                "question_status": "ocr_only",
                "ocr_number": ocr_q.get("number"),
                "gold_number": None,
                "ocr_stem": ocr_q.get("stem", ""),
                "gold_stem": None,
                "stem_similarity": None,
                "content_similarity": None,
                "combined_similarity": None,
                "block_alignment": None,
            })
            continue

        if operation == "right_only":
            gold_q = gold_questions[gi]
            final_results.append({
                "question_status": "gold_only",
                "ocr_number": None,
                "gold_number": gold_q.get("number"),
                "ocr_stem": None,
                "gold_stem": gold_q.get("stem", ""),
                "stem_similarity": None,
                "content_similarity": None,
                "combined_similarity": None,
                "block_alignment": None,
            })
            continue

        ocr_q = ocr_questions[oi]
        gold_q = gold_questions[gi]

        stem_score, _ = question_alignment_score(ocr_q, gold_q)

        block_result = align_blocks(ocr_q, gold_q)
        content_score = block_result["content_similarity"]

        combined_score = combined_similarity(
            stem_score,
            content_score,
        )

        final_results.append({
            "question_status": similarity_status(combined_score),
            "ocr_number": ocr_q.get("number"),
            "gold_number": gold_q.get("number"),
            "ocr_stem": ocr_q.get("stem", ""),
            "gold_stem": gold_q.get("stem", ""),
            "stem_similarity": round(stem_score, 3),
            "content_similarity": content_score,
            "combined_similarity": round(combined_score, 3),
            "combined_weights": {
                "stem": 0.60,
                "content": 0.40 if content_score is not None else None,
            },
            "block_alignment": block_result,
        })

    return final_results


# ---------------------------------------------------------
# 9. Display a readable alignment report
# ---------------------------------------------------------

def print_report(results):
    for result in results:
        print("\n" + "=" * 90)
        print(
            f"OCR Q{result['ocr_number']} "
            f"<-> GOLD Q{result['gold_number']} | "
            f"{result['question_status']}"
        )

        if result["stem_similarity"] is None:
            print("Question exists on only one side.")
            continue

        print(
            f"Stem similarity:     {result['stem_similarity']:.3f}"
        )
        print(
            f"Content similarity:  {result['content_similarity']}"
        )
        print(
            f"Combined similarity: {result['combined_similarity']:.3f}"
        )

        print("\n[STEM ALIGNMENT]")
        print("OCR :")
        print(result["ocr_stem"] or "<EMPTY>")
        print("GOLD:")
        print(result["gold_stem"] or "<EMPTY>")

        block_result = result["block_alignment"]

        if block_result is None:
            continue

        print("\n[CONTENT ALIGNMENT]")
        print("Per-kind similarity:", block_result["per_kind_similarity"])

        for block in block_result["alignment"]:
            print(
                f"\n  Kind: {block['kind']} | "
                f"{block['operation']} | "
                f"similarity={block['similarity']:.3f}"
            )
            print("  OCR : ", block["ocr_text"] or "<MISSING>")
            print("  GOLD: ", block["gold_text"] or "<MISSING>")


# ---------------------------------------------------------
# 10. Main
# ---------------------------------------------------------

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
