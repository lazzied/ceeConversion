"""Parse question/solution markers and prepare separate LaTeX document bodies."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

from .models import (
    QUESTION_ENVIRONMENT,
    SOLUTION_ENVIRONMENTS,
    VERBATIM_ENVIRONMENTS,
    QuestionSpan,
    Span,
    SplitError,
    SplitPlan,
)

_ENVIRONMENT_TOKEN = re.compile(
    r"\\(?P<action>begin|end)\s*\{(?P<environment>[A-Za-z*]+)\}"
)
_LEGACY_SOLUTION_COMMAND = re.compile(r"\\(?P<command>answer|solution)\s*\{")
_LABEL_COMMAND = re.compile(r"\\label\s*\{")
_OPTIONAL_ARGUMENT = re.compile(r"\s*(\[(?:[^\]\\]|\\.)*\])")


def _mask_comments_and_verbatim(source: str) -> str:
    """Blank comments and verbatim regions while preserving every character offset."""

    masked = list(source)
    length = len(source)
    index = 0

    def blank(start: int, end: int) -> None:
        for position in range(start, end):
            if masked[position] not in "\r\n":
                masked[position] = " "

    while index < length:
        if source[index] == "%":
            slash_count = 0
            previous = index - 1
            while previous >= 0 and source[previous] == "\\":
                slash_count += 1
                previous -= 1
            if slash_count % 2 == 0:
                line_end = source.find("\n", index)
                if line_end < 0:
                    line_end = length
                blank(index, line_end)
                index = line_end
                continue

        if source.startswith("\\verb", index):
            command_end = index + len("\\verb")
            if command_end < length and source[command_end] == "*":
                command_end += 1
            if command_end < length and not source[command_end].isalnum():
                delimiter = source[command_end]
                end = source.find(delimiter, command_end + 1)
                end = length if end < 0 else end + 1
                blank(index, end)
                index = end
                continue

        if source.startswith("\\begin", index):
            token = _ENVIRONMENT_TOKEN.match(source, index)
            if token and token.group("environment") in VERBATIM_ENVIRONMENTS:
                environment = token.group("environment")
                closing = re.compile(
                    rf"\\end\s*\{{{re.escape(environment)}\}}"
                ).search(source, token.end())
                end = length if closing is None else closing.end()
                blank(index, end)
                index = end
                continue
        index += 1
    return "".join(masked)


def _matching_environment_end(
    masked: str, opening: re.Match[str], environment: str
) -> int:
    """Return the end offset for a balanced environment, or raise clearly."""

    depth = 1
    for token in _ENVIRONMENT_TOKEN.finditer(masked, opening.end()):
        if token.group("environment") != environment:
            continue
        depth += 1 if token.group("action") == "begin" else -1
        if depth == 0:
            return token.end()
    raise SplitError(f"Unclosed \\begin{{{environment}}} at character {opening.start()}")


def _environment_spans(source: str, masked: str) -> list[Span]:
    """Find complete problem, answer, and solution environments."""

    spans: list[Span] = []
    for token in _ENVIRONMENT_TOKEN.finditer(masked):
        if token.group("action") != "begin":
            continue
        environment = token.group("environment")
        if environment != QUESTION_ENVIRONMENT and environment not in SOLUTION_ENVIRONMENTS:
            continue
        end = _matching_environment_end(masked, token, environment)
        spans.append(Span(token.start(), end, environment, source[token.start() : end]))
    return spans


def _group_end(masked: str, opening_brace: int) -> int:
    """Find the close of a balanced TeX brace group starting at ``{``."""

    depth = 0
    index = opening_brace
    while index < len(masked):
        character = masked[index]
        if character == "\\":
            index += 2  # Escaped braces do not affect TeX group depth.
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise SplitError(f"Unclosed solution command argument at character {opening_brace}")


def _legacy_solution_spans(source: str, masked: str) -> list[Span]:
    r"""Find legacy ``\answer{...}`` and ``\solution{...}`` command calls."""

    spans: list[Span] = []
    for token in _LEGACY_SOLUTION_COMMAND.finditer(masked):
        end = _group_end(masked, token.end() - 1)
        spans.append(Span(token.start(), end, token.group("command"), source[token.start() : end]))
    return spans


def _optional_argument(source: str, opening_end: int) -> str:
    """Return a problem's optional counter suffix, if present."""

    match = _OPTIONAL_ARGUMENT.match(source, opening_end)
    return match.group(1) if match else ""


def _question_labels(masked: str, problem: Span) -> tuple[str, ...]:
    """Collect question labels so references in the solution copy stay defined."""

    labels: list[str] = []
    for match in _LABEL_COMMAND.finditer(masked, problem.start, problem.end):
        end = _group_end(masked, match.end() - 1)
        labels.append(masked[match.start() : end])
    return tuple(labels)


def _apply_edits(source: str, edits: Iterable[tuple[int, int, str]]) -> str:
    """Apply non-overlapping source edits from right to left."""

    result = source
    ordered = sorted(edits, key=lambda edit: (edit[0], edit[1]))
    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] > current[0]:
            raise SplitError("Overlapping LaTeX solution markers make this split ambiguous")
    for start, end, replacement in reversed(ordered):
        result = result[:start] + replacement + result[end:]
    return result


def _solution_heading(
    optional_argument: str,
    inline_solutions: Sequence[Span],
    labels: Sequence[str] = (),
) -> str:
    """Create a numbered solution heading and retain inline answer blocks."""

    # Starting a paragraph triggers the corpus style's normal question-number hook.
    heading = (
        f"\\problem{optional_argument}\\textbf{{Solutions}}"
        + "".join(labels)
        + r"\problembodyend"
    )
    inline = "\n".join(solution.raw for solution in inline_solutions)
    return heading + ("\n" + inline if inline else "") + "\n"


def _find_style_file(source: Path, corpus_root: Path | None) -> Path | None:
    """Locate the shared style file without assuming one fixed corpus layout."""

    if corpus_root is not None:
        candidate = corpus_root / "styles.tex"
        if candidate.is_file():
            return candidate.resolve()
    for parent in source.resolve().parents:
        candidate = parent / "styles.tex"
        if candidate.is_file():
            return candidate
    return None


def analyze_document(source: Path, corpus_root: Path | None = None) -> SplitPlan | None:
    """Analyze one LaTeX file and return a plan when both parts are present.

    Files without recognizable problem environments or solution content are
    left alone. Malformed or unpairable markers raise :class:`SplitError`.
    """

    source = Path(source).resolve()
    text = source.read_text(encoding="utf-8")
    masked = _mask_comments_and_verbatim(text)
    document_start_match = re.search(r"\\begin\s*\{document\}", masked)
    document_end_match = re.search(r"\\end\s*\{document\}", masked)
    document_preamble: str | None = None
    document_start: str | None = None
    document_end: str | None = None
    document_trailer = ""
    body_offset = 0
    if document_start_match or document_end_match:
        if (
            not document_start_match
            or not document_end_match
            or document_end_match.start() < document_start_match.end()
        ):
            raise SplitError(f"Malformed standalone document boundaries in {source}")
        document_preamble = text[: document_start_match.start()]
        document_start = text[document_start_match.start() : document_start_match.end()]
        document_end = text[document_end_match.start() : document_end_match.end()]
        document_trailer = text[document_end_match.end() :]
        body_offset = document_start_match.end()
        text = text[body_offset : document_end_match.start()]
        masked = masked[body_offset : document_end_match.start()]

    environment_spans = _environment_spans(text, masked)
    problem_spans = [span for span in environment_spans if span.kind == QUESTION_ENVIRONMENT]
    if not problem_spans:
        return None

    solution_environment_spans = [
        span for span in environment_spans if span.kind in SOLUTION_ENVIRONMENTS
    ]
    legacy_spans = _legacy_solution_spans(text, masked)
    solution_spans = solution_environment_spans + legacy_spans
    solution_spans = [
        span
        for span in solution_spans
        if not any(
            outer.start <= span.start and span.end <= outer.end
            for outer in solution_environment_spans
            if outer.start != span.start
        )
    ]

    questions: list[QuestionSpan] = []
    for problem in problem_spans:
        inline = tuple(
            span for span in solution_spans if problem.start <= span.start and span.end <= problem.end
        )
        opening = _ENVIRONMENT_TOKEN.match(masked, problem.start)
        if opening is None:
            raise SplitError(f"Could not parse problem marker at character {problem.start}")
        questions.append(
            QuestionSpan(
                span=problem,
                optional_argument=_optional_argument(text, opening.end()),
                inline_solutions=inline,
                following_solutions=(),
                labels=_question_labels(masked, problem),
            )
        )

    attached: list[list[Span]] = [[] for _ in questions]
    for solution in solution_spans:
        if any(
            question.span.start <= solution.start and solution.end <= question.span.end
            for question in questions
        ):
            continue
        previous_questions = [
            index for index, question in enumerate(questions) if question.span.end <= solution.start
        ]
        if not previous_questions:
            raise SplitError(
                f"Solution block at character {solution.start} appears before the first problem in {source}"
            )
        attached[previous_questions[-1]].append(solution)

    questions = [
        QuestionSpan(
            span=question.span,
            optional_argument=question.optional_argument,
            inline_solutions=question.inline_solutions,
            following_solutions=tuple(attached[index]),
            labels=question.labels,
        )
        for index, question in enumerate(questions)
    ]
    paired_count = sum(question.has_solution for question in questions)
    if paired_count == 0:
        return None

    questions_body = _apply_edits(text, [(span.start, span.end, "") for span in solution_spans])
    solution_edits: list[tuple[int, int, str]] = []
    for question in questions:
        replacement = _solution_heading(
            question.optional_argument,
            question.inline_solutions if question.has_solution else (),
            question.labels,
        )
        if not question.has_solution:
            replacement = f"\\problem{question.optional_argument}\\problembodyend\n"
        solution_edits.append((question.span.start, question.span.end, replacement))
    solution_edits.extend(
        (span.start, span.end, "")
        for span in legacy_spans
        if not any(problem.start <= span.start and span.end <= problem.end for problem in problem_spans)
    )
    solutions_body = _apply_edits(text, solution_edits)

    style_file = _find_style_file(source, corpus_root)
    if document_preamble is None and style_file is None:
        raise SplitError(f"Could not locate styles.tex for LaTeX fragment {source}")
    compilation_directory = (
        source.parent if document_preamble is not None else (style_file.parent if style_file else source.parent)
    )
    return SplitPlan(
        source=source,
        questions_body=questions_body,
        solutions_body=solutions_body,
        question_count=len(questions),
        paired_question_count=paired_count,
        style_file=style_file,
        compilation_directory=compilation_directory,
        document_preamble=document_preamble,
        document_start=document_start,
        document_end=document_end,
        document_trailer=document_trailer,
    )
