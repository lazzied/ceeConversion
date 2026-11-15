"""Shared data models and settings for LaTeX exam splitting."""

from dataclasses import dataclass
from pathlib import Path


QUESTION_ENVIRONMENT = "problem"
SOLUTION_ENVIRONMENTS = frozenset({"answer", "solution"})
VERBATIM_ENVIRONMENTS = ("verbatim", "lstlisting", "minted", "comment")
DEFAULT_COMPILE_TIMEOUT_SECONDS = 180


class SplitError(RuntimeError):
    """Raised when a source cannot be split safely or an output fails checks."""


@dataclass(frozen=True)
class Span:
    """A source interval and its LaTeX marker type."""

    start: int
    end: int
    kind: str
    raw: str


@dataclass(frozen=True)
class QuestionSpan:
    """One numbered problem environment and its associated solution blocks."""

    span: Span
    optional_argument: str
    inline_solutions: tuple[Span, ...]
    following_solutions: tuple[Span, ...]
    labels: tuple[str, ...] = ()

    @property
    def has_solution(self) -> bool:
        """Whether this problem has any extracted answer or solution content."""

        return bool(self.inline_solutions or self.following_solutions)


@dataclass(frozen=True)
class SplitPlan:
    """Prepared question and solution bodies for one combined source."""

    source: Path
    questions_body: str
    solutions_body: str
    question_count: int
    paired_question_count: int
    style_file: Path | None
    compilation_directory: Path
    document_preamble: str | None = None
    document_start: str | None = None
    document_end: str | None = None
    document_trailer: str = ""


@dataclass(frozen=True)
class SplitOutput:
    """Paths and validation results for a generated question/solution pair."""

    questions_file: Path
    solutions_file: Path
    question_count: int
    paired_question_count: int
    compiled: bool
