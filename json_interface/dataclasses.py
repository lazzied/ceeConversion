"""Exam object model.

The model keeps *content* (stems, options, cells...) as raw LaTeX strings so
math survives untouched, and makes *structure* explicit with typed classes.

Reading order of a question when rendering:

    stem -> blocks -> options | sub_questions -> footer_blocks

Every block carries a ``type`` discriminator in the JSON output
("text", "figure", "table", "statement_list", "callout").
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Iterator, Optional, Union

SCHEMA_VERSION = "1.0"


class QuestionKind(str, Enum):
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    FILL_BLANK = "fill_blank"
    FREE_RESPONSE = "free_response"
    OTHER = "other"


CHOICE_KINDS = {QuestionKind.SINGLE_CHOICE, QuestionKind.MULTIPLE_CHOICE}


# --------------------------------------------------------------------- blocks
@dataclass
class TextBlock:
    TYPE: ClassVar[str] = "text"
    content: str


@dataclass
class FigureBlock:
    TYPE: ClassVar[str] = "figure"
    label: str                                # caption / placeholder label
    width_cm: Optional[float] = None
    height_cm: Optional[float] = None
    is_placeholder: bool = True               # True while it is an \imgph box
    source_path: Optional[str] = None         # real image path if one was hinted


@dataclass
class TableBlock:
    TYPE: ClassVar[str] = "table"
    column_spec: str                          # e.g. "lcccc"
    rows: list[list[str]] = field(default_factory=list)   # raw LaTeX cells
    header_rows: int = 0                      # rows above the first inner rule
    rules: list[int] = field(default_factory=list)        # inner rule after N rows


@dataclass
class StatementItem:
    label: str                                # "①", "②", ...
    text: str


@dataclass
class StatementListBlock:
    """Labelled propositions inside a stem (the ①②③④ lists)."""
    TYPE: ClassVar[str] = "statement_list"
    items: list[StatementItem] = field(default_factory=list)


@dataclass
class CalloutBlock:
    """Boxed reference material, e.g. the '附:' formula + critical-value table."""
    TYPE: ClassVar[str] = "callout"
    label: Optional[str] = None
    blocks: list["Block"] = field(default_factory=list)


Block = Union[TextBlock, FigureBlock, TableBlock, StatementListBlock, CalloutBlock]


# ------------------------------------------------------------------ questions
@dataclass
class Option:
    label: str                                # "A".."D"
    text: Optional[str] = None
    figure: Optional[FigureBlock] = None      # image-only options (question 8)


@dataclass
class SubQuestion:
    label: str                                # "(1)", "(2)" ...
    text: str = ""
    blocks: list[Block] = field(default_factory=list)
    options: list[Option] = field(default_factory=list)
    sub_questions: list["SubQuestion"] = field(default_factory=list)
    footer_blocks: list[Block] = field(default_factory=list)
    blank_count: int = 0


@dataclass
class Question:
    number: int
    kind: QuestionKind = QuestionKind.FREE_RESPONSE
    stem: str = ""
    blocks: list[Block] = field(default_factory=list)
    options: list[Option] = field(default_factory=list)
    sub_questions: list[SubQuestion] = field(default_factory=list)
    footer_blocks: list[Block] = field(default_factory=list)
    blank_count: int = 0                      # number of \blank in the stem
    points: Optional[float] = None            # not present in the source
    raw: Optional[str] = None                 # original LaTeX (parser option)


@dataclass
class Section:
    order: int
    title: str                                # "单选题"
    ordinal: Optional[str] = None             # "一"
    kind: QuestionKind = QuestionKind.OTHER
    description: Optional[str] = None
    questions: list[Question] = field(default_factory=list)


@dataclass
class ExamMetadata:
    title: Optional[str] = None               # 2024 年普通高等学校招生全国统一考试
    subtitle: Optional[str] = None            # 数学（文科） · 全国甲卷
    year: Optional[int] = None
    subject: Optional[str] = None
    paper: Optional[str] = None
    header_left: Optional[str] = None
    header_right: Optional[str] = None
    source_file: Optional[str] = None


# ----------------------------------------------------------------------- exam
def serialize_dataclass_value(obj: Any, exclude_none: bool) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        type_tag = getattr(obj, "TYPE", None)
        if type_tag:
            out["type"] = type_tag
        for f in fields(obj):
            value = getattr(obj, f.name)
            if exclude_none and value is None:
                continue
            out[f.name] = serialize_dataclass_value(value, exclude_none)
        return out
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [serialize_dataclass_value(v, exclude_none) for v in obj]
    return obj


@dataclass
class Exam:
    metadata: ExamMetadata = field(default_factory=ExamMetadata)
    instructions: Optional[str] = None
    sections: list[Section] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    # -- navigation
    def iter_questions(self) -> Iterator[Question]:
        for section in self.sections:
            yield from section.questions

    def get_question(self, number: int) -> Optional[Question]:
        return next((q for q in self.iter_questions() if q.number == number), None)

    # -- output
    def to_dict(self, exclude_none: bool = False) -> dict[str, Any]:
        return serialize_dataclass_value(self, exclude_none)

    def to_json(self, indent: int = 2, exclude_none: bool = False) -> str:
        return json.dumps(self.to_dict(exclude_none), ensure_ascii=False, indent=indent)

    def save(self, path: Union[str, Path], **kwargs: Any) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(**kwargs), encoding="utf-8")
        return path

    # -- sanity checks
    def validate(self) -> list[str]:
        issues: list[str] = []
        numbers = [q.number for q in self.iter_questions()]
        if numbers:
            expected = list(range(numbers[0], numbers[0] + len(numbers)))
            if numbers != expected:
                issues.append(f"question numbering is not consecutive: {numbers}")
        for q in self.iter_questions():
            if q.kind in CHOICE_KINDS:
                if len(q.options) != 4:
                    issues.append(f"Q{q.number}: choice question has {len(q.options)} options")
                labels = [o.label for o in q.options]
                if labels != list("ABCDEFGH"[:len(labels)]):
                    issues.append(f"Q{q.number}: option labels are not A, B, C... in order: {labels}")
            if q.kind == QuestionKind.FILL_BLANK and q.blank_count == 0:
                issues.append(f"Q{q.number}: fill_blank question without any \\blank")
            if not q.stem and not q.blocks and not q.sub_questions:
                issues.append(f"Q{q.number}: empty question")
        return issues

    def summary(self) -> str:
        lines = [f"{self.metadata.title} | {self.metadata.subtitle}"]
        for s in self.sections:
            lines.append(f"  [{s.ordinal or s.order}] {s.title} ({s.kind.value}): "
                         f"{len(s.questions)} questions")
            for q in s.questions:
                extras = []
                if q.options:
                    extras.append(f"{len(q.options)} options")
                if q.sub_questions:
                    extras.append(f"{len(q.sub_questions)} sub-questions")
                if q.blank_count:
                    extras.append(f"{q.blank_count} blank")
                kinds = [b.TYPE for b in q.blocks + q.footer_blocks]
                if kinds:
                    extras.append("blocks=" + ",".join(kinds))
                lines.append(f"      Q{q.number:<3} {q.kind.value:<14} {'; '.join(extras)}")
        return "\n".join(lines)
