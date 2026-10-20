"""Shared rules for assigning exam section and question kinds."""

from .dataclasses import CHOICE_KINDS, QuestionKind


class Classifier:
    """Rules that decide section and question kinds. Edit the tables to adapt
    to other exam formats."""

    SECTION_RULES = [
        ("多选", QuestionKind.MULTIPLE_CHOICE),
        ("单选", QuestionKind.SINGLE_CHOICE),
        ("选择", QuestionKind.SINGLE_CHOICE),
        ("填空", QuestionKind.FILL_BLANK),
        ("解答", QuestionKind.FREE_RESPONSE),
        ("证明", QuestionKind.FREE_RESPONSE),
        ("计算", QuestionKind.FREE_RESPONSE),
    ]

    @classmethod
    def section_kind(cls, title: str) -> QuestionKind:
        """Infer a section's question type from recognizable heading words."""
        for needle, kind in cls.SECTION_RULES:
            if needle in title:
                return kind
        return QuestionKind.OTHER

    @staticmethod
    def question_kind(has_options: bool, blank_count: int, section_kind: QuestionKind) -> QuestionKind:
        """Prefer observed content, using the section kind only as a fallback."""
        # Structure wins; the section heading only breaks ties.
        if has_options:
            return section_kind if section_kind in CHOICE_KINDS else QuestionKind.SINGLE_CHOICE
        if blank_count:
            return QuestionKind.FILL_BLANK
        return QuestionKind.FREE_RESPONSE
