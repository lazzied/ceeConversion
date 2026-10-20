"""Mutable references to translatable strings inside a copied JSON tree."""

from dataclasses import dataclass, field


@dataclass
class Unit:
    """One complete JSON field and the contextual ancestors for translation."""

    id: int
    path: str
    group: str
    text: str
    parent: dict | list
    key: str | int
    segments: list[str] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)
    context: str = ""
    field_text: str = ""
    rejection_reason: str = ""
    validation_text: str | None = None
