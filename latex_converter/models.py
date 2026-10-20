
"""Configuration and errors shared by the JSON-to-LaTeX renderer."""

from dataclasses import dataclass


SUPPORTED_SCHEMA = "1.0"


@dataclass
class RenderOptions:
    """Rendering choices that do not change the exam's source content."""

    figures: str = "placeholder"      # "placeholder" | "image" (use source_path)
    wrap: int = 88                    # soft line width; 0 disables wrapping
    normalize_parens: bool = True     # （\ \ ） -> \choiceparen{}
    font_size: str = "12pt"
    document_class: str = "ctexart"

    def __post_init__(self) -> None:
        """Reject unsupported figure modes and negative wrapping widths."""
        if self.figures not in {"placeholder", "image"}:
            raise ValueError("figures must be 'placeholder' or 'image'")
        if self.wrap < 0:
            raise ValueError("wrap must be zero (disabled) or a positive width")


class ExamFormatError(ValueError):
    """The input JSON does not follow the exam schema."""
