"""Tools for splitting combined exam and solution LaTeX documents.

Use :func:`analyze_document` to inspect a source without writing files, or
:func:`split_document` to create and validate separate question and solution
documents. Run ``python -m latex_exam_splitter --help`` for the batch CLI.
"""

from .discovery import discover_tex_files, scan_corpus
from .models import QuestionSpan, Span, SplitError, SplitOutput, SplitPlan
from .output import split_document
from .parser import analyze_document

__all__ = [
    "QuestionSpan",
    "Span",
    "SplitError",
    "SplitOutput",
    "SplitPlan",
    "analyze_document",
    "discover_tex_files",
    "scan_corpus",
    "split_document",
]
