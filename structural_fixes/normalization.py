"""Deterministic cleanup passes for OCR-derived exam Markdown."""

from pathlib import Path
import re
from collections.abc import Callable

# Each match is kept intact while text and math-specific cleanup passes run.
SPLIT_PATTERN = re.compile(r'(\$\$.+?\$\$|\$[^$\n]+\$|<[^>]+>|!\[[^\]]*\]\([^)]*\))', re.S)
"""
SPLIT_PATTERN matches the things you don’t want to damage: $$...$$ display math, $...$ inline math, HTML tags, and Markdown image links.
"""


CHINESE = re.compile(r'[\u4e00-\u9fff]')
MIN_OCR_FIGURE_WIDTH_FRACTION = 0.4

UNICODE_TO_LATEX = {
    '∥': r'\parallel', '⊥': r'\perp',
    '⩾': r'\geqslant', '≥': r'\geqslant',
    '⩽': r'\leqslant', '≤': r'\leqslant',
    '≈': r'\approx', '∈': r'\in', '∩': r'\cap', '∪': r'\cup',
}
UNICODE_PATTERN = re.compile('|'.join(UNICODE_TO_LATEX))


def edit_parts(
    markdown: str,
    edit_text: Callable[[str], str] | None = None,
    edit_math: Callable[[str], str] | None = None,
) -> str:
    """Apply separate cleanup functions to prose and math spans.

    HTML tags and Markdown image links are protected tokens: they are passed
    through untouched so a text rule cannot accidentally rewrite their syntax.
    A missing callback means that representation is preserved as-is.
    """
    parts = SPLIT_PATTERN.split(markdown)
    for i, part in enumerate(parts):
        if i % 2 == 0 and edit_text:
            parts[i] = edit_text(part)
        elif i % 2 == 1 and part.startswith('$') and edit_math:
            parts[i] = edit_math(part)
    return ''.join(parts)



def convert_figures(markdown: str) -> str:
    """Convert a simple HTML image wrapper to a centered LaTeX figure.

    OCR often records a crop's bounding-box width as a tiny page-relative
    percentage. The lower width bound keeps those figures readable; the
    original image path is preserved for the renderer.
    """
    def to_latex(match: re.Match[str]) -> str:
        """Build one figure while retaining the source image path."""
        img_tag = match.group(1)
        src = re.search(r'src=["\']([^"\']+)["\']', img_tag)
        if not src:
            return match.group(0)
        width = re.search(r'width=["\'](\d+(?:\.\d+)?)%', img_tag)
        fraction = max(float(width.group(1)) / 100, MIN_OCR_FIGURE_WIDTH_FRACTION) if width else 1
        return (
            "\\begin{figure}[h]\n"
            "\\centering\n"
            f"\\includegraphics[width={fraction:g}\\linewidth]{{{src.group(1)}}}\n"
            "\\end{figure}"
        )
    return re.sub(r"<div[^>]*>\s*(<img\b[^>]*>)\s*</div>", to_latex, markdown,
                  flags=re.IGNORECASE)


def unwrap_divs(markdown: str) -> str:
    """Remove remaining layout-only div wrappers after figure conversion."""
    return re.sub(r"<div[^>]*>(.*?)</div>", lambda m: m.group(1).strip(), markdown,
                  flags=re.DOTALL | re.IGNORECASE)



def replace_unicode_symbols(markdown: str) -> str:
    """Replace supported relation and set glyphs with LaTeX equivalents.

    In prose the command is placed in a new math span. Inside math, only the
    glyph is replaced, preserving the surrounding expression.
    """
    return edit_parts(
        markdown,
        edit_text=lambda t: UNICODE_PATTERN.sub(lambda m: '$' + UNICODE_TO_LATEX[m.group()] + '$', t),
        edit_math=lambda m: UNICODE_PATTERN.sub(lambda s: UNICODE_TO_LATEX[s.group()] + ' ', m),
    )


def replace_degree_sign(markdown: str) -> str:
    """Represent a degree mark as a math-mode ``\\circ`` command."""
    return edit_parts(
        markdown,
        edit_text=lambda t: re.sub(r'(\d+)\s*[°◦]', r'$\1^{\\circ}$', t),
        edit_math=lambda m: re.sub(r'(\d)\s*[°◦]', r'\1^{\\circ}', m),
    )


def fullwidth_to_halfwidth_in_math(markdown: str) -> str:
    """Convert full-width ASCII-like characters inside math spans only."""
    def convert(math: str) -> str:
        """Map full-width ASCII code points to their half-width forms."""
        return ''.join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in math)
    return edit_parts(markdown, edit_math=convert)


def trim_math_spaces(markdown: str) -> str:
    """Remove padding immediately inside single- or double-dollar delimiters."""
    return edit_parts(
        markdown,
        edit_math=lambda m: re.sub(r'^(\$+)\s*(.*?)\s*(\$+)$', r'\1\2\3', m, flags=re.S),
    )


def remove_page_numbers(markdown: str) -> str:
    """Drop standalone three- or four-digit OCR page-number lines."""
    return re.sub(r'(?m)^[ \t]*\d{3,4}[ \t]*(?:\n|\Z)', '', markdown)


def fix_answer_brackets(markdown: str) -> str:
    """Normalize empty answer parentheses without touching mathematical text."""
    return edit_parts(markdown, edit_text=lambda t: re.sub(r'[（(]\s*[）)]', '（ ）', t))


def fix_fill_in_blanks(markdown: str) -> str:
    """Convert repeated underscore blanks to an escaped LaTeX blank marker."""
    def fix(text: str) -> str:
        """Keep an optional sentence-ending period after the blank marker."""
        return re.sub(r'(?:\\_|_){3,}([.。]?)',
                      lambda m: r'\_\_\_' + ('。' if m.group(1) else ''), text)
    return edit_parts(markdown, edit_text=fix)


def split_options(markdown: str) -> str:
    """Separate adjacent A–D choices so each begins its own paragraph."""
    next_option = r'\([A-D]\)(?:\s|$)'
    normalized_labels = re.sub(
        r'(?m)(?<![\w$])[（(]([A-D])[）)](?=\s|$)', r'(\1)', markdown
    )
    separated_inline_options = re.sub(
        rf'(?m)[ \t]+(?={next_option})', '\n\n', normalized_labels
    )
    return re.sub(
        rf'(?m)([^\n])\n(?={next_option})', r'\1\n\n', separated_inline_options
    )


def clean_whitespace(markdown: str) -> str:
    """Trim line endings and excess whitespace while preserving paragraph breaks."""
    markdown = re.sub(r'(?<=\S)[ \t]{2,}', ' ', markdown)
    markdown = re.sub(r'(?m)[ \t]+$', '', markdown)
    return re.sub(r'\n{3,}', '\n\n', markdown).strip() + '\n'


def wrap_loose_symbols(markdown: str) -> str:
    """Heuristically wrap short Latin math symbols in Chinese prose.

    This is intentionally limited to lines containing Chinese characters; on
    ordinary English prose, short words could otherwise be mistaken for names.
    """
    loose_symbol = re.compile(
        r'(?<![A-Za-z0-9$\\(])[A-Za-z]{1,4}(?:\s*[,、]\s*[A-Za-z]{1,4})*(?![A-Za-z0-9$])')

    def wrap_line(line: str) -> str:
        """Leave non-Chinese lines alone and edit only their prose segments."""
        if not CHINESE.search(line):
            return line
        return edit_parts(line, edit_text=lambda t: loose_symbol.sub(lambda m: '$' + m.group() + '$', t))

    return '\n'.join(wrap_line(line) for line in markdown.split('\n'))


def normalize_markdown(markdown: str, wrap_symbols: bool = True) -> str:
    """Run deterministic cleanup passes in their dependency order.

    Figure wrappers are converted before generic divs are removed. Text and
    math edits use ``edit_parts`` to avoid applying prose rules to formulas.
    The final symbol-wrapping pass is heuristic and can be disabled when the
    input is already normalized or contains many short Latin words.
    """
    # Order matters: preserve figure markup before removing layout wrappers,
    # then normalize symbols and blanks before splitting adjacent options.
    steps = [
        convert_figures,
        unwrap_divs,
        replace_unicode_symbols,
        replace_degree_sign,
        fullwidth_to_halfwidth_in_math,
        trim_math_spaces,
        remove_page_numbers,
        fix_answer_brackets,
        fix_fill_in_blanks,
        split_options,
        clean_whitespace,
    ]
    if wrap_symbols:
        steps.append(wrap_loose_symbols)

    for step in steps:
        markdown = step(markdown)
    return markdown

if __name__ == "__main__":
    # Keep a direct script mode for quick manual normalization of the sample.
    input_path = Path("data/input/exam.md")
    output_path = Path("data/output/exam.md")

    normalized = normalize_markdown(input_path.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(normalized, encoding="utf-8")
    print(f"Written to {output_path}")
