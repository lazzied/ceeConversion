
"""Deterministic cleanup passes for OCR-derived exam Markdown."""

from pathlib import Path
import html
import re
from collections.abc import Callable


# Protect mathematical expressions, HTML tags and Markdown images.
SPLIT_PATTERN = re.compile(
    r'(\$\$.+?\$\$|\$[^$\n]+\$|<[^>]+>|!\[[^\]]*\]\([^)]*\))',
    re.S,
)

CHINESE = re.compile(r'[\u4e00-\u9fff]')

UNICODE_TO_LATEX = {
    '∥': r'\parallel',
    '⊥': r'\perp',
    '⩾': r'\geqslant',
    '≥': r'\geqslant',
    '⩽': r'\leqslant',
    '≤': r'\leqslant',
    '≈': r'\approx',
    '∈': r'\in',
    '∩': r'\cap',
    '∪': r'\cup',
}

UNICODE_PATTERN = re.compile(
    '|'.join(re.escape(k) for k in UNICODE_TO_LATEX)
)

PUNCTUATION_MAP = str.maketrans({
    '，': ',',
    '；': ';',
    '：': ':',
    '（': '(',
    '）': ')',
    '。': '.',
})


def edit_parts(
    markdown: str,
    edit_text: Callable[[str], str] | None = None,
    edit_math: Callable[[str], str] | None = None,
) -> str:
    """Apply separate cleanup functions to prose and math spans.

    HTML tags and Markdown images are protected tokens.
    """

    parts = SPLIT_PATTERN.split(markdown)

    for i, part in enumerate(parts):
        if i % 2 == 0 and edit_text:
            parts[i] = edit_text(part)

        elif i % 2 == 1 and part.startswith('$') and edit_math:
            parts[i] = edit_math(part)

    return ''.join(parts)


# ---------------------------------------------------------
# 1. BASIC NORMALIZATION
# ---------------------------------------------------------

def decode_html_entities(markdown: str) -> str:
    """Decode HTML character references."""

    return html.unescape(markdown)


def remove_math_emphasis(markdown: str) -> str:
    """Remove accidental Markdown bold markers around LaTeX commands."""

    def clean(math: str) -> str:
        return re.sub(
            r'\*\*(\\[a-zA-Z]+)\*\*',
            r'\1',
            math,
        )

    return edit_parts(markdown, edit_math=clean)


def normalize_punctuation(markdown: str) -> str:
    """Normalize equivalent Chinese and ASCII punctuation in prose."""

    return edit_parts(
        markdown,
        edit_text=lambda text: text.translate(PUNCTUATION_MAP),
    )


def normalize_latex_commands(markdown: str) -> str:
    """Canonicalize selected equivalent LaTeX commands."""

    def normalize(math: str) -> str:
        math = re.sub(
            r'\\(?:dfrac|tfrac)(?![a-zA-Z])',
            r'\\frac',
            math,
        )

        # \bar{z} -> \overline{z}
        math = re.sub(
            r'\\bar\s*\{([^{}]*)\}',
            r'\\overline{\1}',
            math,
        )

        # \bar z -> \overline{z}, for a single Latin symbol.
        math = re.sub(
            r'\\bar\s*([A-Za-z])(?![A-Za-z])',
            r'\\overline{\1}',
            math,
        )

        return math

    return edit_parts(markdown, edit_math=normalize)


# ---------------------------------------------------------
# 2. FIGURE NORMALIZATION
# ---------------------------------------------------------

def convert_figures(markdown: str) -> str:
    """Convert HTML images and Markdown images to includegraphics only."""

    def convert_html_image(match: re.Match[str]) -> str:
        tag = match.group(0)

        src = re.search(
            r'''src\s*=\s*["']([^"']+)["']''',
            tag,
            re.IGNORECASE,
        )

        if not src:
            return tag

        return rf"\includegraphics{{{src.group(1).strip()}}}"

    def convert_markdown_image(match: re.Match[str]) -> str:
        image_path = match.group(2).strip()
        return rf"\includegraphics{{{image_path}}}"

    # HTML <img> tags.
    markdown = re.sub(
        r'<img\b[^>]*>',
        convert_html_image,
        markdown,
        flags=re.IGNORECASE,
    )

    # Markdown ![alt](path) images.
    markdown = re.sub(
        r'!\[([^\]]*)\]\(([^)]+)\)',
        convert_markdown_image,
        markdown,
    )

    return markdown


def unwrap_divs(markdown: str) -> str:
    """Remove layout-only div wrappers."""

    return re.sub(
        r'</?div\b[^>]*>',
        '',
        markdown,
        flags=re.IGNORECASE,
    )


# ---------------------------------------------------------
# 3. TABLE NORMALIZATION
# ---------------------------------------------------------

def normalize_tables(markdown: str) -> str:
    """Preserve table structure and remove all table styling attributes.

    Preserved:
        table, tr, td, th

    Other HTML tags are left untouched.
    """

    allowed_tags = {"table", "tr", "td", "th"}

    def clean_tag(match: re.Match[str]) -> str:
        tag = match.group(0)

        tag_match = re.match(
            r'<\s*(/?)\s*([a-zA-Z0-9]+)',
            tag,
        )

        if not tag_match:
            return tag

        closing, tag_name = tag_match.groups()
        tag_name = tag_name.lower()

        if tag_name not in allowed_tags:
            return tag

        if closing:
            return f"</{tag_name}>"

        return f"<{tag_name}>"

    return re.sub(
        r'<[^>]+>',
        clean_tag,
        markdown,
        flags=re.IGNORECASE,
    )


# ---------------------------------------------------------
# 4. MATHEMATICAL SYMBOL NORMALIZATION
# ---------------------------------------------------------

def replace_unicode_symbols(markdown: str) -> str:
    """Replace supported Unicode mathematical symbols with LaTeX."""

    return edit_parts(
        markdown,
        edit_text=lambda text: UNICODE_PATTERN.sub(
            lambda m: '$' + UNICODE_TO_LATEX[m.group()] + '$',
            text,
        ),
        edit_math=lambda math: UNICODE_PATTERN.sub(
            lambda m: UNICODE_TO_LATEX[m.group()],
            math,
        ),
    )


def replace_degree_sign(markdown: str) -> str:
    """Convert degree symbols into LaTeX math notation."""

    return edit_parts(
        markdown,
        edit_text=lambda text: re.sub(
            r'(\d+)\s*[°◦]',
            r'$\1^{\\circ}$',
            text,
        ),
        edit_math=lambda math: re.sub(
            r'(\d+)\s*[°◦]',
            r'\1^{\\circ}',
            math,
        ),
    )


def fullwidth_to_halfwidth_in_math(markdown: str) -> str:
    """Convert full-width ASCII characters inside math only."""

    def convert(math: str) -> str:
        return ''.join(
            chr(ord(c) - 0xFEE0)
            if 0xFF01 <= ord(c) <= 0xFF5E
            else c
            for c in math
        )

    return edit_parts(markdown, edit_math=convert)


def trim_math_spaces(markdown: str) -> str:
    """Remove padding immediately inside math delimiters."""

    return edit_parts(
        markdown,
        edit_math=lambda math: re.sub(
            r'^(\$+)\s*(.*?)\s*(\$+)$',
            r'\1\2\3',
            math,
            flags=re.S,
        ),
    )


# ---------------------------------------------------------
# 5. OCR-SPECIFIC CORRECTIONS
# ---------------------------------------------------------

def remove_page_numbers(markdown: str) -> str:
    """Remove standalone three- or four-digit page numbers."""

    return re.sub(
        r'(?m)^[ \t]*\d{3,4}[ \t]*(?:\n|\Z)',
        '',
        markdown,
    )


def fix_answer_brackets(markdown: str) -> str:
    """Normalize empty answer parentheses in prose."""

    return edit_parts(
        markdown,
        edit_text=lambda text: re.sub(
            r'[（(]\s*[）)]',
            '( )',
            text,
        ),
    )


def fix_fill_in_blanks(markdown: str) -> str:
    """Normalize repeated underscore blanks."""

    def fix(text: str) -> str:
        return re.sub(
            r'(?:\\_|_){3,}([.。]?)',
            lambda m: r'\_\_\_' + m.group(1),
            text,
        )

    return edit_parts(markdown, edit_text=fix)




def split_options(markdown: str) -> str:
    """
    Put every multiple-choice option (A)-(D) on a separate line.

    Supports:
      - Options separated by spaces.
      - Options separated by newlines.
      - ASCII and full-width parentheses.
      - Options containing inline LaTeX expressions.
    """
    option_pattern = r"(?:\([A-D]\)|（[A-D]）)"

    def separate(text: str) -> str:
        return re.sub(
            rf"[ \t]+(?={option_pattern})",
            "\n",
            text,
        )

    return edit_parts(markdown, edit_text=separate)



# ---------------------------------------------------------
# 6. WHITESPACE NORMALIZATION
# ---------------------------------------------------------

def clean_whitespace(markdown: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""

    markdown = re.sub(
        r'(?<=\S)[ \t]{2,}',
        ' ',
        markdown,
    )

    markdown = re.sub(
        r'(?m)[ \t]+$',
        '',
        markdown,
    )

    markdown = re.sub(
        r'\n{3,}',
        '\n\n',
        markdown,
    )

    return markdown.strip() + '\n'


# ---------------------------------------------------------
# 7. OPTIONAL CHINESE MATHEMATICAL SYMBOL DETECTION
# ---------------------------------------------------------

def wrap_loose_symbols(markdown: str) -> str:
    """Wrap short Latin mathematical symbols in Chinese prose.

    Heuristic: only operates on lines containing Chinese characters.
    """

    loose_symbol = re.compile(
        r'(?<![A-Za-z0-9$\\(])'
        r'[A-Za-z]{1,4}'
        r'(?:\s*[,、]\s*[A-Za-z]{1,4})*'
        r'(?![A-Za-z0-9$])'
    )

    def wrap_line(line: str) -> str:
        if not CHINESE.search(line):
            return line

        return edit_parts(
            line,
            edit_text=lambda text: loose_symbol.sub(
                lambda m: '$' + m.group() + '$',
                text,
            ),
        )

    return '\n'.join(
        wrap_line(line)
        for line in markdown.split('\n')
    )


# ---------------------------------------------------------
# 8. COMPLETE NORMALIZATION PIPELINE
# ---------------------------------------------------------

def normalize_markdown(
    markdown: str,
    wrap_symbols: bool = True,
) -> str:
    """Normalize OCR Markdown using deterministic cleanup passes."""

    steps = [
        decode_html_entities,
        convert_figures,
        normalize_tables,
        unwrap_divs,
        remove_math_emphasis,
        normalize_punctuation,
        normalize_latex_commands,
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


# ---------------------------------------------------------
# 9. SCRIPT ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":

    input_path = Path("data/input/exam.md")
    output_path = Path("data/output/exam.md")

    normalized = normalize_markdown(
        input_path.read_text(encoding="utf-8")
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        normalized,
        encoding="utf-8",
    )

    print(f"Written to {output_path}")
