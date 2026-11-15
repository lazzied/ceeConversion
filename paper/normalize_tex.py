
"""Convert gold-standard LaTeX exam documents into canonical Markdown."""

import html
import re


CIRCLED = {
    "1": "①",
    "2": "②",
    "3": "③",
    "4": "④",
    "5": "⑤",
}

PUNCTUATION_MAP = str.maketrans({
    "，": ",",
    "；": ";",
    "：": ":",
    "（": "(",
    "）": ")",
    "。": ".",
})


# ---------------------------------------------------------
# 1. BASIC LATEX PARSING
# ---------------------------------------------------------

def read_group(s: str, i: int) -> tuple[str, int]:
    """Read a balanced LaTeX brace group starting at s[i]."""

    if i >= len(s) or s[i] != "{":
        raise ValueError("Expected opening brace")

    depth = 0

    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1

            if depth == 0:
                return s[i + 1:j], j + 1

    raise ValueError("Unbalanced braces")


def split_math_parts(markdown: str) -> list[str]:
    """
    Split Markdown into alternating text and math segments.

    Math segments include:
      - Inline math: $...$
      - Display math: $$...$$
    """

    pattern = re.compile(
        r"(\$\$.+?\$\$|\$[^$\n]+\$)",
        re.S,
    )

    return pattern.split(markdown)


def edit_math_parts(markdown: str, callback) -> str:
    """Apply a transformation only to inline and display math."""

    parts = split_math_parts(markdown)

    for i in range(1, len(parts), 2):
        parts[i] = callback(parts[i])

    return "".join(parts)


# ---------------------------------------------------------
# 2. FIGURE NORMALIZATION
# ---------------------------------------------------------

def make_figure(path: str) -> str:
    """Generate the canonical image representation."""

    return rf"\includegraphics{{{path.strip()}}}"


def convert_images(tex: str) -> str:
    """Convert custom image commands into includegraphics."""

    # Images inside answer choices.
    tex = re.sub(
        r"\\choicebitmap(?:\[[^\]]*\])?\{([^}]*)\}",
        lambda m: make_figure(m.group(1)),
        tex,
    )

    # Main figure of a problem.
    tex = re.sub(
        r"\\bitmapfigure(?:\[[^\]]*\])?\{([^}]*)\}",
        lambda m: "\n\n" + make_figure(m.group(1)) + "\n\n",
        tex,
    )

    return tex


def normalize_includegraphics(markdown: str) -> str:
    """Remove optional image presentation arguments."""

    return re.sub(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}",
        lambda m: make_figure(m.group(1)),
        markdown,
    )


# ---------------------------------------------------------
# 3. TABLE NORMALIZATION
# ---------------------------------------------------------

def convert_tables(tex: str) -> str:
    """Convert tabular environments into clean HTML tables."""

    def to_html(match: re.Match[str]) -> str:
        content = match.group(1)

        # Remove horizontal rules.
        content = re.sub(
            r"\\(?:hline|toprule|midrule|bottomrule)",
            "",
            content,
        )

        # Split rows.
        rows = re.split(
            r"\\\\(?:\s*\[[^\]]*\])?",
            content,
        )

        output = ["<table>"]

        for row in rows:
            if not row.strip():
                continue

            output.append("<tr>")

            cells = row.split("&")

            for cell in cells:
                cell = cell.replace(
                    r"\fillinblank{}",
                    "___",
                )

                # Convert text commands into readable cell text.
                cell = re.sub(
                    r"\\text\{([^{}]*)\}",
                    r"\1",
                    cell,
                )

                output.append(f"<td>{cell.strip()}</td>")

            output.append("</tr>")

        output.append("</table>")

        return "\n\n" + "".join(output) + "\n\n"

    return re.sub(
        r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
        to_html,
        tex,
        flags=re.S,
    )


# ---------------------------------------------------------
# 4. CHOICE NORMALIZATION
# ---------------------------------------------------------

def convert_choices(tex: str) -> str:
    """
    Convert \\choices {A}{B}{C}{D} into Markdown options.

    Every option is always placed on its own line,
    regardless of its length or whether it contains an image.
    """

    result = []
    pos = 0
    command = r"\choices"

    while True:
        k = tex.find(command, pos)

        if k < 0:
            result.append(tex[pos:])
            break

        result.append(tex[pos:k])

        j = k + len(command)
        options = []

        while len(options) < 4:
            j2 = j + len(tex[j:]) - len(tex[j:].lstrip())

            if j2 < len(tex) and tex[j2] == "{":
                content, j = read_group(tex, j2)
                options.append(content.strip())
            else:
                break

        # Avoid silently destroying an incomplete choices command.
        if len(options) != 4:
            result.append(tex[k:j])
            pos = j
            continue

        items = [
            f"({'ABCD'[n]}) {option}"
            for n, option in enumerate(options)
        ]

        # Always put each choice on a separate line.
        result.append(
            "\n\n" + "\n".join(items) + "\n"
        )

        pos = j

    return "".join(result)


def split_inline_choices(markdown: str) -> str:
    """
    Split inline multiple-choice options onto separate lines.

    Supports:
      (A), (B), (C), (D)
      （A）, （B）, （C）, （D）

    Only transforms text outside math expressions.

    The function requires at least two option labels on the same
    line to reduce accidental splitting of ordinary prose.
    """

    option_pattern = r"[（(][A-D][）)]"

    def split_text(text: str) -> str:
        lines = text.splitlines(keepends=True)
        output = []

        for line in lines:
            # Only process lines containing at least two option labels.
            labels = list(re.finditer(option_pattern, line))

            if len(labels) < 2:
                output.append(line)
                continue

            # Split at whitespace immediately preceding option labels.
            line = re.sub(
                rf"[ \t]+(?={option_pattern})",
                "\n",
                line,
            )

            output.append(line)

        return "".join(output)

    parts = split_math_parts(markdown)

    # Even indices are normal text; odd indices are math.
    for i in range(0, len(parts), 2):
        parts[i] = split_text(parts[i])

    return "".join(parts)


# ---------------------------------------------------------
# 5. LIST NORMALIZATION
# ---------------------------------------------------------

def convert_lists(tex: str) -> str:
    """Convert custom and standard lists into Markdown paragraphs."""

    def circle_list(match: re.Match[str]) -> str:
        items = match.group(1).split(r"\item")[1:]
        lines = []

        for n, item in enumerate(items, 1):
            if str(n) not in CIRCLED:
                lines.append(item.strip())
            else:
                lines.append(
                    f"{CIRCLED[str(n)]} {item.strip()}"
                )

        return "\n\n" + "\n\n".join(lines) + "\n\n"

    def number_list(match: re.Match[str]) -> str:
        items = match.group(1).split(r"\item")[1:]

        lines = [
            f"({n}) {item.strip()}"
            for n, item in enumerate(items, 1)
        ]

        return "\n\n" + "\n\n".join(lines) + "\n\n"

    tex = re.sub(
        r"\\begin\{circlelist\}(.*?)\\end\{circlelist\}",
        circle_list,
        tex,
        flags=re.S,
    )

    tex = re.sub(
        r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}",
        number_list,
        tex,
        flags=re.S,
    )

    return tex


# ---------------------------------------------------------
# 6. MATHEMATICAL NORMALIZATION
# ---------------------------------------------------------

def normalize_latex_commands(markdown: str) -> str:
    """Canonicalize equivalent LaTeX presentation commands."""

    def normalize(math: str) -> str:
        math = re.sub(
            r"\\(?:dfrac|tfrac)(?![a-zA-Z])",
            r"\\frac",
            math,
        )

        math = re.sub(
            r"\\bar\s*\{([^{}]*)\}",
            r"\\overline{\1}",
            math,
        )

        math = re.sub(
            r"\\bar\s*([A-Za-z])(?![A-Za-z])",
            r"\\overline{\1}",
            math,
        )

        return math

    return edit_math_parts(markdown, normalize)


def convert_math(tex: str) -> str:
    """Normalize custom mathematical commands and math delimiters."""

    # \circled{n} -> Unicode circled numbers.
    tex = re.sub(
        r"\\\(((?:\s*\\circled\{\d\}\s*)+)\\\)",
        lambda m: "".join(
            CIRCLED[d]
            for d in re.findall(r"\d", m.group(1))
            if d in CIRCLED
        ),
        tex,
    )

    tex = re.sub(
        r"\\circled\{(\d)\}",
        lambda m: CIRCLED.get(m.group(1), m.group(0)),
        tex,
    )

    # Custom mathematical macros.
    tex = re.sub(
        r"\\bs\s*\{([^{}]*)\}",
        r"\\boldsymbol{\1}",
        tex,
    )

    tex = re.sub(
        r"\\bs\s*([A-Za-z])",
        r"\\boldsymbol{\1}",
        tex,
    )

    tex = re.sub(
        r"\\e(?![A-Za-z])",
        r"\\mathrm{e}",
        tex,
    )

    tex = re.sub(
        r"\\i(?![A-Za-z])",
        "i",
        tex,
    )

    # Display math: \[ ... \] -> $$ ... $$
    tex = re.sub(
        r"\\\[\s*(.*?)\s*\\\]",
        lambda m: "\n\n$$\n" + m.group(1) + "\n$$\n\n",
        tex,
        flags=re.S,
    )

    # Inline math: \( ... \) -> $ ... $
    tex = re.sub(
        r"\\\(\s*(.*?)\s*\\\)",
        lambda m: "$" + m.group(1) + "$",
        tex,
        flags=re.S,
    )

    return tex


# ---------------------------------------------------------
# 7. SHARED REPRESENTATION NORMALIZATION
# ---------------------------------------------------------

def remove_math_emphasis(markdown: str) -> str:
    """Remove accidental Markdown bold around LaTeX commands."""

    return edit_math_parts(
        markdown,
        lambda math: re.sub(
            r"\*\*(\\[a-zA-Z]+)\*\*",
            r"\1",
            math,
        ),
    )


def normalize_punctuation(markdown: str) -> str:
    """Normalize punctuation in prose without changing math."""

    parts = split_math_parts(markdown)

    for i in range(0, len(parts), 2):
        parts[i] = parts[i].translate(PUNCTUATION_MAP)

    return "".join(parts)


def normalize_whitespace(markdown: str) -> str:
    """Normalize insignificant whitespace."""

    markdown = re.sub(
        r"[ \t]+\n",
        "\n",
        markdown,
    )

    markdown = re.sub(
        r"\n{3,}",
        "\n\n",
        markdown,
    )

    return markdown.strip()


# ---------------------------------------------------------
# 8. BLOCK NORMALIZATION
# ---------------------------------------------------------

def clean_block(tex: str) -> str:
    """Convert one exam block into canonical Markdown."""

    tex = convert_images(tex)
    tex = convert_tables(tex)
    tex = convert_lists(tex)
    tex = convert_choices(tex)

    # Empty answer brackets.
    tex = tex.replace(
        "（\\quad）",
        "( )",
    )

    # Fill-in-the-blank markers.
    tex = tex.replace(
        r"\fillinblank{}",
        r"\_\_\_",
    )

    # Remove presentation-only center environments.
    tex = tex.replace(
        r"\begin{center}",
        "",
    )

    tex = tex.replace(
        r"\end{center}",
        "",
    )

    tex = convert_math(tex)

    # Safety net: split any remaining inline choices.
    # This operates only on text outside math expressions.
    tex = split_inline_choices(tex)

    # Shared normalization.
    tex = html.unescape(tex)
    tex = normalize_includegraphics(tex)
    tex = remove_math_emphasis(tex)
    tex = normalize_latex_commands(tex)
    tex = normalize_punctuation(tex)

    return normalize_whitespace(tex)


# ---------------------------------------------------------
# 9. COMPLETE DOCUMENT CONVERSION
# ---------------------------------------------------------

def convert_tex(tex: str, keep_solutions: bool = False) -> str:
    """Convert a complete LaTeX exam document into Markdown."""

    pattern = re.compile(
        r"\\chapter\{(?P<chapter>[^}]*)\}"
        r"|\\section\{(?P<section>[^}]*)\}"
        r"|\\begin\{problem\}(?P<problem>.*?)\\end\{problem\}"
        r"|\\begin\{answer\}(?P<answer>.*?)\\end\{answer\}"
        r"|\\begin\{solution\}(?P<solution>.*?)\\end\{solution\}",
        re.S,
    )

    parts = []
    number = 0

    for match in pattern.finditer(tex):

        if match.group("chapter") is not None:
            parts.append(
                "# " + match.group("chapter").strip()
            )

        elif match.group("section") is not None:
            parts.append(
                "## " + match.group("section").strip()
            )

        elif match.group("problem") is not None:
            number += 1

            parts.append(
                f"{number}. "
                + clean_block(match.group("problem"))
            )

        elif keep_solutions and match.group("answer") is not None:
            parts.append(
                "**答案：** "
                + clean_block(match.group("answer"))
            )

        elif keep_solutions and match.group("solution") is not None:
            parts.append(
                "**解析：**\n\n"
                + clean_block(match.group("solution"))
            )

    return "\n\n".join(parts).strip() + "\n"


# ---------------------------------------------------------
# 10. SCRIPT ENTRY POINT
# ---------------------------------------------------------

def main():
    input_file = (
        "Gaokao-Math-Problems-Compilation/content/2024/"
        "national_paper_a_liberal.tex"
    )

    output_file = "paper/exam_out.md"

    keep_solutions = False

    with open(input_file, encoding="utf-8") as file:
        tex = file.read()

    markdown = convert_tex(
        tex,
        keep_solutions=keep_solutions,
    )

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(markdown)

    print("Wrote", output_file)


if __name__ == "__main__":
    main()
