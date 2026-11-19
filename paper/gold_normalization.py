"""Convert gold-standard LaTeX exam documents into canonical Markdown.
Structure conversion lives here. ALL style normalization (dfrac, punctuation,
spacing, blanks, brackets, number wrapping ...) is delegated to
normalize_md.normalize_markdown, the same function the OCR side uses.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import html
import re

from structural_fixes.normalization import normalize_markdown, edit_parts  # shared, single source of truth

CIRCLED = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}


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


# ---------------------------------------------------------
# 2. FIGURES
# ---------------------------------------------------------

def make_figure(path: str) -> str:
    """Convert an image path to the shared Markdown figure representation."""
    return rf"\includegraphics{{{path.strip()}}}"


def convert_images(tex: str) -> str:
    """Convert supported TeX image commands to Markdown figure references."""
    tex = re.sub(r"\\choicebitmap(?:\[[^\]]*\])?\{([^}]*)\}",
                 lambda m: make_figure(m.group(1)), tex)
    return re.sub(r"\\bitmapfigure(?:\[[^\]]*\])?\{([^}]*)\}",
                  lambda m: "\n\n" + make_figure(m.group(1)) + "\n\n", tex)


def normalize_includegraphics(markdown: str) -> str:
    """Drop optional image arguments (same canonical form as the OCR side)."""
    return re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}",
                  lambda m: make_figure(m.group(1)), markdown)


# ---------------------------------------------------------
# 3. TABLES
# ---------------------------------------------------------

def convert_tables(tex: str) -> str:
    """Convert supported TeX tabular environments into HTML tables."""
    def to_html(match: re.Match[str]) -> str:
        """Turn one matched tabular body into rows and cells."""
        content = re.sub(r"\\(?:hline|toprule|midrule|bottomrule)", "", match.group(1))
        rows = re.split(r"\\\\(?:\s*\[[^\]]*\])?", content)
        out = ["<table>"]
        for row in rows:
            if not row.strip():
                continue
            out.append("<tr>")
            for cell in row.split("&"):
                cell = cell.replace(r"\fillinblank{}", "___")
                cell = re.sub(r"\\text\{([^{}]*)\}", r"\1", cell)
                out.append(f"<td>{cell.strip()}</td>")
            out.append("</tr>")
        out.append("</table>")
        return "\n\n" + "".join(out) + "\n\n"

    return re.sub(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
                  to_html, tex, flags=re.S)


# ---------------------------------------------------------
# 4. CHOICES (one option per line; the shared normalizer enforces the same)
# ---------------------------------------------------------

def convert_choices(tex: str) -> str:
    """Convert common multiple-choice TeX layouts into Markdown text."""
    result, pos, command = [], 0, r"\choices"
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
        if len(options) != 4:                     # never silently destroy a command
            result.append(tex[k:j])
            pos = j
            continue
        items = [f"({'ABCD'[n]}) {o}" for n, o in enumerate(options)]
        result.append("\n\n" + "\n".join(items) + "\n")
        pos = j
    return "".join(result)

# ---------------------------------------------------------
# 5. LISTS
# ---------------------------------------------------------

def convert_lists(tex: str) -> str:
    """Convert supported enumerate and itemize environments to Markdown."""
    def circle_list(m):
        """Render list items with the source's circled-number labels."""
        items = m.group(1).split(r"\item")[1:]
        lines = [f"{CIRCLED[str(n)]} {x.strip()}" if str(n) in CIRCLED else x.strip()
                 for n, x in enumerate(items, 1)]
        return "\n\n" + "\n\n".join(lines) + "\n\n"

    def number_list(m):
        """Render each enumerate item with a simple numbered label."""
        items = m.group(1).split(r"\item")[1:]
        return "\n\n" + "\n\n".join(f"({n}) {x.strip()}" for n, x in enumerate(items, 1)) + "\n\n"

    tex = re.sub(r"\\begin\{circlelist\}(.*?)\\end\{circlelist\}", circle_list, tex, flags=re.S)
    return re.sub(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", number_list, tex, flags=re.S)


# ---------------------------------------------------------
# 6. SOURCE-SPECIFIC MACROS AND DELIMITERS (not style: these are the .tex dialect)
# ---------------------------------------------------------

def convert_math(tex: str) -> str:
    """Normalize supported TeX math environments to Markdown delimiters."""
    tex = re.sub(r"\\\(((?:\s*\\circled\{\d\}\s*)+)\\\)",
                 lambda m: "".join(CIRCLED[d] for d in re.findall(r"\d", m.group(1)) if d in CIRCLED),
                 tex)
    tex = re.sub(r"\\circled\{(\d)\}", lambda m: CIRCLED.get(m.group(1), m.group(0)), tex)

    tex = re.sub(r"\\bs\s*\{([^{}]*)\}", r"\\boldsymbol{\1}", tex)
    tex = re.sub(r"\\bs\s*([A-Za-z])", r"\\boldsymbol{\1}", tex)
    tex = re.sub(r"\\e(?![A-Za-z])", lambda m: r"\mathrm{e}", tex)
    tex = re.sub(r"\\i(?![A-Za-z])", "i", tex)

    tex = re.sub(r"\\\[\s*(.*?)\s*\\\]",
                 lambda m: "\n\n$$\n" + m.group(1) + "\n$$\n\n", tex, flags=re.S)
    return re.sub(r"\\\(\s*(.*?)\s*\\\)", lambda m: "$" + m.group(1) + "$", tex, flags=re.S)


# ---------------------------------------------------------
# 7. ONE BLOCK
# ---------------------------------------------------------

def clean_block(tex: str, *, normalize: bool = True) -> str:
    """Remove layout commands and clean one TeX block for Markdown output."""
    tex = convert_images(tex)
    tex = convert_tables(tex)
    tex = convert_lists(tex)
    tex = convert_choices(tex)

    tex = tex.replace("（\\quad）", "（ ）")
    tex = tex.replace(r"\fillinblank{}", "___")
    tex = tex.replace(r"\begin{center}", "").replace(r"\end{center}", "")

    tex = convert_math(tex)
    tex = html.unescape(tex)
    tex = normalize_includegraphics(tex)

    # Everything style-related happens here, identical to the OCR side.
    return (normalize_markdown(tex) if normalize else tex).strip()


# ---------------------------------------------------------
# 8. WHOLE DOCUMENT
# ---------------------------------------------------------

def convert_tex(tex: str, keep_solutions: bool = False, *, normalize: bool = True) -> str:
    """Convert a gold TeX exam to Markdown while preserving reading order."""
    pattern = re.compile(
        r"\\chapter\{(?P<chapter>[^}]*)\}"
        r"|\\section\{(?P<section>[^}]*)\}"
        r"|\\begin\{problem\}(?P<problem>.*?)\\end\{problem\}"
        r"|\\begin\{answer\}(?P<answer>.*?)\\end\{answer\}"
        r"|\\begin\{solution\}(?P<solution>.*?)\\end\{solution\}",
        re.S,
    )
    parts, number = [], 0
    for m in pattern.finditer(tex):
        if m.group("chapter") is not None:
            parts.append("# " + m.group("chapter").strip())
        elif m.group("section") is not None:
            parts.append("## " + m.group("section").strip())
        elif m.group("problem") is not None:
            number += 1
            parts.append(f"{number}. " + clean_block(m.group("problem"), normalize=normalize))
        elif keep_solutions and m.group("answer") is not None:
            parts.append("**答案：** " + clean_block(m.group("answer"), normalize=normalize))
        elif keep_solutions and m.group("solution") is not None:
            parts.append("**解析：**\n\n" + clean_block(m.group("solution"), normalize=normalize))
    return "\n\n".join(parts).strip() + "\n"


def main(input_path=PROJECT_ROOT / "Gaokao-Math-Problems-Compilation/content/2024/national_paper_a_liberal.tex",
         output_path=PROJECT_ROOT / "paper/example_01_exam_markdown.md", *, keep_solutions=False):
    """Convert a reference TeX file using editable function arguments."""
    input_file, output_file = Path(input_path), Path(output_path)
    with open(input_file, encoding="utf-8") as f:
        tex = f.read()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(convert_tex(tex, keep_solutions=keep_solutions))
    print("Wrote", output_file)


if __name__ == "__main__":
    main()
