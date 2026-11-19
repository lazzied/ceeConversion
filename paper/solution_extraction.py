"""Extract answer and solution sections from a gold LaTeX exam."""

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CIRCLED = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}


def read_group(s, i):
    """s[i] is '{'. Return (content, index after the matching '}')."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def make_figure(path):
    """Format an image path as the figure command used by the Markdown parser."""
    return f"\\includegraphics{{{path.strip()}}}"


def convert_images(t):
    """Convert the source's custom image commands to standard image references."""
    # images inside answer choices
    t = re.sub(r"\\choicebitmap(?:\[[^\]]*\])?\{([^}]*)\}",
               lambda m: make_figure(m.group(1)), t)
    # main figure of a problem
    t = re.sub(r"\\bitmapfigure(?:\[[^\]]*\])?\{([^}]*)\}",
               lambda m: "\n\n" + make_figure(m.group(1)) + "\n\n", t)
    return t


def convert_choices(t):
    """\\choices {..}{..}{..}{..}  ->  (A) .. (B) .. """
    result, pos = "", 0
    while True:
        k = t.find("\\choices", pos)
        if k < 0:
            return result + t[pos:]
        result += t[pos:k]
        j = k + len("\\choices")
        options = []
        while len(options) < 4:
            j2 = j + len(t[j:]) - len(t[j:].lstrip())
            if j2 < len(t) and t[j2] == "{":
                content, j = read_group(t, j2)
                options.append(content.strip())
            else:
                break
        items = [f"({'ABCD'[n]}) {o}" for n, o in enumerate(options)]
        is_short = all(len(o) < 28 and "\\includegraphics" not in o for o in options)
        sep = "  " if is_short else "\n\n"
        result += "\n\n" + sep.join(items) + "\n"
        pos = j


def convert_tables(t):
    """Convert supported tabular environments to HTML table markup."""
    def to_html(m):
        """Convert one matched tabular body to HTML rows and cells."""
        rows = m.group(1).replace("\\hline", "").split("\\\\")
        html = "<table>"
        for row in rows:
            if not row.strip():
                continue
            html += "<tr>"
            for cell in row.split("&"):
                cell = cell.replace("\\fillinblank{}", "")
                cell = re.sub(r"\\text\{([^}]*)\}", r"\1", cell)
                html += f"<td>{cell.strip()}</td>"
            html += "</tr>"
        return "\n\n" + html + "</table>\n\n"

    return re.sub(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", to_html, t, flags=re.S)


def convert_lists(t):
    """Convert circled and numbered lists to readable Markdown lines."""
    def circle_list(m):
        """Render each source list item with a circled number."""
        items = m.group(1).split("\\item")[1:]
        lines = [f"{CIRCLED[str(n)]} {x.strip()}" for n, x in enumerate(items, 1)]
        return "\n\n" + "\n\n".join(lines) + "\n\n"

    def number_list(m):
        """Render each enumerate item with a simple numbered label."""
        items = m.group(1).split("\\item")[1:]
        lines = [f"({n}) {x.strip()}" for n, x in enumerate(items, 1)]
        return "\n\n" + "\n\n".join(lines) + "\n\n"

    t = re.sub(r"\\begin\{circlelist\}(.*?)\\end\{circlelist\}", circle_list, t, flags=re.S)
    t = re.sub(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", number_list, t, flags=re.S)
    return t


def convert_math(t):
    """Expand source-specific math macros and convert math delimiters."""
    # \circled{n} -> unicode circled numbers
    t = re.sub(r"\\\(((?:\s*\\circled\{\d\}\s*)+)\\\)",
               lambda m: "".join(CIRCLED[d] for d in re.findall(r"\d", m.group(1))), t)
    t = re.sub(r"\\circled\{(\d)\}", lambda m: CIRCLED[m.group(1)], t)

    # custom math macros -> standard LaTeX
    t = re.sub(r"\\bs\s*\{([^}]*)\}", r"\\boldsymbol{\1}", t)
    t = re.sub(r"\\bs\s*([A-Za-z])", r"\\boldsymbol{\1}", t)
    t = re.sub(r"\\e(?![A-Za-z])", r"\\mathrm{e}", t)
    t = re.sub(r"\\i(?![A-Za-z])", "i", t)

    # math delimiters
    t = re.sub(r"\\\[\s*(.*?)\s*\\\]", lambda m: "\n\n$$\n" + m.group(1) + "\n$$\n\n", t, flags=re.S)
    t = re.sub(r"\\\(\s*(.*?)\s*\\\)", lambda m: "$" + m.group(1) + "$", t, flags=re.S)
    return t


def clean_block(t):
    """Clean one answer or solution block using the supported TeX conversions."""
    t = convert_images(t)
    t = convert_tables(t)
    t = convert_lists(t)
    t = convert_choices(t)
    t = t.replace("（\\quad）", "（ ）")
    t = t.replace("\\fillinblank{}", "___")
    t = t.replace("\\begin{center}", "").replace("\\end{center}", "")
    t = convert_math(t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def convert_solutions(tex):
    """Keep titles, question numbers, answers and solutions. Drop problem statements."""
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
            parts.append(f"### {number}.")
        elif m.group("answer") is not None:
            parts.append("**答案：** " + clean_block(m.group("answer")))
        elif m.group("solution") is not None:
            parts.append("**解析：**\n\n" + clean_block(m.group("solution")))
    return "\n\n".join(parts) + "\n"


def main(input_path=PROJECT_ROOT / "Gaokao-Math-Problems-Compilation/content/2024/national_paper_a_liberal.tex",
         output_path=PROJECT_ROOT / "paper/results/solutions/solutions_out.md"):
    """Extract solutions from the paths provided as function arguments."""
    input_file, output_file = Path(input_path), Path(output_path)

    with open(input_file, encoding="utf-8") as f:
        tex = f.read()
    md = convert_solutions(tex)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote", output_file)


if __name__ == "__main__":
    main()
