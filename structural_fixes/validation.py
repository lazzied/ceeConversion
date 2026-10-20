import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import fitz  # pip install pymupdf

MATH = re.compile(r'\$[^$\n]+\$')
CHINESE = re.compile(r'[\u4e00-\u9fff]')
QUESTION_START = re.compile(r'^(\d{1,2})\. ')
OPTION_LABEL = re.compile(r'\(([A-D])\)')
IMAGE_PATH = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}|!\[[^\]]*\]\(([^)]+)\)')
UNCONVERTED_SYMBOLS = '∥⊥⩾⩽≥≤°◦'
MATH_FUNCTIONS = r'\\(sin|cos|tan|ln|log|lg|lim|max|min|exp)\b'


def make_issue(severity, line, question, check, message, snippet=""):
    return {"severity": severity, "line": line, "question": question,
            "check": check, "message": message, "snippet": snippet}


# ======================================================================
# 1. Line checks: take one line, return a list of messages
# ======================================================================

def check_dollar_signs(line):
    messages = []
    if '$$' in line:
        messages.append("'$$' found (two math spans merged)")
    if line.count('$') % 2:
        messages.append("odd number of '$' (unclosed math)")
    return messages


def check_math_brackets(line):
    messages = []
    for math in MATH.findall(line):
        inner = re.sub(r'\\[{}]', '', math.strip('$'))      # \{ \} are literal braces
        if inner.count('(') != inner.count(')'):
            messages.append(f"unbalanced ( ) in {math}")
        if inner.count('{') != inner.count('}'):
            messages.append(f"unbalanced {{ }} in {math}")
        if inner.count('\\left') != inner.count('\\right'):
            messages.append(f"\\left / \\right mismatch in {math}")
        if inner.count('\\begin') != inner.count('\\end'):
            messages.append(f"\\begin / \\end mismatch in {math}")
    return messages


def check_stray_glyph(line):
    if re.search(r'\$\s*[A-Za-z]_\{\\(?:down|up)arrow\}\s*\$', line):
        return ["stray axis glyph from a figure (like y_{\\downarrow})"]
    return []


def check_leftover_html(line):
    if re.search(r'</?(?:div|table|tr|td|th|img|span|br)\b', line):
        return ["leftover HTML tag"]
    return []


def check_unicode_symbols(line):
    found = sorted({char for char in line if char in UNCONVERTED_SYMBOLS})
    return [f"unconverted symbols: {' '.join(found)}"] if found else []


def check_slash_parallel(line):
    if re.search(r'(?<![:/])//(?!/)', line):
        return ["'//' probably means \\parallel"]
    return []


def check_page_number(line):
    if re.fullmatch(r'\s*\d{3,4}\s*', line):
        return ["line is only a page number"]
    return []


def check_k_table(line):
    if r'\frac{P(K' in line and not line.lstrip().startswith('|'):
        return ["K² table row not converted to a markdown table"]
    return []


def check_options_on_one_line(line):
    if len(OPTION_LABEL.findall(MATH.sub(' ', line))) > 1:
        return ["several options on one line"]
    return []


def check_math_padding(line):
    for math in MATH.findall(line):
        inner = math[1:-1]
        if inner != inner.strip():
            return [f"spaces inside the $ delimiters: {math}"]
    return []


def check_fullwidth_in_math(line):
    for math in MATH.findall(line):
        if re.search(r'[\uff01-\uff5e\u3000]', math):
            return [f"full-width character inside math: {math}"]
    return []


def check_blank_style(line):
    if re.search(r'(?<![\\_])_{3,}', MATH.sub(' ', line)):
        return ["fill-in blank is not in \\_\\_\\_ form"]
    return []


def check_stem_brackets(line):
    odd = [m for m in re.findall(r'[（(]\s*[）)]', MATH.sub(' ', line)) if m != '（ ）']
    return ["answer brackets are not （ ）"] if odd else []


def check_loose_symbols(line):
    """Warning: Latin symbols in a Chinese line that are outside $...$."""
    if not CHINESE.search(line):
        return []
    text = re.sub(r'<[^>]+>|\([A-D]\)', ' ', MATH.sub(' ', line))
    symbols = re.findall(r'(?<![A-Za-z0-9\\(])[A-Za-z]{1,4}(?![A-Za-z0-9])', text)
    return [f"loose symbols outside $...$: {', '.join(symbols[:6])}"] if symbols else []


LINE_CHECKS = [   # (function, severity)
    (check_dollar_signs, "error"),
    (check_math_brackets, "error"),
    (check_stray_glyph, "error"),
    (check_leftover_html, "error"),
    (check_unicode_symbols, "error"),
    (check_slash_parallel, "error"),
    (check_page_number, "error"),
    (check_k_table, "error"),
    (check_options_on_one_line, "warning"),
    (check_math_padding, "warning"),
    (check_fullwidth_in_math, "warning"),
    (check_blank_style, "warning"),
    (check_stem_brackets, "warning"),
    (check_loose_symbols, "warning"),
]


# ======================================================================
# 2. Exam checks: need the whole document
# ======================================================================

def parse_questions(lines):
    """Find every question: its number, section heading, first and last line."""
    questions, section, current = [], "", None
    for line_no, line in enumerate(lines, start=1):
        is_heading = line.startswith('#')
        match = QUESTION_START.match(line)
        if (is_heading or match) and current:
            current["last_line"] = line_no - 1
            current = None
        if is_heading:
            section = line.lstrip('#').strip()
        if match:
            current = {"number": int(match.group(1)), "first_line": line_no,
                       "last_line": len(lines), "section": section}
            questions.append(current)
    return questions


def question_text(question, lines):
    return '\n'.join(lines[question["first_line"] - 1:question["last_line"]])


def check_sections(lines):
    issues = []
    headings = [line for line in lines if line.startswith('#')]
    for name in ("单选题", "填空题", "解答题"):
        count = sum(name in heading for heading in headings)
        if count != 1:
            issues.append(make_issue("error", None, None, "sections",
                                     f"section '{name}' found {count} times (expected 1)"))
    return issues


def check_question_numbers(questions, expected_total):
    issues, previous = [], 0
    for position, question in enumerate(questions, start=1):
        if question["number"] not in (position, previous + 1):
            issues.append(make_issue(
                "error", question["first_line"], question["number"], "numbering",
                f"question printed as {question['number']} at position {position} "
                f"(previous was {previous})"))
        previous = question["number"]
    if expected_total and len(questions) != expected_total:
        issues.append(make_issue("error", None, None, "numbering",
                                 f"found {len(questions)} questions, expected {expected_total}"))
    return issues


def check_options(questions, lines):
    """Every multiple-choice question must have exactly one each of (A) (B) (C) (D)."""
    issues = []
    for question in questions:
        if "单选题" not in question["section"]:
            continue
        text = MATH.sub(' ', question_text(question, lines))
        counts = Counter(OPTION_LABEL.findall(text))
        missing = [label for label in "ABCD" if counts[label] == 0]
        repeated = [label for label in "ABCD" if counts[label] > 1]
        if missing or repeated:
            issues.append(make_issue(
                "error", question["first_line"], question["number"], "options",
                f"missing options: {missing or '-'}; repeated options: {repeated or '-'}"))
    return issues


def check_fill_in_blanks(questions, lines):
    issues = []
    for question in questions:
        if "填空题" in question["section"]:
            text = question_text(question, lines)
            if r'\_\_\_' not in text and '___' not in text:
                issues.append(make_issue("error", question["first_line"], question["number"],
                                         "blank", "fill-in question has no blank"))
    return issues


def check_sub_questions(questions, lines):
    """In 解答题, sub-questions must read (1), (2), (3)... in order."""
    issues = []
    for question in questions:
        if "解答题" not in question["section"]:
            continue
        numbers = [int(m.group(1)) for line in question_text(question, lines).split('\n')
                   if (m := re.match(r'\((\d)\)', line))]
        if numbers and numbers != list(range(1, len(numbers) + 1)):
            issues.append(make_issue("error", question["first_line"], question["number"],
                                     "sub_questions", f"sub-question order is {numbers}"))
    return issues


def check_tables(lines):
    issues, start = [], None
    for line_no, line in enumerate(lines + [""], start=1):
        is_row = line.lstrip().startswith('|')
        if is_row and start is None:
            start = line_no
        if not is_row and start is not None:
            rows = lines[start - 1:line_no - 1]
            widths = {len(re.findall(r'(?<!\\)\|', row)) for row in rows}
            if len(widths) > 1:
                issues.append(make_issue("error", start, None, "table",
                                         "rows have different column counts"))
            if len(rows) < 2 or not re.fullmatch(r'[\s|:\-]+', rows[1]):
                issues.append(make_issue("error", start, None, "table",
                                         "second row is not a |---| separator"))
            start = None
    return issues


def check_images(lines, base_dir):
    issues = []
    for line_no, line in enumerate(lines, start=1):
        for match in IMAGE_PATH.finditer(line):
            path = match.group(1) or match.group(2)
            if not (base_dir / path).exists():
                issues.append(make_issue("error", line_no, None, "image",
                                         f"image file not found: {path}"))
    return issues


# ======================================================================
# 3. PDF comparison (only if the PDF has a text layer)
# ======================================================================

def extract_pdf_text(pdf_path, min_chars_per_page=30):
    """Return the PDF text layer, or None if the PDF is scanned (no text layer)."""
    pages = [page.get_text() for page in fitz.open(pdf_path)]
    average = sum(len(page.strip()) for page in pages) / max(len(pages), 1)
    return None if average < min_chars_per_page else '\n'.join(pages)


def markdown_for_comparison(markdown):
    """Remove everything that exists only in the markdown (image tags, LaTeX wrappers)."""
    markdown = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}|!\[[^\]]*\]\([^)]*\)|<[^>]+>',
                      ' ', markdown)
    return re.sub(r'\\(?:begin|end)\{[^}]*\}', ' ', markdown)


def count_letters_and_digits(text):
    text = re.sub(MATH_FUNCTIONS, r'\1', text)       # \sin -> sin (the PDF shows "sin")
    text = re.sub(r'\\[A-Za-z]+', ' ', text)         # drop other LaTeX commands
    return Counter(re.findall(r'[A-Za-z0-9]', text))


def compare_chinese(pdf_text, markdown):
    """Diff only the Chinese characters. Catches dropped, added or misread words."""
    pdf_chars = ''.join(CHINESE.findall(pdf_text))
    md_chars = ''.join(CHINESE.findall(markdown))
    issues = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, pdf_chars, md_chars, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        issues.append(make_issue(
            "warning", None, None, "pdf_chinese",
            f"PDF has '{pdf_chars[i1:i2]}', markdown has '{md_chars[j1:j2]}' "
            f"after '{pdf_chars[max(0, i1 - 8):i1]}'"))
    return issues


def compare_letters_and_digits(pdf_text, markdown):
    """Compare how many times each letter/digit occurs. Order-independent, so split
    fractions in the PDF text don't matter. Catches wrong or missing digits and symbols."""
    pdf_counts = count_letters_and_digits(pdf_text)
    md_counts = count_letters_and_digits(markdown)
    differences = sorted(
        ((char, pdf_counts[char], md_counts[char])
         for char in set(pdf_counts) | set(md_counts) if pdf_counts[char] != md_counts[char]),
        key=lambda item: -abs(item[1] - item[2]))
    return [make_issue("warning", None, None, "pdf_characters",
                       f"'{char}': PDF {pdf_n}x, markdown {md_n}x")
            for char, pdf_n, md_n in differences[:15]]


def compare_with_pdf(markdown, pdf_path):
    pdf_text = extract_pdf_text(pdf_path)
    if pdf_text is None:
        return [], "PDF comparison skipped: PDF is scanned (no text layer)."
    markdown = markdown_for_comparison(markdown)
    issues = compare_chinese(pdf_text, markdown) + compare_letters_and_digits(pdf_text, markdown)
    return issues, "PDF comparison done (warnings only: PDF text order is noisy)."


# ======================================================================
# 4. Report + orchestrator
# ======================================================================

def format_report(issues, markdown_path, pdf_note):
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    out = [f"Validation report: {markdown_path}",
           f"{len(errors)} errors, {len(warnings)} warnings", pdf_note, ""]
    for issue in sorted(issues, key=lambda i: (i["severity"] != "error", i["line"] or 10**9)):
        where = f"line {issue['line']}" if issue["line"] else "document"
        question = f"Q{issue['question']}" if issue["question"] else "-"
        out.append(f"[{issue['severity'].upper()}] {where} | {question} | "
                   f"{issue['check']} | {issue['message']}")
        if issue["snippet"]:
            out.append(f"    {issue['snippet']}")
    return '\n'.join(out) + '\n'


def validate_exam(
    markdown_path,
    pdf_path=None,
    report_path=None,
    expected_questions=None,
    *,
    write_report=True,
):
    """Run Markdown checks and optionally write their standalone text report.

    ``write_report=False`` lets the main pipeline collect these findings in its
    combined JSON validation report without creating a second report file.
    """
    markdown_path = Path(markdown_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    lines = markdown.split('\n')
    questions = parse_questions(lines)
    issues = []

    # 1. line checks
    current_question = None
    for line_no, line in enumerate(lines, start=1):
        if line.startswith('#'):
            current_question = None
        match = QUESTION_START.match(line)
        if match:
            current_question = int(match.group(1))
        for check, severity in LINE_CHECKS:
            for message in check(line):
                issues.append(make_issue(severity, line_no, current_question,
                                         check.__name__.removeprefix("check_"),
                                         message, line.strip()[:80]))

    # 2. exam checks
    issues += check_sections(lines)
    issues += check_question_numbers(questions, expected_questions)
    issues += check_options(questions, lines)
    issues += check_fill_in_blanks(questions, lines)
    issues += check_sub_questions(questions, lines)
    issues += check_tables(lines)
    issues += check_images(lines, markdown_path.parent)

    # 3. PDF comparison
    pdf_note = "No PDF given: comparison skipped."
    if pdf_path:
        pdf_issues, pdf_note = compare_with_pdf(markdown, pdf_path)
        issues += pdf_issues

    # 4. report
    if write_report:
        report_path = Path(report_path or markdown_path.with_suffix(".report.txt"))
        report_path.write_text(format_report(issues, markdown_path, pdf_note), encoding="utf-8")
        print(f"{sum(i['severity'] == 'error' for i in issues)} errors, "
              f"{sum(i['severity'] == 'warning' for i in issues)} warnings -> {report_path}")
    return issues


if __name__ == "__main__":
    validate_exam(
        markdown_path="data/output/exam.md",
        pdf_path="data/input/exam.pdf",
        expected_questions=23,
    )
