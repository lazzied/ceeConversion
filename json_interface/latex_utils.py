"""Small, dependency-free LaTeX scanning helpers (brace matching, environments,
tabular parsing, layout-noise stripping)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ------------------------------------------------------------- comments / prep
_FIGURE_HINT = re.compile(
    r"^[ \t]*%[ \t]*\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}.*$", re.M)


def preserve_figure_hints(text: str) -> str:
    """A commented-out \\includegraphics line is the only record of the real
    image path, so turn it into a \\figsrc{path} marker before comments go."""
    return _FIGURE_HINT.sub(r"\\figsrc{\1}", text)


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


# ------------------------------------------------------------------- grouping
def skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def read_group(s: str, i: int, open_: str = "{", close: str = "}") -> tuple[str, int]:
    """Read a balanced group starting at (or after whitespace from) index i.
    Returns (inner_text, index_after_group). Escaped braces are ignored."""
    i = skip_ws(s, i)
    if i >= len(s) or s[i] != open_:
        raise ValueError(f"expected {open_!r} at {i}: {s[i:i + 30]!r}")
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError(f"unbalanced {open_}{close} starting at {i}")


def read_groups(s: str, i: int, n: int) -> tuple[list[str], int]:
    out = []
    for _ in range(n):
        g, i = read_group(s, i)
        out.append(g)
    return out, i


def split_top(s: str, sep: str) -> list[str]:
    """Split on '&' or the row break '\\\\' at brace depth 0."""
    parts: list[str] = []
    depth, i, last = 0, 0, 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            if sep == "\\\\" and s[i:i + 2] == "\\\\" and depth == 0:
                parts.append(s[last:i])
                i += 2
                last = i
                continue
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif sep == "&" and c == "&" and depth == 0:
            parts.append(s[last:i])
            last = i + 1
        i += 1
    parts.append(s[last:])
    return parts


# --------------------------------------------------------------- environments
@dataclass
class Segment:
    kind: str                 # "text" | "env"
    text: str                 # text, or env body (still including its arguments)
    name: Optional[str] = None


def split_top_level(s: str, names: list[str]) -> list[Segment]:
    """Split ``s`` into text runs and top-level environments with the given
    names. Nested environments of the same name are matched correctly, and
    anything nested inside an environment stays inside its body."""
    alt = "|".join(re.escape(n) for n in names)
    begin = re.compile(r"\\begin\{(%s)\}" % alt)
    segs: list[Segment] = []
    pos = 0
    while True:
        m = begin.search(s, pos)
        if not m:
            break
        name = m.group(1)
        pat = re.compile(r"\\(begin|end)\{%s\}" % re.escape(name))
        depth, end_m = 1, None
        for e in pat.finditer(s, m.end()):
            depth += 1 if e.group(1) == "begin" else -1
            if depth == 0:
                end_m = e
                break
        if end_m is None:
            raise ValueError(f"\\begin{{{name}}} at {m.start()} is never closed")
        if m.start() > pos:
            segs.append(Segment("text", s[pos:m.start()]))
        segs.append(Segment("env", s[m.end():end_m.start()], name))
        pos = end_m.end()
    if pos < len(s):
        segs.append(Segment("text", s[pos:]))
    return segs


# ------------------------------------------------------------------- tabulars
_RULE = re.compile(r"\\(toprule|midrule|bottomrule|hline)(?![A-Za-z])")


def parse_tabular(env_body: str) -> tuple[str, list[list[str]], list[int]]:
    """Return (column_spec, rows, inner_rules). ``inner_rules`` lists after how
    many rows a midrule/hline occurs (top and bottom rules are dropped)."""
    spec, k = read_group(env_body, 0)
    rows: list[list[str]] = []
    rules: list[int] = []
    for chunk in split_top(env_body[k:], "\\\\"):
        for m in _RULE.finditer(chunk):
            if m.group(1) in ("midrule", "hline") and rows and (not rules or rules[-1] != len(rows)):
                rules.append(len(rows))
        chunk = _RULE.sub("", chunk).strip()
        if not chunk:
            continue
        rows.append([c.strip() for c in split_top(chunk, "&")])
    rules = [r for r in rules if r != len(rows)]
    return spec, rows, rules


# -------------------------------------------------------------- text cleaning
_LAYOUT = [re.compile(p) for p in (
    r"\\vspace\*?\{[^{}]*\}",
    r"\\noindent(?![A-Za-z])",
    r"\\par(?![A-Za-z])",
    r"\\nopagebreak(?:\[[^\]]*\])?",
    r"\\centering(?![A-Za-z])",
    r"\\renewcommand\{\\arraystretch\}\{[^{}]*\}",
    r"\\(?:smallskip|medskip|bigskip)(?![A-Za-z])",
)]

_MATH_SPAN = re.compile(r"(?<!\\)\$.*?(?<!\\)\$", re.S)
_INLINE_MATH_SPAN = re.compile(r"(?<!\\)(?<!\$)\$(?!\$).*?(?<!\\)\$(?!\$)", re.S)
_STRAY_SUPERSCRIPT = re.compile(r"(?m)^[ \t]*[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾]+[ \t]*(?:\n|$)")
_MATH_CONNECTOR = re.compile(r"^[ \t\d.;:+*/=<>!|&^_()\[\]{}\\-]*$")
_MATH_PREFIX = re.compile(r"(?:[A-Za-z0-9]+(?:\s*[-+*/=<>]\s*[A-Za-z0-9]+)*\s*[-+*/=<>]\s*)$")
_MATH_SUFFIX = re.compile(r"^(?:\s*(?:[-+*/=<>]|\|)\s*(?:[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?))+")
_OPTION_MATH = re.compile(r"(?:\\[A-Za-z]+|[{}_^=<>+*/]|\d|\|)")


def normalize_math_markup(text: str) -> str:
    """Repair and join inline math fragments without changing surrounding prose.

    OCR sometimes preserves the dollar signs but drops a closing parenthesis
    from a coordinate or grouped expression. When a math span has only
    unmatched opening delimiters, their obvious closing partners are restored
    at the end of that span. A mismatched or extra closing delimiter is left
    alone for the renderer's path-aware validation to report.
    """
    text = balance_inline_math_groups(text)
    text = re.sub(r"\|\s*\$([^$]+)\$\s*\|", lambda match: "$|" + match.group(1) + "|$", text)
    spans = list(_MATH_SPAN.finditer(text))
    if len(spans) > 1:
        parts: list[str] = []
        cursor = 0
        index = 0
        while index < len(spans):
            current = spans[index]
            start, end = current.start(), current.end()
            body = current.group()[1:-1]
            while index + 1 < len(spans):
                gap = text[end:spans[index + 1].start()]
                if not _MATH_CONNECTOR.fullmatch(gap):
                    break
                body += gap + spans[index + 1].group()[1:-1]
                end = spans[index + 1].end()
                index += 1
            prefix = _MATH_PREFIX.search(text[cursor:start])
            suffix = _MATH_SUFFIX.match(text[end:])
            if prefix:
                start = cursor + prefix.start()
            if start > cursor and text[start - 1] in "|([":
                start -= 1
            if suffix:
                end += suffix.end()
            if end < len(text) and text[end] in ")]|":
                end += 1
            parts.append(text[cursor:start] + "$" + text[start:end].replace("$", "").strip() + "$")
            cursor = end
            index += 1
        parts.append(text[cursor:])
        return "".join(parts)

    if spans:
        match = spans[0]
        before = text[:match.start()]
        after = text[match.end():]
        prefix = _MATH_PREFIX.search(before)
        suffix = _MATH_SUFFIX.match(after)
        start = prefix.start() if prefix else match.start()
        end = match.end() + (suffix.end() if suffix else 0)
        if start > 0 and text[start - 1] in "|([":
            start -= 1
        if end < len(text) and text[end] in ")]|":
            end += 1
        if prefix or suffix:
            return text[:start] + "$" + text[start:end].replace("$", "").strip() + "$" + text[end:]
    return text


def balance_inline_math_groups(text: str) -> str:
    """Close unambiguous open ``()``, ``[]``, or ``{}`` groups inside math.

    This is deliberately conservative: if a closer is missing, extra, or in
    the wrong order, the original span is preserved so downstream validation
    can fail with the field path instead of guessing at the expression.
    Escaped delimiters such as ``\\{`` are LaTeX symbols, not group syntax.
    """
    closing_for = {"(": ")", "[": "]", "{": "}"}
    opening_for = {closing: opening for opening, closing in closing_for.items()}

    def balance_span(match: re.Match[str]) -> str:
        span = match.group(0)
        expression = span[1:-1]
        stack: list[str] = []
        index = 0

        while index < len(expression):
            character = expression[index]
            if character == "\\":
                # Skip the escaped character, but continue scanning ordinary
                # command names so argument braces such as in \\frac{a}{b}
                # remain visible to the group balancer.
                if index + 1 < len(expression) and expression[index + 1] in r"{}[]()\\":
                    index += 2
                    continue
                index += 1
                continue

            if character in closing_for:
                stack.append(character)
            elif character in opening_for:
                if not stack or stack[-1] != opening_for[character]:
                    return span
                stack.pop()
            index += 1

        if not stack:
            return span
        missing_closers = "".join(closing_for[opening] for opening in reversed(stack))
        return f"${expression}{missing_closers}$"

    return _INLINE_MATH_SPAN.sub(balance_span, text)


def normalize_option_math(text: str) -> str:
    """Keep compact answer expressions in math mode, including escaped sets."""
    text = normalize_math_markup(text)
    if re.fullmatch(r"\s*_{3,}\s*", text):
        return r"\_\_\_"
    if "$" in text or not _OPTION_MATH.search(text) or re.search(r"[\u3400-\u9fff]", text):
        return text
    text = re.sub(r"(?<!\\)([{}])", r"\\\1", text.strip())
    return f"${text}$"


def clean_text(s: str) -> str:
    """Drop layout-only commands and normalise whitespace; keeps paragraph
    breaks (blank lines) and leaves math / content macros untouched."""
    for pat in _LAYOUT:
        s = pat.sub("", s)
    s = _STRAY_SUPERSCRIPT.sub("", s)
    paras = re.split(r"\n\s*\n", s)
    paras = [normalize_math_markup(" ".join(p.split())) for p in paras]
    return "\n\n".join(p for p in paras if p)


_STYLE = re.compile(
    r"\\(?:small|footnotesize|large|Large|LARGE|huge|Huge|bfseries|itshape|centering)(?![A-Za-z])"
    r"|\\color\{[^}]*\}|\\vspace\*?\{[^}]*\}|\\par(?![A-Za-z])|\\\s|\\,")


def strip_styling(s: str) -> str:
    """For titles/headers: remove font/colour commands and control spaces."""
    s = _STYLE.sub(" ", s)
    return " ".join(s.split())
