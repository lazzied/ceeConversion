"""Deterministic style normalization for exam Markdown.

Direction: OCR style -> gold style. Apply the SAME function to gold and OCR
(on gold it only rewrites the gold's own inconsistent outliers).
"""
from pathlib import Path
import html
import re

CJK = r'\u4e00-\u9fff'
CHINESE = re.compile(f'[{CJK}]')

# Protected tokens (odd split indices). Only '$...' tokens are ever edited.
SPLIT_PATTERN = re.compile(
    r'(<table\b.*?</table>'
    r'|\\includegraphics\{[^}]*\}'
    r'|(?m:^\#{1,6}[ \t][^\n]*)'
    r'|\$\$.+?\$\$'
    r'|\$(?:(?!\n[ \t]*\n)[^$])+\$'          # inline math, may span single newlines
    r'|<[^>]+>'
    r'|!\[[^\]]*\]\([^)]*\))',
    re.S | re.I,
)
INLINE = re.compile(r'^\$(?!\$)(.*)\$$', re.S)

UNICODE_TO_LATEX = {
    '∥': r'\parallel', '⊥': r'\perp', '⩾': r'\ge', '≥': r'\ge',
    '⩽': r'\le', '≤': r'\le', '≈': r'\approx', '∈': r'\in',
    '∩': r'\cap', '∪': r'\cup',
}
UNICODE_PATTERN = re.compile('|'.join(re.escape(k) for k in UNICODE_TO_LATEX))


def edit_parts(markdown, edit_text=None, edit_math=None):
    """Apply separate edit functions to ordinary text and math spans."""
    parts = SPLIT_PATTERN.split(markdown)
    for i, part in enumerate(parts):
        if i % 2 == 0 and edit_text:
            parts[i] = edit_text(part)
        elif i % 2 == 1 and part.startswith('$') and edit_math:
            parts[i] = edit_math(part)
    return ''.join(parts)


def edit_text_ctx(markdown, fn):
    """Edit prose between math spans while preserving where each line begins.

    The leading marker lets rules distinguish the start of a line from text
    immediately after a math span, where the same characters can mean something
    different.
    """
    parts = SPLIT_PATTERN.split(markdown)
    for i in range(0, len(parts), 2):
        pre = '$' if i else ''
        parts[i] = fn(pre + parts[i])[len(pre):]
    return ''.join(parts)


# ------------------------------------------------------------------ basics
def decode_html_entities(md):
    """Decode HTML entities without changing the surrounding Markdown."""
    return html.unescape(md)


def convert_figures(md):
    """Standardize supported figure references while preserving their paths."""
    def img(m):
        """Convert one HTML image tag when it contains a usable source path."""
        src = re.search(r'''src\s*=\s*["']([^"']+)["']''', m.group(0), re.I)
        return rf"\includegraphics{{{src.group(1).strip()}}}" if src else m.group(0)
    md = re.sub(r'<img\b[^>]*>', img, md, flags=re.I)
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',
                  lambda m: rf"\includegraphics{{{m.group(2).strip()}}}", md)


def unwrap_divs(md):
    """Remove presentation-only div wrappers around document content."""
    return re.sub(r'</?div\b[^>]*>', '', md, flags=re.I)


def normalize_tables(md):
    """Standardize table cell spacing and layout without changing cell values."""
    def clean(m):
        """Simplify supported table tags while leaving other HTML tags alone."""
        t = re.match(r'<\s*(/?)\s*([a-zA-Z0-9]+)', m.group(0))
        if not t or t.group(2).lower() not in {'table', 'tr', 'td', 'th'}:
            return m.group(0)
        return f"<{t.group(1)}{t.group(2).lower()}>"
    return re.sub(r'<[^>]+>', clean, md)


def replace_unicode_symbols(md):
    """Map known Unicode math symbols to the shared canonical spelling."""
    return edit_parts(
        md,
        edit_text=lambda t: UNICODE_PATTERN.sub(lambda m: '$' + UNICODE_TO_LATEX[m.group()] + '$', t),
    )  # inside math it is handled by canon_math


def replace_degree_sign(md):
    """Normalize degree symbols in text and mathematical expressions."""
    return edit_parts(md, edit_text=lambda t: re.sub(r'(\d+)\s*[°◦]', r'$\1^\\circ$', t))


# --------------------------------------------------------- math-internal
SINGLE_TOKEN = re.compile(r'[A-Za-z0-9]|\\[A-Za-z]+')
FRAC = re.compile(r'\\dfrac(?![A-Za-z])')


def _strip_spaces(s):
    """Remove unnecessary math spacing but keep spaces inside text and commands."""
    def keep(m):
        """Keep a space when removing it would join a command or escape."""
        before, after = s[:m.start()], s[m.end():]
        if before.endswith('\\') and not before.endswith('\\\\'):
            return ' '
        if re.search(r'\\[A-Za-z]+$', before) and re.match(r'[A-Za-z]', after):
            return ' '
        return ''
    pieces = re.split(r'(\\text\{[^{}]*\})', s)
    out = []
    for i, p in enumerate(pieces):
        if i % 2:
            out.append(p)
        else:
            # re-run keep() with full-string context is unnecessary inside a piece
            def keep_p(m, p=p):
                """Apply the same safe spacing rule within this math fragment."""
                before, after = p[:m.start()], p[m.end():]
                if before.endswith('\\') and not before.endswith('\\\\'):
                    return ' '
                if re.search(r'\\[A-Za-z]+$', before) and re.match(r'[A-Za-z]', after):
                    return ' '
                return ''
            out.append(re.sub(r'\s+', keep_p, p))
    return ''.join(out)


def _read_arg(s, i):
    """Read one LaTeX argument at ``i`` and return its text and ending index."""
    while i < len(s) and s[i] == ' ':
        i += 1
    if i >= len(s):
        return None
    if s[i] == '{':
        depth = 0
        for j in range(i, len(s)):
            if s[j] in '{}' and s[j - 1:j] == '\\':
                continue
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], True, j + 1
        return None
    m = re.match(r'\\[A-Za-z]+|.', s[i:], re.S)
    return m.group(), False, i + m.end()


def _canon_fracs(s):
    """Gold style: \\dfrac52 when both args are single tokens, else \\dfrac{..}{..}."""
    result, i = '', 0
    while True:
        m = FRAC.search(s, i)
        if not m:
            return result + s[i:]
        result += s[i:m.start()]
        a = _read_arg(s, m.end())
        b = _read_arg(s, a[2]) if a else None
        if not a or not b:
            result += m.group()
            i = m.end()
            continue
        first, second = (_canon_fracs(t) if br else t for t, br, _ in (a, b))
        if SINGLE_TOKEN.fullmatch(first) and SINGLE_TOKEN.fullmatch(second):
            out = r'\dfrac' + (' ' if first[0].isalpha() else '') + first
            if re.fullmatch(r'\\[A-Za-z]+', first) and second[0].isalpha():
                out += ' '
            out += second
        else:
            out = '\\dfrac{' + first + '}{' + second + '}'
        result += out
        i = b[2]


LR = re.compile(r'\\(left|right)(?![A-Za-z])\s*(.)', re.S)
CLOSER = {'(': ')', '[': ']'}


def _strip_flat_left_right(s):
    """Gold keeps \\left( \\right) only when the inside holds a fraction."""
    stack, pairs = [], []
    for m in LR.finditer(s):
        if m.group(1) == 'left':
            stack.append(m)
        elif stack:
            pairs.append((stack.pop(), m))
    edits = []
    for l, r in pairs:
        if CLOSER.get(l.group(2)) == r.group(2) and not re.search(r'\\dfrac|\\frac', s[l.end():r.start()]):
            edits += [(l.start(), l.end(), l.group(2)), (r.start(), r.end(), r.group(2))]
    for a, b, rep in sorted(edits, reverse=True):
        s = s[:a] + rep + s[b:]
    return s


def canon_math(span):
    """Canonicalize formatting variations inside one mathematical expression."""
    d = '$$' if span.startswith('$$') else '$'
    s = span[len(d):-len(d)]
    s = re.sub(r'\*\*(\\[A-Za-z]+)\*\*', r'\1', s)
    s = ''.join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in s)
    s = UNICODE_PATTERN.sub(lambda m: UNICODE_TO_LATEX[m.group()] + ' ', s)
    s = re.sub(r'(\d)\s*[°◦]', r'\1^\\circ', s)
    s = re.sub(r'\\t?frac(?![A-Za-z])', lambda m: r'\dfrac', s)
    s = re.sub(r'\\(?:overline|bar)\s*\{([A-Za-z])\}', r'\\bar \1', s)
    s = re.sub(r'\\(?:geqslant|geq)(?![A-Za-z])', lambda m: r'\ge', s)
    s = re.sub(r'\\(?:leqslant|leq)(?![A-Za-z])', lambda m: r'\le', s)
    s = re.sub(r'\\(?:left|right)(?![A-Za-z])\s*\|', '|', s)
    s = _strip_flat_left_right(s)
    s = re.sub(r'(?<![A-Za-z\\])e(?=\^)', lambda m: r'\mathrm{e}', s)
    s = _strip_spaces(s)
    s = re.sub(r'([\^_])\{([A-Za-z])\}', r'\1\2', s)
    s = re.sub(r'([\^_])\{(\d)\}(?!\d)', r'\1\2', s)
    s = re.sub(r'([\^_])\{(\\[A-Za-z]+)\}(?![A-Za-z])', r'\1\2', s)
    s = re.sub(r'\\sqrt\{(\d)\}', r'\\sqrt\1', s)
    s = _canon_fracs(s)
    return d + s + d


def canon_math_spans(md):
    """Apply math canonicalization to every recognized Markdown math span."""
    return edit_parts(md, edit_math=canon_math)


# ------------------------------------------------------- prose-level rules
def remove_page_numbers(md):
    """Remove standalone page-number lines introduced by OCR or conversion."""
    return re.sub(r'(?m)^[ \t]*\d{3,4}[ \t]*(?:\n|\Z)', '', md)


def fix_answer_brackets(md):
    """Standardize answer blanks and brackets used in exam questions."""
    return edit_parts(md, edit_text=lambda t: re.sub(r'[ \t]*[（(]\s*[）)]', '（ ）', t))


def fix_fill_in_blanks(md):
    """Convert supported fill-in-blank marks to one stable representation."""
    return edit_parts(md, edit_text=lambda t: re.sub(r'[ \t]*(?:\\_|_){3,}', '___', t))


def wrap_set_literals(md):
    r"""{1, 2, 3} -> $\{1,2,3\}$ (before number wrapping)."""
    pat = re.compile(r'\{\s*([0-9A-Za-z.\-]+(?:\s*[,，]\s*[0-9A-Za-z.\-]+)*)\s*\}')
    return edit_parts(md, edit_text=lambda t: pat.sub(
        lambda m: '$\\{' + re.sub(r'\s*[,，]\s*', ',', m.group(1)) + '\\}$', t))


NUMBER = re.compile(
    r'(?m)(^[ \t]*(?:\d{1,2}[.．、](?!\d)|[（(]\d{1,2}[）)]))'
    r'|(?<![A-Za-z0-9_$\\.])(?<!\d-)(-?\d+(?:\.\d+)?%?)(?![A-Za-z0-9_$])'
)


def wrap_standalone_numbers(md):
    """Mark standalone numbers that belong to math rather than prose."""
    def repl(m):
        """Wrap a matched number in math while leaving question labels intact."""
        if m.group(1):
            return m.group(1)
        n = m.group(2)
        return '$' + (n[:-1] + r'\%' if n.endswith('%') else n) + '$'
    return edit_text_ctx(md, lambda t: NUMBER.sub(repl, t))


def wrap_abs_bars(md):
    r"""|$AB$| -> $|AB|$"""
    return re.sub(r'(?<![\\$\w|])\|[ \t]*\$([^$\n]+)\$[ \t]*\|', r'$|\1|$', md)


OPS = r'[=<>+\-]'


def wrap_bare_operands(md):
    """A bare token glued to a math span by an operator joins that span: 2x - $y$ -> $2x-$$y$."""
    parts = SPLIT_PATTERN.split(md)
    isinl = lambda k: 0 <= k < len(parts) and bool(INLINE.match(parts[k]))
    for i in range(0, len(parts), 2):
        t = parts[i]
        if isinl(i + 1):
            t = re.sub(rf'(?<![\w$\\.])([A-Za-z0-9]{{1,3}})[ \t]*({OPS})[ \t]*$',
                       lambda m: '$' + m.group(1) + m.group(2) + '$', t)
        if isinl(i - 1):
            t = re.sub(rf'^[ \t]*({OPS})[ \t]*([A-Za-z0-9]{{1,3}})(?![A-Za-z0-9])',
                       lambda m: '$' + m.group(1) + m.group(2) + '$', t)
        parts[i] = t
    return ''.join(parts)


REL_ONLY = re.compile(r'^\\(?:ge|le|parallel|perp|approx|in|cap|cup)$')
REL_END = re.compile(r'\\(?:ge|le|parallel|perp|approx|in|cap|cup)$')
JOIN_OP = re.compile(rf'^[ \t]*({OPS})[ \t]*$')


def _glue(a, b):
    """Join math pieces without merging a LaTeX command with the next letter."""
    if re.search(r'\\[A-Za-z]+$', a) and re.match(r'[A-Za-z]', b):
        return a + ' ' + b
    return a + b


def merge_math_spans(md):
    """$a$$b$ , $a$ = $b$ , $a$ \\perp $b$  ->  one span."""
    parts = SPLIT_PATTERN.split(md)
    out = [parts[0]]
    for k in range(1, len(parts), 2):
        tok = parts[k]
        after = parts[k + 1] if k + 1 < len(parts) else ''
        merged = False
        if len(out) >= 3 and INLINE.match(out[-2]) and INLINE.match(tok):
            gap = out[-1]
            a, b = INLINE.match(out[-2]).group(1), INLINE.match(tok).group(1)
            op = JOIN_OP.match(gap)
            call = gap.strip() == '' and re.fullmatch(r'[A-Za-z]', a) and re.match(r'(?:\\left)?\(', b)
            colon = re.fullmatch(r'[ \t]*[:：][ \t]*', gap) and re.fullmatch(r'[A-Za-z]', a) and '=' in b
            coef = gap.strip() == '' and re.search(r'\d$', a) and re.match(r'\\(?:sqrt|pi)', b)
            if gap == '' or op or call or coef or colon or (gap.strip() == '' and (REL_ONLY.match(a) or REL_ONLY.match(b) or REL_END.search(a))):
                glue = op.group(1) if op else (':' if colon else '')
                out[-2] = '$' + _glue(_glue(a, glue) if glue else a, b) + '$'
                out[-1] = after
                merged = True
        if not merged:
            out += [tok, after]
    return ''.join(out)


SYM = r'(?:[A-Za-z](?:_[A-Za-z0-9])?|\\[A-Za-z]+)'
SYMLIST = re.compile(rf'^{SYM}(?:\s*,\s*{SYM})+$')
UNITS = ('件',)


def split_symbol_lists(md):
    r"""$A, B$ -> $A$，$B$  (gold keeps symbol lists as separate spans)."""
    def fix(span):
        """Split a comma-separated list into separate inline math spans."""
        m = INLINE.match(span)
        if m and SYMLIST.match(m.group(1)):
            return '，'.join('$' + x.strip() + '$' for x in m.group(1).split(','))
        return span
    return edit_parts(md, edit_math=fix)


def attach_units(md):
    r"""$150$ 件 -> $150\text{ 件}$"""
    parts = SPLIT_PATTERN.split(md)
    for k in range(1, len(parts) - 1, 2):
        m = INLINE.match(parts[k])
        u = re.match(r'^[ \t]*(' + '|'.join(UNITS) + ')', parts[k + 1])
        if m and u and re.fullmatch(r'[A-Za-z0-9.\\{}^_]+', m.group(1)):
            parts[k] = '$' + m.group(1) + '\\text{ ' + u.group(1) + '}$'
            parts[k + 1] = parts[k + 1][u.end():]
    return ''.join(parts)


def param_note(md):
    r"""$\begin{cases}..\end{cases}$ (t 为参数) -> ...\end{cases}\ (t\text{ 为参数})$"""
    return re.sub(r'\\end\{cases\}\$[ \t]*[（(]\s*([A-Za-z])\s*为参数\s*[）)]',
                  lambda m: '\\end{cases}\\ (' + m.group(1) + '\\text{ 为参数})$', md)


def move_trailing_punct(md):
    """$...;$ -> $...$；  (punctuation belongs outside the span)."""
    parts = SPLIT_PATTERN.split(md)
    for k in range(1, len(parts) - 1, 2):
        m = INLINE.match(parts[k])
        if m and len(m.group(1)) > 1 and m.group(1)[-1] in ';,.' and not m.group(1).endswith('\\,'):
            parts[k + 1] = m.group(1)[-1] + parts[k + 1]
            parts[k] = '$' + m.group(1)[:-1] + '$'
    return ''.join(parts)


def gold_punctuation(md):
    """Prose punctuation in gold style: ， ； ： ？ ！ fullwidth, '.' ASCII."""
    def fix(t):
        """Apply the reference paper's punctuation and spacing conventions."""
        t = t.replace('。', '.').replace('、', '，')
        t = t.replace(',', '，').replace(';', '；').replace(':', '：')
        t = t.replace('?', '？').replace('!', '！')
        t = re.sub(r'(?<![0-9])\.(?=[' + CJK + r'])', '. ', t)
        t = re.sub(r'[ \t]+(?=[，；：？！])', '', t)
        t = re.sub(r'(?<=[，；：？！])[ \t]+', '', t)
        return t
    return edit_parts(md, edit_text=fix)


def space_cjk_math(md):
    """Gold: one space between a Chinese character and an adjacent $span$."""
    parts = SPLIT_PATTERN.split(md)
    for k in range(1, len(parts), 2):
        if not INLINE.match(parts[k]):
            continue
        parts[k - 1] = re.sub(f'([{CJK}])[ \t]*$', r'\1 ', parts[k - 1])
        if k + 1 < len(parts):
            parts[k + 1] = re.sub(f'^[ \t]*(?=[{CJK}])', ' ', parts[k + 1])
    return ''.join(parts)


def fullwidth_parens_around_math(md):
    r"""($a>b>0$) -> （$a>b>0$）"""
    parts = SPLIT_PATTERN.split(md)
    for k in range(1, len(parts) - 1, 2):
        if INLINE.match(parts[k]) and parts[k - 1].endswith('(') and parts[k + 1].startswith(')') \
                and not re.search(r'\([A-D1-9]$', parts[k - 1]):
            parts[k - 1] = parts[k - 1][:-1].rstrip(' \t') + '（'
            parts[k + 1] = '）' + parts[k + 1][1:]
    return ''.join(parts)


def label_spacing(md):
    """Option labels: '（A）x' / '(A)x' -> '(A) x' at line start."""
    return edit_text_ctx(md, lambda t: re.sub(
        r'(?m)^([ \t]*)[（(]([A-D])[）)][ \t]*', r'\1(\2) ', t))


def absorb_trailing_equals(md):
    """$x$ =（ ） / $a$ = ___  ->  $x=$（ ） / $a=$___"""
    parts = SPLIT_PATTERN.split(md)
    for k in range(1, len(parts) - 1, 2):
        m = INLINE.match(parts[k])
        g = re.match(r'^[ \t]*=[ \t]*(?=（ ）|___)', parts[k + 1])
        if m and g:
            parts[k] = '$' + m.group(1) + '=$'
            parts[k + 1] = parts[k + 1][g.end():]
    return ''.join(parts)


def split_options(md):
    """Normalize inline answer choices while keeping their labels and order."""
    opt = r"(?:\([A-D]\)|（[A-D]）)"
    return edit_parts(md, edit_text=lambda t: re.sub(rf"[ \t]+(?={opt})", "\n", t))


def clean_whitespace(md):
    """Clean repeated spaces and excess blank lines without altering content."""
    md = re.sub(r'(?<=\S)[ \t]{2,}', ' ', md)
    md = re.sub(r'(?m)[ \t]+$', '', md)
    return re.sub(r'\n{3,}', '\n\n', md).strip() + '\n'


def wrap_loose_symbols(md):
    """Optionally wrap loose mathematical symbols as inline math spans."""
    loose = re.compile(r'(?<![A-Za-z0-9$\\(])[A-Za-z]{1,4}(?:\s*[,、]\s*[A-Za-z]{1,4})*(?![A-Za-z0-9$])')
    def line(l):
        """Only consider short loose symbols on lines containing Chinese text."""
        if not CHINESE.search(l):
            return l
        return edit_parts(l, edit_text=lambda t: loose.sub(lambda m: '$' + m.group() + '$', t))
    return '\n'.join(line(l) for l in md.split('\n'))


STEPS = [
    decode_html_entities, convert_figures, normalize_tables, unwrap_divs,
    replace_unicode_symbols, replace_degree_sign,
    canon_math_spans,
    wrap_set_literals, remove_page_numbers,
    fix_answer_brackets, fix_fill_in_blanks,
    wrap_standalone_numbers, wrap_abs_bars, wrap_bare_operands,
    merge_math_spans, split_symbol_lists, attach_units, param_note,
    absorb_trailing_equals, move_trailing_punct, gold_punctuation,
    fullwidth_parens_around_math, space_cjk_math,
    label_spacing, split_options, clean_whitespace,
]


def normalize_markdown(markdown, wrap_symbols=False):
    """Style-only normalization. Use the SAME call for gold and OCR.

    wrap_symbols=True adds a content-changing OCR heuristic; keep it False when
    measuring the baseline or building training pairs, and use fix_ocr() later.
    """
    for step in STEPS:
        markdown = step(markdown)
    return wrap_loose_symbols(markdown) if wrap_symbols else markdown


def fix_ocr(markdown):
    """OCR-only repairs (content-changing). Run after the baseline is measured."""
    return clean_whitespace(wrap_loose_symbols(markdown))


if __name__ == "__main__":
    src, dst = Path("data/input/exam.md"), Path("data/output/exam.md")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(normalize_markdown(src.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Written to {dst}")
