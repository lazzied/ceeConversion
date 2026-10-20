import re
import unicodedata

CJK_BREAK_AFTER = "，。；：？！、"


def display_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def is_cjk(c: str) -> bool:
    return "\u4e00" <= c <= "\u9fff"


def visual_len(s: str) -> int:
    """Rough rendered length of a short LaTeX snippet (for column choice)."""
    s = re.sub(r"\\[A-Za-z]+", "x", s)
    s = re.sub(r"[${}^_\s]", "", s)
    return len(s)


def split_text_pieces(text: str) -> list[tuple[str, str]]:
    """Split text into (chunk, separator-that-follows) at safe break points:
    ASCII spaces outside math, and after full-width punctuation that is
    followed by another CJK character."""
    out: list[tuple[str, str]] = []
    cur: list[str] = []
    in_math = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\":                          # keep escapes (\$, \ , \{ ...) whole
            cur.append(text[i:i + 2])
            i += 2
            continue
        if c == "$":
            in_math = not in_math
            cur.append(c)
        elif not in_math and c == " ":
            out.append(("".join(cur), " "))
            cur = []
        else:
            cur.append(c)
            if not in_math and c in CJK_BREAK_AFTER and i + 1 < len(text):
                nxt = text[i + 1]
                if is_cjk(nxt):                  # newline between CJK chars is ignored
                    out.append(("".join(cur), ""))
                    cur = []
                elif nxt == "$":                  # needs a trailing % to swallow the newline
                    out.append(("".join(cur), "%"))
                    cur = []
        i += 1
    out.append(("".join(cur), ""))
    return [(t, s) for t, s in out if t]


def wrap_paragraph(text: str, width: int, first: str, cont: str) -> list[str]:
    """Greedy wrap; ``first``/``cont`` are the prefixes of the first and the
    following lines. A separator of "%" means: break here only with a
    trailing % (so the newline cannot turn into a space)."""
    if width <= 0:
        return [first + text]
    lines: list[str] = []
    line, has_content, prev = first, False, ""
    for chunk, sep in split_text_pieces(text):
        glue = "" if prev == "%" else prev
        if has_content and display_width(line + glue + chunk) > width:
            lines.append(line + ("%" if prev == "%" else ""))
            line, has_content = cont, False
        else:
            line += glue
        line += chunk
        has_content, prev = True, sep
    lines.append(line)
    return lines
