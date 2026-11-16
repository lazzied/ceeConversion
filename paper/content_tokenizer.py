"""Whitespace-free character and LaTeX-aware tokenization."""

from __future__ import annotations

import re


MATH_SEGMENT = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\$(?!\$)(.+?)\$", re.S)
DISPLAY_SEGMENT = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.S)
CJK_CHAR = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
LATIN_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
MATH_ATOM = re.compile(r"\\[A-Za-z]+|\\.|[A-Za-z]|\d|[^\s]")


def _text_tokens(text):
    result = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        cjk = CJK_CHAR.match(text, i)
        latin = LATIN_WORD.match(text, i)
        number = NUMBER.match(text, i)
        match = cjk or latin or number
        if match:
            value = match.group(0)
            category = "cjk_text" if cjk else "latin_text" if latin else "other_text"
            result.append({"token": value, "category": category})
            i = match.end()
        else:
            result.append({"token": text[i], "category": "other_text"})
            i += 1
    return result


def _math_tokens(text, open_delimiter=None, close_delimiter=None):
    result = []
    if open_delimiter:
        result.append({"token": open_delimiter, "category": "math_syntax"})
    result.extend({"token": match.group(0), "category": "math"} for match in MATH_ATOM.finditer(text))
    if close_delimiter:
        result.append({"token": close_delimiter, "category": "math_syntax"})
    return result


def tokenize_content(text):
    """Tokenize mixed Markdown; CJK is character-wise and math is LaTeX-aware."""
    text = str(text or "")
    result, cursor = [], 0
    for match in MATH_SEGMENT.finditer(text):
        result.extend(_text_tokens(text[cursor:match.start()]))
        if match.group(1) is not None:
            formula, open_delimiter, close_delimiter = match.group(1), "$$", "$$"
        elif match.group(2) is not None:
            formula, open_delimiter, close_delimiter = match.group(2), r"\[", r"\]"
        else:
            formula, open_delimiter, close_delimiter = match.group(3), "$", "$"
        result.extend(_math_tokens(formula, open_delimiter, close_delimiter))
        cursor = match.end()
    result.extend(_text_tokens(text[cursor:]))
    return result


def character_records(text):
    """Return visible characters labelled by CJK/Latin/math/other content class."""
    records = []
    cursor = 0
    for match in MATH_SEGMENT.finditer(str(text or "")):
        records.extend(_chars(str(text or "")[cursor:match.start()], "text"))
        formula = next((value for value in match.groups() if value is not None), "")
        if match.group(1) is not None:
            delimiters = "$$$$"
        elif match.group(2) is not None:
            delimiters = r"\[\]"
        else:
            delimiters = "$$"
        records.extend(_chars(delimiters[: len(delimiters) // 2] + formula + delimiters[len(delimiters) // 2 :], "math"))
        cursor = match.end()
    records.extend(_chars(str(text or "")[cursor:], "text"))
    return records


def _chars(text, mode):
    result = []
    for char in text:
        if char.isspace():
            continue
        if mode == "math":
            category = "math"
        elif CJK_CHAR.fullmatch(char):
            category = "cjk_text"
        elif char.isascii() and char.isalpha():
            category = "latin_text"
        else:
            category = "other_text"
        result.append({"token": char, "category": category})
    return result


def display_equations(text):
    """Find block/display equations, including a whole paragraph wrapped inline."""
    text = str(text or "")
    result = []
    for match in DISPLAY_SEGMENT.finditer(text):
        equation = match.group(1) if match.group(1) is not None else match.group(2)
        result.append({"text": equation.strip(), "start": match.start(), "end": match.end()})
    if not result:
        stripped = text.strip()
        whole = re.fullmatch(r"\$(?!\$)(.+)\$", stripped, re.S)
        if whole:
            result.append({"text": whole.group(1).strip(), "start": 0, "end": len(text)})
    return result
