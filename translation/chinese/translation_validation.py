"""Validation and normalization helpers for translated exam strings."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

CJK = re.compile(r"[\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323AF]")
CJK_RUN = re.compile(r"[\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323AF]+")
MATH_SPAN = re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$.*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{(?P<environment>equation\*?|align\*?|gather\*?|multline\*?|cases|array|matrix|pmatrix|bmatrix|vmatrix|Vmatrix)\}.*?\\end\{(?P=environment)\}", re.S)
MATH_ATOM = re.compile(r"\\[A-Za-z]+|\\.|[A-Za-z]+|\d+(?:\.\d+)?|[^\s,;.{}]")
PRESERVED_FIELDS = {"raw", "source_file", "source_path", "warnings"}
FULLWIDTH_PUNCTUATION = str.maketrans({"，": ",", "。": ".", "：": ":", "；": ";", "？": "?", "、": ",", "（": "(", "）": ")"})
CHINESE_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10_000}
ENGLISH_ORDINAL_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


def is_preserved_path(path: str) -> bool:
    """Return whether a JSON string is metadata/source that must stay exact."""
    fields = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)(?=\.|\[|$)", path)
    return any(field in PRESERVED_FIELDS for field in fields)


def chinese_spans(text: str) -> list[tuple[int, int]]:
    """Return Chinese runs outside math (kept for callers and final checks)."""
    math_ranges = [(match.start(), match.end()) for match in MATH_SPAN.finditer(text)]
    spans: list[tuple[int, int]] = []
    for match in CJK_RUN.finditer(text):
        if not any(start <= match.start() < end for start, end in math_ranges):
            spans.append((match.start(), match.end()))
    return spans


def has_chinese_in_math(text: str) -> bool:
    """Return whether Chinese occurs inside a math span."""
    return any(CJK.search(match.group()) for match in MATH_SPAN.finditer(text))


def math_atoms(text: str) -> Counter[str]:
    """Collect math commands, identifiers, numbers, and symbols independent of span layout."""
    atoms: Counter[str] = Counter()
    for match in MATH_SPAN.finditer(text):
        expression = match.group()
        if expression.startswith("$$") and expression.endswith("$$"):
            expression = expression[2:-2]
        elif expression.startswith("$") and expression.endswith("$"):
            expression = expression[1:-1]
        elif expression.startswith(r"\(") and expression.endswith(r"\)"):
            expression = expression[2:-2]
        elif expression.startswith(r"\[") and expression.endswith(r"\]"):
            expression = expression[2:-2]
        else:
            expression = re.sub(r"^\\begin\{[^}]+\}|\\end\{[^}]+\}$", "", expression)
        atoms.update(MATH_ATOM.findall(expression))
    return atoms


def math_atom_difference(source: str, translated: str) -> str | None:
    """Describe missing math atoms while allowing equivalent span grouping."""
    expected, actual = math_atoms(source), math_atoms(translated)
    missing = list((expected - actual).elements())
    added = list((actual - expected).elements())
    if not missing:
        return None
    return f"missing math atoms={missing}; expected={sorted(expected.elements())}; actual={sorted(actual.elements())}"


def added_math_atoms(source: str, translated: str) -> list[str]:
    """Return output-only math atoms for informational diagnostics."""
    return list((math_atoms(translated) - math_atoms(source)).elements())


def normalize_translated_text(text: str) -> str:
    """Normalize English punctuation outside math, including translated headings."""
    chunks: list[str] = []
    position = 0
    for match in MATH_SPAN.finditer(text):
        chunks.append(normalize_plain_text(text[position:match.start()]))
        chunks.append(match.group())
        position = match.end()
    chunks.append(normalize_plain_text(text[position:]))
    return "".join(chunks)


def normalize_plain_text(text: str) -> str:
    """Normalize one plain-text region; this function never receives math."""
    text = normalize_ordinal_heading(text)
    text = text.translate(FULLWIDTH_PUNCTUATION)
    text = re.sub(r"::+", ":", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r",\s*、\s*|、\s*,\s*", ",", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
    ordinal = re.fullmatch(r"\s*([零〇一二两三四五六七八九十百千万]+|[A-Za-z]+)\s*", text)
    if ordinal:
        word = ordinal.group(1).lower()
        value = ENGLISH_ORDINAL_WORDS.get(word)
        if value is None:
            value = parse_chinese_number(ordinal.group(1))
        if value is not None:
            text = roman_numeral(value)
    return text


def parse_chinese_number(value: str) -> int | None:
    """Parse common Chinese section ordinals, including values above ten."""
    if not value or any(char not in CHINESE_DIGITS and char not in CHINESE_UNITS for char in value):
        return None
    if not any(char in CHINESE_UNITS for char in value):
        return int("".join(str(CHINESE_DIGITS[char]) for char in value))
    total = 0
    section = 0
    digit = 0
    for char in value:
        if char in CHINESE_DIGITS:
            digit = CHINESE_DIGITS[char]
        elif char == "万":
            section += digit
            total += section * 10_000
            section = digit = 0
        else:
            unit = CHINESE_UNITS[char]
            section += (digit or (1 if unit == 10 else 0)) * unit
            digit = 0
    return total + section + digit


def roman_numeral(number: int) -> str:
    """Convert a positive ordinal to a conventional Roman numeral."""
    if number <= 0:
        return str(number)
    values = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    result: list[str] = []
    for value, symbol in values:
        count, number = divmod(number, value)
        result.append(symbol * count)
    return "".join(result)


def normalize_ordinal_heading(text: str) -> str:
    """Render a leading Chinese ordinal heading as a Roman numeral heading."""
    match = re.match(r"^(\s*)([零〇一二两三四五六七八九十百千万]+|[A-Za-z]+|[IVXLCDM]+)、\s*(.+)$", text, re.S | re.I)
    if not match:
        return text
    token = match.group(2)
    value = ENGLISH_ORDINAL_WORDS.get(token.lower())
    if value is None and re.fullmatch(r"[IVXLCDM]+", token, re.I):
        value = roman_to_integer(token)
    if value is None:
        value = parse_chinese_number(token)
    if value is None:
        return text
    return f"{match.group(1)}{roman_numeral(value)}. {match.group(3)}"


def roman_to_integer(token: str) -> int:
    """Parse a Roman numeral used as a section ordinal."""
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(token.upper()):
        value = values[character]
        total += -value if value < previous else value
        previous = max(previous, value)
    return total


def validation_errors(source: str, translated: Any) -> list[str]:
    """Explain why one complete translated field is invalid."""
    if not isinstance(translated, str):
        return ["translation is missing or is not a string"]
    if not translated.strip():
        return ["translation is empty"]
    errors: list[str] = []
    if CJK.search(translated):
        errors.append("Chinese characters remain")
    atom_difference = math_atom_difference(source, translated)
    if atom_difference:
        errors.append(atom_difference)
    return errors


def extract_non_chinese_tokens(text: str) -> list[str]:
    """Extract ASCII identifiers/numbers outside math for content checks."""
    unprotected_text = MATH_SPAN.sub(" ", text)
    return re.findall(r"[A-Za-z]+|\d+(?:[.,]\d+)*", unprotected_text)


def find_untranslated_strings(data: Any) -> list[tuple[str, str]]:
    """Find translatable Chinese left in string values, excluding protected paths/math."""
    findings: list[tuple[str, str]] = []
    def visit(value: Any, path: str) -> None:
        if isinstance(value, str) and not is_preserved_path(path) and chinese_spans(value):
            findings.append((path, value))
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
    visit(data, "$")
    return findings


def validate_final_json(original: Any, translated: Any) -> list[str]:
    """Compare structure and math while allowing prose and punctuation to translate."""
    errors: list[str] = []
    def compare(source: Any, output: Any, path: str) -> None:
        if isinstance(source, dict):
            if not isinstance(output, dict) or source.keys() != output.keys():
                errors.append(f"{path}: JSON object structure changed")
                return
            for key in source:
                compare(source[key], output[key], f"{path}.{key}")
        elif isinstance(source, list):
            if not isinstance(output, list) or len(source) != len(output):
                errors.append(f"{path}: list structure changed")
                return
            for index, (left, right) in enumerate(zip(source, output)):
                compare(left, right, f"{path}[{index}]")
        elif isinstance(source, str):
            if not isinstance(output, str):
                errors.append(f"{path}: string value changed type")
            elif is_preserved_path(path) or not chinese_spans(source):
                if source != output:
                    errors.append(f"{path}: preserved string was modified")
            else:
                errors.extend(f"{path}: {problem}" for problem in validation_errors(source, output))
        elif source != output:
            errors.append(f"{path}: non-string value changed")
    compare(original, translated, "$")
    for path, text in find_untranslated_strings(translated):
        errors.append(f"{path}: Chinese remains in translatable text: {text[:80]!r}")
    return errors


def is_valid(source: str, translated: Any) -> bool:
    """Compatibility helper retained for older callers."""
    return not validation_errors(source, translated)
