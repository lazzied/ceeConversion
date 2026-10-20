"""Exam JSON -> readable, standard LaTeX.

This module is self-contained: it knows only the JSON format described in
SCHEMA.md (no imports from the parser package, standard library only).

Design goals for the generated .tex
-----------------------------------
* Preamble is split into labelled blocks (packages, metadata, colours, page
  style, lists, macros) and only loads/defines what the exam actually uses.
* The body uses *semantic* environments instead of layout hacks::

      \\begin{questions}      numbered questions (continues across sections)
        \\item stem
        \\begin{tasks}(4)     A-D options, 4 / 2 / 1 columns picked by length
        \\begin{subquestions} (1) (2) ... , nests as (a) (b)
        \\begin{statements}   ①②③④ proposition lists
        \\begin{reference}   unframed reference material (e.g. the 附 table)

* Tables use booktabs with column-aligned ``&`` so they read well as source.
* Long lines are wrapped at safe points only (never inside ``$...$`` and
  never where a space would change CJK typesetting).

All strings coming from the JSON are treated as LaTeX source (that is what the
parser stores), so they are written out verbatim and not escaped.
"""
from __future__ import annotations

import json
import html
import re
from pathlib import Path
from typing import Any, Optional, Union

from latex_converter.models import SUPPORTED_SCHEMA, ExamFormatError, RenderOptions
from latex_converter.preamble import build_preamble
from latex_converter.utils import CJK_BREAK_AFTER, display_width, visual_len, wrap_paragraph

INDENT = "  "
DEFAULT_FIGURE_WIDTH_CM = 8.0
MAX_FIGURE_WIDTH_CM = 15.5
TASKS_AVAILABLE_WIDTH_CM = 15.5
TASKS_COLUMN_GAP_CM = 1.0
MIN_STANDALONE_FIGURE_FRACTION = 0.25
MAX_STANDALONE_FIGURE_FRACTION = 0.75
DEFAULT_STANDALONE_FIGURE_FRACTION = 0.4
DEFAULT_STANDALONE_FIGURE_HEIGHT_CM = 3.0
TEXTUAL_MATH_SUFFIXES = ("axis", "coordinate", "coordinates")
FLOATING_FIGURE = re.compile(r"\\begin\{figure\}(?:\[[^\]]*\])?(.*?)\\end\{figure\}", re.S)
GRAPHIC_COMMAND = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}")
DISPLAY_ENVIRONMENT_START = re.compile(r"\\begin\{(cases|aligned|(?:p|b|B|v|V)matrix)\}")
MATH_RELATION = re.compile(
    r"(?:=|(?<!\\)[<>]|\\(?:parallel|perp|leq(?:slant)?|geq(?:slant)?|neq|approx|in|notin|subset(?:eq)?|supseteq)\b)"
)
SUBQUESTION_MARKER = re.compile(r"^\((\d+|[a-z])\)\s*(.*)$", re.S)
STATEMENT_MARKER = re.compile(r"^([①②③④⑤⑥⑦⑧⑨⑩])\s*(.*)$", re.S)
STRUCTURED_LINE = re.compile(r"(?m)(?=^[ \t]*(?:\(\d+\)|\([a-z]\)|[①②③④⑤⑥⑦⑧⑨⑩])(?:[ \t]+|$))")
STRAY_SUPERSCRIPT_LINE = re.compile(r"(?m)^[ \t]*[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾]+[ \t]*(?:\n|$)")
MATH_SEGMENT = re.compile(r"(?<!\\)\$.*?(?<!\\)\$", re.S)
MATH_OR_DISPLAY_SEGMENT = re.compile(r"(?<!\\)\$.*?(?<!\\)\$|\\\[.*?\\\]", re.S)
MATHISH_OPTION = re.compile(r"(?:\\[A-Za-z]+|[{}_^=<>+*/]|\d|\|)")
MATH_CONNECTOR = re.compile(r"^[ \t\d,;:+*/=<>!|&^_()\[\]{}\\-]*$")
CJK_FONT_PATTERN = re.compile(r"CJK|WenQuanYi|AR PL|Noto (?:Sans|Serif) CJK", re.I)

# ------------------------------------------------------------------- renderer
class LatexRenderer:
    def __init__(self, options: Optional[RenderOptions] = None):
        self.opt = options or RenderOptions()
        self.uses: set[str] = set()
        self.notes: list[str] = []

    # ---- public --------------------------------------------------------
    def render(self, exam: Union[dict, str, Path]) -> str:
        """``exam`` is the exam JSON as a dict, a JSON string, or a file path."""
        data = self.load(exam)
        self.validate_math_delimiters(data)
        self.uses = set()
        self.notes = []

        body = self.render_body(data)
        body_text = "\n".join(body)
        for macro, flag in ((r"\blank", "blank"), (r"\choiceparen", "choiceparen")):
            if macro in body_text:
                self.uses.add(flag)
        if re.search("[\u2460-\u24ff]", body_text):      # ① ② ... need the CJK font
            self.uses.add("circled")
        text = self.build_preamble(data) + "\n".join(body) + "\n"
        return re.sub(r"\n{3,}", "\n\n", text)          # never more than one blank line

    @staticmethod
    def validate_math_delimiters(data: Any) -> None:
        """Fail before rendering if any source field has unmatched dollar math."""
        def visit(value: Any, path: str) -> None:
            if isinstance(value, str):
                dollars = re.findall(r"(?<!\\)\$", value)
                if len(dollars) % 2:
                    raise ExamFormatError(f"unbalanced $ math delimiter at {path}")
                for expression in MATH_SEGMENT.findall(value):
                    stack: list[str] = []
                    pairs = {")": "(", "]": "[", "}": "{"}
                    for character in expression[1:-1]:
                        if character in "([{":
                            stack.append(character)
                        elif character in ")]}" and (not stack or stack.pop() != pairs[character]):
                            raise ExamFormatError(f"unbalanced {character!r} delimiter in math at {path}")
                    if stack:
                        raise ExamFormatError(f"unbalanced {stack[-1]!r} delimiter in math at {path}")
            elif isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")
        visit(data, "$")

    # ---- input -----------------------------------------------------------
    @staticmethod
    def load(source: Union[dict, str, Path]) -> dict:
        """Accept a dict, a JSON string or a path; validate the essentials."""
        if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith("{")):
            source = json.loads(Path(source).read_text(encoding="utf-8"))
        elif isinstance(source, str):
            source = json.loads(source)
        if not isinstance(source, dict):
            raise ExamFormatError("exam JSON must be an object")
        version = str(source.get("schema_version", SUPPORTED_SCHEMA))
        if version.split(".")[0] != SUPPORTED_SCHEMA.split(".")[0]:
            raise ExamFormatError(f"unsupported schema_version {version!r} "
                                  f"(this renderer reads {SUPPORTED_SCHEMA}.x)")
        sections = source.get("sections")
        if not isinstance(sections, list):
            raise ExamFormatError("missing 'sections' list")
        for s in sections:
            if not isinstance(s, dict):
                raise ExamFormatError(f"section must be an object: {s!r:.80}")
            if not isinstance(s.get("title"), str) or not isinstance(s.get("questions", []), list):
                raise ExamFormatError(f"malformed section: {s!r:.80}")
            for q in s.get("questions", []):
                if not isinstance(q, dict) or "number" not in q:
                    raise ExamFormatError(f"question without 'number' in section {s['title']!r}")
        return source

    def render_file(self, json_path: Union[str, Path], out_path: Union[str, Path]) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render(Path(json_path)), encoding="utf-8")
        return out

    # ---- document body ---------------------------------------------------
    def render_body(self, data: dict) -> list[str]:
        meta = data.get("metadata") or {}
        lines = [r"\begin{document}", ""]
        if meta.get("title"):
            lines += [r"\makeexamtitle", ""]
        if data.get("instructions"):
            lines += self.render_paragraphs(data["instructions"], "") + [""]
        for section in data.get("sections", []):
            lines += self.render_section(section)
        lines.append(r"\end{document}")
        return lines

    def render_section(self, s: dict) -> list[str]:
        rule = "%" + "=" * 66
        ordinal = str(s.get("ordinal") or "")
        separator = ". " if re.fullmatch(r"[IVXLCDM]+", ordinal, re.I) else "、"
        heading = f"{ordinal}{separator}{s['title']}" if ordinal else s["title"]
        heading = self.normalize_latex_text(heading)
        lines = [rule, rf"\section*{{{heading}}}", rule, ""]
        if s.get("description"):
            lines += self.render_paragraphs(s["description"], "") + [""]

        questions = s.get("questions", [])
        if not questions:
            return lines
        first = questions[0]["number"]
        lines.append(r"\begin{questions}" + (f"[start={first}]" if first != 1 else ""))
        expected = first
        for q in questions:
            lines += ["", f"{INDENT}% Question {q['number']}"]
            if q["number"] != expected:        # numbering gap: force the counter
                lines.append(rf"{INDENT}\setcounter{{questionsi}}{{{q['number'] - 1}}}")
            expected = q["number"] + 1
            need = self.get_required_space_cm(q)
            if need:                       # tall options: keep stem and options on one page
                self.uses.add("needspace")
                lines.append(f"{INDENT}\\Needspace{{{need}cm}}")
            lines += self.render_item(q.get("stem", ""), q, level=1, depth=0)
        lines += ["", r"\end{questions}", ""]
        return lines

    @staticmethod
    def get_required_space_cm(q: dict) -> Optional[int]:
        """Rough height (cm) of a question whose options are figures; None for
        ordinary questions, which are short enough not to need protection."""
        figs = [o["figure"] for o in q.get("options", []) if o.get("figure")]
        if not figs:
            return None
        height = max((f.get("height_cm") or 3) for f in figs)
        rows = 2 if len(figs) > 2 and height * len(figs) > 8 else 1
        return int(1.5 + rows * (height + 0.8) + 0.5) + 1

    # ---- a list item (question or sub-question) ----------------------------
    def render_item(self, text: str, node: dict, level: int, depth: int) -> list[str]:
        ind = INDENT * level
        inner = INDENT * (level + 1)
        lines: list[str] = []
        parent_item_started = False
        subquestion_depth = 0
        after_content = False

        def close_subquestions() -> None:
            nonlocal subquestion_depth
            while subquestion_depth:
                lines.append(INDENT * (level + subquestion_depth) + r"\end{subquestions}")
                subquestion_depth -= 1

        def append_text(content: str) -> None:
            nonlocal parent_item_started, after_content, subquestion_depth
            events = self.parse_structured_text(content)
            figure_events = [event for event in events if event[0] == "figure"]
            events = [event for event in events if event[0] != "figure"]
            first_subquestion = next((i for i, event in enumerate(events) if event[0] == "subquestion"), len(events))
            events[first_subquestion:first_subquestion] = figure_events
            for event_type, event_value in events:
                if event_type == "paragraph":
                    if not event_value.strip():
                        continue
                    if subquestion_depth:
                        event_indent = INDENT * (level + subquestion_depth + 1)
                        lines.extend(self.render_paragraphs(event_value, event_indent))
                    else:
                        first = "" if parent_item_started else r"\item "
                        lines.extend(self.render_paragraphs(event_value, ind, first=first, cont=inner))
                        parent_item_started = True
                    after_content = True
                elif event_type == "subquestion":
                    marker_depth, content_text = event_value
                    if marker_depth == 2 and subquestion_depth == 0:
                        # A nested label without a numeric parent is ordinary text.
                        first = "" if parent_item_started else r"\item "
                        lines.extend(self.render_paragraphs(f"({content_text[0]}) {content_text[1]}", ind, first=first, cont=inner))
                        parent_item_started = True
                        continue
                    if not parent_item_started:
                        lines.append(ind + r"\item")
                        parent_item_started = True
                    while subquestion_depth > marker_depth:
                        lines.append(INDENT * (level + subquestion_depth) + r"\end{subquestions}")
                        subquestion_depth -= 1
                    while subquestion_depth < marker_depth:
                        env_indent = INDENT * (level + subquestion_depth + 1)
                        lines.append(env_indent + r"\begin{subquestions}")
                        subquestion_depth += 1
                    item_indent = INDENT * (level + marker_depth + 1)
                    lines.extend(self.render_paragraphs(content_text[1], item_indent, first=r"\item ", cont=item_indent + INDENT))
                    parent_item_started = True
                    after_content = True
                elif event_type == "statements":
                    statement_block = {"type": "statement_list", "items": event_value}
                    block_indent = INDENT * (level + subquestion_depth + 1) if subquestion_depth else inner
                    if not subquestion_depth and not parent_item_started:
                        lines.append(ind + r"\item")
                        parent_item_started = True
                    lines.extend(self.render_statements(statement_block, block_indent))
                    after_content = True
                elif event_type == "shared":
                    close_subquestions()
                    if after_content:
                        lines.append("")
                    lines.extend(self.render_paragraphs(event_value, inner))
                    after_content = True
                elif event_type == "figure":
                    if after_content:
                        lines.append("")
                    lines.extend(self.render_centered_graphic(event_value, inner))
                    after_content = True

        append_text(text)
        if not parent_item_started and (
            node.get("blocks") or node.get("options") or node.get("sub_questions") or node.get("footer_blocks")
        ):
            lines.append(ind + r"\item")
            parent_item_started = True
        for block in node.get("blocks", []):
            if after_content:
                lines.append("")
            if not parent_item_started and not subquestion_depth:
                lines.append(ind + r"\item")
                parent_item_started = True
            if block.get("type") == "text":
                append_text(block.get("content", ""))
            else:
                block_indent = INDENT * (level + subquestion_depth + 1) if subquestion_depth else inner
                lines.extend(self.render_block(block, block_indent))
                after_content = True

        close_subquestions()
        if node.get("options"):
            if not parent_item_started:
                lines.append(ind + r"\item")
                parent_item_started = True
            if after_content:
                lines.append("")
            lines.extend(self.render_options(node["options"], inner))
            after_content = False
        if node.get("sub_questions"):
            if after_content:
                lines.append("")
            lines += self.render_subquestions(
                node["sub_questions"], level + 1, depth)
        for block in node.get("footer_blocks", []):
            lines += [""] + self.render_block(block, inner)
        if not lines:
            lines.append(ind + r"\item")
        return lines

    @staticmethod
    def parse_structured_text(text: str) -> list[tuple[str, Any]]:
        """Split prose into parent text, subquestions, shared tails and figures."""
        events: list[tuple[str, Any]] = []
        pending_statements: list[dict[str, str]] = []

        def flush_statements() -> None:
            if pending_statements:
                events.append(("statements", pending_statements.copy()))
                pending_statements.clear()

        figures: list[str] = []

        def extract_figure(match: re.Match[str]) -> str:
            graphic = GRAPHIC_COMMAND.search(match.group(1))
            if graphic:
                figures.append(graphic.group(0))
                return "\n\n"
            return match.group(0)

        text_without_figures = FLOATING_FIGURE.sub(extract_figure, text or "")
        paragraphs = re.split(r"\n([ \t]*)\n+", text_without_figures)
        paragraph_records = [
            (paragraph, len(paragraph) - len(paragraph.lstrip(" \t")))
            for paragraph in paragraphs[::2]
        ]
        last_top_level_marker = max(
            (index for index, (paragraph, _) in enumerate(paragraph_records)
             if (marker := SUBQUESTION_MARKER.match(paragraph.strip())) and marker.group(1).isdigit()),
            default=-1,
        )
        last_marker_indent = (
            paragraph_records[last_top_level_marker][1] if last_top_level_marker >= 0 else 0
        )
        for paragraph_index, (paragraph, indentation) in enumerate(paragraph_records):
            for segment in STRUCTURED_LINE.split(paragraph):
                segment = segment.strip()
                if not segment:
                    continue
                subquestion = SUBQUESTION_MARKER.match(segment)
                statement = STATEMENT_MARKER.match(segment)
                if subquestion:
                    flush_statements()
                    label = subquestion.group(1)
                    marker_depth = 1 if label.isdigit() else 2
                    events.append(("subquestion", (marker_depth, (label, subquestion.group(2).strip()))))
                elif statement:
                    pending_statements.append({"label": statement.group(1), "text": statement.group(2).strip()})
                else:
                    flush_statements()
                    event_type = (
                        "shared"
                        if last_top_level_marker >= 0
                        and paragraph_index > last_top_level_marker
                        and indentation <= last_marker_indent
                        else "paragraph"
                    )
                    events.append((event_type, segment))
        flush_statements()
        if figures:
            first_subquestion = next((i for i, event in enumerate(events) if event[0] == "subquestion"), len(events))
            events[first_subquestion:first_subquestion] = [("figure", graphic) for graphic in figures]
        return events

    def render_centered_graphic(self, graphic: str, ind: str) -> list[str]:
        """Render a source figure as a centered, non-floating image block."""
        self.uses.add("graphicx")
        self.uses.add("placeholder")
        normalized_graphic = self.clamp_standalone_figure_widths(graphic)
        image = GRAPHIC_COMMAND.search(normalized_graphic)
        if image:
            path_match = re.search(r"\{([^{}]+)\}$", image.group(0))
            width_match = re.search(r"width\s*=\s*(\d*\.?\d+)\\linewidth", image.group(0))
            if path_match:
                fallback_width = float(width_match.group(1)) * MAX_FIGURE_WIDTH_CM if width_match else DEFAULT_FIGURE_WIDTH_CM
                normalized_graphic = (
                    rf"\IfFileExists{{{path_match.group(1)}}}"
                    rf"{{{image.group(0)}}}"
                    rf"{{\examfigure{{{fallback_width:g}}}{{{DEFAULT_STANDALONE_FIGURE_HEIGHT_CM:g}}}{{Figure unavailable}}}}"
                )
        return [f"{ind}\\begin{{center}}", f"{ind}{INDENT}{normalized_graphic}", f"{ind}\\end{{center}}"]

    def render_subquestions(self, subs: list[dict], level: int, depth: int) -> list[str]:
        ind = INDENT * level
        lines = [ind + r"\begin{subquestions}"]
        for sub in subs:
            lines += self.render_item(sub.get("text", ""), sub, level + 1, depth + 1)
        lines.append(ind + r"\end{subquestions}")
        return lines

    # ---- options -----------------------------------------------------------
    def render_options(self, options: list[dict], ind: str) -> list[str]:
        self.uses.add("tasks")
        cols = self.calculate_option_columns(options)
        option_width = max(
            1.0,
            min(MAX_FIGURE_WIDTH_CM, (TASKS_AVAILABLE_WIDTH_CM - (cols - 1) * TASKS_COLUMN_GAP_CM) / cols),
        )
        lines = [f"{ind}\\begin{{tasks}}({cols})"]
        for o in options:
            parts = []
            if o.get("figure"):
                parts.append(self.render_inline_figure(o["figure"], max_width_cm=option_width))
            elif o.get("text"):
                parts.append(self.normalize_option_math(o["text"]))
            lines.append(f"{ind}{INDENT}\\task {' '.join(parts)}".rstrip())
        lines.append(f"{ind}\\end{{tasks}}")
        return lines

    @staticmethod
    def calculate_option_columns(options: list[dict]) -> int:
        if any(o.get("figure") for o in options):
            widest = max(
                LatexRenderer.figure_width_cm(o["figure"].get("width_cm"))
                for o in options if o.get("figure")
            )
            return 4 if widest * 4 + 3 * TASKS_COLUMN_GAP_CM <= TASKS_AVAILABLE_WIDTH_CM else 2
        longest = max((visual_len(o.get("text") or "") for o in options), default=0)
        if longest <= 8:
            return 4
        return 2 if longest <= 18 else 1

    # ---- blocks ------------------------------------------------------------
    def render_block(self, b: dict, ind: str) -> list[str]:
        kind = b["type"]
        if kind == "text":
            return self.render_paragraphs(b["content"], ind)
        if kind == "figure":
            return self.render_figure_block(b, ind)
        if kind == "table":
            return self.render_table(b, ind)
        if kind == "statement_list":
            return self.render_statements(b, ind)
        if kind == "callout":
            return self.render_callout(b, ind)
        self.notes.append(f"unknown block type {kind!r} skipped")
        return [f"{ind}% unknown block type: {kind}"]

    def render_statements(self, b: dict, ind: str) -> list[str]:
        lines = [f"{ind}\\begin{{statements}}"]
        for item in b["items"]:
            lines += self.render_paragraphs(item["text"], ind + INDENT,
                                      first=f"\\item[{item['label']}] ", cont=ind + INDENT * 2)
        lines.append(f"{ind}\\end{{statements}}")
        return lines

    def render_callout(self, b: dict, ind: str) -> list[str]:
        inner = ind + INDENT
        self.uses.add("reference")
        lines = [f"{ind}\\begin{{reference}}"]
        pending_label = rf"\boxlabel{{{self.normalize_latex_text(b['label'])}}}" if b.get("label") else None
        if pending_label:
            self.uses.add("boxlabel")
        for i, block in enumerate(b.get("blocks", [])):
            if pending_label and block["type"] == "text":
                lines += self.render_paragraphs(block["content"], inner,
                                          first=pending_label + " ", cont=inner)
                pending_label = None
                continue
            if pending_label:
                lines.append(inner + pending_label)
                pending_label = None
            lines += self.render_block(block, inner)
        if pending_label:
            lines.append(inner + pending_label)
        lines.append(f"{ind}\\end{{reference}}")
        return lines

    # ---- figures -----------------------------------------------------------
    def render_inline_figure(self, f: dict, *, max_width_cm: float = MAX_FIGURE_WIDTH_CM) -> str:
        w = min(self.figure_width_cm(f.get("width_cm")), max_width_cm)
        h = self.format_dimension(f.get("height_cm"))
        if self.opt.figures == "image" and f.get("source_path"):
            self.uses.add("graphicx")
            self.uses.add("placeholder")
            width = f"width={w}cm"
            image_path = str(f["source_path"])
            placeholder = rf"\examfigure{{{w}}}{{{h or 3}}}{{{f.get('label', '')}}}"
            return (
                rf"\IfFileExists{{{image_path}}}"
                rf"{{\includegraphics[{width}]{{{image_path}}}}}"
                rf"{{{placeholder}}}"
            )
        if self.opt.figures == "image":
            self.notes.append(f"figure {f.get('label')!r} has no source_path; placeholder used")
        self.uses.add("placeholder")
        return rf"\examfigure{{{w}}}{{{h or 3}}}{{{f.get('label', '')}}}"

    @staticmethod
    def figure_width_cm(value: Any) -> float:
        """Use measured width when present, otherwise the named readable default."""
        width = float(value) if value is not None else DEFAULT_FIGURE_WIDTH_CM
        return min(max(width, 1.0), MAX_FIGURE_WIDTH_CM)

    def render_figure_block(self, f: dict, ind: str) -> list[str]:
        lines = [f"{ind}\\begin{{center}}", f"{ind}{INDENT}{self.render_inline_figure(f)}"]
        if self.opt.figures != "image" and f.get("source_path"):
            lines.append(f"{ind}{INDENT}% real figure: \\includegraphics[width={self.figure_width_cm(f.get('width_cm')):g}cm]"
                         f"{{{f['source_path']}}}")
        lines.append(f"{ind}\\end{{center}}")
        return lines

    # ---- tables ------------------------------------------------------------
    def render_table(self, t: dict, ind: str) -> list[str]:
        self.uses.add("booktabs")
        rows = t.get("rows") or []
        if not rows:
            return []
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        rows = [[self.normalize_table_cell(cell) for cell in row] for row in rows]
        widths = [max(display_width(r[c]) for r in rows) for c in range(ncol)]
        rules = set(t.get("rules") or [])
        spec = t.get("column_spec", "c" * ncol).replace("|", "")   # booktabs: no vertical rules

        def fmt(row: list[str]) -> str:
            cells = [row[c] + " " * (widths[c] - display_width(row[c])) for c in range(ncol - 1)]
            cells.append(row[-1])
            return " & ".join(cells) + r" \\"

        pad = ind + INDENT * 2
        lines = [f"{ind}\\begin{{center}}", f"{ind}{INDENT}\\begin{{tabular}}{{{spec}}}",
                 f"{pad}\\toprule"]
        for i, row in enumerate(rows):
            if i in rules:
                lines.append(f"{pad}\\midrule")
            lines.append(pad + fmt(row))
        lines += [f"{pad}\\bottomrule", f"{ind}{INDENT}\\end{{tabular}}", f"{ind}\\end{{center}}"]
        return lines

    # ---- text helpers ------------------------------------------------------
    def normalize_latex_text(self, text: str) -> str:
        text = self.clean_text_artifacts(text)
        text = STRAY_SUPERSCRIPT_LINE.sub("", text)
        text = self.clamp_standalone_figure_widths(text)
        if self.opt.normalize_parens:
            text = text.replace("（\\ \\ ）", r"\choiceparen{}")
        text = self.wrap_partial_math(text)
        text = self.normalize_textual_math_suffixes(text)
        text = self.normalize_math_spans(text)
        text = self.keep_choice_blank_together(text)
        return self.escape_percent_outside_math(text)

    @staticmethod
    def clean_text_artifacts(text: str) -> str:
        """Remove common Markdown/HTML serialization artifacts conservatively."""
        text = html.unescape(text)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"</?(?:div|span|p|em|strong|b|i)\b[^>]*>", "", text, flags=re.I)

        # Repair doubled serialization escapes only before syntax where two
        # backslashes are not a valid LaTeX command (set braces and percent).
        text = re.sub(r"(?<!\\)\\\\(?=[{}%])", lambda _match: "\\", text)
        text = re.sub(
            r"(?<!\\)\\\\\[(.*?)\\\\\]",
            lambda match: r"\[" + match.group(1) + r"\]",
            text,
            flags=re.S,
        )

        def clean_plain_text(plain: str) -> str:
            plain = re.sub(r"(?m)^([ \t]*)#{1,6}[ \t]+", r"\1", plain)
            plain = re.sub(r"(?<!\\)\*\*(.+?)\*\*(?!\*)", r"\1", plain, flags=re.S)
            plain = re.sub(r"(?<![\w\\])\*([^*\n]+?)\*(?!\*)", r"\1", plain)
            return re.sub(r"(?<!`)`([^`\n]+)`(?!`)", r"\1", plain)

        parts: list[str] = []
        cursor = 0
        for match in MATH_OR_DISPLAY_SEGMENT.finditer(text):
            parts.append(clean_plain_text(text[cursor:match.start()]))
            parts.append(match.group(0))
            cursor = match.end()
        parts.append(clean_plain_text(text[cursor:]))
        return "".join(parts)

    @staticmethod
    def normalize_math_spans(text: str) -> str:
        """Normalize math relations, condition lists, and tall environments."""
        def normalize(match: re.Match[str]) -> str:
            expression = match.group(0)
            expression = re.sub(r"(?<!\\)//", r"\\parallel", expression)
            expression = re.sub(r"(?<!\\)\|\|", r"\\parallel", expression)
            environment = DISPLAY_ENVIRONMENT_START.search(expression)
            if environment:
                environment_name = environment.group(1)
                ending = re.search(rf"\\end\{{{re.escape(environment_name)}\}}", expression[environment.end():])
                if ending:
                    environment_end = environment.end() + ending.end()
                    inline_prefix = expression[1:environment.start()].strip()
                    display_body = expression[environment.start():environment_end]
                    inline_suffix = expression[environment_end:-1].strip()
                    pieces = []
                    if inline_prefix:
                        pieces.append(LatexRenderer.split_math_conditions(inline_prefix))
                    pieces.append(r"\[" + display_body + r"\]")
                    if inline_suffix:
                        pieces.append(LatexRenderer.split_math_conditions(inline_suffix))
                    return " ".join(pieces)
            return LatexRenderer.split_math_conditions(expression[1:-1])

        text = re.sub(
            r"\$([A-Z]),\s*([A-Z])\$\s+(is|was)\b",
            r"$\1$. $\2$ \3",
            text,
        )
        text = MATH_SEGMENT.sub(normalize, text)
        return re.sub(
            r"(\$[^$]*[xy]\$)[ \t]+(axis|coordinate|coordinates)\b",
            r"\1-\2",
            text,
            flags=re.I,
        )

    @staticmethod
    def split_math_conditions(expression: str) -> str:
        """Separate top-level comma-delimited relations into inline math spans.

        Commas inside parentheses, brackets, or braces remain part of the same
        expression, so coordinate tuples and grouped function arguments stay
        intact. Plain comma-separated symbol lists are also left unchanged.
        """
        fragments: list[str] = []
        nesting: list[str] = []
        pairs = {')': '(', ']': '[', '}': '{'}
        start = 0
        for index, character in enumerate(expression):
            if character in '([{':
                nesting.append(character)
            elif character in ')]}':
                if nesting and nesting[-1] == pairs[character]:
                    nesting.pop()
            elif character == ',' and not nesting:
                fragments.append(expression[start:index].strip())
                start = index + 1
        if not fragments:
            return f"${expression}$"
        fragments.append(expression[start:].strip())
        if not all(fragment and MATH_RELATION.search(fragment) for fragment in fragments):
            return f"${expression}$"
        return ", ".join(f"${fragment}$" for fragment in fragments)

    @staticmethod
    def keep_choice_blank_together(text: str) -> str:
        """Keep an empty multiple-choice marker together when paragraph-wrapped."""
        def normalize_plain_text(plain: str) -> str:
            return re.sub(
                r"(?<!\\)\([ \t]+\)",
                lambda _match: r"\mbox{(\kern0.35em)}",
                plain,
            )

        pieces: list[str] = []
        cursor = 0
        for match in MATH_OR_DISPLAY_SEGMENT.finditer(text):
            pieces.append(normalize_plain_text(text[cursor:match.start()]))
            pieces.append(match.group(0))
            cursor = match.end()
        pieces.append(normalize_plain_text(text[cursor:]))
        return "".join(pieces)

    @staticmethod
    def normalize_textual_math_suffixes(text: str) -> str:
        """Keep axis and coordinate words outside math mode after a variable."""
        suffix_pattern = "|".join(map(re.escape, TEXTUAL_MATH_SUFFIXES))
        return re.sub(
            rf"\$([A-Za-z])-({suffix_pattern})\$",
            r"$\1$-\2",
            text,
            flags=re.I,
        )

    @staticmethod
    def clamp_standalone_figure_widths(text: str) -> str:
        """Clamp tiny raw includegraphics widths to a readable page fraction."""
        def clamp(match: re.Match[str]) -> str:
            width = float(match.group(1))
            if width < MIN_STANDALONE_FIGURE_FRACTION:
                width = DEFAULT_STANDALONE_FIGURE_FRACTION
            width = min(width, MAX_STANDALONE_FIGURE_FRACTION)
            return f"width={width:g}\\linewidth"

        return re.sub(
            r"width\s*=\s*(\d*\.?\d+)\s*\\linewidth",
            clamp,
            text,
        )

    @staticmethod
    def escape_percent_outside_math(text: str) -> str:
        """Escape literal percent signs in text but preserve math and existing escapes."""
        def escape_plain(plain: str) -> str:
            plain = re.sub(r"(?<!\\)_{3,}", r"\_\_\_", plain)
            lines: list[str] = []
            for line in plain.splitlines(keepends=True):
                content = line.rstrip("\r\n")
                newline = line[len(content):]
                comment = next(
                    (match for match in re.finditer(r"(?<!\\)%", content)
                     if match.start() == 0 or content[match.start() - 1].isspace()),
                    None,
                )
                if comment:
                    before = re.sub(r"(?<!\\)%", r"\\%", content[:comment.start()])
                    lines.append(before + content[comment.start():] + newline)
                else:
                    lines.append(re.sub(r"(?<!\\)%", r"\\%", content) + newline)
            return "".join(lines)
        chunks: list[str] = []
        position = 0
        for match in MATH_OR_DISPLAY_SEGMENT.finditer(text):
            chunks.append(escape_plain(text[position:match.start()]))
            chunks.append(match.group())
            position = match.end()
        chunks.append(escape_plain(text[position:]))
        return "".join(chunks)

    def normalize_option_math(self, text: str) -> str:
        """Wrap compact mathematical answer choices while leaving prose intact."""
        text = self.normalize_latex_text(text)
        if re.fullmatch(r"\s*(?:\\_){3}\s*", text):
            return text
        if "$" not in text and MATHISH_OPTION.search(text) and not re.search(r"[\u3400-\u9fff]", text):
            text = re.sub(r"(?<!\\)([{}])", r"\\\1", text.strip())
            return f"${text.strip()}$"
        return text

    def normalize_table_cell(self, text: str) -> str:
        """Keep table formulas and numeric values in math mode."""
        text = self.normalize_latex_text(str(text))
        if "$" not in text and re.fullmatch(r"\s*[+-]?(?:\d+(?:\.\d+)?|\\[A-Za-z]+(?:\{[^{}]*\})?)\s*", text):
            return f"${text.strip()}$"
        return text

    @staticmethod
    def wrap_partial_math(text: str) -> str:
        """Join bare ASCII math adjacent to inline spans into one math expression."""
        # Bars around a formula belong to the expression itself.
        text = re.sub(r"\|\s*(\$[^$]+\$)\s*\|", lambda m: "$|" + m.group(1)[1:-1] + "|$", text)
        prefix_pattern = re.compile(r"(?:[A-Za-z0-9]+(?:\s*[-+*/=<>]\s*[A-Za-z0-9]+)*\s*[-+*/=<>]\s*)$")
        suffix_pattern = re.compile(r"^(?:\s*(?:[-+*/=<>]|\|)\s*(?:[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?))+")
        spans = list(MATH_SEGMENT.finditer(text))
        result: list[str] = []
        position = 0
        index = 0
        while index < len(spans):
            match = spans[index]
            start, end = match.start(), match.end()
            while index + 1 < len(spans):
                connector = text[end:spans[index + 1].start()]
                if not MATH_CONNECTOR.fullmatch(connector):
                    break
                end = spans[index + 1].end()
                index += 1
            before_text = text[position:start]
            after_text = text[end:]
            prefix = prefix_pattern.search(before_text)
            suffix = suffix_pattern.match(after_text)
            expanded_start = position + prefix.start() if prefix else start
            expanded_end = end + (suffix.end() if suffix else 0)
            if expanded_start > 0 and text[expanded_start - 1] in "|([":
                expanded_start -= 1
            if expanded_end < len(text) and text[expanded_end] in ")]|":
                expanded_end += 1
            expression = text[expanded_start:expanded_end].replace("$", "").strip()
            result.append(text[position:expanded_start] + "$" + expression + "$")
            position = expanded_end
            index += 1
        result.append(text[position:])
        return "".join(result)

    def render_paragraphs(self, text: str, ind: str, first: str = "", cont: Optional[str] = None) -> list[str]:
        text = self.normalize_latex_text(text or "")
        cont = ind + INDENT if cont is None else cont
        lines: list[str] = []
        for n, para in enumerate(p for p in text.split("\n\n") if p.strip()):
            if n:
                lines.append("")
            lines += wrap_paragraph(para.strip(), self.opt.wrap,
                                    ind + first if n == 0 else ind, cont)
        return lines

    @staticmethod
    def format_dimension(v: Any) -> Optional[str]:
        return None if v is None else f"{float(v):g}"

    # ---- preamble ----------------------------------------------------------
    def build_preamble(self, data: dict[str, Any]) -> str:
        """Build a preamble from metadata and the features used in the body."""
        return build_preamble(data, self.uses, self.opt)
    
