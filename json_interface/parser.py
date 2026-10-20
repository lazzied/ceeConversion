"""Parse the project's structured LaTeX exam format.

This module handles LaTeX documents that use the project's exam commands and
list environments. Markdown input has its own adapter in ``markdown_parser``;
both adapters produce the same dataclasses from ``dataclasses.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from . import latex_utils as lu
from .classification import Classifier
from .latex_utils import Segment
from .dataclasses import (
    Block, CalloutBlock, Exam, ExamMetadata, FigureBlock, Option,
    Question, QuestionKind, Section, StatementItem, StatementListBlock,
    SubQuestion, TableBlock, TextBlock,
)

BODY_ENVS = ["center", "enumerate", "list"]
OPTION_LABELS = "ABCDEFGH"


class PreambleParser:
    """Reads title / subtitle from \\papertitle and running headers."""

    def parse(self, preamble: str, source_file: Optional[str] = None) -> ExamMetadata:
        """Read title commands and running-head values from the preamble."""
        meta = ExamMetadata(source_file=source_file)
        meta.header_left = self.extract_command_argument(preamble, r"\\fancyhead\[L\]")
        meta.header_right = self.extract_command_argument(preamble, r"\\fancyhead\[R\]")

        m = re.search(r"\\newcommand\{\\papertitle\}", preamble)
        if m:
            try:
                body, _ = lu.read_group(preamble, m.end())
            except ValueError:
                body = ""
            styled = []
            for g in re.finditer(r"\{\s*\\(?:LARGE|Large|huge|Huge)(?![A-Za-z])", body):
                inner, _ = lu.read_group(body, g.start())
                styled.append(lu.strip_styling(inner))
            if styled:
                meta.title = styled[0]
            if len(styled) > 1:
                meta.subtitle = styled[1]

        if meta.title:
            ym = re.search(r"(\d{4})\s*年", meta.title)
            meta.year = int(ym.group(1)) if ym else None
        if meta.subtitle:
            parts = [p.strip() for p in re.split(r"[·•]", meta.subtitle)]
            meta.subject = parts[0] or None
            meta.paper = parts[1] if len(parts) > 1 else None
        return meta

    @staticmethod
    def extract_command_argument(text: str, command_regex: str) -> Optional[str]:
        """Return the braced argument following a matching LaTeX command."""
        m = re.search(command_regex + r"(?=\s*\{)", text)
        if not m:
            return None
        try:
            inner, _ = lu.read_group(text, m.end())
        except ValueError:
            return None
        return lu.strip_styling(inner) or None


@dataclass
class ParsedBody:
    """Collect the pieces found while parsing one question body.

    ``blocks`` are content before any sub-question, while ``footer_blocks``
    hold shared material that follows the sub-question list. Keeping these
    pieces separate lets the renderer place each one in the right order.
    """
    text: Optional[str] = None
    blocks: list[Block] = field(default_factory=list)
    options: list[Option] = field(default_factory=list)
    sub_questions: list[SubQuestion] = field(default_factory=list)
    footer_blocks: list[Block] = field(default_factory=list)
    blank_count: int = 0


class BlockParser:
    """Turn a LaTeX question body into text, options, and typed content blocks.

    This parser interprets supported environments at the current nesting
    level. Nested environments are split by ``latex_utils`` so table contents
    and math are not mistaken for question structure.
    """

    def __init__(self, warn):
        """Store the callback used to report recoverable malformed input."""
        self.warn = warn

    # ---- entry point ------------------------------------------------
    def parse_body(self, text: str, depth: int = 0) -> ParsedBody:
        """Split and parse one question or sub-question body."""
        return self.parse_segments(lu.split_top_level(text, BODY_ENVS), depth)

    def parse_segments(self, segs: list[Segment], depth: int) -> ParsedBody:
        """Combine already-split LaTeX segments into a structured result."""
        body = ParsedBody()
        seen_subs = False

        def add_block(block: Block) -> None:
            """Keep material after sub-questions in the shared footer."""
            (body.footer_blocks if seen_subs else body.blocks).append(block)

        for seg in segs:
            if seg.kind == "text":
                body.blank_count += len(re.findall(r"\\blank(?![A-Za-z])", seg.text))
                text, opts = self.extract_options(seg.text)
                body.options.extend(opts)
                text = lu.clean_text(text)
                if not text:
                    continue
                if body.text is None:
                    body.text = text
                else:
                    add_block(TextBlock(text))
                continue

            if body.text is None:          # question starts with a non-text block
                body.text = ""
            if seg.name == "enumerate":
                body.sub_questions.extend(self.parse_enumerate(seg.text, depth))
                seen_subs = True
            elif seg.name == "list":
                add_block(self.parse_statement_list(seg.text))
            elif seg.name == "center":
                opts, blocks = self.parse_center(seg.text)
                body.options.extend(opts)
                for b in blocks:
                    add_block(b)
        if body.text is None:
            body.text = ""
        return body

    # ---- \opts{A}{B}{C}{D} -------------------------------------------
    def extract_options(self, text: str) -> tuple[str, list[Option]]:
        """Remove each complete ``\\opts`` command and return its choices."""
        options: list[Option] = []
        out, pos = [], 0
        for m in re.finditer(r"\\opts(?![A-Za-z])", text):
            if m.start() < pos:
                continue
            try:
                args, end = lu.read_groups(text, m.end(), 4)
            except ValueError as exc:
                self.warn(f"malformed \\opts: {exc}")
                continue
            out.append(text[pos:m.start()])
            base = len(options)
            options.extend([Option(OPTION_LABELS[base + i], lu.normalize_option_math(lu.clean_text(a)))
                            for i, a in enumerate(args)])
            pos = end
        out.append(text[pos:])
        return "".join(out), options

    # ---- center: callout | image options | figure(s) | table ----------
    def parse_center(self, body: str) -> tuple[list[Option], list[Block]]:
        """Interpret a centered block as a callout, image choices, or content."""
        if r"\fcolorbox" in body:
            return [], [self.parse_callout(body)]

        segs = lu.split_top_level(body, ["tabular"])
        tabulars = [s for s in segs if s.kind == "env"]

        if tabulars and r"\imgph" in body and re.search(r"\(\s*[A-H]\s*\)", body):
            options: list[Option] = []
            for t in tabulars:
                options.extend(self.parse_image_options(t.text))
            return options, []

        blocks: list[Block] = []
        for seg in segs:
            if seg.kind == "env":
                blocks.append(self.parse_table(seg.text))
            else:
                figs, leftover = self.parse_figures(seg.text)
                blocks.extend(figs)
                leftover = lu.clean_text(leftover)
                if leftover:
                    blocks.append(TextBlock(leftover))
        return [], blocks

    def parse_figures(self, text: str) -> tuple[list[FigureBlock], str]:
        """Extract figure commands while preserving surrounding text."""
        figs: list[FigureBlock] = []
        out, pos = [], 0
        for m in re.finditer(r"\\(imgph|figsrc)(?![A-Za-z])", text):
            if m.start() < pos:
                continue
            out.append(text[pos:m.start()])
            if m.group(1) == "imgph":
                (w, h, label), pos = lu.read_groups(text, m.end(), 3)
                figs.append(FigureBlock(label=lu.clean_text(label),
                                        width_cm=self.parse_numeric_dimension(w), height_cm=self.parse_numeric_dimension(h)))
            else:
                path, pos = lu.read_group(text, m.end())
                if figs:
                    figs[-1].source_path = path.strip()
        out.append(text[pos:])
        return figs, "".join(out)

    def parse_image_options(self, tabular_body: str) -> list[Option]:
        """Read image placeholders in a tabular block as answer choices."""
        _, rows, _ = lu.parse_tabular(tabular_body)
        options: list[Option] = []
        for cell in (c for row in rows for c in row):
            lm = re.fullmatch(r"\(\s*([A-H])\s*\)", cell.strip())
            if lm:
                options.append(Option(label=lm.group(1)))
                continue
            if not options:
                continue
            figs, leftover = self.parse_figures(cell)
            if figs:
                options[-1].figure = figs[0]
            leftover = lu.clean_text(leftover)
            if leftover:
                options[-1].text = lu.clean_text(leftover)
        return options

    # ---- tabular -----------------------------------------------------
    def parse_table(self, tabular_body: str) -> TableBlock:
        """Convert one ``tabular`` environment to the shared table model."""
        spec, rows, rules = lu.parse_tabular(tabular_body)
        return TableBlock(column_spec=spec, rows=rows,
                          header_rows=rules[0] if rules else 0, rules=rules)

    # ---- \begin{list} with \item[①] ----------------------------------
    def parse_statement_list(self, env_body: str) -> StatementListBlock:
        """Read labeled list items, such as circled-digit propositions."""
        try:                                   # skip the two {..}{..} list arguments
            _, k = lu.read_groups(env_body, 0, 2)
        except ValueError:
            k = 0
        body = env_body[k:]
        marks = list(re.finditer(r"\\item\s*\[([^\]]*)\]", body))
        items = []
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
            items.append(StatementItem(label=m.group(1).strip(),
                                       text=lu.clean_text(body[m.end():end])))
        return StatementListBlock(items=items)

    # ---- \fcolorbox{..}{..}{\begin{minipage}..} ---------------------
    def parse_callout(self, body: str) -> CalloutBlock:
        """Extract the heading and inner blocks from a framed callout."""
        m = re.search(r"\\fcolorbox\{[^{}]*\}\{[^{}]*\}", body)
        inner, _ = lu.read_group(body, m.end())
        content = inner
        for seg in lu.split_top_level(inner, ["minipage"]):
            if seg.kind == "env":
                _, k = lu.read_group(seg.text, 0)          # {0.88\linewidth}
                content = seg.text[k:]
        label: Optional[str] = None
        blocks: list[Block] = []
        for seg in lu.split_top_level(content, ["tabular"]):
            if seg.kind == "env":
                blocks.append(self.parse_table(seg.text))
                continue
            text = seg.text
            lm = re.search(r"\\textbf\{\s*\\color\{[^}]*\}\s*([^{}]*)\}\s*(?:\\quad)?", text)
            if lm and label is None:
                label = lm.group(1).strip()
                text = text[:lm.start()] + text[lm.end():]
            text = lu.clean_text(text)
            if text:
                blocks.append(TextBlock(text))
        return CalloutBlock(label=label, blocks=blocks)

    # ---- enumerate -> sub-questions ---------------------------------
    def parse_enumerate(self, env_body: str, depth: int) -> list[SubQuestion]:
        """Turn an ``enumerate`` environment into depth-aware sub-questions."""
        body = env_body
        if body.lstrip().startswith("["):
            _, k = lu.read_group(body, 0, "[", "]")
            body = body[k:]
        items: list[list[Segment]] = []
        for seg in lu.split_top_level(body, BODY_ENVS):
            if seg.kind == "text":
                parts = re.split(r"\\item(?![A-Za-z])", seg.text)
                if parts[0].strip():
                    if items:
                        items[-1].append(Segment("text", parts[0]))
                    else:
                        self.warn("text before first \\item in enumerate was ignored")
                items.extend([[Segment("text", p)] for p in parts[1:]])
            else:
                if not items:
                    items.append([])
                items[-1].append(seg)

        subs = []
        for idx, segs in enumerate(items, start=1):
            parsed = self.parse_segments(segs, depth + 1)
            subs.append(SubQuestion(
                label=self.sub_label(depth, idx), text=parsed.text or "",
                blocks=parsed.blocks, options=parsed.options,
                sub_questions=parsed.sub_questions, footer_blocks=parsed.footer_blocks,
                blank_count=parsed.blank_count))
        return subs

    @staticmethod
    def sub_label(depth: int, idx: int) -> str:
        """Format a sub-question number or letter for its nesting level."""
        if depth == 0:
            return f"({idx})"
        if depth == 1:
            return f"({chr(96 + idx)})"
        return f"({idx})"

    @staticmethod
    def parse_numeric_dimension(s: str) -> Optional[float]:
        """Parse a numeric image dimension, returning ``None`` if absent."""
        try:
            return float(s)
        except ValueError:
            return None


# ======================================================================
class QuestionParser:
    """Build one schema question from its number, body, and section kind."""

    def __init__(self, warn, keep_raw: bool = False):
        """Create the body parser and choose whether to retain raw source text."""
        self.blocks = BlockParser(warn)
        self.keep_raw = keep_raw

    def parse(self, number: int, raw: str, section_kind: QuestionKind) -> Question:
        """Parse a question and infer its kind from its actual structure."""
        parsed = self.blocks.parse_body(raw)
        return Question(
            number=number,
            kind=Classifier.question_kind(bool(parsed.options), parsed.blank_count, section_kind),
            stem=parsed.text or "",
            blocks=parsed.blocks,
            options=parsed.options,
            sub_questions=parsed.sub_questions,
            footer_blocks=parsed.footer_blocks,
            blank_count=parsed.blank_count,
            raw=raw.strip() if self.keep_raw else None,
        )


class ExamParser:
    """Parse a complete exam document written in the project's LaTeX format.

    Use :meth:`parse_file` for a path or :meth:`parse` for source text already
    in memory. Any recoverable issues are attached to ``exam.warnings``.
    """

    def __init__(self, keep_raw: bool = False):
        """Initialize warning state and configure optional raw-text retention."""
        self.keep_raw = keep_raw
        self.warnings: list[str] = []

    def parse_file(self, path: Union[str, Path]) -> Exam:
        """Read a UTF-8 LaTeX file and parse it as one exam."""
        path = Path(path)
        return self.parse(path.read_text(encoding="utf-8"), source_file=path.name)

    def parse(self, text: str, source_file: Optional[str] = None) -> Exam:
        """Parse preamble metadata, sections, questions, and validation warnings."""
        self.warnings = []
        text = lu.strip_comments(lu.preserve_figure_hints(text))

        begin = re.search(r"\\begin\{document\}", text)
        end = re.search(r"\\end\{document\}", text)
        if not begin:
            raise ValueError("no \\begin{document} found - is this a complete .tex file?")
        preamble = text[:begin.start()]
        body = text[begin.end():end.start() if end else len(text)]

        exam = Exam(metadata=PreambleParser().parse(preamble, source_file))
        body = re.sub(r"\\papertitle(?![A-Za-z])", "", body)

        qparser = QuestionParser(self.warnings.append, self.keep_raw)
        sections = list(re.finditer(r"\\section\*?(?=\s*\{)", body))
        preface = lu.clean_text(body[:sections[0].start()] if sections else body)
        exam.instructions = preface or None

        for order, m in enumerate(sections, start=1):
            title, after = lu.read_group(body, m.end())
            stop = sections[order].start() if order < len(sections) else len(body)
            exam.sections.append(self.parse_section(order, title, body[after:stop], qparser))

        exam.warnings = self.warnings + exam.validate()
        return exam

    def parse_section(self, order: int, title: str, body: str, qparser: QuestionParser) -> Section:
        """Parse one custom-LaTeX section and its numbered question bodies."""
        title = lu.strip_styling(title)
        ordinal_match = re.match(r"^([一二三四五六七八九十]+|\d+)\s*[、.．]\s*(.*)$", title)
        section = Section(
            order=order,
            ordinal=ordinal_match.group(1) if ordinal_match else None,
            title=ordinal_match.group(2) if ordinal_match else title,
        )
        section.kind = Classifier.section_kind(section.title)

        question_markers = list(re.finditer(r"\\q\{\s*(\d+)\s*\}", body))
        intro = lu.clean_text(body[:question_markers[0].start()] if question_markers else body)
        section.description = intro or None
        for index, marker in enumerate(question_markers):
            stop = question_markers[index + 1].start() if index + 1 < len(question_markers) else len(body)
            section.questions.append(
                qparser.parse(int(marker.group(1)), body[marker.end():stop], section.kind)
            )
        return section

def __getattr__(name: str):
    """Load the Markdown adapter only for callers using its old import path.

    New code should import ``MarkdownExamParser`` from ``markdown_parser``.
    This small compatibility hook keeps older ``parser`` imports working
    without making the LaTeX parser depend on the Markdown implementation.
    """
    if name == "MarkdownExamParser":
        from .markdown_parser import MarkdownExamParser
        return MarkdownExamParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
