"""Convert normalized Markdown exams into the shared exam data model.

The Markdown adapter recognizes headings, numbered questions, options, tables,
figures, proposition lists, and sub-questions. Content outside those patterns
is retained as question text so parsing does not silently discard exam wording.
It is kept separate from the LaTeX parser to make each input format easier to
follow and maintain.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
import yaml

from . import latex_utils as lu
from .classification import Classifier
from .dataclasses import (
    CHOICE_KINDS, Block, CalloutBlock, Exam, ExamMetadata, FigureBlock, Option, Question,
    QuestionKind, Section, StatementItem, StatementListBlock, SubQuestion,
    TableBlock, TextBlock,
)

TABLE_BLOCK_PATTERN = re.compile(
    r"<table\b.*?</table>|(?m:^[ \t]*\|[^\n]*\|[ \t]*(?:\n^[ \t]*\|[^\n]*\|[ \t]*)+)",
    re.IGNORECASE | re.DOTALL,
)
TABLE_PLACEHOLDER_PATTERN = re.compile(r"__EXAM_TABLE_BLOCK_(\d+)__")
MARKDOWN_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
METADATA_FIELDS = (
    "title", "subtitle", "year", "subject", "paper", "header_left", "header_right"
)
SUBQUESTION_LINE = re.compile(r"^[ \t]*\((\d+|[a-z])\)[ \t]*(.*)$")
STATEMENT_LINE = re.compile(r"^[ \t]*([①②③④⑤⑥⑦⑧⑨⑩])\s*(.*)$")
MARKDOWN_FIGURE = re.compile(
    r"\\begin\{figure\}(?:\[[^\]]*\])?.*?\\end\{figure\}"
    r"|<div\b[^>]*>\s*<img\b[^>]*>\s*</div>",
    re.IGNORECASE | re.DOTALL,
)
INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}")
FIGURE_TOKEN = re.compile(r"__EXAM_FIGURE_BLOCK_(\d+)__")
SHARED_MATERIAL_MARKER = re.compile(r"^[ \t]*(?:Reference\b|Appendix\b|附\s*[:：])", re.I)
FIGURE_LAYOUT = {
    "page_width_cm": 15.5,
    "minimum_standalone_fraction": 0.4,
    "maximum_standalone_fraction": 0.75,
    "default_standalone_fraction": 0.4,
    "option_width_cm": 3.3,
}
MATH_RELATION_MARKER = re.compile(
    r"(?:=|(?<!\\)[<>]|\\(?:parallel|perp|leq(?:slant)?|geq(?:slant)?|neq|approx)\b)"
)


class MarkdownExamParser:
    """Convert normalized exam Markdown into the shared exam JSON schema.

    This adapter intentionally preserves question wording and inline math as
    source text. It recognizes headings, numbered questions, and A–H options;
    content it does not understand stays in the question stem.
    """

    QUESTION_PATTERN = re.compile(r"^(\d{1,3})\.\s+(.*)$")
    OPTION_PATTERN = re.compile(r"^\(([A-H])\)\s*(.*)$")
    HEADING_PATTERN = re.compile(r"^(#{1,2})\s+(.+?)\s*$")
    GRAPHICAL_OPTION_PATTERN = re.compile(
        rf"{MARKDOWN_FIGURE.pattern}|(?m:^[ \t]*(?:<div\b[^>]*>)?\(([A-D])\)(?:</div>)?[ \t]*$)",
        re.IGNORECASE | re.DOTALL,
    )
    GRAPHICAL_IMAGE_PATTERN = re.compile(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}"
    )

    def parse(self, markdown: str, source_file: Optional[str] = None) -> Exam:
        """Parse Markdown text without changing its original content."""
        markdown_metadata: dict[str, object] = {}
        front_matter = MARKDOWN_FRONT_MATTER.match(markdown)
        if front_matter:
            parsed_metadata = yaml.safe_load(front_matter.group(1)) or {}
            if not isinstance(parsed_metadata, dict):
                raise ValueError("Markdown front matter must contain a metadata mapping")
            markdown_metadata = {
                field_name: parsed_metadata[field_name]
                for field_name in METADATA_FIELDS
                if field_name in parsed_metadata
            }
            markdown = markdown[front_matter.end():]

        headings: list[tuple[int, str, int]] = []
        lines = markdown.splitlines()
        for line_number, line in enumerate(lines):
            match = self.HEADING_PATTERN.match(line)
            if match:
                headings.append((len(match.group(1)), match.group(2).strip(), line_number))

        title = next((text for level, text, _ in headings if level == 1), None)
        section_headings = [item for item in headings if item[0] == 2]
        if not section_headings:
            raise ValueError("Markdown exam must contain at least one '##' section heading")

        metadata_values = {
            field_name: markdown_metadata[field_name]
            for field_name in METADATA_FIELDS
            if field_name in markdown_metadata
        }
        metadata_values.setdefault("title", title)
        if "year" not in metadata_values and isinstance(metadata_values.get("title"), str):
            year_match = re.search(r"(?<!\d)(\d{4})(?!\d)", metadata_values["title"])
            if year_match:
                metadata_values["year"] = int(year_match.group(1))
        metadata_values["source_file"] = source_file
        metadata = ExamMetadata(**metadata_values)
        exam = Exam(metadata=metadata)
        for order, (_, title_text, heading_line) in enumerate(section_headings, start=1):
            next_heading_line = next(
                (line_no for next_level, _, line_no in section_headings[order:]),
                len(lines),
            )
            section_content = lines[heading_line + 1:next_heading_line]
            section = self.parse_markdown_section(order, title_text, section_content)
            exam.sections.append(section)

        exam.warnings = exam.validate()
        return exam

    @staticmethod
    def apply_metadata_sidecar(exam: Exam, sidecar_path: str | Path) -> bool:
        """Apply generated YAML metadata without editing the Markdown.

        The caller chooses the output path. Only existing schema fields are
        accepted; absent values do not overwrite values parsed from Markdown.
        """
        path = Path(sidecar_path)
        if not path.is_file():
            return False
        sidecar = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(sidecar, dict):
            raise ValueError(f"Metadata sidecar must contain a mapping: {path}")
        for field_name in METADATA_FIELDS:
            value = sidecar.get(field_name)
            if value is not None:
                setattr(exam.metadata, field_name, value)
        if exam.metadata.year is None and exam.metadata.title:
            year_match = re.search(r"(?<!\d)(\d{4})(?!\d)", exam.metadata.title)
            if year_match:
                exam.metadata.year = int(year_match.group(1))
        return True

    def parse_file(self, path: str | Path) -> Exam:
        """Read and parse one UTF-8 Markdown file."""
        markdown_path = Path(path)
        return self.parse(markdown_path.read_text(encoding="utf-8"), markdown_path.name)

    def parse_markdown_section(
        self,
        order: int,
        heading: str,
        lines: list[str],
    ) -> Section:
        """Parse one section heading and divide its body into numbered questions."""
        ordinal_match = re.match(r"^([一二三四五六七八九十\d]+)[、.．]\s*(.*)$", heading)
        ordinal = ordinal_match.group(1) if ordinal_match else None
        title = ordinal_match.group(2) if ordinal_match else heading
        kind = Classifier.section_kind(title)
        section = Section(order=order, title=title, ordinal=ordinal, kind=kind)

        question_starts = [
            index for index, line in enumerate(lines)
            if self.QUESTION_PATTERN.match(line)
        ]
        if not question_starts:
            section.description = "\n".join(lines).strip() or None
            return section

        section.description = "\n".join(lines[:question_starts[0]]).strip() or None
        for question_index, start in enumerate(question_starts):
            stop = question_starts[question_index + 1] if question_index + 1 < len(question_starts) else len(lines)
            question_match = self.QUESTION_PATTERN.match(lines[start])
            assert question_match is not None
            number = int(question_match.group(1))
            content_lines = [question_match.group(2), *lines[start + 1:stop]]
            graphical_parse = (
                self.parse_graphical_options(content_lines)
                if kind in CHOICE_KINDS else None
            )
            table_blocks: list[TableBlock] = []
            if graphical_parse is None:
                content_lines, table_blocks = self.extract_tables(content_lines)
            if graphical_parse is not None:
                stem, options = graphical_parse
                question_blocks: list[Block] = []
            else:
                content_lines = self.split_inline_options(content_lines) if kind in CHOICE_KINDS else content_lines
                stem_lines: list[str] = []
                options = []
                active_option: Option | None = None
                for line in content_lines:
                    option_match = self.OPTION_PATTERN.match(line.strip())
                    if option_match:
                        active_option = Option(
                            option_match.group(1),
                            lu.normalize_option_math(lu.clean_text(option_match.group(2).strip())),
                        )
                        options.append(active_option)
                    elif active_option is not None:
                        active_option.text = lu.clean_text("\n".join(filter(None, (active_option.text, line.strip()))))
                    else:
                        stem_lines.append(line)
                stem = "\n".join(stem_lines).strip()
                stem, question_blocks = self.restore_tables(stem, table_blocks)

            stem, question_blocks, sub_questions, footer_blocks = self.structure_markdown_question(
                stem, question_blocks
            )
            blank_count = self.count_markdown_blanks(stem, question_blocks, sub_questions, footer_blocks)
            question_kind = Classifier.question_kind(bool(options), blank_count, kind)
            section.questions.append(Question(
                number=number,
                kind=question_kind,
                stem=stem,
                blocks=question_blocks,
                options=options,
                sub_questions=sub_questions,
                footer_blocks=footer_blocks,
                blank_count=blank_count,
            ))
        return section

    @staticmethod
    def split_inline_options(lines: list[str]) -> list[str]:
        """Split adjacent A–H option labels when OCR placed choices on one line."""
        option_marker = re.compile(r"(?<![\w$])\(([A-H])\)")
        expanded: list[str] = []
        for line in lines:
            matches = list(option_marker.finditer(line))
            labels = [match.group(1) for match in matches]
            if len(matches) < 2 or labels != sorted(set(labels)):
                expanded.append(line)
                continue
            if line[:matches[0].start()].strip():
                expanded.append(line[:matches[0].start()].rstrip())
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
                option_line = line[match.start():end].strip()
                if option_line:
                    expanded.append(option_line)
        return expanded

    def structure_markdown_question(
        self,
        stem: str,
        blocks: list[Block],
    ) -> tuple[str, list[Block], list[SubQuestion], list[Block]]:
        """Extract sub-questions, proposition lists, figures, and shared tails.

        Paragraph order is retained as typed question or sub-question blocks.
        A figure is question-level material even if OCR placed it after a part
        marker, so the renderer can keep it with the stem before the parts.
        """
        events: list[tuple[str, object]] = []
        if stem.strip():
            events.extend(self.markdown_text_events(stem))
        for block in blocks:
            if isinstance(block, TextBlock):
                events.extend(self.markdown_text_events(block.content))
            else:
                events.append(("block", block))

        question_stem = ""
        question_blocks: list[Block] = []
        sub_questions: list[SubQuestion] = []
        footer_blocks: list[Block] = []
        current_top: SubQuestion | None = None
        current_nested: SubQuestion | None = None
        shared_tail = False
        parent_has_block = False

        def append_text(target: list[Block], value: str) -> None:
            """Append non-empty normalized text as an ordered content block."""
            cleaned = self.normalize_markdown_content(value)
            if cleaned.strip():
                target.append(TextBlock(cleaned.strip()))

        for event_type, value in events:
            if event_type == "shared":
                shared_tail = True
                if isinstance(value, str):
                    append_text(footer_blocks, value)
                continue
            if event_type == "subquestion":
                label, remainder = value  # type: ignore[misc]
                if label.isdigit():
                    current_top = SubQuestion(label=f"({label})")
                    sub_questions.append(current_top)
                    current_nested = None
                elif current_top is not None:
                    current_nested = SubQuestion(label=f"({label})")
                    current_top.sub_questions.append(current_nested)
                else:
                    question_blocks.append(TextBlock(self.normalize_markdown_content(remainder or f"({label})")))
                    parent_has_block = True
                    continue
                self.append_to_active(remainder, current_top, current_nested)
                continue

            if event_type == "text":
                if shared_tail:
                    append_text(footer_blocks, str(value))
                elif current_top is not None:
                    self.append_to_active(str(value), current_top, current_nested)
                else:
                    before_count = len(question_blocks)
                    question_stem = self.append_parent_text(
                        str(value), question_stem, question_blocks, parent_has_block
                    )
                    parent_has_block = parent_has_block or len(question_blocks) > before_count
                continue

            if event_type == "statement":
                if shared_tail:
                    footer_blocks.append(value)  # type: ignore[arg-type]
                elif current_top is not None:
                    self.append_to_active_block(value, current_top, current_nested)  # type: ignore[arg-type]
                else:
                    question_blocks.append(value)  # type: ignore[arg-type]
                    parent_has_block = True
                continue

            if event_type == "figure":
                if shared_tail:
                    footer_blocks.append(value)  # type: ignore[arg-type]
                else:
                    question_blocks.append(value)  # type: ignore[arg-type]
                    parent_has_block = True
                continue

            if event_type == "block":
                if shared_tail:
                    footer_blocks.append(value)  # type: ignore[arg-type]
                elif current_top is not None:
                    self.append_to_active_block(value, current_top, current_nested)  # type: ignore[arg-type]
                else:
                    question_blocks.append(value)  # type: ignore[arg-type]
                    parent_has_block = True

        return question_stem.strip(), question_blocks, sub_questions, footer_blocks

    def append_parent_text(
        self,
        text: str,
        stem: str,
        blocks: list[Block],
        has_block: bool,
    ) -> str:
        """Keep initial prose in the stem and preserve later prose block order."""
        normalized = self.normalize_markdown_content(text).strip()
        if not normalized:
            return stem
        if not stem and not has_block:
            return normalized
        if not has_block:
            return f"{stem}\n\n{normalized}".strip()
        blocks.append(TextBlock(normalized))
        return stem

    def append_to_active(
        self,
        text: str,
        top: SubQuestion,
        nested: SubQuestion | None,
    ) -> None:
        """Attach prose to the most recently opened sub-question."""
        cleaned = self.normalize_markdown_content(text).strip()
        if cleaned:
            (nested or top).blocks.append(TextBlock(cleaned))

    @staticmethod
    def append_to_active_block(block: Block, top: SubQuestion, nested: SubQuestion | None) -> None:
        """Attach a non-text block to the active nested or top-level part."""
        (nested or top).blocks.append(block)

    def markdown_text_events(self, text: str) -> list[tuple[str, object]]:
        """Tokenize Markdown text while retaining inline content and line order."""
        figures: list[FigureBlock] = []

        def extract_figure(match: re.Match[str]) -> str:
            """Save a figure block and replace it with an ordering token."""
            figure = self.figure_from_markup(match.group(0), option=False)
            if figure is None:
                return match.group(0)
            index = len(figures)
            figures.append(figure)
            return f"\n__EXAM_FIGURE_BLOCK_{index}__\n"

        prepared = MARKDOWN_FIGURE.sub(extract_figure, text)
        events: list[tuple[str, object]] = []
        text_buffer: list[str] = []
        pending_statements: list[StatementItem] = []
        pending_statement_lines: list[str] = []
        pending_blank_lines: list[str] = []

        def flush_text() -> None:
            """Emit buffered prose before the next structural marker."""
            if text_buffer:
                content = "".join(text_buffer)
                if content.strip():
                    events.append(("text", content))
                text_buffer.clear()

        def flush_statements() -> None:
            """Emit a proposition list once its consecutive lines end."""
            if pending_statements:
                if len(pending_statements) > 1:
                    flush_text()
                    events.append(("statement", StatementListBlock(items=pending_statements.copy())))
                else:
                    text_buffer.extend(pending_statement_lines)
                pending_statements.clear()
                pending_statement_lines.clear()

        for line in prepared.splitlines(keepends=True):
            figure_token = FIGURE_TOKEN.fullmatch(line.strip())
            if figure_token:
                flush_statements()
                flush_text()
                events.append(("figure", figures[int(figure_token.group(1))]))
                continue

            statement_match = STATEMENT_LINE.match(line.rstrip("\r\n"))
            if statement_match:
                if pending_blank_lines:
                    pending_statement_lines.extend(pending_blank_lines)
                    pending_blank_lines.clear()
                pending_statements.append(StatementItem(
                    label=statement_match.group(1),
                    text=self.normalize_markdown_content(statement_match.group(2)).strip(),
                ))
                pending_statement_lines.append(line)
                continue

            if pending_statements and not line.strip():
                pending_blank_lines.append(line)
                continue

            flush_statements()
            if pending_blank_lines:
                text_buffer.extend(pending_blank_lines)
                pending_blank_lines.clear()

            subquestion_match = SUBQUESTION_LINE.match(line.rstrip("\r\n"))
            if subquestion_match:
                flush_text()
                events.append(("subquestion", (subquestion_match.group(1), subquestion_match.group(2))))
                continue

            if SHARED_MATERIAL_MARKER.match(line):
                flush_text()
                events.append(("shared", line))
                continue

            text_buffer.append(line)

        flush_statements()
        if pending_blank_lines:
            text_buffer.extend(pending_blank_lines)
        flush_text()
        return events

    @staticmethod
    def normalize_markdown_content(text: str) -> str:
        """Normalize partial math and blank markers without changing prose."""
        normalized = text
        relation_chain = re.compile(
            r"(?<![A-Za-z0-9$])([A-Za-z]{1,4})\s*(?://|\|\|)\s*([A-Za-z]{1,4})"
            r"(?:\s*,\s*[A-Za-z]{1,4}\s*(?://|\|\|)\s*[A-Za-z]{1,4})*"
        )

        def wrap_relation_chain(match: re.Match[str]) -> str:
            """Wrap each comma-separated relation so LaTeX can break lines."""
            conditions = re.split(r"\s*,\s*", match.group(0))
            normalized_conditions = [
                re.sub(r"\s*(?://|\|\|)\s*", r" \\parallel ", condition).strip()
                for condition in conditions
            ]
            return ", ".join(f"${condition}$" for condition in normalized_conditions)

        normalized = relation_chain.sub(wrap_relation_chain, normalized)
        normalized = lu.normalize_math_markup(normalized)
        normalized = re.sub(
            r"(?<!\\)\$.*?(?<!\\)\$",
            lambda match: MarkdownExamParser.split_math_conditions(match.group(0)),
            normalized,
            flags=re.DOTALL,
        )
        normalized = re.sub(
            r"(\$[^$]*[xy]\$)[ \t]+(axis|coordinate|coordinates)\b",
            r"\1-\2",
            normalized,
            flags=re.I,
        )
        normalized = re.sub(
            r"(?<![A-Za-z$])([A-Z]{1,4})[ \t]+\$(\\(?:perp|parallel)[ \t]*[xy])\$",
            lambda match: f"${match.group(1)} {match.group(2)}$",
            normalized,
            flags=re.I,
        )
        chunks: list[str] = []
        cursor = 0
        for match in re.finditer(r"(?<!\\)\$.*?(?<!\\)\$", normalized, re.DOTALL):
            plain = normalized[cursor:match.start()]
            escaped_plain = re.sub(r"(?<!\\)_{3,}|(?:\\_){3,}", r"\\blank", plain)
            chunks.append(re.sub(r"(?<!\\)%", r"\\%", escaped_plain))
            chunks.append(re.sub(r"(?<!\\)//", r"\\parallel", match.group(0)))
            cursor = match.end()
        trailing_plain = re.sub(r"(?<!\\)_{3,}|(?:\\_){3,}", r"\\blank", normalized[cursor:])
        chunks.append(re.sub(r"(?<!\\)%", r"\\%", trailing_plain))
        return "".join(chunks)

    @staticmethod
    def split_math_conditions(math_span: str) -> str:
        """Split comma-separated top-level relations while retaining groups."""
        expression = math_span[1:-1]
        expression = re.sub(r"(?<!\\)//|(?<!\\)\|\|", r"\\parallel", expression)
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
            return math_span
        fragments.append(expression[start:].strip())
        if not all(part and MATH_RELATION_MARKER.search(part) for part in fragments):
            return math_span
        return ", ".join(f"${part}$" for part in fragments)

    @staticmethod
    def count_markdown_blanks(
        stem: str,
        blocks: list[Block],
        sub_questions: list[SubQuestion],
        footer_blocks: list[Block],
    ) -> int:
        """Count normalized blank macros throughout question content."""
        count = len(re.findall(r"\\blank(?![A-Za-z])", stem))

        def count_block(block: Block) -> int:
            """Count blanks recursively for each supported block type."""
            if isinstance(block, TextBlock):
                return len(re.findall(r"\\blank(?![A-Za-z])", block.content))
            if isinstance(block, StatementListBlock):
                return sum(len(re.findall(r"\\blank(?![A-Za-z])", item.text)) for item in block.items)
            if isinstance(block, CalloutBlock):
                return sum(count_block(child) for child in block.blocks)
            return 0

        def count_subquestion(question: SubQuestion) -> int:
            """Include blanks from this part and all nested parts."""
            return sum(count_block(block) for block in question.blocks) + sum(
                count_subquestion(child) for child in question.sub_questions
            )

        return count + sum(count_block(block) for block in blocks + footer_blocks) + sum(
            count_subquestion(question) for question in sub_questions
        )

    @staticmethod
    def figure_from_markup(markup: str, *, option: bool) -> FigureBlock | None:
        """Build a sized figure block from an image path and its aspect ratio."""
        image = INCLUDEGRAPHICS.search(markup)
        html_image = BeautifulSoup(markup, "html.parser").find("img")
        if image is None and html_image is None:
            return None
        path = image.group(2).strip() if image else str(html_image.get("src", "")).strip()
        if not path:
            return None
        pixel_size = None
        box = re.search(r"img_in_image_box_(\d+)_(\d+)_(\d+)_(\d+)", Path(path).stem)
        if box:
            x0, y0, x1, y1 = map(int, box.groups())
            pixel_size = (max(x1 - x0, 1), max(y1 - y0, 1))
        if pixel_size is None and Path(path).is_file():
            try:
                from PIL import Image
                with Image.open(path) as image_file:
                    pixel_size = image_file.size
            except (ImportError, OSError):
                pixel_size = None

        if option:
            width_cm = float(FIGURE_LAYOUT["option_width_cm"])
        else:
            width_fraction = float(FIGURE_LAYOUT["default_standalone_fraction"])
            declared_width = (
                re.search(r"width\s*=\s*(\d*\.?\d+)\\linewidth", image.group(1) or "")
                if image else None
            )
            if declared_width is None and html_image is not None:
                width_attribute = html_image.get("width")
                width_match = re.fullmatch(r"\s*(\d*\.?\d+)\s*%\s*", str(width_attribute or ""))
                if width_match:
                    width_fraction = float(width_match.group(1)) / 100
            if declared_width:
                width_fraction = float(declared_width.group(1))
            width_fraction = min(
                max(width_fraction, float(FIGURE_LAYOUT["minimum_standalone_fraction"])),
                float(FIGURE_LAYOUT["maximum_standalone_fraction"]),
            )
            width_cm = width_fraction * float(FIGURE_LAYOUT["page_width_cm"])
        height_cm = width_cm * pixel_size[1] / pixel_size[0] if pixel_size else width_cm * 0.7
        return FigureBlock(
            label="Figure",
            width_cm=round(width_cm, 2),
            height_cm=round(height_cm, 2),
            is_placeholder=False,
            source_path=path,
        )

    def extract_tables(self, content_lines: list[str]) -> tuple[list[str], list[TableBlock]]:
        """Replace HTML or Markdown tables with markers and structured rows."""
        content = "\n".join(content_lines)
        tables: list[TableBlock] = []

        def replace_table(match: re.Match[str]) -> str:
            """Parse a source table and replace it with a stable marker."""
            table_text = match.group(0)
            if table_text.lstrip().lower().startswith("<table"):
                table = self.parse_html_table(table_text)
            else:
                table = self.parse_markdown_table(table_text)
            if table is None:
                return table_text
            table_index = len(tables)
            tables.append(table)
            return f"\n__EXAM_TABLE_BLOCK_{table_index}__\n"

        transformed = TABLE_BLOCK_PATTERN.sub(replace_table, content)
        return transformed.splitlines(), tables

    @staticmethod
    def parse_html_table(html_table: str) -> TableBlock | None:
        """Parse HTML table cells directly into a structured table block."""
        soup = BeautifulSoup(html_table, "html.parser")
        table_tag = soup.find("table")
        if table_tag is None:
            return None

        rows: list[list[str]] = []
        has_header = False
        for row_index, row in enumerate(table_tag.find_all("tr")):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                continue
            if row_index == 0:
                has_header = any(cell.name == "th" for cell in cells)
            rows.append([cell.get_text(" ", strip=True) for cell in cells])
        return MarkdownExamParser.make_table_block(rows, has_header)

    @staticmethod
    def parse_markdown_table(markdown_table: str) -> TableBlock | None:
        """Parse a pipe table directly into the same table block representation."""
        rows: list[list[str]] = []
        has_header = False
        for line in markdown_table.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                continue
            cells = [
                cell.strip().replace(r"\|", "|")
                for cell in re.split(r"(?<!\\)\|", stripped_line.strip("|"))
            ]
            if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                has_header = bool(rows)
                continue
            rows.append(cells)
        return MarkdownExamParser.make_table_block(rows, has_header)

    @staticmethod
    def make_table_block(rows: list[list[str]], has_header: bool) -> TableBlock | None:
        """Pad ragged rows and choose basic LaTeX column alignment."""
        if not rows:
            return None
        column_count = max(len(row) for row in rows)
        padded_rows = [row + [""] * (column_count - len(row)) for row in rows]
        return TableBlock(
            column_spec="l" + "c" * (column_count - 1),
            rows=padded_rows,
            header_rows=1 if has_header else 0,
            rules=[1] if has_header and len(padded_rows) > 1 else [],
        )

    @staticmethod
    def restore_tables(
        stem: str,
        tables: list[TableBlock],
    ) -> tuple[str, list[TextBlock | TableBlock]]:
        """Restore markers as ordered table/text blocks after the stem prefix."""
        matches = list(TABLE_PLACEHOLDER_PATTERN.finditer(stem))
        if not matches:
            return stem, []

        question_stem = stem[:matches[0].start()].strip()
        blocks: list[TextBlock | TableBlock] = []
        for match_index, match in enumerate(matches):
            table_index = int(match.group(1))
            if table_index >= len(tables):
                continue
            blocks.append(tables[table_index])
            next_start = (
                matches[match_index + 1].start()
                if match_index + 1 < len(matches) else len(stem)
            )
            following_text = stem[match.end():next_start].strip()
            if following_text:
                blocks.append(TextBlock(following_text))
        return question_stem, blocks

    def parse_graphical_options(self, content_lines: list[str]) -> tuple[str, list[Option]] | None:
        """Pair four extracted option images with A–D when OCR order split labels.

        Some Markdown exports place graph images before their labels or read a
        two-column option layout in an inconsistent order. A strict four-image
        check avoids treating arbitrary figures as choices. Existing text after
        an explicit option label is retained with that option.
        """
        content = "\n".join(content_lines)
        matches = list(self.GRAPHICAL_OPTION_PATTERN.finditer(content))
        image_matches = [match for match in matches if self.figure_from_markup(match.group(0), option=True)]
        explicit_labels = [match.group(1) for match in matches if match.group(1)]
        if len(image_matches) != 4 or len(set(explicit_labels)) < 2:
            return None

        image_paths: list[str] = []
        for match in image_matches:
            figure = self.figure_from_markup(match.group(0), option=True)
            if figure is None or not figure.source_path:
                return None
            image_paths.append(figure.source_path)

        stem_parts: list[str] = []
        option_text: dict[str, list[str]] = {label: [] for label in "ABCD"}
        active_label: str | None = None
        cursor = 0
        for match in matches:
            preceding_text = content[cursor:match.start()].strip()
            if preceding_text:
                if active_label is None:
                    stem_parts.append(preceding_text)
                else:
                    option_text[active_label].append(preceding_text)

            if match.group(1):
                active_label = match.group(1)
            cursor = match.end()

        trailing_text = content[cursor:].strip()
        if trailing_text:
            if active_label is None:
                stem_parts.append(trailing_text)
            else:
                option_text[active_label].append(trailing_text)

        options = [
            self.graphical_option(label, image_paths[index], option_text[label])
            for index, label in enumerate("ABCD")
        ]
        return "\n\n".join(stem_parts).strip(), options

    @classmethod
    def graphical_option(cls, label: str, image_path: str, text_parts: list[str]) -> Option:
        """Create an image choice with absent text represented as JSON null."""
        figure = cls.figure_from_markup(
            rf"\includegraphics{{{image_path}}}", option=True
        )
        if figure is not None:
            figure.label = f"Option {label}"
        text_markup = "\n".join(text_parts).strip()
        text = BeautifulSoup(text_markup, "html.parser").get_text(" ", strip=True) or None
        return Option(label=label, text=text, figure=figure)
