"""Turn exam Markdown into an ordered section/question/block representation."""

import json
import re
import html
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBQUESTION_LABEL = re.compile(r"^[（(]([0-9０-９]+|[一二三四五六七八九十]+)[)）]\s*(.*)$", re.S)

# ---------- data model ----------
@dataclass
class Figure:
    """Reference to an image in a parsed Markdown block."""
    path: str


@dataclass
class Block:
    """One piece of a question, kept in its original order.

    kind: "text" | "table" | "figure" | "statement" | "choice" | "sub_question"
      text   -> text        : a plain paragraph
      table  -> text        : the table HTML
      figure -> figure      : a Figure
      statement    -> label ("①"), text
      choice       -> label ("A"), text and/or figure
      sub_question -> label ("1"), text
    follows_sub_question: label of the most recent sub-question printed before this
      block (None if no sub-question has appeared yet).
    """
    kind: str
    text: str = ""
    label: str = ""
    figure: Optional[Figure] = None
    follows_sub_question: Optional[str] = None


@dataclass
class Question:
    """A numbered question with its stem and ordered content blocks."""
    number: int
    stem: str
    blocks: list[Block] = field(default_factory=list)  # everything after the stem, in order

    # convenient filtered views (not stored in the JSON)
    def _of(self, kind):
        """Select blocks of one kind without changing their document order."""
        return [b for b in self.blocks if b.kind == kind]

    @property
    def choices(self):
        """Return answer choices in their original order."""
        return self._of("choice")

    @property
    def sub_questions(self):
        """Return sub-questions in their original order."""
        return self._of("sub_question")

    @property
    def statements(self):
        """Return statement blocks in their original order."""
        return self._of("statement")

    @property
    def tables(self):
        """Return table blocks in their original order."""
        return self._of("table")

    @property
    def figures(self):
        """Return figure blocks in their original order."""
        return self._of("figure")

    @property
    def other_text(self):
        """Return ordinary text blocks in their original order."""
        return self._of("text")

    @property
    def kind(self):
        """Return the broad question kind inferred from its choices and stem."""
        if self.choices:
            return "choice"
        if "___" in self.stem:
            return "fill_in_blank"
        return "free_response"


@dataclass
class Section:
    """A titled section containing questions in document order."""
    title: str
    questions: list[Question] = field(default_factory=list)


@dataclass
class Exam:
    """A parsed exam document containing ordered sections."""
    title: str = ""
    sections: list[Section] = field(default_factory=list)

    def all_questions(self):
        """Return questions from every section in reading order."""
        return [q for s in self.sections for q in s.questions]


# ---------- parsing helpers ----------
def parse_figure(text):
    """Read a TeX image reference from one Markdown content block."""
    m = re.search(r"\\includegraphics\{([^}]*)\}", text)
    return Figure(path=m.group(1)) if m else None


def parse_choices(block):
    """'(A) x  (B) y' or '(A) \\includegraphics{...}' -> list of choice Blocks."""
    parts = re.split(r"(?:^|\n|\s{2,})\(([A-H])\)\s*", block)
    choices = []
    for label, text in zip(parts[1::2], parts[2::2]):
        text = text.strip()
        figure = parse_figure(text) if "\\includegraphics" in text else None
        choices.append(Block(kind="choice", label=label,
                             text="" if figure else text, figure=figure))
    return choices


def split_blocks(md):
    """Separate Markdown into paragraphs, tables, and equation blocks."""
    return [b.strip() for b in re.split(r"\n\s*\n", md) if b.strip()]


def _markdown_table_html(block):
    """Convert a pipe table to the HTML representation used for table scoring."""
    lines = [line.strip() for line in block.splitlines() if "|" in line]
    if len(lines) < 2 or not all(line.startswith("|") and line.endswith("|") for line in lines):
        return None
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    separator = re.compile(r"^:?-{3,}:?$")
    if len(rows) > 1 and rows[1] and all(separator.fullmatch(cell.replace(" ", "")) for cell in rows[1]):
        rows.pop(1)
    if not rows:
        return None
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table>{body}</table>"


def _question_number(text):
    """Read a supported printed number and return the remaining question stem."""
    patterns = (
        r"^\s*(?:第\s*)?([0-9０-９]+)\s*[.．、]\s*",
        r"^\s*第\s*([0-9０-９]+)\s*题\s*[：:]?\s*",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            digits = "".join(str(unicodedata.digit(char)) for char in match.group(1))
            return int(digits), text[match.end():].strip()
    return None


# ---------- main parser ----------
def parse_exam(md):
    """Parse exam Markdown into section, question, and content records."""
    exam = Exam()
    section = None
    question = None
    current_sub = None  # label of the last sub-question seen in this question

    for text in split_blocks(md):
        pipe_table = None if text.startswith("<table") else _markdown_table_html(text)
        if re.fullmatch(r"<!--\s*page_number:\s*\d+\s*-->", text):
            continue
        if text.startswith("## "):
            section = Section(title=text[3:].strip())
            exam.sections.append(section)
            question = None
        elif text.startswith("# "):
            exam.title = text[2:].strip()
        elif (numbered := _question_number(text)) is not None:
            number, stem = numbered
            question = Question(number=number, stem=stem)
            current_sub = None
            if section is None:  # exam without "##" headings
                section = Section(title="")
                exam.sections.append(section)
            section.questions.append(question)
        elif question is None:
            continue
        elif text.startswith("\\includegraphics"):
            question.blocks.append(Block("figure", figure=parse_figure(text),
                                         follows_sub_question=current_sub))
        elif text.startswith("<table") or pipe_table:
            table_text = text if text.startswith("<table") else pipe_table
            question.blocks.append(Block("table", text=table_text, follows_sub_question=current_sub))
        elif re.match(r"\([A-H]\)", text):
            question.blocks.extend(parse_choices(text))
        elif SUBQUESTION_LABEL.match(text):
            m = SUBQUESTION_LABEL.match(text)
            current_sub = "".join(str(unicodedata.digit(char)) for char in m.group(1)) if m.group(1).isdigit() else m.group(1)
            question.blocks.append(Block("sub_question", label=current_sub,
                                         text=m.group(2).strip(),
                                         follows_sub_question=current_sub))
        elif text[0] in CIRCLED:
            question.blocks.append(Block("statement", label=text[0], text=text[1:].strip()))
        else:
            question.blocks.append(Block("text", text=text, follows_sub_question=current_sub))
    return exam


def describe(q):
    """Short one-line view of a question's block order, e.g. 'sub(1) > table > text > sub(2)'."""
    names = []
    for b in q.blocks:
        if b.kind in ("sub_question", "statement", "choice"):
            names.append(f"{b.kind[:3]}({b.label})")
        else:
            names.append(b.kind)
    return " > ".join(names)


def main(gold_markdown_path=PROJECT_ROOT / "paper/example_01_exam_markdown.md",
         ocr_markdown_path=PROJECT_ROOT / "data/input/exam.md",
         gold_output_path=PROJECT_ROOT / "paper/results/segmentations/manual/reference.json",
         ocr_output_path=PROJECT_ROOT / "paper/results/segmentations/manual/ocr.json"):
    """Segment the two Markdown inputs supplied as function arguments."""
    for input_file, output_json in (
        (Path(gold_markdown_path), Path(gold_output_path)),
        (Path(ocr_markdown_path), Path(ocr_output_path)),
    ):
        print(f"\nProcessing: {input_file}")

        with open(input_file, encoding="utf-8") as f:
            exam = parse_exam(f.read())

        print(exam.title)

        for s in exam.sections:
            print(f"  {s.title}: {len(s.questions)} questions")
            for q in s.questions:
                print(f"    Q{q.number:<2} {q.kind:<13} {describe(q)}")

        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(asdict(exam), f, ensure_ascii=False, indent=2)

        print("Wrote", output_json)


if __name__ == "__main__":
    main()
