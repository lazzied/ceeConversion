import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"


# ---------- data model ----------
@dataclass
class Figure:
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
    number: int
    stem: str
    blocks: list[Block] = field(default_factory=list)  # everything after the stem, in order

    # convenient filtered views (not stored in the JSON)
    def _of(self, kind):
        return [b for b in self.blocks if b.kind == kind]

    @property
    def choices(self):
        return self._of("choice")

    @property
    def sub_questions(self):
        return self._of("sub_question")

    @property
    def statements(self):
        return self._of("statement")

    @property
    def tables(self):
        return self._of("table")

    @property
    def figures(self):
        return self._of("figure")

    @property
    def other_text(self):
        return self._of("text")

    @property
    def kind(self):
        if self.choices:
            return "choice"
        if "___" in self.stem:
            return "fill_in_blank"
        return "free_response"


@dataclass
class Section:
    title: str
    questions: list[Question] = field(default_factory=list)


@dataclass
class Exam:
    title: str = ""
    sections: list[Section] = field(default_factory=list)

    def all_questions(self):
        return [q for s in self.sections for q in s.questions]


# ---------- parsing helpers ----------
def parse_figure(text):
    m = re.search(r"\\includegraphics\{([^}]*)\}", text)
    return Figure(path=m.group(1)) if m else None


def parse_choices(block):
    """'(A) x  (B) y' or '(A) \\includegraphics{...}' -> list of choice Blocks."""
    parts = re.split(r"(?:^|\s{2,})\(([A-H])\)\s*", block)
    choices = []
    for label, text in zip(parts[1::2], parts[2::2]):
        text = text.strip()
        figure = parse_figure(text) if "\\includegraphics" in text else None
        choices.append(Block(kind="choice", label=label,
                             text="" if figure else text, figure=figure))
    return choices


def split_blocks(md):
    return [b.strip() for b in re.split(r"\n\s*\n", md) if b.strip()]


# ---------- main parser ----------
def parse_exam(md):
    exam = Exam()
    section = None
    question = None
    current_sub = None  # label of the last sub-question seen in this question

    for text in split_blocks(md):
        if text.startswith("## "):
            section = Section(title=text[3:].strip())
            exam.sections.append(section)
            question = None
        elif text.startswith("# "):
            exam.title = text[2:].strip()
        elif re.match(r"\d+\.\s", text):
            m = re.match(r"(\d+)\.\s+(.*)", text, re.S)
            question = Question(number=int(m.group(1)), stem=m.group(2).strip())
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
        elif text.startswith("<table"):
            question.blocks.append(Block("table", text=text, follows_sub_question=current_sub))
        elif re.match(r"\([A-H]\)\s", text):
            question.blocks.extend(parse_choices(text))
        elif re.match(r"\(\d+\)\s", text):
            m = re.match(r"\((\d+)\)\s+(.*)", text, re.S)
            current_sub = m.group(1)
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


def main():

    input_file = "paper/exam_out.md"
    output_json = "paper/exam_segmented.json"


    #input_file = "data/output/exam.md"
    #output_json = "paper/ocr_exam_segmented.json"



    with open(input_file, encoding="utf-8") as f:
        exam = parse_exam(f.read())

    print(exam.title)
    for s in exam.sections:
        print(f"  {s.title}: {len(s.questions)} questions")
        for q in s.questions:
            print(f"    Q{q.number:<2} {q.kind:<13} {describe(q)}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(asdict(exam), f, ensure_ascii=False, indent=2)
    print("Wrote", output_json)


if __name__ == "__main__":
    main()