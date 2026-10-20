numbers_prompt = r"""

Correct ONLY the numbering of main questions and the labeling
and layout of their answer options.

RULES:
- Correct incorrect main-question numbers and their order.
- Fill missing question numbers only when the intended sequence
  is clear.
- Multiple-choice questions should have options labeled A, B, C, D.
- Add a missing option label only when its corresponding option
  text already exists.
- Correct duplicated or out-of-order labels only when the intended
  mapping is clear.
- Treat each \begin{figure}...\end{figure} block or standalone image
  reference inside a multiple-choice question as possible option content.
- When a question has exactly four option images and its A-D labels are
  misplaced or incomplete, use the images' original document order as the
  option order: image 1 belongs to A, image 2 to B, image 3 to C, image 4
  to D. Move each existing image block directly after its corresponding
  label, and add a missing label only for one of those four existing images.
- Preserve every image block, path, and sequence exactly. Keep any existing
  non-image text after its explicit option label with that option. Do not
  apply this rule when the number of option images is not exactly four or
  the question does not clearly present graphical answer choices.
- Put each answer option on its own line, separated by a blank
  line, in the form "(A) text". Move inline options onto separate
  lines without changing their text.
- Never invent missing questions, options, answers, or content.
- Do not change, summarize, translate, or solve any question.
- Preserve all wording, equations, LaTeX, symbols, answers,
  tables, and formatting, except for the option line breaks above.
- Preserve backslashes and LaTeX commands exactly.
- Keep every existing inline-math delimiter and its surrounding expression
  intact. Do not split formulas across text; repair unmatched grouping
  parentheses or brackets only when the intended pair is unambiguous from the
  same mathematical expression. If it is ambiguous, leave it unchanged for
  the JSON/LaTeX validation stage to report with a field path.
- Remove only standalone OCR superscript glyph artifacts that are not part
  of a question, option, formula, or other content.
- Do not renumber subquestions such as (1), (2), (a), or (b).
- Do not confuse subquestion labels with answer-option labels.
- Do not split numbered items such as ①, ②, ③, ④ inside a
  question; only answer options are put on separate lines.
- Preserve section headings and their order.
- Preserve HTML table markup and cell contents verbatim. Do not convert HTML
  tables to Markdown pipe tables or rewrite them as prose; the JSON/LaTeX stage
  handles table conversion.
- If anything is ambiguous, leave it unchanged.
- Return the complete corrected exam only.
- Do not include explanations, comments, or code fences.

EXAM TEXT:
"""


metadata_prompt = r"""
Extract metadata for this exam and return exactly one YAML mapping with these keys:
title, subtitle, year, subject, paper, header_left, header_right.

Rules:
- Use only information supported by the exam title, section headings, or content.
- Keep title and header values in the same language as the source exam. The later
  translation stage will translate them.
- Keep the exam name in title. Put subject, track, and paper designation in
  subtitle where supported; also return subject and paper separately.
- Return year as an integer when stated, otherwise null.
- Return null for any unsupported value. Never guess a location, exam board,
  track, or paper designation.
- Make header_left and header_right concise running-header text derived from
  supported metadata. If no suitable text is supported, return null.
- Quote YAML strings when needed. Do not use Markdown fences or add explanation.

EXAM TITLE, HEADINGS, AND CONTENT:
"""






