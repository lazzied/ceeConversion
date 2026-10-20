# Code guide

This project turns an exam document into structured JSON, an English translation,
LaTeX, and PDFs. The folders are organized by the kind of work they do; most
changes should stay inside the stage that owns that behavior.

## Pipeline at a glance

Run `python main.py [input.md] --output-dir data/output/pipeline` from the project
root. The stages are coordinated in `main.py`:

1. `structural_fixes` normalizes OCR/Markdown details and asks the configured
   correction model to repair document structure.
2. `json_interface.markdown_parser.MarkdownExamParser` parses the corrected
   Markdown into the shared exam model.
3. `translation.chinese.ExamTranslator` translates the JSON while preserving its
   schema and mathematical content.
4. `semantic_review` can review the Chinese and translated JSON when enabled.
5. `latex_converter.LatexRenderer` renders the original and translated JSON;
   the pipeline then compiles both documents to PDF.

The pipeline writes intermediate files to its output directory so each stage's
input and output can be inspected independently.

## Where to make a change

- **OCR/Markdown cleanup:** `structural_fixes/normalization.py`. Each cleanup
  function handles one transformation; `normalize_markdown` lists the order.
- **LLM structural correction:** `structural_fixes/main.py`, `client.py`, and
  `prompts.py`.
- **Markdown to exam data:** `json_interface/markdown_parser.py`.
- **Custom LaTeX to exam data:** `json_interface/parser.py`.
- **Question-kind rules used by both parsers:** `json_interface/classification.py`.
- **Shared exam fields and block types:** `json_interface/dataclasses.py`.
- **Exam JSON to LaTeX:** `latex_converter/latex_renderer.py`; document setup is
  in `latex_converter/preamble.py`.
- **Translation behavior and validation:** `translation/chinese/`.
- **Read-only semantic checks:** `semantic_review/`.

Both parsers create the same dataclasses. That shared model is the handoff
between reading a source document and rendering or translating it. Prefer
adding a block or field there only when the schema itself needs to change;
format-specific recognition belongs in the corresponding parser.

## Parsing entry points

New Markdown code should import `MarkdownExamParser` from
`json_interface.markdown_parser`. LaTeX source code should use `ExamParser` from
`json_interface.parser`. The old import
`from json_interface.parser import MarkdownExamParser` remains available for
existing callers.

Parsing warnings are attached to the returned exam as `exam.warnings`. Recoverable
format issues should be reported there; malformed input that cannot be safely
interpreted should raise an error rather than silently dropping exam content.
