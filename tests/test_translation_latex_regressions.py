"""Synthetic regression coverage for translation and LaTeX defect classes."""

import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from latex_converter.latex_renderer import ExamFormatError, LatexRenderer, RenderOptions
from json_interface.parser import MarkdownExamParser
from json_interface.latex_utils import normalize_math_markup, normalize_option_math as normalize_json_option_math
import main as pipeline_main
from translation.chinese.exam_translator import ExamTranslator
from translation.chinese.prompt import GLOSSARY, SYSTEM_PROMPT, render_glossary
from translation.chinese.translation_validation import (
    added_math_atoms,
    math_atom_difference,
    normalize_translated_text,
    validation_errors,
)

ROOT = Path(__file__).resolve().parents[1]
REGRESSION_JSON = ROOT / "json_interface" / "gaokao_2024_national_A_wen.json"


class FakeCompletions:
    """Scriptable OpenAI-compatible completion endpoint for tests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=json.dumps(value, ensure_ascii=False)),
        )])


class FakeLLM:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def test_math_validation_compares_atoms_not_span_boundaries():
    assert validation_errors("已知 $A, B$", "Given $A$ and $B$") == []
    for source, translated in [
        ("$a$ - 2b", "$a - 2b$"),
        ("$(-6,4$", "$(-6,4)$"),
        ("$x$ > 0", "$x > 0$"),
        ("$p$ = 0.5", "$p = 0.5$"),
        ("|$AB$|=2", "$|AB|=2$"),
        ("$AD$ = 4, $AB$ = $BC$ = $EF$ = 2", "$AD = 4, AB = BC = EF = 2$"),
        ("$x; y$", "$x y$;"),
    ]:
        assert validation_errors(source, translated) == []
    assert "\\perp" in (math_atom_difference("若 $m\\perp n$", "If $m n$" ) or "")
    assert "\\%" in (math_atom_difference("置信度 $95\\%$", "confidence $95$" ) or "")
    assert validation_errors("若 $m\\perp n$", "If $m\\parallel n$")
    assert validation_errors("置信度 $95\\%$", "confidence $95$ percent")
    assert validation_errors("$150$", "$15$")
    assert validation_errors("$a$", "$a+b$") == []
    assert added_math_atoms("$a$", "$a+b$") == ["+", "b"]


def test_complete_paragraph_is_one_translation_unit_and_retry_feedback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "fake-model")
    source = "已知 $x>0$，求值。"
    fake = FakeLLM([
        {"0": "Given $x=0$, find the value."},
        {"0": "Given $x=0$, find the value."},
        {"0": "Given ⟦MATH_0⟧, find the value."},
    ])
    translator = ExamTranslator(client=fake, model="fake-model", attempts=3, backoff=0)
    data = {"sections": [{"ordinal": "I", "title": "Choice", "questions": [{"number": 1, "stem": source}]}]}

    result = translator.translate(data)

    assert result["sections"][0]["questions"][0]["stem"] == "Given $x>0$, find the value."
    calls = fake.chat.completions.calls
    first_payload = json.loads(calls[0]["messages"][1]["content"])
    assert first_payload["items"][0]["segments"] == [source]
    retry_payload = json.loads(calls[1]["messages"][1]["content"])
    assert "validation_feedback" in retry_payload["items"][0]
    assert "missing math atoms" in retry_payload["items"][0]["validation_feedback"]
    fallback_payload = json.loads(calls[2]["messages"][1]["content"])
    assert "⟦MATH_0⟧" in fallback_payload["items"][0]["segments"][0]
    assert len(calls) == 3


def test_translation_units_split_on_blank_lines_and_keep_each_paragraph_whole():
    source = "已知 $x>0$，求值。\n\n(1) 求 $a$。\n\n(2) 证明 $b$。"
    translator = ExamTranslator(client=FakeLLM([]), model="fake-model")
    units = translator.collect_translation_units([{"stem": source}])
    assert [unit.text for unit in units] == ["已知 $x>0$，求值。", "(1) 求 $a$。", "(2) 证明 $b$。"]
    assert [source[start:end] for unit in units for start, end in unit.spans] == [unit.text for unit in units]


def test_translated_paragraphs_rejoin_with_identical_blank_line_separators():
    fake = FakeLLM([{"0": "Given $x>0$, evaluate.", "1": "(1) Find $a$.", "2": "(2) Prove $b$."}])
    translator = ExamTranslator(client=fake, model="fake-model", attempts=0, backoff=0)
    source = "已知 $x>0$，求值。\n\n(1) 求 $a$。\n\n(2) 证明 $b$。"
    result = translator.translate({"stem": source})
    assert result["stem"] == "Given $x>0$, evaluate.\n\n(1) Find $a$.\n\n(2) Prove $b$."


@pytest.mark.parametrize(("source", "expected"), [
    ("一、Single-Choice Questions", "I. Single-Choice Questions"),
    ("十二、Fill-in-the-Blank Questions", "XII. Fill-in-the-Blank Questions"),
    ("二十一、Free-Response Questions", "XXI. Free-Response Questions"),
    ("Check， Alpha、Beta:: gamma,, delta", "Check, Alpha, Beta: gamma, delta"),
])
def test_translation_normalization_outside_math_only(source, expected):
    assert normalize_translated_text(source) == expected


def test_glossary_is_single_extendable_source_and_prompt_has_clause_rules():
    assert GLOSSARY["甲"] == "A"
    assert GLOSSARY["通项公式"] == "general term"
    assert render_glossary() in SYSTEM_PROMPT.format(context="(none)", glossary=render_glossary())
    assert "Never drop" in SYSTEM_PROMPT
    assert "one complete original field string" in SYSTEM_PROMPT


def test_renderer_repairs_text_math_options_tables_figures_and_percent():
    renderer = LatexRenderer()
    assert renderer.normalize_latex_text("95% confidence") == r"95\% confidence"
    assert renderer.normalize_latex_text("2x - $y$ - 2 = 0") == "$2x - y - 2 = 0$"
    assert renderer.normalize_latex_text("$x$ > 0 when x varies") == "$x > 0$ when x varies"
    assert renderer.normalize_latex_text("|$AB$|") == "$|AB|$"
    assert renderer.normalize_latex_text("expression\n²\ncontinues") == "expression\ncontinues"
    assert renderer.normalize_option_math("-2") == "$-2$"
    assert renderer.normalize_option_math(r"{1, 2, 3}") == r"$\{1, 2, 3\}$"
    assert renderer.normalize_option_math("___") == r"\_\_\_"
    assert renderer.normalize_table_cell("0.050") == "$0.050$"
    assert renderer.normalize_latex_text("$x，y$") == "$x，y$"

    tex = renderer.render({
        "schema_version": "1.0",
        "metadata": {"title": "Synthetic 95% Exam"},
        "sections": [{"ordinal": "I", "title": "Choice", "questions": [{
            "number": 1,
            "stem": "Graph choice",
            "options": [{"label": "A", "text": "(D) y_{down}", "figure": {"label": "Option A", "width_cm": 40, "height_cm": 2}}],
            "blocks": [{"type": "table", "rows": [["Statistic", "0.050"], [r"$\frac{a}{b}$", "2"]]}],
        }]}],
    })
    assert r"\%" in tex
    assert "(D) y_{down}" not in tex
    assert r"\examfigure{7.25}{2}{Option A}" in tex
    assert "0.050" in tex and "$0.050$" in tex
    assert r"\_\_\_" in renderer.render({"sections": [{"title": "Fill", "questions": [{"number": 1, "stem": "Fill ___ now"}]}]})


def test_renderer_cleans_markdown_html_and_duplicate_escape_artifacts():
    renderer = LatexRenderer()
    assert renderer.normalize_latex_text("&#x20; **\\newcommand**") == r"  \newcommand"
    assert renderer.normalize_latex_text("*% generated comment *") == "% generated comment "
    assert renderer.normalize_latex_text(r"$\\{x_1, x_2\\}$") == r"$\{x_1, x_2\}$"
    assert renderer.normalize_latex_text(r"95\\% confidence") == r"95\% confidence"
    assert renderer.normalize_latex_text(r"\\[\frac{a_1}{b^2}\\]") == r"\[\frac{a_1}{b^2}\]"
    assert renderer.normalize_latex_text(r"$\begin{pmatrix}a&b\\c&d\end{pmatrix}$") == (
        r"\[\begin{pmatrix}a&b\\c&d\end{pmatrix}\]"
    )


def test_image_rendering_uses_active_includegraphics_with_missing_file_fallback():
    renderer = LatexRenderer(RenderOptions(figures="image"))
    tex = renderer.render({
        "sections": [{"ordinal": "I", "title": "Geometry", "questions": [{
            "number": 1,
            "stem": "Select the diagram.",
            "options": [{"label": "A", "text": None, "figure": {
                "label": "Option A", "width_cm": 3.3, "height_cm": 2.5,
                "source_path": "imgs/diagram.jpg",
            }}],
            "blocks": [{"type": "figure", "label": "Main figure", "width_cm": 6.2,
                        "height_cm": 5.8, "source_path": "imgs/main.jpg"}],
            "sub_questions": [],
        }]}],
    })
    assert r"\usepackage{graphicx}" in tex
    assert r"\IfFileExists{imgs/diagram.jpg}{\includegraphics[width=3.3cm]{imgs/diagram.jpg}}" in tex
    assert r"\IfFileExists{imgs/main.jpg}{\includegraphics[width=6.2cm]{imgs/main.jpg}}" in tex
    assert "% real figure:" not in tex
    assert "\\section\\*" not in tex
    assert r"\arabic\*" not in tex and r"\alph\*" not in tex and r"\Alph\*" not in tex


def test_renderer_rejects_unbalanced_dollar_math():
    renderer = LatexRenderer()
    exam = {"sections": [{"title": "Synthetic", "questions": [{"number": 1, "stem": "broken $x + 1"}]}]}
    with pytest.raises(ExamFormatError, match=r"unbalanced \$ math delimiter"):
        renderer.render(exam)


def test_renderer_rejects_unbalanced_grouping_inside_math_with_path():
    renderer = LatexRenderer()
    exam = {"sections": [{"title": "Synthetic", "questions": [{"number": 1, "stem": "point $(-6,4$"}]}]}
    with pytest.raises(ExamFormatError, match=r"unbalanced .* at \$\.sections\[0\]\.questions\[0\]\.stem"):
        renderer.render(exam)


def test_stage_two_normalizes_partial_math_and_preserves_set_braces():
    examples = {
        "$x$ > 0": "$x > 0$",
        "$p$ = 0.5": "$p = 0.5$",
        "|$AB$|=2": "$|AB|=2$",
        "2x - $y$ - 2 = 0": "$2x - y - 2 = 0$",
        "$AB$ = $BC$ = $EF$ = 2": "$AB = BC = EF = 2$",
        "(a > $b$ > 0)": "$(a > b > 0)$",
        "$AD$ = 4, $AB$ = $BC$ = $EF$ = 2": "$AD = 4$, $AB = BC = EF = 2$",
    }
    for source, expected in examples.items():
        assert normalize_math_markup(source) == expected
    assert normalize_json_option_math("{1, 2, 3}") == r"$\{1, 2, 3\}$"


def test_translated_chinese_gate_paths_and_tex_font_preflight(monkeypatch):
    assert pipeline_main.chinese_string_paths({"stem": "English", "choice": "仍有中文"}) == ["$.choice"]
    monkeypatch.setattr(pipeline_main.shutil, "which", lambda name: f"/usr/bin/{name}")
    outputs = [SimpleNamespace(stdout="/texmf/fonts/lmroman12-regular.otf", returncode=0),
               SimpleNamespace(stdout="Noto Sans CJK SC", returncode=0)]
    monkeypatch.setattr(pipeline_main.subprocess, "run", lambda *args, **kwargs: outputs.pop(0))
    pipeline_main.check_tex_prerequisites()

    outputs = [SimpleNamespace(stdout="", returncode=0), SimpleNamespace(stdout="Noto Sans CJK SC", returncode=0)]
    monkeypatch.setattr(pipeline_main.subprocess, "run", lambda *args, **kwargs: outputs.pop(0))
    with pytest.raises(RuntimeError, match="fonts-lmodern"):
        pipeline_main.check_tex_prerequisites()


def test_pipeline_stops_before_latex_if_translation_contains_chinese(tmp_path, monkeypatch):
    from types import SimpleNamespace
    input_path = tmp_path / "synthetic.md"
    input_path.write_text("synthetic source", encoding="utf-8")
    output_dir = tmp_path / "out"
    source_exam = {
        "schema_version": "1.0",
        "sections": [{"title": "Section", "questions": [{"number": 1, "stem": "Chinese"}]}],
    }
    parsed_exam = SimpleNamespace(
        to_dict=lambda: source_exam,
        warnings=[],
        sections=[SimpleNamespace(questions=[object()])],
    )
    monkeypatch.setattr(pipeline_main, "fix_markdown", lambda _: "fixed markdown")
    generated_metadata = {
        "title": "Synthetic Examination", "subtitle": "Mathematics · Sample Paper",
        "year": 2025, "subject": "Mathematics", "paper": "Sample Paper",
        "header_left": "Sample Paper", "header_right": "Mathematics Examination",
    }
    monkeypatch.setattr(pipeline_main, "generate_exam_metadata", lambda _: generated_metadata)
    monkeypatch.setattr(pipeline_main, "MarkdownExamParser", lambda: SimpleNamespace(parse=lambda *_: parsed_exam))
    monkeypatch.setattr(pipeline_main, "ExamTranslator", lambda: SimpleNamespace(
        translate=lambda _: {"sections": [{"title": "Section", "questions": [{"number": 1, "stem": "仍有中文"}]}]},
        failed=[], failed_paths=[],
    ))
    monkeypatch.setattr(pipeline_main, "validate_final_json", lambda *_: [])
    monkeypatch.setattr(pipeline_main, "LatexRenderer", lambda *_: pytest.fail("LaTeX stage must not run"))
    status = pipeline_main.run_pipeline(input_path, output_dir)
    assert status == 2
    assert (output_dir / "synthetic_translated.json").is_file()
    assert (output_dir / "synthetic.metadata.yaml").is_file()
    assert not (output_dir / "synthetic.tex").exists()


def test_metadata_is_generated_from_the_metadata_prompt_and_validated(monkeypatch):
    import structural_fixes.main as structural_main

    captured = {}
    response = (
        "title: Regional Examination\n"
        "subtitle: Mathematics · Sample Paper\n"
        "year: 2025\nsubject: Mathematics\npaper: Sample Paper\n"
        "header_left: Sample Paper\nheader_right: Mathematics Examination\n"
    )

    def fake_process(prompt, context, *, client=None):
        captured["prompt"] = prompt
        captured["context"] = context
        return response

    monkeypatch.setattr(structural_main, "process", fake_process)
    metadata = structural_main.generate_exam_metadata(
        "# Regional Examination\n\n## I. Mathematics\n\n1. Find x."
    )

    assert metadata["subject"] == "Mathematics"
    assert metadata["year"] == 2025
    assert "title, subtitle, year, subject, paper" in captured["prompt"]
    assert "## I. Mathematics" in captured["context"]


def test_metadata_generation_rejects_invalid_types_and_missing_title(monkeypatch):
    import structural_fixes.main as structural_main

    for response in ("not: [valid", "subject: 123\nyear: 20.5", "subject: Mathematics"):
        monkeypatch.setattr(structural_main, "process", lambda *args, response=response, **kwargs: response)
        with pytest.raises(ValueError):
            structural_main.generate_exam_metadata("# Exam\n\n## Questions\n\n1. Find x.")


def test_gaokao_regression_fixture_preserves_closing_math_delimiter_and_figures():
    exam = json.loads(REGRESSION_JSON.read_text(encoding="utf-8"))
    rendered = LatexRenderer().render(exam)
    assert "$(-6,4)$" in rendered
    assert "\\begin{tabular}" in rendered
    assert "\\examfigure" in rendered
    assert re.search(r"(?<!\\)\$(?:[^$]|\\\$)*?(?<!\\)\$", rendered)


def test_renderer_converts_textual_math_words_and_clamps_raw_figure_width():
    exam = {
        "sections": [{"ordinal": "I", "title": "Geometry", "questions": [{
            "number": 1,
            "stem": r"Use the $x-axis$ and $y-coordinate$.",
            "blocks": [{"type": "text", "content": r"\includegraphics[width=0.09\linewidth]{fig.jpg}"}],
        }]}],
    }
    tex = LatexRenderer().render(exam)
    assert r"I. Geometry" in tex
    assert r"$x$-axis" in tex
    assert r"$y$-coordinate" in tex
    assert r"width=0.4\linewidth" in tex


def test_renderer_wraps_circled_propositions_and_paren_subquestions_around_tables():
    exam = {
        "sections": [{"ordinal": "II", "title": "Proof", "questions": [{
            "number": 1,
            "stem": "Consider the following statements:\n\n① First claim.\n\n② Second claim.\n\nSelect the true claims.",
            "blocks": [
                {"type": "text", "content": "(1) Complete the table:"},
                {"type": "table", "rows": [["x", "y"], ["1", "2"]]},
                {"type": "text", "content": "(2) Explain your answer.\n\n(a) Give a reason.\n\n(b) State the result."},
            ],
        }]}],
    }
    tex = LatexRenderer().render(exam)
    assert r"\section*{II. Proof}" in tex
    assert tex.count(r"\begin{statements}") == 1
    assert r"\item[①] First claim." in tex
    assert r"\item[②] Second claim." in tex
    assert tex.index(r"\end{statements}") < tex.index("Select the true claims.")
    assert tex.count(r"\begin{subquestions}") == 2
    assert tex.count(r"\end{subquestions}") == 2
    assert "Complete the table" in tex and r"\begin{tabular}" in tex
    assert tex.index("Complete the table") < tex.index(r"\begin{tabular}") < tex.index("Explain your answer")


def test_renderer_places_parent_figures_and_shared_tail_outside_subquestions():
    exam = {
        "sections": [{"title": "Geometry", "questions": [{
            "number": 1,
            "stem": (
                "Question stem.\n\n(1) First part.\n\n(2) Second part.\n\n"
                r"\begin{figure}[h]\centering\includegraphics[width=0.09\linewidth]{diagram.jpg}\end{figure}"
            ),
            "blocks": [{"type": "text", "content": "(1) Complete the shared table:"},
                       {"type": "table", "rows": [["Group", "Count"], ["A", "2"]]},
                       {"type": "text", "content": "(2) Explain the result.\n\nReference: $K^2=1$.\n\nLookup values: 0.05 0.01"}],
        }]}],
    }
    tex = LatexRenderer().render(exam)
    center_start = tex.index(r"\begin{center}", tex.index("Question stem"))
    subquestions_start = tex.index(r"\begin{subquestions}")
    shared_reference = tex.index("Reference:")
    subquestions_end = tex.index(r"\end{subquestions}")
    assert center_start < subquestions_start
    assert tex.index(r"\includegraphics[width=0.4\linewidth]") < subquestions_start
    assert "\\begin{figure}" not in tex
    assert r"\usepackage{graphicx}" in tex
    assert subquestions_end < shared_reference
    assert tex.index("Complete the shared table") < tex.index(r"\begin{tabular}") < subquestions_end
    assert "$K^2=1$" in tex and "Lookup values: 0.05 0.01" in tex


def test_renderer_normalizes_math_shortcuts_cases_axes_and_choice_blank():
    renderer = LatexRenderer()
    assert renderer.normalize_latex_text(r"$r \parallel s, u \parallel v, d = 4$") == r"$r \parallel s$, $u \parallel v$, $d = 4$"
    assert renderer.normalize_latex_text(r"$P=(1, 2), Q=(3, 4)$") == r"$P=(1, 2)$, $Q=(3, 4)$"
    assert renderer.normalize_latex_text(r"$EF // AD, BC // AD$") == r"$EF \parallel AD$, $BC \parallel AD$"
    assert renderer.normalize_latex_text(r"$l || m$") == r"$l \parallel m$"
    assert renderer.normalize_latex_text(r"$MF \perp x$ axis and $AQ \perp y$ axis") == r"$MF \perp x$-axis and $AQ \perp y$-axis"
    assert renderer.normalize_latex_text(r"$B, N$ is the midpoint") == r"$B$. $N$ is the midpoint"
    assert renderer.normalize_latex_text(r"$B$. $N$ is the midpoint") == r"$B$. $N$ is the midpoint"
    cases = renderer.normalize_latex_text(r"$\begin{cases}x=t\\y=t+a\end{cases}$")
    assert cases == r"\[\begin{cases}x=t\\y=t+a\end{cases}\]"
    line_cases = renderer.normalize_latex_text(r"$l: \begin{cases}x=t\\y=t+a\end{cases}$")
    assert line_cases == r"$l:$ \[\begin{cases}x=t\\y=t+a\end{cases}\]"
    assert renderer.escape_percent_outside_math(r"\[\text{95%}\]") == r"\[\text{95%}\]"
    assert r"\mbox{(\kern0.35em)}" in renderer.normalize_latex_text("Find x = ( )")


def test_markdown_source_edit_is_visible_after_parse_and_render():
    source_markdown = "# Synthetic Exam\n\n## I. Questions\n\n1. Original wording.\n"
    edited_markdown = source_markdown.replace("Original wording", "Updated wording")
    source_exam = MarkdownExamParser().parse(source_markdown).to_dict()
    edited_exam = MarkdownExamParser().parse(edited_markdown).to_dict()
    renderer = LatexRenderer()
    assert "Original wording" in renderer.render(source_exam)
    assert "Updated wording" in renderer.render(edited_exam)


def test_markdown_front_matter_populates_exam_metadata_and_latex_headers():
    markdown = """---
title: Regional Mathematics Examination
subtitle: Mathematics · Sample Paper
year: 2025
subject: Mathematics
paper: Sample Paper
header_left: Sample Paper
header_right: Regional Examination
---

# Regional Mathematics Examination

## I. Questions

1. Find the value of $x$.
"""
    exam_data = MarkdownExamParser().parse(markdown, "sample.md").to_dict()
    metadata = exam_data["metadata"]
    assert metadata["title"] == "Regional Mathematics Examination"
    assert metadata["subject"] == "Mathematics"
    assert metadata["header_left"] == "Sample Paper"
    assert metadata["header_right"] == "Regional Examination"

    latex = LatexRenderer().render(exam_data)
    assert r"\fancyhead[L]{\small\examheaderleft}" in latex
    assert r"\fancyhead[R]{\small\examheaderright}" in latex
    assert r"\newcommand{\examsubtitle}{Mathematics · Sample Paper}" in latex


def test_markdown_parser_structures_parts_statements_figures_and_blanks():
    markdown = r"""# Synthetic Examination

## 一、单选题

1. Consider these claims:

① First proposition.

② Second proposition.

2. Find ___ when x > 0.

## II. Problems

3. Given the geometry:

<div><img src="imgs/img_in_image_box_10_20_410_320.jpg" width="5%" /></div>

(1) Establish $r // s$.

(2) Calculate the value.

(a) First nested step.

(b) Second nested step.
"""
    exam = MarkdownExamParser().parse(markdown)
    proposition_question = exam.get_question(1)
    blank_question = exam.get_question(2)
    geometry_question = exam.get_question(3)

    assert proposition_question is not None
    assert any(block.TYPE == "statement_list" for block in proposition_question.blocks)
    assert blank_question is not None and blank_question.blank_count == 1
    assert r"\blank" in blank_question.stem
    assert geometry_question is not None and len(geometry_question.sub_questions) == 2
    assert len(geometry_question.sub_questions[1].sub_questions) == 2
    figure = next(block for block in geometry_question.blocks if block.TYPE == "figure")
    assert figure.width_cm == 6.2
    assert figure.height_cm == 4.65
    assert r"$r \parallel s$" in geometry_question.sub_questions[0].blocks[0].content


def test_html_graphical_choices_use_null_for_empty_text_and_crop_dimensions():
    image_markup = "\n".join(
        f'<div><img src="imgs/img_in_image_box_{i * 500}_0_{i * 500 + 400}_300.jpg" width="8%" /></div>'
        for i in range(4)
    )
    markdown = f"""# Synthetic Examination

## 一、单选题

1. Choose the matching graph:

{image_markup}

<div>(A)</div>

<div>(B)</div>

<div>(C)</div>

<div>(D)</div>
"""
    question = MarkdownExamParser().parse(markdown).get_question(1)
    assert question is not None
    assert [option.label for option in question.options] == list("ABCD")
    assert all(option.text is None for option in question.options)
    assert all(option.figure is not None for option in question.options)
    assert all(option.figure.width_cm == 3.3 for option in question.options)
    assert all(option.figure.height_cm == 2.48 for option in question.options)


def test_metadata_sidecar_populates_schema_without_editing_markdown(tmp_path):
    markdown_path = tmp_path / "sample.md"
    original = "# Regional Examination\n\n## I. Questions\n\n1. Find $x$.\n"
    markdown_path.write_text(original, encoding="utf-8")
    metadata_path = tmp_path / "sample.metadata.yaml"
    metadata_path.write_text(
        "title: Regional Examination\nsubtitle: Mathematics · Sample Paper\n"
        "year: 2025\nsubject: Mathematics\npaper: Sample Paper\n"
        "header_left: Sample Paper\nheader_right: Mathematics Examination\n",
        encoding="utf-8",
    )

    parser = MarkdownExamParser()
    exam = parser.parse_file(markdown_path)
    assert parser.apply_metadata_sidecar(exam, metadata_path)
    assert exam.metadata.subject == "Mathematics"
    assert exam.metadata.paper == "Sample Paper"
    assert exam.metadata.year == 2025
    assert exam.metadata.header_left == "Sample Paper"
    assert markdown_path.read_text(encoding="utf-8") == original
    rendered = LatexRenderer().render(exam.to_dict())
    assert r"\fancyhead[L]{\small\examheaderleft}" in rendered
    assert r"\fancyhead[R]{\small\examheaderright}" in rendered


def test_markdown_parser_promotes_structure_and_normalizes_latex_markup(tmp_path):
    markdown = r"""# Synthetic Exam

## 一、单选题

1. Choose an answer.

(A) First (B) Second (C) Third (D) Fourth

2. Four statements:

① First statement.

② Second statement.

3. Stem.

(1) First part.

(a) Nested part.

(2) Second part.

4. Fill ____.

5. Conditions EF // AD, BC // AD; and MF $\perp$ $x$ axis.

6. Diagram.

\begin{figure}[h]
\centering
\includegraphics[width=0.09\linewidth]{imgs/img_in_image_box_2059_1248_2291_1465.jpg}
\end{figure}

7. Confidence is 95%.
"""
    parser = MarkdownExamParser()
    exam = parser.parse(markdown, "synthetic.md")
    choices, statements, multipart, blank, math, figure_question, percent = exam.iter_questions()
    assert [option.label for option in choices.options] == ["A", "B", "C", "D"]
    assert any(block.TYPE == "statement_list" for block in statements.blocks)
    assert [part.label for part in multipart.sub_questions] == ["(1)", "(2)"]
    assert multipart.sub_questions[0].sub_questions[0].label == "(a)"
    assert r"\blank" in blank.stem and blank.blank_count == 1
    assert r"$EF \parallel AD$, $BC \parallel AD$" in math.stem
    assert r"$MF \perp x$-axis" in math.stem
    figure = next(block for block in figure_question.blocks if block.TYPE == "figure")
    assert figure.width_cm == 6.2 and figure.height_cm > 0
    assert r"95\%" in percent.stem

    sidecar = tmp_path / "sample.metadata.yaml"
    sidecar.write_text(
        "title: Mathematics Examination\nyear: 2024\nsubject: Mathematics\n"
        "header_left: Paper A\nheader_right: Exam Board\n",
        encoding="utf-8",
    )
    parser.apply_metadata_sidecar(exam, sidecar)
    assert exam.metadata.year == 2024
    assert exam.metadata.subject == "Mathematics"
    assert exam.metadata.header_left == "Paper A"
