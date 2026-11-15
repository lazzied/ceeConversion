"""Regression tests for corpus-style and standalone LaTeX splitting."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import latex_exam_splitter as splitter
from latex_exam_splitter import output as splitter_output


class LatexExamSplitterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.tmp_path = Path(self.temporary_directory.name)

    def write_source(self, body: str) -> Path:
        (self.tmp_path / "styles.tex").write_text("% synthetic styles\n", encoding="utf-8")
        source = self.tmp_path / "paper.tex"
        source.write_text(body, encoding="utf-8")
        return source

    def test_environment_format_keeps_numbering_suffix_and_labels(self) -> None:
        source = self.write_source(
            r"""\chapter{Exam}
\section{Multiple choice}
\begin{problem}Question one: $x^2=1$?\label{q:first}\end{problem}
\begin{answer}A\end{answer}
\begin{solution}Because $x=1$.\end{solution}
\begin{problem}[A]Question two.\end{problem}
\begin{solution}Second explanation.\end{solution}
"""
        )

        plan = splitter.analyze_document(source, corpus_root=self.tmp_path)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.question_count, 2)
        self.assertEqual(plan.paired_question_count, 2)
        self.assertIn("Question one", plan.questions_body)
        self.assertNotIn("Because $x=1$.", plan.questions_body)
        self.assertIn(r"\problem[A]", plan.solutions_body)
        self.assertIn(r"\label{q:first}", plan.solutions_body)
        self.assertIn("Second explanation.", plan.solutions_body)
        wrapper = splitter_output._document_wrapper(
            plan.questions_body, plan.style_file, show_solutions=False, graphics_directory=self.tmp_path
        )
        self.assertIn(r"\graphicspath{{\detokenize{", wrapper)

    def test_legacy_inline_commands_are_moved_with_their_math(self) -> None:
        source = self.write_source(
            r"""\chapter{Older format}
\begin{problem}Find $a$.
\answer{B}
\end{problem}
\begin{problem}Find $b$.
\solution{Since $b=2$, the answer is $2$.}
\end{problem}
"""
        )

        plan = splitter.analyze_document(source, corpus_root=self.tmp_path)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertNotIn(r"\answer{B}", plan.questions_body)
        self.assertNotIn(r"\solution{Since $b=2$, the answer is $2$.}", plan.questions_body)
        self.assertIn(r"\answer{B}", plan.solutions_body)
        self.assertIn("$b=2$", plan.solutions_body)

    def test_reused_numbering_control_stays_before_the_next_problem(self) -> None:
        source = self.write_source(
            r"""\chapter{Exam}
\begin{problem}Question one.\end{problem}
\begin{solution}One.\end{solution}
\reuseproblemnumber
\begin{problem}[B]Question one part B.\end{problem}
\begin{solution}Part B.\end{solution}
"""
        )

        plan = splitter.analyze_document(source, corpus_root=self.tmp_path)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.solutions_body.count(r"\textbf{Solutions}"), 2)
        self.assertLess(
            plan.solutions_body.index(r"\reuseproblemnumber"),
            plan.solutions_body.index(r"\problem[B]"),
        )

    def test_missing_solution_keeps_later_original_question_numbers(self) -> None:
        source = self.write_source(
            r"""\chapter{Exam}
\begin{problem}First question.\end{problem}
\begin{solution}First solution.\end{solution}
\begin{problem}Question without a supplied solution.\end{problem}
\begin{problem}Third question.\end{problem}
\begin{solution}Third solution.\end{solution}
"""
        )

        plan = splitter.analyze_document(source, corpus_root=self.tmp_path)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.question_count, 3)
        self.assertEqual(plan.paired_question_count, 2)
        self.assertEqual(plan.solutions_body.count(r"\textbf{Solutions}"), 2)
        self.assertIn(r"\problem\problembodyend", plan.solutions_body)

    def test_comments_and_verbatim_examples_are_not_treated_as_markers(self) -> None:
        source = self.write_source(
            r"""\begin{problem}Question.\end{problem}
% \begin{solution}not real\end{solution}
\begin{verbatim}
\begin{solution}also not real\end{solution}
\end{verbatim}
\begin{solution}Real solution.\end{solution}
"""
        )

        plan = splitter.analyze_document(source, corpus_root=self.tmp_path)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.paired_question_count, 1)
        self.assertIn("not real", plan.questions_body)
        self.assertIn("also not real", plan.questions_body)
        self.assertIn("Real solution.", plan.solutions_body)

    def test_orphan_solution_before_first_problem_fails_loudly(self) -> None:
        source = self.write_source(
            r"""\begin{solution}Orphan.\end{solution}
\begin{problem}Question.\end{problem}
"""
        )

        with self.assertRaisesRegex(splitter.SplitError, "before the first problem"):
            splitter.analyze_document(source, corpus_root=self.tmp_path)

    def test_question_only_document_is_not_split(self) -> None:
        source = self.write_source(r"\begin{problem}Question only.\end{problem}")
        self.assertIsNone(splitter.analyze_document(source, corpus_root=self.tmp_path))

    def test_standalone_document_preamble_and_trailer_are_preserved(self) -> None:
        source = self.tmp_path / "standalone.tex"
        source.write_text(
            r"""\documentclass{article}
\usepackage{amsmath}
\newif\ifshowanswer
\begin{document}
\section{Exam}
\begin{problem}Question.\end{problem}
\begin{solution}Solution.\end{solution}
\end{document}
% trailing source comment
""",
            encoding="utf-8",
        )

        plan = splitter.analyze_document(source)

        self.assertIsNotNone(plan)
        assert plan is not None
        questions_document = splitter_output._document_wrapper(
            plan.questions_body,
            plan.style_file,
            show_solutions=False,
            preamble=plan.document_preamble,
            document_start=plan.document_start,
            document_end=plan.document_end,
            trailer=plan.document_trailer,
        )
        self.assertIn(r"\documentclass{article}", questions_document)
        self.assertIn(r"\usepackage{amsmath}", questions_document)
        self.assertIn(r"\showanswerfalse", questions_document)
        self.assertIn("% trailing source comment", questions_document)

    def test_outputs_are_separate_and_both_are_compiler_checked(self) -> None:
        source = self.write_source(
            r"\begin{problem}Question.\end{problem}\begin{solution}Answer.\end{solution}"
        )
        output = self.tmp_path / "generated"

        def fake_compile(tex_file: Path, _engine: str, _working_directory: Path) -> None:
            tex_file.with_suffix(".pdf").touch()

        with patch.object(splitter_output, "_compile_latex", side_effect=fake_compile) as compile_file:
            result = splitter.split_document(source, output, self.tmp_path)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotEqual(result.questions_file, source)
        self.assertNotEqual(result.solutions_file, source)
        self.assertTrue(source.read_text(encoding="utf-8").startswith(r"\begin{problem}"))
        self.assertEqual(compile_file.call_count, 2)
        self.assertTrue(result.compiled)

    def test_existing_output_is_not_overwritten(self) -> None:
        source = self.write_source(
            r"\begin{problem}Question.\end{problem}\begin{solution}Answer.\end{solution}"
        )
        output = self.tmp_path / "generated"
        output.mkdir()
        existing = output / "paper_questions.tex"
        existing.write_text("keep me", encoding="utf-8")

        with self.assertRaisesRegex(splitter.SplitError, "Output already exists"):
            splitter.split_document(source, output, self.tmp_path, compile_outputs=False)
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep me")

    def test_compilation_failure_reports_compiler_diagnostic(self) -> None:
        tex_file = self.tmp_path / "broken.tex"
        tex_file.write_text("", encoding="utf-8")
        failed = type(
            "Completed",
            (),
            {"returncode": 1, "stdout": "! Undefined control sequence.\n", "stderr": ""},
        )()

        with (
            patch.object(splitter_output.shutil, "which", return_value="/usr/bin/xelatex"),
            patch.object(splitter_output.subprocess, "run", return_value=failed),
            self.assertRaisesRegex(splitter.SplitError, "Undefined control sequence"),
        ):
            splitter_output._compile_latex(tex_file, "xelatex", self.tmp_path)

    def test_pair_validation_attempts_both_generated_documents(self) -> None:
        source = self.write_source(
            r"\begin{problem}Question.\end{problem}\begin{solution}Answer.\end{solution}"
        )
        failures: list[Path] = []

        def fail_compile(tex_file: Path, _engine: str, _working_directory: Path) -> None:
            failures.append(tex_file)
            raise splitter.SplitError(f"failure for {tex_file.name}")

        with patch.object(splitter_output, "_compile_latex", side_effect=fail_compile):
            with self.assertRaisesRegex(splitter.SplitError, "did not fully compile"):
                splitter.split_document(
                    source,
                    self.tmp_path / "generated",
                    self.tmp_path,
                    compile_outputs=True,
                )

        self.assertEqual(len(failures), 2)


if __name__ == "__main__":
    unittest.main()
