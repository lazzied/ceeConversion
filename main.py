"""Run exam conversion from Markdown through translated LaTeX."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from json_interface.markdown_parser import MarkdownExamParser
from latex_converter.latex_renderer import ExamFormatError, LatexRenderer, RenderOptions
from semantic_review.reviewers import (
    ChineseSourceReviewer,
    ReviewReport,
    ReviewUncertainty,
    TranslationReviewer,
    schema_warning_uncertainties,
)
from structural_fixes.main import fix_markdown, generate_exam_metadata
from structural_fixes.validation import validate_exam
from translation.chinese.exam_translator import ExamTranslator
from translation.chinese.translation_errors import TranslationError
from translation.chinese.translation_validation import validate_final_json
from pipeline_validation import PipelineValidationReport

LATIN_TEX_FONT = "lmroman12-regular.otf"
CJK_FONT_PATTERN = r"CJK|WenQuanYi|AR PL|Noto (?:Sans|Serif) CJK"
FONT_INSTALL_HINT = "sudo apt install fonts-lmodern texlive-fonts-recommended fonts-noto-cjk; fc-cache -f"

log = logging.getLogger("exam_pipeline")
PIPELINE_STAGES = (
    "input",
    "structural_fixes",
    "json_interface",
    "translation",
    "latex_rendering",
    "pdf_compilation",
    "semantic_review",
)


def write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def write_json(path: Path, value: Any) -> None:
    """Serialize JSON without escaping the exam's Chinese text."""
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def inspect_latex_log(log_path: Path) -> dict[str, list[str]]:
    """Collect compiler errors, warnings, and layout diagnostics from a log."""
    if not log_path.is_file():
        return {"errors": [], "warnings": [f"compiler log is missing: {log_path}"]}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    errors = [line.strip() for line in lines if line.lstrip().startswith("!")]
    warnings = [
        line.strip() for line in lines
        if "Warning:" in line or "Overfull \\hbox" in line or "Underfull \\hbox" in line
    ]
    return {"errors": errors, "warnings": warnings}


def compile_latex_to_pdf(tex_path: Path, output_dir: Path, working_directory: Path) -> Path:
    """Compile one UTF-8 LaTeX document with XeLaTeX or LuaLaTeX."""
    engine = shutil.which("xelatex") or shutil.which("lualatex")
    if engine is None:
        raise RuntimeError(
            "PDF generation requires XeLaTeX or LuaLaTeX; install a TeX distribution "
            "that provides one of these commands and rerun the pipeline"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={output_dir.resolve()}",
        str(tex_path.resolve()),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=working_directory.resolve(),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"LaTeX compilation timed out for {tex_path.name}") from exc

    pdf_path = output_dir / f"{tex_path.stem}.pdf"
    if result.returncode != 0 or not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        compiler_output = result.stdout + "\n" + result.stderr
        useful_lines = "\n".join(compiler_output.splitlines()[-25:])
        log_path = output_dir / f"{tex_path.stem}.log"
        first_tex_error = ""
        if log_path.is_file():
            first_tex_error = next(
                (line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if line.lstrip().startswith("!")),
                "",
            )
        raise RuntimeError(
            f"{Path(engine).name} failed to compile {tex_path.name}. "
            f"{f'First LaTeX error: {first_tex_error}. ' if first_tex_error else ''}"
            f"Last compiler output:\n{useful_lines}"
        )
    log.info("Created PDF: %s", pdf_path)
    return pdf_path


def check_tex_prerequisites() -> None:
    """Check the default Latin Modern and at least one CJK font before XeLaTeX."""
    kpsewhich = shutil.which("kpsewhich")
    fc_list = shutil.which("fc-list")
    latin_font = ""
    cjk_output = ""
    if kpsewhich:
        result = subprocess.run([kpsewhich, LATIN_TEX_FONT], capture_output=True, text=True, check=False)
        latin_font = result.stdout.strip()
    if fc_list:
        result = subprocess.run([fc_list], capture_output=True, text=True, check=False)
        cjk_output = result.stdout
    missing = []
    if not latin_font:
        missing.append(f"Latin Modern OpenType font ({LATIN_TEX_FONT} via kpsewhich)")
    if not fc_list or not re.search(CJK_FONT_PATTERN, cjk_output, re.I):
        missing.append("a CJK font listed by fc-list")
    if missing:
        raise RuntimeError(
            "Missing XeLaTeX font prerequisite(s): " + ", ".join(missing) + ". "
            f"Install them with: {FONT_INSTALL_HINT}"
        )


def chinese_string_paths(value: Any) -> list[str]:
    """Return exact JSON paths for every string that still contains Chinese."""
    pattern = re.compile(r"[\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323AF]")
    paths: list[str] = []
    def visit(item: Any, path: str) -> None:
        if isinstance(item, str) and pattern.search(item):
            paths.append(path)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
    visit(value, "$")
    return paths


def review_unavailable_report(checker: str, explanation: str) -> ReviewReport:
    """Represent a checker outage as a report uncertainty, not a false pass."""
    report = ReviewReport(checker)
    report.uncertainties.append(ReviewUncertainty(
        path="$",
        question="Could the semantic review complete?",
        missing_evidence=explanation,
    ))
    return report


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    *,
    semantic_review: bool = False,
    source_pdf: Path | None = None,
    review_model: str | None = None,
    allow_untranslated: bool = False,
) -> int:
    """Run Markdown → structural fixes → JSON → translation → LaTeX → PDFs.

    Intermediate files are retained in ``output_dir`` to make each handoff
    inspectable. Semantic review runs only when explicitly enabled. Return 0 on
    success, 2 when translation or enabled review needs attention, or 1 when a
    stage cannot run.
    """
    corrected_path = output_dir / f"{input_path.stem}_corrected.md"
    json_path = output_dir / f"{input_path.stem}.json"
    translated_json_path = output_dir / f"{input_path.stem}_translated.json"
    latex_path = output_dir / f"{input_path.stem}_translated.tex"

    try:
        log.info("Reading Markdown input: %s", input_path)
        markdown = input_path.read_text(encoding="utf-8")
        if not markdown.strip():
            raise ValueError("Markdown input is empty")

        log.info("Stage 1/5: structural fixes")
        corrected_markdown = fix_markdown(markdown)
        write_text(corrected_path, corrected_markdown)

        log.info("Stage 2/5: Generate exam metadata and parse Markdown to JSON")
        metadata = generate_exam_metadata(corrected_markdown)
        metadata_path = output_dir / f"{input_path.stem}.metadata.yaml"
        write_text(metadata_path, yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False))
        log.info("Wrote generated exam metadata: %s", metadata_path)

        markdown_parser = MarkdownExamParser()
        exam = markdown_parser.parse(corrected_markdown, input_path.name)
        apply_metadata_sidecar = getattr(markdown_parser, "apply_metadata_sidecar", None)
        if callable(apply_metadata_sidecar) and apply_metadata_sidecar(exam, metadata_path):
            log.info("Applied generated metadata from %s", metadata_path)
        exam_data = exam.to_dict()
        if not exam_data.get("sections"):
            raise ValueError("JSON interface produced no exam sections")
        if not any(section.questions for section in exam.sections):
            raise ValueError("JSON interface found no numbered questions")
        for warning in exam.warnings:
            log.warning("JSON interface: %s", warning)
        write_json(json_path, exam_data)

        log.info("Stage 3/5: translate exam JSON")
        translator = ExamTranslator()
        translated_data = translator.translate(exam_data)
        if not isinstance(translated_data, dict):
            raise TranslationError("Translation stage did not return an exam JSON object")
        validation_issues = validate_final_json(exam_data, translated_data)
        write_json(translated_json_path, translated_data)
        remaining_chinese_paths = chinese_string_paths(translated_data)
        if remaining_chinese_paths and not allow_untranslated:
            log.error(
                "Pipeline stopped after translation: %d JSON string(s) still contain Chinese; "
                "stages 4 and 5 were skipped",
                len(remaining_chinese_paths),
            )
            for path in remaining_chinese_paths[:100]:
                log.error("  Untranslated path: %s", path)
            return 2
        if remaining_chinese_paths:
            log.warning(
                "Debug override enabled; continuing with %d untranslated JSON string(s)",
                len(remaining_chinese_paths),
            )
            for path in remaining_chinese_paths[:100]:
                log.warning("  Untranslated path: %s", path)

        log.info("Stage 4/5: render original and translated JSON to LaTeX")
        original_latex_path = output_dir / f"{input_path.stem}.tex"
        original_renderer = LatexRenderer(RenderOptions(figures="image"))
        original_latex = original_renderer.render(exam_data)
        if not original_latex.strip():
            raise ExamFormatError("LaTeX converter produced empty original output")
        write_text(original_latex_path, original_latex)
        for note in original_renderer.notes:
            log.warning("Original LaTeX converter: %s", note)

        translated_renderer = LatexRenderer(RenderOptions(figures="image"))
        translated_latex = translated_renderer.render(translated_data)
        if not translated_latex.strip():
            raise ExamFormatError("LaTeX converter produced empty translated output")
        write_text(latex_path, translated_latex)
        for note in translated_renderer.notes:
            log.warning("Translated LaTeX converter: %s", note)

        log.info("Stage 5/5: compile original and translated LaTeX to PDF")
        check_tex_prerequisites()
        compile_latex_to_pdf(original_latex_path, output_dir, input_path.parent)
        compile_latex_to_pdf(latex_path, output_dir, input_path.parent)

        log.info("Wrote corrected Markdown: %s", corrected_path)
        log.info("Wrote structured JSON: %s", json_path)
        log.info("Wrote translated JSON: %s", translated_json_path)
        log.info("Wrote original LaTeX: %s", original_latex_path)
        log.info("Wrote translated LaTeX: %s", latex_path)

        semantic_findings = 0
        semantic_uncertainties = 0
        if semantic_review:
            pdf_path = source_pdf or input_path.with_suffix(".pdf")
            chinese_review_path = output_dir / f"{input_path.stem}_chinese_review.json"
            translation_review_path = output_dir / f"{input_path.stem}_translation_review.json"

            log.info("Stage 6/6: semantic review of generated JSON files")
            # Read the persisted artifacts so the reviewers inspect exactly the
            # JSON files handed off by the pipeline, not transient objects.
            persisted_chinese = json.loads(json_path.read_text(encoding="utf-8"))
            persisted_english = json.loads(translated_json_path.read_text(encoding="utf-8"))

            try:
                chinese_report = ChineseSourceReviewer(model=review_model).review(
                    persisted_chinese, pdf_path
                )
            except TranslationError as exc:
                # A provider credential/project failure affects both Gemini
                # checkers. Save explicit reports and avoid a redundant call.
                log.error("Chinese semantic review unavailable: %s", exc)
                chinese_report = review_unavailable_report("chinese_source", str(exc))
                chinese_report.uncertainties.extend(
                    schema_warning_uncertainties(persisted_chinese)
                )
                translation_report = review_unavailable_report(
                    "translation_comparison",
                    f"Skipped because the shared Gemini review service failed: {exc}",
                )
            else:
                try:
                    translation_report = TranslationReviewer(model=review_model).review(
                        persisted_chinese, persisted_english
                    )
                except TranslationError as exc:
                    log.error("Translation semantic review unavailable: %s", exc)
                    translation_report = review_unavailable_report(
                        "translation_comparison", str(exc)
                    )
            write_json(chinese_review_path, chinese_report.to_dict())
            write_json(translation_review_path, translation_report.to_dict())
            semantic_findings = len(chinese_report.findings) + len(translation_report.findings)
            semantic_uncertainties = (
                len(chinese_report.uncertainties) + len(translation_report.uncertainties)
            )
            log.info("Wrote Chinese semantic review: %s", chinese_review_path)
            log.info("Wrote translation semantic review: %s", translation_review_path)
            for review in (chinese_report, translation_report):
                for finding in review.findings:
                    log.warning("%s review %s at %s: %s", review.checker,
                                finding.severity, finding.path, finding.explanation)
                for uncertainty in review.uncertainties:
                    log.info("%s review uncertain at %s: %s", review.checker,
                             uncertainty.path, uncertainty.question)

        if translator.failed or validation_issues or semantic_findings or semantic_uncertainties:
            log.error(
                "Review needs attention: %d translation failure(s), %d translation validation issue(s), "
                "%d semantic finding(s), %d uncertainty note(s)",
                len(translator.failed),
                len(validation_issues),
                semantic_findings,
                semantic_uncertainties,
            )
            for path in translator.failed_paths[:50]:
                log.error("  Untranslated field: %s", path)
            for issue in validation_issues[:20]:
                log.error("  %s", issue)
            return 2
        log.info("Pipeline completed successfully")
        return 0
    except (OSError, ValueError, RuntimeError, ExamFormatError, TranslationError) as exc:
        log.error("Pipeline stopped: %s", exc)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the configured pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("data/input/exam.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output/pipeline"))
    parser.add_argument(
        "--allow-untranslated", action="store_true",
        help="Debug only: continue to LaTeX/PDF stages when translated JSON still contains Chinese",
    )
    parser.add_argument(
        "--source-pdf", type=Path,
        help="Source PDF for Chinese semantic review; defaults to <input stem>.pdf beside Markdown",
    )
    parser.add_argument(
        "--review-model", help="Gemini model for both semantic reviews (defaults to GEMINI_MODEL or gemini-3.8-flash)",
    )
    review_options = parser.add_mutually_exclusive_group()
    review_options.add_argument(
        "--semantic-review", dest="semantic_review", action="store_true",
        help="Run the Gemini semantic reviews after conversion (disabled by default)",
    )
    review_options.add_argument(
        "--skip-semantic-review", dest="semantic_review", action="store_false",
        help="Skip semantic reviews (default)",
    )
    parser.set_defaults(semantic_review=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return run_pipeline(
        args.input,
        args.output_dir,
        semantic_review=args.semantic_review,
        source_pdf=args.source_pdf,
        review_model=args.review_model,
        allow_untranslated=args.allow_untranslated,
    )


if __name__ == "__main__":
    sys.exit(main())
