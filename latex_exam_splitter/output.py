"""Build standalone LaTeX outputs and validate them with a TeX compiler."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .models import DEFAULT_COMPILE_TIMEOUT_SECONDS, SplitError, SplitOutput
from .parser import analyze_document


def _document_wrapper(
    body: str,
    style_file: Path | None,
    show_solutions: bool,
    *,
    preamble: str | None = None,
    document_start: str | None = None,
    document_end: str | None = None,
    trailer: str = "",
    graphics_directory: Path | None = None,
) -> str:
    """Wrap a corpus fragment or preserve an existing standalone preamble."""

    answer_setting = r"\showanswertrue" if show_solutions else r"\showanswerfalse"
    if preamble is not None and document_start is not None and document_end is not None:
        return "\n".join(
            [preamble.rstrip(), answer_setting, document_start, body, document_end + trailer]
        )
    if style_file is None:
        raise SplitError("A LaTeX fragment requires a shared styles.tex file")

    graphics_path = graphics_directory or style_file.parent
    graphics_root = graphics_path.as_posix().rstrip("/") + "/"
    return "\n".join(
        [
            r"\documentclass[12pt, oneside, UTF8]{ctexbook}",
            r"\usepackage{graphicx}",
            rf"\input{{\detokenize{{{style_file.as_posix()}}}}}",
            rf"\graphicspath{{{{\detokenize{{{graphics_root}}}}}}}",
            answer_setting,
            r"\tallpagefalse",
            r"\begin{document}",
            r"\mainmatter",
            body,
            r"\end{document}",
            "",
        ]
    )


def _compile_latex(tex_file: Path, engine: str, working_directory: Path) -> None:
    """Compile a generated file and include useful compiler diagnostics on failure."""

    executable = shutil.which(engine)
    if not executable:
        raise SplitError(f"LaTeX compiler '{engine}' is not installed or not on PATH")
    command = [
        executable,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={tex_file.parent.resolve()}",
        str(tex_file.resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=DEFAULT_COMPILE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise SplitError(f"Timed out compiling {tex_file} with {engine}") from error

    if completed.returncode != 0:
        details = (completed.stdout + "\n" + completed.stderr).strip()
        lines = details.splitlines()
        error_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.startswith("!") or " Error:" in line or line.lstrip().startswith("Error:")
            ),
            None,
        )
        diagnostic = (
            "\n".join(lines[error_index : error_index + 5])
            if error_index is not None
            else details[-2500:]
        )
        raise SplitError(f"Compilation failed for {tex_file}:\n{diagnostic}")


def split_document(
    source: Path,
    output_directory: Path,
    corpus_root: Path | None = None,
    *,
    compile_outputs: bool = True,
    engine: str = "xelatex",
    overwrite_output: bool = False,
) -> SplitOutput | None:
    """Write a separate questions/solutions pair without modifying ``source``.

    Outputs already present are protected by default. When compilation is
    enabled, both files are attempted and a missing PDF or TeX error is reported.
    """

    source = Path(source).resolve()
    plan = analyze_document(source, corpus_root=corpus_root)
    if plan is None:
        return None

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    questions_file = output_directory / f"{source.stem}_questions.tex"
    solutions_file = output_directory / f"{source.stem}_solutions.tex"
    if source in {questions_file, solutions_file}:
        raise SplitError("Refusing to use the source file as an output target")
    if not overwrite_output and (questions_file.exists() or solutions_file.exists()):
        raise SplitError(
            f"Output already exists; choose another output directory or enable overwrite: {output_directory}"
        )

    wrapper_options = {
        "preamble": plan.document_preamble,
        "document_start": plan.document_start,
        "document_end": plan.document_end,
        "trailer": plan.document_trailer,
        "graphics_directory": plan.compilation_directory,
    }
    questions_file.write_text(
        _document_wrapper(
            plan.questions_body,
            plan.style_file,
            show_solutions=False,
            **wrapper_options,
        ),
        encoding="utf-8",
    )
    solutions_file.write_text(
        _document_wrapper(
            plan.solutions_body,
            plan.style_file,
            show_solutions=True,
            **wrapper_options,
        ),
        encoding="utf-8",
    )

    if compile_outputs:
        compile_errors: list[str] = []
        for tex_file in (questions_file, solutions_file):
            try:
                _compile_latex(tex_file, engine, plan.compilation_directory)
                pdf_file = tex_file.with_suffix(".pdf")
                if not pdf_file.is_file():
                    raise SplitError(f"Compiler reported success but did not create {pdf_file}")
            except SplitError as error:
                compile_errors.append(str(error))
        if compile_errors:
            raise SplitError("Generated pair did not fully compile:\n" + "\n".join(compile_errors))

    return SplitOutput(
        questions_file=questions_file,
        solutions_file=solutions_file,
        question_count=plan.question_count,
        paired_question_count=plan.paired_question_count,
        compiled=compile_outputs,
    )
