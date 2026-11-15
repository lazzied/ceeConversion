"""Command-line interface for corpus scanning and batch splitting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .discovery import scan_corpus
from .models import SplitError
from .output import split_document
from .parser import analyze_document


LOGGER = logging.getLogger("latex_exam_splitter")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Separate combined exam questions and solutions into new LaTeX files."
    )
    parser.add_argument("--root", type=Path, required=True, help="Corpus root containing styles.tex")
    parser.add_argument("--output", type=Path, required=True, help="Separate output directory")
    parser.add_argument("--source", type=Path, help="Split one source instead of scanning recursively")
    parser.add_argument("--write", action="store_true", help="Write outputs; otherwise only report matches")
    parser.add_argument("--no-compile", action="store_true", help="Skip XeLaTeX validation after writing")
    parser.add_argument("--engine", default="xelatex", help="LaTeX engine to use (default: xelatex)")
    parser.add_argument("--overwrite-output", action="store_true", help="Replace existing generated files")
    parser.add_argument("--verbose", action="store_true", help="List every matched document")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Scan a corpus, optionally write question/solution pairs, and return a status."""

    args = _build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.source:
        files = [args.source.resolve()]
        try:
            plan = analyze_document(args.source, corpus_root=args.root)
        except (OSError, UnicodeError, SplitError) as error:
            LOGGER.error("Cannot analyze %s: %s", args.source, error)
            return 2
        combined = [(args.source.resolve(), plan)] if plan is not None else []
    else:
        files, combined = scan_corpus(args.root, exclude_paths=(args.output,))

    print(f"Scanned {len(files)} .tex file(s); found {len(combined)} combined document(s).")
    displayed = combined if args.verbose else combined[:10]
    for source, plan in displayed:
        print(
            f"{source}: {plan.question_count} questions, "
            f"{plan.paired_question_count} with answer/solution material"
        )
        if args.write:
            try:
                relative_parent = source.parent.relative_to(args.root.resolve())
            except ValueError:
                relative_parent = Path()
            destination = args.output / relative_parent
            try:
                split_document(
                    source,
                    destination,
                    corpus_root=args.root,
                    compile_outputs=not args.no_compile,
                    engine=args.engine,
                    overwrite_output=args.overwrite_output,
                )
            except (OSError, UnicodeError, SplitError) as error:
                LOGGER.error("Failed to split %s: %s", source, error)
                return 1

    if len(displayed) < len(combined):
        print(f"... {len(combined) - len(displayed)} more; pass --verbose to list all matches.")
    if not args.write:
        print("No files written. Pass --write to create output pairs.")
    return 0
