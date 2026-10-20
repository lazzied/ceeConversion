"""CLI:  python -m exam_parser paper.tex -o paper.json [--keep-raw] [--summary]"""
import argparse
from pathlib import Path

from .parser import ExamParser


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse an exam .tex file into JSON.")
    ap.add_argument("tex", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="default: <tex>.json")
    ap.add_argument("--keep-raw", action="store_true", help="store original LaTeX per question")
    ap.add_argument("--exclude-none", action="store_true", help="omit null fields")
    ap.add_argument("--summary", action="store_true", help="print the structure tree")
    args = ap.parse_args()

    exam = ExamParser(keep_raw=args.keep_raw).parse_file(args.tex)
    out = exam.save(args.output or args.tex.with_suffix(".json"), exclude_none=args.exclude_none)
    if args.summary:
        print(exam.summary())
    for w in exam.warnings:
        print("WARNING:", w)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
