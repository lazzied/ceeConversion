"""CLI:  python -m exam_latex exam.json -o exam.tex [--figures image] [--wrap 88]"""
import argparse
from pathlib import Path

from latex_converter.latex_renderer import ExamFormatError, LatexRenderer, RenderOptions



def main() -> None:
    ap = argparse.ArgumentParser(description="Render exam JSON to a LaTeX document.")
    ap.add_argument("json", type=Path, help="exam JSON (see SCHEMA.md)")
    ap.add_argument("-o", "--output", type=Path, help="default: <json>.tex")
    ap.add_argument("--figures", choices=["placeholder", "image"], default="image",
                    help="file-aware \\includegraphics (placeholder only if missing), or labelled placeholders")
    ap.add_argument("--wrap", type=int, default=88, help="soft line width (0 = no wrapping)")
    ap.add_argument("--keep-parens", action="store_true",
                    help="keep （\\ \\ ） literally instead of \\choiceparen{}")
    args = ap.parse_args()

    renderer = LatexRenderer(RenderOptions(figures=args.figures, wrap=args.wrap,
                                           normalize_parens=not args.keep_parens))
    out = args.output or args.json.with_suffix(".tex")
    try:
        renderer.render_file(args.json, out)
    except ExamFormatError as exc:
        raise SystemExit(f"error: {exc}")
    for note in renderer.notes:
        print("NOTE:", note)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
