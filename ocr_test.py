"""Compatibility entry point for importing externally generated OCR output."""

from paper.ocr_baseline import main


if __name__ == "__main__":
    raise SystemExit(main())
