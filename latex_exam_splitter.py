"""Compatibility entry point for the :mod:`latex_exam_splitter` package.

Use ``python -m latex_exam_splitter`` for the package CLI. This small shim
keeps the earlier ``python latex_exam_splitter.py`` command working.
"""

from latex_exam_splitter import *  # noqa: F403
from latex_exam_splitter.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
