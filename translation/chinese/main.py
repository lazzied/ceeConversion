from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

from translation.chinese.exam_translator import ExamTranslator
from translation.chinese.translation_errors import TranslationError
from translation.chinese.translation_validation import validate_final_json

log = logging.getLogger(__name__)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("input", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path; defaults to <input>_en.json",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_path = args.output or args.input.with_name(
        args.input.stem + "_en.json"
    )

    try:
        original = json.loads(args.input.read_text(encoding="utf-8"))
    except OSError as exc:
        log.error("Cannot read input file %s: %s", args.input, exc)
        return 1
    except json.JSONDecodeError as exc:
        log.error("Input is not valid JSON: %s", exc)
        return 1

    try:
        translator = ExamTranslator()
        result = translator.translate(original)
    except TranslationError as exc:
        log.error("%s", exc)
        return 1

    # Final validation happens before writing the output file.
    validation_issues = validate_final_json(original, result)

    if validation_issues:
        log.error(
            "Final validation found %d issue(s):",
            len(validation_issues),
        )

        for error in validation_issues[:50]:
            log.error("  %s", error)

        if len(validation_issues) > 50:
            log.error(
                "  ... and %d more issue(s)",
                len(validation_issues) - 50,
            )
    else:
        log.info("Final validation passed.")

    # Save successful translations even if some strings failed, so that
    # valid work is not lost. Exit code 2 signals that manual review is needed.
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(output_path)

    except OSError as exc:
        log.error("Cannot write output file %s: %s", output_path, exc)
        return 1

    log.info("Wrote %s", output_path)

    if translator.failed:
        log.error(
            "%d source string(s) could not be translated",
            len(translator.failed),
        )
        for path in translator.failed_paths[:50]:
            log.error("  Untranslated field: %s", path)

    if validation_issues:
        log.error("Output needs review; it did not pass final validation.")
        return 2

    if translator.failed:
        # Defensive check: failed items should also be visible in the final
        # validation report because their source Chinese remains in the JSON.
        log.error("Some translations remain incomplete.")
        return 2

    log.info("Translation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
