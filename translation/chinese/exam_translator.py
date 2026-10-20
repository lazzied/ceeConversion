from __future__ import annotations
import copy
import itertools
import json
import logging
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from translation.chinese.translation_client import TranslationClient
from translation.chinese.translation_units import Unit
from translation.chinese.translation_validation import (
    chinese_spans,
    has_chinese_in_math,
    is_preserved_path,
    MATH_SPAN,
    validation_errors,
)

# Load .env from the script's directory, regardless of the working directory.
load_dotenv()
log = logging.getLogger("exam_translator")
# Keep section and question items together where possible. Context is also
# attached explicitly to each group so batching cannot separate it from its text.
GROUP_START = re.compile(
    r"^\$\.[^.\[]+(\[\d+\])?$|^\$\.sections\[\d+\]$|\.questions\[\d+\]$"
)
SECTION_PATH = re.compile(r"^\$\.sections\[\d+\]$")
QUESTION_PATH = re.compile(r"\.questions\[\d+\]$")
MATH_PLACEHOLDER_PREFIX = "⟦MATH_"

class ExamTranslator:
    """Translate Chinese strings in an exam JSON structure."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        client: OpenAI | None = None,
        max_chars: int = 6000,
        max_items: int = 40,
        attempts: int = 3,
        timeout: float = 120.0,
        backoff: float = 2.0,
    ) -> None:
        if max_chars <= 0 or max_items <= 0:
            raise ValueError("max_chars and max_items must be positive")
        if attempts < 0 or backoff < 0:
            raise ValueError("attempts and backoff cannot be negative")

        self._provided_client = client
        self._client_config = {
            "api_key": api_key,
            "model": model,
            "timeout": timeout,
        }
        self._translation_client: TranslationClient | None = None

        self.max_chars = max_chars
        self.max_items = max_items
        self.attempts = attempts
        self.backoff = backoff

        self.failed: list[str] = []
        self.failed_paths: list[str] = []
        self.protected_chinese: list[str] = []

    @property
    def client(self) -> TranslationClient:
        """Create the API client only when a translation request is needed."""
        if self._translation_client is None:
            self._translation_client = TranslationClient(
                **self._client_config,
                client=self._provided_client,
            )
        return self._translation_client


    def translate(self, data: Any) -> Any:
        """Return a translated deep copy without modifying the original."""
        self.failed = []
        self.failed_paths = []
        self.protected_chinese = []

        holder = [copy.deepcopy(data)]
        units = self.collect_translation_units(holder)

        log.info("Found %d Chinese-containing paragraph(s) for translation", len(units))

        context = self.build_exam_context(data)

        done = self.translate_units(units, context)
        if self.protected_chinese:
            log.warning(
                "Preserved Chinese inside protected math in %d field(s): %s",
                len(self.protected_chinese),
                ", ".join(self.protected_chinese[:10]),
            )

        # Translate one paragraph per unit, then splice completed paragraphs
        # back while leaving the original blank-line separators untouched.
        field_units: dict[tuple[int, str | int], list[Unit]] = {}
        for unit in units:
            field_units.setdefault((id(unit.parent), unit.key), []).append(unit)
        for grouped_units in field_units.values():
            first_unit = grouped_units[0]
            restored = first_unit.parent[first_unit.key]
            replacements = [
                (unit.spans[0][0], unit.spans[0][1], done[unit.id][0])
                for unit in grouped_units if unit.id in done
            ]
            for start, end, translated_paragraph in sorted(replacements, reverse=True):
                restored = restored[:start] + translated_paragraph + restored[end:]
            first_unit.parent[first_unit.key] = restored

        return holder[0]

    # -----------------------------------------------------------------------
    # Collect strings
    # -----------------------------------------------------------------------

    def collect_translation_units(self, root: list) -> list[Unit]:
        """Collect Chinese strings in document order with contextual grouping."""
        units: list[Unit] = []

        def visit(
            parent: dict | list,
            key: str | int,
            path: str,
            group: str,
            context_parts: tuple[str, ...] = (),
        ) -> None:
            value = parent[key]

            if GROUP_START.search(path):
                group = path

            if isinstance(value, str):
                spans = [] if is_preserved_path(path) else chinese_spans(value)
                if not is_preserved_path(path) and has_chinese_in_math(value):
                    self.protected_chinese.append(path)
                if spans:
                    segment_context = "\n".join(context_parts)[:5000]
                    cursor = 0
                    for part in re.split(r"(\n[ \t]*\n+)", value):
                        start, end = cursor, cursor + len(part)
                        cursor = end
                        if not part or re.fullmatch(r"\n[ \t]*\n+", part):
                            continue
                        if not part.strip() or not chinese_spans(part):
                            continue
                        units.append(
                            Unit(
                                id=len(units),
                                path=path,
                                group=group,
                                text=part,
                                parent=parent,
                                key=key,
                                segments=[part],
                                spans=[(start, end)],
                                context=segment_context,
                                field_text=value,
                            )
                        )

            elif isinstance(value, dict):
                object_context = self.build_object_context(path, value)
                child_context = context_parts + ((object_context,) if object_context else ())
                for child_key in value:
                    visit(
                        value,
                        child_key,
                        f"{path}.{child_key}",
                        group,
                        child_context,
                    )

            elif isinstance(value, list):
                for index in range(len(value)):
                    visit(
                        value,
                        index,
                        f"{path}[{index}]",
                        group,
                        context_parts,
                    )

        visit(root, 0, "$", "$")
        return units

    @staticmethod
    def build_exam_context(data: Any) -> str:
        """Provide exam-wide metadata and instructions once per API batch."""
        if not isinstance(data, dict):
            return ""
        context: dict[str, Any] = {}
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            context["metadata"] = {
                key: value for key, value in metadata.items()
                if key not in {"source_file", "raw"}
            }
        if "instructions" in data:
            context["instructions"] = data["instructions"]
        for key in ("title", "description", "passage", "shared_passage", "passages", "context"):
            if key in data:
                context[key] = data[key]
        return json.dumps(context, ensure_ascii=False)[:2500] if context else ""

    @staticmethod
    def build_object_context(path: str, value: dict[str, Any]) -> str:
        """Serialize the useful parent context for a section/question group."""
        if SECTION_PATH.fullmatch(path):
            keys = ("ordinal", "title", "kind", "instructions", "description",
                    "passage", "shared_passage", "context")
            label = "Section context"
            field_limits = {
                "description": 700,
                "instructions": 250,
                "passage": 500,
                "shared_passage": 500,
                "context": 300,
            }
            total_limit = 1200
        elif QUESTION_PATH.search(path):
            keys = (
                "number", "kind", "stem", "sub_questions", "options",
                "instructions", "passage", "shared_passage", "context",
                "blocks", "footer_blocks",
            )
            label = "Question context"
            field_limits = {
                "stem": 1000,
                "sub_questions": 1000,
                "options": 500,
                "instructions": 350,
                "passage": 600,
                "shared_passage": 600,
                "context": 350,
                "blocks": 650,
                "footer_blocks": 350,
            }
            total_limit = 3500
        else:
            return ""

        parts = [label]
        remaining = total_limit - len(label) - 2
        for key in keys:
            if key not in value or remaining <= len(key) + 4:
                continue
            value_text = json.dumps(value[key], ensure_ascii=False, separators=(",", ":"))
            field_limit = field_limits.get(key, 100)
            available = min(field_limit, remaining - len(key) - 2)
            if available <= 0:
                continue
            parts.append(f"{key}={value_text[:available]}")
            remaining -= len(parts[-1]) + 2
        return "; ".join(parts) if len(parts) > 1 else ""

    # -----------------------------------------------------------------------
    # Batching
    # -----------------------------------------------------------------------

    def split_group(self, group: list[Unit]) -> list[list[Unit]]:
        """Split a group only when it exceeds the batch limits."""
        parts: list[list[Unit]] = []
        current: list[Unit] = []
        size = 0
        context_size = min(len(group[0].context), 5000) if group else 0

        for unit in group:
            exceeds_limit = (
                current
                and (
                    len(current) >= self.max_items
                    or size + len(unit.text) > self.max_chars - context_size
                )
            )

            if exceeds_limit:
                parts.append(current)
                current, size = [], 0

            current.append(unit)
            size += len(unit.text)

        if current:
            parts.append(current)

        return parts

    def create_batches(self, units: list[Unit]) -> list[list[Unit]]:
        """Pack contextual groups into batches without splitting unnecessarily."""
        batches: list[list[Unit]] = []
        current: list[Unit] = []
        size = 0

        for _, grouped_units in itertools.groupby(
            units,
            key=lambda unit: unit.group,
        ):
            for part in self.split_group(list(grouped_units)):
                part_contexts = {unit.group: unit.context for unit in part if unit.context}
                part_text_size = sum(len(unit.text) for unit in part)
                part_size = part_text_size + sum(
                    min(len(value), 5000) for value in part_contexts.values()
                )
                current_contexts = {unit.group: unit.context for unit in current if unit.context}
                current_size = size + sum(
                    min(len(value), 5000) for value in current_contexts.values()
                )

                exceeds_limit = (
                    current
                    and (
                        len(current) + len(part) > self.max_items
                        or current_size + part_size > self.max_chars
                    )
                )

                if exceeds_limit:
                    batches.append(current)
                    current, size = [], 0

                current.extend(part)
                size += part_text_size

        if current:
            batches.append(current)

        return batches

    # -----------------------------------------------------------------------
    # Translation and retries
    # -----------------------------------------------------------------------

    def translate_units(
        self,
        units: list[Unit],
        context: str,
    ) -> dict[int, list[str]]:
        """Translate batches, then retry unsuccessful groups individually."""
        done: dict[int, list[str]] = {}

        for batch in self.create_batches(units):
            # If a mixed batch still has failures, retry its original groups
            # separately. Do only one mixed attempt first; repeating the whole
            # batch several times wastes calls when one question is problematic.
            groups = [
                list(group)
                for _, group in itertools.groupby(
                    batch,
                    key=lambda unit: unit.group,
                )
            ]
            self.translate_pending_units(
                batch,
                done,
                context,
                max_attempts=1 if len(groups) > 1 else self.attempts + 1,
            )

            if len(groups) > 1:
                if not any(unit.id in done for unit in batch):
                    # A wholly invalid response is likely a batch-level format
                    # or transient API issue. Retry the batch before fanning
                    # out into many per-question calls.
                    self.translate_pending_units(
                        batch,
                        done,
                        context,
                        max_attempts=self.attempts,
                        retry_first=True,
                    )
                else:
                    for group in groups:
                        if any(unit.id not in done for unit in group):
                            self.translate_pending_units(
                                group,
                                done,
                                context,
                                max_attempts=self.attempts,
                                retry_first=True,
                            )

        math_rejections = [
            unit for unit in units
            if unit.id not in done and MATH_SPAN.search(unit.text)
        ]
        if math_rejections:
            self.translate_masked_fallback(math_rejections, context, done)

        for unit in units:
            if unit.id not in done:
                log.error("Translation failed at %s: %.100s", unit.path, unit.text)
                self.failed.append(unit.text)
                if unit.path not in self.failed_paths:
                    self.failed_paths.append(unit.path)

        return done

    def translate_masked_fallback(
        self,
        rejected_units: list[Unit],
        context: str,
        done: dict[int, list[str]],
    ) -> None:
        """Last resort: mask formulas, translate each full sentence, restore math."""
        from dataclasses import replace

        masked_units: list[Unit] = []
        restorations: dict[int, list[str]] = {}
        for unit in rejected_units:
            formulas: list[str] = []
            def mask(match: re.Match[str]) -> str:
                formulas.append(match.group())
                return f"{MATH_PLACEHOLDER_PREFIX}{len(formulas) - 1}⟧"
            masked = MATH_SPAN.sub(mask, unit.text)
            if not formulas:
                continue
            fallback_unit = replace(
                unit,
                text=masked,
                segments=[masked],
                validation_text=masked,
                rejection_reason=(unit.rejection_reason + "; retry with formula placeholders, then preserve all placeholders exactly").strip("; "),
            )
            masked_units.append(fallback_unit)
            restorations[unit.id] = formulas
        if not masked_units:
            return

        log.warning("Retrying %d rejected math-bearing field(s) with protected placeholders", len(masked_units))
        try:
            fallback = self.client.translate_batch(masked_units, {}, context, retry=True)
        except Exception as exc:
            log.warning("Math-placeholder fallback failed: %s", exc)
            return
        for unit in masked_units:
            translated = fallback.get(unit.id)
            if not translated:
                log.warning(
                    "Math-placeholder fallback returned no valid text for %s; raw masked response: %s",
                    unit.path,
                    getattr(self.client, "last_raw_response", "")[:2000],
                )
                continue
            restored = translated[0]
            formulas = restorations[unit.id]
            placeholders_valid = True
            for index, formula in enumerate(formulas):
                placeholder = f"{MATH_PLACEHOLDER_PREFIX}{index}⟧"
                if restored.count(placeholder) != 1:
                    placeholders_valid = False
                restored = restored.replace(placeholder, formula)
            problems = validation_errors(unit.text, restored)
            if not placeholders_valid:
                problems.append("math placeholders were missing, duplicated, or altered")
            if problems:
                log.warning(
                    "Math-placeholder fallback rejected at %s (%s); raw masked response: %s",
                    unit.path,
                    "; ".join(problems),
                    getattr(self.client, "last_raw_response", "")[:2000],
                )
                continue
            done[unit.id] = [restored]

    def translate_pending_units(
        self,
        units: list[Unit],
        done: dict[int, list[str]],
        context: str,
        *,
        max_attempts: int | None = None,
        retry_first: bool = False,
    ) -> None:
        """Retry invalid or missing translations up to the configured limit."""
        attempt_limit = max_attempts if max_attempts is not None else self.attempts + 1
        repeated_rejections: set[int] = set()
        for attempt in range(attempt_limit):
            pending = [unit for unit in units if unit.id not in done and unit.id not in repeated_rejections]
            if not pending:
                return

            if attempt:
                delay = self.backoff * (2 ** (attempt - 1))
                log.info(
                    "Retry %d/%d for %d pending item(s); waiting %.1fs",
                    attempt,
                    attempt_limit - 1,
                    len(pending),
                    delay,
                )
                time.sleep(delay)

            previous_rejections = {unit.id: unit.rejection_reason for unit in pending}
            result = self.request_translation(
                pending,
                done,
                context,
                retry=retry_first or attempt > 0,
            )
            done.update(result)
            for unit in pending:
                previous_reason = previous_rejections[unit.id]
                if previous_reason and unit.id not in done and unit.rejection_reason == previous_reason:
                    repeated_rejections.add(unit.id)
                    log.warning(
                        "Same rejection repeated for %s; escalating directly to math-placeholder fallback: %s",
                        unit.path,
                        unit.rejection_reason,
                    )

    
    def request_translation(
        self,
        units: list[Unit],
        done: dict[int, list[str]],
        context: str,
        *,
        retry: bool = False,
    ) -> dict[int, list[str]]:
        """Delegate one translation attempt to the API client."""
        return self.client.translate_batch(
            units,
            done,
            context,
            retry=retry,
    )



