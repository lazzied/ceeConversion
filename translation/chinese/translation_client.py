
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import (
    APIError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
)

from translation.chinese.translation_errors import TranslationError
from translation.chinese.translation_units import Unit
from translation.chinese.translation_validation import added_math_atoms, normalize_translated_text, validation_errors
from translation.chinese.prompt import SYSTEM_PROMPT, RETRY_PROMPT, render_glossary

log = logging.getLogger(__name__)

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def decode_json_object(raw: str) -> dict[str, Any]:
    """Decode one JSON object, rejecting duplicate keys and optional prose."""
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    try:
        value = decoder.decode(raw)
        if isinstance(value, dict):
            return value
    except (json.JSONDecodeError, ValueError):
        pass

    # Some providers prepend commentary despite the JSON-only instruction.
    # Find the first decodable object instead of slicing at the last brace,
    # which can break when translated strings contain literal braces.
    for start, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw, start)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    raise TranslationError("Model response does not contain a valid JSON object")


class TranslationClient:
    """Handle OpenRouter requests and validate their responses."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        client: OpenAI | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.environ.get("OPENROUTER_MODEL", "")

        if not self.model:
            raise TranslationError("OPENROUTER_MODEL is not set")

        if client is None:
            key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
            if not key:
                raise TranslationError("OPENROUTER_API_KEY is not set")

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
                timeout=timeout,
                # The translator owns content retries; keep transport retries
                # low so one invalid batch cannot multiply into many requests.
                max_retries=1,
            )

        self.client = client
        self.last_raw_response = ""

    def translate_batch(
        self,
        units: list[Unit],
        done: dict[int, list[str]],
        context: str,
        *,
        retry: bool = False,
    ) -> dict[int, list[str]]:
        """Request translations for pending units and validate the response."""
        pending = [unit for unit in units if unit.id not in done]

        if not pending:
            return {}

        contexts = {
            unit.group: unit.context
            for unit in units
            if unit.context
        }
        payload_items = []
        for unit in units:
            segments = unit.segments or [unit.text]
            if unit.id in done:
                payload_items.append({
                    "id": unit.id,
                    "path": unit.path,
                    "source_segments": segments,
                    "done": done[unit.id],
                })
            else:
                payload_items.append({
                    "id": unit.id,
                    "path": unit.path,
                    "segments": segments,
                    **({"validation_feedback": unit.rejection_reason} if unit.rejection_reason else {}),
                })
        payload = {"group_context": contexts, "items": payload_items}

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    context=context or "(none)",
                    glossary=render_glossary(),
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

        if retry:
            messages.append(
                {"role": "user", "content": RETRY_PROMPT}
            )

        source_character_count = sum(
            len(segment)
            for unit in pending
            for segment in (unit.segments or [unit.text])
        )
        output_token_limit = min(
            8_192,
            max(256, source_character_count * 2 + len(pending) * 16),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=output_token_limit,
                messages=messages,
            )
            response_choices = getattr(response, "choices", None)
            finish_reason = (
                getattr(response_choices[0], "finish_reason", None)
                if response_choices else None
            )
            if finish_reason == "length":
                raise TranslationError("Model response was truncated by the output token limit")
            return self.parse_response(response, pending)

        except (
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
        ) as exc:
            raise TranslationError(
                f"Fatal API error; check the API key and model: {exc}"
            ) from exc

        except (APIError, TranslationError) as exc:
            log.warning(
                "API call failed for %d item(s): %s",
                len(pending),
                exc,
            )
            return {}

    def parse_response(
        self,
        response: Any,
        pending: list[Unit],
    ) -> dict[int, list[str]]:
        """Parse segment arrays and accept only complete, valid translations."""
        choices = getattr(response, "choices", None)

        if not choices:
            raise TranslationError("Model returned an empty response")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise TranslationError("Model returned an empty response")
        raw = FENCE.sub("", content.strip())
        self.last_raw_response = raw

        obj = decode_json_object(raw)

        expected_ids = {str(unit.id) for unit in pending}
        returned_ids = set(obj)

        missing_ids = expected_ids - returned_ids
        extra_ids = returned_ids - expected_ids

        if missing_ids:
            log.warning(
                "Model omitted %d pending ID(s): %s",
                len(missing_ids),
                ", ".join(sorted(missing_ids)[:20]),
            )

        if extra_ids:
            log.warning(
                "Model returned %d unexpected ID(s); ignoring them",
                len(extra_ids),
            )

        translations: dict[int, list[str]] = {}

        for unit in pending:
            translated_segments = obj.get(str(unit.id))
            source_segments = unit.segments or [unit.text]
            if isinstance(translated_segments, str) and len(source_segments) == 1:
                translated_segments = [translated_segments]
            if not isinstance(translated_segments, list):
                problems = ["translation must be an array of strings"]
            elif len(translated_segments) != len(source_segments):
                problems = [
                    f"expected {len(source_segments)} translated segment(s), "
                    f"received {len(translated_segments)}"
                ]
            else:
                problems = []
                for index, (source_segment, translated_segment) in enumerate(
                    zip(source_segments, translated_segments), start=1
                ):
                    normalized_translation = normalize_translated_text(translated_segment) if isinstance(translated_segment, str) else translated_segment
                    validation_source = unit.validation_text or source_segment
                    segment_problems = validation_errors(validation_source, normalized_translation)
                    additions = added_math_atoms(validation_source, normalized_translation)
                    if additions:
                        log.info(
                            "Translation %s added math atoms that were not required by the source: %s",
                            unit.path,
                            additions,
                        )
                    if isinstance(normalized_translation, str):
                        translated_segments[index - 1] = normalized_translation
                    problems.extend(f"segment {index}: {problem}" for problem in segment_problems)

            if problems:
                unit.rejection_reason = "; ".join(problems)
                log.warning(
                    "Rejected %s: %s",
                    unit.path,
                    unit.rejection_reason,
                )
                continue

            translations[unit.id] = [segment.strip() for segment in translated_segments]

        return translations
