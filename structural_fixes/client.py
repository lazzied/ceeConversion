"""Groq-compatible OpenAI client used by the structural correction stage."""

from __future__ import annotations

import os
import time
from typing import Any

import openai
from openai import OpenAI

from structural_fixes.utils import strip_fences

TPM_LIMIT = 8_000
SAFETY_MARGIN = 200
_recent_calls: list[tuple[float, int]] = []


def estimate_tokens(text: str) -> int:
    """Estimate tokens conservatively for mixed Chinese and LaTeX text."""
    return len(text) // 2 + 1


def wait_for_budget(token_count: int) -> None:
    """Wait until ``token_count`` fits the rolling 60-second request budget."""
    while True:
        now = time.monotonic()
        _recent_calls[:] = [(stamp, count) for stamp, count in _recent_calls if now - stamp < 60]
        if sum(count for _, count in _recent_calls) + token_count <= TPM_LIMIT - SAFETY_MARGIN:
            _recent_calls.append((now, token_count))
            return
        oldest_call = _recent_calls[0][0]
        time.sleep(max(1.0, 61.0 - (now - oldest_call)))


def call_llm(text: str, model: str = "openai/gpt-oss-20b", *, client: Any = None) -> str:
    """Submit one correction request, retrying rate limits three times."""
    input_tokens = estimate_tokens(text)
    max_output_tokens = input_tokens + 1_000
    request_tokens = input_tokens + max_output_tokens
    if request_tokens > TPM_LIMIT - SAFETY_MARGIN:
        raise ValueError(
            f"Section needs about {request_tokens} tokens, above the configured "
            f"{TPM_LIMIT} token-per-minute limit. Split it into smaller sections."
        )
    wait_for_budget(request_tokens)

    api_key = os.environ.get("GROQ_API_KEY")
    if client is None:
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for structural corrections")
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    for attempt in range(3):
        try:
            response = client.responses.create(
                input=text,
                model=model,
                temperature=0,
                max_output_tokens=max_output_tokens,
                reasoning={"effort": "low"},
            )
            if response.status == "incomplete":
                raise RuntimeError("Structural correction response was truncated")
            return response.output_text.strip()
        except openai.RateLimitError:
            if attempt == 2:
                raise RuntimeError("Rate limit persisted after three attempts")
            time.sleep(20)
        except openai.OpenAIError as exc:
            raise RuntimeError(f"Structural correction API request failed: {exc}") from exc
    raise RuntimeError("Structural correction request failed")


def process(prompt: str, text: str, *, client: Any = None) -> str:
    """Correct a Markdown section and remove accidental response fences."""
    return strip_fences(call_llm(prompt + text, client=client))
