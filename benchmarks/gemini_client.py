"""Thin Gemini client used by all benchmark domain agents.

Tracks token counts and latency. Falls back to pre-written responses
when GEMINI_API_KEY is not set (useful for offline benchmarking).
"""

from __future__ import annotations

import os
import time
from typing import Optional

_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("PRISMRAG_GEMINI_KEY")
_MODEL = "gemini-2.0-flash"

# Lazy client — only initialised if key is available
_client = None


def _get_client():
    global _client
    if _client is None and _API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=_API_KEY)
        _client = genai.GenerativeModel(_MODEL)
    return _client


def call(prompt: str, fallback: str) -> tuple[str, int, int, float]:
    """Call Gemini and return (text, prompt_tokens, output_tokens, elapsed_ms).

    Falls back to the provided pre-written response when no API key is set.
    Token counts for fallback are estimated from whitespace-split word count.
    """
    client = _get_client()
    t0 = time.perf_counter()

    if client is None:
        # Offline mode — simulate realistic latency
        time.sleep(0.05)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        prompt_tokens = max(1, len(prompt.split()))
        output_tokens = max(1, len(fallback.split()))
        return fallback, prompt_tokens, output_tokens, elapsed_ms

    response = client.generate_content(prompt)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    text = response.text.strip()

    # Extract token counts from usage_metadata when available
    meta = getattr(response, "usage_metadata", None)
    if meta:
        prompt_tokens = getattr(meta, "prompt_token_count", len(prompt.split()))
        output_tokens = getattr(meta, "candidates_token_count", len(text.split()))
    else:
        prompt_tokens = len(prompt.split())
        output_tokens = len(text.split())

    return text, prompt_tokens, output_tokens, elapsed_ms
