"""Thread context: combine caption + prior replies + target comment for modeling."""

from __future__ import annotations

import json
import re
from typing import Iterable


def format_contextual_input(
    comment: str,
    parent_caption: str | None = None,
    prior_replies: str | list[str] | None = None,
    max_prior: int = 5,
    max_chars_per_prior: int = 500,
) -> str:
    """
    Build a single string for the classifier. Markers help the model separate roles.
    `prior_replies` can be JSON array string or list of strings (oldest → newest).
    """
    parts: list[str] = []

    if parent_caption and str(parent_caption).strip():
        cap = _truncate(str(parent_caption).strip(), 2000)
        parts.append(f"[caption] {cap}")

    replies = _parse_prior_replies(prior_replies)
    if replies:
        for r in replies[-max_prior:]:
            r = _truncate(str(r).strip(), max_chars_per_prior)
            if r:
                parts.append(f"[thread] {r}")

    parts.append(f"[comment] {str(comment).strip()}")
    return "\n".join(parts)


def _truncate(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[: n - 3] + "..."


def _parse_prior_replies(prior_replies: str | list[str] | None) -> list[str]:
    if prior_replies is None:
        return []
    if isinstance(prior_replies, list):
        return [x for x in prior_replies if x]
    s = str(prior_replies).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines if lines else [s]


def detect_language_hint(text: str) -> str | None:
    """
    Optional language tag using `langdetect` if installed.
    Returns ISO 639-1 code or None. Does not translate; use multilingual models for mixed audiences.
    """
    if not text or not str(text).strip():
        return None
    try:
        from langdetect import detect

        return detect(text[:5000])
    except Exception:
        return None


def summarize_multilingual_note(langs: Iterable[str | None]) -> str:
    """Human-readable note for UI when multiple languages may appear."""
    codes = sorted({c for c in langs if c})
    if not codes:
        return "Language not detected (install `langdetect` for hints)."
    if len(codes) == 1:
        return f"Detected language hint: {codes[0]} (heuristic; use a multilingual model for robust coverage)."
    return f"Multiple language hints: {', '.join(codes)}. Consider mBERT / XLM-R fine-tuning."
