"""
Preprocessing utilities for the strict 2-stage pipeline.

Key requirement: EMOJIS MUST BE PRESERVED end-to-end.
We therefore do NOT strip non-ASCII characters and we do NOT use regex like `[^a-z\\s]`.
"""

from __future__ import annotations

import re
from typing import List


_URL_RE = re.compile(r"http\S+|www\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+", flags=re.UNICODE)
_HASHTAG_RE = re.compile(r"#(\w+)", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_sexual_threat_typos(text: str) -> str:
    """
    Fix a small set of common misspellings/obfuscations so sexual-threat language
    is not missed by token patterns.

    Must NOT remove emojis.
    """

    s = str(text).lower()

    # Common leetspeak / obfuscations
    s = s.replace("r@pe", "rape")
    s = s.replace("r*pe", "rape")
    s = s.replace("k!ll", "kill")
    s = s.replace("k1ll", "kill")

    # "rapped" (common typo) -> "raped" in threat-like phrases
    patterns = [
        (r"\b(deserve to be|going to be|will be|gonna be)\s+rapped\b", r"\1 raped"),
        (r"\b(you|u)\s+deserve to be\s+rapped\b", r"\1 deserve to be raped"),
        (r"\b(be|get)\s+rap(ped|ping)\b", r"\1 raped"),  # be/get rap -> be/get rape-ish
        (r"\b(be|get)\s+rapped by\b", r"\1 raped by"),
        (r"\braping\b", "raping"),
        (r"\brapist\b", "rapist"),
    ]

    for pat, repl in patterns:
        s = re.sub(pat, repl, s)

    return s


def normalize_indian_threats(text: str) -> str:
    """Example mapping for some slang/abusive terms (extend as needed)."""

    s = str(text).lower()
    replacements = {
        "maar dunga": "kill you",
        "randi": "slut",
        "harami": "bastard",
        "kutte": "dog",
        "madarchod": "abusive",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def preprocess_text(text: str) -> str:
    """
    Basic normalization while preserving emojis.

    Steps:
    1) targeted typo normalization
    2) lowercase
    3) remove URLs and @mentions
    4) keep hashtag text but remove '#'
    5) normalize whitespace
    """

    s = normalize_indian_threats(normalize_sexual_threat_typos(text))
    s = s.lower()
    s = _URL_RE.sub("", s)
    s = _MENTION_RE.sub("", s)
    s = _HASHTAG_RE.sub(r"\1", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def identity_preprocess(x: str) -> str:
    """Pickle-safe identity preprocessor for TfidfVectorizer."""
    return x


def tokenize_keep_emojis(text: str) -> List[str]:
    """
    Tokenizer for TF-IDF that preserves emojis.

    - alphanumeric sequences become tokens
    - non-whitespace, non-ascii characters (ord>127) become single-char tokens
    """

    s = str(text)
    tokens: list[str] = []
    buf: list[str] = []

    def flush_buf() -> None:
        nonlocal buf
        if buf:
            tokens.append("".join(buf))
            buf = []

    for ch in s:
        if ch.isalnum():
            buf.append(ch.lower())
            continue

        flush_buf()
        if ch.isspace():
            continue

        if ord(ch) > 127:
            tokens.append(ch)

    flush_buf()
    return tokens


def prepare_for_model(text: str, parent_caption: str | None = None, prior_replies=None) -> tuple[str, int]:
    """
    Backwards-compatible helper used by older notebook code.
    For the new pipeline, only the cleaned text is used.
    """

    if parent_caption is not None or prior_replies is not None:
        from thread_context import format_contextual_input

        text = format_contextual_input(text, parent_caption=parent_caption, prior_replies=prior_replies)

    return preprocess_text(text), 0

