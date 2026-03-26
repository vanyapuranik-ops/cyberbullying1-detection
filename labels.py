"""
Label mapping for the strict 2-stage pipeline.

Binary stage:
  - threat
  - non_threat

Multiclass stage (only trained on non_threat):
  - harassment
  - sexual
  - neutral
"""

from __future__ import annotations

from typing import Tuple


BINARY_LABELS = ["non_threat", "threat"]
BINARY_LABEL_TO_ID = {"non_threat": 0, "threat": 1}

MULTICLASS_LABELS = ["harassment", "sexual", "neutral"]
MULTICLASS_LABEL_TO_ID = {"harassment": 0, "sexual": 1, "neutral": 2}


def _normalize_label(raw: str) -> str:
    s = str(raw).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    return s


def map_raw_label_to_two_stage(raw_label: str) -> Tuple[str, str]:
    """
    Map a dataset label into:
      (binary_label, category)

    category:
      - if binary_label == "threat" -> "threat"
      - if binary_label == "non_threat" -> one of "harassment"|"sexual"|"neutral"

    Required merges enforced:
      - "sexual threat" -> threat
      - "threat violence" -> threat
      - "safe" -> neutral
    """

    s = _normalize_label(raw_label)

    # ---- Threat mapping (binary: threat) ----
    # Some datasets already use the collapsed label directly.
    if s == "threat":
        return "threat", "threat"

    if s in {
        "sexual_threat",
        "threat_sexual",
        "rape_threat",
        "threat_violence",
        "sexual_violation_threat",
        "violent_threat",
        "sexual_threats",
        "threat_violences",
    }:
        return "threat", "threat"

    # Heuristic: threat keywords in label name
    if "threat" in s and ("sexual" in s or "rape" in s or "violence" in s):
        return "threat", "threat"

    # ---- Neutral mapping (safe -> neutral) ----
    if s in {"safe", "neutral", "ok", "clean", "normal", "benign", "none"}:
        return "non_threat", "neutral"

    # ---- Harassment vs Sexual (non-threat only) ----
    if (
        s in {
            "harassment",
            "harassment_general",
            "misogyny",
            "misogynistic",
            "insult",
            "abusive",
            "abuse",
            "bullying",
        }
        or "harass" in s
        or "misogyn" in s
    ):
        return "non_threat", "harassment"

    # Sexual class here means sexual harassment, NOT sexual threat
    if (
        s in {"sexual", "sexual_harassment", "unwanted_sex", "sexual_comment"}
        or ("sexual" in s and "threat" not in s and "rape" not in s)
    ):
        return "non_threat", "sexual"

    raise ValueError(f"Unknown/unsupported label for 2-stage mapping: {raw_label!r} (normalized={s!r})")


def map_raw_label_to_training_ids(raw_label: str) -> Tuple[int, int | None]:
    """
    Returns:
      (binary_id, multiclass_id_or_None)

    multiclass_id_or_None is None when the label maps to the threat class.
    """

    binary_label, category = map_raw_label_to_two_stage(raw_label)
    binary_id = BINARY_LABEL_TO_ID[binary_label]
    if binary_id == BINARY_LABEL_TO_ID["threat"]:
        return binary_id, None
    return binary_id, MULTICLASS_LABEL_TO_ID[category]

