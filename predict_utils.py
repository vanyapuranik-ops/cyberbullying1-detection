"""
Prediction utilities for the strict 2-stage pipeline.

Required output format:
{
  "binary_label": "threat" | "non_threat",
  "category": "harassment" | "sexual" | "neutral" | "threat"
}
"""

from __future__ import annotations

import joblib
from functools import lru_cache
from pathlib import Path

from labels import BINARY_LABELS, MULTICLASS_LABELS
from preprocessing import preprocess_text


ROOT = Path(__file__).resolve().parent

VECTOR_PATH = ROOT / "vectorizer.pkl"
BINARY_MODEL_PATH = ROOT / "binary_model.pkl"
MULTICLASS_MODEL_PATH = ROOT / "multiclass_model.pkl"
THRESHOLD_PATH = ROOT / "threat_threshold.txt"


import joblib

@lru_cache(maxsize=1)
def load_two_stage_artifacts():
    if not VECTOR_PATH.is_file():
        raise FileNotFoundError(f"Missing {VECTOR_PATH}")
    if not BINARY_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing {BINARY_MODEL_PATH}")
    if not MULTICLASS_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing {MULTICLASS_MODEL_PATH}")

    vectorizer = joblib.load(VECTOR_PATH)
    binary_model = joblib.load(BINARY_MODEL_PATH)
    multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)

    return vectorizer, binary_model, multiclass_model



def predict_two_stage(text_input: str, parent_caption=None, prior_replies=None):

    text_lower = text_input.lower()

    # --- STRONG IMPLICIT + SEXUAL THREAT RULE ---
    if any(phrase in text_lower for phrase in [
        "rape", "raped", "raping",
        "force myself", "force yourself",
        "take advantage of you",
        "you will regret this", "you'll regret this",
        "wait and see what happens",
        "you'll see what happens",
        "won't be safe", "not safe",
        "hurt you", "kill you", "beat you"
    ]):
        return {"binary_label": "threat", "category": "threat"}

    # Load models
    vectorizer, binary_model, multiclass_model = load_two_stage_artifacts()

    from preprocessing import prepare_for_model

    # Add context
    processed, _ = prepare_for_model(
        text_input,
        parent_caption=parent_caption,
        prior_replies=prior_replies,
    )

    X = vectorizer.transform([processed])

    # Threshold logic
    threshold = 0.5
    if THRESHOLD_PATH.is_file():
        try:
            threshold = float(THRESHOLD_PATH.read_text().strip())
        except:
            pass

    proba = binary_model.predict_proba(X)[0]
    threat_prob = float(proba[1])

    binary_label = "threat" if threat_prob >= threshold else "non_threat"

    if binary_label == "threat":
        return {"binary_label": "threat", "category": "threat"}

    # Multiclass prediction
    category = multiclass_model.predict(X)[0]

    # Normalize labels
    if str(category).lower() == "safe":
        category = "neutral"
    elif "harassment" in str(category).lower():
        category = "harassment"

    return {"binary_label": "non_threat", "category": category}

    # Multiclass prediction
    category = multiclass_model.predict(X)[0]

    # Normalize label
    if str(category).lower() == "safe":
        category = "neutral"
    elif "harassment" in str(category).lower():
        category = "harassment"

    return {"binary_label": "non_threat", "category": category}

    from preprocessing import prepare_for_model

    # Add context here
    processed, _ = prepare_for_model(
        text_input,
        parent_caption=parent_caption,
        prior_replies=prior_replies,
    )

    X = vectorizer.transform([processed])

    # Threshold logic
    threshold = 0.5
    if THRESHOLD_PATH.is_file():
        try:
            threshold = float(THRESHOLD_PATH.read_text().strip())
        except:
            pass

    proba = binary_model.predict_proba(X)[0]
    threat_prob = float(proba[1])

    binary_label = "threat" if threat_prob >= threshold else "non_threat"

    if binary_label == "threat":
        return {"binary_label": "threat", "category": "threat"}

    category = multiclass_model.predict(X)[0]

    if str(category).lower() == "safe":
        category = "neutral"

    return {"binary_label": "non_threat", "category": category}


def predict_two_stage_with_scores(text_input: str) -> dict:
    """
    Extra helper for UI: returns threat_prob and multiclass probs.
    Does NOT change the required output of `predict_two_stage`.
    """
    vectorizer, binary_model, multiclass_model = load_two_stage_artifacts()

    processed = preprocess_text(text_input)
    X = vectorizer.transform([processed])

    threshold = 0.5
    if THRESHOLD_PATH.is_file():
        try:
            threshold = float(THRESHOLD_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            threshold = 0.5

    proba_bin = binary_model.predict_proba(X)[0]
    threat_prob = float(proba_bin[1])
    binary_label = "threat" if threat_prob >= threshold else "non_threat"

    if binary_label == "threat":
        return {
            "binary_label": "threat",
            "category": "threat",
            "threat_prob": threat_prob,
        }

    proba_multi = multiclass_model.predict_proba(X)[0]
    category_probs = {MULTICLASS_LABELS[i]: float(p) for i, p in enumerate(proba_multi)}
    multiclass_id = int(multiclass_model.predict(X)[0])
    category = MULTICLASS_LABELS[multiclass_id]

    return {
        "binary_label": "non_threat",
        "category": category,
        "threat_prob": threat_prob,
        "category_probs": category_probs,
    }

