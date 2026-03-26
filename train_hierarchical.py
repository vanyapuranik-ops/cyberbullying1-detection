#!/usr/bin/env python3
"""
2-stage threat detection training (strict pipeline).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from labels import BINARY_LABEL_TO_ID, MULTICLASS_LABEL_TO_ID, map_raw_label_to_training_ids
from preprocessing import identity_preprocess, preprocess_text, tokenize_keep_emojis

ROOT = Path(__file__).resolve().parent


def _load_csv(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}. Found columns={list(df.columns)}")
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    df = df.rename(columns={text_col: "text", label_col: "label"})
    return df[["text", "label"]]


def _train_logreg(X, y, *, max_iter: int = 900) -> LogisticRegression:
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", solver="lbfgs")
    clf.fit(X, y)
    return clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dataset", type=Path, required=True, help="CSV with columns: text,label")
    ap.add_argument("--emoji-dataset", type=Path, required=True, help="CSV with columns: text,label")
    ap.add_argument("--out-dir", type=Path, default=ROOT, help="Where to write vectorizer/models")
    ap.add_argument("--dataset-version", type=str, default="unspecified")
    ap.add_argument("--max-features", type=int, default=50000)
    ap.add_argument("--min-df", type=int, default=1)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_text = _load_csv(args.text_dataset, "text", "label")
    df_emoji = _load_csv(args.emoji_dataset, "text", "label")
    df = pd.concat([df_text, df_emoji], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    df["processed_text"] = df["text"].apply(preprocess_text)
    mapped = df["label"].map(map_raw_label_to_training_ids)
    df["binary_id"] = mapped.apply(lambda x: x[0])
    df["multiclass_id"] = mapped.apply(lambda x: x[1])

    threat_id = BINARY_LABEL_TO_ID["threat"]
    non_threat_id = BINARY_LABEL_TO_ID["non_threat"]

    if df["binary_id"].isna().any():
        raise ValueError("Some rows mapped to NaN binary_id; check labels mapping.")
    if ((df["binary_id"] == threat_id) & df["multiclass_id"].notna()).any():
        raise ValueError("Threat rows must NOT have a multiclass_id.")
    if ((df["binary_id"] == non_threat_id) & df["multiclass_id"].isna()).any():
        raise ValueError("Non-threat rows must have a multiclass_id.")

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_keep_emojis,
        preprocessor=identity_preprocess,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=args.max_features,
        min_df=args.min_df,
        sublinear_tf=True,
    )
    X_all = vectorizer.fit_transform(df["processed_text"])

    y_bin = df["binary_id"].astype(int).values
    strat_bin = y_bin if pd.Series(y_bin).value_counts().min() >= 2 else None
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_all, y_bin, test_size=0.2, random_state=42, stratify=strat_bin
    )
    binary_model = _train_logreg(X_train_bin, y_train_bin)

    proba_bin = binary_model.predict_proba(X_test_bin)
    threat_proba = proba_bin[:, threat_id]
    thresholds = [i / 100 for i in range(5, 96, 5)]
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (threat_proba >= t).astype(int)
        f1 = f1_score(y_test_bin, pred, pos_label=threat_id)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    (out_dir / "threat_threshold.txt").write_text(str(best_t), encoding="utf-8")

    non_threat_mask = df["binary_id"].astype(int).values == non_threat_id
    X_non = X_all[non_threat_mask]
    y_multi = df.loc[non_threat_mask, "multiclass_id"].astype(int).values
    expected_classes = set(MULTICLASS_LABEL_TO_ID.values())
    if not set(y_multi.tolist()).issubset(expected_classes):
        raise ValueError(f"Unexpected multiclass ids. Got={set(y_multi.tolist())} expected subset={expected_classes}")
    strat_multi = y_multi if pd.Series(y_multi).value_counts().min() >= 2 else None
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_non, y_multi, test_size=0.2, random_state=42, stratify=strat_multi
    )
    multiclass_model = _train_logreg(X_train_m, y_train_m)

    with open(out_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(out_dir / "binary_model.pkl", "wb") as f:
        pickle.dump(binary_model, f)
    with open(out_dir / "multiclass_model.pkl", "wb") as f:
        pickle.dump(multiclass_model, f)

    (out_dir / "two_stage_manifest.txt").write_text(
        f"dataset_version={args.dataset_version}\ntext_dataset={args.text_dataset}\nemoji_dataset={args.emoji_dataset}\n",
        encoding="utf-8",
    )
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
