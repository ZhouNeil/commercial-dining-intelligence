#!/usr/bin/env python3
"""
Train a supervised business score model: P(still open) from environment + category features only.

Does NOT use survival_probability or rating-model outputs (avoids circular supervision with y=is_open).

Usage (repo root):
  python models/train_business_score_ml.py
  python models/train_business_score_ml.py --csv data/train_merchant_split.csv --out models/artifacts/business_score_ml.pkl
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DROP_ALWAYS = {
    "is_open",  # label
    "business_id",
    "name",
    "address",
    "city",
    "state",
    "postal_code",
    "categories",
    "hours",
    "stars",  # avoid outcome leakage; model uses env + category only
}


def default_train_path(repo: Path) -> Path:
    p = repo / "data" / "train_merchant_split.csv"
    if p.is_file():
        return p
    return repo / "train_merchant_split.csv"


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if "is_open" not in df.columns:
        raise ValueError("CSV must contain is_open")
    y = df["is_open"].astype(int).to_numpy()
    drop = [c for c in DROP_ALWAYS if c in df.columns]
    X = df.drop(columns=drop, errors="ignore")
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    X = X.drop(columns=non_numeric, errors="ignore")
    X = X.fillna(0.0)
    return X, y


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None, help="Training table (default: data/train_merchant_split.csv)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output joblib path (default: models/artifacts/business_score_ml.pkl)",
    )
    ap.add_argument("--test-size", type=float, default=0.2, dest="test_size")
    ap.add_argument("--random-state", type=int, default=42, dest="random_state")
    ap.add_argument("--max-samples", type=int, default=0, help="Subsample rows for speed (0 = all)")
    args = ap.parse_args()
    os.chdir(repo)

    csv_path = args.csv or default_train_path(repo)
    if not csv_path.is_file():
        raise SystemExit(f"Training CSV not found: {csv_path.resolve()}")

    out_path = args.out or (repo / "models" / "artifacts" / "business_score_ml.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.random_state)
        print(f"Subsampled to n={args.max_samples}")

    X, y = build_xy(df)
    print(f"Features: {X.shape[1]} columns, n={X.shape[0]}, positive rate={y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    clf = HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=200,
        random_state=args.random_state,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    proba_tr = clf.predict_proba(X_train)[:, 1]
    proba_te = clf.predict_proba(X_test)[:, 1]
    print(f"Train ROC-AUC: {roc_auc_score(y_train, proba_tr):.4f}")
    print(f"Test  ROC-AUC: {roc_auc_score(y_test, proba_te):.4f}")

    # Refit on full data for deployment artifact
    clf_full = HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=200,
        random_state=args.random_state,
        class_weight="balanced",
    )
    clf_full.fit(X, y)
    joblib.dump(clf_full, out_path)
    print(f"Wrote {out_path.resolve()} (fit on all {len(X)} rows)")


if __name__ == "__main__":
    main()
