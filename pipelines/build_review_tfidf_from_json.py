"""
Build a TF-IDF sparse matrix from Yelp's raw review.json (JSON Lines).

Streams the file line-by-line (memory-safe for large dumps). Aggregates review
`text` per `business_id`, restricted to businesses present in business_dining.csv
to match the dining subset and bound memory.

Outputs (default: models/artifacts/review_json_tfidf/):
  - restaurant_matrix.npz   (CSR sparse TF-IDF)
  - restaurant_ids.npy
  - vectorizer.joblib
  - meta.csv
  - build_config.json

Environment:
  YELP_REVIEW_JSON  Optional default path to review.json if --review-json is omitted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_JSON = REPO_ROOT / "data" / "raw" / "yelp" / "review.json"
DEFAULT_BUSINESS_CSV = REPO_ROOT / "data" / "cleaned" / "business_dining.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "models" / "artifacts" / "review_json_tfidf"

BUILD_VERSION = 1


def _resolve_review_json(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    env = os.environ.get("YELP_REVIEW_JSON", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_REVIEW_JSON.resolve()


def _select_businesses(
    business_df: pd.DataFrame,
    max_businesses: int | None,
) -> pd.DataFrame:
    """Mirror app.retrieval.TouristRetrieval candidate selection when possible."""
    business_df = business_df.copy()
    if "review_count" not in business_df.columns:
        raise ValueError("business CSV must contain 'review_count'.")
    business_df = business_df.sort_values("review_count", ascending=False)

    if max_businesses is not None and len(business_df) > max_businesses and "state" in business_df.columns:
        business_df["state"] = business_df["state"].astype(str).str.strip().str.upper()
        states = business_df["state"].unique().tolist()
        n_states = max(len(states), 1)
        per_state = max(50, max_businesses // n_states)
        business_df = business_df.groupby("state", group_keys=False).head(per_state)
        if len(business_df) > max_businesses:
            business_df = business_df.head(max_businesses)
    elif max_businesses is not None and len(business_df) > max_businesses:
        business_df = business_df.head(max_businesses)

    business_df["business_id"] = business_df["business_id"].astype(str)
    if "city" not in business_df.columns:
        business_df["city"] = ""
    if "state" not in business_df.columns:
        business_df["state"] = ""
    business_df["city"] = business_df["city"].astype(str).str.strip()
    business_df["state"] = business_df["state"].astype(str).str.strip().str.upper()
    return business_df


def stream_aggregate_texts(
    review_json_path: Path,
    selected_ids: set[str],
    max_reviews_per_business: int,
) -> dict[str, list[str]]:
    text_bank: dict[str, list[str]] = {bid: [] for bid in selected_ids}
    lines_read = 0
    matched = 0

    with open(review_json_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines_read += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = str(obj.get("business_id", ""))
            if bid not in selected_ids:
                continue
            txt = obj.get("text")
            if not isinstance(txt, str) or not txt.strip():
                continue
            bucket = text_bank[bid]
            if len(bucket) >= max_reviews_per_business:
                continue
            bucket.append(txt)
            matched += 1

    print(
        f"  Parsed {lines_read} non-empty lines; "
        f"stored {matched} review texts for selected businesses."
    )
    return text_bank


def build_documents_from_banks(
    business_df: pd.DataFrame,
    text_bank: dict[str, list[str]],
    max_reviews_per_business: int,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    business_ids = business_df["business_id"].to_numpy()
    cat_bank: dict[str, str] = {}
    if "categories" in business_df.columns:
        cat_bank = (
            business_df.set_index("business_id")["categories"].fillna("").astype(str).to_dict()
        )

    docs: list[str] = []
    for bid in business_ids:
        bid = str(bid)
        texts = text_bank.get(bid, [])
        cats = cat_bank.get(bid, "")
        if not texts:
            docs.append(cats)
        else:
            docs.append(cats + " " + " ".join(texts[:max_reviews_per_business]))

    meta = business_df.copy()
    meta["business_id"] = meta["business_id"].astype(str)
    meta["state_norm"] = meta["state"].astype(str).str.strip().str.upper()
    meta["city_norm"] = meta["city"].astype(str).str.strip().str.lower()
    meta["categories_norm"] = meta.get("categories", "").fillna("").astype(str).str.lower()
    return business_ids, docs, meta


def run_build(
    review_json_path: Path,
    business_csv_path: Path,
    out_dir: Path,
    max_businesses: int | None,
    max_reviews_per_business: int,
    max_features: int,
    min_df: int,
    ngram_lo: int,
    ngram_hi: int,
    force: bool,
) -> None:
    if not review_json_path.is_file():
        print(
            f"ERROR: review.json not found at:\n  {review_json_path}\n"
            "Place the Yelp review.json there, set YELP_REVIEW_JSON, or pass --review-json.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not business_csv_path.is_file():
        print(f"ERROR: business CSV not found: {business_csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = out_dir.resolve()
    paths = {
        "vectorizer": out_dir / "vectorizer.joblib",
        "matrix": out_dir / "restaurant_matrix.npz",
        "ids": out_dir / "restaurant_ids.npy",
        "meta": out_dir / "meta.csv",
        "config": out_dir / "build_config.json",
    }

    if not force and all(p.exists() for k, p in paths.items() if k != "config"):
        print(f"Artifacts already exist under {out_dir}. Use --force to rebuild.")
        return

    business_df = pd.read_csv(business_csv_path)
    business_df = _select_businesses(business_df, max_businesses)
    selected_set = set(business_df["business_id"].astype(str).tolist())

    print(f"Streaming {review_json_path} for {len(selected_set)} businesses (max {max_reviews_per_business} reviews each)...")
    text_bank = stream_aggregate_texts(
        review_json_path,
        selected_set,
        max_reviews_per_business,
    )

    business_ids, docs, meta = build_documents_from_banks(
        business_df,
        text_bank,
        max_reviews_per_business,
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=min_df,
        ngram_range=(ngram_lo, ngram_hi),
    )
    restaurant_matrix = vectorizer.fit_transform(docs).tocsr()

    out_dir.mkdir(parents=True, exist_ok=True)
    dump(vectorizer, paths["vectorizer"])
    save_npz(paths["matrix"], restaurant_matrix)
    np.save(paths["ids"], np.asarray(business_ids, dtype=str), allow_pickle=False)
    meta.to_csv(paths["meta"], index=False)

    cfg = {
        "build_version": BUILD_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_json": str(review_json_path),
        "business_csv": str(business_csv_path),
        "out_dir": str(out_dir),
        "max_businesses": max_businesses,
        "max_reviews_per_business": max_reviews_per_business,
        "max_features": max_features,
        "vectorizer_min_df": min_df,
        "vectorizer_ngram_range": [ngram_lo, ngram_hi],
        "n_rows": int(restaurant_matrix.shape[0]),
        "n_features": int(restaurant_matrix.shape[1]),
    }
    paths["config"].write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Matrix shape: {restaurant_matrix.shape}. Saved under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TF-IDF matrix from Yelp review.json (JSON Lines) for dining businesses."
    )
    parser.add_argument(
        "--review-json",
        default=None,
        help=f"Path to review.json (default: YELP_REVIEW_JSON env or {DEFAULT_REVIEW_JSON})",
    )
    parser.add_argument(
        "--business-csv",
        type=Path,
        default=DEFAULT_BUSINESS_CSV,
        help=f"Path to business_dining.csv (default: {DEFAULT_BUSINESS_CSV})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--max-businesses", type=int, default=None, help="Cap businesses (like tourist index)")
    parser.add_argument("--max-reviews-per-business", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Rebuild even if artifacts exist")
    args = parser.parse_args()

    review_path = _resolve_review_json(args.review_json)
    run_build(
        review_json_path=review_path,
        business_csv_path=args.business_csv.resolve(),
        out_dir=args.out_dir.resolve(),
        max_businesses=args.max_businesses,
        max_reviews_per_business=args.max_reviews_per_business,
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_lo=args.ngram_min,
        ngram_hi=args.ngram_max,
        force=args.force,
    )


if __name__ == "__main__":
    main()
