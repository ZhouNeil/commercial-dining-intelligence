from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class RestaurantSearchIndex:
    """
    Precomputed search artifacts for restaurant retrieval.

    restaurant_matrix: TF-IDF vectors for each selected restaurant (rows aligned with restaurant_ids)
    restaurant_norms: precomputed L2 norms for cosine similarity
    """

    restaurant_ids: np.ndarray
    restaurant_matrix: csr_matrix
    restaurant_norms: np.ndarray
    vectorizer: TfidfVectorizer
    meta: pd.DataFrame
    stars_norm: np.ndarray  # normalized to [0, 1] for ranking prior


class TouristRetrieval:
    """
    MVP: keyword-based restaurant recommendation.

    Build:
      - Aggregate review texts per restaurant into a single document (for a subset of restaurants).
      - Fit a TF-IDF vectorizer over those documents.
      - Retrieve by cosine similarity (implemented manually) + rerank prior from stars.
    """

    def __init__(
        self,
        data_dir: str | Path = "../data/cleaned",
        index_dir: str | Path = "../models/artifacts",
        max_businesses: int = 5000,
        max_reviews_per_business: int = 20,
        max_features: int = 50000,
        random_state: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.max_businesses = max_businesses
        self.max_reviews_per_business = max_reviews_per_business
        self.max_features = max_features
        self.random_state = random_state
        # Bump this whenever retrieval/index-building logic changes meaningfully.
        self.index_version = 2

    @property
    def business_path(self) -> Path:
        return self.data_dir / "business_dining.csv"

    @property
    def review_path(self) -> Path:
        return self.data_dir / "review_dining.csv"

    def _index_paths(self) -> dict[str, Path]:
        return {
            "vectorizer": self.index_dir / "vectorizer.joblib",
            "matrix": self.index_dir / "restaurant_matrix.npz",
            "ids": self.index_dir / "restaurant_ids.npy",
            "meta": self.index_dir / "meta.csv",
            "config": self.index_dir / "index_config.json",
        }

    def _should_rebuild(self) -> bool:
        paths = self._index_paths()
        # If any artifact is missing, rebuild.
        missing = any(not p.exists() for k, p in paths.items() if k != "config")
        if missing:
            return True

        # Config mismatch -> rebuild.
        cfg_path = paths["config"]
        if not cfg_path.exists():
            return True
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return True

        expected = {
            "index_version": self.index_version,
            "max_businesses": self.max_businesses,
            "max_reviews_per_business": self.max_reviews_per_business,
            "max_features": self.max_features,
            "vectorizer_min_df": 1,
            "vectorizer_ngram_range": [1, 2],
            "categories_norm_in_docs": True,
        }
        return cfg != expected

    def _build_documents(self) -> tuple[np.ndarray, list[str], pd.DataFrame]:
        """
        Build restaurant "documents" by aggregating up to max_reviews_per_business texts per restaurant.

        We select the top-N restaurants by review_count for a stable MVP candidate set.
        """
        business_df = pd.read_csv(self.business_path)
        if "review_count" not in business_df.columns:
            raise ValueError("business_dining.csv must contain 'review_count' column.")

        # Candidate set selection:
        # Previously we picked a global Top-N by review_count.
        # That can cause state/city filters to become "empty" for users even if restaurants exist,
        # because they were never included in the index candidate set.
        business_df = business_df.sort_values("review_count", ascending=False)
        if self.max_businesses is not None and len(business_df) > self.max_businesses and "state" in business_df.columns:
            # Allocate index budget roughly uniformly across states to improve filter coverage.
            business_df["state"] = business_df["state"].astype(str).str.strip().str.upper()
            states = business_df["state"].unique().tolist()
            n_states = max(len(states), 1)
            per_state = max(50, self.max_businesses // n_states)
            business_df = business_df.groupby("state", group_keys=False).head(per_state)
            if len(business_df) > self.max_businesses:
                business_df = business_df.head(self.max_businesses)
        elif self.max_businesses is not None and len(business_df) > self.max_businesses:
            business_df = business_df.head(self.max_businesses)

        # Ensure stable types for matching
        business_df["business_id"] = business_df["business_id"].astype(str)
        if "city" not in business_df.columns:
            business_df["city"] = ""
        if "state" not in business_df.columns:
            business_df["state"] = ""
        business_df["city"] = business_df["city"].astype(str).str.strip()
        business_df["state"] = business_df["state"].astype(str).str.strip().str.upper()

        business_ids = business_df["business_id"].to_numpy()

        # Collect review texts per restaurant.
        # Note: we read in chunks to avoid loading the whole review_dining.csv.
        selected_set = set(business_ids.tolist())
        text_bank: dict[str, list[str]] = {bid: [] for bid in selected_set}
        cat_bank: dict[str, str] = {}
        if "categories" in business_df.columns:
            cat_bank = business_df.set_index("business_id")["categories"].fillna("").astype(str).to_dict()

        chunksize = 100_000
        for chunk in pd.read_csv(self.review_path, chunksize=chunksize):
            # Fast filter
            chunk = chunk[chunk["business_id"].astype(str).isin(selected_set)]
            if chunk.empty:
                continue

            # Append texts, up to limit
            # We keep it simple: join raw text; TF-IDF handles tokenization.
            for bid, txt in zip(chunk["business_id"].astype(str).values, chunk["text"].values):
                if len(text_bank[bid]) >= self.max_reviews_per_business:
                    continue
                if isinstance(txt, str) and txt.strip():
                    text_bank[bid].append(txt)

            # Early stop if all filled
            done = all(len(v) >= min(3, self.max_reviews_per_business) for v in text_bank.values())
            if done:
                break

        # Final docs in the same order as business_ids
        docs: list[str] = []
        for bid in business_ids:
            texts = text_bank.get(bid, [])
            cats = cat_bank.get(bid, "")
            if not texts:
                # Include category text to improve keyword matching even if review text is missing.
                docs.append(cats)
            else:
                # Limit total length a bit for MVP stability
                docs.append(cats + " " + " ".join(texts[: self.max_reviews_per_business]))

        meta = business_df.copy()
        meta["business_id"] = meta["business_id"].astype(str)
        # Normalized fields for strict filtering (case/space insensitive)
        meta["state_norm"] = meta["state"].astype(str).str.strip().str.upper()
        meta["city_norm"] = meta["city"].astype(str).str.strip().str.lower()
        meta["categories_norm"] = meta.get("categories", "").fillna("").astype(str).str.lower()
        return business_ids, docs, meta

    def build_or_load_index(self, force_rebuild: bool = False) -> RestaurantSearchIndex:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if force_rebuild or self._should_rebuild():
            business_ids, docs, meta = self._build_documents()

            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words="english",
                # Use more permissive thresholds for MVP keyword matching (e.g. "steakhouse")
                min_df=1,
                ngram_range=(1, 2),
            )
            restaurant_matrix = vectorizer.fit_transform(docs).tocsr()

            # Precompute L2 norms for cosine similarity
            norms = np.sqrt(restaurant_matrix.power(2).sum(axis=1)).A1

            # Normalized stars prior for reranking
            stars = meta["stars"].astype(float).to_numpy()
            stars_norm = (stars - 1.0) / 4.0
            stars_norm = np.clip(stars_norm, 0.0, 1.0)

            paths = self._index_paths()
            dump(vectorizer, paths["vectorizer"])
            save_npz(paths["matrix"], restaurant_matrix)

            # Save ids as a plain string array to avoid pickle issues on load.
            business_ids = np.asarray(business_ids, dtype=str)
            np.save(paths["ids"], business_ids, allow_pickle=False)
            meta.to_csv(paths["meta"], index=False)

            # Write config for future rebuild decisions
            cfg = {
                "index_version": self.index_version,
                "max_businesses": self.max_businesses,
                "max_reviews_per_business": self.max_reviews_per_business,
                "max_features": self.max_features,
                "vectorizer_min_df": 1,
                "vectorizer_ngram_range": [1, 2],
                "categories_norm_in_docs": True,
            }
            paths["config"].write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        # Load artifacts
        paths = self._index_paths()
        vectorizer = load(paths["vectorizer"])
        restaurant_matrix = load_npz(paths["matrix"])

        # Older/broken artifacts may have been saved as dtype=object; allow_pickle=True for robustness.
        restaurant_ids = np.load(paths["ids"], allow_pickle=True).astype(str)
        meta = pd.read_csv(paths["meta"])

        # Align meta order with restaurant_ids (in case CSV order differs)
        meta = meta.set_index("business_id").loc[restaurant_ids].reset_index()

        # Backward compatible normalization columns (older artifacts may miss them)
        if "state" not in meta.columns:
            meta["state"] = ""
        if "city" not in meta.columns:
            meta["city"] = ""
        if "state_norm" not in meta.columns:
            meta["state_norm"] = meta["state"].astype(str).str.strip().str.upper()
        if "city_norm" not in meta.columns:
            meta["city_norm"] = meta["city"].astype(str).str.strip().str.lower()
        if "categories_norm" not in meta.columns:
            meta["categories_norm"] = meta.get("categories", "").fillna("").astype(str).str.lower()

        stars = meta["stars"].astype(float).to_numpy()
        stars_norm = (stars - 1.0) / 4.0
        stars_norm = np.clip(stars_norm, 0.0, 1.0)
        norms = np.sqrt(restaurant_matrix.power(2).sum(axis=1)).A1

        return RestaurantSearchIndex(
            restaurant_ids=restaurant_ids,
            restaurant_matrix=restaurant_matrix.tocsr(),
            restaurant_norms=norms,
            vectorizer=vectorizer,
            meta=meta,
            stars_norm=stars_norm,
        )

    @staticmethod
    def _cosine_scores(restaurant_matrix: csr_matrix, restaurant_norms: np.ndarray, query_vec) -> np.ndarray:
        """
        Cosine similarity implemented manually (not using sklearn nearest neighbors).
        query_vec is a sparse row vector.
        """
        # raw dot products: (1 x n) -> (n,)
        dot = restaurant_matrix @ query_vec.T  # (n x 1) sparse
        dot = dot.toarray().ravel()
        q_norm = np.sqrt(query_vec.power(2).sum()).item()
        denom = (restaurant_norms * q_norm) + 1e-9
        return dot / denom

    def recommend_keywords(
        self,
        keywords: str,
        index: RestaurantSearchIndex,
        state: Optional[str] = None,
        city: Optional[str] = None,
        cuisines: Optional[list[str]] = None,
        top_k: int = 10,
        alpha: float = 1.0,
        beta: float = 0.2,
    ) -> pd.DataFrame:
        if not isinstance(keywords, str) or not keywords.strip():
            raise ValueError("keywords must be a non-empty string.")

        query_text = keywords.strip()
        query_vec = index.vectorizer.transform([query_text])  # sparse (1 x d)

        sim = self._cosine_scores(index.restaurant_matrix, index.restaurant_norms, query_vec)

        # Optional filtering by state/city (strict, normalized match)
        mask = np.ones_like(sim, dtype=bool)
        if state and state.strip() and state.strip().lower() != "all":
            state_norm_q = state.strip().upper()
            mask &= index.meta["state_norm"].astype(str) == state_norm_q
        if city and city.strip() and city.strip().lower() != "all":
            city_norm_q = city.strip().lower()
            mask &= index.meta["city_norm"].astype(str) == city_norm_q

        # Optional filtering by cuisines (semantic rules over business `categories`)
        # Strict for state/city, but we relax cuisines if it makes the result empty,
        # so users still get something relevant within the chosen location.
        if cuisines:
            categories_norm = index.meta["categories_norm"].astype(str)

            # Map UI cuisine labels to category keywords (semantic rules, no pre-tagging required).
            cuisine_kw = {
                "Sushi": ["sushi", "japanese"],
                "Steakhouse": ["steak", "steakhouse"],
                "Korean": ["korean"],
                "Fast Food": ["fast food", "burger", "fries", "takeout"],
                "Chinese": [
                    "chinese",
                    "cantonese",
                    "sichuan",
                    "mandarin",
                    "hunan",
                    "taiwanese",
                    "dim sum",
                    "noodles",
                    "mongolian",
                ],
                "Burger": ["burger", "hamburgers", "burgers"],
                "Healthy": ["salad", "healthy", "vegan", "vegetarian"],
            }

            # Start from state/city filtered mask; apply cuisine filter on top.
            mask_statecity = mask.copy()

            cuisine_mask = np.zeros_like(mask, dtype=bool)
            for c in cuisines:
                c_key = str(c)
                kws = cuisine_kw.get(c_key, [])
                if not kws:
                    continue
                matched = np.zeros_like(mask, dtype=bool)
                for kw in kws:
                    matched |= categories_norm.str.contains(kw, na=False)
                cuisine_mask |= matched

            combined = mask_statecity & cuisine_mask
            if combined.sum() > 0:
                mask = combined
            else:
                # Relax cuisines filter to avoid returning results from wrong states/cities.
                mask = mask_statecity

        if mask.sum() == 0:
            # If filter becomes empty, return empty results (better than returning wrong states).
            cols = [
                "name",
                "address",
                "city",
                "state",
                "stars",
                "review_count",
                "similarity",
                "final_score",
                "latitude",
                "longitude",
            ]
            return pd.DataFrame(columns=cols)

        idx = np.where(mask)[0]
        final_score = alpha * sim[idx] + beta * index.stars_norm[idx]

        if len(idx) <= top_k:
            top_local = np.argsort(final_score)[::-1]
        else:
            # partial sort then order
            top_local = np.argpartition(final_score, -top_k)[-top_k:]
            top_local = top_local[np.argsort(final_score[top_local])[::-1]]

        top_idx = idx[top_local]
        out = index.meta.iloc[top_idx].copy()
        out["similarity"] = sim[top_idx]
        out["final_score"] = final_score[top_local]
        out = out.sort_values("final_score", ascending=False).head(top_k)

        # Frontend-friendly fields (avoid showing business_id directly)
        return out[
            [
                "name",
                "address",
                "city",
                "state",
                "stars",
                "review_count",
                "similarity",
                "final_score",
                "latitude",
                "longitude",
            ]
        ]

