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
        embed_google_maps: bool = True,
        restrict_index_cities: bool = True,
        rating_trust_ref_reviews: float = 150.0,
    ):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.max_businesses = max_businesses
        self.max_reviews_per_business = max_reviews_per_business
        self.max_features = max_features
        self.random_state = random_state
        self.embed_google_maps = embed_google_maps
        self.restrict_index_cities = restrict_index_cities
        self.rating_trust_ref_reviews = float(rating_trust_ref_reviews)
        # Bump this whenever retrieval/index-building logic changes meaningfully.
        self.index_version = 5

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

        from app.core.index_cities import INDEX_FILTER_ID

        expected = {
            "index_version": self.index_version,
            "max_businesses": self.max_businesses,
            "max_reviews_per_business": self.max_reviews_per_business,
            "max_features": self.max_features,
            "vectorizer_min_df": 1,
            "vectorizer_ngram_range": [1, 2],
            "categories_norm_in_docs": True,
            "embed_google_maps": self.embed_google_maps,
            "restrict_index_cities": self.restrict_index_cities,
            "index_city_filter_id": INDEX_FILTER_ID if self.restrict_index_cities else "none",
            "rating_trust_ref_reviews": self.rating_trust_ref_reviews,
        }
        return cfg != expected

    def _build_documents(self) -> tuple[np.ndarray, list[str], pd.DataFrame]:
        """
        Build restaurant "documents" by aggregating up to max_reviews_per_business texts per restaurant.

        We select the top-N restaurants by review_count for a stable MVP candidate set.
        """
        from app.core.google_maps_loader import (
            google_maps_csv_path,
            load_google_maps_as_yelp_schema,
            synthetic_google_profile_snippets,
        )

        business_df = pd.read_csv(self.business_path, low_memory=False)
        if "review_count" not in business_df.columns:
            raise ValueError("business_dining.csv must contain 'review_count' column.")

        for _c in ("_gm_detail", "_gm_url"):
            if _c not in business_df.columns:
                business_df[_c] = ""
        if self.embed_google_maps:
            _gmp = google_maps_csv_path(self.data_dir)
            if _gmp.exists():
                _gm = load_google_maps_as_yelp_schema(_gmp)
                if _gm is not None and not _gm.empty:
                    business_df = pd.concat([business_df, _gm], ignore_index=True)

        if self.restrict_index_cities:
            from app.core.index_cities import INDEX_ALLOWED_CITY_STATE

            if "city" not in business_df.columns:
                business_df["city"] = ""
            if "state" not in business_df.columns:
                business_df["state"] = ""
            _ck = business_df["city"].astype(str).str.strip().str.lower()
            _sk = business_df["state"].astype(str).str.strip().str.upper()
            _in_scope = [
                (str(c), str(s)) in INDEX_ALLOWED_CITY_STATE for c, s in zip(_ck, _sk)
            ]
            business_df = business_df.loc[_in_scope].copy()
            if business_df.empty:
                raise ValueError(
                    "Index city filter removed all businesses. Check `app/core/index_cities.py` "
                    "or set restrict_index_cities=False on TouristRetrieval."
                )

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

        business_ids = business_df["business_id"].astype(str).to_numpy()

        # Collect review texts per restaurant.
        # Note: we read in chunks to avoid loading the whole review_dining.csv.
        selected_set = set(business_ids.tolist())
        # Split reviews by sentiment (positive/negative) and extract themes.
        pos_text_bank: dict[str, list[str]] = {str(bid): [] for bid in selected_set}
        neg_text_bank: dict[str, list[str]] = {str(bid): [] for bid in selected_set}
        cat_bank: dict[str, str] = {}
        name_bank: dict[str, str] = {}
        city_bank: dict[str, str] = {}
        state_bank: dict[str, str] = {}
        attrs_bank: dict[str, str] = {}
        if "categories" in business_df.columns:
            cat_bank = business_df.set_index("business_id")["categories"].fillna("").astype(str).to_dict()
        if "name" in business_df.columns:
            name_bank = business_df.set_index("business_id")["name"].fillna("").astype(str).to_dict()
        if "city" in business_df.columns:
            city_bank = business_df.set_index("business_id")["city"].fillna("").astype(str).to_dict()
        if "state" in business_df.columns:
            state_bank = business_df.set_index("business_id")["state"].fillna("").astype(str).to_dict()
        if "attributes" in business_df.columns:
            attrs_bank = business_df.set_index("business_id")["attributes"].fillna("").astype(str).to_dict()

        chunksize = 100_000
        for chunk in pd.read_csv(self.review_path, chunksize=chunksize):
            # Fast filter
            chunk = chunk[chunk["business_id"].astype(str).isin(selected_set)]
            if chunk.empty:
                continue

            # Append texts by sentiment up to per-restaurant limits.
            # Positive: stars >= 4 ; Negative: stars <= 2
            pos_limit = max(1, self.max_reviews_per_business // 2)
            neg_limit = max(1, self.max_reviews_per_business - pos_limit)

            for bid, txt, stars in zip(
                chunk["business_id"].astype(str).values,
                chunk["text"].values,
                chunk["stars"].values if "stars" in chunk.columns else [None] * len(chunk),
            ):
                if not isinstance(txt, str) or not txt.strip():
                    continue
                try:
                    s = float(stars)
                except Exception:
                    continue

                if s >= 4.0:
                    if len(pos_text_bank[bid]) < pos_limit:
                        pos_text_bank[bid].append(txt)
                elif s <= 2.0:
                    if len(neg_text_bank[bid]) < neg_limit:
                        neg_text_bank[bid].append(txt)

            # Early stop when each Yelp-scoped restaurant has enough sentiment text.
            # Google Maps rows have no matching reviews in review_dining.csv — exclude from this check.
            min_total = min(3, self.max_reviews_per_business)
            yelp_only = {str(b) for b in selected_set if not str(b).startswith("gm_")}
            if yelp_only:
                done = all(
                    (len(pos_text_bank[bid]) + len(neg_text_bank[bid])) >= min_total for bid in yelp_only
                )
                if done:
                    break

        # Final docs in the same order as business_ids.
        # Instead of concatenating raw reviews, we build an explainable profile text:
        # business metadata + extracted positive/negative themes.
        from app.core.profile_builder import build_profile_text

        row_by_bid = business_df.set_index("business_id")
        docs: list[str] = []
        for bid in business_ids:
            bid_s = str(bid)
            pos_list = list(pos_text_bank.get(bid_s, []))
            neg_list = list(neg_text_bank.get(bid_s, []))
            if bid_s.startswith("gm_"):
                try:
                    px, nx = synthetic_google_profile_snippets(row_by_bid.loc[bid_s])
                    pos_list.extend(px)
                    neg_list.extend(nx)
                except (KeyError, TypeError):
                    pass
            business_meta = {
                "name": name_bank.get(bid_s, ""),
                "categories": cat_bank.get(bid_s, ""),
                "city": city_bank.get(bid_s, ""),
                "state": state_bank.get(bid_s, ""),
                "attributes": attrs_bank.get(bid_s, ""),
            }
            docs.append(
                build_profile_text(
                    business=business_meta,
                    positive_texts=pos_list,
                    negative_texts=neg_list,
                    top_k_phrases=6,
                )
            )

        meta = business_df.copy()
        meta["business_id"] = meta["business_id"].astype(str)
        meta.drop(columns=["_gm_detail", "_gm_url"], inplace=True, errors="ignore")
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
            from app.core.index_cities import INDEX_FILTER_ID

            cfg = {
                "index_version": self.index_version,
                "max_businesses": self.max_businesses,
                "max_reviews_per_business": self.max_reviews_per_business,
                "max_features": self.max_features,
                "vectorizer_min_df": 1,
                "vectorizer_ngram_range": [1, 2],
                "categories_norm_in_docs": True,
                "embed_google_maps": self.embed_google_maps,
                "restrict_index_cities": self.restrict_index_cities,
                "index_city_filter_id": INDEX_FILTER_ID if self.restrict_index_cities else "none",
                "rating_trust_ref_reviews": self.rating_trust_ref_reviews,
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
    def _haversine_km(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Great-circle distance in km (vectorized over lat/lon arrays)."""
        rlat0 = np.radians(lat0)
        rlon0 = np.radians(lon0)
        rlat = np.radians(lat)
        rlon = np.radians(lon)
        dlat = rlat - rlat0
        dlon = rlon - rlon0
        a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat0) * np.cos(rlat) * np.sin(dlon / 2.0) ** 2
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arcsin(np.sqrt(a))
        return 6371.0 * c

    @staticmethod
    def _parse_price_tiers(meta: pd.DataFrame) -> np.ndarray:
        """Extract Yelp RestaurantsPriceRange2 (1–4) from attributes string; NaN if missing."""
        s = meta.get("attributes", pd.Series([""] * len(meta))).astype(str)
        extracted = s.str.extract(r"RestaurantsPriceRange2['\"]?:\s*['\"]?(\d)", expand=False)
        return pd.to_numeric(extracted, errors="coerce").to_numpy(dtype=float)

    @staticmethod
    def _stars_with_review_trust(
        stars_norm: np.ndarray, review_count: np.ndarray, ref_rc: float
    ) -> np.ndarray:
        """
        Shrink stars_norm toward 0.5 (avg on 1–5 scale) when review_count is small.
        ref_rc: ~count at which trust reaches 1 (log1p scale).
        """
        rc = np.maximum(review_count.astype(float), 0.0)
        trust = np.clip(np.log1p(rc) / np.log1p(max(ref_rc, 1.0)), 0.0, 1.0)
        return stars_norm * trust + 0.5 * (1.0 - trust)

    @staticmethod
    def _budget_target_tier(budget: Optional[str]) -> Optional[float]:
        if not budget:
            return None
        b = budget.strip().lower()
        if b == "cheap":
            return 1.0
        if b == "moderate":
            return 2.0
        if b == "expensive":
            return 4.0
        return None

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
        pool_k: Optional[int] = None,
        include_business_id: bool = False,
        budget: Optional[str] = None,
        ref_lat: Optional[float] = None,
        ref_lon: Optional[float] = None,
        max_radius_km: Optional[float] = None,
        w_semantic: float = 1.0,
        w_rating: float = 0.25,
        w_price: float = 0.2,
        w_distance: float = 0.25,
        w_popularity: float = 0.15,
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

        # Optional hard distance filter (reference point + max radius)
        dist_km_all = np.full(len(sim), np.nan, dtype=float)
        if ref_lat is not None and ref_lon is not None:
            lat = pd.to_numeric(index.meta["latitude"], errors="coerce").to_numpy(dtype=float)
            lon = pd.to_numeric(index.meta["longitude"], errors="coerce").to_numpy(dtype=float)
            valid_geo = np.isfinite(lat) & np.isfinite(lon)
            dist_km_all[valid_geo] = self._haversine_km(float(ref_lat), float(ref_lon), lat[valid_geo], lon[valid_geo])
            if max_radius_km is not None and float(max_radius_km) > 0:
                mask &= valid_geo & (dist_km_all <= float(max_radius_km))

        if mask.sum() == 0:
            # If filter becomes empty, return empty results (better than returning wrong states).
            cols = [
                "name",
                "address",
                "city",
                "state",
                "categories",
                "stars",
                "review_count",
                "similarity",
                "final_score",
                "latitude",
                "longitude",
                "distance_km",
                "price_tier",
                "price_match",
            ]
            if include_business_id:
                cols.insert(0, "business_id")
            return pd.DataFrame(columns=cols)

        idx = np.where(mask)[0]
        sim_c = sim[idx]
        stars_c = index.stars_norm[idx]
        rc = index.meta.iloc[idx]["review_count"].astype(float).to_numpy()
        stars_rank = self._stars_with_review_trust(stars_c, rc, self.rating_trust_ref_reviews)
        pop = np.log1p(np.maximum(rc, 0.0))
        pop_n = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)

        sim_n = (sim_c - sim_c.min()) / (sim_c.max() - sim_c.min() + 1e-9)

        # Price match vs budget (soft signal; missing tier -> neutral 0.5)
        tiers = self._parse_price_tiers(index.meta.iloc[idx])
        target_tier = self._budget_target_tier(budget)
        price_match = np.full(len(idx), 0.5, dtype=float)
        if target_tier is not None:
            known = np.isfinite(tiers)
            if known.any():
                diff = np.abs(tiers[known] - target_tier)
                price_match[known] = np.clip(1.0 - np.minimum(diff, 3.0) / 3.0, 0.0, 1.0)

        # Distance score (higher is better). No ref -> neutral 0.5
        dist_score = np.full(len(idx), 0.5, dtype=float)
        dist_c = dist_km_all[idx]
        if ref_lat is not None and ref_lon is not None and np.isfinite(dist_c).any():
            dmax = float(max_radius_km) if max_radius_km and float(max_radius_km) > 0 else 50.0
            finite = np.isfinite(dist_c)
            dist_score[finite] = np.clip(1.0 - dist_c[finite] / (dmax + 1e-9), 0.0, 1.0)

        # Multi-factor score (d1doc.md §3.5).
        # stars_rank: stars weighted by review trust (low counts shrink toward neutral).
        final_score = (
            w_semantic * sim_n
            + w_rating * stars_rank
            + w_price * price_match
            + w_distance * dist_score
            + w_popularity * pop_n
        )

        rank_k = max(top_k, int(pool_k)) if pool_k is not None else top_k
        rank_k = max(1, min(rank_k, len(idx)))

        if len(idx) <= rank_k:
            top_local = np.argsort(final_score)[::-1]
        else:
            top_local = np.argpartition(final_score, -rank_k)[-rank_k:]
            top_local = top_local[np.argsort(final_score[top_local])[::-1]]

        top_idx = idx[top_local]
        out = index.meta.iloc[top_idx].copy()
        out["similarity"] = sim[top_idx]
        out["final_score"] = final_score[top_local]
        out["distance_km"] = dist_km_all[top_idx]
        out["price_tier"] = tiers[top_local]
        out["price_match"] = price_match[top_local]
        out = out.sort_values("final_score", ascending=False).head(rank_k)

        cols = [
            "name",
            "address",
            "city",
            "state",
            "categories",
            "stars",
            "review_count",
            "similarity",
            "final_score",
            "latitude",
            "longitude",
            "distance_km",
            "price_tier",
            "price_match",
        ]
        if include_business_id and "business_id" in out.columns:
            cols = ["business_id"] + cols
        return out[cols]

