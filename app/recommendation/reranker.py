from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from app.core.retrieval import RestaurantSearchIndex
from app.recommendation.preference_state import UserPreferenceState


def _row_index_for_business_id(index: RestaurantSearchIndex, business_id: str) -> Optional[int]:
    bids = index.restaurant_ids
    for i, rid in enumerate(bids):
        if str(rid) == str(business_id):
            return int(i)
    return None


def _cosine_rows(M: csr_matrix, norms: np.ndarray, i: int, j: int) -> float:
    """Cosine similarity between TF-IDF rows i and j (same space as retrieval)."""
    if i < 0 or j < 0 or i >= M.shape[0] or j >= M.shape[0]:
        return 0.0
    dot = float(M.getrow(i).dot(M.getrow(j).T).A[0, 0])
    denom = float(norms[i]) * float(norms[j]) + 1e-9
    return dot / denom


def _avg_sim_to_businesses(
    M: csr_matrix,
    norms: np.ndarray,
    row_i: int,
    target_business_ids: list[str],
    index: RestaurantSearchIndex,
) -> float:
    sims: list[float] = []
    for bid in target_business_ids:
        j = _row_index_for_business_id(index, bid)
        if j is None:
            continue
        sims.append(_cosine_rows(M, norms, row_i, j))
    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))


def _minmax(x: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)


def _preference_row_score(
    row: pd.Series,
    pref: UserPreferenceState,
    categories_norm: str,
) -> float:
    """Heuristic [0,1] match to structured preferences (no training)."""
    parts: list[float] = []

    if pref.min_rating is not None:
        try:
            stars = float(row.get("stars", 0))
            parts.append(1.0 if stars >= float(pref.min_rating) else 0.0)
        except (TypeError, ValueError):
            parts.append(0.5)

    if pref.max_distance_km is not None:
        dist = row.get("distance_km")
        try:
            if dist is not None and float(dist) == float(dist):
                d = float(dist)
                cap = float(pref.max_distance_km)
                parts.append(1.0 if d <= cap else max(0.0, 1.0 - (d - cap) / (cap + 1e-9)))
            else:
                parts.append(0.5)
        except (TypeError, ValueError):
            parts.append(0.5)

    for dc in pref.disliked_cuisines:
        k = str(dc).strip().lower()
        if k and k in categories_norm:
            parts.append(0.0)
        elif k:
            parts.append(1.0)

    for pc in pref.preferred_cuisines:
        k = str(pc).strip().lower()
        if k and k in categories_norm:
            parts.append(1.0)
        elif k:
            parts.append(0.5)

    if not parts:
        return 0.5
    return float(sum(parts) / len(parts))


def rerank_pool(
    pool_df: pd.DataFrame,
    index: RestaurantSearchIndex,
    pref: UserPreferenceState,
    *,
    w_base: float = 1.0,
    w_like: float = 0.35,
    w_dislike: float = 0.25,
    w_pref: float = 0.2,
) -> pd.DataFrame:
    """
    Re-rank rows in pool_df using v1 final_score plus like/dislike TF-IDF similarity.

    pool_df must include `business_id` and columns used by preference heuristics.
    """
    if pool_df.empty or "business_id" not in pool_df.columns:
        return pool_df.copy()

    out = pool_df.copy()
    n = len(out)
    M = index.restaurant_matrix
    norms = index.restaurant_norms

    base = pd.to_numeric(out["final_score"], errors="coerce").to_numpy(dtype=float)
    base = np.nan_to_num(base, nan=0.0)
    base_n = _minmax(base)

    sim_liked = np.zeros(n, dtype=float)
    sim_disliked = np.zeros(n, dtype=float)
    row_indices: list[Optional[int]] = []
    for bid in out["business_id"].astype(str).tolist():
        row_indices.append(_row_index_for_business_id(index, bid))

    if pref.liked_business_ids:
        for i in range(n):
            ri = row_indices[i]
            if ri is None:
                continue
            sim_liked[i] = _avg_sim_to_businesses(M, norms, ri, pref.liked_business_ids, index)
        sim_liked = _minmax(sim_liked)
    else:
        sim_liked = np.zeros(n, dtype=float)

    if pref.disliked_business_ids:
        for i in range(n):
            ri = row_indices[i]
            if ri is None:
                continue
            sim_disliked[i] = _avg_sim_to_businesses(
                M, norms, ri, pref.disliked_business_ids, index
            )
        sim_disliked = _minmax(sim_disliked)
    else:
        sim_disliked = np.zeros(n, dtype=float)

    cats = out.get("categories", pd.Series([""] * n)).astype(str).str.lower()
    pref_scores = np.array([_preference_row_score(out.iloc[i], pref, cats.iloc[i]) for i in range(n)])

    v2 = (
        w_base * base_n
        + w_like * sim_liked
        - w_dislike * sim_disliked
        + w_pref * pref_scores
    )
    out = out.assign(
        v2_score=v2,
        v2_sim_liked=sim_liked,
        v2_sim_disliked=sim_disliked,
        v2_pref_match=pref_scores,
    )
    return out.sort_values("v2_score", ascending=False).reset_index(drop=True)
