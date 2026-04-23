"""
检索服务薄封装：TouristRetrieval + parse_query（与前端 `/search` 请求体一致）。
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dining_retrieval.core.retrieval import RestaurantSearchIndex, TouristRetrieval
from dining_retrieval.recommendation.preference_state import UserPreferenceState
from dining_retrieval.recommendation.reranker import rerank_pool
from dining_retrieval.core.yelp_photos import (
    load_business_photo_ids,
    resolve_photos_json,
    yelp_bphoto_cdn_url,
)
from dining_retrieval.search.query_parser import ParsedQuery, extract_budget_hint, parse_query

# 无 Yelp photo_id 时的占位图（与前端 @error 回退使用同一组，便于一致）
_FALLBACK_FOOD_IMAGES: tuple[str, ...] = (
    "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=320&h=200&fit=crop&q=80",
    "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=320&h=200&fit=crop&q=80",
    "https://images.unsplash.com/photo-1414235077428-338989a841e3?w=320&h=200&fit=crop&q=80",
    "https://images.unsplash.com/photo-1466978913421-dad2ebd01d17?w=320&h=200&fit=crop&q=80",
    "https://images.unsplash.com/photo-1552566626-52f8b828add9?w=320&h=200&fit=crop&q=80",
)


def _fallback_photo_url(business_id: str) -> str:
    key = (business_id or "unknown").encode()
    h = int(hashlib.md5(key).hexdigest()[:8], 16)
    return _FALLBACK_FOOD_IMAGES[h % len(_FALLBACK_FOOD_IMAGES)]


def _nl_cuisine_map() -> dict[str, str]:
    return {
        "sushi": "Sushi",
        "japanese": "Sushi",
        "steakhouse": "Steakhouse",
        "steak": "Steakhouse",
        "korean": "Korean",
        "chinese": "Chinese",
        "fast food": "Fast Food",
        "burger": "Burger",
        "healthy": "Healthy",
        "salad": "Healthy",
        "vegan": "Healthy",
        "vegetarian": "Healthy",
    }


def _compose_query_text(
    parsed: ParsedQuery,
    keywords_extra: Optional[str],
    effective_cuisines: Optional[List[str]],
) -> str:
    semantic_parts: list[str] = []
    if parsed.semantic_query and str(parsed.semantic_query).strip():
        semantic_parts.append(str(parsed.semantic_query).strip())
    if keywords_extra and str(keywords_extra).strip():
        semantic_parts.append(str(keywords_extra).strip())
    if effective_cuisines:
        semantic_parts.extend([str(c) for c in effective_cuisines if str(c).strip()])
    seen: set[str] = set()
    ordered: list[str] = []
    for p in semantic_parts:
        p = str(p).strip()
        if not p or p in seen:
            continue
        seen.add(p)
        ordered.append(p)
    query_text = " ".join(ordered).strip()
    if not query_text:
        query_text = (parsed.raw or "").strip() or "restaurants"
    return query_text


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or len(df) == 0:
        return []
    out = df.replace({np.nan: None})
    return out.to_dict(orient="records")


class RetrievalSearchService:
    def __init__(self, repo_root: Optional[Path] = None):
        # backend/services/... → 仓库根
        self.repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
        self._retrieval = TouristRetrieval(
            data_dir=self.repo_root / "data" / "cleaned",
            index_dir=self.repo_root / "models" / "artifacts",
            max_businesses=20000,
            max_reviews_per_business=10,
            restrict_index_cities=True,
            rating_trust_ref_reviews=150.0,
        )
        self._index: Optional[RestaurantSearchIndex] = None
        self._photo_ids_by_business: Optional[dict[str, list[str]]] = None

    def _business_photo_map(self) -> dict[str, list[str]]:
        if self._photo_ids_by_business is None:
            self._photo_ids_by_business = load_business_photo_ids(
                resolve_photos_json(self.repo_root),
                max_per_business=1,
            )
        return self._photo_ids_by_business

    def _attach_photo_urls(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return df
        out = df.copy()
        pmap = self._business_photo_map()
        urls: list[str] = []
        if "business_id" in out.columns:
            for bid in out["business_id"].astype(str):
                ids = pmap.get(bid, [])
                if ids:
                    try:
                        urls.append(yelp_bphoto_cdn_url(ids[0]))
                    except ValueError:
                        urls.append(_fallback_photo_url(bid))
                else:
                    urls.append(_fallback_photo_url(bid))
        else:
            urls = [_fallback_photo_url("") for _ in range(len(out))]
        out["photo_url"] = urls
        return out

    def load_index(self, force_rebuild: bool = False) -> RestaurantSearchIndex:
        self._index = self._retrieval.build_or_load_index(force_rebuild=force_rebuild)
        return self._index

    def ensure_index(self) -> RestaurantSearchIndex:
        if self._index is None:
            return self.load_index(False)
        return self._index

    def search(
        self,
        *,
        query: str = "",
        state: str,
        city: Optional[str] = None,
        top_k: int = 10,
        pool_k: Optional[int] = None,
        keywords_extra: Optional[str] = None,
        force_rebuild_index: bool = False,
        discover_only: bool = False,
        cuisines: Optional[List[str]] = None,
        w_semantic: float = 0.85,
        w_rating: float = 1.05,
        w_price: float = 0.15,
        w_distance: float = 0.2,
        w_popularity: float = 0.1,
        liked_business_ids: Optional[List[str]] = None,
        disliked_business_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        对齐前端流程：Discover（泛检索）或 Refine（NL + 菜系 + 权重 + pool），
        可选根据 👍/👎 在候选池内 v2 重排。
        """
        index = self.load_index(force_rebuild=force_rebuild_index)
        semantic_state_note = ""

        if discover_only:
            parsed = ParsedQuery(raw="")
            query_text = "restaurants"
            effective_cuisines = None
        else:
            parsed = parse_query(query or "")
            constraint = " ".join(
                x for x in (query or "", keywords_extra or "") if str(x).strip()
            ).strip()
            budget_hint = extract_budget_hint(constraint) if constraint else None
            if budget_hint and not parsed.budget:
                parsed = replace(parsed, budget=budget_hint)

            nl_map = _nl_cuisine_map()
            cuisine_from_nl: list[str] = []
            if parsed.cuisine and parsed.cuisine in nl_map:
                cuisine_from_nl.append(nl_map[parsed.cuisine])
            merged = list(dict.fromkeys(list(cuisines or []) + cuisine_from_nl))
            effective_cuisines = merged or None

            query_text = _compose_query_text(parsed, keywords_extra, effective_cuisines)
            if not query_text.strip():
                query_text = (query or "").strip() or "restaurants"

        st = str(state).strip().upper()
        if not st or st == "ALL":
            raise ValueError("state 不能为空（请传 USPS 两字母州码，如 PA）。")

        city_f = city.strip() if city and str(city).strip() else None
        pk = int(pool_k) if pool_k is not None else 45
        pool_eff = max(int(top_k), pk)

        pool_df = self._retrieval.recommend_keywords(
            keywords=query_text,
            index=index,
            state=st,
            city=city_f,
            cuisines=effective_cuisines,
            top_k=int(top_k),
            pool_k=pool_eff,
            include_business_id=True,
            budget=parsed.budget,
            ref_lat=parsed.ref_lat,
            ref_lon=parsed.ref_lon,
            max_radius_km=parsed.radius_km,
            w_semantic=w_semantic,
            w_rating=w_rating,
            w_price=w_price,
            w_distance=w_distance,
            w_popularity=w_popularity,
        )

        likes = [str(x).strip() for x in (liked_business_ids or []) if str(x).strip()]
        dislikes = [str(x).strip() for x in (disliked_business_ids or []) if str(x).strip()]
        pref = UserPreferenceState(liked_business_ids=likes, disliked_business_ids=dislikes)
        interactive_on = bool(likes or dislikes)
        if interactive_on:
            ranked = rerank_pool(pool_df, index, pref)
            result_df = ranked.head(int(top_k))
        else:
            result_df = pool_df.head(int(top_k))

        result_df = self._attach_photo_urls(result_df)

        meta: Dict[str, Any] = {
            "query_text": query_text,
            "parsed": parsed.to_dict(),
            "discover_only": discover_only,
            "pool_rows": int(len(pool_df)),
            "pool_k": pool_eff,
            "reranked": interactive_on,
            "semantic_state_note": semantic_state_note,
        }
        return _df_to_records(result_df), meta
