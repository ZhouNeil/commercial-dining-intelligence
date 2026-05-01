"""
Thin service wrapper around TouristRetrieval + parse_query, aligned with the frontend `/search` request body.
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
from models.rl_feedback_loop import RLFeedbackLoop, classify_query_intent
from dining_retrieval.core.yelp_photos import (
    load_business_photo_ids,
    resolve_photos_json,
    yelp_bphoto_cdn_url,
)
from dining_retrieval.search.query_parser import ParsedQuery, extract_budget_hint, parse_query
from dining_retrieval.core.geocoder import geocode_address

# Fallback images used when no Yelp photo_id is available (same set as the frontend @error fallback).
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


def _strategy_label(arm_name: Optional[str], intent_name: Optional[str]) -> str:
    if arm_name == "convenience":
        return "Detected intent: Quick Meal. Prioritizing: Convenience."
    if arm_name == "reputation":
        return "Detected intent: Special Occasion. Prioritizing: Reputation."
    if arm_name == "explorer":
        return "Detected intent: Discovery. Prioritizing: Explorer picks."
    if intent_name == "intent_quick":
        return "Detected intent: Quick Meal. Manual weights active."
    if intent_name == "intent_romantic":
        return "Detected intent: Romantic Dining. Manual weights active."
    return "Detected intent: General Search. Manual weights active."


class RetrievalSearchService:
    def __init__(self, repo_root: Optional[Path] = None):
        # backend/services/... → repo root
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
        self._rl_engine = self._create_rl_engine()

    _GALLERY_MAX = 8
    _FALLBACK_GALLERY = 4

    # Keep presets centralized so RL can steer the same retrieval formula
    # without changing the downstream TF-IDF or v2 reranker behavior.
    _RL_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
        "explorer": {
            "w_semantic": 1.35,
            "w_rating": 0.65,
            "w_price": 0.15,
            "w_distance": 0.15,
            "w_popularity": 0.2,
        },
        "reputation": {
            "w_semantic": 0.75,
            "w_rating": 1.45,
            "w_price": 0.15,
            "w_distance": 0.1,
            "w_popularity": 0.3,
        },
        "convenience": {
            "w_semantic": 0.7,
            "w_rating": 0.85,
            "w_price": 0.2,
            "w_distance": 1.45,
            "w_popularity": 0.15,
        },
    }

    def _create_rl_engine(self) -> Optional[RLFeedbackLoop]:
        try:
            return RLFeedbackLoop()
        except Exception:  # noqa: BLE001 - search should degrade gracefully if RL init fails.
            return None

    def _normalized_weights(
        self,
        *,
        w_semantic: float,
        w_rating: float,
        w_price: float,
        w_distance: float,
        w_popularity: float,
    ) -> dict[str, float]:
        return {
            "w_semantic": float(w_semantic),
            "w_rating": float(w_rating),
            "w_price": float(w_price),
            "w_distance": float(w_distance),
            "w_popularity": float(w_popularity),
        }

    def _preset_for_arm(self, arm_name: str) -> dict[str, float]:
        return dict(self._RL_WEIGHT_PRESETS.get(arm_name, self._RL_WEIGHT_PRESETS["explorer"]))

    def _reward_for_action(self, action_name: str) -> Optional[float]:
        # Normalize the UI events into a tiny, explicit reward surface for v1 RL.
        if action_name in {"detail_open", "like"}:
            return 1.0
        if action_name == "pass":
            return -0.5
        if action_name in {"refresh", "slider_override"}:
            return -0.1
        return None

    def _log_rl_feedback(
        self,
        *,
        rl_prev_selected_arm: Optional[str],
        rl_prev_intent_name: Optional[str],
        rl_action_events: Optional[List[Dict[str, Any]]],
    ) -> int:
        if self._rl_engine is None or not rl_prev_selected_arm or not rl_prev_intent_name:
            return 0

        logged = 0
        for event in rl_action_events or []:
            if not isinstance(event, dict):
                continue
            reward = self._reward_for_action(str(event.get("action") or "").strip())
            if reward is None:
                continue
            self._rl_engine.log_user_feedback(
                rl_prev_selected_arm,
                reward,
                rl_prev_intent_name,
                query=str(event.get("query_text") or ""),
            )
            logged += 1
        return logged

    def _business_photo_map(self) -> dict[str, list[str]]:
        if self._photo_ids_by_business is None:
            self._photo_ids_by_business = load_business_photo_ids(
                resolve_photos_json(self.repo_root),
                max_per_business=self._GALLERY_MAX,
            )
        return self._photo_ids_by_business

    def _attach_photo_urls(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return df
        out = df.copy()
        pmap = self._business_photo_map()
        primary: list[str] = []
        galleries: list[list[str]] = []
        if "business_id" in out.columns:
            for bid in out["business_id"].astype(str):
                ids = pmap.get(bid, [])
                g: list[str] = []
                for pid in ids[: self._GALLERY_MAX]:
                    try:
                        g.append(yelp_bphoto_cdn_url(pid))
                    except ValueError:
                        continue
                if not g:
                    for j in range(self._FALLBACK_GALLERY):
                        g.append(_fallback_photo_url(f"{bid}|{j}"))
                else:
                    g = g[: self._GALLERY_MAX]
                galleries.append(g)
                primary.append(g[0] if g else _fallback_photo_url(bid))
        else:
            for _ in range(len(out)):
                galleries.append([_fallback_photo_url("") for j in range(self._FALLBACK_GALLERY)])
            primary = [g[0] for g in galleries]
        out["photo_url"] = primary
        out["photo_urls"] = galleries
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
        user_location: Optional[str] = None,
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
        rl_enabled: bool = True,
        rl_user_overrode: bool = False,
        rl_prev_selected_arm: Optional[str] = None,
        rl_prev_intent_name: Optional[str] = None,
        rl_action_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Aligned with the frontend flow: Discover (broad retrieval) or Refine (NL + cuisines + weights + pool),
        with optional v2 re-ranking within the candidate pool based on likes/dislikes.
        """
        index = self.load_index(force_rebuild=force_rebuild_index)
        semantic_state_note = ""
        rl_feedback_logged = 0
        if rl_enabled:
            try:
                rl_feedback_logged = self._log_rl_feedback(
                    rl_prev_selected_arm=rl_prev_selected_arm,
                    rl_prev_intent_name=rl_prev_intent_name,
                    rl_action_events=rl_action_events,
                )
            except Exception:  # noqa: BLE001 - RL feedback should never break search results.
                rl_feedback_logged = 0

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

        if parsed.ref_lat is None and parsed.ref_lon is None and user_location and str(user_location).strip():
            geo_res = geocode_address(str(user_location).strip())
            if geo_res:
                parsed = replace(parsed, ref_lat=geo_res[0], ref_lon=geo_res[1])

        st = str(state).strip().upper()
        if not st or st == "ALL":
            raise ValueError("state is required (USPS two-letter code, e.g. PA).")

        city_f = city.strip() if city and str(city).strip() else None
        pk = int(pool_k) if pool_k is not None else 45
        pool_eff = max(int(top_k), pk)
        user_weights = self._normalized_weights(
            w_semantic=w_semantic,
            w_rating=w_rating,
            w_price=w_price,
            w_distance=w_distance,
            w_popularity=w_popularity,
        )

        rl_intent_name = classify_query_intent(query_text)
        rl_selected_arm: Optional[str] = None
        rl_applied = False
        if rl_enabled and not rl_user_overrode and self._rl_engine is not None:
            try:
                rl_selected_arm = self._rl_engine.select_strategy(rl_intent_name)
                effective_weights = self._preset_for_arm(rl_selected_arm)
                rl_applied = True
            except Exception:  # noqa: BLE001 - fall back to manual weights if RL selection fails.
                effective_weights = user_weights
                rl_selected_arm = None
        else:
            effective_weights = user_weights

        # Once the user touches a slider, the manual request weights win for that round.
        if rl_user_overrode:
            effective_weights = user_weights

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
            w_semantic=effective_weights["w_semantic"],
            w_rating=effective_weights["w_rating"],
            w_price=effective_weights["w_price"],
            w_distance=effective_weights["w_distance"],
            w_popularity=effective_weights["w_popularity"],
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
            "semantic_match_count": int((result_df["similarity"] > 0.0).sum()) if "similarity" in result_df.columns else 0,
            "reranked": interactive_on,
            "semantic_state_note": semantic_state_note,
            "rl_applied": rl_applied,
            "rl_intent_name": rl_intent_name,
            "rl_selected_arm": rl_selected_arm,
            "rl_strategy_label": _strategy_label(rl_selected_arm, rl_intent_name),
            "rl_effective_weights": effective_weights,
            "rl_user_override_active": bool(rl_user_overrode),
            "rl_feedback_logged": rl_feedback_logged,
        }
        return _df_to_records(result_df), meta
