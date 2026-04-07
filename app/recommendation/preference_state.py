from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class UserPreferenceState:
    """Session-level aggregates for v2 interactive re-ranking."""

    liked_business_ids: list[str] = field(default_factory=list)
    disliked_business_ids: list[str] = field(default_factory=list)
    preferred_cuisines: list[str] = field(default_factory=list)
    disliked_cuisines: list[str] = field(default_factory=list)
    price_preference: Optional[str] = None  # cheap | moderate | expensive
    max_distance_km: Optional[float] = None
    min_rating: Optional[float] = None


def preference_from_session(raw: dict[str, Any]) -> UserPreferenceState:
    """Build state from Streamlit session_state dict (JSON-serializable lists only)."""
    return UserPreferenceState(
        liked_business_ids=[str(x) for x in raw.get("liked_business_ids") or [] if str(x).strip()],
        disliked_business_ids=[
            str(x) for x in raw.get("disliked_business_ids") or [] if str(x).strip()
        ],
        preferred_cuisines=[str(x) for x in raw.get("preferred_cuisines") or [] if str(x).strip()],
        disliked_cuisines=[str(x) for x in raw.get("disliked_cuisines") or [] if str(x).strip()],
        price_preference=raw.get("price_preference"),
        max_distance_km=raw.get("max_distance_km"),
        min_rating=float(raw["min_rating"]) if raw.get("min_rating") is not None else None,
    )


def session_dict_from_preference(pref: UserPreferenceState) -> dict[str, Any]:
    return asdict(pref)


def toggle_like(pref: UserPreferenceState, business_id: str) -> None:
    bid = str(business_id).strip()
    if not bid:
        return
    if bid in pref.disliked_business_ids:
        pref.disliked_business_ids = [x for x in pref.disliked_business_ids if x != bid]
    if bid not in pref.liked_business_ids:
        pref.liked_business_ids.append(bid)


def toggle_dislike(pref: UserPreferenceState, business_id: str) -> None:
    bid = str(business_id).strip()
    if not bid:
        return
    if bid in pref.liked_business_ids:
        pref.liked_business_ids = [x for x in pref.liked_business_ids if x != bid]
    if bid not in pref.disliked_business_ids:
        pref.disliked_business_ids.append(bid)
