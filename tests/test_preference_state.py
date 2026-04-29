"""Tests for dining_retrieval/recommendation/preference_state.py."""
from __future__ import annotations

import pytest

from dining_retrieval.recommendation.preference_state import (
    UserPreferenceState,
    preference_from_session,
    session_dict_from_preference,
    toggle_dislike,
    toggle_like,
)


class TestToggleLike:
    def test_adds_business_to_liked(self) -> None:
        pref = UserPreferenceState()
        toggle_like(pref, "biz1")
        assert "biz1" in pref.liked_business_ids

    def test_does_not_add_duplicate_liked(self) -> None:
        pref = UserPreferenceState()
        toggle_like(pref, "biz1")
        toggle_like(pref, "biz1")
        assert pref.liked_business_ids.count("biz1") == 1

    def test_removes_from_disliked_when_liking(self) -> None:
        pref = UserPreferenceState(disliked_business_ids=["biz1"])
        toggle_like(pref, "biz1")
        assert "biz1" not in pref.disliked_business_ids
        assert "biz1" in pref.liked_business_ids

    def test_ignores_empty_string(self) -> None:
        pref = UserPreferenceState()
        toggle_like(pref, "")
        assert pref.liked_business_ids == []

    def test_ignores_whitespace_only_id(self) -> None:
        pref = UserPreferenceState()
        toggle_like(pref, "   ")
        assert pref.liked_business_ids == []

    def test_strips_and_uses_id(self) -> None:
        pref = UserPreferenceState()
        toggle_like(pref, "  biz1  ")
        assert "biz1" in pref.liked_business_ids

    def test_does_not_affect_other_liked_entries(self) -> None:
        pref = UserPreferenceState(liked_business_ids=["biz2", "biz3"])
        toggle_like(pref, "biz1")
        assert "biz2" in pref.liked_business_ids
        assert "biz3" in pref.liked_business_ids

    def test_coerces_id_to_string(self) -> None:
        pref = UserPreferenceState()
        toggle_like(pref, 123)  # type: ignore[arg-type]
        assert "123" in pref.liked_business_ids


class TestToggleDislike:
    def test_adds_business_to_disliked(self) -> None:
        pref = UserPreferenceState()
        toggle_dislike(pref, "biz1")
        assert "biz1" in pref.disliked_business_ids

    def test_does_not_add_duplicate_disliked(self) -> None:
        pref = UserPreferenceState()
        toggle_dislike(pref, "biz1")
        toggle_dislike(pref, "biz1")
        assert pref.disliked_business_ids.count("biz1") == 1

    def test_removes_from_liked_when_disliking(self) -> None:
        pref = UserPreferenceState(liked_business_ids=["biz1"])
        toggle_dislike(pref, "biz1")
        assert "biz1" not in pref.liked_business_ids
        assert "biz1" in pref.disliked_business_ids

    def test_ignores_empty_string(self) -> None:
        pref = UserPreferenceState()
        toggle_dislike(pref, "")
        assert pref.disliked_business_ids == []

    def test_ignores_whitespace_only_id(self) -> None:
        pref = UserPreferenceState()
        toggle_dislike(pref, "   ")
        assert pref.disliked_business_ids == []

    def test_does_not_affect_other_disliked_entries(self) -> None:
        pref = UserPreferenceState(disliked_business_ids=["biz2", "biz3"])
        toggle_dislike(pref, "biz1")
        assert "biz2" in pref.disliked_business_ids
        assert "biz3" in pref.disliked_business_ids

    def test_strips_and_uses_id(self) -> None:
        pref = UserPreferenceState()
        toggle_dislike(pref, "  biz1  ")
        assert "biz1" in pref.disliked_business_ids


class TestLikeDislikeMutualExclusion:
    def test_like_then_dislike_moves_between_lists(self) -> None:
        pref = UserPreferenceState()
        toggle_like(pref, "biz1")
        assert "biz1" in pref.liked_business_ids
        toggle_dislike(pref, "biz1")
        assert "biz1" not in pref.liked_business_ids
        assert "biz1" in pref.disliked_business_ids

    def test_dislike_then_like_moves_between_lists(self) -> None:
        pref = UserPreferenceState()
        toggle_dislike(pref, "biz1")
        assert "biz1" in pref.disliked_business_ids
        toggle_like(pref, "biz1")
        assert "biz1" not in pref.disliked_business_ids
        assert "biz1" in pref.liked_business_ids

    def test_multiple_businesses_independent(self) -> None:
        pref = UserPreferenceState()
        toggle_like(pref, "biz1")
        toggle_dislike(pref, "biz2")
        assert "biz1" in pref.liked_business_ids
        assert "biz2" in pref.disliked_business_ids
        assert "biz1" not in pref.disliked_business_ids
        assert "biz2" not in pref.liked_business_ids


class TestPreferenceFromSession:
    def test_empty_dict_gives_defaults(self) -> None:
        pref = preference_from_session({})
        assert pref.liked_business_ids == []
        assert pref.disliked_business_ids == []
        assert pref.preferred_cuisines == []
        assert pref.disliked_cuisines == []
        assert pref.price_preference is None
        assert pref.max_distance_km is None
        assert pref.min_rating is None

    def test_parses_liked_and_disliked_ids(self) -> None:
        pref = preference_from_session({
            "liked_business_ids": ["b1", "b2"],
            "disliked_business_ids": ["b3"],
        })
        assert pref.liked_business_ids == ["b1", "b2"]
        assert pref.disliked_business_ids == ["b3"]

    def test_filters_blank_ids(self) -> None:
        pref = preference_from_session({
            "liked_business_ids": ["b1", "", "  ", "b2"],
        })
        assert pref.liked_business_ids == ["b1", "b2"]

    def test_parses_cuisines(self) -> None:
        pref = preference_from_session({
            "preferred_cuisines": ["Italian", "Sushi"],
            "disliked_cuisines": ["Fast Food"],
        })
        assert pref.preferred_cuisines == ["Italian", "Sushi"]
        assert pref.disliked_cuisines == ["Fast Food"]

    def test_parses_price_preference(self) -> None:
        pref = preference_from_session({"price_preference": "moderate"})
        assert pref.price_preference == "moderate"

    def test_parses_max_distance(self) -> None:
        pref = preference_from_session({"max_distance_km": 5.0})
        assert pref.max_distance_km == 5.0

    def test_parses_min_rating(self) -> None:
        pref = preference_from_session({"min_rating": 4.0})
        assert pref.min_rating == 4.0

    def test_min_rating_coerced_to_float(self) -> None:
        pref = preference_from_session({"min_rating": "3.5"})
        assert pref.min_rating == 3.5
        assert isinstance(pref.min_rating, float)

    def test_none_liked_ids_treated_as_empty(self) -> None:
        pref = preference_from_session({"liked_business_ids": None})
        assert pref.liked_business_ids == []

    def test_ids_coerced_to_strings(self) -> None:
        pref = preference_from_session({"liked_business_ids": [1, 2, 3]})
        assert pref.liked_business_ids == ["1", "2", "3"]


class TestSessionDictRoundtrip:
    def test_roundtrip_preserves_liked_ids(self) -> None:
        pref = UserPreferenceState(liked_business_ids=["b1", "b2"])
        d = session_dict_from_preference(pref)
        restored = preference_from_session(d)
        assert restored.liked_business_ids == ["b1", "b2"]

    def test_roundtrip_preserves_min_rating(self) -> None:
        pref = UserPreferenceState(min_rating=4.5)
        d = session_dict_from_preference(pref)
        restored = preference_from_session(d)
        assert restored.min_rating == 4.5

    def test_roundtrip_preserves_cuisines(self) -> None:
        pref = UserPreferenceState(
            preferred_cuisines=["Italian"],
            disliked_cuisines=["Fast Food"],
        )
        d = session_dict_from_preference(pref)
        restored = preference_from_session(d)
        assert restored.preferred_cuisines == ["Italian"]
        assert restored.disliked_cuisines == ["Fast Food"]

    def test_session_dict_contains_all_fields(self) -> None:
        pref = UserPreferenceState()
        d = session_dict_from_preference(pref)
        expected_keys = {
            "liked_business_ids",
            "disliked_business_ids",
            "preferred_cuisines",
            "disliked_cuisines",
            "price_preference",
            "max_distance_km",
            "min_rating",
        }
        assert set(d.keys()) == expected_keys
