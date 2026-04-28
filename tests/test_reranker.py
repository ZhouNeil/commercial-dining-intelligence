"""Tests for dining_retrieval/recommendation/reranker.py."""
from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from dining_retrieval.recommendation.preference_state import UserPreferenceState
from dining_retrieval.recommendation.reranker import (
    _avg_sim_to_businesses,
    _build_bid_to_row,
    _cosine_rows,
    _minmax,
    _preference_row_score,
    rerank_pool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index(n: int = 5):
    """Build a minimal RestaurantSearchIndex stub with an n×n identity matrix."""
    ids = np.array([f"b{i}" for i in range(n)], dtype=str)
    M = csr_matrix(np.eye(n, dtype=float))
    norms = np.sqrt(M.power(2).sum(axis=1)).A1
    meta = pd.DataFrame({
        "business_id": ids,
        "name": [f"Restaurant {i}" for i in range(n)],
        "stars": [3.0 + i * 0.5 for i in range(n)],
        "categories": ["Pizza"] * n,
    })
    index = MagicMock()
    index.restaurant_ids = ids
    index.restaurant_matrix = M
    index.restaurant_norms = norms
    index.meta = meta
    index.stars_norm = np.linspace(0.3, 0.9, n)
    return index


def _make_pool(n: int = 4, *, ids: Optional[list[str]] = None) -> pd.DataFrame:
    """Build a minimal pool DataFrame compatible with rerank_pool."""
    bids = ids if ids is not None else [f"b{i}" for i in range(n)]
    return pd.DataFrame({
        "business_id": bids,
        "name": [f"Place {i}" for i in range(n)],
        "final_score": [0.9 - i * 0.1 for i in range(n)],
        "stars": [4.5 - i * 0.2 for i in range(n)],
        "categories": ["Italian", "Sushi", "Pizza", "Burger"][:n],
        "distance_km": [0.5 + i * 0.5 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# _build_bid_to_row
# ---------------------------------------------------------------------------

class TestBuildBidToRow:
    def test_returns_dict(self) -> None:
        index = _make_index(3)
        result = _build_bid_to_row(index)
        assert isinstance(result, dict)

    def test_maps_each_id_to_its_position(self) -> None:
        index = _make_index(4)
        result = _build_bid_to_row(index)
        assert result["b0"] == 0
        assert result["b1"] == 1
        assert result["b2"] == 2
        assert result["b3"] == 3

    def test_all_ids_present(self) -> None:
        n = 6
        index = _make_index(n)
        result = _build_bid_to_row(index)
        assert len(result) == n
        for i in range(n):
            assert f"b{i}" in result

    def test_empty_index(self) -> None:
        index = _make_index(0)
        index.restaurant_ids = np.array([], dtype=str)
        result = _build_bid_to_row(index)
        assert result == {}

    def test_values_are_ints(self) -> None:
        index = _make_index(3)
        result = _build_bid_to_row(index)
        for v in result.values():
            assert isinstance(v, int)

    def test_keys_are_strings(self) -> None:
        index = _make_index(3)
        result = _build_bid_to_row(index)
        for k in result.keys():
            assert isinstance(k, str)

    def test_unknown_id_not_in_dict(self) -> None:
        index = _make_index(3)
        result = _build_bid_to_row(index)
        assert "bXXX" not in result

    def test_single_entry_index(self) -> None:
        index = _make_index(1)
        result = _build_bid_to_row(index)
        assert result == {"b0": 0}


# ---------------------------------------------------------------------------
# _cosine_rows
# ---------------------------------------------------------------------------

class TestCosineRows:
    def setup_method(self) -> None:
        self.index = _make_index(4)
        self.M = self.index.restaurant_matrix
        self.norms = self.index.restaurant_norms

    def test_identical_row_similarity_is_one(self) -> None:
        sim = _cosine_rows(self.M, self.norms, 0, 0)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_rows_similarity_is_zero(self) -> None:
        sim = _cosine_rows(self.M, self.norms, 0, 1)
        assert abs(sim) < 1e-6

    def test_symmetric(self) -> None:
        sim_01 = _cosine_rows(self.M, self.norms, 0, 1)
        sim_10 = _cosine_rows(self.M, self.norms, 1, 0)
        assert abs(sim_01 - sim_10) < 1e-9

    def test_negative_index_returns_zero(self) -> None:
        sim = _cosine_rows(self.M, self.norms, -1, 0)
        assert sim == 0.0

    def test_out_of_bounds_i_returns_zero(self) -> None:
        sim = _cosine_rows(self.M, self.norms, 999, 0)
        assert sim == 0.0

    def test_out_of_bounds_j_returns_zero(self) -> None:
        sim = _cosine_rows(self.M, self.norms, 0, 999)
        assert sim == 0.0

    def test_returns_float(self) -> None:
        sim = _cosine_rows(self.M, self.norms, 0, 0)
        assert isinstance(sim, float)

    def test_similarity_in_valid_range(self) -> None:
        n = self.M.shape[0]
        for i in range(n):
            for j in range(n):
                sim = _cosine_rows(self.M, self.norms, i, j)
                assert -1.0 - 1e-9 <= sim <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# _avg_sim_to_businesses
# ---------------------------------------------------------------------------

class TestAvgSimToBusinesses:
    def setup_method(self) -> None:
        self.index = _make_index(5)
        self.M = self.index.restaurant_matrix
        self.norms = self.index.restaurant_norms
        self.bid_to_row = _build_bid_to_row(self.index)

    def test_empty_target_list_returns_zero(self) -> None:
        result = _avg_sim_to_businesses(self.M, self.norms, 0, [], self.bid_to_row)
        assert result == 0.0

    def test_self_similarity_is_one(self) -> None:
        result = _avg_sim_to_businesses(self.M, self.norms, 0, ["b0"], self.bid_to_row)
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_target_returns_zero(self) -> None:
        result = _avg_sim_to_businesses(self.M, self.norms, 0, ["b1"], self.bid_to_row)
        assert abs(result) < 1e-6

    def test_unknown_target_id_skipped(self) -> None:
        result = _avg_sim_to_businesses(self.M, self.norms, 0, ["unknown_id"], self.bid_to_row)
        assert result == 0.0

    def test_mix_of_known_and_unknown_ids(self) -> None:
        result = _avg_sim_to_businesses(self.M, self.norms, 0, ["b0", "unknown"], self.bid_to_row)
        assert abs(result - 1.0) < 1e-6

    def test_average_over_multiple_targets(self) -> None:
        result = _avg_sim_to_businesses(self.M, self.norms, 0, ["b0", "b1"], self.bid_to_row)
        # sim(0,0)=1.0, sim(0,1)=0.0 → average = 0.5
        assert abs(result - 0.5) < 1e-6

    def test_all_unknowns_returns_zero(self) -> None:
        result = _avg_sim_to_businesses(self.M, self.norms, 0, ["x1", "x2", "x3"], self.bid_to_row)
        assert result == 0.0

    def test_returns_float(self) -> None:
        result = _avg_sim_to_businesses(self.M, self.norms, 0, ["b0"], self.bid_to_row)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _minmax
# ---------------------------------------------------------------------------

class TestMinmax:
    def test_all_same_values_returns_half(self) -> None:
        x = np.array([5.0, 5.0, 5.0])
        result = _minmax(x)
        assert np.allclose(result, 0.5)

    def test_range_is_zero_to_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _minmax(x)
        assert abs(result.min() - 0.0) < 1e-9
        assert abs(result.max() - 1.0) < 1e-9

    def test_preserves_order(self) -> None:
        x = np.array([1.0, 3.0, 2.0])
        result = _minmax(x)
        assert result[0] < result[2] < result[1]

    def test_single_element_returns_half(self) -> None:
        x = np.array([42.0])
        result = _minmax(x)
        assert abs(result[0] - 0.5) < 1e-9

    def test_handles_zeros(self) -> None:
        x = np.array([0.0, 0.5, 1.0])
        result = _minmax(x)
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[2] - 1.0) < 1e-9

    def test_negative_values(self) -> None:
        x = np.array([-2.0, 0.0, 2.0])
        result = _minmax(x)
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[1] - 0.5) < 1e-9
        assert abs(result[2] - 1.0) < 1e-9

    def test_output_dtype_is_float(self) -> None:
        x = np.array([1, 2, 3], dtype=int)
        result = _minmax(x)
        assert result.dtype == float or np.issubdtype(result.dtype, np.floating)

    def test_two_elements(self) -> None:
        x = np.array([0.0, 10.0])
        result = _minmax(x)
        assert abs(result[0] - 0.0) < 1e-9
        assert abs(result[1] - 1.0) < 1e-9

    def test_large_array_stays_bounded(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.uniform(-1000, 1000, size=1000)
        result = _minmax(x)
        assert result.min() >= 0.0 - 1e-9
        assert result.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# _preference_row_score
# ---------------------------------------------------------------------------

class TestPreferenceRowScore:
    def _row(self, **kwargs) -> pd.Series:
        defaults = {"stars": 4.0, "distance_km": 1.0, "categories": "italian"}
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_no_preferences_returns_half(self) -> None:
        pref = UserPreferenceState()
        score = _preference_row_score(self._row(), pref, "italian")
        assert abs(score - 0.5) < 1e-9

    def test_min_rating_met_contributes_one(self) -> None:
        pref = UserPreferenceState(min_rating=3.5)
        score = _preference_row_score(self._row(stars=4.0), pref, "italian")
        assert score == 1.0

    def test_min_rating_not_met_contributes_zero(self) -> None:
        pref = UserPreferenceState(min_rating=4.5)
        score = _preference_row_score(self._row(stars=3.0), pref, "italian")
        assert score == 0.0

    def test_distance_within_cap_contributes_one(self) -> None:
        pref = UserPreferenceState(max_distance_km=5.0)
        score = _preference_row_score(self._row(distance_km=2.0), pref, "italian")
        assert score == 1.0

    def test_distance_beyond_cap_penalized(self) -> None:
        pref = UserPreferenceState(max_distance_km=5.0)
        score = _preference_row_score(self._row(distance_km=10.0), pref, "italian")
        assert 0.0 <= score < 1.0

    def test_distance_none_gets_neutral_half(self) -> None:
        pref = UserPreferenceState(max_distance_km=5.0)
        score = _preference_row_score(self._row(distance_km=None), pref, "italian")
        assert abs(score - 0.5) < 1e-9

    def test_liked_cuisine_present_contributes_one(self) -> None:
        pref = UserPreferenceState(preferred_cuisines=["italian"])
        score = _preference_row_score(self._row(), pref, "italian pizza")
        assert score == 1.0

    def test_liked_cuisine_absent_contributes_half(self) -> None:
        pref = UserPreferenceState(preferred_cuisines=["sushi"])
        score = _preference_row_score(self._row(), pref, "italian pizza")
        assert abs(score - 0.5) < 1e-9

    def test_disliked_cuisine_present_contributes_zero(self) -> None:
        pref = UserPreferenceState(disliked_cuisines=["italian"])
        score = _preference_row_score(self._row(), pref, "italian pizza")
        assert score == 0.0

    def test_disliked_cuisine_absent_contributes_one(self) -> None:
        pref = UserPreferenceState(disliked_cuisines=["sushi"])
        score = _preference_row_score(self._row(), pref, "italian pizza")
        assert score == 1.0

    def test_result_bounded_between_zero_and_one(self) -> None:
        pref = UserPreferenceState(
            min_rating=4.0,
            max_distance_km=2.0,
            preferred_cuisines=["italian"],
            disliked_cuisines=["sushi"],
        )
        score = _preference_row_score(self._row(stars=5.0, distance_km=0.5), pref, "italian")
        assert 0.0 <= score <= 1.0

    def test_invalid_stars_gets_neutral_half(self) -> None:
        pref = UserPreferenceState(min_rating=4.0)
        score = _preference_row_score(self._row(stars="N/A"), pref, "italian")
        assert abs(score - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# rerank_pool
# ---------------------------------------------------------------------------

class TestRerankPool:
    def setup_method(self) -> None:
        self.index = _make_index(5)
        self.empty_pref = UserPreferenceState()

    def test_empty_pool_returned_as_is(self) -> None:
        pool = pd.DataFrame()
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert result.empty

    def test_empty_pool_not_copied(self) -> None:
        pool = pd.DataFrame()
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert result is pool

    def test_missing_business_id_column_returned_as_is(self) -> None:
        pool = pd.DataFrame({"name": ["A", "B"], "final_score": [0.9, 0.5]})
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert result is pool

    def test_output_has_v2_score_column(self) -> None:
        pool = _make_pool(4)
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert "v2_score" in result.columns

    def test_output_has_v2_sim_liked_column(self) -> None:
        pool = _make_pool(4)
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert "v2_sim_liked" in result.columns

    def test_output_has_v2_sim_disliked_column(self) -> None:
        pool = _make_pool(4)
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert "v2_sim_disliked" in result.columns

    def test_output_has_v2_pref_match_column(self) -> None:
        pool = _make_pool(4)
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert "v2_pref_match" in result.columns

    def test_output_sorted_by_v2_score_descending(self) -> None:
        pool = _make_pool(4)
        result = rerank_pool(pool, self.index, self.empty_pref)
        scores = result["v2_score"].to_numpy()
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_row_count_preserved(self) -> None:
        pool = _make_pool(4)
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert len(result) == 4

    def test_all_columns_from_pool_preserved(self) -> None:
        pool = _make_pool(4)
        result = rerank_pool(pool, self.index, self.empty_pref)
        for col in pool.columns:
            assert col in result.columns

    def test_single_row_pool(self) -> None:
        pool = _make_pool(1)
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert len(result) == 1
        assert "v2_score" in result.columns

    def test_liked_business_boosts_similar_entries(self) -> None:
        # b0 is liked; b0 is identical (sim=1) to itself in identity matrix.
        pref = UserPreferenceState(liked_business_ids=["b0"])
        pool = _make_pool(4, ids=["b0", "b1", "b2", "b3"])
        # Make scores equal so reranking signal comes only from like similarity
        pool["final_score"] = 0.5

        result = rerank_pool(pool, self.index, pref, w_base=0.0, w_like=1.0, w_dislike=0.0, w_pref=0.0)
        # b0 is most similar to b0 (self-sim=1), so it should rank first
        assert result.iloc[0]["business_id"] == "b0"

    def test_disliked_business_penalizes_similar_entries(self) -> None:
        pref = UserPreferenceState(disliked_business_ids=["b0"])
        pool = _make_pool(4, ids=["b0", "b1", "b2", "b3"])
        pool["final_score"] = 0.5

        result = rerank_pool(pool, self.index, pref, w_base=0.0, w_like=0.0, w_dislike=1.0, w_pref=0.0)
        # b0 is most similar to b0, so it should be penalized most → rank last
        assert result.iloc[-1]["business_id"] == "b0"

    def test_ids_not_in_index_handled_gracefully(self) -> None:
        pool = _make_pool(3, ids=["unknown_x", "unknown_y", "b0"])
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert len(result) == 3

    def test_nan_final_score_treated_as_zero(self) -> None:
        pool = _make_pool(3)
        pool.loc[0, "final_score"] = float("nan")
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert len(result) == 3
        assert not result["v2_score"].isna().any()

    def test_no_preferences_sim_liked_is_zero(self) -> None:
        pool = _make_pool(3)
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert np.allclose(result["v2_sim_liked"].to_numpy(), 0.0)

    def test_no_preferences_sim_disliked_is_zero(self) -> None:
        pool = _make_pool(3)
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert np.allclose(result["v2_sim_disliked"].to_numpy(), 0.0)

    def test_index_reset_after_rerank(self) -> None:
        pool = _make_pool(4)
        result = rerank_pool(pool, self.index, self.empty_pref)
        assert list(result.index) == list(range(len(result)))

    def test_custom_weights_affect_scores(self) -> None:
        pool = _make_pool(4)
        pref = UserPreferenceState(liked_business_ids=["b0"])
        result_low = rerank_pool(pool, self.index, pref, w_like=0.01)
        result_high = rerank_pool(pool, self.index, pref, w_like=2.0)
        # With higher like weight, score spread should differ
        spread_low = result_low["v2_score"].max() - result_low["v2_score"].min()
        spread_high = result_high["v2_score"].max() - result_high["v2_score"].min()
        assert spread_high >= spread_low
