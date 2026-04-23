"""Merchant prediction inference: loads models/reference data once, exposes predict_location()."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Absolute paths — works regardless of CWD when Streamlit runs
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_SPATIAL = _REPO_ROOT / "train_spatial.csv"
_SURVIVAL_MODEL = _REPO_ROOT / "models" / "artifacts" / "global_survival_model.pkl"
_RATING_MODEL = _REPO_ROOT / "models" / "artifacts" / "global_rating_model.pkl"


@st.cache_resource(show_spinner="Loading merchant models…")
def _load_resources():
    ref_df = pd.read_csv(_TRAIN_SPATIAL)
    surv_model = joblib.load(_SURVIVAL_MODEL)
    rate_model = joblib.load(_RATING_MODEL)
    cat_cols = [c for c in ref_df.columns if c.startswith("cat_")]
    return ref_df, surv_model, rate_model, cat_cols


def _align_features(
    live_features: pd.DataFrame,
    cat_array: np.ndarray,
    cat_cols: list[str],
    expected_features: np.ndarray,
) -> pd.DataFrame:
    """Zero-pad to model's expected columns, then fill in computed values."""
    df = pd.DataFrame(0.0, index=[0], columns=expected_features)
    for col in live_features.columns:
        if col in df.columns:
            df[col] = live_features[col].values
    for idx, col in enumerate(cat_cols):
        if col in df.columns:
            df[col] = cat_array[idx]
    return df


def predict_location(
    city: str,
    lat: float,
    lon: float,
    selected_cat_keys: list[str],
) -> tuple[float, float, int]:
    """
    Run live spatial inference for a prospective merchant location.

    Returns (survival_probability 0-1, predicted_stars 1-5, competitor_count_3km).
    """
    from pipelines.spatial_feature_engineer import SpatialFeatureEngineer

    ref_df, surv_model, rate_model, cat_cols = _load_resources()

    # Filter to target city; fall back to nearest 2000 rows if city not found
    local_ref = ref_df[ref_df["city"].str.lower() == city.lower()]
    if len(local_ref) < 10:
        local_ref = ref_df.head(2000)

    # Build one-hot category array aligned to reference cat columns
    cat_array = np.array(
        [1.0 if col in selected_cat_keys else 0.0 for col in cat_cols]
    )

    # SpatialFeatureEngineer(None) — engineer_single_target() takes reference_df directly
    engineer = SpatialFeatureEngineer(None)
    live_features = engineer.engineer_single_target(
        (lat, lon), cat_array, local_ref.reset_index(drop=True)
    )

    surv_df = _align_features(live_features, cat_array, cat_cols, surv_model.feature_names_in_)
    rate_df = _align_features(live_features, cat_array, cat_cols, rate_model.feature_names_in_)

    surv_prob: float = float(surv_model.predict_proba(surv_df)[:, 1][0])
    stars_pred: float = float(rate_model.predict(rate_df)[0])
    all_competitors: int = int(live_features["count_all_3.0km"].iloc[0])
    same_cat_competitors: int = int(live_features["count_same_cat_3.0km"].iloc[0])

    return surv_prob, stars_pred, all_competitors, same_cat_competitors


def models_available() -> bool:
    return _SURVIVAL_MODEL.exists() and _RATING_MODEL.exists() and _TRAIN_SPATIAL.exists()
