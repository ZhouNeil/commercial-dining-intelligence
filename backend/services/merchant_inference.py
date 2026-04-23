"""
商家选址推理：空间特征 + 全局生存/评分模型（无 UI 框架依赖）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from pipelines.spatial_feature_engineer import SpatialFeatureEngineer


@dataclass(frozen=True)
class MerchantPredictResult:
    survival_probability: float
    predicted_stars: float
    reference_row_count: int
    city_filter: Optional[str]
    metrics: dict[str, float]
    live_feature_preview: dict[str, float] = field(default_factory=dict)


def resolve_repo_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    # backend/services/... → 仓库根
    return Path(__file__).resolve().parents[2]


def spatial_train_csv_path(repo_root: Path) -> Path:
    for rel in ("data/train_spatial.csv", "train_spatial.csv"):
        p = (repo_root / rel).resolve()
        if p.is_file():
            return p
    raise FileNotFoundError(
        "未找到 train_spatial：请将 train_spatial.csv 放在 data/ 或仓库根目录。"
    )


def artifact_paths(repo_root: Path) -> Tuple[Path, Path]:
    d = repo_root / "models" / "artifacts"
    return d / "global_survival_model.pkl", d / "global_rating_model.pkl"


def load_spatial_reference(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(spatial_train_csv_path(repo_root))


def slice_local_reference(
    global_ref: pd.DataFrame,
    city: Optional[str],
    *,
    max_rows_if_no_city: int = 2000,
) -> Tuple[pd.DataFrame, Optional[str]]:
    if "city" in global_ref.columns and city and str(city).strip():
        c = str(city).strip().lower()
        local = global_ref[global_ref["city"].astype(str).str.lower() == c].copy()
        return local, str(city).strip()
    if "city" in global_ref.columns:
        return global_ref.head(max_rows_if_no_city).copy(), None
    return global_ref.head(max_rows_if_no_city).copy(), None


@lru_cache(maxsize=4)
def _survival_model(repo_root_s: str):
    p = Path(repo_root_s) / "models" / "artifacts" / "global_survival_model.pkl"
    return joblib.load(p)


@lru_cache(maxsize=4)
def _rating_model(repo_root_s: str):
    p = Path(repo_root_s) / "models" / "artifacts" / "global_rating_model.pkl"
    return joblib.load(p)


def clear_model_cache() -> None:
    """测试或热重载时清空 joblib 单例缓存。"""
    _survival_model.cache_clear()
    _rating_model.cache_clear()


def predict_merchant_site(
    *,
    city: Optional[str],
    lat: float,
    lon: float,
    selected_category_columns: list[str],
    repo_root: Optional[Path] = None,
    max_rows_if_no_city: int = 2000,
    reference_df: Optional[pd.DataFrame] = None,
) -> MerchantPredictResult:
    """
    :param reference_df: 若传入（例如调用方已按城市筛好的子表），则不再从 CSV 切片，
        此时 ``city`` 仅用于展示 ``city_filter``。
    """
    root = resolve_repo_root(repo_root)
    if reference_df is not None:
        local_ref = reference_df.copy()
        city_used = str(city).strip() if city and str(city).strip() else None
    else:
        global_ref = load_spatial_reference(root)
        local_ref, city_used = slice_local_reference(
            global_ref, city, max_rows_if_no_city=max_rows_if_no_city
        )
    if len(local_ref) < 10:
        raise ValueError("参考商户过少（<10），请更换城市或检查 train_spatial。")

    cat_cols = [c for c in local_ref.columns if c.startswith("cat_")]
    vec = np.zeros(len(cat_cols), dtype=float)
    unknown = [c for c in selected_category_columns if c not in cat_cols]
    if unknown:
        raise ValueError(f"未知品类列: {unknown[:5]}…" if len(unknown) > 5 else f"未知品类列: {unknown}")
    for i, col in enumerate(cat_cols):
        if col in selected_category_columns:
            vec[i] = 1.0

    coord = (float(lat), float(lon))
    engineer = SpatialFeatureEngineer(None)
    live_df = engineer.engineer_single_target(coord, vec, local_ref)

    rs = str(root.resolve())
    survival_model = _survival_model(rs)
    rating_model = _rating_model(rs)

    model_df = pd.DataFrame(0.0, index=[0], columns=survival_model.feature_names_in_)
    for col in live_df.columns:
        if col in model_df.columns:
            model_df[col] = live_df[col].values
    for i, col in enumerate(cat_cols):
        if col in model_df.columns:
            model_df[col] = vec[i]
    surv_prob = float(survival_model.predict_proba(model_df)[:, 1][0])

    model_df_reg = pd.DataFrame(0.0, index=[0], columns=rating_model.feature_names_in_)
    for col in live_df.columns:
        if col in model_df_reg.columns:
            model_df_reg[col] = live_df[col].values
    for i, col in enumerate(cat_cols):
        if col in model_df_reg.columns:
            model_df_reg[col] = vec[i]
    stars = float(rating_model.predict(model_df_reg)[0])

    metrics = {}
    for key in ("count_all_3.0km", "survival_top5_similar", "dist_nearest_same_cat"):
        if key in live_df.columns:
            v = live_df[key].iloc[0]
            metrics[key] = float(v) if pd.notna(v) else float("nan")

    priority = [
        "count_all_0.5km",
        "count_all_3.0km",
        "avg_rating_all_3.0km",
        "survival_top5_similar",
        "avg_rating_top5_similar",
        "dist_nearest_same_cat",
        "log_dist_nearest_same_cat",
    ]
    preview: dict[str, float] = {}
    ordered_cols = [c for c in priority if c in live_df.columns]
    ordered_cols += [c for c in live_df.columns if c not in ordered_cols][:24]
    for col in ordered_cols[:32]:
        v = live_df[col].iloc[0]
        if pd.isna(v):
            continue
        try:
            preview[col] = float(v)
        except (TypeError, ValueError):
            continue

    return MerchantPredictResult(
        survival_probability=surv_prob,
        predicted_stars=stars,
        reference_row_count=len(local_ref),
        city_filter=city_used,
        metrics=metrics,
        live_feature_preview=preview,
    )


def predict_merchant_site_safe(
    **kwargs: Any,
) -> Tuple[Optional[MerchantPredictResult], Optional[str]]:
    """返回 (结果, 错误信息)。"""
    try:
        return predict_merchant_site(**kwargs), None
    except Exception as ex:  # noqa: BLE001 — 边界 API 聚合错误
        return None, str(ex)
