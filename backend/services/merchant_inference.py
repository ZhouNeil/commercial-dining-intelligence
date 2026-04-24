"""
Merchant site inference: spatial features + saved sklearn models (no UI).

Survival: ``models/artifacts/advanced_survival_classifier.pkl`` (binary proba).
Stars: ``models/artifacts/global_rating_model.pkl`` (regression).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from pipelines.spatial_feature_engineer import SpatialFeatureEngineer


@dataclass(frozen=True)
class MerchantPredictResult:
    survival_probability: float
    predicted_stars: float
    reference_row_count: int
    city_filter: Optional[str]
    metrics: dict[str, float]
    live_feature_preview: dict[str, float] = field(default_factory=dict)
    inside_reference_hull: bool = True


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
        "train_spatial.csv not found: place it under data/ or the repository root."
    )


def artifact_paths(repo_root: Path) -> Tuple[Path, Path]:
    """(survival_classifier, rating_regressor) under models/artifacts/."""
    d = repo_root / "models" / "artifacts"
    return d / "advanced_survival_classifier.pkl", d / "global_rating_model.pkl"


def load_spatial_reference(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(spatial_train_csv_path(repo_root))


def slice_local_reference(
    global_ref: pd.DataFrame,
    city: Optional[str],
    *,
    max_rows_if_no_city: int = 2000,
    state: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    if "city" in global_ref.columns and city and str(city).strip():
        c = str(city).strip().lower()
        local = global_ref[global_ref["city"].astype(str).str.lower() == c].copy()
        if state and str(state).strip() and "state" in local.columns:
            st = str(state).strip().upper()
            local = local[local["state"].astype(str).str.strip().str.upper() == st].copy()
        return local, str(city).strip()
    if "city" in global_ref.columns:
        return global_ref.head(max_rows_if_no_city).copy(), None
    return global_ref.head(max_rows_if_no_city).copy(), None


@lru_cache(maxsize=4)
def _survival_model(repo_root_s: str):
    p = Path(repo_root_s) / "models" / "artifacts" / "advanced_survival_classifier.pkl"
    return joblib.load(p)


@lru_cache(maxsize=4)
def _rating_model(repo_root_s: str):
    p = Path(repo_root_s) / "models" / "artifacts" / "global_rating_model.pkl"
    return joblib.load(p)


def clear_model_cache() -> None:
    """测试或热重载时清空 joblib 单例缓存。"""
    _survival_model.cache_clear()
    _rating_model.cache_clear()
    _list_merchant_spatial_cities_cached.cache_clear()


def _reference_lonlat_xy(local_ref: pd.DataFrame) -> np.ndarray:
    """Rows of [lon, lat] with finite coordinates."""
    if "longitude" not in local_ref.columns or "latitude" not in local_ref.columns:
        return np.empty((0, 2))
    lon = pd.to_numeric(local_ref["longitude"], errors="coerce")
    lat = pd.to_numeric(local_ref["latitude"], errors="coerce")
    m = lon.notna() & lat.notna()
    return np.column_stack([lon[m].to_numpy(dtype=float), lat[m].to_numpy(dtype=float)])


def convex_hull_closed_ring_lonlat(xy: np.ndarray) -> Optional[np.ndarray]:
    """
    Closed ring of [lon, lat] for GeoJSON Polygon exterior (first point repeated at end).
    Returns None if fewer than 3 finite points or hull fails.
    """
    if xy.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(xy)
    except Exception:
        return None
    ordered = xy[hull.vertices]
    if ordered.shape[0] < 3:
        return None
    return np.vstack([ordered, ordered[0]])


def point_in_hull_ring(lon: float, lat: float, ring: Optional[np.ndarray]) -> bool:
    """Ray casting; ring is closed [lon,lat] with first vertex repeated at end."""
    if ring is None or ring.shape[0] < 4:
        return False
    verts = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    if verts.shape[0] < 3:
        return False
    x, y = float(lon), float(lat)
    n = int(verts.shape[0])
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(verts[i, 0]), float(verts[i, 1])
        xj, yj = float(verts[j, 0]), float(verts[j, 1])
        if (yi > y) != (yj > y):
            denom = (yj - yi) or 1e-18
            xinters = (xj - xi) * (y - yi) / denom + xi
            if x < xinters:
                inside = not inside
        j = i
    return inside


def _geojson_polygon_feature(ring: np.ndarray) -> dict[str, Any]:
    coords = [[[float(p[0]), float(p[1])] for p in ring]]
    return {
        "type": "Feature",
        "properties": {"kind": "reference_convex_hull"},
        "geometry": {"type": "Polygon", "coordinates": coords},
    }


def _geojson_sample_points_feature(xy: np.ndarray, max_points: int) -> dict[str, Any]:
    if xy.shape[0] == 0:
        return {"type": "Feature", "properties": {"kind": "reference_sample"}, "geometry": {"type": "MultiPoint", "coordinates": []}}
    if xy.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(xy.shape[0], size=max_points, replace=False)
        samp = xy[idx]
    else:
        samp = xy
    coords = [[float(p[0]), float(p[1])] for p in samp]
    return {
        "type": "Feature",
        "properties": {"kind": "reference_sample"},
        "geometry": {"type": "MultiPoint", "coordinates": coords},
    }


@lru_cache(maxsize=4)
def _list_merchant_spatial_cities_cached(
    repo_root_s: str,
    min_rows: int,
) -> tuple[tuple[str, str, int, float, float], ...]:
    """Immutable rows: (city_display, state_upper, row_count, center_lat, center_lon)."""
    root = Path(repo_root_s)
    path = spatial_train_csv_path(root)
    hdr = pd.read_csv(path, nrows=0).columns.tolist()
    cols = ["city", "latitude", "longitude"]
    if "state" in hdr:
        cols.append("state")
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    if "state" not in df.columns:
        df["state"] = ""
    df["city"] = df["city"].astype(str).str.strip()
    df["state"] = df["state"].fillna("").astype(str).str.strip().str.upper()
    df = df[df["city"].str.len() > 0]
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["_city_l"] = df["city"].str.lower()
    out: list[tuple[str, str, int, float, float]] = []
    for (_cl, st), sub in df.groupby(["_city_l", "state"], sort=False):
        n = int(len(sub))
        if n < min_rows:
            continue
        disp = str(sub["city"].iloc[0]).strip()
        lat_m = float(np.nanmean(sub["lat"].to_numpy(dtype=float)))
        lon_m = float(np.nanmean(sub["lon"].to_numpy(dtype=float)))
        if not (np.isfinite(lat_m) and np.isfinite(lon_m)):
            continue
        st_clean = str(st).strip() if st and str(st) not in ("", "NAN") else ""
        out.append((disp, st_clean, n, lat_m, lon_m))
    out.sort(key=lambda r: (-r[2], r[0].lower()))
    return tuple(out)


def list_merchant_spatial_cities(
    repo_root: Optional[Path] = None,
    *,
    min_rows: int = 10,
) -> list[dict[str, Any]]:
    root = resolve_repo_root(repo_root)
    rows = _list_merchant_spatial_cities_cached(str(root.resolve()), int(min_rows))
    return [
        {
            "city": r[0],
            "state": r[1],
            "row_count": r[2],
            "center_lat": r[3],
            "center_lon": r[4],
        }
        for r in rows
    ]


def get_merchant_coverage(
    *,
    city: Optional[str],
    repo_root: Optional[Path] = None,
    max_rows_if_no_city: int = 2000,
    max_sample_points: int = 400,
    state: Optional[str] = None,
) -> dict[str, Any]:
    """
    Spatial extent of the same reference slice used for /merchant/predict:
    bbox, centroid, convex hull of training points, and a subsampled MultiPoint for map dots.
    """
    root = resolve_repo_root(repo_root)
    global_ref = load_spatial_reference(root)
    local_ref, city_used = slice_local_reference(
        global_ref, city, max_rows_if_no_city=max_rows_if_no_city, state=state
    )
    xy = _reference_lonlat_xy(local_ref)
    n = int(xy.shape[0])
    if n == 0:
        return {
            "city_filter": city_used,
            "reference_count": len(local_ref),
            "geo_count": 0,
            "min_lon": 0.0,
            "min_lat": 0.0,
            "max_lon": 0.0,
            "max_lat": 0.0,
            "center_lon": 0.0,
            "center_lat": 0.0,
            "hull_geojson": None,
            "sample_points_geojson": None,
            "valid_hull": False,
        }

    min_lon, min_lat = float(xy[:, 0].min()), float(xy[:, 1].min())
    max_lon, max_lat = float(xy[:, 0].max()), float(xy[:, 1].max())
    center_lon = float(xy[:, 0].mean())
    center_lat = float(xy[:, 1].mean())

    ring = convex_hull_closed_ring_lonlat(xy)
    hull_f = _geojson_polygon_feature(ring) if ring is not None else None
    sample_f = _geojson_sample_points_feature(xy, max_sample_points)

    return {
        "city_filter": city_used,
        "reference_count": len(local_ref),
        "geo_count": n,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "center_lon": center_lon,
        "center_lat": center_lat,
        "hull_geojson": hull_f,
        "sample_points_geojson": sample_f,
        "valid_hull": ring is not None,
    }


def predict_merchant_site(
    *,
    city: Optional[str],
    lat: float,
    lon: float,
    selected_category_columns: list[str],
    repo_root: Optional[Path] = None,
    max_rows_if_no_city: int = 2000,
    reference_df: Optional[pd.DataFrame] = None,
    state: Optional[str] = None,
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
            global_ref, city, max_rows_if_no_city=max_rows_if_no_city, state=state
        )
    if len(local_ref) < 10:
        raise ValueError("Too few reference businesses (<10). Try another city or check train_spatial.csv.")

    xy_ref = _reference_lonlat_xy(local_ref)
    hull_ring = convex_hull_closed_ring_lonlat(xy_ref)
    inside_hull = point_in_hull_ring(float(lon), float(lat), hull_ring)

    cat_cols = [c for c in local_ref.columns if c.startswith("cat_")]
    vec = np.zeros(len(cat_cols), dtype=float)
    unknown = [c for c in selected_category_columns if c not in cat_cols]
    if unknown:
        preview = ", ".join(unknown[:5]) + (" …" if len(unknown) > 5 else "")
        raise ValueError(f"Unknown category column(s): {preview}")
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
        inside_reference_hull=inside_hull,
    )


def predict_merchant_site_safe(
    **kwargs: Any,
) -> Tuple[Optional[MerchantPredictResult], Optional[str]]:
    """返回 (结果, 错误信息)。"""
    try:
        return predict_merchant_site(**kwargs), None
    except Exception as ex:  # noqa: BLE001 — 边界 API 聚合错误
        return None, str(ex)
