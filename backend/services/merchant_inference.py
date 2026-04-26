"""
Merchant site inference: spatial features + saved sklearn models (no UI).

Survival: ``models/artifacts/global_survival_model.pkl`` (binary proba).
Stars: ``models/artifacts/global_rating_model.pkl`` (regression).
"""
from __future__ import annotations

import re
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
    return d / "global_survival_model.pkl", d / "global_rating_model.pkl"


def load_spatial_reference(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(spatial_train_csv_path(repo_root))


def list_merchant_category_keys(
    *,
    city: Optional[str] = None,
    state: Optional[str] = None,
    max_rows_if_no_city: int = 2000,
    repo_root: Optional[Path] = None,
) -> list[str]:
    """
    ``cat_*`` column names used by ``predict_merchant_site`` for the same reference
    slice (city / state / head). Returns columns that have at least one non-zero
    in the slice when possible, else all ``cat_*`` headers in the slice.
    """
    root = resolve_repo_root(repo_root)
    global_ref = load_spatial_reference(root)
    local_ref, _ = slice_local_reference(
        global_ref, city, max_rows_if_no_city=max_rows_if_no_city, state=state
    )
    raw = [c for c in local_ref.columns if c.startswith("cat_")]
    present: list[str] = []
    for c in raw:
        try:
            s = pd.to_numeric(local_ref[c], errors="coerce").fillna(0.0)
            if float(s.sum()) > 0.0:
                present.append(c)
        except (TypeError, ValueError):
            continue
    return sorted(present) if present else sorted(raw)


# User-facing phrases (match substrings) -> substrings in cat_* / labels to boost
_MERCHANT_QUERY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fast food", ("fast_food", "fast food")),
    ("fastfood", ("fast_food", "fast food")),
    ("fast-food", ("fast_food", "fast food")),
    ("burger", ("burgers", "hamburger", "hamburgers", " burger")),
    ("burgers", ("burgers", "hamburger", "hamburgers")),
    ("coffee", ("coffee", "coffeeshops", "coffeeshop", "coffee_&_tea", " coffee")),
    # NOTE: keep "tea" strict (word boundary) to avoid "sTEAkh..." false hits.
    # We avoid mapping plain "tea" to "team_*" style categories by scoring rules.
    ("tea", ("tea", "bubble tea", "coffee_&_tea")),
    ("pizza", ("pizza",)),
    ("sushi", ("sushi", "japanese", "japan")),
    ("taco", ("mexican", "taco", "tex-mex", "mexic")),
    ("dessert", ("desserts", "baker", "bakeries")),
    ("bar", ("bars", "lounges", "pub", "sports bar")),
    ("vegan", ("vegan", "vegetarian", "veggie")),
)


def _cat_key_to_label(key: str) -> str:
    s = key[4:] if key.startswith("cat_") else key
    return s.replace("_", " ").lower()


def _score_phrase_on_label(query: str, label: str) -> float:
    """Higher = better. ``label`` is humanized ``cat_*`` (lowercase, spaces)."""
    q, p = (query or "").strip().lower(), (label or "").strip().lower()
    if not q or not p:
        return 0.0
    if q == p or p == q:
        return 100.0
    if p.startswith(q) and len(q) >= 2:
        return 92.0
    if len(q) >= 3 and q in p:
        return 86.0
    if len(q) >= 2 and (q in p or p in q):
        return 80.0
    parts = [
        t
        for t in re.split(r"[^a-z0-9&]+", q)
        if t and len(t) >= 2 and t not in ("or", "and", "the", "a", "an")
    ]
    if not parts:
        return 0.0
    if all(t in p for t in parts):
        return 78.0
    # Avoid tiny substrings matching unrelated words (e.g. "tea" in "team").
    p_words = [w for w in re.split(r"[^a-z0-9&]+", p) if w]
    hits = sum(1 for t in parts if any((t == w or (len(t) >= 4 and t in w)) for w in p_words))
    if hits:
        return 45.0 + 25.0 * (hits / len(parts))
    for t in parts:
        for w in p.replace("&", " ").split():
            if len(t) >= 4 and (t in w or w.startswith(t[:4])):
                return 35.0
    return 0.0


def _score_merchant_category_query_against_key(query: str, key: str) -> float:
    label = _cat_key_to_label(key)
    raw_unders = key[4:].lower() if key.startswith("cat_") else key.lower()
    s = _score_phrase_on_label(query, label)
    s = max(s, _score_phrase_on_label(query, raw_unders.replace("_", " ")))
    s = max(s, _score_phrase_on_label(query, raw_unders))
    ql = query.lower()
    for phrase, toks in _MERCHANT_QUERY_ALIASES:
        # Avoid accidental substring hits like "sTEAkh..." triggering the "tea" alias.
        # For single-word phrases, require word boundaries; for multi-word phrases, allow substring.
        if " " in phrase:
            hit = phrase in ql
        else:
            hit = re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", ql) is not None
        if hit or (len(phrase) <= 6 and query.strip().lower() == phrase):
            for t in toks:
                if t in label or t in raw_unders:
                    s = max(s, 90.0)
    return s


def resolve_merchant_category_text(
    text: str,
    *,
    city: Optional[str] = None,
    state: Optional[str] = None,
    max_rows_if_no_city: int = 2000,
    repo_root: Optional[Path] = None,
    max_keys: int = 3,
) -> list[str]:
    """
    Map free text (e.g. ``"burger"``, ``"fast food, coffee"``) to ``cat_*`` names
    in the current training slice, using humanized label matching + small phrase boosts.
    """
    available = list_merchant_category_keys(
        city=city, state=state, max_rows_if_no_city=max_rows_if_no_city, repo_root=repo_root
    )
    if not available:
        return []
    raw = (text or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"[,;\n]+", raw) if p.strip()] or [raw]
    single_segment = len(parts) == 1
    norm = " ".join(raw.replace(",", " ").split())
    chunks = list({raw, *parts, " ".join(parts), norm})
    chunks = [c for c in chunks if c]
    all_scores: dict[str, float] = {k: 0.0 for k in available}
    for part in chunks:
        for k in available:
            s = _score_merchant_category_query_against_key(part, k)
            if s > all_scores[k]:
                all_scores[k] = s
    ranked = sorted(all_scores.items(), key=lambda x: (-x[1], x[0]))
    if not ranked or ranked[0][1] < 8.0:
        return []
    best_s, s2 = ranked[0][1], ranked[1][1] if len(ranked) > 1 else 0.0
    # For a single-type query like "steakhouse", aggressively prefer the best match.
    if single_segment and best_s >= 80.0 and (best_s - s2) >= 6.0:
        return [ranked[0][0]]
    if best_s >= 88.0 and (best_s - s2) >= 12.0:
        return [ranked[0][0]]
    strong = [k for k, s in ranked if s >= 28.0][:max_keys]
    if strong:
        return strong
    return [ranked[0][0]]


def suggest_merchant_category_text(
    text: str,
    *,
    city: Optional[str] = None,
    state: Optional[str] = None,
    max_rows_if_no_city: int = 2000,
    repo_root: Optional[Path] = None,
    limit: int = 8,
) -> list[str]:
    """
    Return top-N suggested ``cat_*`` keys for a free-text query, even when strict
    matching fails. Used to make 400 errors actionable (e.g. "fine dining").
    """
    available = list_merchant_category_keys(
        city=city, state=state, max_rows_if_no_city=max_rows_if_no_city, repo_root=repo_root
    )
    raw = (text or "").strip()
    if not available or not raw:
        return []
    norm = " ".join(raw.replace(",", " ").split())
    scored = [(k, _score_merchant_category_query_against_key(norm, k)) for k in available]
    scored.sort(key=lambda x: (-x[1], x[0]))
    out: list[str] = []
    for k, s in scored:
        if len(out) >= max(1, int(limit)):
            break
        # Keep only somewhat related keys; otherwise suggestions become noisy.
        if s < 8.0:
            break
        out.append(k)
    if out:
        return out
    # If nothing matches at all, provide stable "good starting point" dining suggestions
    # present in the current slice.
    fallback_priority = [
        "cat_american_(new)",
        "cat_french",
        "cat_steakhouses",
        "cat_wine_bars",
        "cat_cocktail_bars",
        "cat_seafood",
        "cat_italian",
        "cat_sushi_bars",
        "cat_tapas_bars",
        "cat_bars",
        "cat_restaurants",
    ]
    have = [k for k in fallback_priority if k in set(available)]
    if have:
        return have[: max(1, int(limit))]
    return available[: max(1, int(limit))]


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

    # Some survival artifacts were trained with a small "local_*" feature set.
    # When the engineer emits the newer radius-based columns, add aliases so we
    # don't silently feed all-zeros into the classifier (which makes outputs constant).
    def _augment_live_for_model_columns(model_cols: set[str], df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Legacy feature names (older training code) -> current engineered columns
        legacy_map = {
            "local_restaurant_count": "count_all_3.0km",
            "local_same_category_count": "count_same_cat_3.0km",
            "local_same_category_ratio": "same_cat_ratio_3.0km",
            "distance_to_nearest_same_category": "dist_nearest_same_cat",
            "local_same_category_avg_rating": "avg_rating_same_cat_3.0km",
            "local_same_category_survival_rate": "survival_same_cat_3.0km",
            "local_category_diversity": "diversity_3.0km",
        }
        for legacy, cur in legacy_map.items():
            if legacy in model_cols and legacy not in out.columns and cur in out.columns:
                out[legacy] = out[cur]
        return out

    live_df_surv = _augment_live_for_model_columns(set(survival_model.feature_names_in_), live_df)
    live_df_rat = _augment_live_for_model_columns(set(rating_model.feature_names_in_), live_df)

    model_df = pd.DataFrame(0.0, index=[0], columns=survival_model.feature_names_in_)
    for col in live_df_surv.columns:
        if col in model_df.columns:
            model_df[col] = live_df_surv[col].values
    for i, col in enumerate(cat_cols):
        if col in model_df.columns:
            model_df[col] = vec[i]
    surv_prob = float(survival_model.predict_proba(model_df)[:, 1][0])

    model_df_reg = pd.DataFrame(0.0, index=[0], columns=rating_model.feature_names_in_)
    for col in live_df_rat.columns:
        if col in model_df_reg.columns:
            model_df_reg[col] = live_df_rat[col].values
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
