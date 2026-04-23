"""
K-Means clustering helpers (mirrors notebooks/kmeans.ipynb); 供 notebook 或后续前端复用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURE_COLS = ("latitude", "longitude", "stars", "review_count")


def load_business_for_clustering(
    csv_path: Path,
    state: str | None = None,
    extra_usecols: tuple[str, ...] = ("name", "business_id"),
) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    want = {*DEFAULT_FEATURE_COLS, "state", *extra_usecols}
    use = [c for c in header.columns if c in want]
    df = pd.read_csv(csv_path, usecols=use, low_memory=False)
    df = df.dropna(subset=list(DEFAULT_FEATURE_COLS))
    if state and str(state).strip().upper() not in ("", "ALL"):
        s = df["state"].astype(str).str.strip().str.upper()
        df = df.loc[s == str(state).strip().upper()]
    return df.reset_index(drop=True)


def add_log1p_review_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["review_count"] = np.log1p(pd.to_numeric(out["review_count"], errors="coerce").fillna(0).clip(lower=0))
    return out


def fit_kmeans(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    out = df.copy()
    out["cluster"] = labels.astype(int)
    return out, km, scaler, X_scaled


def cluster_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    g = df.groupby("cluster", as_index=True)[list(feature_cols)].mean()
    g["n_merchants"] = df.groupby("cluster").size()
    return g


def elbow_inertias(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> tuple[list[int], list[float]]:
    ks: list[int] = []
    inertias: list[float] = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        ks.append(k)
        inertias.append(float(km.inertia_))
    return ks, inertias


def folium_cluster_map(df: pd.DataFrame, color_list: list[str] | None = None) -> Any:
    """Build a Folium map (requires folium installed)."""
    import folium

    if color_list is None:
        color_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]

    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")
    for _, row in df.iterrows():
        cid = int(row["cluster"])
        color = color_list[cid % len(color_list)]
        name = str(row.get("name", ""))[:80]
        tip = f"Cluster {cid}" + (f" · {name}" if name else "")
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            tooltip=tip,
        ).add_to(m)
    return m
