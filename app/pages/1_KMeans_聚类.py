"""
K-Means merchant clustering visualization (mirrors notebooks/kmeans.ipynb).
Run: from project root, `streamlit run app/main.py`, then open this page from the sidebar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from app.clustering.kmeans_clustering import (  # noqa: E402
    DEFAULT_FEATURE_COLS,
    add_log1p_review_count,
    cluster_summary,
    elbow_inertias,
    fit_kmeans,
    folium_cluster_map,
    load_business_for_clustering,
)

st.set_page_config(page_title="K-Means Clustering", layout="wide")
st.title("K-Means Clustering (Merchants)")
st.caption("Mirrors `notebooks/kmeans.ipynb`: geo + stars + review_count (optional log1p), then standardize + K-Means.")

csv_candidates = [
    BASE / "data" / "cleaned" / "business_dining.csv",
    BASE / "data" / "slice_representative" / "business_dining.csv",
]
data_path = None
for p in csv_candidates:
    if p.is_file():
        data_path = p
        break

if data_path is None:
    st.error(
        "Merchant dataset not found. Put `business_dining.csv` under `data/cleaned/`, "
        "or run `scripts/build_representative_slice.py` to generate `data/slice_representative/`."
    )
    st.stop()

with st.sidebar:
    st.header("Data & features")
    st.text(f"Data file:\n{data_path.relative_to(BASE)}")
    try:
        states_df = pd.read_csv(data_path, usecols=["state"], low_memory=False)
        states = (
            states_df["state"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": "", "NONE": ""})
        )
        state_opts = ["ALL"] + sorted(
            {
                str(s).strip().upper()
                for s in states.unique()
                if str(s).strip() and str(s).strip().upper() not in ("NAN", "NONE", "")
            }
        )
    except Exception:
        state_opts = ["ALL"]
    state_sel = st.selectbox("Filter by state (like selecting NJ in the notebook)", state_opts, index=0)
    state_val = None if state_sel == "ALL" else state_sel

    use_log1p = st.checkbox("Apply log1p to review_count", value=True, help="Reduce the impact of extreme review counts")
    n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=12, value=4, step=1)

    st.subheader("Elbow method")
    k_elbow_max = st.slider("Max K for elbow curve", min_value=5, max_value=15, value=10)

btn = st.button("Run clustering", type="primary")

if not btn and "kmeans_result" not in st.session_state:
    st.info("Click **Run clustering** to generate results. On first load, you can select state and K first.")
    st.stop()

if btn:
    with st.spinner("Loading data and clustering..."):
        raw = load_business_for_clustering(data_path, state=state_val)
        if raw.empty:
            st.error("No data after filtering. Try another state or check the CSV.")
            st.stop()
        feat_df = add_log1p_review_count(raw) if use_log1p else raw.copy()
        feature_cols = list(DEFAULT_FEATURE_COLS)
        labeled, kmeans, scaler, X_scaled = fit_kmeans(feat_df, feature_cols, n_clusters=n_clusters)
        ks, inertias = elbow_inertias(X_scaled, k_min=2, k_max=k_elbow_max)
        st.session_state["kmeans_result"] = {
            "labeled": labeled,
            "feature_cols": feature_cols,
            "kmeans": kmeans,
            "X_scaled": X_scaled,
            "ks": ks,
            "inertias": inertias,
            "use_log1p": use_log1p,
            "state": state_val or "ALL",
            "n_clusters": n_clusters,
        }

res = st.session_state["kmeans_result"]
labeled = res["labeled"]
feature_cols = res["feature_cols"]
ks, inertias = res["ks"], res["inertias"]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Merchants", len(labeled))
with c2:
    st.metric("Clusters (K)", res["n_clusters"])
with c3:
    st.metric("State filter", str(res["state"]))

tab_map, tab_scatter, tab_table, tab_elbow = st.tabs(["Map", "Scatter", "Cluster summary", "Elbow curve"])

with tab_map:
    try:
        from streamlit_folium import st_folium

        m = folium_cluster_map(labeled)
        st_folium(m, width=1200, height=480, returned_objects=[])
    except ImportError:
        st.warning("`streamlit-folium` not installed. Falling back to a simple scatter chart. Install: pip install streamlit-folium folium")
        st.scatter_chart(
            labeled.rename(columns={"cluster": "cluster_id"}),
            x="longitude",
            y="latitude",
            color="cluster_id",
        )

with tab_scatter:
    # Use English text to avoid missing CJK glyphs in default fonts.
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        labeled["longitude"],
        labeled["latitude"],
        c=labeled["cluster"],
        cmap="tab10",
        alpha=0.65,
        s=22,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Merchants by cluster (lat / lon)")
    plt.colorbar(sc, ax=ax, label="cluster id")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

with tab_table:
    summ = cluster_summary(labeled, feature_cols)
    st.dataframe(summ.style.format("{:.4f}", subset=feature_cols), width="stretch")
    st.download_button(
        "Download labeled CSV (with cluster id)",
        labeled.to_csv(index=False).encode("utf-8"),
        file_name="kmeans_labeled_merchants.csv",
        mime="text/csv",
    )

with tab_elbow:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(ks, inertias, marker="o")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Inertia")
    ax2.set_title("Elbow curve (inertia vs K)")
    ax2.axvline(
        res["n_clusters"],
        color="crimson",
        linestyle="--",
        alpha=0.7,
        label=f"chosen K={res['n_clusters']}",
    )
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)
