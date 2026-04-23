"""Merchant Location Analyzer — predicts survival probability and expected rating for a new restaurant."""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

# Ensure repo root is on path so pipelines/ and models/ can be imported
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.merchant_inference import models_available, predict_location

# ── Constants ────────────────────────────────────────────────────────────────

CITY_CENTERS: dict[str, tuple[float, float]] = {
    "Philadelphia": (39.9526, -75.1652),
    "Tampa": (27.9506, -82.4572),
    "Indianapolis": (39.7684, -86.1581),
    "Tucson": (32.2226, -110.9747),
    "Nashville": (36.1627, -86.7816),
    "New Orleans": (29.9511, -90.0715),
    "Edmonton": (53.5461, -113.4938),
    "Saint Louis": (38.6270, -90.1994),
    "Reno": (39.5296, -119.8138),
    "Boise": (43.6150, -116.2023),
    "Santa Barbara": (34.4208, -119.6982),
}

# Display label → cat_* key in the model
CATEGORY_MAP: dict[str, str] = {
    "Pizza": "cat_pizza",
    "Coffee & Tea": "cat_coffee_&_tea",
    "Fast Food": "cat_fast_food",
    "Nightlife": "cat_nightlife",
    "Bars": "cat_bars",
    "Sandwiches": "cat_sandwiches",
    "Mexican": "cat_mexican",
    "Burgers": "cat_burgers",
    "Chinese": "cat_chinese",
    "Italian": "cat_italian",
    "Breakfast & Brunch": "cat_breakfast_&_brunch",
    "Seafood": "cat_seafood",
    "Sushi Bars": "cat_sushi_bars",
    "Sports Bars": "cat_sports_bars",
}

DEFAULT_ZOOM = 13


# ── Helpers ──────────────────────────────────────────────────────────────────

def _star_display(stars: float) -> str:
    full = int(stars)
    half = 1 if (stars - full) >= 0.25 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


def _survival_color(pct: float) -> str:
    if pct >= 0.70:
        return "green"
    if pct >= 0.40:
        return "orange"
    return "red"


# ── Page layout ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Merchant Location Analyzer", page_icon="🏪", layout="wide")
st.title("🏪 Merchant Location Analyzer")
st.caption("Drop a pin, pick your concept, and see your predicted survival odds and star rating.")

if not models_available():
    st.error(
        "Model artifacts not found. Run `AblationMerchantPredictor.train_pipeline()` "
        "to generate `global_survival_model.pkl` and `global_rating_model.pkl` in `models/artifacts/`."
    )
    st.stop()

# ── Sidebar controls ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Location Settings")
    city = st.selectbox("City", list(CITY_CENTERS.keys()), index=0)
    st.markdown("---")
    st.header("Restaurant Concept")
    selected_labels = st.multiselect(
        "Select categories",
        list(CATEGORY_MAP.keys()),
        placeholder="Choose one or more…",
    )
    selected_cat_keys = [CATEGORY_MAP[lbl] for lbl in selected_labels]

# ── Map ───────────────────────────────────────────────────────────────────────

center = CITY_CENTERS[city]

col_map, col_results = st.columns([3, 2], gap="large")

with col_map:
    st.subheader(f"Drop a pin in {city}")
    st.caption("Click anywhere on the map to set your proposed location.")

    m = folium.Map(location=center, zoom_start=DEFAULT_ZOOM, tiles="CartoDB positron")
    folium.Marker(location=center, tooltip="City center", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
    m.add_child(folium.LatLngPopup())

    map_data = st_folium(m, width="100%", height=420, key=f"map_{city}")

    # Extract clicked coordinates
    pin_lat: float | None = None
    pin_lon: float | None = None
    if map_data and map_data.get("last_clicked"):
        pin_lat = map_data["last_clicked"]["lat"]
        pin_lon = map_data["last_clicked"]["lng"]
        st.success(f"Pin set: {pin_lat:.4f}, {pin_lon:.4f}")
    else:
        st.info("Click the map to drop your pin.")

# ── Results ───────────────────────────────────────────────────────────────────

with col_results:
    st.subheader("Prediction")

    analyze = st.button(
        "Analyze My Location",
        type="primary",
        disabled=(pin_lat is None),
        use_container_width=True,
    )

    if pin_lat is None:
        st.markdown("*Drop a pin on the map first.*")

    elif analyze:
        with st.spinner("Running spatial analysis…"):
            try:
                surv_prob, stars_pred, competitors, same_cat_competitors = predict_location(
                    city, pin_lat, pin_lon, selected_cat_keys
                )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                st.stop()

        pct = surv_prob * 100
        color = _survival_color(surv_prob)

        st.markdown("#### Survival Probability")
        st.markdown(
            f"<h2 style='color:{color};margin:0'>{pct:.1f}%</h2>",
            unsafe_allow_html=True,
        )
        st.progress(surv_prob)
        if color == "green" and competitors > 200:
            st.caption("Strong odds despite high competition — the model sees this location as viable.")
        elif color == "green":
            st.caption("Strong location with manageable competition.")
        elif color == "orange":
            st.caption("Moderate survival odds — concept and execution will matter.")
        else:
            st.caption("Low survival odds — poor market fit or overwhelming competition.")

        st.markdown("---")
        st.markdown("#### Expected Star Rating")
        stars_clipped = max(1.0, min(5.0, stars_pred))
        st.markdown(f"**{_star_display(stars_clipped)}** &nbsp; {stars_clipped:.2f} / 5.0")

        st.markdown("---")
        if selected_cat_keys:
            st.metric(f"Same-concept competitors within 3 km", same_cat_competitors)
            st.caption(f"All restaurants within 3 km: {competitors}. Both counts are factored into the survival probability above.")
        else:
            st.metric("Competitors within 3 km", competitors)
            st.caption("Select a concept above to see same-category competitor count. This count is factored into the survival probability above.")

        if selected_labels:
            st.markdown("---")
            st.markdown(f"**Concept:** {', '.join(selected_labels)}")
