from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def _result_signature(result: pd.DataFrame) -> str:
    """Stable short id for this recommendation list (resets picker when results change)."""
    parts = [f"{row.get('name', '')}|{row.get('city', '')}" for _, row in result.iterrows()]
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _fmt_price_tier_ui(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    try:
        t = int(round(float(x)))
    except (TypeError, ValueError):
        return "N/A"
    t = max(1, min(t, 4))
    return "$" * t


@st.cache_data
def _yelp_photo_business_map_cached(_root: str):
    from app.core.yelp_photos import load_business_photo_ids, resolve_photos_json

    return load_business_photo_ids(resolve_photos_json(Path(_root)), max_per_business=10)


@st.cache_resource
def _yelp_photo_tar_cached(_root: str):
    from app.core.yelp_photos import YelpPhotoTarReader, photos_tar_path

    p = photos_tar_path(Path(_root))
    return YelpPhotoTarReader(p) if p.is_file() else None


# Dialog layout for `width="medium"` (~750px max-width).
_MODAL_MAP_WIDTH = 700
_MODAL_MAP_HEIGHT = 260


def _yelp_cover_photo_bytes(
    business_id: str,
    base_dir: Path,
    photo_map: dict[str, list[str]],
    photo_tar,
) -> bytes | None:
    """First Yelp bundle photo for list cards; None if unavailable."""
    from app.core.yelp_photos import read_photo_jpg_bytes, resolve_photos_json

    bid = str(business_id or "").strip()
    if not bid or bid.startswith("gm_"):
        return None
    if resolve_photos_json(base_dir) is None:
        return None
    pids = photo_map.get(bid, [])
    if not pids:
        return None
    return read_photo_jpg_bytes(base_dir, pids[0], photo_tar)


@st.dialog("Restaurant details", width="medium")
def restaurant_modal_dialog() -> None:
    """Pop-up: full scores, insight, map, Yelp photos. Uses `modal_business_id` or `modal_idx` + `_last_display_records`."""
    from app.search.insights import generate_insight
    from app.search.query_parser import ParsedQuery
    from streamlit_folium import st_folium

    recs = st.session_state.get("_last_display_records")
    root = st.session_state.get("_ui_root")
    if recs is None or root is None:
        return

    result = pd.DataFrame.from_records(recs)
    n = len(result)
    if n == 0:
        st.session_state.pop("modal_idx", None)
        st.session_state.pop("modal_business_id", None)
        return

    bid_m = st.session_state.get("modal_business_id")
    idx = None
    if bid_m is not None and str(bid_m).strip() and "business_id" in result.columns:
        sub = result[result["business_id"].astype(str) == str(bid_m).strip()]
        if len(sub) == 1:
            idx = int(sub.index[0])
    if idx is None and st.session_state.get("modal_idx") is not None:
        idx = int(st.session_state["modal_idx"])
    if idx is None or idx < 0 or idx >= n:
        st.session_state.pop("modal_idx", None)
        st.session_state.pop("modal_business_id", None)
        return

    row = result.iloc[idx]
    base_dir = Path(root)
    parsed = ParsedQuery(**st.session_state["rec_parsed"])
    query_text = st.session_state.get("rec_query_text", "")
    sig = st.session_state.get("_last_result_sig", "x")

    name_esc = str(row.get("name", ""))
    st.markdown(f"#### {name_esc}")
    lat_q = pd.to_numeric(row.get("latitude"), errors="coerce")
    lon_q = pd.to_numeric(row.get("longitude"), errors="coerce")
    if pd.notna(lat_q) and pd.notna(lon_q):
        gmaps = f"https://www.google.com/maps/search/?api=1&query={float(lat_q)},{float(lon_q)}"
        st.markdown(f"[Google Maps]({gmaps})")
    st.caption(
        f"{row.get('address', 'N/A')}, {row.get('city', '')}, {row.get('state', '')}"
    )
    if str(row.get("categories", "")).strip():
        st.caption(str(row.get("categories", ""))[:180])

    fs = float(row["final_score"]) if pd.notna(row.get("final_score")) else 0.0
    sim = float(row["similarity"]) if pd.notna(row.get("similarity")) else 0.0
    pm_def = row.get("price_match")
    pm_s = f"{float(pm_def):.2f}" if pm_def is not None and pd.notna(pm_def) else "—"
    dist = row.get("distance_km")
    dist_s = f"{float(dist):.1f} km" if dist is not None and pd.notna(dist) else "—"
    st.caption(
        f"{float(row['stars']):.1f}★ · {int(row['review_count'])} reviews · "
        f"{_fmt_price_tier_ui(row.get('price_tier'))} (match {pm_s}) · {dist_s}"
    )
    score_line = f"final {fs:.2f} · sim {sim:.2f}"
    if pd.notna(row.get("v2_score")):
        score_line += f" · v2 {float(row['v2_score']):.2f}"
    st.caption(score_line)
    st.progress(min(max(float(row["stars"]) / 5.0, 0.0), 1.0))

    insight = generate_insight(row, parsed, query_text)
    with st.expander("Why this pick?", expanded=True):
        st.markdown(insight["why"])
        st.markdown(f"**Pros:** {insight['pros']}")
        st.markdown(f"**Cons:** {insight['cons']}")

    st.divider()
    st.markdown("**Map** (red = this place, blue = other Top-K)")
    if "latitude" in result.columns and "longitude" in result.columns:
        m = _build_recommendation_folium_map(result, idx)
        st_folium(
            m,
            width=_MODAL_MAP_WIDTH,
            height=_MODAL_MAP_HEIGHT,
            returned_objects=[],
            key=f"modal_map_{sig}_{idx}",
        )
    else:
        st.info("Latitude/longitude missing.")

    st.divider()
    st.markdown("**Photos** (Yelp bundle)")
    pmap = _yelp_photo_business_map_cached(root)
    ptar = _yelp_photo_tar_cached(root)
    _render_yelp_photos_section(row, base_dir, pmap, ptar, num_columns=3)

    st.divider()
    if st.button("Close", key=f"modal_close_{sig}_{idx}"):
        st.session_state.pop("modal_idx", None)
        st.session_state.pop("modal_business_id", None)
        st.rerun()


def _build_recommendation_folium_map(result: pd.DataFrame, focus_idx: int):
    """Center/zoom on focused row; emphasize selected marker, show peers as smaller points."""
    import folium

    n = len(result)
    if n == 0:
        return folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    focus_idx = max(0, min(int(focus_idx), n - 1))
    rows = list(result.iterrows())

    def _coords(ser: pd.Series) -> tuple[float, float] | None:
        lat = float(pd.to_numeric(ser.get("latitude"), errors="coerce"))
        lon = float(pd.to_numeric(ser.get("longitude"), errors="coerce"))
        if np.isfinite(lat) and np.isfinite(lon):
            return lat, lon
        return None

    _, focus_row = rows[focus_idx]
    center = _coords(focus_row)
    if center is None:
        lat_m = pd.to_numeric(result["latitude"], errors="coerce")
        lon_m = pd.to_numeric(result["longitude"], errors="coerce")
        center = (float(lat_m.mean()), float(lon_m.mean()))

    m = folium.Map(
        location=[center[0], center[1]],
        zoom_start=16,
        tiles="cartodbpositron",
        control_scale=True,
    )

    for i, (_, r) in enumerate(rows):
        if i == focus_idx:
            continue
        c = _coords(r)
        if c is None:
            continue
        name = str(r.get("name", ""))
        folium.CircleMarker(
            c,
            radius=5,
            color="#2471a3",
            weight=1,
            fill=True,
            fill_color="#5dade2",
            fill_opacity=0.35,
            tooltip=f"#{i + 1} {name}",
        ).add_to(m)

    fc = _coords(focus_row)
    if fc is not None:
        fn = str(focus_row.get("name", ""))
        folium.Circle(
            fc,
            radius=220,
            color="#cb4335",
            fill=True,
            fill_opacity=0.08,
            weight=1,
        ).add_to(m)
        folium.CircleMarker(
            fc,
            radius=18,
            color="#943126",
            weight=4,
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.95,
            tooltip=f"#{focus_idx + 1} {fn} · selected",
            popup=folium.Popup(
                f"<b>#{focus_idx + 1} {fn}</b><br/>"
                f"{focus_row.get('address', 'N/A')}, {focus_row.get('city', '')}, {focus_row.get('state', '')}",
                max_width=320,
            ),
        ).add_to(m)

    return m


def _render_yelp_photos_section(
    row: pd.Series,
    base_dir: Path,
    photo_map: dict[str, list[str]],
    photo_tar,
    *,
    num_columns: int = 3,
) -> None:
    """Yelp Photos thumbnails from `data/Yelp Photos/` (disk or tar)."""
    from app.core.yelp_photos import (
        has_local_photo_folder,
        photos_tar_path,
        read_photo_jpg_bytes,
        resolve_photos_json,
    )

    bid = str(row.get("business_id", "") or "").strip()
    if not bid:
        st.info("No Yelp business id — photos unavailable.")
        return
    if bid.startswith("gm_"):
        st.info("Google Maps listings are not in the Yelp Photos bundle.")
        return
    if resolve_photos_json(base_dir) is None:
        st.warning(
            "Missing photos manifest: add `data/Yelp Photos/photos.json` "
            "(or `data/Yelp Photos/yelp_photos/photos.json`)."
        )
        return

    pids = photo_map.get(bid, [])
    if not pids:
        st.info(
            "No photos for this business in the bundled Yelp Photos dataset "
            "(subset may not cover every indexed restaurant)."
        )
        return

    has_tar = photos_tar_path(base_dir).is_file()
    local = has_local_photo_folder(base_dir)
    if local and has_tar:
        src = "local `photos/` folder (tar also present)"
    elif local:
        src = "local `photos/` or `yelp_photos/photos/` — **no tar needed**"
    elif has_tar:
        src = "`yelp_photos.tar` on demand"
    else:
        src = "no image files or tar found"
    st.caption(f"Showing up to **{len(pids)}** image(s) — {src}.")

    nc = max(1, min(int(num_columns), 4))
    cols = st.columns(min(nc, len(pids)))
    for j, pid in enumerate(pids):
        data = read_photo_jpg_bytes(base_dir, pid, photo_tar)
        with cols[j % len(cols)]:
            if data:
                st.image(data, use_container_width=True)
            else:
                st.caption(f"Could not load `{pid}`")


def main():
    """
    Streamlit MVP:
    - Build/load an offline restaurant retrieval index (TF-IDF + cosine similarity)
    - Provide keyword + optional state/city filtering
    - Return top-K recommended restaurants
    """

    st.title("Yelp Commercial & Dining Intelligence (MVP Recommendation System)")
    st.caption(
        "Ranking blends sidebar weights (stars, text similarity, price, distance, popularity). "
        "Use **Search & filters** below, then open each result in **View details**."
    )

    base_dir = Path(__file__).resolve().parents[1]
    # Ensure project root is importable when Streamlit runs from different CWDs.
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    if "v2_prefs" not in st.session_state:
        st.session_state["v2_prefs"] = {
            "liked_business_ids": [],
            "disliked_business_ids": [],
            "preferred_cuisines": [],
            "disliked_cuisines": [],
            "price_preference": None,
            "max_distance_km": None,
            "min_rating": None,
        }

    # Import after sys.path adjustment (Streamlit may change CWD).
    from app.search.query_parser import extract_budget_hint, parse_query
    from app.core.retrieval import TouristRetrieval
    retriever = TouristRetrieval(
        data_dir=base_dir / "data" / "cleaned",
        index_dir=base_dir / "models" / "artifacts",
        max_businesses=20000,
        max_reviews_per_business=10,
        restrict_index_cities=True,
        rating_trust_ref_reviews=150.0,
    )

    @st.cache_resource
    def load_index(force_rebuild: bool = False):
        with st.spinner("Loading/building the retrieval index (may take a moment)..."):
            return retriever.build_or_load_index(force_rebuild=force_rebuild)

    @st.cache_data
    def load_state_options() -> list[str]:
        from app.core.google_maps_loader import union_state_options

        return union_state_options(base_dir / "data" / "cleaned")

    _root_s = str(base_dir.resolve())

    with st.sidebar:
        st.header("Search Settings")
        st.caption(
            "Multi-page app: open **K-Means Clustering** from the sidebar page list. "
            "If you use `google_maps_restaurants(cleaned).csv`, click **Force Rebuild Index** once to include it."
        )
        rebuild = st.button(
            "Force Rebuild Index",
            help="Rebuild TF-IDF from `business_dining` + `review_dining`, and merge Google Maps CSV if present.",
        )
        st.subheader("Ranking weights (d1doc)")
        st.caption(
            "Rating uses Yelp stars **shrunk** toward neutral when `review_count` is small "
            "(~full trust at 150+ reviews). Defaults favor stars over raw text match."
        )
        w_semantic = st.slider("w_semantic (text similarity)", 0.0, 2.0, 0.85, 0.05)
        w_rating = st.slider("w_rating (stars × review trust)", 0.0, 2.0, 1.05, 0.05)
        w_price = st.slider("w_price", 0.0, 2.0, 0.15, 0.05)
        w_distance = st.slider("w_distance", 0.0, 2.0, 0.2, 0.05)
        w_popularity = st.slider("w_popularity (log reviews)", 0.0, 2.0, 0.1, 0.05)

        st.subheader("Interactive v2 — candidate pool")
        pool_k_ui = st.slider(
            "Internal pool size (retrieve Top-N, then show Top-K from Advanced)",
            min_value=15,
            max_value=120,
            value=45,
            step=5,
            help="Larger pool enables re-ranking after 👍/👎 without a new search. Must be ≥ Top-K.",
        )

        st.subheader("NL → Location filter (optional)")
        _semantic_ok = importlib.util.find_spec("sentence_transformers") is not None
        if not _semantic_ok:
            st.caption("`sentence-transformers` not installed.")
        use_minilm_state = st.checkbox(
            "Infer state from NL if query mentions a place but Step 1 state seems missing",
            value=False,
            disabled=not _semantic_ok,
            help="Usually off: Step 1 already sets state. Uses all-MiniLM-L6-v2.",
        )

    @st.cache_resource
    def _load_minilm_cached():
        from app.search.semantic_filters import load_minilm_model

        return load_minilm_model()

    with st.container(border=True):
        st.markdown("### Search & filters")
        st.caption(
            "**Step 1:** state (required) + optional city → **Find general restaurants**. "
            "**Step 2:** cuisine, budget, keywords, or natural language → **Update with preferences**."
        )
        st.markdown("#### Step 1 — Where are you dining?")
        st.caption(
            "Pick a **state** (required). City is optional (exact match, case-insensitive). "
            "The first search only ranks **general restaurants** in that area—no cuisine or budget yet."
        )
        state_options = load_state_options()
        if not state_options:
            st.error("No states found under `data/cleaned/`.")
            st.stop()
        wiz_cols = st.columns((1, 1, 1))
        with wiz_cols[0]:
            browse_state = st.selectbox(
                "State (required)",
                options=state_options,
                index=0,
                key="wiz_browse_state",
                help="Only states present in the dataset / index.",
            )
        with wiz_cols[1]:
            browse_city = st.text_input(
                "City (optional)",
                value="",
                key="wiz_browse_city",
                help="Exact city name match. Leave blank for the whole state.",
            )
        with wiz_cols[2]:
            st.write("")
            st.write("")
            discover_btn = st.button("Find general restaurants here", type="primary", key="wiz_discover")

        st.markdown("#### Step 2 — Refine (cuisine, budget, keywords)")
        _refine_expanded = bool(st.session_state.get("rec_ok"))
        with st.expander("Preferences & natural language — use after Step 1", expanded=_refine_expanded):
            nl_query = st.text_area(
                "Natural language (e.g. cheap sushi, near NYU, within 3 km)",
                value="",
                height=88,
                key="wiz_nl_query",
                help="Parsed for budget, cuisine hints, landmarks, radius. Applied when you click **Update with preferences**.",
            )
            cuisine_options = ["Sushi", "Steakhouse", "Korean", "Fast Food", "Chinese", "Burger", "Healthy"]
            cuisines = st.multiselect("Cuisines", options=cuisine_options, default=[], key="wiz_cuisines")
            keywords = st.text_input(
                "Extra keywords (optional)",
                value="",
                key="wiz_keywords",
                help="Merged into the text query; budget words here also count.",
            )
            top_k = st.slider("Top-K", min_value=3, max_value=30, value=10, step=1, key="wiz_top_k")
        refine_btn = st.button("Update with preferences", type="secondary", key="wiz_refine")

    _nl_map = {
        "sushi": "Sushi",
        "japanese": "Sushi",
        "steakhouse": "Steakhouse",
        "steak": "Steakhouse",
        "korean": "Korean",
        "chinese": "Chinese",
        "fast food": "Fast Food",
        "burger": "Burger",
        "healthy": "Healthy",
        "salad": "Healthy",
        "vegan": "Healthy",
        "vegetarian": "Healthy",
    }

    discover_run = discover_btn
    refine_run = refine_btn

    if discover_run or refine_run:
        try:
            st.session_state["v2_prefs"]["liked_business_ids"] = []
            st.session_state["v2_prefs"]["disliked_business_ids"] = []

            if not browse_state or not str(browse_state).strip():
                st.warning("Please choose a **state** in Step 1.")
                st.stop()

            index = load_index(force_rebuild=rebuild)
            effective_state = str(browse_state).strip().upper()
            city_filter = browse_city.strip() if browse_city.strip() else None
            semantic_state_note = ""

            if discover_run:
                parsed = parse_query("")
                query_text = "restaurants"
                effective_cuisines = None
            else:
                parsed = parse_query(nl_query or "")
                _constraint_str = " ".join(
                    x for x in (nl_query or "", keywords or "") if str(x).strip()
                ).strip()
                _budget_from_extra = extract_budget_hint(_constraint_str) if _constraint_str else None
                if _budget_from_extra and not parsed.budget:
                    parsed = replace(parsed, budget=_budget_from_extra)

                if use_minilm_state and (nl_query or "").strip():
                    try:
                        enc = _load_minilm_cached()
                        st_codes = load_state_options()
                        from app.search.semantic_filters import infer_state_minilm

                        guessed, sc = infer_state_minilm(nl_query, st_codes, enc)
                        if guessed:
                            semantic_state_note = (
                                f"MiniLM suggested **{guessed}** (cosine **{sc:.3f}**) — "
                                "Step 1 state still applies unless you clear it."
                            )
                    except Exception as ex:
                        semantic_state_note = f"MiniLM inference failed (ignored): `{ex}`"

                cuisine_from_nl: list[str] = []
                if parsed.cuisine and parsed.cuisine in _nl_map:
                    cuisine_from_nl.append(_nl_map[parsed.cuisine])
                effective_cuisines = list(dict.fromkeys(list(cuisines or []) + cuisine_from_nl))
                effective_cuisines = effective_cuisines or None

                semantic_parts: list[str] = []
                if parsed.semantic_query and str(parsed.semantic_query).strip():
                    semantic_parts.append(str(parsed.semantic_query).strip())
                if keywords and str(keywords).strip():
                    semantic_parts.append(str(keywords).strip())
                if effective_cuisines:
                    semantic_parts.extend([str(c) for c in effective_cuisines if str(c).strip()])
                seen: set[str] = set()
                ordered: list[str] = []
                for p in semantic_parts:
                    p = str(p).strip()
                    if not p or p in seen:
                        continue
                    seen.add(p)
                    ordered.append(p)
                query_text = " ".join(ordered).strip()
                if not query_text:
                    query_text = (nl_query or "").strip() or "restaurants"

            candidates = index.meta
            if effective_state:
                state_norm_q = effective_state.upper()
                candidates = candidates[candidates["state_norm"].astype(str) == state_norm_q]
            if city_filter:
                city_norm_q = city_filter.lower()
                candidates = candidates[candidates["city_norm"].astype(str) == city_norm_q]

            if len(candidates) == 0:
                available_states = sorted(
                    index.meta["state"].astype(str).str.strip().str.upper().unique().tolist()
                )
                st.session_state["rec_ok"] = False
                st.warning(
                    "No matching restaurants in the current dataset for the selected state/city. "
                    "This is a data-coverage issue (rebuilding the index will not help). "
                    f"Available states include: {', '.join(available_states[:10])}..."
                )
                st.stop()

            pool_k_eff = max(int(top_k), int(pool_k_ui))
            _pool = retriever.recommend_keywords(
                keywords=query_text,
                index=index,
                state=effective_state,
                city=city_filter,
                cuisines=effective_cuisines,
                top_k=top_k,
                pool_k=pool_k_eff,
                include_business_id=True,
                budget=parsed.budget,
                ref_lat=parsed.ref_lat,
                ref_lon=parsed.ref_lon,
                max_radius_km=parsed.radius_km,
                w_semantic=w_semantic,
                w_rating=w_rating,
                w_price=w_price,
                w_distance=w_distance,
                w_popularity=w_popularity,
            )
            st.session_state["rec_pool_df"] = _pool.copy()
            st.session_state["rec_top_k"] = int(top_k)
            st.session_state["rec_pool_k"] = pool_k_eff
            st.session_state["rec_parsed"] = asdict(parsed)
            st.session_state["rec_query_text"] = query_text
            st.session_state["rec_semantic_note"] = semantic_state_note
            st.session_state["rec_effective_state"] = effective_state
            st.session_state["rec_city_filter"] = city_filter
            st.session_state["rec_discovery_only"] = bool(discover_run)
            st.session_state["rec_ok"] = True
            st.session_state.pop("modal_idx", None)
            st.session_state.pop("modal_business_id", None)

        except FileNotFoundError as e:
            st.session_state["rec_ok"] = False
            st.error(f"Data file not found: {e}")
        except Exception as e:
            st.session_state["rec_ok"] = False
            st.exception(e)

    if st.session_state.get("rec_ok") and "rec_pool_df" in st.session_state:
        from app.recommendation.preference_state import (
            preference_from_session,
            session_dict_from_preference,
            toggle_dislike,
            toggle_like,
        )
        from app.recommendation.reranker import rerank_pool
        from app.search.query_parser import ParsedQuery

        index = load_index(force_rebuild=rebuild)
        pool = st.session_state["rec_pool_df"]
        show_k = int(st.session_state.get("rec_top_k") or 10)
        pref = preference_from_session(st.session_state["v2_prefs"])
        interactive_on = bool(pref.liked_business_ids or pref.disliked_business_ids)
        if interactive_on:
            ranked = rerank_pool(pool, index, pref)
            result = ranked.head(show_k)
        else:
            result = pool.head(show_k)
        parsed = ParsedQuery(**st.session_state["rec_parsed"])
        semantic_state_note = st.session_state.get("rec_semantic_note", "")
        effective_state = st.session_state.get("rec_effective_state")

        with st.container(border=True):
            st.markdown("### Recommendations")
            with st.expander("Parsed constraints (rule-based + applied filters)", expanded=False):
                st.json(parsed.to_dict())
                _cf = st.session_state.get("rec_city_filter")
                if effective_state:
                    st.markdown(f"**Step 1 state:** `{effective_state}`")
                if _cf:
                    st.markdown(f"**Step 1 city:** `{_cf}`")
                if st.session_state.get("rec_discovery_only"):
                    st.caption("Last run: **general discovery** (no cuisine / budget from Step 2).")
                if semantic_state_note:
                    st.markdown(semantic_state_note)
                st.caption(
                    "Results reflect the last **Find general restaurants here** or **Update with preferences** run. "
                    "Change Step 1–2 and search again to refresh."
                )

            pk = int(st.session_state.get("rec_pool_k") or len(pool))
            st.caption(
                f"v2 candidate pool: **{len(pool)}** rows (internal Top-{pk}). "
                f"Showing **{show_k}**. "
                + (
                    "Re-ranked with 👍/👎 (TF-IDF similarity + base score)."
                    if interactive_on
                    else "Use 👍 / 👎 on a card to re-rank within the pool without searching again."
                )
            )
            b_reset, _ = st.columns([1, 3])
            with b_reset:
                if st.button("Reset v2 feedback (likes / dislikes)", key="v2_reset_feedback"):
                    st.session_state["v2_prefs"]["liked_business_ids"] = []
                    st.session_state["v2_prefs"]["disliked_business_ids"] = []
                    st.rerun()
            if len(result) == 0:
                st.warning(
                    "No matching restaurants under the current filters. "
                    "Try: clear city/state, widen radius, remove cuisine filters, or pick another state "
                    "present in this dataset."
                )
            else:
                n_res = len(result)
                sig = _result_signature(result)
                st.session_state["_last_display_records"] = result.to_dict("records")
                st.session_state["_ui_root"] = _root_s
                st.session_state["_last_result_sig"] = sig

                st.caption(
                    "Each block is one result (cover from Yelp Photos when available). **View details** opens the full dialog."
                )

                photo_map = _yelp_photo_business_map_cached(_root_s)
                photo_tar = _yelp_photo_tar_cached(_root_s)

                for i in range(n_res):
                    row_i = result.iloc[i]
                    rank = i + 1
                    name_esc = str(row_i.get("name", ""))
                    addr = f"{row_i.get('address', 'N/A')}, {row_i['city']}, {row_i['state']}"
                    dist = row_i.get("distance_km")
                    dist_s = (
                        f"{float(dist):.1f} km"
                        if dist is not None and pd.notna(dist)
                        else "—"
                    )
                    fs = float(row_i["final_score"]) if pd.notna(row_i.get("final_score")) else 0.0
                    sim = float(row_i["similarity"]) if pd.notna(row_i.get("similarity")) else 0.0
                    score_bits = f"v1 {fs:.2f} · sim {sim:.2f}"
                    if pd.notna(row_i.get("v2_score")):
                        score_bits += f" · v2 {float(row_i['v2_score']):.2f}"
                    bid = str(row_i.get("business_id", "") or "").strip()
                    cover_bytes = _yelp_cover_photo_bytes(bid, base_dir, photo_map, photo_tar)

                    with st.container(border=True):
                        c_pic, c_txt = st.columns([1, 2.35], vertical_alignment="center")
                        with c_pic:
                            if cover_bytes:
                                st.image(cover_bytes, use_container_width=True)
                            else:
                                st.caption("No photo")
                        with c_txt:
                            h1, h2 = st.columns([3, 1])
                            with h1:
                                st.markdown(f"**#{rank}. {name_esc}**")
                                st.caption(
                                    f"{addr} · {float(row_i['stars']):.1f}★ ({int(row_i['review_count'])} reviews) · "
                                    f"{_fmt_price_tier_ui(row_i.get('price_tier'))} · {dist_s} · {score_bits}"
                                )
                            with h2:
                                st.write("")
                                if st.button(
                                    "View details", key=f"modal_open_{sig}_{i}", use_container_width=True
                                ):
                                    if bid:
                                        st.session_state["modal_business_id"] = bid
                                        st.session_state.pop("modal_idx", None)
                                    else:
                                        st.session_state["modal_idx"] = i
                                        st.session_state.pop("modal_business_id", None)
                                    st.rerun()
                        if bid:
                            kh = hashlib.md5(bid.encode()).hexdigest()[:10]
                            c_a, c_b, c_c = st.columns(3)
                            with c_a:
                                if st.button("👍 Like", key=f"v2like_{sig}_{rank}_{kh}"):
                                    p = preference_from_session(st.session_state["v2_prefs"])
                                    toggle_like(p, bid)
                                    st.session_state["v2_prefs"] = session_dict_from_preference(p)
                                    st.rerun()
                            with c_b:
                                if st.button("👎 Dislike", key=f"v2dis_{sig}_{rank}_{kh}"):
                                    p = preference_from_session(st.session_state["v2_prefs"])
                                    toggle_dislike(p, bid)
                                    st.session_state["v2_prefs"] = session_dict_from_preference(p)
                                    st.rerun()
                            with c_c:
                                liked_here = bid in (
                                    st.session_state["v2_prefs"].get("liked_business_ids") or []
                                )
                                disliked_here = bid in (
                                    st.session_state["v2_prefs"].get("disliked_business_ids") or []
                                )
                                if liked_here:
                                    fc = "Marked liked"
                                elif disliked_here:
                                    fc = "Marked disliked"
                                else:
                                    fc = "No feedback"
                                st.caption(fc)

                        st.progress(min(max(float(row_i["stars"]) / 5.0, 0.0), 1.0))

                if (
                    st.session_state.get("modal_business_id") is not None
                    or st.session_state.get("modal_idx") is not None
                ):
                    restaurant_modal_dialog()

        st.write(
            "How it works: TF-IDF cosine similarity on aggregated review text, then a multi-factor "
            "score (sidebar weights) over normalized similarity, stars, price fit vs budget, "
            "distance to a reference point when provided, and log review count."
        )

if __name__ == "__main__":
    main()
