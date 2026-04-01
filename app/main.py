from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def _result_signature(result: pd.DataFrame) -> str:
    """Stable short id for this recommendation list (resets picker when results change)."""
    parts = [f"{row.get('name', '')}|{row.get('city', '')}" for _, row in result.iterrows()]
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()[:16]


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
                max_width=440,
            ),
        ).add_to(m)

    return m


def main():
    """
    Streamlit MVP:
    - Build/load an offline restaurant retrieval index (TF-IDF + cosine similarity)
    - Provide keyword + optional state/city filtering
    - Return top-K recommended restaurants
    """

    st.title("Yelp Commercial & Dining Intelligence (MVP Recommendation System)")
    st.write(
        "Describe what you want in natural language and/or use advanced filters. "
        "We parse cuisine, budget, and optional “near … / within … km”, then rank with "
        "semantic similarity, rating, price fit, distance, and review popularity (see sidebar weights)."
    )

    base_dir = Path(__file__).resolve().parents[1]
    # Ensure project root is importable when Streamlit runs from different CWDs.
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    # Import after sys.path adjustment (Streamlit may change CWD).
    from app.search.query_parser import parse_query
    from app.search.insights import generate_insight
    from app.core.retrieval import TouristRetrieval
    retriever = TouristRetrieval(
        data_dir=base_dir / "data" / "cleaned",
        index_dir=base_dir / "models" / "artifacts",
        max_businesses=20000,
        max_reviews_per_business=10,
    )

    @st.cache_resource
    def load_index(force_rebuild: bool = False):
        with st.spinner("Loading/building the retrieval index (may take a moment)..."):
            return retriever.build_or_load_index(force_rebuild=force_rebuild)

    @st.cache_data
    def load_state_options() -> list[str]:
        from app.core.google_maps_loader import union_state_options

        return union_state_options(base_dir / "data" / "cleaned")

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
        w_semantic = st.slider("w_semantic", 0.0, 2.0, 1.0, 0.05)
        w_rating = st.slider("w_rating", 0.0, 2.0, 0.25, 0.05)
        w_price = st.slider("w_price", 0.0, 2.0, 0.2, 0.05)
        w_distance = st.slider("w_distance", 0.0, 2.0, 0.25, 0.05)
        w_popularity = st.slider("w_popularity", 0.0, 2.0, 0.15, 0.05)

        st.subheader("NL → Location filter")
        _semantic_ok = importlib.util.find_spec("sentence_transformers") is not None
        if not _semantic_ok:
            st.caption("`sentence-transformers` not installed: only rule-based state parsing will be used.")
        use_minilm_state = st.checkbox(
            "If no state is detected, use MiniLM to infer a state from dataset states",
            value=_semantic_ok,
            disabled=not _semantic_ok,
            help="Uses sentence-transformers / all-MiniLM-L6-v2. First run downloads ~90MB model.",
        )

    @st.cache_resource
    def _load_minilm_cached():
        from app.search.semantic_filters import load_minilm_model

        return load_minilm_model()

    tab_nl, tab_adv = st.tabs(["Natural language query", "Advanced filters"])

    with tab_nl:
        st.subheader("Describe what you want")
        nl_query = st.text_area(
            "Example: cheap sushi near Philadelphia within 5 km",
            value="",
            height=100,
            help="Rule-based parser extracts cuisine, budget, landmarks / radius when possible.",
        )

    with tab_adv:
        st.subheader("Choose cuisines (multi-select)")
        cuisine_options = ["Sushi", "Steakhouse", "Korean", "Fast Food", "Chinese", "Burger", "Healthy"]
        cuisines = st.multiselect("Cuisines", options=cuisine_options, default=[])

        st.subheader("Keyword Search")
        keywords = st.text_input(
            "Extra keywords (optional). Combined with natural language and cuisines.",
            value="",
        )
        top_k = st.slider("Top-K", min_value=3, max_value=30, value=10, step=1)

        state_options = load_state_options()
        state = st.selectbox(
            "State (optional)",
            options=["All"] + state_options,
            index=0,
            help="Only includes states present in this dataset.",
        )
        city = st.text_input("City (optional, exact match; leave blank for no filter)", value="")

    do_search = st.button("Get Recommendations", type="primary")

    if do_search:
        try:
            index = load_index(force_rebuild=rebuild)
            parsed = parse_query(nl_query or "")

            city_filter = city.strip() if city.strip() else None
            state_filter = None if state == "All" else state.strip()

            effective_state = state_filter
            semantic_state_note = ""
            if not effective_state and parsed.state_code:
                effective_state = str(parsed.state_code).strip().upper()
            if (
                not effective_state
                and use_minilm_state
                and (nl_query or "").strip()
            ):
                try:
                    enc = _load_minilm_cached()
                    st_codes = load_state_options()
                    from app.search.semantic_filters import infer_state_minilm

                    guessed, sc = infer_state_minilm(nl_query, st_codes, enc)
                    if guessed:
                        effective_state = guessed
                        semantic_state_note = f"MiniLM inferred state **{guessed}** (cosine **{sc:.3f}**)."
                except Exception as ex:
                    semantic_state_note = f"MiniLM inference failed (ignored): `{ex}`"

            # Map parser cuisine labels to advanced-tab multiselect keys (category rules).
            cuisine_from_nl: list[str] = []
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
            if parsed.cuisine and parsed.cuisine in _nl_map:
                cuisine_from_nl.append(_nl_map[parsed.cuisine])

            effective_cuisines = list(dict.fromkeys(list(cuisines or []) + cuisine_from_nl))
            effective_cuisines = effective_cuisines or None

            # Quick coverage check: state/city in this dataset?
            candidates = index.meta
            if effective_state:
                state_norm_q = effective_state.upper()
                candidates = candidates[candidates["state_norm"].astype(str) == state_norm_q]
            if city_filter:
                city_norm_q = city_filter.lower()
                candidates = candidates[candidates["city_norm"].astype(str) == city_norm_q]

            if len(candidates) == 0:
                # Provide a helpful dataset-coverage message instead of "rebuild index".
                available_states = sorted(index.meta["state"].astype(str).str.strip().str.upper().unique().tolist())
                st.session_state["rec_ok"] = False
                st.warning(
                    "No matching restaurants in the current dataset for the selected state/city. "
                    "This is a data-coverage issue (rebuilding the index will not help). "
                    f"Available states include: {', '.join(available_states[:10])}..."
                )
                st.stop()

            # semantic_query already removes budget/location/radius noise (see `app/search/query_parser.py`).
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
                query_text = (nl_query or "").strip() or "restaurant"

            _result = retriever.recommend_keywords(
                keywords=query_text,
                index=index,
                state=effective_state,
                city=city_filter,
                cuisines=effective_cuisines,
                top_k=top_k,
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
            st.session_state["rec_df"] = _result.copy()
            st.session_state["rec_parsed"] = asdict(parsed)
            st.session_state["rec_query_text"] = query_text
            st.session_state["rec_semantic_note"] = semantic_state_note
            st.session_state["rec_effective_state"] = effective_state
            st.session_state["rec_ok"] = True

        except FileNotFoundError as e:
            st.session_state["rec_ok"] = False
            st.error(f"Data file not found: {e}")
        except Exception as e:
            st.session_state["rec_ok"] = False
            st.exception(e)

    if st.session_state.get("rec_ok") and "rec_df" in st.session_state:
        from app.search.query_parser import ParsedQuery
        from streamlit_folium import st_folium

        result = st.session_state["rec_df"]
        parsed = ParsedQuery(**st.session_state["rec_parsed"])
        query_text = st.session_state["rec_query_text"]
        semantic_state_note = st.session_state.get("rec_semantic_note", "")
        effective_state = st.session_state.get("rec_effective_state")

        with st.expander("Parsed constraints (rule-based + applied filters)", expanded=False):
            st.json(parsed.to_dict())
            if effective_state:
                st.markdown(f"**Applied state filter:** `{effective_state}`")
            if semantic_state_note:
                st.markdown(semantic_state_note)
            st.caption(
                "Switching tabs will rerun the page, but results come from the last **Get Recommendations** run. "
                "After changing filters, click **Get Recommendations** again. "
                "State names/abbreviations in NL (e.g., CA/California) are merged into the filter when possible."
            )

        st.subheader("Recommendations (Top-K)")
        if len(result) == 0:
            st.warning(
                "No matching restaurants under the current filters. "
                "Try: clear city/state, widen radius, remove cuisine filters, or pick another state "
                "present in this dataset."
            )
        else:
            def _fmt_price_tier(x) -> str:
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return "N/A"
                try:
                    t = int(round(float(x)))
                except (TypeError, ValueError):
                    return "N/A"
                t = max(1, min(t, 4))
                return "$" * t

            n_res = len(result)
            sig = _result_signature(result)

            def _tab_title(i: int) -> str:
                nm = str(result.iloc[i].get("name", ""))
                if len(nm) <= 22:
                    return f"{i + 1}. {nm}"
                return f"{i + 1}. {nm[:20]}…"

            st.caption(
                "Each tab is one recommended restaurant. Switching tabs shows its details, and the map centers "
                "on that restaurant (red highlight); light-blue markers are the other Top-K results."
            )

            tab_labels = [_tab_title(i) for i in range(n_res)]
            rec_tabs = st.tabs(tab_labels)

            def _render_rec_card(rank: int, row: pd.Series) -> None:
                lat_q = pd.to_numeric(row.get("latitude"), errors="coerce")
                lon_q = pd.to_numeric(row.get("longitude"), errors="coerce")
                name_esc = str(row.get("name", ""))
                if pd.notna(lat_q) and pd.notna(lon_q):
                    gmaps = f"https://www.google.com/maps/search/?api=1&query={float(lat_q)},{float(lon_q)}"
                    st.markdown(f"### #{rank}. {name_esc} [🔗]({gmaps})")
                else:
                    st.markdown(f"### #{rank}. {name_esc}")
                st.write(
                    f"Address: {row.get('address', 'N/A')}, {row['city']}, {row['state']}"
                )
                fs = float(row["final_score"]) if pd.notna(row.get("final_score")) else 0.0
                sim = float(row["similarity"]) if pd.notna(row.get("similarity")) else 0.0
                pm = row.get("price_match")
                pm_s = f"{float(pm):.2f}" if pm is not None and pd.notna(pm) else "—"
                dist = row.get("distance_km")
                dist_s = (
                    f"{float(dist):.1f} km"
                    if dist is not None and pd.notna(dist)
                    else "—"
                )
                st.write(
                    f"Rating: {float(row['stars']):.1f} stars "
                    f"({int(row['review_count'])} reviews) · "
                    f"Price tier: {_fmt_price_tier(row.get('price_tier'))} "
                    f"(budget match {pm_s}) · Distance: {dist_s}"
                )
                st.write(f"Scores — final: {fs:.3f}, text similarity: {sim:.3f}")
                st.progress(min(max(float(row["stars"]) / 5.0, 0.0), 1.0))
                insight = generate_insight(row, parsed, query_text)
                with st.expander("Why this result?", expanded=False):
                    st.write(insight["why"])
                    st.markdown(f"**Pros:** {insight['pros']}")
                    st.markdown(f"**Cons:** {insight['cons']}")

            for i, tab in enumerate(rec_tabs):
                with tab:
                    row_i = result.iloc[i]
                    _render_rec_card(i + 1, row_i)
                    st.markdown("#### Map")
                    if "latitude" in result.columns and "longitude" in result.columns:
                        m = _build_recommendation_folium_map(result, i)
                        st_folium(
                            m,
                            width=700,
                            height=395,
                            returned_objects=[],
                            key=f"rec_folium_{sig}_{i}",
                            use_container_width=True,
                        )
                    else:
                        st.info("Latitude/longitude missing; cannot render map.")

            st.write(
                "How it works: TF-IDF cosine similarity on aggregated review text, then a multi-factor "
                "score (sidebar weights) over normalized similarity, stars, price fit vs budget, "
                "distance to a reference point when provided, and log review count."
            )

if __name__ == "__main__":
    main()
