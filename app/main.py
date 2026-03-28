import streamlit as st

from pathlib import Path
import sys
import pandas as pd


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
    from app.query_parser import parse_query
    from app.insights import generate_insight
    from app.retrieval import TouristRetrieval
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
        # Build valid state options from cleaned business table.
        p = base_dir / "data" / "cleaned" / "business_dining.csv"
        df = pd.read_csv(p, usecols=["state"])
        states = (
            df["state"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": ""})
        )
        states = sorted([s for s in states.unique().tolist() if s and s != ""])
        return states

    with st.sidebar:
        st.header("Search Settings")
        rebuild = st.button(
            "Force Rebuild Index",
            help="Rebuild TF-IDF vectors from `data/cleaned` (slower, but improves matching).",
        )
        st.subheader("Ranking weights (d1doc)")
        w_semantic = st.slider("w_semantic", 0.0, 2.0, 1.0, 0.05)
        w_rating = st.slider("w_rating", 0.0, 2.0, 0.25, 0.05)
        w_price = st.slider("w_price", 0.0, 2.0, 0.2, 0.05)
        w_distance = st.slider("w_distance", 0.0, 2.0, 0.25, 0.05)
        w_popularity = st.slider("w_popularity", 0.0, 2.0, 0.15, 0.05)

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
            if state_filter:
                state_norm_q = state_filter.upper()
                candidates = candidates[candidates["state_norm"].astype(str) == state_norm_q]
            if city_filter:
                city_norm_q = city_filter.lower()
                candidates = candidates[candidates["city_norm"].astype(str) == city_norm_q]

            if len(candidates) == 0:
                # Provide a helpful dataset-coverage message instead of "rebuild index".
                available_states = sorted(index.meta["state"].astype(str).str.strip().str.upper().unique().tolist())
                st.warning(
                    "No matching restaurants in the current dataset for the selected state/city. "
                    "This is a data-coverage issue (rebuilding the index will not help). "
                    f"Available states include: {', '.join(available_states[:10])}..."
                )
                st.stop()

            # semantic_query already removes budget/location/radius noise (see `app/query_parser.py`).
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

            with st.expander("Parsed constraints (rule-based)", expanded=False):
                st.json(parsed.to_dict())
                st.caption(
                    "Landmarks set a reference point for distance scoring and optional radius filtering. "
                    "This academic subset may omit some states (e.g. NY); pick a state that exists in the data."
                )

            result = retriever.recommend_keywords(
                keywords=query_text,
                index=index,
                state=state_filter,
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

            st.subheader("Recommendations (Top-K)")
            if len(result) == 0:
                st.warning(
                    "No matching restaurants under the current filters. "
                    "Try: clear city/state, widen radius, remove cuisine filters, or pick another state "
                    "present in this dataset."
                )
            else:
                # Left: card list; Right: map preview
                left_col, right_col = st.columns([2, 1])

                def _fmt_price_tier(x) -> str:
                    if x is None or (isinstance(x, float) and pd.isna(x)):
                        return "N/A"
                    try:
                        t = int(round(float(x)))
                    except (TypeError, ValueError):
                        return "N/A"
                    t = max(1, min(t, 4))
                    return "$" * t

                with left_col:
                    for rank, (_, row) in enumerate(result.iterrows(), start=1):
                        with st.container(border=True):
                            st.markdown(f"### {rank}. {row['name']}")
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
                            st.write(
                                f"Scores — final: {fs:.3f}, text similarity: {sim:.3f}"
                            )
                            st.progress(min(max(float(row["stars"]) / 5.0, 0.0), 1.0))
                            insight = generate_insight(row, parsed, query_text)
                            with st.expander("Why this result?", expanded=False):
                                st.write(insight["why"])
                                st.markdown(f"**Pros:** {insight['pros']}")
                                st.markdown(f"**Cons:** {insight['cons']}")

                with right_col:
                    st.markdown("### Location Preview")
                    if "latitude" in result.columns and "longitude" in result.columns:
                        map_df = result[["latitude", "longitude"]].copy()
                        map_df.columns = ["lat", "lon"]
                        st.map(map_df)

                st.write(
                    "How it works: TF-IDF cosine similarity on aggregated review text, then a multi-factor "
                    "score (sidebar weights) over normalized similarity, stars, price fit vs budget, "
                    "distance to a reference point when provided, and log review count."
                )
        except FileNotFoundError as e:
            st.error(f"Data file not found: {e}")
        except Exception as e:
            st.exception(e)

if __name__ == "__main__":
    main()
