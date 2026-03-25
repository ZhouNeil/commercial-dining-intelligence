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
        "Enter restaurant type/keywords and we will return a Top-K list of similar restaurants "
        "(based on review text retrieval)."
    )

    base_dir = Path(__file__).resolve().parents[1]
    # Ensure project root is importable when Streamlit runs from different CWDs.
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    # Import after sys.path adjustment (Streamlit may change CWD).
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

    # UI controls
    st.subheader("Choose cuisines (multi-select)")
    cuisine_options = ["Sushi", "Steakhouse", "Korean", "Fast Food", "Chinese", "Burger", "Healthy"]
    cuisines = st.multiselect("Cuisines", options=cuisine_options, default=[])

    st.subheader("Keyword Search")
    keywords = st.text_input(
        "Restaurant keywords (optional). If empty, we will use selected cuisines as query.",
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
            city_filter = city.strip() if city.strip() else None
            state_filter = None if state == "All" else state.strip()

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

            query_text = keywords
            if not isinstance(query_text, str):
                query_text = ""
            query_text = query_text.strip()
            if not query_text:
                # Use cuisines as query if keywords are empty
                if cuisines:
                    query_text = " ".join(cuisines)
                else:
                    query_text = "restaurant"

            result = retriever.recommend_keywords(
                keywords=query_text,
                index=index,
                state=state_filter,
                city=city_filter,
                cuisines=cuisines if cuisines else None,
                top_k=top_k,
                alpha=1.0,
                beta=0.2,
            )

            st.subheader("Recommendations (Top-K)")
            if len(result) == 0:
                st.warning(
                    "No matching restaurants found under the current state/city filters "
                    "(strict but case/space-insensitive matching). "
                    "Try: 1) leaving city/state blank, or 2) clicking "
                    "“Force Rebuild Index” in the sidebar."
                )
            else:
                # Left: card list; Right: map preview
                left_col, right_col = st.columns([2, 1])

                with left_col:
                    for rank, (_, row) in enumerate(result.iterrows(), start=1):
                        with st.container(border=True):
                            st.markdown(f"### {rank}. {row['name']}")
                            st.write(
                                f"Address: {row.get('address', 'N/A')}, {row['city']}, {row['state']}"
                            )
                            st.write(
                                f"Rating: {float(row['stars']):.1f} stars "
                                f"({int(row['review_count'])} reviews)"
                            )
                            st.progress(min(max(float(row["stars"]) / 5.0, 0.0), 1.0))
                            st.caption("Source: TF-IDF text similarity + stars prior reranking")

                with right_col:
                    st.markdown("### Location Preview")
                    if "latitude" in result.columns and "longitude" in result.columns:
                        map_df = result[["latitude", "longitude"]].copy()
                        map_df.columns = ["lat", "lon"]
                        st.map(map_df)

                st.write(
                    "How it works: cosine similarity over TF-IDF vectors built from restaurant review text, "
                    "then a lightweight re-ranking prior from the restaurant's `stars`."
                )
        except FileNotFoundError as e:
            st.error(f"Data file not found: {e}")
        except Exception as e:
            st.exception(e)

if __name__ == "__main__":
    main()
