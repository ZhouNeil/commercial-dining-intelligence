import ast
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import folium


def extract_price(attr_str):
    """Extract RestaurantsPriceRange2 from the raw attributes column."""
    if pd.isna(attr_str) or str(attr_str).strip() in ["{}", "None", "", "nan"]:
        return 2
    try:
        attr_dict = ast.literal_eval(attr_str) if isinstance(attr_str, str) else attr_str
        if not isinstance(attr_dict, dict):
            return 2
        price = attr_dict.get('RestaurantsPriceRange2')
        if price is None or str(price).strip().lower() in ['none', 'nan', '']:
            return 2
        return int(price)
    except Exception:
        return 2


def build_kmeans_ready_features_from_output(df_raw):
    """
    Build the same 9-column feature matrix as kmeans_ready_features.csv
    directly from output_philly.csv.

    Uses existing cat_ columns when available, which is cleaner and more robust
    than reparsing the categories string column.
    """
    df = df_raw.copy()

    # 1) price_range
    if 'price_range' not in df.columns:
        if 'attr_restaurantspricerange2' in df.columns:
            # Prefer pre-split attribute column if present
            df['price_range'] = pd.to_numeric(df['attr_restaurantspricerange2'], errors='coerce').fillna(2).astype(int)
        elif 'attributes' in df.columns:
            df['price_range'] = df['attributes'].apply(extract_price)
        else:
            df['price_range'] = 2

    # 2) category PCA input
    cat_cols = [c for c in df.columns if c.startswith('cat_')]
    if cat_cols:
        cat_matrix = df[cat_cols].fillna(0)
    else:
        # Fallback: parse raw categories string if cat_ columns do not exist
        categories = df.get('categories', pd.Series('', index=df.index)).fillna('').astype(str).str.lower()
        cat_matrix = categories.str.get_dummies(sep=', ')

    # Ensure numeric matrix
    cat_matrix = cat_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)

    # PCA with up to 4 components
    n_components = min(4, cat_matrix.shape[1])
    if n_components == 0:
        pca_scores = np.zeros((len(df), 4))
    else:
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(cat_matrix)
        if n_components < 4:
            pca_scores = np.zeros((len(df), 4))
            pca_scores[:, :n_components] = transformed
        else:
            pca_scores = transformed

    df_pca = pd.DataFrame(
        pca_scores,
        columns=[
            'PC1_Cafe_Score',
            'PC2_Nightlife_Score',
            'PC3_Brunch_Score',
            'PC4_Pizza_Score',
        ],
        index=df.index,
    )

    # 3) final 9-column feature matrix
    base_features = df[['latitude', 'longitude', 'stars', 'review_count', 'price_range']].copy()
    df_kmeans_ready = pd.concat([base_features, df_pca], axis=1)

    return df_kmeans_ready


def load_and_preprocess_data(file_path):
    """
    Step 1: Load data and prepare the scaled feature matrix X_scaled.

    Supports both:
    - output_philly.csv (raw engineered dataset with cat_ columns)
    - kmeans_ready_features.csv (already prepared 9-column dataset)
    """
    df = pd.read_csv(file_path)

    # If raw output file is given, construct the 9-column kmeans-ready matrix first
    required_ready_cols = [
        'latitude', 'longitude', 'stars', 'review_count', 'price_range',
        'PC1_Cafe_Score', 'PC2_Nightlife_Score', 'PC3_Brunch_Score', 'PC4_Pizza_Score'
    ]
    if not all(col in df.columns for col in required_ready_cols):
        df = build_kmeans_ready_features_from_output(df)

    features = required_ready_cols

    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Input data is missing required columns: {missing_features}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[features].dropna().copy()

    # Handle extreme values (Long-tail distribution of reviews)
    df['review_count'] = np.log1p(df['review_count'])

    # Scale all 9 clustering features, including price_range
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    return df, X_scaled


def run_kmeans_clustering(df, X_scaled, n_clusters=4):
    """
    Step 2: Execute clustering.
    Returns the DataFrame with cluster labels and the aggregated summary table.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_summary = df_clustered.groupby('cluster').mean()

    return df_clustered, cluster_summary


def generate_cluster_map(df_clustered):
    """
    Step 3: Generate the interactive Folium Map.
    """
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightgray', 'pink']

    m = folium.Map(
        location=[df_clustered['latitude'].mean(), df_clustered['longitude'].mean()],
        zoom_start=12,
    )

    for _, row in df_clustered.iterrows():
        cluster_idx = int(row['cluster'])
        color = color_list[cluster_idx % len(color_list)]

        tooltip_html = f"""
        <b>Cluster ID:</b> {cluster_idx}<br>
        <b>Shop Rating:</b> {row['stars']:.1f}<br>
        <b>Shop Reviews (Log):</b> {row['review_count']:.2f}<br>
        <b>Price Range:</b> {row['price_range']}
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=tooltip_html,
        ).add_to(m)

    return m


def generate_business_insight(cluster_summary, target_category='PC4_Pizza_Score'):
    """
    Step 4: Dynamically generate business location advice.
    """
    busiest_cluster = cluster_summary['review_count'].idxmax()
    red_ocean_cluster = cluster_summary[target_category].idxmax()
    blue_ocean_cluster = cluster_summary[target_category].idxmin()

    insight_markdown = f"""
### Smart Location Intelligence Report

**Avoid the Red Ocean (Saturated Market): Cluster {red_ocean_cluster}**
> Our system detects that this zone has the highest concentration of `{target_category}` features. This area is heavily saturated with direct competitors. Unless you have a massive brand advantage, it is highly recommended to avoid opening a new shop here.

**Dive into the Blue Ocean (Market Gap): Cluster {blue_ocean_cluster}**
> This is your prime strategic gap! This cluster shows almost zero presence of your target niche. By establishing your business here, you can capture the local demographic without fighting established competitors.

**Traffic Hub (High Footfall): Cluster {busiest_cluster}**
> If your business model relies heavily on walk-in traffic, Cluster {busiest_cluster} has the highest historical customer engagement. Be prepared, however, as high traffic usually correlates with premium real estate costs.
    """

    return insight_markdown


if __name__ == "__main__":
    test_file_path = "data/output_philly.csv"

    try:
        print("Loading data...")
        df, X_scaled = load_and_preprocess_data(test_file_path)
        print("Prepared feature columns:", list(df.columns))
        print("Feature matrix shape:", df.shape)

        print("Running clustering...")
        df_clustered, cluster_summary = run_kmeans_clustering(df, X_scaled, n_clusters=4)

        print("\n--- Testing Business Insight Generator ---")
        test_insight = generate_business_insight(cluster_summary, target_category="PC4_Pizza_Score")
        print(test_insight)
        print("\nSuccess! The API is ready for the frontend.")

    except FileNotFoundError:
        print(f"Error: Please make sure '{test_file_path}' is in the same folder, or update the path.")
