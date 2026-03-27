import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import folium


REQUIRED_READY_COLS = [
    'latitude', 'longitude', 'stars', 'review_count', 'price_range',
    'PC1_Cafe_Score', 'PC2_Nightlife_Score', 'PC3_Brunch_Score', 'PC4_Pizza_Score'
]

REQUIRED_BASE_COLS = ['latitude', 'longitude', 'stars', 'review_count']
REQUIRED_PRICE_COL = 'attr_restaurantspricerange2'


def build_kmeans_ready_features_from_output(df_raw):
    """
    Build the 9-column k-means feature matrix directly from output_philly.csv.

    Strict mode:
    - only uses pre-split attr_ columns for price_range
    - only uses pre-split cat_ columns for category PCA
    - raises explicit errors when required columns are missing or invalid
    """
    df = df_raw.copy()

    # 1) validate base columns
    missing_base_cols = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing_base_cols:
        raise ValueError(
            f"Missing required base columns in output data: {missing_base_cols}"
        )

    # 2) validate and build price_range from pre-split attr_ column only
    if REQUIRED_PRICE_COL not in df.columns:
        raise ValueError(
            f"Missing required price column in output data: {REQUIRED_PRICE_COL}"
        )

    price_range = pd.to_numeric(df[REQUIRED_PRICE_COL], errors='coerce')
    if price_range.isna().any():
        bad_count = int(price_range.isna().sum())
        raise ValueError(
            f"Column {REQUIRED_PRICE_COL} contains {bad_count} null or non-numeric values. "
            "Please fix this in the upstream feature pipeline before clustering."
        )
    df['price_range'] = price_range.astype(int)

    # 3) validate and build category PCA from pre-split cat_ columns only
    cat_cols = [c for c in df.columns if c.startswith('cat_')]
    if not cat_cols:
        raise ValueError(
            "No pre-split category columns found. Expected columns with prefix 'cat_'."
        )

    cat_matrix = df[cat_cols].apply(pd.to_numeric, errors='coerce')
    if cat_matrix.isna().any().any():
        bad_cols = cat_matrix.columns[cat_matrix.isna().any()].tolist()
        raise ValueError(
            f"Some cat_ columns contain null or non-numeric values: {bad_cols}. "
            "Please fix this in the upstream feature pipeline before clustering."
        )

    # PCA requires n_components <= min(n_samples, n_features)
    n_samples = len(df)
    n_features = cat_matrix.shape[1]
    n_components = min(4, n_samples, n_features)

    pca_scores = np.zeros((len(df), 4))
    if n_components > 0:
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(cat_matrix)
        pca_scores[:, :n_components] = transformed

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

    # 4) final 9-column feature matrix with explicit validation
    required_cols = ['latitude', 'longitude', 'stars', 'review_count', 'price_range']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for k-means features: {', '.join(missing_cols)}"
        )

    base_features = df[required_cols].copy()
    df_kmeans_ready = pd.concat([base_features, df_pca], axis=1)

    return df_kmeans_ready



def load_and_preprocess_data(file_path):
    """
    Step 1: Load data and prepare the scaled feature matrix X_scaled.

    Supports both:
    - output_philly.csv (strictly requires pre-split attr_ and cat_ columns)
    - kmeans_ready_features.csv (already prepared 9-column dataset)
    """
    df = pd.read_csv(file_path)

    if not all(col in df.columns for col in REQUIRED_READY_COLS):
        df = build_kmeans_ready_features_from_output(df)

    missing_features = [col for col in REQUIRED_READY_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Input data is missing required columns: {missing_features}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[REQUIRED_READY_COLS].dropna().copy()

    # Handle extreme values (long-tail distribution of reviews)
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


