import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium

def load_and_preprocess_data(file_path):
    """
    Step 1: Load and clean data (Frontend should cache this with @st.cache_data)
    Returns the cleaned DataFrame and the scaled feature matrix X_scaled.
    """
    df = pd.read_csv(file_path)
    
    # Feature selection (Spatial + Commercial DNA)
    features = [
        'latitude', 'longitude', 'stars', 'review_count',
        'PC1_Cafe_Score', 'PC2_Nightlife_Score', 'PC3_Brunch_Score', 'PC4_Pizza_Score'
    ]
    df = df[features].dropna()
    
    # Handle extreme values (Long-tail distribution of reviews)
    df['review_count'] = np.log1p(df['review_count'])
    
    # Feature Scaling is crucial so location doesn't overpower business DNA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    return df, X_scaled

def run_kmeans_clustering(df, X_scaled, n_clusters=4):
    """
    Step 2: Execute clustering (Frontend can pass n_clusters dynamically via a slider)
    Returns the DataFrame with cluster labels and the aggregated summary table.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Generate the commercial insight summary (Mean of features per cluster)
    cluster_summary = df_clustered.groupby('cluster').mean()
    
    return df_clustered, cluster_summary

def generate_cluster_map(df_clustered):
    """
    Step 3: Generate the interactive Folium Map
    Returns a Folium Map object for the frontend to render using st_folium().
    """
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightgray', 'pink']
    
    m = folium.Map(
        location=[df_clustered['latitude'].mean(), df_clustered['longitude'].mean()],
        zoom_start=12
    )
    
    for _, row in df_clustered.iterrows():
        cluster_idx = int(row['cluster'])
        color = color_list[cluster_idx % len(color_list)]
        
        # Enhanced Tooltip for the UI
        tooltip_html = f"""
        <b>Cluster ID:</b> {cluster_idx}<br>
        <b>Avg Stars:</b> {row['stars']:.1f}<br>
        <b>Reviews (Log):</b> {row['review_count']:.2f}
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=tooltip_html
        ).add_to(m)
        
    return m

def generate_business_insight(cluster_summary, target_category='PC4_Pizza_Score'):
    """
    Step 4: Dynamically generate business location advice
    Uses heuristic rules to find the highest/lowest scoring clusters for a specific niche.
    """
    # 1. Traffic Hub (Highest average review count)
    busiest_cluster = cluster_summary['review_count'].idxmax()
    
    # 2. Red Ocean (Highest competition for the selected category)
    red_ocean_cluster = cluster_summary[target_category].idxmax()
    
    # 3. Blue Ocean (Lowest competition / Market Gap for the selected category)
    blue_ocean_cluster = cluster_summary[target_category].idxmin()
    
    # Dynamic Markdown Generation
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

"""
# ==========================================
# LOCAL TESTING BLOCK (Run this file directly to test)
# ==========================================
if __name__ == "__main__":
    test_file_path = "data/kmeans_ready_features.csv" 
    
    try:
        print("Loading data...")
        df, X_scaled = load_and_preprocess_data(test_file_path)
        
        print("Running Clustering...")
        df_clustered, cluster_summary = run_kmeans_clustering(df, X_scaled, n_clusters=4)
        
        print("\n--- Testing Business Insight Generator ---")
        # Let's pretend the user selected "PC4_Pizza_Score" from the frontend dropdown
        test_insight = generate_business_insight(cluster_summary, target_category="PC4_Pizza_Score")
        
        print(test_insight)
        print("\nSuccess! The API is ready for the frontend.")
        
    except FileNotFoundError:
        print(f"Error: Please make sure '{test_file_path}' is in the same folder, or update the path.")
"""
