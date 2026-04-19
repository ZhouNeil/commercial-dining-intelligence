"""
Online Feature Aggregation Engine
Owner: Ethan (Role 1)
Requirement: Calculate spatial features for Merchant Mode AND dynamic features for Tourist Mode RL.
"""
import pandas as pd
import numpy as np

# ==========================================
# Module A: Merchant Mode (Site Selection) - Retaining Original Logic
# ==========================================

def calculate_competition_density(neighbor_indices):
    """
    Calculate the business density within the selected radius.
    
    Parameters:
    neighbor_indices (list or np.ndarray): Output from CustomKNN.retrieve_by_radius.
    
    Returns:
    int: The total count of competing businesses nearby.
    """
    # TODO for Ethan: Return the length of the neighbor_indices list.
    pass

def calculate_local_avg_rating(neighbor_indices, df_historical):
    """
    Calculate the average rating of nearby businesses to gauge neighborhood quality.
    
    Parameters:
    neighbor_indices (list or np.ndarray): Output from CustomKNN.retrieve_by_radius.
    df_historical (pd.DataFrame): The cleaned city-specific Yelp dataframe.
    
    Returns:
    float: The average star rating of the neighbors.
    """
    # TODO for Ethan: Extract the 'stars' column for these indices and compute np.mean().
    pass

def build_merchant_feature_vector(query_lat_lon, neighbor_indices, df_historical, pca_features):
    """
    Construct the final 1D feature vector (X) to feed into the predictive model.
    """
    # TODO for Ethan: Combine density, avg rating, and PCA features into a single row.
    pass


# ==========================================
# Module B: Tourist Mode (RL Recommendation) - New Additions for Yihang
# ==========================================

def calculate_haversine_distance(user_lat, user_lon, candidates_df):
    """
    Calculate the actual physical Haversine distance (in kilometers) between the user and candidate restaurants.
    
    Parameters:
    user_lat, user_lon (float): The user's current latitude and longitude.
    candidates_df (pd.DataFrame): Top 100 restaurants retrieved by Guyu. Must contain 'latitude' and 'longitude' columns.
    
    Returns:
    pd.DataFrame: The original DataFrame with an appended 'distance_km' column.
    """
    # TODO for Ethan: Implement the Haversine formula to calculate the spherical distance between the user and each restaurant.
    pass

def enrich_and_normalize_for_rl(user_lat, user_lon, candidates_df):
    """
    Prepare and normalize features (scaled 0-1) for Yihang's Reinforcement Learning (RL) engine.
    
    Parameters:
    user_lat, user_lon (float): The user's current latitude and longitude.
    candidates_df (pd.DataFrame): The candidate restaurants from Guyu.
    
    Returns:
    pd.DataFrame: The DataFrame with two additional normalized columns for RL weighting:
                  - 'norm_distance_score': 0 to 1 (Closer distance = higher score. e.g., 1 is right next door, 0 is very far).
    """
    # TODO for Ethan: 
    # 1. Call `calculate_haversine_distance` to get the actual kilometers.
    # 2. Reverse-normalize the distance (Closer -> Higher score). You can use a Min-Max Scaler or cap it at a max threshold (e.g., 10km).
    # 4. Return the processed DataFrame.
    pass
