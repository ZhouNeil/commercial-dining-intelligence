import os
import sys
import joblib
import pandas as pd
import numpy as np
from pipelines.spatial_feature_engineer import SpatialFeatureEngineer

def run_simulation(city="Philadelphia"):
    print(f"--- MOCKING LIVE USER SELECTION ON FRONTEND ---")
    print(f"1. Loading reference data for {city} into server memory...")
    
    # In production, the backend would just load the raw output of YelpDataProcessor
    # Here, we can cheat and load the global train file, but only keep Philly rows
    global_ref = pd.read_csv("../train_spatial.csv")
    
    # Try to filter by city if the column exists, otherwise just use a subset (e.g. first 2000 rows as 'local context')
    if 'city' in global_ref.columns:
        local_ref = global_ref[global_ref['city'].str.lower() == city.lower()]
    else:
        # Fallback if cleaner dropped the city column: arbitrarily subset to simulate a single city reference map
        local_ref = global_ref.head(2000)
    
    print(f"   Loaded {len(local_ref)} local restaurants perfectly.")

    # A user drops a pin via maps API for their new concept
    # Coordinates in central Philadelphia roughly
    user_target_coord = (39.9526, -75.1652) 
    
    # They specify it is a fast-food coffee shop
    cat_cols = [c for c in local_ref.columns if c.startswith('cat_')]
    user_target_categories = np.zeros(len(cat_cols))
    
    # We will simulate turning on a couple category tags manually
    for idx, col in enumerate(cat_cols):
        if col in ['cat_coffee_&_tea', 'cat_fast_food']:
            user_target_categories[idx] = 1.0

    print(f"2. User placed pin on map at coordinates {user_target_coord}")
    print(f"   Concept: Coffee shop / Fast Food")

    print("\n--- INFERENCE LAYER ---")
    print("3. Executing engineer_single_target (BallTree & Custom KNN live execution)...")
    spatial_engineer = SpatialFeatureEngineer(None)
    live_features_df = spatial_engineer.engineer_single_target(
        user_target_coord, 
        user_target_categories, 
        local_ref
    )
    
    # Assemble the final unified DataFrame to pass to model
    # The baseline prediction model requires exact columns
    # In production, ensure the features mapped here perfectly correspond to 'current_best_features'
    # For now, let's just grab the feature layout directly from the trained model's expectation if possible
    # Note: For prediction, HistGradientBoosting expects matching columns.
    survival_model = joblib.load("models/artifacts/global_survival_model.pkl")
    
    # We create a dummy matching df populated with zeros initially
    model_df = pd.DataFrame(0.0, index=[0], columns=survival_model.feature_names_in_)
    
    # Safely slot in the spatial features generated
    for col in live_features_df.columns:
        if col in model_df.columns:
            model_df[col] = live_features_df[col].values
            
    # Safely slot in the category definitions the user manually enabled
    for idx, col in enumerate(cat_cols):
        if col in model_df.columns:
            model_df[col] = user_target_categories[idx]

    print("4. Features Extracted! Sample metrics generated instantly:")
    print(f"   - Local Competitors within 3km: {live_features_df['count_all_3.0km'].iloc[0]}")
    print(f"   - Predicted Similar Tourist Peers Rate: {live_features_df['survival_top5_similar'].iloc[0]:.2f}")

    print("\n--- PREDICTION ---")
    surv_prob = survival_model.predict_proba(model_df)[:, 1][0]
    rating_model = joblib.load("models/artifacts/global_rating_model.pkl")
    
    # Rating model has a different configured feature set than Survival model
    model_df_reg = pd.DataFrame(0.0, index=[0], columns=rating_model.feature_names_in_)
    for col in live_features_df.columns:
        if col in model_df_reg.columns:
            model_df_reg[col] = live_features_df[col].values
    for idx, col in enumerate(cat_cols):
        if col in model_df_reg.columns:
            model_df_reg[col] = user_target_categories[idx]
            
    stars_pred = rating_model.predict(model_df_reg)[0]
    
    print(f"🔥 FINAL VERDICT 🔥")
    print(f"   Survival Probability: {surv_prob * 100:.1f}%")
    print(f"   Predicted Star Rating: {stars_pred:.2f} / 5.0")
    print("-----------------------------------------------")

if __name__ == "__main__":
    run_simulation()
