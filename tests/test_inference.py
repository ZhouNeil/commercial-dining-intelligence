import os

from services.merchant_inference import predict_merchant_site, resolve_repo_root


def run_simulation(city="Philadelphia"):
    print(f"--- MOCKING LIVE USER SELECTION ON FRONTEND ---")
    print(f"1. Loading reference data for {city} into server memory...")

    # A user drops a pin via maps API for their new concept
    if city.lower() == "philadelphia":
        user_target_coord = (39.9526, -75.1652) # Central Philadelphia
    elif city.lower() == "tucson":
        user_target_coord = (32.2226, -110.9747) # Central Tucson
    else:
        user_target_coord = (39.9526, -75.1652) # Default
    
    # They specify it is a fast-food coffee shop
    cat_cols = [c for c in local_ref.columns if c.startswith('cat_')]
    user_target_categories = np.zeros(len(cat_cols))
    
    # We will simulate turning on a couple category tags manually
    for idx, col in enumerate(cat_cols):
        if col in ['cat_coffee_&_tea', 'cat_fast_food']:
            user_target_categories[idx] = 1.0

    print(f"   Loaded {r.reference_row_count} local restaurants perfectly.")
    print(f"2. User placed pin on map at coordinates (39.9526, -75.1652)")
    print(f"   Concept: Coffee shop / Fast Food")

    print("\n--- INFERENCE LAYER ---")
    print("3. engineer_single_target (via services.merchant_inference)...")
    km = r.metrics.get("count_all_3.0km", float("nan"))
    sim = r.metrics.get("survival_top5_similar", float("nan"))
    print("4. Features Extracted! Sample metrics generated instantly:")
    print(f"   - Local Competitors within 3km: {km}")
    print(f"   - Predicted Similar Tourist Peers Rate: {sim:.2f}")

    print("\n--- PREDICTION ---")
    print(f"🔥 FINAL VERDICT 🔥")
    print(f"   Survival Probability: {r.survival_probability * 100:.1f}%")
    print(f"   Predicted Star Rating: {r.predicted_stars:.2f} / 5.0")
    print("-----------------------------------------------")


if __name__ == "__main__":
    run_simulation(city="Tucson")
