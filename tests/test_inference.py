import os

from services.merchant_inference import predict_merchant_site, resolve_repo_root


def run_simulation(city="Philadelphia"):
    print(f"--- MOCKING LIVE USER SELECTION ON FRONTEND ---")
    print(f"1. Loading reference data for {city} into server memory...")

    repo = resolve_repo_root()
    r = predict_merchant_site(
        city=city,
        lat=39.9526,
        lon=-75.1652,
        selected_category_columns=["cat_coffee_&_tea", "cat_fast_food"],
        repo_root=repo,
    )

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
    run_simulation()
