from pathlib import Path

import pytest

from services.merchant_inference import artifact_paths, predict_merchant_site, resolve_repo_root

REPO = Path(__file__).resolve().parents[1]
surv, rat = artifact_paths(REPO)


@pytest.mark.skipif(not surv.is_file() or not rat.is_file(), reason="缺少 global_*_model.pkl")
def test_predict_philly_coffee_fastfood():
    r = predict_merchant_site(
        city="Philadelphia",
        lat=39.9526,
        lon=-75.1652,
        selected_category_columns=["cat_coffee_&_tea", "cat_fast_food"],
        repo_root=resolve_repo_root(REPO),
    )
    assert 0.0 <= r.survival_probability <= 1.0
    assert 0.0 <= r.predicted_stars <= 5.5
    assert r.reference_row_count >= 10
    assert "count_all_3.0km" in r.metrics or r.metrics


def test_predict_requires_valid_category():
    if not surv.is_file():
        pytest.skip("no pkl")
    with pytest.raises(ValueError, match="未知品类列"):
        predict_merchant_site(
            city="Philadelphia",
            lat=39.9526,
            lon=-75.1652,
            selected_category_columns=["cat___definitely_missing___"],
            repo_root=REPO,
        )
