from pathlib import Path

import pytest

from services.merchant_inference import (
    artifact_paths,
    list_merchant_category_keys,
    predict_merchant_site,
    resolve_merchant_category_text,
    resolve_repo_root,
    spatial_train_csv_path,
)

REPO = Path(__file__).resolve().parents[1]
surv, rat = artifact_paths(REPO)


def _has_spatial_train() -> bool:
    try:
        return spatial_train_csv_path(REPO).is_file()
    except OSError:
        return False


@pytest.mark.skipif(not surv.is_file() or not rat.is_file(), reason="missing merchant .pkl artifacts")
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
    assert isinstance(r.inside_reference_hull, bool)
    assert "count_all_3.0km" in r.metrics or r.metrics


@pytest.mark.skipif(not _has_spatial_train(), reason="missing train_spatial.csv")
def test_resolve_burger_to_burgers():
    keys = resolve_merchant_category_text("burger", city="Philadelphia", repo_root=REPO, max_keys=2)
    assert "cat_burgers" in keys


@pytest.mark.skipif(not _has_spatial_train(), reason="missing train_spatial.csv")
def test_list_category_keys_philly():
    keys = list_merchant_category_keys(city="Philadelphia", repo_root=REPO)
    assert len(keys) >= 1
    assert all(k.startswith("cat_") for k in keys)


def test_predict_requires_valid_category():
    if not surv.is_file():
        pytest.skip("no pkl")
    with pytest.raises(ValueError, match="Unknown category column"):
        predict_merchant_site(
            city="Philadelphia",
            lat=39.9526,
            lon=-75.1652,
            selected_category_columns=["cat___definitely_missing___"],
            repo_root=REPO,
        )
