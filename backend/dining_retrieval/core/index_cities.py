"""
Indexed city scope for `business_dining.csv`.

The full dataset contains thousands of distinct (city, state) combinations. The index is
intentionally restricted to the top 11 metro areas by business count to keep the index
size manageable and focused on the most data-rich markets.

If a metro area appears under multiple spellings in the CSV (e.g. Saint Louis / St. Louis),
both variants are included in the set.
"""

from __future__ import annotations

# (city lower stripped, state USPS upper) — must align with `meta.city_norm` / `state_norm`
INDEX_ALLOWED_CITY_STATE: frozenset[tuple[str, str]] = frozenset(
    {
        ("philadelphia", "PA"),
        ("tampa", "FL"),
        ("indianapolis", "IN"),
        ("tucson", "AZ"),
        ("nashville", "TN"),
        ("new orleans", "LA"),
        ("edmonton", "AB"),
        ("saint louis", "MO"),
        ("st. louis", "MO"),
        ("reno", "NV"),
        ("santa barbara", "CA"),
        ("boise", "ID"),
    }
)

INDEX_FILTER_ID = "top11_by_business_count_v1"
