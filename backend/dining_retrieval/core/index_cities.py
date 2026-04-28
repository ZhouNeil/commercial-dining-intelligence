"""
Indexed city scope for `business_dining.csv`.

At build time, the full table can have many (city, state) pairs; project slices often focus
on the top metros by business count. The default index lists only these pairs to keep the
index small and target primary markets.

If the CSV uses multiple spellings for the same metro (e.g. Saint Louis / St. Louis), list both.
"""

from __future__ import annotations

# (city lower stripped, state USPS upper) — matches `meta.city_norm` / `state_norm`
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
