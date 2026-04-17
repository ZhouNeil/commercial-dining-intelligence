"""
Indexed city scope for `business_dining.csv`.

Verification: 截至构建时，全表约有上千个不同的 (city, state) 组合；课堂/项目切片里
「主体体量」集中在按 **商户数降序** 的前 11 个都会区。索引默认只收录这些组合，
以减小索引、聚焦主市场。

若在 CSV 中有同一都会区的不同写法（如 Saint Louis / St. Louis），在集合里一并列出。
"""

from __future__ import annotations

# (city lower stripped, state USPS upper) — 与 `meta.city_norm` / `state_norm` 一致
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
