"""
Load Google Maps restaurant CSV and normalize to the same columns as `business_dining.csv`
for indexing alongside the Yelp dining slice.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

GOOGLE_MAPS_CSV_NAME = "google_maps_restaurants(cleaned).csv"

_US_ADDR_TAIL = re.compile(
    r",\s*([^,]+),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\s*$",
    re.IGNORECASE,
)


def google_maps_csv_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / GOOGLE_MAPS_CSV_NAME


def parse_city_state_zip(address: str, zip_fallback: str | float | None = "") -> tuple[str, str, str]:
    if not isinstance(address, str) or not address.strip():
        z = "" if zip_fallback is None or (isinstance(zip_fallback, float) and np.isnan(zip_fallback)) else str(zip_fallback).strip()
        return "", "", z
    m = _US_ADDR_TAIL.search(address.strip())
    if m:
        city = m.group(1).strip()
        state = m.group(2).strip().upper()
        postal = m.group(3).strip()
        return city, state, postal
    z = "" if zip_fallback is None or (isinstance(zip_fallback, float) and np.isnan(zip_fallback)) else str(zip_fallback).strip()
    return "", "", z


def _stable_gm_id(url: str, seq: int) -> str:
    u = str(url or "").strip()
    if u:
        h = hashlib.sha256(u.encode("utf-8")).hexdigest()[:16]
        return f"gm_{h}"
    return f"gm_idx_{seq}"


def load_google_maps_as_yelp_schema(csv_path: str | Path) -> pd.DataFrame:
    """
    Map Google export columns to Yelp business schema. Rows without coordinates or name are dropped.
    """
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame()

    raw = pd.read_csv(p, low_memory=False)
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["_lat"] = pd.to_numeric(df.get("Lat"), errors="coerce")
    df["_lon"] = pd.to_numeric(df.get("Lon"), errors="coerce")
    df = df[df["_lat"].notna() & df["_lon"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    names = df.get("Name", pd.Series([], dtype=str)).astype(str).str.strip()
    df = df.loc[names != ""].copy()
    if df.empty:
        return pd.DataFrame()

    df.reset_index(drop=True, inplace=True)
    n = len(df)

    urls = df.get("URL", pd.Series([""] * n)).astype(str).fillna("")
    bids = [_stable_gm_id(urls.iloc[i], i) for i in range(n)]

    cities: list[str] = []
    states: list[str] = []
    postals: list[str] = []
    for _, r in df.iterrows():
        addr = r.get("Address", "")
        z = r.get("ZipCode", "")
        c, s, pc = parse_city_state_zip(str(addr) if pd.notna(addr) else "", z)
        cities.append(c)
        states.append(s)
        postals.append(pc or (str(z).strip() if pd.notna(z) else ""))

    stars = pd.to_numeric(df.get("Rating"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=5.0)
    rc = pd.to_numeric(df.get("Rating Count"), errors="coerce").fillna(0.0).astype(int)

    price_col = df.get("Price Category")
    attrs: list[str] = []
    if price_col is None:
        attrs = [str({"DataSource": "GoogleMaps"})] * n
    else:
        for pi in price_col:
            try:
                if pd.isna(pi) or str(pi).strip() == "":
                    attrs.append(str({"DataSource": "GoogleMaps"}))
                else:
                    tier = int(round(float(pi)))
                    tier = max(1, min(tier, 4))
                    attrs.append(
                        str({"RestaurantsPriceRange2": str(tier), "DataSource": "GoogleMaps"})
                    )
            except (TypeError, ValueError):
                attrs.append(str({"DataSource": "GoogleMaps"}))
        if len(attrs) != n:
            attrs = [str({"DataSource": "GoogleMaps"})] * n

    meta_text = df.get("Detailed Ratings", pd.Series([""] * n)).astype(str).fillna("")

    out = pd.DataFrame(
        {
            "business_id": bids,
            "name": df["Name"].astype(str).values,
            "address": df.get("Address", pd.Series([""] * n)).astype(str).fillna("").values,
            "city": cities,
            "state": states,
            "postal_code": postals,
            "latitude": df["_lat"].astype(float).values,
            "longitude": df["_lon"].astype(float).values,
            "stars": stars.values,
            "review_count": rc.values,
            "is_open": np.ones(n, dtype=int),
            "attributes": attrs,
            "categories": np.array(["Restaurants, Food"] * n, dtype=object),
            "hours": np.array([""] * n, dtype=object),
            "_gm_detail": meta_text.values,
            "_gm_url": urls.values,
        }
    )

    out["state"] = out["state"].astype(str).str.strip().str.upper()
    out["city"] = out["city"].astype(str).str.strip()
    return out


def synthetic_google_profile_snippets(series_row: pd.Series) -> tuple[list[str], list[str]]:
    """Pseudo-review text so TF-IDF has signal without Yelp reviews."""
    parts: list[str] = []
    nm = str(series_row.get("name", ""))
    addr = str(series_row.get("address", ""))
    cats = str(series_row.get("categories", ""))
    if nm:
        parts.append(nm)
    if addr:
        parts.append(addr)
    if cats:
        parts.append(cats)
    detail = str(series_row.get("_gm_detail", ""))
    if detail and detail not in ("{}", "nan"):
        parts.append(detail[:400])
    blob = " ".join(parts)
    if not blob.strip():
        return [], []
    return [blob], []


def union_state_options(data_dir: str | Path) -> list[str]:
    """States appearing in Yelp dining slice and/or Google Maps CSV (for filter UI)."""
    data_dir = Path(data_dir)
    states: set[str] = set()
    yelp_p = data_dir / "business_dining.csv"
    if yelp_p.exists():
        y = pd.read_csv(yelp_p, usecols=["state"], low_memory=False)
        s = y["state"].astype(str).str.strip().str.upper().replace({"NAN": ""})
        states.update(x for x in s.unique() if x)
    gm_p = google_maps_csv_path(data_dir)
    if gm_p.exists():
        g = load_google_maps_as_yelp_schema(gm_p)
        if not g.empty:
            s = g["state"].astype(str).str.strip().str.upper()
            states.update(x for x in s.unique() if x)
    return sorted(states)
