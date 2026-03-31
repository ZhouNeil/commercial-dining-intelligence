"""
Lightweight insight strings for each recommendation (MVP, no LLM).

Aligned with d1doc.md §3.6 — basic keyword / heuristic explanations.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from app.search.query_parser import ParsedQuery


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def generate_insight(
    row: pd.Series,
    parsed: Optional[ParsedQuery],
    query_text: str,
) -> dict[str, str]:
    """Return pros, cons, and a one-line 'why recommended'."""
    name = str(row.get("name", "This restaurant"))
    stars = _safe_float(row.get("stars"), 0.0)
    rc = int(_safe_float(row.get("review_count"), 0.0))
    cats = str(row.get("categories", "") or "")[:120]
    sim = _safe_float(row.get("similarity"), 0.0)
    dist_km = row.get("distance_km")
    price_tier = row.get("price_tier")

    pros: list[str] = []
    cons: list[str] = []

    if stars >= 4.0:
        pros.append("Strong average rating on Yelp.")
    elif stars >= 3.0:
        pros.append("Solid average rating.")
    else:
        cons.append("Lower average rating — read recent reviews carefully.")

    if rc >= 200:
        pros.append("Many reviews — easier to judge consistency.")
    elif rc >= 50:
        pros.append("A healthy number of reviews.")
    elif rc > 0:
        cons.append("Fewer reviews — ratings can be noisier.")

    if cats:
        pros.append(f"Categories include: {cats}{'...' if len(str(row.get('categories',''))) > 120 else ''}")

    if parsed and parsed.budget == "cheap" and price_tier is not None and not pd.isna(price_tier):
        pt = float(price_tier)
        if pt >= 3:
            cons.append("Yelp price tier suggests $$$ or higher — may not match a 'cheap' request.")

    if dist_km is not None and not pd.isna(dist_km):
        pros.append(f"About {float(dist_km):.1f} km from your reference point.")

    if not pros:
        pros.append("Returned by text similarity over aggregated review content.")

    why = (
        f"{name} ranks highly for your query because its review/category text is similar "
        f"(similarity {sim:.3f})"
    )
    if stars > 0:
        why += f", with {stars:.1f}-star average"
    if rc > 0:
        why += f" across {rc} reviews"
    why += "."

    return {
        "why": why,
        "pros": " • ".join(pros),
        "cons": " • ".join(cons) if cons else "—",
    }
