"""
Rule-based natural language query parser (MVP).

Maps free text to structured constraints aligned with d1doc.md:
- cuisine hint
- budget (cheap / moderate / expensive)
- location text + optional landmark coordinates
- radius hint (km)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# Known landmarks -> (lat, lon) for distance filtering when user says "near NYU" etc.
LANDMARKS: dict[str, tuple[float, float]] = {
    "nyu": (40.7295, -73.9965),
    "new york university": (40.7295, -73.9965),
    "times square": (40.7580, -73.9855),
    "philadelphia": (39.9526, -75.1652),
    "philly": (39.9526, -75.1652),
    "center city philadelphia": (39.9526, -75.1652),
    "las vegas strip": (36.1147, -115.1728),
    "nashville": (36.1627, -86.7816),
    "tampa": (27.9506, -82.4572),
}


@dataclass
class ParsedQuery:
    """Structured constraints extracted from user text."""

    raw: str
    cuisine: Optional[str] = None
    budget: Optional[str] = None  # cheap | moderate | expensive
    location_text: Optional[str] = None
    ref_lat: Optional[float] = None
    ref_lon: Optional[float] = None
    radius_km: Optional[float] = None
    vibe_terms: list[str] = field(default_factory=list)
    # Query text intended for semantic embedding/retrieval (should avoid budget/location noise).
    semantic_query: str = ""
    # Kept for backward compatibility; currently equals normalized raw text.
    extra_keywords: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuisine": self.cuisine,
            "budget": self.budget,
            "location": self.location_text,
            "radius_km": self.radius_km,
            "ref_lat": self.ref_lat,
            "ref_lon": self.ref_lon,
            "vibe_terms": self.vibe_terms,
            "semantic_query": self.semantic_query,
            "extra_keywords": self.extra_keywords,
        }


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


_STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "near",
    "around",
    "close",
    "within",
    "inexpensive",
    "affordable",
    "cheap",
    "moderate",
    "moderately",
    "reasonable",
    "midrange",
    "mid-range",
    "expensive",
    "fancy",
    "upscale",
    "fine",
    "dining",
    "splurge",
    "budget",
    "low",
    "cost",
    "km",
    "kilometer",
    "kilometers",
    "miles",
    "mi",
    "restaurant",
    "restaurants",
    "place",
    "want",
    "need",
    "looking",
    "best",
    "food",
}


def _build_semantic_query(raw: str, parsed: ParsedQuery) -> tuple[list[str], str]:
    """
    Build semantic query text by removing obvious budget/location/radius parts.
    """
    q = raw.lower()

    # Remove budget phrases/words.
    q = re.sub(
        r"\b(cheap|affordable|budget|inexpensive|low[- ]?cost|expensive|fancy|upscale|fine dining|splurge|moderate|mid[- ]?range|midrange|reasonable)\b",
        " ",
        q,
    )

    # Remove distance constraints.
    q = re.sub(
        r"\bwithin\s+\d+(?:\.\d+)?\s*(km|kilometers?|miles?|mi)\b",
        " ",
        q,
    )

    # Remove landmark/location patterns.
    q = re.sub(r"\b(?:near|around|close to)\s+[a-z0-9][a-z0-9\s\-\']{1,80}", " ", q)
    q = re.sub(r"\bin\s+[a-z][a-z\s\-]{1,60}", " ", q)

    tokens = re.findall(r"[a-zA-Z']+", q)
    filtered: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        t = tok.strip().lower()
        if not t or t in _STOPWORDS:
            continue
        if len(t) <= 1:
            continue
        if t in seen:
            continue
        seen.add(t)
        filtered.append(t)

    # In case the query is only structured constraints (e.g. "cheap near NYU").
    if not filtered:
        if parsed.cuisine:
            filtered = [str(parsed.cuisine)]
        else:
            filtered = ["restaurant"]

    semantic_query = " ".join(filtered[:10]).strip()
    return filtered[:10], semantic_query


def parse_query(text: str) -> ParsedQuery:
    if not isinstance(text, str) or not text.strip():
        return ParsedQuery(raw=text or "")

    raw = text.strip()
    q = _norm(raw)

    cuisine = None
    cuisine_patterns = [
        ("sushi", "sushi"),
        ("japanese", "japanese"),
        ("steakhouse", "steakhouse"),
        ("steak", "steak"),
        ("korean", "korean"),
        ("chinese", "chinese"),
        ("italian", "italian"),
        ("mexican", "mexican"),
        ("thai", "thai"),
        ("indian", "indian"),
        ("burger", "burger"),
        ("pizza", "pizza"),
        ("fast food", "fast food"),
        ("vegan", "vegan"),
        ("vegetarian", "vegetarian"),
        ("healthy", "healthy"),
        ("salad", "salad"),
    ]
    for needle, label in cuisine_patterns:
        if needle in q:
            cuisine = label
            break

    budget = None
    if re.search(r"\b(cheap|affordable|budget|inexpensive|low[- ]?cost)\b", q):
        budget = "cheap"
    elif re.search(r"\b(expensive|fancy|upscale|fine dining|splurge)\b", q):
        budget = "expensive"
    elif re.search(r"\b(moderate|mid[- ]?range|midrange|reasonable)\b", q):
        budget = "moderate"

    radius_km = None
    m = re.search(r"within\s+(\d+(?:\.\d+)?)\s*(km|kilometers?|miles?|mi)\b", q)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("mi"):
            radius_km = val * 1.60934
        else:
            radius_km = val
    elif "near" in q or "around" in q or "close to" in q:
        radius_km = 2.0

    location_text = None
    ref_lat, ref_lon = None, None

    near = re.search(
        r"\b(?:near|around|close to)\s+([a-z0-9][a-z0-9\s\-\']{1,80})(?:\s|$|,|\.)",
        q,
    )
    if near:
        location_text = near.group(1).strip()

    in_loc = re.search(r"\bin\s+([a-z][a-z\s\-]{1,60})(?:\s|$|,|\.)", q)
    if in_loc and not location_text:
        location_text = in_loc.group(1).strip()

    if location_text:
        # If the extracted location accidentally includes trailing budget words
        # (e.g. "in Philadelphia moderate"), strip them for a cleaner output.
        loc_tokens = location_text.lower().split()
        budget_tail = {
            "cheap",
            "affordable",
            "budget",
            "inexpensive",
            "low",
            "cost",
            "expensive",
            "fancy",
            "upscale",
            "fine",
            "dining",
            "splurge",
            "moderate",
            "midrange",
            "mid-range",
            "reasonable",
        }
        while loc_tokens and loc_tokens[-1] in budget_tail:
            loc_tokens.pop()
        location_text = " ".join(loc_tokens).strip() or location_text

        key = _norm(location_text)
        for landmark, coords in LANDMARKS.items():
            if landmark in key or key in landmark:
                ref_lat, ref_lon = coords
                break

    # Keep a normalized raw version for debug/compat.
    extra = raw
    extra_keywords = _norm(extra)

    # Build semantic query (used for embedding/retrieval).
    # It should avoid budget/location/radius noise but keep meaningful hints.
    parsed_tmp = ParsedQuery(
        raw=raw,
        cuisine=cuisine,
        budget=budget,
        location_text=location_text,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        radius_km=radius_km,
        extra_keywords=extra_keywords,
    )
    vibe_terms, semantic_query = _build_semantic_query(raw, parsed_tmp)

    return ParsedQuery(
        raw=raw,
        cuisine=cuisine,
        budget=budget,
        location_text=location_text,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        radius_km=radius_km,
        vibe_terms=vibe_terms,
        semantic_query=semantic_query,
        extra_keywords=extra_keywords,
    )
