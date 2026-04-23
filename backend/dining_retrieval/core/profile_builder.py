"""
Restaurant profile construction utilities (MVP).

This module is intentionally written with only the Python standard library so
you can quickly verify its behavior without needing numpy/pandas.

Profile text conventions:
- Split reviews into positive (stars >= 4) and negative (stars <= 2)
- Extract top positive themes and top negative themes
- Build a unified restaurant profile text template
"""

from __future__ import annotations

import csv
import re
from typing import Any, Iterable, Optional


_WORD_RE = re.compile(r"[a-zA-Z']+")

# Keep this list small and practical; the goal is to avoid obvious query filler.
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
    "about",
    "into",
    "over",
    "under",
    "near",
    "around",
    "close",
    "within",
    "just",
    "very",
    "really",
    "like",
    "is",
    "it",
    "its",
    "was",
    "were",
    "be",
    "been",
    "being",
    "are",
    "i",
    "we",
    "you",
    "they",
    "them",
    "our",
    "your",
    "my",
    "me",
    "this",
    "that",
    "these",
    "those",
    "food",  # generic; but keep other cuisine words
    "restaurant",  # generic
    "restaurants",
}


def tokenize(text: str) -> list[str]:
    """Basic word tokenizer used for lightweight theme extraction."""
    words = _WORD_RE.findall(text or "")
    return [w.lower() for w in words]


def extract_themes(
    texts: Iterable[str],
    top_k: int = 6,
    ngram_range: tuple[int, int] = (1, 2),
    min_token_len: int = 3,
) -> list[str]:
    """
    Extract top themes from a list of review texts using simple term frequency.

    Returns phrases like:
    - "friendly staff"
    - "great service"
    """
    counts: dict[str, int] = {}
    min_n, max_n = ngram_range
    for t in texts:
        raw_tokens = [w for w in tokenize(t) if w not in _STOPWORDS and len(w) >= min_token_len]
        if not raw_tokens:
            continue

        if min_n <= 1:
            for tok in raw_tokens:
                counts[tok] = counts.get(tok, 0) + 1

        # Bigrams (optional)
        if max_n >= 2 and len(raw_tokens) >= 2:
            for i in range(len(raw_tokens) - 1):
                a, b = raw_tokens[i], raw_tokens[i + 1]
                phrase = f"{a} {b}"
                counts[phrase] = counts.get(phrase, 0) + 1

    if not counts:
        return []

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [p for p, _ in ordered[: max(1, top_k)]]


def price_symbol_from_attributes(attributes: Any) -> str:
    """
    Extract RestaurantsPriceRange2 from Yelp attributes and convert to "$" symbols.
    """
    s = str(attributes or "")
    m = re.search(r"RestaurantsPriceRange2['\"]?:\s*['\"]?(\d)", s)
    if not m:
        return "N/A"
    tier = int(m.group(1))
    tier = max(1, min(tier, 4))
    return "$" * tier


def build_profile_text(
    business: dict[str, Any],
    positive_texts: list[str],
    negative_texts: list[str],
    top_k_phrases: int = 6,
) -> str:
    """
    Unified restaurant profile template used as TF-IDF "document" text.
    """
    name = str(business.get("name", "") or "").strip() or "This restaurant"
    categories = str(business.get("categories", "") or "").strip()
    city = str(business.get("city", "") or "").strip()
    state = str(business.get("state", "") or "").strip()
    attributes = business.get("attributes", "")

    price = price_symbol_from_attributes(attributes)

    pos_phrases = extract_themes(positive_texts, top_k=top_k_phrases)
    neg_phrases = extract_themes(negative_texts, top_k=top_k_phrases)

    pos_str = ", ".join(pos_phrases) if pos_phrases else ""
    neg_str = ", ".join(neg_phrases) if neg_phrases else ""
    loc = ", ".join([x for x in [city, state] if x])

    # Template format intentionally simple to maximize token overlap with queries.
    return (
        f"Restaurant: {name}. "
        f"Category: {categories}. "
        f"City: {loc}. "
        f"Price: {price}. "
        f"Positive themes: {pos_str}. "
        f"Negative themes: {neg_str}."
    )


def build_profile_for_business_csv(
    business_path: str,
    reviews_path: str,
    business_id: Optional[str] = None,
    positive_stars_threshold: float = 4.0,
    negative_stars_threshold: float = 2.0,
    max_positive_reviews: int = 8,
    max_negative_reviews: int = 8,
    top_k_phrases: int = 6,
) -> dict[str, Any]:
    """
    Build and return a profile for ONE business by scanning the CSVs.

    This is meant for quick manual validation and avoids pandas/numpy.
    """
    # Load the chosen business row (or the first one).
    chosen: Optional[dict[str, str]] = None
    chosen_id: Optional[str] = None

    with open(business_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = (row.get("business_id") or "").strip()
            if not bid:
                continue
            if business_id is None or bid == str(business_id):
                chosen = row
                chosen_id = bid
                break

    if chosen is None or chosen_id is None:
        raise ValueError(f"business_id not found in {business_path}: {business_id}")

    positive_texts: list[str] = []
    negative_texts: list[str] = []

    # Scan reviews and collect positive/negative texts.
    with open(reviews_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("business_id") or "").strip() != chosen_id:
                continue

            stars_raw = row.get("stars")
            try:
                stars = float(stars_raw) if stars_raw is not None else float("nan")
            except Exception:
                continue

            text = (row.get("text") or "").strip()
            if not text:
                continue

            if stars >= positive_stars_threshold and len(positive_texts) < max_positive_reviews:
                positive_texts.append(text)
            elif stars <= negative_stars_threshold and len(negative_texts) < max_negative_reviews:
                negative_texts.append(text)

            if (
                len(positive_texts) >= max_positive_reviews
                and len(negative_texts) >= max_negative_reviews
            ):
                break

    profile_text = build_profile_text(
        business=chosen,
        positive_texts=positive_texts,
        negative_texts=negative_texts,
        top_k_phrases=top_k_phrases,
    )

    return {
        "business_id": chosen_id,
        "name": chosen.get("name"),
        "positive_review_count": len(positive_texts),
        "negative_review_count": len(negative_texts),
        "profile_text": profile_text,
    }

