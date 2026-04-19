"""
Optional MiniLM (sentence-transformers) mapping: user query → USPS state among dataset states.

Used only when rule-based `parse_query` does not set `state_code` and the UI enables this path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.search.geo_constants import US_STATE_CODES, display_name_for_code

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_minilm_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def infer_state_minilm(
    query: str,
    dataset_state_codes: list[str],
    model,
    min_similarity: float = 0.32,
    min_margin: float = 0.02,
) -> tuple[Optional[str], float]:
    """
    Pick the most similar USPS state label to the query using cosine similarity.
    Returns (state_code_or_none, best_score). Empty / irrelevant queries should fail the threshold.
    """
    q = (query or "").strip()
    if not q or not dataset_state_codes:
        return None, 0.0

    codes = [
        str(c).strip().upper()
        for c in dataset_state_codes
        if str(c).strip().upper() in US_STATE_CODES
    ]
    if not codes:
        return None, 0.0

    labels = [f"{display_name_for_code(c)}, United States ({c})" for c in codes]
    q_emb = model.encode(q, normalize_embeddings=True)
    lab_emb = model.encode(labels, normalize_embeddings=True)
    sims = np.asarray(lab_emb @ q_emb, dtype=float).ravel()
    if sims.size == 0:
        return None, 0.0

    order = np.argsort(sims)[::-1]
    best_i = int(order[0])
    best = float(sims[best_i])
    second = float(sims[int(order[1])]) if sims.size > 1 else 0.0
    if best < min_similarity or (best - second) < min_margin:
        return None, best
    return codes[best_i], best
