# Restaurant Recommendation System — Algorithm Overview

This document summarizes the retrieval and recommendation implementation in the repository (offline index, online scoring, optional RL policy, like/dislike re-ranking) as a reference alongside the code.

---

## 1. Offline Index: Review-based TF-IDF

**Module**: `backend/dining_retrieval/core/retrieval.py` (`TouristRetrieval`, `RestaurantSearchIndex`)

- Each restaurant is represented as a document: review texts are aggregated and combined with metadata (categories, name, city/state). Google Maps descriptions can be optionally mixed in.
- **sklearn `TfidfVectorizer`** is fit on the document corpus to produce a **sparse TF-IDF vector** and pre-computed **L2 norm** per restaurant, used for **cosine similarity** at query time.
- Star ratings are normalized and subject to **review-count trust shrinkage** (`rating_trust_ref_reviews`): restaurants with few reviews have their rating pulled toward neutral to reduce noise from small samples.
- The index candidate set is sampled by `review_count`; coverage can be restricted by `restrict_index_cities` and a city allowlist.

---

## 2. Online Retrieval: `recommend_keywords` Multi-factor Scoring

**Module**: `TouristRetrieval.recommend_keywords`

The final `query_text` is assembled from NL parsing, extra keywords, and cuisine selections (Discover mode uses a broad query like `"restaurants"`).

1. **Semantic relevance**: `query_text` is vectorized with the same `vectorizer` and cosine similarity is computed against all filtered restaurant vectors.
2. **Filtering**: state, city; optional **cuisine** (keyword matching on the `categories` column, relaxed if no results); optional **reference lat/lon + max radius** (Haversine distance in km).
3. Each signal is normalized then **linearly combined** into **`final_score`**:
   - **Semantic** `sim_n`
   - **Stars** `stars_rank` (with trust shrinkage)
   - **Price match** `price_match`: soft match against the budget tier (cheap / moderate / expensive)
   - **Distance** `dist_score`: higher for closer venues; neutral when no reference point is given
   - **Popularity** `pop_n`: min-max over `log1p(review_count)`

Weights (`w_semantic`, `w_rating`, `w_price`, `w_distance`, `w_popularity`) are passed in from the API/frontend. Results are ranked over a larger **`pool_k`** candidate pool and then truncated to **`top_k`** for display.

---

## 3. Query Understanding and API Alignment

**Modules**: `backend/dining_retrieval/search/query_parser.py`, `backend/services/retrieval_service.py`

- **`parse_query`**: extracts semantic tokens, budget hint, reference point, and radius from natural language input.
- **`extract_budget_hint`**: can infer a budget hint from a constraints string even when the NL parser misses it.
- Fields are aligned with the frontend Discover / Refine flow, cuisine multi-select, and weight sliders (see `SearchRequest` in `backend/api/schemas.py`).

---

## 4. Reinforcement Learning Layer: Intent-bucketed Contextual Bandit (UCB)

**Modules**: `models/rl_feedback_loop.py`, `_RL_WEIGHT_PRESETS` and `search()` in `RetrievalSearchService`

- **`classify_query_intent`**: regex rules assign the query to a coarse bucket: `intent_quick`, `intent_romantic`, or `intent_default`.
- **`RLFeedbackLoop`**: maintains three **arms** per intent bucket: `explorer`, `reputation`, `convenience`.
- **Arm selection**: **UCB** (`select_strategy`) — untried arms are force-explored first; otherwise selects by `Q + c * sqrt(log N / n)`.
- **Arm → retrieval weights**: each arm maps to a preset five-weight vector (same formula as `recommend_keywords`). If the user has touched a slider (`rl_user_overrode`), manual weights from the request body are used for that round instead.
- **Feedback**: the request carries the previous arm name, intent, and **`rl_action_events`**. Events like `detail_open` and `like` produce positive rewards; `refresh` and `slider_override` produce small negative rewards. **`log_user_feedback`** updates Q via `new_q = old_q + alpha * (reward - old_q)` and persists state to JSON/CSV.

This is a **"coarse-intent context + multi-arm bandit strategy selection"** approach, not an end-to-end trained deep ranking model.

---

## 5. Interactive Re-ranking: Like / Dislike (v2)

**Modules**: `backend/dining_retrieval/recommendation/reranker.py`, `UserPreferenceState`

When `liked_business_ids` or `disliked_business_ids` are present, **`rerank_pool`** is called on the candidate pool:

- On top of the base **`final_score`** (min-max normalized), the average cosine similarity to liked and disliked venues is computed using the same TF-IDF row vectors from the index.
- Combined with structured preference heuristics (min rating, max distance, category keywords) to produce **`v2_score`**. The pool is then re-sorted by `v2_score` descending and truncated to `top_k`.

---

## 6. Data Flow Summary

```text
Offline:  reviews/metadata → documents → TF-IDF index (vectors + meta + stars_norm)
Online:   query_text → cosine similarity + filters → multi-factor final_score → pool_k → top_k
Optional: intent classification → UCB arm selection → replace five-weight vector
Optional: like/dislike → v2 re-ranking
Async:    user events → reward signal → update Q-values per arm per intent bucket
```

---

## 7. File Index

| Topic | Path |
|-------|------|
| Index build and `recommend_keywords` | `backend/dining_retrieval/core/retrieval.py` |
| Retrieval service entry point (RL, re-ranking, photos) | `backend/services/retrieval_service.py` |
| Query parsing | `backend/dining_retrieval/search/query_parser.py` |
| UCB, intent buckets, feedback logging | `models/rl_feedback_loop.py` |
| Like/dislike re-ranking | `backend/dining_retrieval/recommendation/reranker.py` |
| Request body fields | `backend/api/schemas.py` |
