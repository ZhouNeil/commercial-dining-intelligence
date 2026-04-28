# Restaurant recommendation — algorithms

This document summarizes the **retrieval and recommendation** stack in this repo (offline index, online scoring, optional RL, like/dislike reranking) for reports and code cross-reference.

---

## 1. Offline index: review-based TF-IDF

**Module**: `backend/dining_retrieval/core/retrieval.py` (`TouristRetrieval`, `RestaurantSearchIndex`)

- Each restaurant is a “document”: aggregated review text plus category, name, city/state metadata; optional Google Maps blurbs.
- **sklearn `TfidfVectorizer`** fits on the corpus; each venue gets a **sparse TF-IDF vector** and precomputed **L2 norm** for **cosine similarity**.
- Stars are normalized; `rating_trust_ref_reviews` shrinks low-review-count ratings toward neutral to reduce spam/sample bias.
- The index candidate set is sampled by `review_count` (and related rules); optional `restrict_index_cities` + city whitelist narrows coverage.

---

## 2. Online retrieval: `recommend_keywords` multi-factor score

**Module**: `TouristRetrieval.recommend_keywords`

The client ends up with a **`query_text`** (NL parsing + extra keywords + cuisines; Discover may use a generic query like `restaurants`).

1. **Semantic**: embed **`query_text`** with the same `vectorizer`, cosine similarity vs filtered venue vectors.
2. **Filters**: state, city; optional **cuisine** (keyword rules on `categories`; relaxed if empty); optional **reference lat/lon + max radius** (Haversine km).
3. Normalized signals are **linearly combined** into **`final_score`**:
   - **Semantic** `sim_n`
   - **Stars** `stars_rank` (with trust shrinkage)
   - **Price** `price_match`: soft match vs budget tier (cheap / moderate / expensive)
   - **Distance** `dist_score`: higher near the reference point; neutral if no reference
   - **Popularity** `pop_n`: min-max of `log1p(review_count)`

Weights **`w_semantic`, `w_rating`, `w_price`, `w_distance`, `w_popularity`** come from the API/client. Sort on a larger **`pool_k`**, then take **`top_k`** for display.

---

## 3. Query understanding and API alignment

**Modules**: `backend/dining_retrieval/search/query_parser.py`, `backend/services/retrieval_service.py`

- **`parse_query`**: extracts semantic span, budget, reference point/radius from NL.
- **`extract_budget_hint`**: may infer budget hints from constraint strings.
- Aligns with **Discover / Refine**, cuisine multiselect, weight sliders (`backend/api/schemas.py` → `SearchRequest`).

---

## 4. RL: contextual bandit (UCB) per intent bucket

**Modules**: `models/rl_feedback_loop.py`, `RetrievalSearchService` (`_RL_WEIGHT_PRESETS`, `search()`)

- **`classify_query_intent`**: regex rules → `intent_quick`, `intent_romantic`, `intent_default` (quick/cheap vs date/romantic keywords).
- **`RLFeedbackLoop`**: per intent bucket, three **arms**: `explorer`, `reputation`, `convenience`.
- **Arm selection**: **UCB** (`select_strategy`) — unexplored arms first; else `Q + c * sqrt(log N / n)`.
- **Arm → weights**: each arm maps to a five-weight preset (same `recommend_keywords` formula as manual sliders). If **`rl_user_overrode`**, use request weights.
- **Feedback**: prior arm, intent, **`rl_action_events`**; `detail_open`, `like` → positive reward; `refresh`, `slider_override` → small negative; **`log_user_feedback`** updates Q (`new_q = old_q + alpha * (reward - old_q)`), persisted to JSON/CSV.

This is **coarse intent + multi-arm bandit over ranking strategies**, not an end-to-end learned ranker.

---

## 5. Interactive rerank: likes / dislikes (v2)

**Modules**: `backend/dining_retrieval/recommendation/reranker.py`, `UserPreferenceState`

With **`liked_business_ids` / `disliked_business_ids`**, **`rerank_pool`**:

- On top of **`final_score`** (min-maxed), average **row cosine similarity** in the same TF-IDF space to liked/disliked venues.
- Structured heuristics (min stars, max distance, category keywords, …) → **`v2_score`**, sort descending, then **`top_k`**.

---

## 6. Data flow summary

```text
Offline: reviews/meta → documents → TF-IDF index (vectors + meta + stars_norm)
Online: query_text → cosine + filters → final_score → pool_k → top_k
Optional: intent → UCB pick arm → replace five weights
Optional: likes/dislikes → v2 rerank
Async: actions → reward → update Q per intent/arm
```

---

## 7. File index

| Topic | Path |
|------|------|
| Index + `recommend_keywords` | `backend/dining_retrieval/core/retrieval.py` |
| Search entry (RL, rerank, photos) | `backend/services/retrieval_service.py` |
| Query parsing | `backend/dining_retrieval/search/query_parser.py` |
| UCB, intent, feedback log | `models/rl_feedback_loop.py` |
| Like/dislike rerank | `backend/dining_retrieval/recommendation/reranker.py` |
| Request schema | `backend/api/schemas.py` |

Add sections later for offline eval (`models/offline_evaluator.py`) or sentiment buckets in the index if needed.
