# Recommendation v2 (interactive) — implementation tracker

This file tracks **Restaurant Recommendation System — Interactive Upgrade (v2)** described in `docs/餐厅推荐系统说明.md`. Use checkboxes for progress; add dates next to items when done.

---

## Phases

### Phase A — Foundation: candidate pool + session state + minimal rerank (in progress)

- [x] **A1** Tracker doc `docs/recommendation_v2_tracker.md`
- [x] **A2** `recommend_keywords` supports `pool_k` (keep a larger pool than `top_k`) and `include_business_id`
- [x] **A3** `app/recommendation/preference_state.py`: session-level preference struct (like / dislike, etc.)
- [x] **A4** `app/recommendation/reranker.py`: secondary scoring from TF-IDF row vectors and v1 `final_score`
- [x] **A5** `main.py`: sidebar pool size, clear feedback on new search, `👍` / `👎`, show Top-K from pool and reranked list

### Phase B — Richer feedback types

- [ ] **B1** Explicit buttons: `Too expensive` / `Too far` / `Wrong cuisine` / `More like this` (map into `preference_state` fields)
- [ ] **B2** `feedback_parser.py`: short NL feedback (e.g. “cheaper”, “not sushi”) reusing or calling a subset of `parse_query`
- [ ] **B3** On strong constraint change, **re-run retrieval** (budget / radius / cuisine), then same pool + rerank flow

### Phase C — UX and robustness

- [ ] **C1** Sidebar sliders for v2 weights (`w_base` / `w_like` / `w_dislike` / `w_pref`)
- [ ] **C2** Fallback when a liked `business_id` is not in the index (use category / price / geo only)
- [ ] **C3** Tie into `generate_insight` (“boosted / demoted because you liked X”)

### Phase D — Long term (see doc Section 14)

- [ ] **D1** Dense vectors + FAISS / approximate nearest neighbors
- [ ] **D2** Learning-to-rank or weight tuning from logs

---

## Design notes (stay on track)

1. **With a fixed candidate pool**, reranking alone cannot fix “nothing cheaper in the pool” — Phase B3 must allow another retrieval pass.
2. **Item–item similarity** must use cosine in the **same** TF-IDF space as `RestaurantSearchIndex`.
3. A **new search** should **reset** session like/dislike (or an explicit “keep preferences” option — not implemented yet).

---

*Last updated: Phase A initialized; A1–A5 completed.*
