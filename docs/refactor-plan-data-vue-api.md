# Full-stack refactor plan: data, storage, Vue, model API

This document tracks the migration from **dual CSV tracks → evolvable data/API → Vue front end** for team review and task breakdown. Scope: **data modeling and SQLite/DB**, **backend API for existing models**, **Vue rewrite**, and **phased work**.

> **`docs/` is trimmed**: aside from this file, older long-form notes were removed. **Run commands and tree** live in the root [`README.md`](../README.md); **field-level contracts** in `frontend/openapi.json` and `backend/api/schemas.py`. Add a short `PROJECT_OVERVIEW.md` later if needed.

### Progress (landed in repo)

| Phase | Status | Notes |
|------|--------|--------|
| P0 | Done | `data/manifests/schema.sample.json`, `scripts/write_data_manifest.py` (fill data contract in a future overview or here) |
| P1 | Done | `backend/services/merchant_inference.py`, `backend/services/retrieval_service.py`; `tests/test_inference.py` via services |
| P2 | Done | `backend/api/` (FastAPI), `scripts/run_api.sh`, `Dockerfile.api`, `requirements-dev.txt` + pytest |
| P3 | Done | `scripts/etl_csv_to_sqlite.py` (`business_dining.csv` → SQLite) |
| P4 | Done (UI evolving) | `frontend/`: Vue Router, `/search` & `/merchant`, OpenAPI + `gen:api`, root `package.json` passthrough, `deploy/nginx-frontend.example.conf`; map/detail polish TBD |
| P5 | Not done | Parquet, model registry, rate limits, … |

---

## 1. Why refactor

### 1.1 Two mostly separate tracks

| Track | Use case | Data (today) | Main consumers |
|-------|----------|----------------|----------------|
| **A. Search / recommend** | NL + filters, TF-IDF, maps | `data/cleaned/` (e.g. `business_dining.csv`), optional `data/slice_representative/`; index + vectors in `models/artifacts/` | `dining_retrieval`, `services`, `api`; `frontend` (Vue) |
| **B. Spatial / merchant** | Site selection, spatial features, survival + stars | `data/train_spatial.csv` (and `train_merchant_split` / `test_spatial`); `*.pkl` in `models/artifacts/` | `merchant_predictor`, `SpatialFeatureEngineer`, `tests/test_inference`, `/merchant` |

Both share “restaurant” semantics but **differ in schema, grain, and refresh**. Plain CSV collab often means **scattered paths, duplicate copies, no clear “source of truth”, bad Git for large files**.

### 1.2 Pain points (and goals)

- **Two CSV families**: same concept, mismatched fields; missing unified ID and lineage.
- **Blurry folder roles**: “cleaned”, “features”, “train” mixed — hard for newcomers.
- **(Fixed)**: old Streamlit MVP tightly coupled UI + models; now Vue + OpenAPI + `backend/services`.
- **Hard-coded paths** for train/infer: awkward for fixed deploy paths or containers.

**Goal**: **clear source(s) of truth**, **versioned pipelines**, **HTTP API decoupled from UI**, **Vue SPA**.

---

## 2. Data layer

### 2.1 Principles

1. **Domain model before storage** — define lifecycles for businesses, reviews, spatial snapshots, index metadata, model artifacts.
2. **Split “online read path” from “offline batch”** — small/indexable online; wide tables in columnar / external storage.
3. **One primary key** — consistent Yelp `business_id` (or internal UUID); link A/B by ID, not fuzzy name+lat.

### 2.2 Suggested layout (even while still on CSV)

```

data/
  raw/              # raw Yelp etc. (optional, or metadata only)
  curated/          # cleaned “business” tables: merchants, review summaries (today cleaned / slice)
  features/         # ML wide tables: train_spatial, interim parquet, …
  manifests/        # version manifests: file names, sha256, script commit, row counts
models/
  artifacts/        # index, vectorizer, pkl (gitignored; describe in manifest)
```

**Manifest (JSON/YAML)**: each pipeline run records the active version pointer, not a hard-coded filename in code.

### 2.3 Converging the “two tables” (options)

- **Option 1 (recommended, incremental)**: **one merchant spine** in `curated/`; pipeline **joins** into `features/train_spatial` for B; A reads `curated` + artifacts. Align on `business_id`.
- **Option 2**: keep two tables but **document + contract** field tables and lineage in README / OpenAPI; API exposes DTOs only.
- **Option 3 (long term)**: spine in DB; feature tables still Parquet/SQLite columnar for pandas/sklearn (next section).

---

## 3. Database? Is SQLite enough?

### 3.1 Fit by data type

| Data | CSV cost | SQLite | PostgreSQL | Files (Parquet/npz) |
|------|----------|--------|------------|---------------------|
| Merchants, review meta, small prefs | full scans, weak concurrency | **Good** MVP: single file, low ops, SQL index | multi-writer, scale, geospatial | optional Parquet for analytics |
| Wide feature (many columns) | slow, memory-heavy | possible; **worse than Parquet** for columnar | same | **Prefer** for train/infer |
| TF-IDF sparse, big vectors | bad for hot IO | bad for huge blobs | blob or filesystem | **Keep npz/joblib** + manifest |
| Real OLTP | n/a | single-writer | scale up | — |

### 3.2 Recommendation

- **Phase 1**: **SQLite** (or strong manifest + CSV) as **one queryable source** for **merchant/review/config**; **wide + matrices stay in files** (Parquet + existing joblib/npz) — do not shove 1000-d features into raw SQL.
- **PostgreSQL** when you need multi-writer, permissions, PostGIS in-DB, or managed cloud.
- **Do not expect SQLite to replace a warehouse** — it’s for **structured metadata and relations**; **pandas + Parquet** still fits sklearn best.

### 3.3 Example SQLite schema (sketch)

- `merchants`: `business_id` PK, `name`, `lat`, `lon`, `city`, `state`, `stars`, `review_count`, `is_open`, `categories_json`, …
- `reviews`: `review_id` PK, `business_id` FK, `text`, `stars`, `date`, … (sample if needed)
- `dataset_builds`: `id`, `kind` (spatial_train / retrieval_index), `path`, `checksum`, `created_at`, `git_sha`
- `model_registry`: `name` (survival / rating), `path`, `feature_schema_hash`, `created_at`

Huge spatial feature tables can store **`business_id` + a few fields + Parquet path**, or stay fully file-backed with mmap as needed.

---

## 4. Target architecture: Vue + API + models

### 4.1 Logical layers

```

Vue SPA (Vite)
    │  HTTPS / JSON
    ▼
API Gateway (optional)
    ▼
Python API (FastAPI)
    ├── Search: load TouristRetrieval / index, POST /search
    ├── Merchant: SpatialFeatureEngineer + joblib, POST /merchant/…
    ├── Meta: GET /health, GET /datasets/active
    └── (opt.) SQLite for detail / paging
```

### 4.2 Wiring existing models

- **Search track**: `build_or_load_index` at startup; per request query + rerank; DTOs match Vue/map components.
- **Predict track**: logic from `tests/test_inference.py` lives in **`backend/services/merchant_inference.py`**; API validates and maps errors.
- **Process/memory**: lazy singleton loads; **multi-worker = one copy per process** (or read-only mmap later).

### 4.3 Example API table (for planning)

| Method | Path | Notes |
|--------|------|--------|
| GET | `/api/health` | version, data presence |
| POST | `/api/v1/search` | body: query, filters, `top_k`, … |
| POST | `/api/v1/merchant/predict` | body: `city`, `lat`, `lon`, `category_keys` / `category_query` |
| GET | `/api/v1/merchants/{business_id}` | detail (SQLite or curated) |

Use `application/json`; errors `{ "code", "message", "detail" }`.

### 4.4 Vue notes

- **Vite + Vue 3 + TypeScript**; Pinia optional; UI lib optional (Element / Naive).
- **Map**: MapLibre / Leaflet; WGS84 to match backend.
- **MVP**: Streamlit retired; **OpenAPI is the contract** for Vue.

---

## 5. Roadmap (suggested order)

| Phase | Work | Outcome |
|------|------|---------|
| **P0** | Manifest; unify `business_id`; document CSV lineage | `data/manifests/*` + this doc / future overview |
| **P1** | Inference service without Streamlit; unit tests for predict | `backend/services/*` + pytest |
| **P2** | FastAPI minimal: `/health` + `/merchant/predict` + `/search` | Docker-runnable |
| **P3** | ETL curated merchants → SQLite; API read path | `scripts/etl_to_sqlite.py` |
| **P4** | Vue app; OpenAPI types; main flows | static build + API reverse proxy |
| **P5** | Parquet features, model registry, observability, rate limits | ops-ready |

---

## 6. Task breakdown (epics)

### Epic A — data governance

- [x] **A1** Inventory all CSV paths and consumers.
- [x] **A2** Core conventions in this doc §1–2, §4, OpenAPI; full field list in a future overview.
- [x] **A3** Manifest generation (optional CI check for pointer files).
- [ ] **A4** Decide **SQLite** scope (merchants only vs + reviews) and **Parquet** for `train_spatial`.
- [x] **A5** (optional) `etl_csv_to_sqlite.py` + minimal schema notes.

### Epic B — model serving

- [x] **B1** `merchant_inference.predict` (lat/lon/city/categories → proba, stars, features).
- [x] **B2** Thin `retrieval` / `TouristRetrieval` wrapper.
- [x] **B3** Pydantic + OpenAPI + error shapes.
- [x] **B4** FastAPI routes + run docs (`.venv` to avoid NumPy issues).
- [x] **B5** `Dockerfile` (API) + volume conventions.

### Epic C — Vue

- [x] **C1** Vue3+TS+Vite in `frontend/`.
- [x] **C2** OpenAPI → `gen:api` → `src/api/generated.d.ts`, `client.ts` fetch.
- [x] **C3** `/`, `/search`, `/merchant`.
- [x] **C4** `VITE_API_BASE_URL`; dev proxy.
- [x] **C5** `npm run build` → `dist`; `deploy/nginx-frontend.example.conf`.

### Epic D — cleanup

- [x] **D1** Capabilities live in `frontend/` + OpenAPI.
- [ ] **D2** Remove dead code; keep root `README` + optional `docs/PROJECT_OVERVIEW.md`.
- [ ] **D3** (optional) Playwright for critical API + pages.

---

## 7. Risks

- **Memory**: index + spatial table resident → measure peaks; **split services** (search vs predict) if needed.
- **Drift**: **feature_schema_hash** in manifest; validate on load.
- **Parallel teams**: **freeze OpenAPI**; version paths `/v1` → `/v2`.

---

## 8. Doc maintenance

- This file: `docs/refactor-plan-data-vue-api.md` (**only long-term planning doc under `docs/`**).
- **Table layouts, API base URL, env vars**: root `README.md`; request/response fields: **OpenAPI**. Add a one-pager for stakeholders as `docs/PROJECT_OVERVIEW.md` if needed.

*Version: draft for review; trim P3/P5 for small teams.*
