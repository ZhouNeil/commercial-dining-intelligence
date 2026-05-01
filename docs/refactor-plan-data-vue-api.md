# Full-Stack Refactor Plan: Data Layer, Storage, Vue Frontend, and Model API

This document covers the overall refactor from a dual-track CSV data setup to a maintainable data and API layer with a Vue frontend. Scope includes: **data modeling and whether to introduce SQLite/DB**, **backend API wiring for existing models**, **Vue frontend rewrite**, and **phased task breakdown**.

> **`docs/` has been trimmed**: aside from this file, historical specs, long contract docs, and comparison tables have been removed. **Run commands and directory layout** are documented in the repo root [`README.md`](../README.md); **field-level contracts** are in `frontend/openapi.json` (and `backend/api/schemas.py`). A short project overview page can be added separately if needed.

### Implementation Status (landed in repo)

| Phase | Status | Notes |
|-------|--------|-------|
| P0 | Done | `data/manifests/schema.sample.json`, `scripts/write_data_manifest.py` (field-level data contract details to be added in a future project overview or next iteration of this doc) |
| P1 | Done | `backend/services/merchant_inference.py`, `backend/services/retrieval_service.py`; `tests/test_inference.py` exercises the service layer |
| P2 | Done | `backend/api/` (FastAPI), `scripts/run_api.sh`, `Dockerfile.api`, `requirements-dev.txt` + pytest |
| P3 | Done | `scripts/etl_csv_to_sqlite.py` (imports `business_dining.csv` → SQLite) |
| P4 | Done (UI still in progress) | `frontend/`: Vue Router, `/search` and `/merchant` views, OpenAPI + `gen:api`, root `package.json` forwarding, `deploy/nginx-frontend.example.conf`; map detail views still pending |
| P5 | Not started | Parquet, model registry, rate limiting, etc. |

---

## 1. Current State (Why Refactor)

### 1.1 Two Largely Independent Data/Product Tracks

| Track | Typical use | Data format (current) | Main consumers |
|-------|------------|----------------------|----------------|
| **A. Retrieval / Recommendation** | NL search + filters, TF-IDF retrieval, map display | `data/cleaned/` (e.g. `business_dining.csv`), `data/slice_representative/` as fallback; index and vectors in `models/artifacts/` | `backend/dining_retrieval`, `services`, `api`; Vue frontend |
| **B. Spatial / Merchant Prediction** | Site selection, spatial features, survival probability and star rating regression | `data/train_spatial.csv` (and splits `train_merchant_split` / `test_spatial`); `.pkl` models in `models/artifacts/` | `merchant_predictor`, `SpatialFeatureEngineer`, `tests/test_inference`, frontend `/merchant` |

Both tracks share the concept of a "restaurant", but differ in **column structure, granularity, and update cadence**: Track A is oriented toward display and retrieval, Track B toward wide feature tables. With CSV files, local collaboration leads to: **scattered paths, duplicate copies, no clear "source of truth", and large files that don't version well in Git**.

### 1.2 Current Pain Points (and Refactor Goals)

- **Dual CSV sets**: the same business entity appears with inconsistent fields across A and B, with no shared entity ID or lineage documentation.
- **Unclear responsibility boundaries**: "cleaned output", "feature table", and "training set" sit at the same directory level, making it hard for new contributors to know where to read from.
- **(Resolved)** The old Streamlit MVP coupled frontend and model code; this has been replaced by Vue + OpenAPI + `backend/services`.
- **Hard-coded inference and training paths**: multiple `Path` / relative path references make deployment to a fixed directory or container harder.

Refactor goals: **single source of truth (or clearly layered multi-source) + versionable data pipeline + HTTP API decoupled from frontend + Vue SPA**.

---

## 2. Data Layer Refactor Approach

### 2.1 Principles

1. **Define the domain model before choosing storage**: what is the lifecycle of merchants, reviews, spatial feature snapshots, retrieval index metadata, and model artifacts?
2. **Separate "data read by the online service" from "offline batch outputs"**: online data should be small and indexable; offline wide tables can remain in columnar/file format or external object storage.
3. **Unified primary key**: use Yelp `business_id` (or an internal `merchant_uuid`) consistently throughout the project so that tracks A and B join on that key rather than fuzzy name+lat matching.

### 2.2 Suggested Directory and Logical Layers (even if CSV files are kept short-term)

A "logical refactor" before introducing a DB can reduce dual-track confusion:

```
data/
  raw/              # original Yelp data etc. (not necessarily imported to DB)
  curated/          # cleaned "business-readable" narrow tables: merchant master, review summaries, etc. (corresponds to current cleaned / slice)
  features/         # ML wide tables: train_spatial, intermediate parquet files, etc.
  manifests/        # data version manifests: filenames, sha256, generating script commit, row counts
models/
  artifacts/        # index, vectorizer, pkl files (keep gitignored, described via manifest)
```

**Manifest (JSON/YAML)**: each pipeline run writes one record. The API and training scripts depend on "current active version pointer" rather than hard-coded filenames.

### 2.3 How to Converge the Two Data Sets (Strategy Options)

- **Option 1 (recommended, incremental)**: **one merchant master** (curated), spatial features generated by the pipeline via **JOIN from master + rules** into `features/train_spatial`. Track B reads only from features; Track A reads from curated + artifacts. Both sides align via `business_id`.
- **Option 2**: keep dual tables but **enforce documentation + contracts**: maintain a field table and lineage in README / OpenAPI / a separate overview doc. The API layer only exposes DTOs, not raw CSV paths, to the frontend.
- **Option 3 (long-term)**: master data moves to a DB; feature tables remain as Parquet/SQLite columnar files for pandas/sklearn batch reads (see below).

---

## 3. Should We Use a Database? Is SQLite Sufficient?

### 3.1 Appropriate Storage for Each Data Type

| Data type | Cost of continuing with CSV | SQLite | PostgreSQL etc. | Files (Parquet/npz) |
|-----------|---------------------------|--------|-----------------|---------------------|
| Merchant master, review metadata, user preferences (small) | Full table scans, poor concurrency | **Good fit** for MVP: single file, zero ops, SQL indexes | Multiple writers, high concurrency, geospatial extensions | Optional: export master to Parquet for analysis |
| Wide feature tables (many columns) | Slow, memory-intensive | Possible but **worse than Parquet** for columnar access | Same | **Recommended**: Parquet for feature training/inference |
| TF-IDF sparse matrix, large vectors | Not suited for frequent I/O | Not suitable for large blobs | Store as blob or keep on filesystem | **Keep as npz/joblib** + manifest |
| Online OLTP (orders, sessions) | Not applicable | Fine for single-machine workloads | Migrate when team scales | — |

### 3.2 Recommendation

- **Phase 1**: introduce **SQLite** (or keep CSV + strong manifest) as the **single query source for merchant/review/config data**; **keep spatial wide tables and matrices as files** (Parquet + existing joblib/npz) — don't force thousands of feature columns into SQL.
- **When to move to PostgreSQL**: when multiple-instance writes, complex permissions, PostGIS in-database geo queries, or managed cloud services are needed.
- **SQLite is not a silver bullet**: it excels at **structured metadata and relational data**, not columnar analytics. For sklearn pipelines, **pandas + Parquet** is still the smoothest path for wide feature tables.

### 3.3 Sketch SQLite Schema (if adopted)

- `merchants`: `business_id` PK, `name`, `lat`, `lon`, `city`, `state`, `stars`, `review_count`, `is_open`, `categories_json`, …
- `reviews`: `review_id` PK, `business_id` FK, `text`, `stars`, `date`, … (can sample as needed)
- `dataset_builds`: `id`, `kind` (spatial_train / retrieval_index), `path`, `checksum`, `created_at`, `git_sha`
- `model_registry`: `name` (survival / rating), `path`, `feature_schema_hash`, `created_at`

If the spatial feature table has a very large number of columns, store only `business_id` + a few key metrics + a Parquet path in SQLite, and keep the full feature table as a file loaded into memory on demand by the API service.

---

## 4. Target Architecture: Vue Frontend + API + Existing Models

### 4.1 Logical Layers

```
Vue SPA (Vite)
    │  HTTPS / JSON
    ▼
API Gateway (optional)
    │
    ▼
Python API Service (FastAPI)
    ├── Retrieval service: loads TouristRetrieval / index, exposes POST /search
    ├── Merchant prediction service: loads SpatialFeatureEngineer + joblib models, exposes POST /merchant/site-score
    ├── Metadata: GET /health, GET /datasets/active
    └── (optional) reads SQLite to serve merchant details and paginated lists
```

### 4.2 Wiring to Existing Models

- **Retrieval track**: `build_or_load_index` at startup; requests only do query + re-ranking. Response DTOs are aligned with Vue table fields so map and list components can share the same data shape.
- **Prediction track**: the logic from `tests/test_inference.py` has been extracted into a **pure function service layer** (`backend/services/merchant_inference.py`). The API layer only handles parameter validation, dispatch, and error mapping.
- **Process and memory**: large CSV/index files can be **lazily loaded + singleton**; with multiple workers, note that each worker gets its own copy in memory — consider shared read-only mmap for larger deployments.

### 4.3 API Contract (examples, for task breakdown)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Version, whether dependent data files exist |
| POST | `/api/v1/search` | body: `{ "query", "filters", "top_k" }` |
| POST | `/api/v1/merchant/predict` | body: `{ "city", "lat", "lon", "category_keys": [] }` |
| GET | `/api/v1/merchants/{business_id}` | Merchant detail (from SQLite or curated) |

All responses use `application/json`; error body is `{ "code", "message", "detail" }`.

### 4.4 Vue Notes

- **Vite + Vue 3 + TypeScript**; state management via Pinia; UI library is open (Element Plus / Naive UI).
- **Maps**: MapLibre / Leaflet (decoupled from Folium); coordinates in WGS84 matching the backend.
- **Relation to old MVP**: Streamlit has been decommissioned; Vue + OpenAPI contract is the canonical interface.

---

## 5. Phased Roadmap (Recommended Order)

| Phase | Content | Output |
|-------|---------|--------|
| **P0** | Data manifest; unify `business_id`; document lineage of both CSV tracks | `data/manifests/*` + this file / future project overview |
| **P1** | Extract inference service modules with no Streamlit dependency; unit tests cover predict | `backend/services/*` + pytest |
| **P2** | Minimal FastAPI: `/health` + `/merchant/predict` + `/search` | Runnable via Docker |
| **P3** | SQLite ETL for curated merchants; API reads DB to return detail | `scripts/etl_to_sqlite.py` |
| **P4** | Vue project setup; wire OpenAPI types; replace main flow pages | Static deploy + reverse proxy to API |
| **P5** | Feature table Parquet migration, model registry; observability and rate limiting | Production-ready |

---

## 6. Task Breakdown (Epic → Actionable Items)

### Epic A — Data Governance and Storage Decisions

- [x] **A1** Inventory all CSV paths in the repo and their consumers (scripts, notebooks, `backend/`, etc.); output a table (path, purpose, update method).
- [x] **A2** Data contract: primary key and A/B track rules were previously documented separately; core conventions are now in this file §1–2, §4, and OpenAPI. Full field table to be added to a new project overview.
- [x] **A3** Implement manifest generation step (in `pipelines` or `scripts`); CI can optionally validate "pointer exists".
- [ ] **A4** Review: confirm **SQLite scope** (merchant master only vs. including reviews) and **Parquet scope** (train_spatial).
- [x] **A5** (optional) Implement `etl_csv_to_sqlite.py` + minimal schema + migration notes.

### Epic B — Model and Inference Service Extraction

- [x] **B1** Extract `merchant_inference.predict(...)` (inputs: lat/lon/city/categories; outputs: probability + stars + select features).
- [x] **B2** Extract `retrieval_search.search(...)` or thin wrapper around `TouristRetrieval`.
- [x] **B3** Define Pydantic request/response models and OpenAPI; standardize error codes.
- [x] **B4** FastAPI route implementation + startup docs (including using `.venv` to avoid NumPy conflicts).
- [x] **B5** Containerize `Dockerfile` (API) + data volume mount conventions.

### Epic C — Vue Frontend

- [x] **C1** Initialize Vue 3 + TS + Vite project (`frontend/` subdirectory vs. separate repo decision).
- [x] **C2** Generate OpenAPI types: `frontend/openapi.json` + `npm run gen:api` → `src/api/generated.d.ts`; `src/api/client.ts` wraps fetch.
- [x] **C3** Pages: `/` health check, `/search` retrieval results table, `/merchant` site analysis form and result card (map pin selection as follow-up).
- [x] **C4** Environment variables: `VITE_API_BASE_URL`; local dev proxy.
- [x] **C5** Build and deploy: `npm run build` → `frontend/dist`; example `deploy/nginx-frontend.example.conf`.

### Epic D — Decommission and Migration

- [x] **D1** Frontend routing and capabilities now live in `frontend/` and OpenAPI (standalone comparison tables decommissioned).
- [ ] **D2** Delete or archive dead code paths; maintain root `README` and (planned) `docs/PROJECT_OVERVIEW.md`.
- [ ] **D3** (optional) E2E: Playwright for critical API + page flows.

---

## 7. Risks and Mitigations

- **Memory footprint**: index + spatial reference table both resident → benchmark peak usage; consider **splitting services** if needed (separate retrieval API and prediction API).
- **Data drift**: training features vs. online features consistency → write **feature_schema_hash** into the manifest and validate at load time.
- **Parallel team work**: lock the **OpenAPI contract first** so frontend and backend can develop in parallel; contract changes go through version numbers `/v1` → `/v2`.

---

## 8. Document Maintenance

- This document lives at `docs/refactor-plan-data-vue-api.md` (**the only long-lived planning/migration doc in `docs/`**).
- **Actual table schemas, API base URLs, and environment variables**: maintained in the root `README.md`. Field-level request/response contracts live in OpenAPI. A one-page project overview can be added at `docs/PROJECT_OVERVIEW.md` if needed for external presentations.

---

*Document version: initial draft, for sprint review and task breakdown. P3/P5 can be trimmed depending on team size.*
