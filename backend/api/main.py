"""
FastAPI entry: health, merchant predict, search.

From repo root: `./scripts/run_api.sh` (sets `PYTHONPATH=backend:.`)
or: `PYTHONPATH=backend:. .venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    MerchantCategoriesResponse,
    MerchantCitiesResponse,
    MerchantCoverageResponse,
    MerchantHeatmapRequest,
    MerchantHeatmapResponse,
    MerchantPredictRequest,
    MerchantPredictResponse,
    SearchRequest,
    SearchResponse,
    StatesResponse,
)
from services.merchant_inference import (
    SPATIAL_CITY_MIN_TRAIN_ROWS,
    artifact_paths,
    clear_merchant_spatial_cities_cache,
    get_merchant_coverage,
    list_merchant_category_keys,
    list_merchant_spatial_cities,
    predict_merchant_heatmap,
    predict_merchant_site,
    resolve_merchant_category_text,
    suggest_merchant_category_text,
    spatial_train_csv_path,
)
from services.retrieval_service import RetrievalSearchService

API_VERSION = "0.1.0"


def get_repo_root() -> Path:
    env = os.environ.get("API_REPO_ROOT")
    if env:
        return Path(env).resolve()
    # backend/api/main.py -> repo root
    return Path(__file__).resolve().parents[2]


app = FastAPI(title="Commercial Dining Intelligence API", version=API_VERSION)

_origins = os.environ.get("CORS_ORIGINS", "*")
_allow = [o.strip() for o in _origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow if _allow != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_repo = get_repo_root()
_retrieval: Optional[RetrievalSearchService] = None


@app.on_event("startup")
def _startup_clear_spatial_city_cache() -> None:
    # Avoid stale spatial city LRU in gunicorn --preload children; no cost for single process
    clear_merchant_spatial_cities_cache()


def retrieval_service() -> RetrievalSearchService:
    global _retrieval
    if _retrieval is None:
        _retrieval = RetrievalSearchService(_repo)
    return _retrieval


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    root = _repo
    surv, rat = artifact_paths(root)
    spatial_ok = False
    try:
        spatial_train_csv_path(root)
        spatial_ok = True
    except OSError:
        pass
    biz = root / "data" / "cleaned" / "business_dining.csv"
    idx = root / "models" / "artifacts" / "restaurant_matrix.npz"
    bs_ml = root / "models" / "artifacts" / "business_score_ml.pkl"
    return HealthResponse(
        ok=True,
        version=API_VERSION,
        repo_root=str(root),
        spatial_csv=spatial_ok,
        survival_pkl=surv.is_file(),
        rating_pkl=rat.is_file(),
        business_score_ml_pkl=bs_ml.is_file(),
        retrieval_business_csv=biz.is_file(),
        retrieval_index=idx.is_file(),
    )


@app.post("/api/v1/merchant/predict", response_model=MerchantPredictResponse)
def merchant_predict(body: MerchantPredictRequest) -> MerchantPredictResponse:
    q = (body.category_query or "").strip()
    if q:
        try:
            keys = resolve_merchant_category_text(
                q,
                city=body.city,
                state=body.state.strip().upper() if body.state and str(body.state).strip() else None,
                max_rows_if_no_city=body.max_rows_if_no_city,
                repo_root=_repo,
            )
        except FileNotFoundError as ex:
            raise HTTPException(status_code=503, detail=str(ex)) from ex
        if not keys:
            sugg = suggest_merchant_category_text(
                q,
                city=body.city,
                state=body.state.strip().upper() if body.state and str(body.state).strip() else None,
                max_rows_if_no_city=body.max_rows_if_no_city,
                repo_root=_repo,
                limit=8,
            )
            sugg_txt = ", ".join(sugg[:8]) if sugg else "pizza, fast food, coffee, burger"
            raise HTTPException(
                status_code=400,
                detail=(
                    "No training category match for this text and city slice. "
                    f"Suggestions (cat_*): {sugg_txt}"
                ),
            )
    else:
        keys = list(body.category_keys)
    if not keys:
        raise HTTPException(
            status_code=400,
            detail="Send category_query (plain text) or non-empty category_keys.",
        )
    try:
        r = predict_merchant_site(
            city=body.city,
            state=body.state,
            lat=body.lat,
            lon=body.lon,
            selected_category_columns=keys,
            repo_root=_repo,
            max_rows_if_no_city=body.max_rows_if_no_city,
            price_level=body.price_level,
            price_per_person=body.price_per_person,
        )
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex)) from ex
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    return MerchantPredictResponse(
        survival_probability=r.survival_probability,
        predicted_stars=r.predicted_stars,
        reference_row_count=r.reference_row_count,
        city_filter=r.city_filter,
        metrics=r.metrics,
        live_feature_preview=r.live_feature_preview,
        inside_reference_hull=r.inside_reference_hull,
        resolved_category_keys=keys,
        price_fit=r.price_fit,
        price_gap=r.price_gap,
        nearby_avg_price_level=r.nearby_avg_price_level,
        risk=r.risk,
        explanation=r.explanation,
        business_score=r.business_score,
        business_score_ml=r.business_score_ml,
    )


@app.get("/api/v1/merchant/heatmap")
def merchant_heatmap_get() -> dict[str, str]:
    """Browser-friendly probe; heavy work uses POST."""
    return {
        "message": "POST JSON here with MerchantHeatmapRequest (same as /merchant/predict but no lat/lon; add grid_size).",
    }


@app.post("/api/v1/merchant/heatmap", response_model=MerchantHeatmapResponse)
def merchant_heatmap(body: MerchantHeatmapRequest) -> MerchantHeatmapResponse:
    q = (body.category_query or "").strip()
    if q:
        try:
            keys = resolve_merchant_category_text(
                q,
                city=body.city,
                state=body.state.strip().upper() if body.state and str(body.state).strip() else None,
                max_rows_if_no_city=body.max_rows_if_no_city,
                repo_root=_repo,
            )
        except FileNotFoundError as ex:
            raise HTTPException(status_code=503, detail=str(ex)) from ex
        if not keys:
            sugg = suggest_merchant_category_text(
                q,
                city=body.city,
                state=body.state.strip().upper() if body.state and str(body.state).strip() else None,
                max_rows_if_no_city=body.max_rows_if_no_city,
                repo_root=_repo,
                limit=8,
            )
            sugg_txt = ", ".join(sugg[:8]) if sugg else "pizza, fast food, coffee, burger"
            raise HTTPException(
                status_code=400,
                detail=(
                    "No training category match for this text and city slice. "
                    f"Suggestions (cat_*): {sugg_txt}"
                ),
            )
    else:
        keys = list(body.category_keys)
    if not keys:
        raise HTTPException(
            status_code=400,
            detail="Send category_query (plain text) or non-empty category_keys.",
        )
    try:
        r = predict_merchant_heatmap(
            city=body.city,
            state=body.state,
            selected_category_columns=keys,
            grid_size=body.grid_size,
            repo_root=_repo,
            max_rows_if_no_city=body.max_rows_if_no_city,
            price_level=body.price_level,
            price_per_person=body.price_per_person,
        )
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex)) from ex
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    return MerchantHeatmapResponse(
        city_filter=r.city_filter,
        reference_row_count=r.reference_row_count,
        grid_size=r.grid_size,
        min_lat=r.min_lat,
        max_lat=r.max_lat,
        min_lon=r.min_lon,
        max_lon=r.max_lon,
        resolved_category_keys=list(r.resolved_category_keys),
        business_score=[list(row) for row in r.business_score],
        business_score_ml=[list(row) for row in r.business_score_ml],
        survival_probability=[list(row) for row in r.survival_probability],
        predicted_stars=[list(row) for row in r.predicted_stars],
    )


@app.get("/api/v1/merchant/cities", response_model=MerchantCitiesResponse)
def merchant_cities(
    min_rows: int = Query(SPATIAL_CITY_MIN_TRAIN_ROWS, ge=1, le=5000),
) -> MerchantCitiesResponse:
    # Enforce project floor for city list (same as frontend / merchant page) even if min_rows is 1/2
    eff = max(int(min_rows), int(SPATIAL_CITY_MIN_TRAIN_ROWS))
    try:
        rows = list_merchant_spatial_cities(repo_root=_repo, min_rows=eff)
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex)) from ex
    return MerchantCitiesResponse(cities=rows)


@app.get("/api/v1/merchant/categories", response_model=MerchantCategoriesResponse)
def merchant_categories(
    city: Optional[str] = None,
    state: Optional[str] = None,
    max_rows_if_no_city: int = Query(2000, ge=100, le=50000),
) -> MerchantCategoriesResponse:
    try:
        keys = list_merchant_category_keys(
            city=city,
            state=state.strip().upper() if state and str(state).strip() else None,
            max_rows_if_no_city=max_rows_if_no_city,
            repo_root=_repo,
        )
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex)) from ex
    return MerchantCategoriesResponse(category_keys=keys)


@app.get("/api/v1/merchant/categories/resolve", response_model=MerchantCategoriesResponse)
def merchant_categories_resolve(
    q: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="Free text, e.g. 'burger', 'fast food, coffee' — matched to train_spatial cat_* in this city slice.",
    ),
    city: Optional[str] = None,
    state: Optional[str] = None,
    max_rows_if_no_city: int = Query(2000, ge=100, le=50000),
) -> MerchantCategoriesResponse:
    try:
        st = state.strip().upper() if state and str(state).strip() else None
        keys = resolve_merchant_category_text(
            q,
            city=city,
            state=st,
            max_rows_if_no_city=max_rows_if_no_city,
            repo_root=_repo,
        )
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex)) from ex
    if not keys:
        sugg = suggest_merchant_category_text(
            q,
            city=city,
            state=st,
            max_rows_if_no_city=max_rows_if_no_city,
            repo_root=_repo,
            limit=8,
        )
        sugg_txt = ", ".join(sugg[:8]) if sugg else "pizza, fast food, coffee, burger"
        raise HTTPException(
            status_code=400,
            detail=(
                "No train_spatial category match for this text and city slice. "
                f"Suggestions (cat_*): {sugg_txt}"
            ),
        )
    return MerchantCategoriesResponse(category_keys=keys)


@app.get("/api/v1/merchant/coverage", response_model=MerchantCoverageResponse)
def merchant_coverage(
    city: Optional[str] = None,
    state: Optional[str] = None,
    max_rows_if_no_city: int = Query(2000, ge=100, le=50000),
    max_sample_points: int = Query(400, ge=50, le=5000),
) -> MerchantCoverageResponse:
    try:
        d = get_merchant_coverage(
            city=city,
            state=state,
            repo_root=_repo,
            max_rows_if_no_city=max_rows_if_no_city,
            max_sample_points=max_sample_points,
        )
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex)) from ex
    return MerchantCoverageResponse(**d)


@app.get("/api/v1/states", response_model=StatesResponse)
def list_states() -> StatesResponse:
    from dining_retrieval.core.google_maps_loader import union_state_options

    opts = set(union_state_options(_repo / "data" / "cleaned"))

    # Filter to only states present in the built index — prevents showing states
    # that would always return zero search results.
    index_meta = _repo / "models" / "artifacts" / "meta.csv"
    if index_meta.exists():
        import pandas as pd
        meta = pd.read_csv(index_meta, usecols=["state"], low_memory=False)
        indexed = set(meta["state"].dropna().astype(str).str.strip().str.upper().unique())
        opts = opts & indexed

    return StatesResponse(states=sorted(opts))


@app.post("/api/v1/search", response_model=SearchResponse)
def search_v1(body: SearchRequest) -> SearchResponse:
    try:
        rows, meta = retrieval_service().search(**body.model_dump())
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex)) from ex
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(ex)) from ex
    return SearchResponse(results=rows, meta=meta)
