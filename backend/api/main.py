"""
FastAPI 入口：健康检查、商家预测、检索。

启动（仓库根目录）：`./scripts/run_api.sh`（已设置 `PYTHONPATH=backend:.`）
或: `PYTHONPATH=backend:. .venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    MerchantCitiesResponse,
    MerchantCoverageResponse,
    MerchantPredictRequest,
    MerchantPredictResponse,
    SearchRequest,
    SearchResponse,
    StatesResponse,
)
from services.merchant_inference import (
    artifact_paths,
    get_merchant_coverage,
    list_merchant_spatial_cities,
    predict_merchant_site,
    spatial_train_csv_path,
)
from services.retrieval_service import RetrievalSearchService

API_VERSION = "0.1.0"


def get_repo_root() -> Path:
    env = os.environ.get("API_REPO_ROOT")
    if env:
        return Path(env).resolve()
    # backend/api/main.py → 仓库根
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
    return HealthResponse(
        ok=True,
        version=API_VERSION,
        repo_root=str(root),
        spatial_csv=spatial_ok,
        survival_pkl=surv.is_file(),
        rating_pkl=rat.is_file(),
        retrieval_business_csv=biz.is_file(),
        retrieval_index=idx.is_file(),
    )


@app.post("/api/v1/merchant/predict", response_model=MerchantPredictResponse)
def merchant_predict(body: MerchantPredictRequest) -> MerchantPredictResponse:
    try:
        r = predict_merchant_site(
            city=body.city,
            state=body.state,
            lat=body.lat,
            lon=body.lon,
            selected_category_columns=body.category_keys,
            repo_root=_repo,
            max_rows_if_no_city=body.max_rows_if_no_city,
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
    )


@app.get("/api/v1/merchant/cities", response_model=MerchantCitiesResponse)
def merchant_cities(
    min_rows: int = Query(10, ge=1, le=5000),
) -> MerchantCitiesResponse:
    try:
        rows = list_merchant_spatial_cities(repo_root=_repo, min_rows=min_rows)
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex)) from ex
    return MerchantCitiesResponse(cities=rows)


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

    opts = union_state_options(_repo / "data" / "cleaned")
    return StatesResponse(states=list(opts or []))


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
