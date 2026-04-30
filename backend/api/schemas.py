from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool
    version: str = "0.1.0"
    repo_root: str
    spatial_csv: bool
    survival_pkl: bool
    rating_pkl: bool
    retrieval_business_csv: bool
    retrieval_index: bool


class MerchantPredictRequest(BaseModel):
    city: Optional[str] = Field(None, description="Filter reference merchants by city; if empty, uses the first N rows.")
    state: Optional[str] = Field(
        None,
        description="Optional USPS state to disambiguate duplicate city names (must match train_spatial.state).",
    )
    lat: float
    lon: float
    category_keys: list[str] = Field(
        default_factory=list,
        description="Explicit train_spatial cat_* columns. Optional if category_query is set.",
    )
    category_query: Optional[str] = Field(
        None,
        description="Plain text (e.g. 'fast food, coffee', 'burger'); server maps to cat_* in the current slice.",
    )
    max_rows_if_no_city: int = Field(2000, ge=100, le=50000)


class MerchantCityRow(BaseModel):
    city: str
    state: str = ""
    row_count: int
    center_lat: float
    center_lon: float


class MerchantCitiesResponse(BaseModel):
    cities: list[MerchantCityRow]


class MerchantCategoriesResponse(BaseModel):
    """cat_* column names matching those used by predict for the current city/state/max_rows slice."""

    category_keys: list[str] = Field(
        default_factory=list,
        description="cat_* column names from train_spatial that appear at least once in the slice (all-zero columns excluded).",
    )


class MerchantPredictResponse(BaseModel):
    survival_probability: float
    predicted_stars: float
    reference_row_count: int
    city_filter: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    live_feature_preview: dict[str, float] = Field(default_factory=dict)
    inside_reference_hull: bool = Field(
        ...,
        description="True if the pin lies inside the convex hull of reference training coordinates for this slice.",
    )
    resolved_category_keys: list[str] = Field(
        default_factory=list,
        description="cat_* columns used for this prediction (from category_query and/or category_keys).",
    )


class MerchantCoverageResponse(BaseModel):
    """Map overlay: bbox, centroid, convex hull of reference rows, sampled training points."""

    city_filter: Optional[str] = None
    reference_count: int
    geo_count: int
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    center_lon: float
    center_lat: float
    hull_geojson: Optional[dict[str, Any]] = None
    sample_points_geojson: Optional[dict[str, Any]] = None
    valid_hull: bool


class StatesResponse(BaseModel):
    states: list[str]


class SearchActionEvent(BaseModel):
    action: str = Field(..., description="detail_open | like | pass | refresh | slider_override")
    business_id: Optional[str] = Field(None, description="Related business ID (if applicable).")
    query_text: Optional[str] = Field(None, description="The query_text active when this action was triggered.")


class SearchRequest(BaseModel):
    """Retrieval request fields aligned with the frontend `/search` flow (Step 1–2 and sidebar weights)."""

    query: str = Field("", description="Natural language query (Step 2); may be empty when discover_only is True.")
    state: str = Field(..., description="USPS state code.")
    city: Optional[str] = Field(None, description="Optional city name (exact match).")
    user_location: Optional[str] = Field(None, description="User's current location input for distance ranking")
    top_k: int = Field(10, ge=1, le=100)
    pool_k: Optional[int] = Field(
        None,
        ge=15,
        le=500,
        description="Internal candidate pool size Top-N (frontend default: 45).",
    )
    keywords_extra: Optional[str] = Field(None, description="Additional keywords to append to the query.")
    force_rebuild_index: bool = False
    discover_only: bool = Field(
        False,
        description="True = broad discovery (ignore NL query, return top restaurants for the area).",
    )
    cuisines: list[str] = Field(
        default_factory=list,
        description="Multi-select cuisine filter: Sushi, Steakhouse, Korean, …",
    )
    w_semantic: float = Field(0.85, ge=0.0, le=2.0)
    w_rating: float = Field(1.05, ge=0.0, le=2.0)
    w_price: float = Field(0.15, ge=0.0, le=2.0)
    w_distance: float = Field(0.2, ge=0.0, le=2.0)
    w_popularity: float = Field(0.1, ge=0.0, le=2.0)
    liked_business_ids: list[str] = Field(default_factory=list)
    disliked_business_ids: list[str] = Field(default_factory=list)
    rl_enabled: bool = Field(True, description="Whether to enable RL-based initial strategy selection.")
    rl_user_overrode: bool = Field(False, description="Whether the user manually overrode the RL ranking weights.")
    rl_prev_selected_arm: Optional[str] = Field(None, description="The RL arm selected in the previous request.")
    rl_prev_intent_name: Optional[str] = Field(None, description="The intent bucket identified in the previous request.")
    rl_action_events: list[SearchActionEvent] = Field(
        default_factory=list,
        description="Frontend action events since the last search, used as RL feedback.",
    )


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]
    meta: dict[str, Any]
