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
    business_score_ml_pkl: bool = False
    retrieval_business_csv: bool
    retrieval_index: bool


class MerchantPredictRequest(BaseModel):
    city: Optional[str] = Field(None, description="Filter reference businesses; if empty, use the first N rows of the table")
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
    price_level: Optional[int] = Field(
        None,
        ge=1,
        le=4,
        description="Yelp-style price tier 1–4; compared to mean attr_restaurantspricerange2 within 1 km.",
    )
    price_per_person: Optional[float] = Field(
        None,
        ge=0.0,
        description="USD-ish per person; mapped to 1–4 if price_level not set.",
    )


class MerchantCityRow(BaseModel):
    city: str
    state: str = ""
    row_count: int
    center_lat: float
    center_lon: float


class MerchantCitiesResponse(BaseModel):
    cities: list[MerchantCityRow]


class MerchantCategoriesResponse(BaseModel):
    """cat_* names aligned with the predict slice for the current city/state/max_rows."""

    category_keys: list[str] = Field(
        default_factory=list,
        description="cat_* names in train_spatial that appear at least once in the slice (all-zero columns dropped unless all are zero).",
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
    price_fit: Optional[str] = Field(
        None,
        description="good | medium | poor — rule-based match vs mean local price level (1 km).",
    )
    price_gap: Optional[float] = None
    nearby_avg_price_level: Optional[float] = None
    risk: dict[str, str] = Field(
        default_factory=dict,
        description="competition, location, price: high|medium|low|unknown (rule-based).",
    )
    explanation: str = Field("", description="Short human-readable summary for the business user.")
    business_score: Optional[float] = Field(
        None,
        description="0–100 composite from survival, stars, competition, price match (rule-based V1).",
    )
    business_score_ml: Optional[float] = Field(
        None,
        description="0–100 supervised score: P(is_open) from spatial/category features only (no survival/rating heads as inputs). Null if artifact missing.",
    )


class MerchantHeatmapRequest(BaseModel):
    """Same filters as merchant predict, without a pin — scores a regular grid over the slice bbox."""

    city: Optional[str] = None
    state: Optional[str] = Field(
        None,
        description="Optional USPS state (must match train_spatial.state when used).",
    )
    category_keys: list[str] = Field(
        default_factory=list,
        description="Explicit cat_* columns; optional if category_query is set.",
    )
    category_query: Optional[str] = Field(
        None,
        description="Plain text resolved to cat_* in the current slice (same as /merchant/predict).",
    )
    max_rows_if_no_city: int = Field(2000, ge=100, le=50000)
    price_level: Optional[int] = Field(
        None,
        ge=1,
        le=4,
        description="Yelp-style tier vs local mean within 1 km.",
    )
    price_per_person: Optional[float] = Field(None, ge=0.0)
    grid_size: int = Field(
        12,
        ge=4,
        le=16,
        description="Number of rows/columns over min/max lat/lon of the reference slice (max 256 cells).",
    )


class MerchantHeatmapResponse(BaseModel):
    city_filter: Optional[str] = None
    reference_row_count: int
    grid_size: int
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    resolved_category_keys: list[str] = Field(default_factory=list)
    business_score: list[list[Optional[float]]] = Field(
        default_factory=list,
        description="Row index ~ south→north, column ~ west→east; null = outside training hull (skipped).",
    )
    business_score_ml: list[list[Optional[float]]] = Field(default_factory=list)
    survival_probability: list[list[Optional[float]]] = Field(
        default_factory=list,
        description="Survival head probability 0–1 per cell; null when cell skipped.",
    )
    predicted_stars: list[list[Optional[float]]] = Field(
        default_factory=list,
        description="Rating regressor: predicted stars (~0–5) per cell; null when cell skipped.",
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
    business_id: Optional[str] = Field(None, description="Related business ID (if any)")
    query_text: Optional[str] = Field(None, description="query_text when the action was triggered")


class SearchRequest(BaseModel):
    """Search request body aligned with the `/search` page (Step1–2 and sidebar weights)."""

    query: str = Field("", description="Step2 natural language; may be empty when discover_only is true")
    state: str = Field(..., description="USPS state code")
    city: Optional[str] = Field(None, description="Optional city name (exact match)")
    user_location: Optional[str] = Field(None, description="User's current location input for distance ranking")
    top_k: int = Field(10, ge=1, le=100)
    pool_k: Optional[int] = Field(
        None,
        ge=15,
        le=500,
        description="Internal candidate pool size Top-N (frontend default 45)",
    )
    keywords_extra: Optional[str] = Field(None, description="Extra keywords")
    force_rebuild_index: bool = False
    discover_only: bool = Field(
        False,
        description="True = discover-only / 'Find general restaurants here' mode",
    )
    cuisines: list[str] = Field(
        default_factory=list,
        description="Cuisine multiselect: Sushi, Steakhouse, Korean, ...",
    )
    w_semantic: float = Field(0.85, ge=0.0, le=2.0)
    w_rating: float = Field(1.05, ge=0.0, le=2.0)
    w_price: float = Field(0.15, ge=0.0, le=2.0)
    w_distance: float = Field(0.2, ge=0.0, le=2.0)
    w_popularity: float = Field(0.1, ge=0.0, le=2.0)
    liked_business_ids: list[str] = Field(default_factory=list)
    disliked_business_ids: list[str] = Field(default_factory=list)
    rl_enabled: bool = Field(True, description="Whether to enable RL initial policy selection")
    rl_user_overrode: bool = Field(False, description="User manually overrode weight sliders")
    rl_prev_selected_arm: Optional[str] = Field(None, description="Last RL selected arm")
    rl_prev_intent_name: Optional[str] = Field(None, description="Last recognized RL intent name")
    rl_action_events: list[SearchActionEvent] = Field(
        default_factory=list,
        description="Client action events since the last search (RL feedback)",
    )


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]
    meta: dict[str, Any]
