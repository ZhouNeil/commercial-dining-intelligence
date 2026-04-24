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
    city: Optional[str] = Field(None, description="过滤参考商户；为空则取表头前 N 行")
    lat: float
    lon: float
    category_keys: list[str] = Field(
        ...,
        description="train_spatial 中的 cat_* 列名，如 cat_coffee_&_tea",
    )
    max_rows_if_no_city: int = Field(2000, ge=100, le=50000)


class MerchantPredictResponse(BaseModel):
    survival_probability: float
    predicted_stars: float
    reference_row_count: int
    city_filter: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    live_feature_preview: dict[str, float] = Field(default_factory=dict)


class StatesResponse(BaseModel):
    states: list[str]


class SearchActionEvent(BaseModel):
    action: str = Field(..., description="detail_open | like | refresh | slider_override")
    business_id: Optional[str] = Field(None, description="相关商户 ID（如适用）")
    query_text: Optional[str] = Field(None, description="触发该动作时的 query_text")


class SearchRequest(BaseModel):
    """与前端 `/search`（Step1–2 与侧栏权重）对齐的检索请求字段。"""

    query: str = Field("", description="Step2 自然语言；discover_only 时可空")
    state: str = Field(..., description="USPS 州码")
    city: Optional[str] = Field(None, description="可选城市名（精确匹配）")
    top_k: int = Field(10, ge=1, le=100)
    pool_k: Optional[int] = Field(
        None,
        ge=15,
        le=500,
        description="内部候选池 Top-N（前端默认 45）",
    )
    keywords_extra: Optional[str] = Field(None, description="额外关键词")
    force_rebuild_index: bool = False
    discover_only: bool = Field(
        False,
        description="True = 「Find general restaurants here」",
    )
    cuisines: list[str] = Field(
        default_factory=list,
        description="菜系多选：Sushi, Steakhouse, Korean, …",
    )
    w_semantic: float = Field(0.85, ge=0.0, le=2.0)
    w_rating: float = Field(1.05, ge=0.0, le=2.0)
    w_price: float = Field(0.15, ge=0.0, le=2.0)
    w_distance: float = Field(0.2, ge=0.0, le=2.0)
    w_popularity: float = Field(0.1, ge=0.0, le=2.0)
    liked_business_ids: list[str] = Field(default_factory=list)
    disliked_business_ids: list[str] = Field(default_factory=list)
    rl_enabled: bool = Field(True, description="是否启用 RL 初始策略选择")
    rl_user_overrode: bool = Field(False, description="用户是否手动接管了权重滑条")
    rl_prev_selected_arm: Optional[str] = Field(None, description="上一次 RL 选中的 arm")
    rl_prev_intent_name: Optional[str] = Field(None, description="上一次 RL 识别的 intent")
    rl_action_events: list[SearchActionEvent] = Field(
        default_factory=list,
        description="自上次搜索以来的前端动作事件，用于 RL 反馈",
    )


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]
    meta: dict[str, Any]
