# 餐厅推荐系统算法说明

本文档概括当前仓库中**检索与推荐**相关实现（离线索引、在线打分、可选 RL 策略、点赞/点踩重排），便于报告与代码对照。

---

## 1. 离线索引：基于评论的 TF-IDF

**模块**：`backend/dining_retrieval/core/retrieval.py`（`TouristRetrieval`、`RestaurantSearchIndex`）

- 将每家餐厅表示为一段「文档」：聚合该店的多条评论文本，并混入类目、名称、城市/州等元信息；可选接入 Google Maps 补充描述。
- 使用 **sklearn `TfidfVectorizer`** 在文档集合上拟合，得到每家店的**稀疏 TF-IDF 向量**及预计算 **L2 范数**，供后续**余弦相似度**计算。
- 星级做归一化，并通过 `rating_trust_ref_reviews` 做**评论量信任收缩**：评论数少时，星级向中性收缩，降低刷单/样本偏差影响。
- 索引候选集按 `review_count` 等规则抽样；可按 `restrict_index_cities` 与城市白名单过滤覆盖范围。

---

## 2. 在线检索：`recommend_keywords` 多因子打分

**模块**：`TouristRetrieval.recommend_keywords`

用户侧最终得到一段 **`query_text`**（由自然语言解析、额外关键词、菜系等组合；Discover 模式可为泛查询如 `restaurants`）。

1. **语义相关**：`query_text` 经同一 `vectorizer` 得到查询向量，与过滤后的餐厅向量计算**余弦相似度**。
2. **过滤**：州、城市；可选**菜系**（在 `categories` 上做关键词规则匹配，若无结果会放宽菜系以避免空结果）；可选**参考经纬度 + 最大半径**（Haversine 距离，公里）。
3. 将各子信号归一化后，**线性加权**得到 **`final_score`**：
   - **语义** `sim_n`
   - **星级** `stars_rank`（带信任收缩）
   - **价格匹配** `price_match`：相对预算档位（cheap / moderate / expensive）的软匹配
   - **距离** `dist_score`：离参考点越近越高；无参考点则为中性
   - **热度** `pop_n`：基于 `log1p(review_count)` 的 min-max

权重为 **`w_semantic`、`w_rating`、`w_price`、`w_distance`、`w_popularity`**，由 API/前端传入。先在较大的 **`pool_k`** 候选池上排序，再取 **`top_k`** 用于展示。

---

## 3. 查询理解与 API 对齐

**模块**：`backend/dining_retrieval/search/query_parser.py`、`backend/services/retrieval_service.py`

- **`parse_query`**：从自然语言中抽取语义片段、预算、参考点与半径等。
- **`extract_budget_hint`**：可从约束字符串中补全预算提示。
- 与前端 **Discover / Refine**、菜系多选、权重滑条等字段对齐（参见 `backend/api/schemas.py` 中 `SearchRequest`）。

---

## 4. 强化学习层：按意图桶的 Contextual Bandit（UCB）

**模块**：`models/rl_feedback_loop.py`、`RetrievalSearchService` 内 `_RL_WEIGHT_PRESETS` 与 `search()`

- **`classify_query_intent`**：用**正则规则**将查询粗分为 `intent_quick`、`intent_romantic`、`intent_default`（例如 quick/cheap 与 date/romantic 等关键词）。
- **`RLFeedbackLoop`**：在每个意图桶下维护三个**臂（arm）**：`explorer`、`reputation`、`convenience`。
- **选臂**：**UCB**（`select_strategy`）— 未尝试过的臂优先探索；否则按 `Q + c * sqrt(log N / n)` 选择。
- **臂 → 检索权重**：每个臂对应一套五维权重预设（与手动滑条同一套 `recommend_keywords` 公式）。若用户 **`rl_user_overrode`**（动过滑条），则当次仍使用请求体中的手动权重。
- **反馈**：请求携带上一轮的臂、意图与 **`rl_action_events`**；`detail_open`、`like` 等映射为**正奖励**，`refresh`、`slider_override` 等为**小负奖励**，通过 **`log_user_feedback`** 更新 Q（形式为 `new_q = old_q + alpha * (reward - old_q)`），状态持久化到 JSON/CSV。

整体上这是**「粗意图上下文 + 多臂 bandit 选择排序策略」**，而非端到端训练的深度排序模型。

---

## 5. 交互式重排：点赞 / 点踩（v2）

**模块**：`backend/dining_retrieval/recommendation/reranker.py`、`UserPreferenceState`

当存在 **`liked_business_ids` / `disliked_business_ids`** 时，在候选池上调用 **`rerank_pool`**：

- 在原有 **`final_score`**（min-max 后）基础上，用索引中**同一 TF-IDF 空间**的**行余弦相似度**，计算与各喜欢/不喜欢商户的平均相似度。
- 结合结构化偏好启发式（如最低星级、最大距离、类目关键词等）得到 **`v2_score`**，按 **`v2_score`** 降序重排后截取 **`top_k`**。

---

## 6. 数据流小结

```text
离线：评论/元数据 → 文档 → TF-IDF 索引（向量 + meta + stars_norm）
在线：query_text → 余弦相似度 + 过滤 → 多因子 final_score → pool_k → top_k
可选：意图分类 → UCB 选臂 → 替换五维权重
可选：点赞/点踩 → v2 重排
异步：用户行为 → reward → 更新各意图下各臂的 Q 值
```

---

## 7. 相关文件索引

| 主题 | 路径 |
|------|------|
| 索引构建与 `recommend_keywords` | `backend/dining_retrieval/core/retrieval.py` |
| 检索服务入口（RL、重排、照片） | `backend/services/retrieval_service.py` |
| 查询解析 | `backend/dining_retrieval/search/query_parser.py` |
| UCB 与意图、反馈日志 | `models/rl_feedback_loop.py` |
| 点赞踩重排 | `backend/dining_retrieval/recommendation/reranker.py` |
| 请求体字段 | `backend/api/schemas.py` |

如需补充**离线评估**（`models/offline_evaluator.py`）或**索引中评论情感分桶**等细节，可在此文档后续追加小节。
