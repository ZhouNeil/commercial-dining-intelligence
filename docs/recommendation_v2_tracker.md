# 推荐系统 v2（交互式）— 实施跟踪

面向《餐厅推荐系统说明》文档中的 **Restaurant Recommendation System — Interactive Upgrade (v2)**。本文件用 checkbox 记录进度；完成后可将日期记在条目后。

---

## 阶段划分

### Phase A — 基础：候选池 + 会话状态 + 最小重排（进行中）

- [x] **A1** 跟踪文档 `docs/recommendation_v2_tracker.md`
- [x] **A2** `recommend_keywords` 支持 `pool_k`（大于 `top_k` 时保留大候选池）与 `include_business_id`
- [x] **A3** `app/recommendation/preference_state.py`：会话级偏好结构（like / dislike 等）
- [x] **A4** `app/recommendation/reranker.py`：基于 TF-IDF 行向量与 v1 `final_score` 的二次打分
- [x] **A5** `main.py`：侧栏候选池大小、新检索清空反馈、`👍`/`👎`、展示池内 Top-K 与重排后列表

### Phase B — 反馈类型扩展

- [ ] **B1** 显式按钮：`Too expensive` / `Too far` / `Wrong cuisine` / `More like this`（映射到 `preference_state` 字段）
- [ ] **B2** `feedback_parser.py`：短 NL 反馈（如 “cheaper”“not sushi”）复用或调用 `parse_query` 子集
- [ ] **B3** 强约束变更时触发 **重新检索**（预算/半径/菜系），再进入同一候选池 + 重排逻辑

### Phase C — 体验与稳健性

- [ ] **C1** 侧栏调节 v2 权重（`w_base` / `w_like` / `w_dislike` / `w_pref`）
- [ ] **C2** 候选处理 liked 不在索引中的降级（仅用类目/价位/地理）
- [ ] **C3** 与 `generate_insight` 联动（展示“因你喜欢 X 而提升/压低”）

### Phase D — 远期（文档 §14）

- [ ] **D1** 稠密向量 + FAISS / 近似最近邻
- [ ] **D2** Learning-to-rank 或从日志学权重

---

## 设计备忘（避免走偏）

1. **候选池固定时**，仅重排无法解决“池里没有更便宜店”的问题 — Phase B3 需支持再拉检索。
2. **item–item 相似度**必须在当前 `RestaurantSearchIndex` 的同一 TF-IDF 空间内算余弦。
3. 新检索应 **重置** 会话内 like/dislike（或显式“保留偏好”高级选项，暂未做）。

---

*最后更新：初始化 Phase A 并完成 A1–A5。*
