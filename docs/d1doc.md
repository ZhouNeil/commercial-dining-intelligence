# 餐厅推荐系统 — 开发说明（d1doc）

> 本文档描述**当前仓库已实现**的架构与技术栈；与 `app/core/retrieval.py`、`app/main.py` 等行为保持一致。历史版本中「以稠密向量 + FAISS 为主检索」的表述已过时，以下文为准。

## 1. 项目概览

本仓库是机器学习课程大作业：**基于 Yelp 开放数据（及可选补充数据）的餐饮商业情报与推荐**。面向用户侧提供：

- **游客/食客模式**：自然语言与筛选条件驱动的餐厅检索、多因子排序、可解释摘要，以及 👍/👎 反馈下的池内二次排序（v2）。
- **商家侧分析（独立页面）**：K-Means 聚类可视化（`app/pages/1_KMeans_聚类.py`），与 `models/`、`notebooks/` 中的实验代码呼应。

与传统关键词搜索相比，系统在返回列表的同时，用**规则化说明**解释「为何推荐」（无需 LLM）。

---

## 2. 仓库目录结构（与实现一致）

```text
commercial-dining-intelligence/
├── app/                          # Streamlit 应用
│   ├── main.py                   # 推荐主流程入口（游客模式 MVP）
│   ├── pages/
│   │   └── 1_KMeans_聚类.py      # 多页面：商家 K-Means 可视化
│   ├── core/                     # 索引构建、数据加载、画像文本
│   │   ├── retrieval.py          # TouristRetrieval：TF-IDF 索引与 recommend_keywords
│   │   ├── profile_builder.py    # 评论片段 → 可检索「店铺文档」
│   │   ├── google_maps_loader.py # 与 Google Maps 衍生 CSV 的合并/州列表等
│   │   ├── index_cities.py       # 索引城市过滤配置
│   │   └── yelp_photos.py        # Yelp Photos 资源（json/tar/本地目录）
│   ├── search/
│   │   ├── query_parser.py       # 规则化 NL → ParsedQuery
│   │   ├── semantic_filters.py   # 可选：MiniLM 与州名候选匹配
│   │   ├── insights.py           # 无 LLM 的 pros/cons/why
│   │   └── geo_constants.py      # 美国州名/缩写等
│   ├── recommendation/
│   │   ├── preference_state.py   # Session 中的偏好状态
│   │   └── reranker.py           # v2：基于 TF-IDF 与 👍/👎 的池内重排
│   └── clustering/               # K-Means Streamlit 页所用逻辑
├── models/                       # 离线 ML、实验与 artifact 约定路径
│   ├── artifacts/                # 检索索引落盘（vectorizer、稀疏矩阵、meta 等，运行时生成）
│   └── …                         # kmeans、merchant_predictor、rl_feedback_loop 等 notebook/脚本侧代码
├── pipelines/                    # 数据清洗、PCA 等批处理脚本
├── notebooks/                    # EDA、实验笔记本
├── scripts/                      # setup_env.sh / activate_env.sh（可选 uv）
├── data/                         # 本地数据目录（通常 gitignore）
│   └── cleaned/                  # business_dining.csv, review_dining.csv 等
├── requirements.txt
└── README.md
```

说明：`README.md` 中若仍出现已删除路径（如根目录 `components.py`），以**实际仓库文件**为准。

---

## 3. 系统架构（当前实现）

### 3.1 端到端数据流

```text
用户（Streamlit）
  → Step 1：必选州（state）+ 可选市（city）→ 「发现」通用列表
  → Step 2：自然语言 + 菜系多选 + 关键词 → 「按偏好更新」
  → query_parser.parse_query：菜系/预算/地标坐标/半径/语义子串等
  → TouristRetrieval.recommend_keywords：
        · 店铺侧文档 = profile_builder 生成的文本（元数据 + 正/负向评论主题）
        · 查询向量化：sklearn TfidfVectorizer（与索引共用）
        · 相似度：稀疏向量余弦相似度（非 FAISS）
        · 过滤：州/市、菜系（categories 子串规则，可放宽）、地标+半径（Haversine）
        · 多因子打分 final_score（侧边栏可调权重）
  → 取 Top-N 作为候选池；展示 Top-K
  → 若用户 👍/👎：reranker 在池内用 TF-IDF 与 liked/disliked 相似度计算 v2_score
  → 详情弹窗：insights + Folium 地图 + 可选 Yelp 图片
```

### 3.2 与旧版设计文档的差异（重要）

| 主题 | 原规划/旧稿 | **当前实现** |
|------|-------------|----------------|
| 主检索 | 句向量 + FAISS | **TF-IDF + 余弦**，`scipy.sparse` + 手工归一化点积 |
| 稀疏关键词 | 可选 BM25 | **未使用 rank-bm25**；检索即 TF-IDF |
| 句向量 | 主检索 | **`sentence-transformers`（如 all-MiniLM-L6-v2）仅用于可选「从 NL 猜州」**，不参与主检索 |
| 洞察 | KeyBERT 等 | **`app/search/insights.py` 规则与启发式**，无 KeyBERT/spaCy 依赖 |
| 地图 | 可选 | **已实现**：`folium` + `streamlit-folium` |

---

## 4. 模块说明

### 4.1 查询解析（Query Parser）

**目标**：将自然语言转为结构化约束（与 `ParsedQuery` 对齐）。

**实现**：`app/search/query_parser.py` — 规则与正则（菜系词、cheap/moderate/expensive、地标 → `ref_lat/ref_lon`、`radius_km`、`semantic_query` 等）。

**可选**：侧边栏开启时，通过 `app/search/semantic_filters.py` 用 MiniLM 在美国州名候选上做相似度提示（**不替代** Step 1 所选州）。

---

### 4.2 索引与候选检索（Candidate Retrieval）

**目标**：从全库子集中得到与查询文本相关的候选店铺。

**文档构建**（`TouristRetrieval._build_documents`）：

- 按 `review_count` 等选取店铺子集；从 `review_dining.csv` 流式读取，按星级拆分正/负向评论槽位；
- 通过 `profile_builder.build_profile_text` 把元数据与主题短语压成**单一文本文档**；
- 支持 `business_id` 以 `gm_` 开头的 **Google Maps 补充行**（无 Yelp 评论时用合成 snippet）。

**索引**：

- `sklearn.feature_extraction.text.TfidfVectorizer`（`max_features` 默认 50000，`ngram_range=(1,2)`，`stop_words='english'`）；
- 持久化：`models/artifacts/` 下 `vectorizer.joblib`、`restaurant_matrix.npz`、`restaurant_ids.npy`、`meta.csv`、`index_config.json`；
- `index_version` 等配置变更会触发自动重建（除非用户强制跳过逻辑由 UI「Force Rebuild」覆盖）。

**检索**：对查询句 `transform` 后与各行计算余弦相似度，得到 `similarity`。

---

### 4.3 过滤（Filtering）

- **州/市**：`state_norm` / `city_norm` 严格匹配（与 UI Step 1 一致）。
- **菜系**：`categories_norm` 上的关键词规则；若交集为空则**放宽菜系**但保留州/市（避免无结果）。
- **距离**：提供 `ref_lat/ref_lon` 时计算 Haversine；若给出 `max_radius_km` 则硬过滤。

---

### 4.4 重排序（Re-ranking）

**目标**：在相似度基础上融合评分、价格、距离、热度。

**特征与归一化**（`recommend_keywords` 内）：

- `sim_n`：池内 min-max 归一化的 TF-IDF 余弦相似度；
- `stars_rank`：星级归一化后再做 **review_count 收缩**（低评论数向中性拉；参考 `rating_trust_ref_reviews`，默认约 150 条为较充分信任）；
- `price_match`：Yelp 价格档与预算档的软匹配（无档则为 0.5）；
- `dist_score`：有参考点时距离越近越高；无参考点 0.5；
- `pop_n`：`log1p(review_count)` 的 min-max。

**打分**：

\[
\text{final\_score} = w_{sem}\cdot sim_n + w_{rat}\cdot stars\_rank + w_{price}\cdot price\_match + w_{dist}\cdot dist\_score + w_{pop}\cdot pop_n
\]

权重在 `app/main.py` 侧边栏可调。

---

### 4.5 交互式 v2 池内重排

**目标**：在同一候选池内，根据 👍/👎 无需重新检索即可调整顺序。

**实现**：`app/recommendation/reranker.py` — 对 liked/disliked 店铺的 TF-IDF 行向量与当前行做余弦相似度，与 `final_score` 归一化项及结构化偏好启发式组合为 `v2_score`。

---

### 4.6 洞察生成（Insight Generation）

**目标**：每条结果生成简短 **why / pros / cons**。

**实现**：`app/search/insights.py` — 基于星级、评论数、类目、预算与价格档、距离、相似度的**规则字符串**，无外部 NLP 库推理。

---

### 4.7 Streamlit UI

**主应用** `app/main.py`：

- 两步向导、侧边栏权重与候选池大小、索引重建按钮；
- 结果卡片（封面图来自 Yelp Photos 若配置）、详情 Dialog（Folium 地图、照片网格）；
- Session 状态保存解析结果与候选 `DataFrame`。

**子页面**：`app/pages/1_KMeans_聚类.py` — 商家聚类地图/导出等。

---

## 5. 数据与流水线

**运行期必需（推荐流程）**：

- `data/cleaned/business_dining.csv`
- `data/cleaned/review_dining.csv`

**可选**：

- Google Maps 清洗表（文件名与合并逻辑见 `google_maps_loader` / `retrieval` 中 `embed_google_maps`）；
- `data/Yelp Photos/` 下 `photos.json` 与图片 tar 或解压目录（用于详情页照片）。

**离线**：`pipelines/` 负责从原始 Yelp 到特征 CSV；`notebooks/` 用于探索。具体列名以清洗脚本输出为准。

---

## 6. 技术栈（与 `requirements.txt` 一致）

| 层级 | 依赖 |
|------|------|
| 语言与数值 | Python 3；`pandas`、`numpy`（`<2`）、`scipy` |
| 机器学习 / 检索 | `scikit-learn`（TF-IDF、余弦相关计算）；`joblib` 持久化 |
| 可选句向量 | `sentence-transformers`（仅语义辅助功能，非主检索） |
| 可视化 / 应用 | `streamlit`、`matplotlib`、`seaborn`、`folium`、`streamlit-folium` |
| 其他建模（仓库内脚本/笔记本） | `xgboost` 等（用于 `models/`、`pipelines/` 中扩展实验，非主应用硬依赖路径） |

**环境**：推荐 `source scripts/setup_env.sh`（可选 `uv`）或 `pip install -r requirements.txt`。

**启动**：`streamlit run app/main.py`；首次会构建 `models/artifacts/` 索引，耗时取决于数据量。

---

## 7. 评测与后续方向

**定量**：可对 `similarity`、`final_score`、v2 重排前后 Top-K 做离线对比；Precision@K 需标注查询相关性。

**定性**：可解释文案、地图与照片对决策路径的帮助。

**后续改进（与代码规划一致者可落地）**：

- 主检索升级为稠密向量 + ANN（FAISS 等）并与现有 TF-IDF 混合；
- 学习排序（LTR）或显式用户画像嵌入；
- LLM 查询解析或评论摘要（当前刻意未引入）；
- 与 README 中 `merchant_predictor`、`rl_feedback_loop` 等模块做产品级联通（当前主应用以推荐 MVP 为主）。

---

## 8. 小结

当前系统是一条**可运行的稀疏检索 + 多因子排序 + 规则化解释 + 可选交互重排**链路：以 Yelp（及可选 Google Maps）店铺与评论为数据基础，用 `profile_builder` 将评论转为可检索文本，用 **TF-IDF** 完成主匹配，并在 Streamlit 中完成过滤、打分与展示。若需对外说明「与最初设计文档的差异」，以本节与 §3.2 对照表为准。
