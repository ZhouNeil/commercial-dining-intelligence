# 全栈重构规划：数据层、存储、Vue 前端与模型 API

本文档面向「CSV 双轨数据 → 可演进的数据与接口层 → Vue 前端」的整体重构，供团队评审与拆任务使用。范围包括：**数据建模与是否引入 SQLite/DB**、**后端 API 对接现有模型**、**前端用 Vue 重写**，以及**分阶段任务拆分**。

> **`docs/` 已精简**：除本文件外，历史说明、契约长文、对照表等均已移除。**运行命令、目录树**以仓库根目录 [`README.md`](../README.md) 为准；**字段级契约**以 `frontend/openapi.json`（及 `backend/api/schemas.py`）为准。后续可另增一份简短的「项目概述」Markdown。

### 实施进度（仓库内已落地）

| 阶段 | 状态 | 说明 |
|------|------|------|
| P0 | 已做 | `data/manifests/schema.sample.json`、`scripts/write_data_manifest.py`（数据契约细节待新「项目概述」或本文件迭代补全） |
| P1 | 已做 | `backend/services/merchant_inference.py`、`backend/services/retrieval_service.py`；`tests/test_inference.py` 走服务层 |
| P2 | 已做 | `backend/api/`（FastAPI）、`scripts/run_api.sh`、`Dockerfile.api`、`requirements-dev.txt` + pytest |
| P3 | 已做 | `scripts/etl_csv_to_sqlite.py`（导入 `business_dining.csv` → SQLite） |
| P4 | 已做（持续补全 UI） | `frontend/`：Vue Router、`/search` 与 `/merchant`、OpenAPI + `gen:api`、根 `package.json` 转发、`deploy/nginx-frontend.example.conf`；地图/详情等待办 |
| P5 | 未做 | Parquet、模型 registry、限流等 |

---

## 1. 现状速览（为何要重构）

### 1.1 两条相对独立的数据/产品轨

| 轨 | 典型用途 | 数据形态（当前） | 主要消费者 |
|----|-----------|------------------|------------|
| **A. 检索 / 推荐** | 自然语言 + 筛选、TF-IDF 检索、地图展示 | `data/cleaned/`（如 `business_dining.csv`）、`data/slice_representative/` 备选；索引与向量在 `models/artifacts/` | `backend/` 下 `dining_retrieval`、`services`、`api`；`frontend`（Vue） |
| **B. 空间 / 商家预测** | 选址、空间特征、生存概率与星级回归 | `data/train_spatial.csv`（及划分出的 `train_merchant_split` / `test_spatial`）；模型 `*.pkl` 在 `models/artifacts/` | `merchant_predictor`、`SpatialFeatureEngineer`、`tests/test_inference`、`frontend` `/merchant` |

两条轨**共享「餐厅」语义**，但**列结构、粒度、更新方式不同**：A 偏「展示与检索」，B 偏「宽表 + 已算好的空间特征」。CSV 在本地协作时易出现：**路径散落、重复拷贝、难以版本化「哪一份是线上真相」、大文件 Git 不友好**。

### 1.2 当前痛点（与重构目标对应）

- **双套 CSV**：同一业务概念（商户）在 A/B 中字段不一致，缺少统一实体 ID 与血缘说明。
- **职责边界不清**：「清洗结果」「特征表」「训练集」混在同一目录层级，新人难判断从哪读。
- **（已解决）** 曾将前端与模型耦合在旧版 Streamlit MVP；现已拆为 Vue + OpenAPI + `backend/services`。
- **推理与训练路径硬编码**：多处 `Path` / 相对路径，不利于部署到固定目录或容器。

重构目标：**单一事实来源（或明确分层的多源）+ 可版本化的数据管线 + 与前端解耦的 HTTP API + Vue SPA**。

---

## 2. 数据层重构思路

### 2.1 原则

1. **先定「领域模型」再定存储**：商户、评论、空间特征快照、检索索引元数据、模型制品（artifact）各是什么生命周期。
2. **区分「在线服务读的数据」与「离线批处理产物」**：在线尽量小、可索引；离线宽表可仍在列式/文件或仓库外对象存储。
3. **统一主键**：全项目对 Yelp `business_id`（或自研 `merchant_uuid`）保持一致，A/B 轨通过该键关联，而不是靠 name+lat 模糊对齐。

### 2.2 建议的目录与逻辑分层（即使短期仍落盘 CSV）

在引入 DB 之前也可先做「逻辑重构」，减少双轨混乱：

```
data/
  raw/              # 原始 Yelp 等（可不入库，或仅存元数据）
  curated/          # 清洗后「业务可读」窄表：商户主档、评论摘要等（对应现 cleaned / slice）
  features/         # 机器学习用宽表：train_spatial、中间 parquet 等
  manifests/        # 数据版本清单：文件名、sha256、生成脚本 commit、行数
models/
  artifacts/        # 索引、vectorizer、pkl（继续 gitignore，用 manifest 描述版本）
```

**manifest（JSON/YAML）**：每次跑 pipeline 写一条记录，API 与训练脚本只依赖「当前激活版本指针」，而不是写死文件名。

### 2.3 「两套数据」如何收敛（策略选项）

- **选项 1（推荐，渐进）**：**商户主档一份**（curated），空间特征由 pipeline **从主档 + 规则 JOIN** 生成 `features/train_spatial`，B 轨只读 features；A 轨读 curated + artifacts。两边通过 `business_id` 对齐。
- **选项 2**：保留双表但**强制文档 + 契约**：在 README / OpenAPI / 独立概述文档中维护字段表与血缘，API 层只暴露 DTO，不暴露原始 CSV 路径给前端。
- **选项 3（长期）**：主档进 DB，特征表仍以 Parquet/SQLite 列存形式供 pandas/sklearn 批量读（见下节）。

---

## 3. 是否使用数据库？SQLite 是否够用？

### 3.1 各类数据的合适载体

| 数据类型 | CSV 继续用的代价 | SQLite | PostgreSQL 等 | 文件（Parquet/npz） |
|----------|------------------|--------|-----------------|---------------------|
| 商户主档、评论元数据、用户偏好（小） | 大表全量扫描、并发差 | **很适合** MVP：单文件、零运维、SQL 索引 | 多写者、高并发、地理扩展 | 可选：主档导出 parquet 做分析 |
| 宽表特征（千列） | 慢、内存爆炸 | 可存但**不如 Parquet** 对列存友好 | 同左 | **推荐**：特征训练/推理用 Parquet |
| TF-IDF 稀疏矩阵、大向量 | 不适合频繁 IO | 不适合存大 blob | 存 blob 或仍用文件系统 | **保持 npz/joblib** + manifest |
| 线上 OLTP（订单、会话） | 不适用 | 单机够用 | 团队规模上来再迁 | — |

### 3.2 结论建议

- **第一阶段**：引入 **SQLite**（或继续 CSV + 强 manifest）作为 **「商户/评论/配置」的单一查询源**；**空间宽表与矩阵仍用文件**（Parquet + 现有 joblib/npz），避免把千维特征硬塞进 SQL。
- **何时上 PostgreSQL**：需要多实例写、复杂权限、PostGIS 地理查询在库内完成、或托管云服务时。
- **不要指望 SQLite 解决所有问题**：它是 **结构化元数据与关系** 的利器，不是列存数仓；与 **现有 sklearn 管道** 最顺的仍是 **pandas + Parquet** 读宽表。

### 3.3 若采用 SQLite 的示意 schema（精简）

- `merchants`：`business_id` PK，`name`, `lat`, `lon`, `city`, `state`, `stars`, `review_count`, `is_open`, `categories_json`, …
- `reviews`：`review_id` PK，`business_id` FK，`text`, `stars`, `date`, …（可按需只存抽样）
- `dataset_builds`：`id`，`kind`（spatial_train / retrieval_index），`path`，`checksum`，`created_at`，`git_sha`
- `model_registry`：`name`（survival / rating），`path`，`feature_schema_hash`，`created_at`

空间特征表若列数极大，可 **仅存 `business_id` + 少量指标 + parquet 路径** 或整表仍放文件，由 API 服务内存映射/按需加载。

---

## 4. 目标架构：Vue 前端 + API + 现有模型

### 4.1 逻辑分层

```
Vue SPA (Vite)
    │  HTTPS / JSON
    ▼
API Gateway (可选)
    │
    ▼
Python API 服务 (推荐 FastAPI)
    ├── 检索服务：加载 TouristRetrieval / 索引，暴露 POST /search
    ├── 商家预测服务：加载 SpatialFeatureEngineer + joblib 模型，暴露 POST /merchant/site-score
    ├── 元数据：GET /health, GET /datasets/active
    └── （可选）读 SQLite 提供商户详情、分页列表
```

### 4.2 与「现有模型」的对接方式

- **检索轨**：启动时 `build_or_load_index`，请求内只做 query + 重排；响应 DTO 与 Vue 表格字段对齐，便于地图等组件复用。
- **预测轨**：将 `tests/test_inference.py` 等中的逻辑抽成 **纯函数服务层**（`backend/services/merchant_inference.py`），API 只做参数校验、调用、错误映射。
- **进程与内存**：大 CSV/索引可 **懒加载 + 单例**；多 worker 时注意 **每进程一份内存** 或改用共享只读 mmap（进阶）。

### 4.3 API 契约（示例，便于拆任务）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 版本、依赖数据是否存在 |
| POST | `/api/v1/search` | body: `{ "query", "filters", "top_k" }` |
| POST | `/api/v1/merchant/predict` | body: `{ "city", "lat", "lon", "category_keys": [] }` |
| GET | `/api/v1/merchants/{business_id}` | 详情（数据来自 SQLite 或 curated） |

统一：`application/json`，错误体 `{ "code", "message", "detail" }`。

### 4.4 Vue 侧建议

- **Vite + Vue 3 + TypeScript**；状态可用 Pinia；UI 库任选（Element Plus / Naive UI）。
- **地图**：MapLibre / Leaflet（与现 Folium 解耦）；坐标与后端一致 WGS84。
- **与旧 MVP 关系**：Streamlit 已下线；以 **OpenAPI 单源契约** 与 Vue 为准。

---

## 5. 分阶段路线图（推荐顺序）

| 阶段 | 内容 | 产出 |
|------|------|------|
| **P0** | 数据清单 + manifest；统一 `business_id`；文档化两套 CSV 血缘 | `data/manifests/*` + 本文件 / 未来「项目概述」 |
| **P1** | 抽出「推理服务模块」无 Streamlit 依赖；单元测试覆盖 predict | `backend/services/*` + pytest |
| **P2** | FastAPI 最小实现：`/health` + `/merchant/predict` + `/search` | 可 Docker 运行 |
| **P3** | SQLite 导入 curated 商户（脚本 ETL）；API 读库返回详情 | `scripts/etl_to_sqlite.py` |
| **P4** | Vue 工程初始化；对接 OpenAPI；替换主流程页面 | 部署静态 + 反向代理 API |
| **P5** | 特征表 Parquet 化、模型 registry；观测与限流 | 运维就绪 |

---

## 6. 任务拆分（Epic → 可执行项）

### Epic A — 数据治理与存储决策

- [x] **A1** 盘点仓库内所有 CSV 路径与消费者（脚本、notebooks、`backend/` 等），输出表格（路径、用途、更新方）。
- [x] **A2** 数据契约：主键与 A/B 轨规则曾独立成文，现已下线；核心约定保留在本文件 §1–2、§4 与 OpenAPI；完整字段表待新「项目概述」。
- [x] **A3** 实现 `manifest` 生成步骤（嵌入 `pipelines` 或 `scripts`），CI 可选校验「指针存在」。
- [ ] **A4** 评审会：确认 **SQLite 范围**（仅主档 vs 含评论）与 **Parquet 范围**（train_spatial）。
- [x] **A5**（可选）实现 `etl_csv_to_sqlite.py` + 最小 schema + 迁移说明。

### Epic B — 模型与推理服务化

- [x] **B1** 抽取 `merchant_inference.predict(...)`（输入 lat/lon/city/categories，输出概率与星级 + 部分特征）。
- [x] **B2** 抽取 `retrieval_search.search(...)` 或与 `TouristRetrieval` 薄封装。
- [x] **B3** 定义 Pydantic 请求/响应模型与 OpenAPI；错误码规范。
- [x] **B4** FastAPI 路由实现 + 启动文档（含用 `.venv` 避免 NumPy 冲突）。
- [x] **B5** 容器化 `Dockerfile`（API）+ 数据卷挂载约定。

### Epic C — Vue 前端

- [x] **C1** 初始化 Vue3+TS+Vite 仓库（monorepo 子目录 `frontend/` 或独立 repo 决策）。
- [x] **C2** 生成 OpenAPI 类型：`frontend/openapi.json` + `npm run gen:api` → `src/api/generated.d.ts`，`src/api/client.ts` 封装 fetch。
- [x] **C3** 页面：`/` 健康、`/search` 检索表格、`/merchant` 选址表单与结果卡（地图选点留作后续）。
- [x] **C4** 环境变量：`VITE_API_BASE_URL`；本地 dev proxy。
- [x] **C5** 构建与部署：`npm run build` → `frontend/dist`；示例 `deploy/nginx-frontend.example.conf`。

### Epic D — 下线与迁移

- [x] **D1** 前端路由与能力由 `frontend/` 与 OpenAPI 体现（独立对照表已下线）。
- [ ] **D2** 删除或归档死代码路径；维护根 `README` 与（计划中的）`docs/PROJECT_OVERVIEW.md`（或同类「项目概述」）。
- [ ] **D3**（可选）E2E：Playwright 对关键 API + 页面。

---

## 7. 风险与缓解

- **内存占用**：索引 + 空间参考表同时常驻 → 评估峰值，必要时 **分服务部署**（检索 API 与预测 API 分离）。
- **数据漂移**：训练特征与线上一致性 → **feature_schema_hash** 写入 manifest，加载时校验。
- **团队并行**：先锁 **OpenAPI 契约**，前后端可并行；契约变更走版本号 `/v1` → `/v2`。

---

## 8. 文档维护

- 本文档路径：`docs/refactor-plan-data-vue-api.md`（**`docs/` 下唯一长期保留的规划/迁移说明**）。
- **实际表结构、API 基址、环境变量**：维护在根目录 `README.md`；字段级请求/响应以 OpenAPI 为准。若需要对外/答辩用的一页纸说明，可新增 `docs/PROJECT_OVERVIEW.md`（名称自定），不必再拆多份旧式长文。

---

*文档版本：初稿，供评审与拆 sprint；可根据团队规模删减 P3/P5。*
