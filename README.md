# Dual-Mode Commercial & Dining Intelligence App

This repository contains the codebase for our Machine Learning course final project. The application utilizes the Yelp Open Dataset to provide insights for both prospective merchants and tourists.

## 📂 Repository Structure

Our codebase strictly follows the Separation of Concerns (SoC) principle. The directory tree below outlines our modular architecture:

```text
commercial-dining-intelligence/
├── backend/                    # Python 后端（FastAPI + services + dining_retrieval）
│   ├── api/                    # FastAPI：health、states、search、merchant/predict
│   ├── services/               # 推理与检索编排（供 API 与脚本调用）
│   └── dining_retrieval/       # TF-IDF 索引、解析、重排、聚类辅助等
├── frontend/                   # Vue 3 + Vite（/search、/merchant）
├── data/                       # 数据集与模型（大文件 Git-ignored）
│   ├── cleaned/                # business_dining / review_dining 等
│   ├── manifests/              # schema.sample.json；active.json 由脚本生成
│   └── ...
├── pipelines/                  # 清洗、特征、空间特征工程
├── models/                     # 训练脚本、artifacts 路径约定
├── notebooks/                  # EDA 与实验
├── scripts/                    # setup、run_api、ETL、manifest、export_openapi
├── docs/                       # 仅存迁移/重构规划（见下）；项目概述待另补
├── pytest.ini                  # pytest 的 pythonpath（含 backend）
├── package.json                # 根目录仅转发 npm 脚本至 frontend/
├── .gitignore
├── README.md
└── requirements.txt
```

**`PYTHONPATH`**：`./scripts/run_api.sh`、`export_openapi.sh` 已设为 `backend` 与仓库根。若你自行 `python` / `uvicorn`，请使用 `PYTHONPATH=backend:.`（或只跑 pytest，已读 `pytest.ini`）。

**规划与迁移说明**（数据层 / API / Vue 路线）：[`docs/refactor-plan-data-vue-api.md`](docs/refactor-plan-data-vue-api.md)。更短的一页式「项目概述」计划后续新增（如 `docs/PROJECT_OVERVIEW.md`）。

## 📥 Data Setup

Due to the massive size of the Yelp Open Dataset, raw and cleaned data files are **not** tracked in this GitHub repository. To run this project, you must manually download the required CSV files.

**1. Download the Data:**
* Fetch the cleaned dataset (e.g., `output_philly.csv`) from our team's shared Google: [`https://drive.google.com/drive/folders/1iqaBfD71GEfOnLrj7LzczDSLWwzz8Awd?usp=sharing`](https://drive.google.com/drive/folders/1iqaBfD71GEfOnLrj7LzczDSLWwzz8Awd?usp=sharing)

**2. Place it in the Repo:**
* Move the downloaded files directly into the `data/processed_csv/` folder. Do not force-add data files to Git.

## 🚀 Setup Environment

We use `uv` for lightning-fast dependency management. To run this project locally, please follow these steps from the root directory of the project:

**1. Initial Setup (Run Once):**
This script will install `uv` (if necessary), create a virtual environment (`.venv`), and install all dependencies from `requirements.txt`.

```bash
source scripts/setup_env.sh
```

**2. Activate Environment (Run every time you code):**
Whenever you open a new terminal session to work on this project, activate the environment by running:

```bash
source scripts/activate_env.sh
```

If you do not want to use `uv`, you can also install dependencies with pip:

```bash
pip install -r requirements.txt
```

## 运行方式（Vue + API）

准备 `data/cleaned/business_dining.csv` 与 `review_dining.csv`。首次调用检索接口时会在 `models/artifacts/` 构建 TF‑IDF 索引（可能较久）。

### HTTP API

FastAPI：`GET /api/health`、`GET /api/v1/states`、`POST /api/v1/merchant/predict`、`POST /api/v1/search`。

```bash
pip install -r requirements.txt   # 含 fastapi / uvicorn
./scripts/run_api.sh              # 默认 http://0.0.0.0:8000
# 或: PYTHONPATH=backend:. .venv/bin/uvicorn api.main:app --reload --port 8000
```

- OpenAPI 文档：`http://localhost:8000/docs`
- 可选环境变量：`API_REPO_ROOT`（仓库根）、`CORS_ORIGINS`（逗号分隔，默认 `*`）
- 容器：`docker build -f Dockerfile.api -t cdi-api .`（需挂载 `data/`、`models/artifacts/`）

开发与契约测试依赖：`pip install -r requirements-dev.txt`

### Vue 前端

先启动 API，再在仓库根执行 `npm run install:frontend`（首次）与 `npm run dev`。详见下文「Vue 前端（P4）」。

### 数据 manifest（P0）

```bash
python scripts/write_data_manifest.py   # 写入 data/manifests/active.json（已 gitignore）
```

### SQLite 商户主档（P3）

```bash
python scripts/etl_csv_to_sqlite.py      # 需先有 data/cleaned/business_dining.csv
```

### Vue 前端（P4）

依赖与脚本定义在 **`frontend/package.json`**；仓库根目录另有 **`package.json`**，仅用于转发常用命令（避免在根目录误跑时报错）。

**方式一（推荐，在仓库根目录）：**

```bash
npm run install:frontend   # 首次
npm run dev                # 等价于在 frontend/ 里 npm run dev
```

**方式二：**

```bash
cd frontend && npm install && npm run dev
```

开发代理将 `/api` 转到 `http://127.0.0.1:8000`，需先启动 `./scripts/run_api.sh`。详见 `frontend/README.md`。

**更新 API 契约后（可选）：**

```bash
./scripts/export_openapi.sh   # 重写 frontend/openapi.json
cd frontend && npm run gen:api
```

**静态部署**：`cd frontend && npm run build`，产物在 `frontend/dist/`；Nginx 示例见 `deploy/nginx-frontend.example.conf`。路由与接口见 `frontend/src/router` 与 `frontend/openapi.json`。
