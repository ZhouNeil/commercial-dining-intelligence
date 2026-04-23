# Vue 前端（重构 P4 骨架）

与仓库根 `README.md` 及 `docs/refactor-plan-data-vue-api.md`（迁移规划）一致：通过 Vite 将 `/api` 代理到 FastAPI（默认 `http://127.0.0.1:8000`）。

## 命令

在 **`frontend/`** 目录下：

```bash
cd frontend
npm install
npm run dev
```

或在**仓库根目录**（根目录已有转发用 `package.json`）：

```bash
npm run install:frontend   # 首次
npm run dev
```

环境变量：

- `VITE_API_PROXY_TARGET`：覆盖代理目标（默认 `http://127.0.0.1:8000`）。
- `VITE_API_BASE_URL`：若前端与 API **不同源**（无代理），设为完整 API 根 URL（如 `https://api.example.com`）；开发时留空即可走 Vite 代理。

**页面**：`/` 健康检查、`/search` 餐厅检索、`/merchant` 商家选址预测。

**OpenAPI 类型**：仓库根执行 `./scripts/export_openapi.sh` 后，在 `frontend/` 下执行 `npm run gen:api` 更新 `src/api/generated.d.ts`。

生产构建静态资源后，由 Nginx 等将 `/api` 反代到后端服务即可（见仓库 `deploy/nginx-frontend.example.conf`）。
