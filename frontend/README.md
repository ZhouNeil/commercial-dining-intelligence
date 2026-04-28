# Vue frontend

Aligned with the root [`README.md`](../README.md) and [`docs/refactor-plan-data-vue-api.md`](../docs/refactor-plan-data-vue-api.md): Vite proxies `/api` to FastAPI (default `http://127.0.0.1:8000`).

## Commands

From **`frontend/`**:

```bash
npm install
npm run dev
npm run build
```

Or from the **repo root** (root `package.json` forwards commands):

```bash
npm run install:frontend   # first time
npm run dev
```

## Environment

- `VITE_API_PROXY_TARGET`: override the dev proxy target (default `http://127.0.0.1:8000`).
- `VITE_API_BASE_URL`: if the UI and API are **not** same-origin (no proxy), set the full API base URL (e.g. `https://api.example.com`); leave empty in dev to use the Vite proxy.

## Pages

- `/` health check
- `/search` restaurant search
- `/merchant` merchant site predictor

## OpenAPI types

From the repo root, run `./scripts/export_openapi.sh`, then in `frontend/` run `npm run gen:api` to refresh `src/api/generated.d.ts`.

For production, build static assets and reverse-proxy `/api` to the backend (see `deploy/nginx-frontend.example.conf` in the repo).
