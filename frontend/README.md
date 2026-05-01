# Vue Frontend

Proxies `/api` to FastAPI (default `http://127.0.0.1:8000`) via Vite. See the repo root `README.md` and `docs/refactor-plan-data-vue-api.md` for full context.

## Commands

From **`frontend/`**:

```bash
cd frontend
npm install
npm run dev
```

Or from the **repo root** (root `package.json` forwards the commands):

```bash
npm run install:frontend   # first time only
npm run dev
```

## Environment variables

- `VITE_API_PROXY_TARGET`: overrides the proxy target (default `http://127.0.0.1:8000`).
- `VITE_API_BASE_URL`: set to the full API root URL (e.g. `https://api.example.com`) if the frontend and API are on different origins (no proxy). Leave empty in development to use the Vite proxy.

## Pages

- `/` — health check
- `/search` — restaurant search
- `/merchant` — merchant site prediction

## OpenAPI types

Run `./scripts/export_openapi.sh` from the repo root, then `npm run gen:api` from `frontend/` to regenerate `src/api/generated.d.ts`.

## Production

Build static assets, then have Nginx (or similar) reverse-proxy `/api` to the backend service. See `deploy/nginx-frontend.example.conf`.
