#!/usr/bin/env bash
# Start FastAPI with the repo .venv (avoids conda base vs NumPy 2 conflicts)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/backend:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
UV="${ROOT}/.venv/bin/uvicorn"
if [[ ! -x "$UV" ]]; then
  echo "Not found: $UV — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
exec "$UV" api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" "$@"
