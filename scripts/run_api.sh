#!/usr/bin/env bash
# 使用仓库 .venv 启动 FastAPI（避免 conda base 与 NumPy 2 冲突）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/backend:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
UV="${ROOT}/.venv/bin/uvicorn"
if [[ ! -x "$UV" ]]; then
  echo "未找到 $UV；请先 python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
exec "$UV" api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" "$@"
