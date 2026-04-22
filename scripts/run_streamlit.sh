#!/usr/bin/env bash
# 使用仓库内 .venv，避免系统/conda base 下 NumPy 2.x 与 pandas 等二进制不兼容。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv/bin/streamlit"
if [[ ! -x "$VENV" ]]; then
  echo "未找到可执行文件: $VENV" >&2
  echo "请在仓库根目录执行: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
exec "$VENV" run "$ROOT/app/main.py" "$@"
