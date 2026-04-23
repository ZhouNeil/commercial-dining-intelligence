#!/usr/bin/env bash
# 从当前代码导出 OpenAPI JSON → frontend/openapi.json，再执行 frontend: npm run gen:api
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/backend:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "${ROOT}/.venv/bin/python" -c "
import json
from pathlib import Path
from api.main import app
out = Path('${ROOT}') / 'frontend' / 'openapi.json'
out.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('Wrote', out)
"
