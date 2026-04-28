#!/usr/bin/env bash
# Export OpenAPI JSON from code → frontend/openapi.json, then: npm run gen:api in frontend/
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
