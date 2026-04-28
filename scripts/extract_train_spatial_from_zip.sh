#!/usr/bin/env bash
# Unzip train_spatial.csv from data/data.zip to data/train_spatial.csv
# Backend reads this for merchant cities, /merchant/predict slices, etc. (the UI does not read CSVs directly)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP="${ROOT}/data/data.zip"
OUT="${ROOT}/data/train_spatial.csv"
if [[ ! -f "$ZIP" ]]; then
  echo "Not found: $ZIP" >&2
  exit 1
fi
unzip -o -j "$ZIP" "train_spatial.csv" -d "${ROOT}/data"
echo "Wrote: $OUT ($(du -h "$OUT" | cut -f1) approx.)"
