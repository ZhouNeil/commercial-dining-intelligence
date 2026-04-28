#!/usr/bin/env python3
"""
Train-spatial helpers (same city keys as /api/v1/merchant/cities).

Usage (repo root):
  PYTHONPATH=backend:. python scripts/spatial_train_diagnostics.py rows [--city NAME] [--state ST]
  PYTHONPATH=backend:. python scripts/spatial_train_diagnostics.py threshold [--csv PATH]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO / "backend"))

import numpy as np
import pandas as pd

from services.merchant_inference import (
    SPATIAL_CITY_MIN_TRAIN_ROWS,
    _city_group_key_spatial,
    spatial_train_csv_path,
)


def _cmd_rows(args: argparse.Namespace) -> None:
    root = Path(_REPO).resolve()
    try:
        path = spatial_train_csv_path(root)
    except OSError as e:
        print(e, file=sys.stderr)
        print("Unzip train_spatial.csv from data.zip into data/ or the repo root.", file=sys.stderr)
        sys.exit(1)

    want_key = _city_group_key_spatial(args.city)
    st = str(args.state).strip().upper()

    hdr = pd.read_csv(path, nrows=0).columns.tolist()
    cols = ["city", "latitude", "longitude"]
    if "state" in hdr:
        cols.append("state")
    n = 0
    raw_labels: set[str] = set()
    for chunk in pd.read_csv(path, usecols=cols, chunksize=200_000, low_memory=False):
        if "state" not in chunk.columns:
            chunk["state"] = ""
        c = chunk["city"].astype(str).str.strip()
        c = c.str.replace(r"\s+", " ", regex=True)
        s = chunk["state"].fillna("").astype(str).str.strip().str.upper()
        m = c.map(_city_group_key_spatial).eq(want_key) & s.eq(st)
        n += int(m.sum())
        if m.any():
            raw_labels.update(c[m].unique().tolist()[:20])

    print(f"File: {path.resolve()}")
    print(f"Rows matching normalized city key {want_key!r} + state {st}: {n}")
    if raw_labels:
        print("Raw city field samples:", sorted(raw_labels)[:8])
    print(f"Project floor SPATIAL_CITY_MIN_TRAIN_ROWS = {SPATIAL_CITY_MIN_TRAIN_ROWS}")
    if n < SPATIAL_CITY_MIN_TRAIN_ROWS:
        print("(Below the floor, this group should not appear in the default /merchant/cities list.)")


def _counts_per_city_state(csv: Path) -> np.ndarray:
    df = pd.read_csv(csv, usecols=["city", "state", "latitude", "longitude"], low_memory=False)
    if "state" not in df.columns:
        df["state"] = ""
    df["city"] = df["city"].astype(str).str.strip()
    df["city"] = df["city"].str.replace(r"\s+", " ", regex=True)
    df["state"] = df["state"].fillna("").astype(str).str.strip().str.upper()
    df = df[df["city"].str.len() > 0]
    df["_k"] = df["city"].map(_city_group_key_spatial) + "|" + df["state"]
    return df.groupby("_k", sort=False).size().to_numpy(dtype=int)


def _cmd_threshold(args: argparse.Namespace) -> None:
    os.chdir(_REPO)
    p = args.csv
    if p is None:
        p = spatial_train_csv_path(_REPO)
    if not p.is_file():
        print(f"Not found: {p} (place train_spatial.csv under data/ and retry)", file=sys.stderr)
        sys.exit(1)
    n = _counts_per_city_state(p)
    n_sorted = np.sort(n)
    m = int(len(n_sorted))

    print(f"File: {p.resolve()}")
    print(f"Normalized (city, state) groups: {m}")
    if m == 0:
        return
    print("Rows per city — quantiles: min, 5%, 25%, 50%, 75%, 90%, 95%, max")
    qs = [0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
    print("  ", [int(np.quantile(n_sorted, q, method="linear")) for q in qs])
    print()
    print("Cities kept for each min_rows (matches cities API behavior):")
    for k in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100):
        kept = int(np.sum(n >= k))
        print(f"  min_rows>={k:3d}  ->  {kept:6d}  cities  ({kept / m * 100:.1f}%)")
    print()
    cur = int(SPATIAL_CITY_MIN_TRAIN_ROWS)
    kept_cur = int(np.sum(n >= cur))
    print(
        f"Code SPATIAL_CITY_MIN_TRAIN_ROWS = {cur} -> keeps {kept_cur} cities ({kept_cur / m * 100:.1f}%)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="train_spatial.csv diagnostics (merchant cities API keys).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rows = sub.add_parser("rows", help="Count rows for one (city, state) after normalization")
    p_rows.add_argument("--city", default="Abington", help="Display name; mixed case is OK")
    p_rows.add_argument("--state", default="PA")
    p_rows.set_defaults(func=_cmd_rows)

    p_thr = sub.add_parser("threshold", help="Distribution of rows/city to tune SPATIAL_CITY_MIN_TRAIN_ROWS")
    p_thr.add_argument("--csv", type=Path, help="Defaults to spatial_train_csv_path(repo root)")
    p_thr.set_defaults(func=_cmd_threshold)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
