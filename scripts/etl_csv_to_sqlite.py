#!/usr/bin/env python3
"""
P3：将 `data/cleaned/business_dining.csv` 导入 SQLite（商户主档最小表）。
不导入空间宽表（仍用 CSV/Parquet）。

用法（仓库根目录）：
  python scripts/etl_csv_to_sqlite.py
  python scripts/etl_csv_to_sqlite.py --out data/merchants.sqlite3
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="SQLite 输出路径（默认 data/merchants.sqlite3）",
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    src = repo / "data" / "cleaned" / "business_dining.csv"
    if not src.is_file():
        raise SystemExit(f"缺少源文件: {src}")

    out = (args.out or (repo / "data" / "merchants.sqlite3")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    df = pd.read_csv(src, low_memory=False)
    cols = {
        "business_id": "TEXT PRIMARY KEY",
        "name": "TEXT",
        "latitude": "REAL",
        "longitude": "REAL",
        "city": "TEXT",
        "state": "TEXT",
        "stars": "REAL",
        "review_count": "INTEGER",
        "is_open": "INTEGER",
        "categories": "TEXT",
    }
    use = [c for c in cols if c in df.columns]
    sub = df[use].copy()
    conn = sqlite3.connect(out)
    try:
        sub.to_sql("merchants", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_merchants_city ON merchants(city);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_merchants_state ON merchants(state);"
        )
        conn.commit()
    finally:
        conn.close()
    print(f"Wrote {out.relative_to(repo)} rows={len(sub)}")


if __name__ == "__main__":
    main()
