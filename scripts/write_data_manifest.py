#!/usr/bin/env python3
"""
写入 data/manifests/active.json：登记关键数据/模型文件的相对路径、大小、mtime；
小于 20MB 的文件额外计算 sha256（大 CSV 跳过哈希以节省时间）。

用法（仓库根目录）：
  python scripts/write_data_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

MAX_HASH_BYTES = 20 * 1024 * 1024

TRACKED = [
    ("spatial_train_csv", "data/train_spatial.csv"),
    ("spatial_train_csv", "train_spatial.csv"),
    ("retrieval_business_csv", "data/cleaned/business_dining.csv"),
    ("retrieval_review_csv", "data/cleaned/review_dining.csv"),
    ("model_survival_pkl", "models/artifacts/advanced_survival_classifier.pkl"),
    ("model_rating_pkl", "models/artifacts/global_rating_model.pkl"),
    ("spatial_split_train", "data/train_merchant_split.csv"),
    ("spatial_split_test", "data/test_spatial.csv"),
]


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _hash_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    if p.stat().st_size > MAX_HASH_BYTES:
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out_path = repo / "data" / "manifests" / "active.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    artifacts: list[dict] = []
    for kind, rel in TRACKED:
        if rel in seen:
            continue
        p = repo / rel
        if not p.is_file():
            continue
        seen.add(rel)
        st = p.stat()
        entry = {
            "kind": kind,
            "relative_path": rel.replace("\\", "/"),
            "bytes": st.st_size,
            "mtime_epoch": st.st_mtime,
        }
        h = _hash_file(p)
        if h:
            entry["sha256"] = h
        artifacts.append(entry)

    doc = {
        "version": "1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(repo),
        "artifacts": artifacts,
    }
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(repo)} ({len(artifacts)} artifacts)")


if __name__ == "__main__":
    main()
