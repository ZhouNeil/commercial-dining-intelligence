#!/usr/bin/env python3
"""
从 cleaned 餐饮数据中按 (state, stars 档位, 营业状态, review_count 四分位) 分层抽样商户，
并截取这些商户对应的 review / checkin / tip。可在无 pandas 环境下运行。
"""
from __future__ import annotations

import argparse
import csv
import sys
import math
import random
from collections import defaultdict
from pathlib import Path


def _relax_csv_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def load_business_rows(business_path: Path) -> list[dict[str, str]]:
    _relax_csv_limit()
    with business_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def review_count_quartile_bins(values: list[int]) -> list[float]:
    if not values:
        return [0.0, 1.0, 2.0, 3.0]
    s = sorted(values)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return float(s[0])
        x = (n - 1) * p
        lo = int(math.floor(x))
        hi = int(math.ceil(x))
        if lo == hi:
            return float(s[lo])
        return float(s[lo] + (s[hi] - s[lo]) * (x - lo))

    return [q(0.25), q(0.50), q(0.75)]


def rc_bucket(v: int, cuts: list[float]) -> int:
    if v <= cuts[0]:
        return 0
    if v <= cuts[1]:
        return 1
    if v <= cuts[2]:
        return 2
    return 3


def star_half_bucket(x: str) -> str:
    try:
        s = float(x)
    except (TypeError, ValueError):
        return "na"
    # 与 Yelp 半星一致
    h = round(s * 2) / 2
    return f"{h:.1f}"


def state_bucket(state: str, counts: dict[str, int], min_per_state: int) -> str:
    st = (state or "").strip() or "UNK"
    if counts.get(st, 0) < min_per_state:
        return "OTHER"
    return st


def stratified_business_ids(
    rows: list[dict[str, str]],
    target_n: int,
    min_per_state: int,
    seed: int,
) -> tuple[set[str], list[dict[str, str]]]:
    rcs: list[int] = []
    for row in rows:
        try:
            rcs.append(int(row.get("review_count") or 0))
        except ValueError:
            rcs.append(0)
    cuts = review_count_quartile_bins(rcs)

    state_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        st = (row.get("state") or "").strip() or "UNK"
        state_counts[st] += 1

    strata: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row, rc in zip(rows, rcs):
        bid = row.get("business_id")
        if not bid:
            continue
        st = state_bucket(row.get("state") or "", dict(state_counts), min_per_state)
        sk = (
            st,
            star_half_bucket(row.get("stars") or ""),
            row.get("is_open") or "0",
            rc_bucket(rc, cuts),
        )
        strata[sk].append(row)

    n_total = sum(len(g) for g in strata.values())
    rng = random.Random(seed)
    keys = sorted(strata.keys())

    # 最大余数法在层间分配 target_n，单层不超过该层规模
    allocations: dict[tuple, int] = {}
    exact: dict[tuple, float] = {}
    floor_sum = 0
    for k in keys:
        L = len(strata[k])
        ex = target_n * L / n_total if n_total else 0.0
        exact[k] = ex
        b = min(L, int(math.floor(ex)))
        allocations[k] = b
        floor_sum += b

    rem = target_n - floor_sum
    frac_order = sorted(keys, key=lambda k: (exact[k] - allocations[k], len(strata[k])), reverse=True)
    while rem > 0:
        progressed = False
        for k in frac_order:
            if rem <= 0:
                break
            if allocations[k] < len(strata[k]):
                allocations[k] += 1
                rem -= 1
                progressed = True
        if not progressed:
            break

    if sum(allocations.values()) > target_n:
        over = sum(allocations.values()) - target_n
        while over > 0:
            k = max(
                (k for k in keys if allocations[k] > 0),
                key=lambda x: allocations[x],
                default=None,
            )
            if k is None or allocations[k] <= 0:
                break
            allocations[k] -= 1
            over -= 1

    chosen: set[str] = set()
    out_rows: list[dict[str, str]] = []
    for k in keys:
        g = strata[k]
        m = min(allocations[k], len(g))
        if m <= 0:
            continue
        pick = rng.sample(g, m)
        for r in pick:
            bid = r["business_id"]
            if bid not in chosen:
                chosen.add(bid)
                out_rows.append(r)

    if len(chosen) < target_n:
        rest = [r for r in rows if r.get("business_id") not in chosen]
        rng.shuffle(rest)
        for r in rest:
            if len(chosen) >= target_n:
                break
            bid = r.get("business_id")
            if bid and bid not in chosen:
                chosen.add(bid)
                out_rows.append(r)

    return chosen, out_rows


def filter_csv_by_business(
    src: Path,
    dst: Path,
    id_field: str,
    keep: set[str],
) -> int:
    _relax_csv_limit()
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(newline="", encoding="utf-8") as fin, dst.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        if id_field not in (reader.fieldnames or []):
            raise ValueError(f"{src}: 缺少列 {id_field}")
        w = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        w.writeheader()
        for row in reader:
            if row.get(id_field) in keep:
                w.writerow(row)
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="构建代表性小样本切片")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="项目根目录",
    )
    ap.add_argument("--target-businesses", type=int, default=2500, help="目标商户数量")
    ap.add_argument("--min-per-state", type=int, default=40, help="低于该数量的州并入 OTHER")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cleaned = args.root / "data" / "cleaned"
    out_dir = args.root / "data" / "slice_representative"
    biz_path = cleaned / "business_dining.csv"

    rows = load_business_rows(biz_path)
    keep_ids, biz_out_rows = stratified_business_ids(
        rows,
        target_n=args.target_businesses,
        min_per_state=args.min_per_state,
        seed=args.seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    biz_dst = out_dir / "business_dining.csv"
    with biz_dst.open("w", newline="", encoding="utf-8") as f:
        if biz_out_rows:
            w = csv.DictWriter(f, fieldnames=list(biz_out_rows[0].keys()))
            w.writeheader()
            w.writerows(biz_out_rows)

    n_rev = filter_csv_by_business(cleaned / "review_dining.csv", out_dir / "review_dining.csv", "business_id", keep_ids)
    n_chk = filter_csv_by_business(cleaned / "checkin_dining.csv", out_dir / "checkin_dining.csv", "business_id", keep_ids)
    n_tip = filter_csv_by_business(cleaned / "tip_dining.csv", out_dir / "tip_dining.csv", "business_id", keep_ids)

    meta = out_dir / "slice_meta.txt"
    with meta.open("w", encoding="utf-8") as f:
        f.write(f"target_businesses={args.target_businesses}\n")
        f.write(f"actual_businesses={len(keep_ids)}\n")
        f.write(f"reviews={n_rev}\n")
        f.write(f"checkins={n_chk}\n")
        f.write(f"tips={n_tip}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"min_per_state={args.min_per_state}\n")

    print(f"商户: {len(keep_ids)} -> {biz_dst}")
    print(f"评论: {n_rev}, 签到: {n_chk}, tips: {n_tip} -> {out_dir}")
    print(f"摘要: {meta}")


if __name__ == "__main__":
    main()
