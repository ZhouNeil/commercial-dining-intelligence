"""
多维评分预测：数据清理、列展开、基于评论关键词的弱监督 aspect 目标。
与 docs/multid.md 对齐；供 notebooks/multi/multid.ipynb 逐步验证使用。
"""
from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 文档 4.2：关键词组（可扩展）
# ---------------------------------------------------------------------------
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "food": [
        "taste",
        "fresh",
        "portion",
        "delicious",
        "flavor",
        "food",
        "meal",
        "dish",
        "cooked",
        "menu",
        "appetizer",
        "dessert",
        "sushi",
        "pizza",
        "burger",
    ],
    "service": [
        "service",
        "staff",
        "waiter",
        "waitress",
        "server",
        "host",
        "manager",
        "rude",
        "friendly",
        "slow",
        "fast",
        "attentive",
    ],
    "atmosphere": [
        "atmosphere",
        "ambiance",
        "ambience",
        "cozy",
        "noisy",
        "noise",
        "decor",
        "music",
        "lighting",
        "romantic",
        "loud",
        "quiet",
        "vibe",
    ],
    "value": [
        "price",
        "cheap",
        "expensive",
        "worth",
        "value",
        "money",
        "deal",
        "overpriced",
        "affordable",
        "bill",
    ],
}

POS_WORDS = frozenset(
    """
    great excellent amazing love loved best fantastic wonderful delicious perfect
    good nice tasty fresh friendly attentive cozy worth affordable
    """.split()
)
NEG_WORDS = frozenset(
    """
    bad terrible awful worst disgusting rude slow dirty overpriced bland cold
    gross disappointing poor horrible nasty inedible stale burnt soggy
    """.split()
)

# 尖锐差评词（词元匹配，突出服务/卫生/速度问题）
HARSH_WORDS: frozenset[str] = frozenset(
    """
    bad rude dirty slow terrible awful disgusting worst gross horrible nasty
    inedible stale burnt soggy cold bland overpriced rudest filthy
    """.split()
)

# 投诉 / 强负面信号（子串匹配，小写）
COMPLAINT_KEYWORDS: tuple[str, ...] = (
    "never again",
    "waste of money",
    "food poisoning",
    "hair in",
    "cockroach",
    "roach",
    "dispute",
    "refund",
    "walked out",
    "health code",
    "filthy",
    "unacceptable",
    "rip off",
    "scam",
    "complaint",
    "horrible service",
    "worst experience",
    "disgusting",
    "got sick",
    "rude",
    "dirty",
)


def relax_csv_field_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def _norm_yelp_repr(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\bu'", "'", s)
    s = re.sub(r'\bu"', '"', s)
    return s


def parse_dict_field(raw: Any) -> dict[str, Any]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return {}
    try:
        v = ast.literal_eval(_norm_yelp_repr(s))
        return v if isinstance(v, dict) else {}
    except (SyntaxError, ValueError, TypeError):
        return {}


def _bool_from_yelp_val(v: Any) -> float:
    """返回 1.0 True, 0.0 False, np.nan 未知"""
    if v is None or v == "None":
        return np.nan
    if isinstance(v, bool):
        return float(v)
    t = str(v).strip().lower()
    if t in ("true", "1", "yes"):
        return 1.0
    if t in ("false", "0", "no"):
        return 0.0
    return np.nan


def extract_price_range(attr_str: str) -> float:
    m = re.search(r"RestaurantsPriceRange2['\"]?\s*:\s*['\"]?(\d)", str(attr_str))
    if m:
        return float(m.group(1))
    return np.nan


def count_open_days(hours_raw: Any) -> float:
    d = parse_dict_field(hours_raw)
    if not d:
        return np.nan
    closed_markers = {"", "none", "0:0-0:0", "closed"}
    n = 0
    for _day, span in d.items():
        span_s = str(span).strip().lower()
        if span_s and span_s not in closed_markers and not span_s.startswith("0:0-0:0"):
            n += 1
    return float(n)


def flatten_attributes_column(series: pd.Series) -> pd.DataFrame:
    """展开常见 Yelp attributes 为数值/0-1 列。"""
    price = series.map(extract_price_range)
    good_kids: list[float] = []
    takeout: list[float] = []
    delivery: list[float] = []
    outdoor: list[float] = []
    credit: list[float] = []
    reservable: list[float] = []
    groups: list[float] = []
    wifi: list[float] = []  # 0 no, 1 free/paid, nan
    for raw in series.astype(str):
        d = parse_dict_field(raw)
        good_kids.append(_bool_from_yelp_val(d.get("GoodForKids")))
        takeout.append(_bool_from_yelp_val(d.get("RestaurantsTakeOut")))
        delivery.append(_bool_from_yelp_val(d.get("RestaurantsDelivery")))
        outdoor.append(_bool_from_yelp_val(d.get("OutdoorSeating")))
        credit.append(_bool_from_yelp_val(d.get("BusinessAcceptsCreditCards")))
        reservable.append(_bool_from_yelp_val(d.get("RestaurantsReservations")))
        groups.append(_bool_from_yelp_val(d.get("RestaurantsGoodForGroups")))
        w = d.get("WiFi")
        ws = str(w).lower()
        if "free" in ws:
            wifi.append(1.0)
        elif "no" in ws or "none" in ws:
            wifi.append(0.0)
        else:
            wifi.append(np.nan)

    return pd.DataFrame(
        {
            "attr_price_range": price,
            "attr_good_for_kids": good_kids,
            "attr_takeout": takeout,
            "attr_delivery": delivery,
            "attr_outdoor_seating": outdoor,
            "attr_credit_card": credit,
            "attr_reservations": reservable,
            "attr_good_for_groups": groups,
            "attr_wifi_indicator": wifi,
        }
    )


def expand_categories_column(series: pd.Series) -> pd.DataFrame:
    """categories 逗号分隔 -> 列表长度、首类、拼接小写串（供后续向量化）。"""
    n_cats: list[int] = []
    primary: list[str] = []
    joined: list[str] = []
    for raw in series.fillna(""):
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        n_cats.append(len(parts))
        primary.append(parts[0] if parts else "")
        joined.append(" ".join(p.lower() for p in parts))
    return pd.DataFrame(
        {
            "categories_count": n_cats,
            "category_primary": primary,
            "categories_text": joined,
        }
    )


def clean_and_expand_business(df: pd.DataFrame) -> pd.DataFrame:
    """基础清理 + 展开 attributes / categories / hours。"""
    out = df.copy()
    out = out.dropna(subset=["business_id", "stars"])
    out["business_id"] = out["business_id"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["business_id"], keep="first")

    num_cols = ["latitude", "longitude", "review_count"]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "is_open" in out.columns:
        out["is_open"] = pd.to_numeric(out["is_open"], errors="coerce").fillna(0).astype(int)
    else:
        out["is_open"] = 0

    if "stars" in out.columns:
        out["stars"] = pd.to_numeric(out["stars"], errors="coerce")
        out = out.dropna(subset=["stars"])

    attr_df = flatten_attributes_column(out["attributes"] if "attributes" in out.columns else pd.Series([""] * len(out)))
    cat_df = expand_categories_column(out["categories"] if "categories" in out.columns else pd.Series([""] * len(out)))
    hours_series = out["hours"] if "hours" in out.columns else pd.Series([""] * len(out))
    days_open = hours_series.map(count_open_days)

    expanded = pd.concat([out.reset_index(drop=True), attr_df, cat_df], axis=1)
    expanded["hours_days_open"] = days_open.values

    return expanded


def _text_hits_aspect(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def _complaint_hit(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in COMPLAINT_KEYWORDS)


def _neg_word_density(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    neg = sum(1 for w in words if w in NEG_WORDS)
    return float(neg / len(words))


def _harsh_word_density(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    h = sum(1 for w in words if w in HARSH_WORDS)
    return float(h / len(words))


def _harsh_review_any(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return any(w in HARSH_WORDS for w in words)


def star_interval_sample_weight(stars: Any) -> np.ndarray:
    """
    按半星档位给训练样本权重：1.0/1.5/5.0 极端档最高，2.0/2.5/4.5 中等，3.0–4.0 中部默认。
    stars 可为 Series 或 ndarray，元素为连续分亦可（先映射到最近半星再查表）。
    """
    s = np.asarray(stars, dtype=float)
    idx = np.clip(np.round((s - 1.0) * 2).astype(int), 0, 8)
    w_table = np.array(
        [3.2, 3.2, 1.9, 1.9, 1.0, 1.0, 1.0, 1.9, 3.2], dtype=float
    )  # 1.0 … 5.0 半星共 9 档
    return w_table[idx]


def _sentiment_score(text: str) -> float:
    """粗粒度 -1~1，用于 aspect 句块与评论信号。"""
    words = re.findall(r"[a-z]+", text.lower())
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg + 1.0)


def build_review_signal_features(
    reviews: pd.DataFrame,
    business_overall: pd.DataFrame,
) -> pd.DataFrame:
    """
    按店聚合评论级负面/投诉信号，缓解「只看结构化特征 → 回归均值」问题。
    列：rev_neg_word_density, rev_complaint_hit_rate, rev_low_review_star_share,
        rev_high_review_star_share, rev_mean_sentiment,
        rev_harsh_word_density, rev_harsh_review_share, rev_one_star_share,
        rev_max_neg_density, rev_neg_density_std
    """
    need = {"business_id", "text", "stars"}
    miss = need - set(reviews.columns)
    if miss:
        raise ValueError(f"reviews 缺少列: {miss}")

    all_ids = [str(x).strip() for x in business_overall["business_id"].tolist()]
    r = reviews.copy()
    r["business_id"] = r["business_id"].astype(str).str.strip()
    r["stars"] = pd.to_numeric(r["stars"], errors="coerce")
    r = r.dropna(subset=["business_id", "stars"])
    r["text"] = r["text"].fillna("").astype(str)

    rows: list[dict[str, Any]] = []
    for bid, grp in r.groupby("business_id"):
        bid = str(bid).strip()
        texts = grp["text"]
        stars = grp["stars"].astype(float)
        nd = texts.map(_neg_word_density)
        rows.append(
            {
                "business_id": bid,
                "rev_neg_word_density": float(nd.mean()),
                "rev_complaint_hit_rate": float(texts.map(_complaint_hit).mean()),
                "rev_low_review_star_share": float((stars <= 2).mean()),
                "rev_high_review_star_share": float((stars >= 4).mean()),
                "rev_mean_sentiment": float(texts.map(_sentiment_score).mean()),
                "rev_harsh_word_density": float(texts.map(_harsh_word_density).mean()),
                "rev_harsh_review_share": float(texts.map(_harsh_review_any).mean()),
                "rev_one_star_share": float((stars <= 1).mean()),
                "rev_max_neg_density": float(nd.max()) if len(nd) else 0.0,
                "rev_neg_density_std": float(nd.std()) if len(nd) > 1 else 0.0,
            }
        )

    _sig_cols = [
        "business_id",
        "rev_neg_word_density",
        "rev_complaint_hit_rate",
        "rev_low_review_star_share",
        "rev_high_review_star_share",
        "rev_mean_sentiment",
        "rev_harsh_word_density",
        "rev_harsh_review_share",
        "rev_one_star_share",
        "rev_max_neg_density",
        "rev_neg_density_std",
    ]
    have = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_sig_cols)
    base = pd.DataFrame({"business_id": all_ids})
    out = base.merge(have, on="business_id", how="left")
    return out


def sentiment_to_rating_1_5(s: float) -> float:
    """将 [-1,1] 线性映射到 [1,5]。"""
    return float(np.clip(1.0 + 2.0 * (s + 1.0), 1.0, 5.0))


def build_weak_aspect_targets(
    reviews: pd.DataFrame,
    business_overall: pd.DataFrame,
    min_reviews_for_aspect: int = 1,
) -> pd.DataFrame:
    """
    弱监督 aspect 目标（文档 4.2）：
    - 主信号：含该 aspect 关键词的评论的 stars 均值（1–5）
    - 若无命中评论：回退到商户 overall stars
    - 辅助：命中评论文本的情感映射到 1–5，与均值blend（可关）
    """
    need = {"business_id", "text", "stars"}
    miss = need - set(reviews.columns)
    if miss:
        raise ValueError(f"reviews 缺少列: {miss}")

    r = reviews.copy()
    r["business_id"] = r["business_id"].astype(str).str.strip()
    r["stars"] = pd.to_numeric(r["stars"], errors="coerce")
    r = r.dropna(subset=["business_id", "stars"])
    r["text"] = r["text"].fillna("").astype(str)

    stars_map = business_overall.set_index("business_id")["stars"].to_dict()
    all_ids = [str(x).strip() for x in business_overall["business_id"].tolist()]

    computed: dict[str, dict[str, Any]] = {}
    for bid, grp in r.groupby("business_id"):
        bid = str(bid).strip()
        overall = float(stars_map.get(bid, np.nan))
        if np.isnan(overall):
            overall = float(grp["stars"].mean())
        rec: dict[str, Any] = {"business_id": bid}
        for aspect, kws in ASPECT_KEYWORDS.items():
            mask = grp["text"].map(lambda t, kw=kws: _text_hits_aspect(t, kw))
            sub = grp.loc[mask]
            n_hit = len(sub)
            rec[f"{aspect}_keyword_hits"] = int(n_hit)
            if n_hit >= min_reviews_for_aspect:
                mean_stars = float(sub["stars"].mean())
                sent = float(sub["text"].map(_sentiment_score).mean())
                blended = 0.7 * mean_stars + 0.3 * sentiment_to_rating_1_5(sent)
                rec[f"target_{aspect}"] = float(np.clip(blended, 1.0, 5.0))
            else:
                rec[f"target_{aspect}"] = overall
        computed[bid] = rec

    rows: list[dict[str, Any]] = []
    for bid in all_ids:
        if bid in computed:
            rows.append(computed[bid])
            continue
        overall = float(stars_map.get(bid, np.nan))
        if np.isnan(overall):
            overall = 3.0
        rec = {"business_id": bid}
        for aspect in ASPECT_KEYWORDS:
            rec[f"{aspect}_keyword_hits"] = 0
            rec[f"target_{aspect}"] = overall
        rows.append(rec)

    return pd.DataFrame(rows)


def verify_merge(
    business: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[str, Any]:
    """Notebook 中打印，确认合并率与缺失。"""
    b_ids = set(business["business_id"].astype(str))
    t_ids = set(targets["business_id"].astype(str))
    return {
        "n_business": len(b_ids),
        "n_targets": len(t_ids),
        "targets_without_business": len(t_ids - b_ids),
        "business_without_targets": len(b_ids - t_ids),
    }


def verify_weak_targets(
    business: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[str, float]:
    """文档「验证」：aspect 与 overall 的相关性。"""
    m = business[["business_id", "stars"]].merge(targets, on="business_id", how="inner")
    out: dict[str, float] = {}
    y0 = m["stars"].astype(float)
    for a in ASPECT_KEYWORDS:
        col = f"target_{a}"
        if col in m.columns:
            out[f"corr_{a}_vs_overall"] = float(m[col].corr(y0))
    return out


def load_csv_business(path: str | Path) -> pd.DataFrame:
    relax_csv_field_limit()
    return pd.read_csv(path, dtype={"business_id": str}, low_memory=False)


def load_csv_reviews(path: str | Path) -> pd.DataFrame:
    relax_csv_field_limit()
    return pd.read_csv(path, dtype={"business_id": str, "review_id": str}, low_memory=False)
