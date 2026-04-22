"""
商家选址：地图落点 + 品类 → 空间特征 → 生存概率 / 预测星级（与 `tests/test_inference.py` 同逻辑）。
需本地 `models/artifacts/global_survival_model.pkl` 与 `global_rating_model.pkl`（见侧栏说明）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from pipelines.spatial_feature_engineer import SpatialFeatureEngineer  # noqa: E402

st.set_page_config(page_title="商家选址预测", layout="wide")
st.title("商家选址预测")
st.caption(
    "基于 `data/train_spatial.csv` 的同城参考商户，计算空间特征并调用全局生存 / 评分模型。"
)


@st.cache_data(show_spinner="加载 train_spatial …")
def _load_spatial_frame() -> pd.DataFrame:
    for rel in ("data/train_spatial.csv", "train_spatial.csv"):
        p = BASE / rel
        if p.is_file():
            return pd.read_csv(p)
    raise FileNotFoundError("未找到 train_spatial.csv（请放在 data/ 或仓库根目录）")


def _artifact_paths():
    d = BASE / "models" / "artifacts"
    return d / "global_survival_model.pkl", d / "global_rating_model.pkl"


def _predict(local_ref: pd.DataFrame, lat: float, lon: float, selected_cat_names: list[str]) -> dict:
    cat_cols = [c for c in local_ref.columns if c.startswith("cat_")]
    user_target_categories = np.zeros(len(cat_cols), dtype=float)
    for idx, col in enumerate(cat_cols):
        if col in selected_cat_names:
            user_target_categories[idx] = 1.0

    coord = (float(lat), float(lon))
    engineer = SpatialFeatureEngineer(None)
    live_df = engineer.engineer_single_target(coord, user_target_categories, local_ref)

    surv_path, rating_path = _artifact_paths()
    survival_model = joblib.load(surv_path)
    model_df = pd.DataFrame(0.0, index=[0], columns=survival_model.feature_names_in_)
    for col in live_df.columns:
        if col in model_df.columns:
            model_df[col] = live_df[col].values
    for idx, col in enumerate(cat_cols):
        if col in model_df.columns:
            model_df[col] = user_target_categories[idx]

    surv_prob = float(survival_model.predict_proba(model_df)[:, 1][0])

    rating_model = joblib.load(rating_path)
    model_df_reg = pd.DataFrame(0.0, index=[0], columns=rating_model.feature_names_in_)
    for col in live_df.columns:
        if col in model_df_reg.columns:
            model_df_reg[col] = live_df[col].values
    for idx, col in enumerate(cat_cols):
        if col in model_df_reg.columns:
            model_df_reg[col] = user_target_categories[idx]
    stars_pred = float(rating_model.predict(model_df_reg)[0])

    return {
        "surv_prob": surv_prob,
        "stars_pred": stars_pred,
        "live_df": live_df,
        "n_ref": len(local_ref),
    }


try:
    global_ref = _load_spatial_frame()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

surv_p, rating_p = _artifact_paths()
with st.sidebar:
    st.header("模型文件")
    if surv_p.is_file() and rating_p.is_file():
        st.success("已检测到 `global_survival_model.pkl` 与 `global_rating_model.pkl`。")
    else:
        st.warning("缺少 artifacts 下的 pkl。请在仓库根目录执行：")
        st.code(
            "cd <仓库根目录>\n"
            "PYTHONPATH=. .venv/bin/python -c \"\n"
            "from pathlib import Path\n"
            "from sklearn.model_selection import train_test_split\n"
            "import pandas as pd\n"
            "B = Path('.')\n"
            "df = pd.read_csv(B / 'data' / 'train_spatial.csv')\n"
            "tr, te = train_test_split(df, test_size=0.2, random_state=42)\n"
            "tr.to_csv(B / 'data' / 'train_merchant_split.csv', index=False)\n"
            "te.to_csv(B / 'data' / 'test_spatial.csv', index=False)\n"
            "\"\n"
            "PYTHONPATH=. .venv/bin/python -c \"\n"
            "from models.merchant_predictor import AblationMerchantPredictor\n"
            "p = AblationMerchantPredictor('data/train_merchant_split.csv', 'data/test_spatial.csv')\n"
            "p.train_pipeline()\n"
            "\"",
            language="bash",
        )

st.sidebar.divider()
st.sidebar.header("参考数据范围")

if "city" in global_ref.columns:
    cities = (
        global_ref["city"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.str.len() > 0]
    )
    city_opts = sorted(cities.str.title().unique(), key=str.lower)
    if not city_opts:
        st.error("数据中有 `city` 列但无有效城市名。")
        st.stop()
    default_city = "Philadelphia" if "Philadelphia" in city_opts else city_opts[0]
    city_sel = st.sidebar.selectbox(
        "城市（过滤参考商户）",
        city_opts,
        index=city_opts.index(default_city),
    )
    local_ref = global_ref[global_ref["city"].astype(str).str.lower() == city_sel.lower()].copy()
else:
    city_sel = None
    n_ctx = st.sidebar.slider("参考样本数（无 city 列时取前 N 行）", 500, 8000, 2000, 500)
    local_ref = global_ref.head(n_ctx).copy()

if len(local_ref) < 10:
    st.error("当前筛选下参考商户过少，请换城市或检查数据。")
    st.stop()

cat_cols = [c for c in local_ref.columns if c.startswith("cat_")]
weights = {c: float(local_ref[c].sum()) for c in cat_cols}
popular = sorted(weights.keys(), key=lambda c: weights[c], reverse=True)
top_pick = popular[: min(120, len(popular))]

st.sidebar.subheader("品类（多选）")
preset = st.sidebar.multiselect(
    "常见品类（按同城出现次数排序）",
    options=top_pick,
    default=[c for c in ("cat_coffee_&_tea", "cat_fast_food") if c in top_pick][:2],
)

c1, c2, c3 = st.columns(3)
with c1:
    lat = st.number_input("纬度 latitude", value=39.9526, format="%.6f")
with c2:
    lon = st.number_input("经度 longitude", value=-75.1652, format="%.6f")
with c3:
    st.metric("参考商户数", f"{len(local_ref):,}")

run = st.button("计算预测", type="primary")

if run:
    if not surv_p.is_file() or not rating_p.is_file():
        st.error("请先训练并生成 models/artifacts 下的两个 pkl 文件。")
        st.stop()
    if not preset:
        st.warning("请至少选择一个品类。")
        st.stop()
    try:
        out = _predict(local_ref, lat, lon, preset)
    except Exception as ex:
        st.exception(ex)
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("开业概率（模型）", f"{out['surv_prob'] * 100:.1f}%")
    m2.metric("预测星级", f"{out['stars_pred']:.2f} / 5.0")
    lf = out["live_df"]
    km3 = lf["count_all_3.0km"].iloc[0] if "count_all_3.0km" in lf.columns else float("nan")
    m3.metric("3km 内竞品数", f"{float(km3):.0f}" if pd.notna(km3) else "N/A")

    st.subheader("部分空间特征")
    priority_cols = [
        "count_all_0.5km",
        "count_all_3.0km",
        "avg_rating_all_3.0km",
        "survival_top5_similar",
        "avg_rating_top5_similar",
        "dist_nearest_same_cat",
        "log_dist_nearest_same_cat",
    ]
    show_cols = [c for c in priority_cols if c in lf.columns]
    rest = [c for c in lf.columns if c not in show_cols]
    show_cols = (show_cols + rest)[:24]
    st.dataframe(lf[show_cols], use_container_width=True)

    st.caption(f"城市：{city_sel or 'N/A'}；已选品类：{', '.join(preset)}")
