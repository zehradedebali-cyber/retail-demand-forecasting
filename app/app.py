import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from catboost import CatBoostRegressor
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide"
)


# --------------------------------------------------
# COMPACT STYLE
# --------------------------------------------------

st.markdown("""
<style>

/* ---------- MAIN LAYOUT ---------- */

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* ---------- GLOBAL FONT SCALE ---------- */

html, body, [class*="css"] {
    font-size: 14px;
}

/* ---------- HEADINGS ---------- */

h2 {
    font-size: 1.55rem !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.35rem !important;
}

h3 {
    font-size: 1.25rem !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.3rem !important;
}

/* ---------- NORMAL TEXT ---------- */

p {
    font-size: 0.9rem;
}

/* ---------- METRICS ---------- */

[data-testid="stMetric"] {
    padding: 0 !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {
    min-width: 250px !important;
    max-width: 250px !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 1.1rem !important;
}

/* ---------- SELECT BOXES ---------- */

[data-baseweb="select"] > div {
    min-height: 2.2rem !important;
    font-size: 0.85rem !important;
}

/* ---------- DIVIDERS ---------- */

hr {
    margin-top: 0.45rem !important;
    margin-bottom: 0.45rem !important;
}

/* ---------- VERTICAL GAPS ---------- */

[data-testid="stVerticalBlock"] {
    gap: 0.55rem !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📦 Retail Demand Forecasting")

st.caption(
    "Next-week demand prediction using the final CatBoost model"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "app_data.pkl"
MODEL_PATH = (
    BASE_DIR.parent
    / "data"
    / "checkpoints"
    / "catboost_v4_final.cbm"
)


# --------------------------------------------------
# LOAD DATA + MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_pickle(DATA_PATH)


@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model


app_data = load_data()
model = load_model()


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

FEATURES = [
    "store_id",
    "item_id",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
    "sell_price",
    "price_change_pct",
    "snap_days",
    "lag_52",
    "lag_3",
    "lag_12",
    "lag_26",
    "rolling_mean_2",
    "rolling_mean_12",
    "momentum_1_4",
    "trend_4_8",
    "avg_price_item_store",
    "price_ratio",
]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Forecast Settings")

stores = sorted(app_data["store_id"].unique())

selected_store = st.sidebar.selectbox(
    "Store",
    stores
)

available_items = sorted(
    app_data.loc[
        app_data["store_id"] == selected_store,
        "item_id"
    ].unique()
)

selected_item = st.sidebar.selectbox(
    "Item",
    available_items
)


# --------------------------------------------------
# SELECT RECORD
# --------------------------------------------------

record = app_data[
    (app_data["store_id"] == selected_store)
    & (app_data["item_id"] == selected_item)
].iloc[0]


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

X_pred = record[FEATURES].to_frame().T.copy()

# CatBoost categorical features
X_pred["store_id"] = X_pred["store_id"].astype(str)
X_pred["item_id"] = X_pred["item_id"].astype(str)

prediction = float(model.predict(X_pred)[0])
prediction = max(0, prediction)


# --------------------------------------------------
# MAIN KPI
# --------------------------------------------------

st.subheader("Next Week Forecast")

col1, col2, col3, col4 = st.columns(4)

last_week_sales = float(record["lag_1"])

if last_week_sales > 0:
    forecast_change_pct = (
        (prediction - last_week_sales) / last_week_sales
    ) * 100
else:
    forecast_change_pct = None

col1.metric(
    "Predicted Demand",
    f"{prediction:.1f} units",
    delta=(
        f"{forecast_change_pct:+.1f}% vs last week"
        if forecast_change_pct is not None
        else None
    )
)

col2.metric(
    "Last Week Sales",
    f"{record['lag_1']:.1f} units"
)

col3.metric(
    "Current Price",
    f"${record['sell_price']:.2f}"
)

col4.metric(
    "Price Change",
    f"{record['price_change_pct']:.1f}%"
)


# --------------------------------------------------
# SALES HISTORY
# --------------------------------------------------

st.divider()
st.subheader("Recent Demand Pattern")

period_order = [
    "52 weeks ago",
    "26 weeks ago",
    "12 weeks ago",
    "8 weeks ago",
    "4 weeks ago",
    "3 weeks ago",
    "2 weeks ago",
    "Last week",
    "Forecast",
]

history = pd.DataFrame({
    "Period": period_order,
    "Sales": [
        record["lag_52"],
        record["lag_26"],
        record["lag_12"],
        record["lag_8"],
        record["lag_4"],
        record["lag_3"],
        record["lag_2"],
        record["lag_1"],
        prediction,
    ]
})

fig = px.line(
    history,
    x="Period",
    y="Sales",
    markers=True,
    category_orders={"Period": period_order},
)

fig.update_layout(
    xaxis_title=None,
    yaxis_title="Units Sold",
    hovermode="x unified",
    height=210,
    margin=dict(l=20, r=20, t=10, b=20),
)

fig.update_xaxes(
    categoryorder="array",
    categoryarray=period_order
)

st.plotly_chart(
    fig,
    width="stretch",
    config={"displayModeBar": False}
)

st.divider()


# --------------------------------------------------
# DEMAND SIGNALS
# --------------------------------------------------

st.divider()

st.subheader("Demand Signals")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "2-Week Average",
    f"{record['rolling_mean_2']:.1f}"
)

c2.metric(
    "4-Week Average",
    f"{record['rolling_mean_4']:.1f}"
)

c3.metric(
    "8-Week Average",
    f"{record['rolling_mean_8']:.1f}"
)

c4.metric(
    "12-Week Average",
    f"{record['rolling_mean_12']:.1f}"
)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.subheader("Final Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric(
    "Test MAE",
    "3.8537"
)

m2.metric(
    "Test RMSE",
    "9.1546"
)

m3.metric(
    "Model",
    "CatBoost"
)

st.caption(
    "Final model selected after comparison with "
    "LightGBM and XGBoost using the same temporal test set."
)

st.divider()

st.subheader("Model Comparison")

model_comparison = pd.DataFrame({
    "Model": [
        "CatBoost",
        "LightGBM",
        "XGBoost"
    ],
    "Test MAE": [
        3.8537,
        3.8659,
        3.8885
    ],
    "Test RMSE": [
        9.1546,
        9.1975,
        9.3442
    ]
})

st.dataframe(
    model_comparison,
    hide_index=True,
    width="stretch"
)

st.caption(
    "CatBoost achieved the best overall performance on the "
    "unseen temporal test set and was selected as the final model."
)

st.info(
    "The forecast is generated using 22 engineered features, "
    "including historical sales lags, rolling demand statistics, "
    "price signals, SNAP information, momentum and long-term "
    "seasonal demand patterns."
)
