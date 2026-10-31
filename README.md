# 📦 Retail Demand Forecasting

An end-to-end machine learning project for forecasting **weekly retail demand at the store-item level**, using historical sales, pricing information, calendar signals, and leakage-safe time-series features.

The project compares multiple gradient boosting approaches and progressively improves the forecasting pipeline through feature engineering, feature selection, objective-specific hyperparameter tuning, and temporal validation.

The final **CatBoost** model achieved:

- **Test MAE: 3.8537**
- **Test RMSE: 9.1546**

A Streamlit application was also developed to demonstrate next-week demand predictions interactively.

---

## 🎯 Business Problem

Accurate demand forecasting is essential for retail inventory planning.

Underestimating demand can lead to:

- stockouts,
- lost sales,
- poor customer experience.

Overestimating demand can lead to:

- excess inventory,
- higher storage costs,
- unnecessary capital tied up in stock.

The goal of this project is therefore to predict **next-week unit demand for each store-item combination** using information that would realistically be available before the forecast week.

---

## 📊 Dataset

The project uses an **M5-style retail dataset** containing:

- daily item-level sales,
- store and product identifiers,
- weekly selling prices,
- calendar information,
- events and holidays,
- SNAP indicators.

Daily sales were transformed into a **weekly store-item forecasting dataset**.

The modeling unit is:

```text
store_id × item_id × week
```

This structure allows the model to learn both item-specific and store-specific demand patterns.

---

## 🔍 Exploratory Data Analysis

The exploratory analysis investigated:

- sales distributions,
- store-level demand,
- category and department demand,
- weekday patterns,
- event effects,
- SNAP-related behavior,
- price changes,
- temporal sales patterns.

Special attention was given to price changes and extreme percentage movements.

For focused price-change analysis, observations were restricted to realistic consecutive-week changes to reduce distortion from extreme denominator effects.

---

## 🛠 Feature Engineering

The forecasting pipeline uses only information available **before the target week**.

To prevent target leakage, rolling statistics are calculated after shifting the sales series by one period.

### Lag Features

Historical demand:

```text
lag_1
lag_2
lag_3
lag_4
lag_8
lag_12
lag_26
lag_52
```

`lag_52` captures annual demand seasonality.

### Rolling Demand Features

```text
rolling_mean_2
rolling_mean_4
rolling_mean_8
rolling_mean_12
rolling_std_4
```

These features summarize short- and medium-term demand behavior.

### Demand Dynamics

Two additional features describe recent demand movement:

```text
momentum_1_4
trend_4_8
```

These help the model distinguish temporary demand spikes from broader trends.

### Price Features

```text
sell_price
price_change_pct
avg_price_item_store
price_ratio
```

Price history was also calculated using past information only.

### Additional Features

```text
store_id
item_id
snap_days
```

The final model uses **22 engineered features**.

---

## ⏱ Temporal Train / Validation / Test Split

Because this is a forecasting problem, random splitting would allow information from future weeks to leak into training.

The dataset was therefore split chronologically.

| Dataset | Week Index |
|---|---|
| Train | 52–269 |
| Validation | 270–273 |
| Test | 274–277 |

Approximate sizes:

| Split | Rows |
|---|---:|
| Train | 4,895,700 |
| Validation | 120,242 |
| Test | 120,530 |

The test set represents the final unseen future period and was not used during model development.

---

## 📏 Evaluation Metrics

Two regression metrics were used.

### MAE — Mean Absolute Error

MAE measures the average absolute forecasting error.

It is easy to interpret because it remains in the same unit as sales.

### RMSE — Root Mean Squared Error

RMSE penalizes large forecasting errors more heavily.

Using both metrics provides a more complete view of model performance, particularly because retail demand contains occasional large spikes.

---

## 🧪 Model Development

The modeling process was intentionally iterative rather than jumping directly to a complex final model.

### Naive Baseline

The first benchmark predicts:

```text
next week's demand = last week's demand
```

Validation results:

| Model | MAE | RMSE |
|---|---:|---:|
| Naive | 3.9089 | 8.4082 |

This establishes the minimum benchmark a machine learning model should outperform.

---

## 🌳 LightGBM Development

Several LightGBM versions were developed progressively.

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Naive | 3.9089 | 8.4082 |
| LightGBM v1 | 3.2782 | 6.9728 |
| LightGBM v2 | 3.2447 | 6.9119 |
| LightGBM v3 + `lag_52` | 3.2443 | 6.8727 |

Adding annual demand information through `lag_52` produced an incremental improvement.

---

## 🔬 v4 Feature Engineering

Nine additional candidate features were evaluated:

```text
lag_3
lag_12
lag_26
rolling_mean_2
rolling_mean_12
momentum_1_4
trend_4_8
avg_price_item_store
price_ratio
```

The 29-feature candidate model achieved:

```text
Validation MAE:  3.2174
Validation RMSE: 6.8459
```

---

## 🧠 Permutation Importance

Permutation importance was used to evaluate the contribution of each feature to forecasting performance.

The strongest signals included:

1. `lag_1`
2. `rolling_mean_4`
3. `rolling_mean_2`
4. `rolling_mean_8`
5. `item_id`
6. `lag_2`
7. `rolling_mean_12`
8. `snap_days`
9. `lag_52`

One particularly important finding was the strength of **`rolling_mean_2`**, which became one of the most predictive features in the model.

Seven features showed zero permutation importance and were removed:

```text
month
year
sporting_days
event_days
religious_days
national_days
cultural_days
```

This reduced the model from **29 to 22 features**.

---

## ⚙️ LightGBM v4

The selected 22-feature LightGBM model achieved:

```text
Validation MAE:  3.2158
Validation RMSE: 6.8583
Best iteration:  392
```

Although the full 29-feature model had a slightly lower RMSE, the selected model achieved slightly better MAE with fewer features.

The 22-feature configuration was therefore adopted as **v4**.

---

## 🎯 L1 vs L2 Objective Comparison

Two LightGBM objectives were tuned separately using Optuna.

### L1 — MAE-Oriented

L1 optimization is less sensitive to extreme demand spikes.

Validation:

```text
MAE:  3.1465
RMSE: 6.9999
```

### L2 — RMSE-Oriented

L2 penalizes large errors more aggressively.

Validation:

```text
MAE:  3.2250
RMSE: 6.9208
```

The experiment demonstrated that optimizing the training objective for a particular metric does not necessarily guarantee superior generalization on the final future period.

---

## 🚀 Final Model Comparison

All final models were evaluated on the same unseen temporal test period.

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Naive | 4.7234 | 10.9338 |
| LightGBM v3 | 4.0473 | 9.2676 |
| LightGBM v4 Baseline | 4.0043 | 9.3281 |
| LightGBM v4 Tuned L2 | 4.0038 | 9.2931 |
| XGBoost v4 | 3.8885 | 9.3442 |
| LightGBM v4 Tuned L1 | 3.8659 | 9.1975 |
| **CatBoost v4** | **3.8537** | **9.1546** |

---

## 🏆 Final Model — CatBoost

CatBoost produced the strongest overall performance on the unseen temporal test set.

Final results:

```text
Validation MAE:  3.1456
Validation RMSE: 6.9398

Test MAE:        3.8537
Test RMSE:       9.1546
```

Best iteration:

```text
1482
```

CatBoost was therefore selected as the final forecasting model.

It narrowly outperformed the tuned LightGBM L1 model on both test MAE and RMSE.

---

## 💡 Key Findings

Several useful conclusions emerged from the project.

### Recent demand is the strongest predictor

`lag_1` and rolling demand statistics dominate permutation importance.

The immediate sales history therefore contains more predictive information than static calendar variables.

### Short-term smoothing is highly informative

`rolling_mean_2` became one of the strongest features despite being introduced relatively late in the project.

### Long-term seasonality matters

`lag_52` improved forecasting performance by providing information about demand during the same period of the previous year.

### Price features have weaker individual importance

Price variables showed relatively small standalone permutation importance.

However, they were retained because they remain economically meaningful and may contribute through interactions with other variables.

### Temporal generalization is harder than validation performance suggests

All models performed worse on the final test period than on validation.

This highlights an important forecasting reality:

> Good validation performance does not guarantee identical performance in a future demand regime.

---

## 🖥 Streamlit Forecasting Demo

An interactive Streamlit application was developed using the final CatBoost model.

The application allows a user to select:

```text
Store
Item
```

and generates a **next-week demand forecast**.

The dashboard displays:

- predicted next-week demand,
- last-week sales,
- expected change versus last week,
- current price,
- price change,
- recent demand history,
- rolling demand signals,
- final model metrics,
- model comparison results.

Run the application locally with:

```bash
python -m streamlit run app/app.py
```

---

## 📁 Project Structure

```text
retail-demand-forecasting/
│
├── app/
│   ├── app.py
│   └── app_data.pkl
│
├── data/
│   ├── raw/
│   └── checkpoints/
│
├── notebooks/
│   └── Retail_Demand_Forecasting_Final.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

Large model checkpoints and intermediate datasets are intentionally excluded from Git.

---

## 🧰 Technologies

The project uses:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- LightGBM
- CatBoost
- XGBoost
- Optuna
- Plotly
- Streamlit
- Jupyter
- Git / GitHub

---

## ⚠️ Limitations

The current forecasting pipeline predicts demand but does not directly optimize inventory decisions.

Potential limitations include:

- demand distribution changes over time,
- extreme sales spikes,
- limited external economic variables,
- no explicit stockout correction,
- point forecasts rather than prediction intervals.

---

## 🔮 Future Work

Possible extensions include:

- inventory optimization and reorder-point modeling,
- safety-stock estimation,
- probabilistic demand forecasting,
- prediction intervals,
- hierarchical forecasting,
- LSTM / GRU sequence models,
- Temporal Fusion Transformer,
- additional promotion and external economic signals.

Deep learning models were intentionally left outside the final project scope so that the project could focus on a robust and well-evaluated gradient boosting pipeline.

---

## 👤 Author

**Zehra Dedebali**

Data Science / Machine Learning Portfolio Project
