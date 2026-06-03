# M5 Demand Forecasting & Dynamic Pricing

End-to-end demand forecasting pipeline built on the M5 Forecasting 
competition dataset — 58.3M rows of daily Walmart retail sales across 
30,490 product-store combinations spanning 5.5 years.

## Business Problem

Retail pricing and inventory decisions depend on accurate demand forecasts. 
This project forecasts daily unit sales at the item-store level — the most 
granular level where pricing decisions are made. A 10% price change on a 
specific product in a specific store needs a demand forecast to determine 
whether it's profitable.

## Dataset

M5 Forecasting Accuracy Competition (Kaggle)
- 3,049 unique items × 10 Walmart stores across CA, TX, WI
- Daily sales from January 2011 to April 2016
- 3 categories: FOODS (68.6%), HOUSEHOLD (22%), HOBBIES (9.3%)
- External signals: sell prices, calendar events, SNAP benefit days

## Project Structure
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature engineering pipeline
│   ├── 03_baseline_models.ipynb      # Naive and moving average baselines
│   ├── 04_lightgbm_training.ipynb    # LightGBM training and Optuna tuning
│   ├── 05_evaluation.ipynb           # RMSE and WRMSSE evaluation
│   └── 06_error_analysis.ipynb       # Where and why the model fails
├── data/
│   ├── raw/                          # Original M5 competition files
│   └── processed/                    # Engineered feature dataset
├── requirements.txt
└── README.md


## Key EDA Findings

| Signal | Impact |
|---|---|
| SNAP benefit days | +17.2% demand lift in FOODS category |
| Weekend vs midweek | +38% Saturday premium vs Wednesday |
| National holidays | -14.6% — store closures dominate |
| Sporting events | +3.8% — only event type above baseline |
| August vs May | +10% — summer surge vs spring trough |

These findings directly motivated feature engineering decisions.

## Feature Engineering

48 features built across 6 groups:

- **Lag features** — lag_7, lag_28, lag_56 (weekly/monthly cycles)
- **Rolling features** — mean, std, max, min across 7/28/56 day windows
- **Price features** — price momentum, relative positioning vs category average
- **Calendar features** — day of week, month, is_weekend, is_month_start
- **Event features** — separate flags per event type (sporting, national, religious, cultural)
- **Aggregated demand** — rolling means at store, category, department level

All rolling features use shift(1) before the window — strict leakage prevention.

## Model

**Algorithm:** LightGBM with Tweedie objective
- Tweedie chosen for sparse count data with many zero-sale days
- Trained on 57.4M rows, validated on last 28 days (853K rows)
- Hyperparameters tuned with Optuna TPE sampler — 20 trials

**Why LightGBM over classical models:**
- 30,490 series makes per-series ARIMA impractical (~63 days compute)
- Classical models cannot use exogenous features (SNAP, price, events)
- Validated by M5 competition — all top 50 finishers used LightGBM

## Results

| Model | RMSE | vs Naive |
|---|---|---|
| Naive baseline | 2.2186 | — |
| Moving average | 2.2920 | -3.3% |
| LightGBM (initial) | 1.9005 | +14.3% |
| LightGBM (Optuna) | TBD | TBD |

WRMSSE evaluation — matching M5 Kaggle competition scoring — weights 
forecast errors by revenue importance. High-value items penalised more 
for missed forecasts.

## How to Run

```bash
# install dependencies
pip install -r requirements.txt

# run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_baseline_models.ipynb
jupyter notebook notebooks/04_lightgbm_training.ipynb
```

Data files are not included in this repository due to size. 
Download from: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

## Requirements
pandas
numpy
lightgbm
optuna
plotly
scikit-learn
pyarrow
