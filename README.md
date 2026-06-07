# Demand Pricing Forecast

Demand forecasting and price-signal modeling on the M5 retail sales dataset. The
project uses notebook-based analysis to move from raw Walmart-style retail data
through feature engineering, baseline validation, and LightGBM training.

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   |   |-- calendar.csv
|   |   |-- sales_train_validation.csv
|   |   |-- sales_train_evaluation.csv
|   |   |-- sell_prices.csv
|   |   `-- sample_submission.csv
|   `-- processed/
|       |-- df_train.parquet
|       `-- df_train.pkl
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_feature_engineering.ipynb
|   |-- 03_baseline_models.ipynb
|   `-- 04_lightgbm_training.ipynb
|-- requirements.txt
`-- README.md
```

## Data

The raw files match the M5 forecasting dataset format:

- `sales_train_validation.csv`: daily unit sales in wide format for 30,490 item-store series.
- `calendar.csv`: date, event, weekday, week, month, year, and SNAP metadata.
- `sell_prices.csv`: weekly item prices by store.
- `sales_train_evaluation.csv` and `sample_submission.csv`: evaluation/submission files.

The feature engineering notebook melts the sales data to long format and builds a
processed training table with 58,327,370 rows and 52 columns.

## Workflow

Run the notebooks in order:

1. `notebooks/01_eda.ipynb`
   - Explores demand, seasonality, category/store volume, events, SNAP effects,
     and price distributions.
   - Key findings include strong weekend seasonality, FOODS dominance, SNAP lift
     for FOODS, and weak raw price-demand correlation before controlling for
     confounders.

2. `notebooks/02_feature_engineering.ipynb`
   - Merges sales, calendar, and price data.
   - Builds calendar, SNAP, event, price, lag, rolling demand, holiday, and
     aggregate demand features.
   - Downcasts data types to reduce memory usage.
   - Saves the processed table to `data/processed/df_train.parquet`.

3. `notebooks/03_baseline_models.ipynb`
   - Creates a time-based split using the final 28 days as validation.
   - Evaluates simple benchmark models:
     - Naive last-28-day average RMSE: `2.2186`
     - Moving-average last-7-day RMSE: `2.0970`

4. `notebooks/04_lightgbm_training.ipynb`
   - Trains a LightGBM model with a Tweedie objective for intermittent retail
     demand.
   - Uses item, department, category, store, and state as categorical features.
   - Includes an Optuna tuning section for hyperparameter search.

## Engineered Features

The processed dataset includes:

- Calendar features: day of week, day of month, week of year, month, quarter,
  weekend flag, month-start flag, and month-end flag.
- SNAP and event features: state-specific SNAP flag, event flags, and separate
  sporting, national, religious, and cultural event flags.
- Price features: price lags, percentage price changes, rolling average price,
  price relative to rolling mean, item average price, category average price,
  and price relative to category average.
- Demand history features: sales lags at 7, 28, and 56 days.
- Rolling demand features: rolling means, standard deviations, maximums, and
  minimums over 7, 28, and 56 day windows.
- Interaction and holiday features: SNAP x FOODS flag, days before next holiday,
  and pre-holiday flag.
- Aggregate demand features: 28-day rolling averages by store, category, and
  department.

All lag and rolling features are shifted so the current day's sales are not used
as predictors.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open the notebooks from the `notebooks/` directory and run them in numeric order.

## Modeling Objective

The immediate modeling target is to beat the best baseline RMSE of `2.0970` on a
28-day holdout split. LightGBM is used because it handles large tabular datasets,
categorical features, sparse intermittent demand, lagged time-series signals,
calendar effects, and price features in one global model.
