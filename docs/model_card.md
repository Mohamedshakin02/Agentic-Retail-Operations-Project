# Model Card: Demand Forecasting Model

## 1. Overview

This model predicts **7-day product demand** for each store-product combination in a retail setting.

It is a supervised machine learning model trained on historical sales, inventory, and contextual features.

The output is used for:
- inventory planning
- stock-out risk detection
- business action recommendations

---

## 2. Model Type

| Attribute | Value |
|----------|------|
| Model | RandomForestRegressor |
| Library | scikit-learn |
| Task | Regression |
| Target | units_sold or units_sold_log |
| Output | forecast_7_day_demand |

---

## 3. Model Selection Strategy

Two models are trained and compared:

### Option 1: Raw Target
Target:
units_sold

### Option 2: Log-Transformed Target
Target:

units_sold_log = log1p(units_sold)

---

### Selection Logic

- Both models are evaluated using WMAPE
- Predictions are always converted back to raw units
- The model with the **lower WMAPE** is selected

Code behavior:
```python
best_key = min(results, key=lambda k: results[k]["metrics"]["WMAPE"])
4. Features Used
The model uses engineered features from feature_engineering.py.
Time Features

day_of_week
week_of_year
month
is_weekend


Sales Features

lag_1_day_sales
lag_7_day_sales
rolling_7_day_avg_sales
rolling_14_day_avg_sales
rolling_28_day_avg_sales


Pricing Features

price
discount
price_change_pct


Inventory Features

inventory_level
rolling_7_day_inventory
inventory_cover_days


Encoded Features

category_encoded
region_encoded
weather_condition_encoded
seasonality_encoded


External Signals

holiday_or_promo_flag


5. Training Process
Data Preparation

Only rows with all required features are used
Missing values are dropped for model training
Feature selection is dynamic (uses only columns present in dataset)


Train-Test Split

Uses random split:

train_test_split(test_size=0.2)


Important Limitation
This is NOT a time-based split.
Implication:

Model may see "future-like" patterns during training
This can lead to optimistic performance metrics


6. Evaluation Metrics
The model is evaluated using:
MAE (Mean Absolute Error)
Average absolute prediction error

RMSE (Root Mean Squared Error)
Penalizes larger errors more heavily

WMAPE (Weighted Mean Absolute Percentage Error)
Formula:
WMAPE = sum(|actual - forecast|) / sum(actual)

Used as primary metric because:

standard in retail forecasting
scale-invariant


7. Baseline Comparison
A naive baseline is calculated:
rolling_7_day_avg_sales

Purpose:

ensures model performs better than simple average
avoids unnecessary model complexity


8. Prediction Logic
Input
Latest available row per:
store_id + product_id


Output
forecast_7_day_demand

Generated using:

trained RandomForest model
most recent feature values


Log Handling
If log model is used:

predictions transformed using:

expm1(prediction)


9. Assumptions

Historical patterns repeat in short-term future
Recent trends (lags, rolling averages) are strong predictors
External signals (promotions, weather) influence demand


10. Limitations
1. Random Split Instead of Time-Based Split

not realistic for production forecasting
risk of data leakage


2. Short-Term Forecast Only

limited to 7-day prediction horizon
not suitable for long-term planning


3. Feature Dependency

relies heavily on feature engineering quality
incorrect features → poor predictions


4. No Sequential Model

RandomForest does not capture temporal dependencies explicitly
lacks true time-series modeling


5. Static Dataset

model is trained on static CSV
no real-time updates


11. Risks

inaccurate forecasts → wrong inventory decisions
overfitting due to random data split
sensitivity to outliers in sales data


12. Mitigation Strategies

compare against baseline model
use WMAPE as primary metric
include human approval before business actions
log predictions for traceability


13. Ethical and Business Considerations

No direct automation of business decisions
All outputs require human approval
Model transparency ensured via documentation


15. Summary
This model provides:

scalable demand forecasting
interpretable input-output relationship
integration-ready outputs for downstream decision systems
It serves as a strong baseline for retail demand forecasting in an agent-based system.


