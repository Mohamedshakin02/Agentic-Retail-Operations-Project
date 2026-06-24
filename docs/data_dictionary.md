# Data Dictionary: Retail Store Inventory Forecasting Dataset

## Overview

This dataset is a **synthetic retail dataset** designed for inventory management and demand forecasting tasks. It contains daily records across multiple stores and products, including sales, inventory, pricing, and external factors such as promotions and weather.

The dataset is used to:
- Forecast product demand
- Analyze sales trends
- Detect stock-out risk
- Optimize inventory and pricing strategies 【1-234e13】

---

## Dataset Structure

- Granularity: Daily (one row per date per store per product)
- Rows: ~73,000+
- Columns: ~15
- Format: CSV

---

## Column Definitions

### Core Identifiers

| Column Name | Description |
|------------|------------|
| `date` | Date of the transaction (daily granularity) |
| `store_id` | Unique identifier for each retail store |
| `product_id` | Unique identifier for each product |
| `category` | Product category (e.g., Electronics, Clothing, Groceries) |
| `region` | Geographic region of the store |

---

### Inventory & Sales

| Column Name | Description |
|------------|------------|
| `inventory_level` | Number of units available at the start of the day |
| `units_sold` | Number of units sold during the day |
| `units_ordered` | Number of units ordered for restocking |

---

### Demand & Forecasting

| Column Name | Description |
|------------|------------|
| `demand_forecast` | Estimated or predicted demand for the product |

---

### Pricing

| Column Name | Description |
|------------|------------|
| `price` | Selling price of the product |
| `discount` | Discount applied (if any) |
| `competitor_pricing` | Price of similar product from competitor |

---

### External Factors

| Column Name | Description |
|------------|------------|
| `weather_condition` | Weather type on that day (e.g., Sunny, Rainy) |
| `holiday_promotion` | Indicator (1 or 0) if it's a holiday or promotion day |
| `seasonality` | Seasonal category (e.g., Winter, Summer) |

---

## Important Notes

### 1. Synthetic Data
This dataset is artificially generated but designed to simulate real retail behavior. 【1-234e13】

---

### 2. Time-Series Nature
- Data must be **sorted by date**
- Forecasting models rely on **temporal patterns**

---

### 3. Key Relationships

- `units_sold` is influenced by:
  - price
  - promotions
  - weather
  - seasonality

- `inventory_level` impacts:
  - stock-out risk
  - sales limitations

---

### 4. Missing Values and Data Quality

Potential issues:
- Missing values in demand or pricing
- Outliers in sales
- Inconsistent inventory values

Recommended handling:
- Forward fill missing values
- Remove negative inventory or sales
- Validate date continuity

---

## Derived Features (Created in Project)

The following features are not part of the raw dataset but are engineered:

### Time-Based Features
- `day_of_week`
- `week_of_year`
- `month`
- `is_weekend`

### Sales Features
- `lag_1_day_sales`
- `lag_7_day_sales`
- `rolling_7_day_avg_sales`
- `rolling_14_day_avg_sales`
- `rolling_28_day_avg_sales`

### Inventory Features
- `inventory_cover_days`

### Pricing Features
- `price_change_pct`

### Flags
- `promotion_flag`
- `holiday_flag`

---

## Usage in Project

This dataset is used across the following modules:

| Module | Usage |
|------|------|
| Data Cleaning | Handle missing and invalid values |
| Feature Engineering | Create time-series and lag features |
| Forecasting | Predict future demand |
| Inventory Risk | Calculate stock-out and overstock risk |
| Recommendation Engine | Suggest business actions |

---

## Summary

This dataset provides a complete foundation for building an end-to-end retail analytics system, combining:
- demand forecasting
- inventory optimization
- business decision support

It supports both machine learning models and agent-based workflows.