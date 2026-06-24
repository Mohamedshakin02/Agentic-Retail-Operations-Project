# Agentic-Retail-Operations-Project--Al-Futtaim
https://www.kaggle.com/code/beshoyemad24/retail-store-inventory-forecastin-eda-prediction#Exploratory-Data-Analysis-(EDA)
Function:What it does
add_date_features: Pulls day_of_week, week_of_year, month, is_weekend out of the date
add_lag_features: Adds "what were sales yesterday / 7 days ago for this exact product"
add_rolling_features: Adds 7/14/28-day rolling average sales, plus rolling inventory
add_price_features: Adds % price change vs. the day before
add_inventory_cover_feature: Adds the days-of-cover number
add_categorical_encoding: Turns text like "Electronics" into numbers models can use
build_features:Runs all of the above, in the right order, with a column-check first

Column names list of cleaned data-

['date',
 'store_id',
 'product_id',
 'category',
 'region',
 'inventory_level',
 'units_sold',
 'units_ordered',
 'dataset_demand_forecast',
 'price',
 'discount',
 'weather_condition',
 'holiday_or_promo_flag',
 'competitor_price',
 'seasonality',
 'store_id_encoded',
 'product_id_encoded',
 'category_encoded',
 'region_encoded',
 'weather_condition_encoded',
 'seasonality_encoded',
 'units_sold_log',
 'demand_forecast_log']