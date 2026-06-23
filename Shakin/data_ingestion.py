import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np

COLUMN_RENAME_MAP = {
    "Date": "date", "Store ID": "store_id", "Product ID": "product_id",
    "Category": "category", "Region": "region",
    "Inventory Level": "inventory_level", "Units Sold": "units_sold",
    "Units Ordered": "units_ordered", "Demand Forecast": "dataset_demand_forecast",
    "Price": "price", "Discount": "discount",
    "Weather Condition": "weather_condition",
    "Holiday/Promotion": "holiday_or_promo_flag",
    "Competitor Pricing": "competitor_price", "Seasonality": "seasonality",
}

RAW_DATA_PATH = Path("Shakin/retail_store_inventory.csv")
PROCESSED_DATA_PATH = Path("data/processed/retail_cleaned.csv")


def load_raw_data():
    try:
        df = pd.read_csv("Shakin/retail_store_inventory.csv")
        print(f"Loaded data: {df.shape}")
        return df
    except FileNotFoundError:
        raise Exception(" Dataset not found. Please ensure 'retail_store_inventory.csv' is in the correct path.")

def basic_cleaning(df):
    df = df.copy()

    # Rename columns
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Replace negative Demand Forecast with 0, as demand forcast has negative values
    df['dataset_demand_forecast'] = df['dataset_demand_forecast'].clip(lower=0)


    # Label Encoding , converting categorical text values into numerical labels
    
    # (Keeping both original version and encoded version)

    store_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    region_encoder = LabelEncoder()
    weather_encoder = LabelEncoder()
    season_encoder = LabelEncoder()

    # Store
    df['store_id_encoded'] = store_encoder.fit_transform(df['store_id'])

    # Product
    df['product_id_encoded'] = product_encoder.fit_transform(df['product_id'])

    # Category
    df['category_encoded'] = category_encoder.fit_transform(df['category'])

    # Region
    df['region_encoded'] = region_encoder.fit_transform(df['region'])

    # Weather
    df['weather_condition_encoded'] = weather_encoder.fit_transform(df['weather_condition'])

    # Seasonality
    df['seasonality_encoded'] = season_encoder.fit_transform(df['seasonality'])

    # Log Transformations, as units sold and demand forecast values were outliers (there were extreme values from most of its values range)
    df['units_sold_log'] = np.log1p(df['units_sold'])
    df['demand_forecast_log'] = np.log1p(df['dataset_demand_forecast'])

    return df

def save_processed_data(df):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    df = load_raw_data()
    df_clean = basic_cleaning(df)
    save_processed_data(df_clean)



df = df.rename(columns=COLUMN_RENAME_MAP)