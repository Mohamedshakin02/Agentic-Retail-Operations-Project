import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("Farah's\retail_store_inventory.csv")
PROCESSED_DATA_PATH = Path("data/processed/retail_cleaned.csv")


def load_raw_data():
    try:
        df = pd.read_csv("Farah's\retail_store_inventory.csv")
        print(f"✅ Loaded data: {df.shape}")
        return df
    except FileNotFoundError:
        raise Exception("❌ Dataset not found. Please ensure 'retail_store_inventory.csv' is in the correct path.")

def basic_cleaning(df):
    df = df.copy()

    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop duplicates
    df = df.drop_duplicates()

    # Remove negative values
    df = df[df['units_sold'] >= 0]
    df = df[df['inventory_level'] >= 0]

    # Fill missing values
    df.fillna(method='ffill', inplace=True)

    return df

def save_processed_data(df):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Saved cleaned data to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    df = load_raw_data()
    df_clean = basic_cleaning(df)
    save_processed_data(df_clean)


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
df = df.rename(columns=COLUMN_RENAME_MAP)