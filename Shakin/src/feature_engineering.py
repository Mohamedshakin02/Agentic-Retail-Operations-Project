import pandas as pd
import numpy as np
from pathlib import Path

# Define input and output paths based on the project structure
INPUT_PATH = Path("data/processed/retail_cleaned.csv")
OUTPUT_PATH = Path("data/processed/retail_features.csv")

def load_cleaned_data():
    """Loads the cleaned dataset, ensuring dates are parsed correctly."""
    try:
        # Parse 'date' column as datetime objects immediately upon loading
        df = pd.read_csv(INPUT_PATH, parse_dates=['date'])
        print(f"Successfully loaded cleaned data: {df.shape}")
        return df
    except FileNotFoundError:
        raise Exception(f"File not found: {INPUT_PATH}. Please ensure the cleaning script has been run.")

def create_date_features(df):
    """Extracts fundamental date components required for forecasting."""
    df = df.copy()
    
    # Standard date features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    
    # Boolean flag for weekends (Saturday=5, Sunday=6)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

def create_time_series_features(df):
    """
    Creates lag and rolling average features. 
    Crucially, groups by store and product to prevent data leakage between different items.
    """
    df = df.copy()
    
    # Sort by store, product, and date to ensure chronological order for shifting/rolling
    df = df.sort_values(by=['store_id', 'product_id', 'date'])
    
    # Create a groupby object to apply transformations per item per store
    grouped = df.groupby(['store_id', 'product_id'])
    
    # 1. Lag Features (Historical Sales)
    df['lag_1_day_sales'] = grouped['units_sold'].shift(1)
    df['lag_7_day_sales'] = grouped['units_sold'].shift(7)
    
    # 2. Rolling Averages (Sales Trends)
    # min_periods=1 allows the calculation to start even if a full 7/14/28 days aren't available yet
    df['rolling_7_day_avg_sales'] = grouped['units_sold'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['rolling_14_day_avg_sales'] = grouped['units_sold'].transform(lambda x: x.rolling(window=14, min_periods=1).mean())
    df['rolling_28_day_avg_sales'] = grouped['units_sold'].transform(lambda x: x.rolling(window=28, min_periods=1).mean())
    
    # 3. Rolling Average (Inventory)
    df['rolling_7_day_inventory'] = grouped['inventory_level'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # 4. Price Change Percentage
    # Calculate previous day's price, find the % change, and handle division by zero
    df['lag_1_day_price'] = grouped['price'].shift(1)
    df['price_change_pct'] = np.where(
        df['lag_1_day_price'].notna() & (df['lag_1_day_price'] > 0),
        (df['price'] - df['lag_1_day_price']) / df['lag_1_day_price'],
        0
    )
    df = df.drop(columns=['lag_1_day_price']) # Drop the temporary column
    
    # 5. Inventory Cover Days (KPI)
    # How many days the current inventory will last based on the 7-day sales trend
    df['inventory_cover_days'] = np.where(
        df['rolling_7_day_avg_sales'] > 0, 
        df['inventory_level'] / df['rolling_7_day_avg_sales'], 
        0
    )
    
    # Drop rows with NaN values introduced by the shift() functions (specifically lag_7_day_sales)
    # This will remove the first 7 days of data for every product/store combination
    df = df.dropna().reset_index(drop=True)
    
    return df

def execute_pipeline():
    """Executes the feature engineering pipeline and saves the output."""
    print("Starting feature engineering process...")
    
    df = load_cleaned_data()
    df = create_date_features(df)
    df = create_time_series_features(df)
    
    # Ensure the output directory exists before saving
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature engineering complete. Saved to {OUTPUT_PATH}")
    print(f"Final dataset shape ready for modeling: {df.shape}")

if __name__ == "__main__":
    execute_pipeline()