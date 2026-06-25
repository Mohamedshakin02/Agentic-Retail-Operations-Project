import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Paths
INPUT_PATH = Path("data/processed/retail_features.csv")
MODEL_DIR = Path("data/outputs")
MODEL_PATH = MODEL_DIR / "rf_forecasting_model.pkl"

def load_features():
    """Loads the engineered features dataset."""
    df = pd.read_csv(INPUT_PATH, parse_dates=['date'])
    # Sort by date to ensure our train/test split is strictly chronological
    df = df.sort_values(by='date').reset_index(drop=True)
    return df

def calculate_wmape(actual, forecast):
    """Calculates Weighted Mean Absolute Percentage Error (WMAPE)."""
    return np.sum(np.abs(actual - forecast)) / np.sum(actual)

def train_and_evaluate(df):
    """Splits data, trains the Random Forest, and calculates metrics."""
    
    # Define the features (X) and the target variable (y)
    # We predict 'units_sold'. We drop columns that are identifiers, dates, or the target itself.
    drop_cols = ['date', 'store_id', 'product_id', 'category', 'region', 
                 'weather_condition', 'seasonality', 'units_sold', 'dataset_demand_forecast',
                 'units_sold_log', 'demand_forecast_log']
    
    X = df.drop(columns=drop_cols)
    y = df['units_sold']

    # Chronological Train-Test Split (80% Train, 20% Test)
    split_index = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training on {len(X_train)} records, Testing on {len(X_test)} records...")

    # Initialize and train the Random Forest model
    # Using 100 trees and a fixed random state for reproducibility
    # model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # model.fit(X_train, y_train)

    # Initialize XGBoost with some basic tuning for time-series
    model = XGBRegressor(
        n_estimators=300,        # More trees
        learning_rate=0.05,      # Learn slower and more carefully
        max_depth=6,             # Prevent overfitting
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Generate predictions on the test set
    predictions = model.predict(X_test)

    # Calculate KPIs defined in Module 3 / Module 5
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    wmape = calculate_wmape(y_test.values, predictions)
    
    # Calculate Business Accuracy (100% - WMAPE)
    accuracy_percentage = 1 - wmape

    print("\n--- Model Evaluation Metrics ---")
    print(f"MAE (Mean Absolute Error): {mae:.2f} units")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} units")
    print(f"WMAPE: {wmape:.2%}")
    print(f"Business Accuracy: {accuracy_percentage:.2%}") # This prints the new score

    return model

def save_model(model):
    """Saves the trained model to the outputs directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    print("Starting Model Training Pipeline...")
    df = load_features()
    trained_model = train_and_evaluate(df)
    save_model(trained_model)