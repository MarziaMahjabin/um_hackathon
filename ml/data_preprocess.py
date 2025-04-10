import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_onchain_data(window_size=7):
    """
    Preprocesses miner-to-miner flow data:
    - Aligns timestamps
    - Fills missing values with forward fill
    - Creates features: rolling average, percent change, rolling std
    
    Args:
        window_size (int): Size of the rolling window in days
        
    Returns:
        pd.DataFrame: Processed dataframe with feature columns
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "miner_to_miner.csv"
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded miner-to-miner data with {len(df)} rows")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    # Ensure datetime column is in datetime format
    date_column = next((col for col in df.columns if any(date_term in col.lower() for date_term in ['time', 'date', 'timestamp'])), None)
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    else:
        print("Warning: No date column found, using row index instead")
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Identify value column (assuming it's the one with 'value', 'amount', or other numeric data)
    value_columns = [col for col in df.columns if any(term in col.lower() for term in ['value', 'amount', 'btc', 'flow'])]
    
    if not value_columns:
        # If no obvious value column, use all numeric columns
        value_columns = df.select_dtypes(include=['number']).columns.tolist()
        
    print(f"Using value columns: {value_columns}")
    
    # Create features for each value column
    for col in value_columns:
        # 1. Rolling average
        df[f'feature_{col}_rolling_avg'] = df[col].rolling(window=window_size).mean()
        
        # 2. Percent change
        df[f'feature_{col}_pct_change'] = df[col].pct_change() * 100
        
        # 3. Rolling standard deviation (volatility)
        df[f'feature_{col}_rolling_std'] = df[col].rolling(window=window_size).std()
        
        # 4. Z-score (number of standard deviations from the mean)
        rolling_mean = df[col].rolling(window=window_size).mean()
        rolling_std = df[col].rolling(window=window_size).std()
        df[f'feature_{col}_z_score'] = (df[col] - rolling_mean) / rolling_std
    
    # Assuming we have some price data, if not, we'll need to add it from another source
    if 'price' not in df.columns:
        # For now, we'll use a value column as a proxy if no price data
        # In a real system, you'd want to join with actual price data
        df['price'] = df[value_columns[0]] if value_columns else 1.0
    
    # Fill NaN values that may be created by rolling calculations
    df.fillna(method='ffill', inplace=True)
    # Fill any remaining NaNs with 0
    df.fillna(0, inplace=True)
    
    print(f"Created preprocessed dataset with {len(df)} rows and {len(df.columns)} columns")
    return df

def get_preprocessed_data(window_size=7):
    """
    Returns preprocessed data for modeling
    
    Args:
        window_size (int): Size of the rolling window in days
        
    Returns:
        pd.DataFrame: Processed dataframe with feature columns
    """
    return preprocess_onchain_data(window_size)

if __name__ == "__main__":
    # If run directly, preprocess the data and show some info
    preprocessed_data = preprocess_onchain_data()
    print("\nFeature columns:")
    feature_cols = [col for col in preprocessed_data.columns if col.startswith('feature_')]
    for col in feature_cols:
        print(f"- {col}")
    
    # Display a sample of the data
    print("\nSample of preprocessed data:")
    print(preprocessed_data.head()) 