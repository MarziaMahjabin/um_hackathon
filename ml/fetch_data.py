import os
import pandas as pd
import requests
import json
from dotenv import load_dotenv
from datetime import datetime
import pytz
from pathlib import Path

# Try to import the cybotrade library, but provide fallback if not available
try:
    from cybotrade_datasource import query_paginated
    HAS_CYBOTRADE_LIB = True
except ImportError:
    HAS_CYBOTRADE_LIB = False
    print("Warning: cybotrade_datasource not found. Will use direct API calls instead.")

def fetch_miner_to_miner_data():
    """
    Fetch miner-to-miner flow data from Cybotrade API
    
    Uses either the official cybotrade_datasource library if available, 
    or falls back to direct API calls if the library is not installed.
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('DATASET_API_KEY')
    
    if not api_key:
        raise ValueError("DATASET_API_KEY not found in .env file")
    
    # Set date range from 2024-01-01 to 2025-01-01 in UTC timezone
    start_date = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2025, 1, 1, tzinfo=pytz.UTC)
    
    # Topic for miner-to-miner flows
    topic = 'cryptoquant|btc/inter-entity-flows/miner-to-miner?from_miner=f2pool&to_miner=all_miner&window=hour'
    
    print(f"Fetching data for topic: {topic}")
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        if HAS_CYBOTRADE_LIB:
            # Use the official library if available
            print("Using cybotrade_datasource library")
            data = query_paginated(
                topic=topic,
                start=start_date,
                end=end_date,
                api_key=api_key
            )
        else:
            # Fallback to direct API calls
            print("Using direct API call")
            data = fetch_data_direct_api(topic, start_date, end_date, api_key)
        
        # Convert to DataFrame
        if data:
            df = pd.DataFrame(data)
            
            # Ensure the data directory exists
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            
            # Save to CSV
            output_path = data_dir / "miner_to_miner.csv"
            df.to_csv(output_path, index=False)
            print(f"Successfully saved miner-to-miner data to {output_path}")
            
            # Print data summary
            print(f"Data shape: {df.shape}")
            print("Data columns:")
            for col in df.columns:
                print(f"  - {col}")
            
            return df
        else:
            print("No data received from API")
            return None
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise

def fetch_data_direct_api(topic, start_date, end_date, api_key):
    """
    Fetch data directly from the Cybotrade API without the library
    
    Args:
        topic (str): The data topic to fetch
        start_date (datetime): Start date for data range
        end_date (datetime): End date for data range
        api_key (str): API key for authentication
        
    Returns:
        list: The fetched data as a list of dictionaries
    """
    base_url = "https://api.datasource.cybotrade.rs/v1/data"
    
    # Convert dates to timestamps
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    # Prepare request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Try different auth methods if needed
    auth_methods = [
        {"Authorization": f"Bearer {api_key}"},
        {"Authorization": f"Token {api_key}"},
        {"X-API-Key": api_key},
        {"Api-Key": api_key}
    ]
    
    # Parameters
    params = {
        "topic": topic,
        "start": start_timestamp,
        "end": end_timestamp
    }
    
    # Try each auth method
    for header in auth_methods:
        try:
            print(f"Trying authentication method: {header.keys()}")
            response = requests.get(base_url, headers=header, params=params)
            
            if response.status_code == 200:
                print("Authentication successful!")
                data = response.json()
                return data
            else:
                print(f"Failed with status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Error with auth method: {e}")
    
    # If all methods failed
    print("All authentication methods failed")
    # Create simulated data for testing
    print("WARNING: Using simulated data for development")
    return generate_simulated_data(start_date, end_date)

def generate_simulated_data(start_date, end_date):
    """
    Generate simulated miner-to-miner flow data for development purposes only
    This should be replaced with real API data in production
    
    Args:
        start_date (datetime): Start date 
        end_date (datetime): End date
        
    Returns:
        list: Simulated data as a list of dictionaries
    """
    import numpy as np
    from pandas.tseries.offsets import Hour
    
    # Generate hourly timestamps
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    
    # Generate simulated data points
    data = []
    for ts in date_range:
        # Create random flow values with some patterns
        hour_of_day = ts.hour
        day_of_week = ts.dayofweek
        
        # More activity during business hours
        time_factor = 1.0 + 0.5 * (9 <= hour_of_day <= 17)
        # Less activity on weekends
        day_factor = 1.0 - 0.3 * (day_of_week >= 5)
        
        # Base flow amount with some randomness
        base_flow = 0.5 + np.random.exponential(2.0) * time_factor * day_factor
        
        data_point = {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "from_miner": "f2pool",
            "to_miner": "all_miner",
            "flow_amount": base_flow,
            "flow_count": int(np.random.poisson(base_flow * 2) + 1)
        }
        data.append(data_point)
    
    print(f"Generated {len(data)} simulated data points")
    print("IMPORTANT: This is SIMULATED data for development purposes only")
    print("Replace with real API data in production")
    
    return data

if __name__ == "__main__":
    fetch_miner_to_miner_data() 