#!/usr/bin/env python
"""
Run Simple Backtest Script

This script uses test data with pre-generated signals to run a backtest
with the BacktestFramework.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from backtest_framework import BacktestFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data():
    """
    Load test data with pre-generated signals
    
    Returns:
        pd.DataFrame: Data with signals
    """
    test_data_path = Path(__file__).parent / "test_data.csv"
    
    if not test_data_path.exists():
        logger.error(f"Test data file not found at {test_data_path}")
        logger.info("Run create_mock_data.py first to generate test data")
        raise FileNotFoundError(f"Test data file not found at {test_data_path}")
    
    logger.info(f"Loading test data from {test_data_path}")
    data = pd.read_csv(test_data_path)
    
    # Debug: Print signal distribution
    signal_counts = data['signal'].value_counts()
    logger.info(f"Signal distribution: {signal_counts}")
    
    return data

def run_backtest(data):
    """
    Run the backtest using BacktestFramework
    
    Args:
        data (pd.DataFrame): Data with signals
    
    Returns:
        dict: Backtest metrics
    """
    # Initialize BacktestFramework
    backtest = BacktestFramework(data=data, signal_column='signal', price_column='price')
    
    # Run the backtest
    backtest.run()
    
    # Debug: Print portfolio history stats
    logger.info(f"Portfolio history length: {len(backtest.portfolio_history)}")
    logger.info(f"Initial portfolio value: {backtest.portfolio_history[0]}")
    logger.info(f"Final portfolio value: {backtest.portfolio_history[-1]}")
    
    # Get metrics
    metrics = backtest.metrics()
    
    # Debug: Print raw metrics
    logger.info(f"Raw metrics: {metrics}")
    
    return metrics

def print_results(metrics):
    """
    Print backtest results in a clear format
    
    Args:
        metrics (dict): Backtest metrics
    """
    print("\n" + "="*50)
    print(" "*15 + "BACKTEST RESULTS")
    print("="*50)
    
    for key, value in metrics.items():
        if key == "Max Drawdown":
            print(f"{key}: -{value}%")
        elif key == "Trade Frequency":
            print(f"{key}: {value}%")
        elif key == "Final Portfolio Value":
            print(f"{key}: ${value:,.2f}")
        else:
            print(f"{key}: {value}")
    
    print("="*50)

def main():
    """Main function to run the backtest"""
    try:
        # Create mock data if needed
        try:
            import create_mock_data
            logger.info("Generated mock data")
        except Exception as e:
            logger.info(f"Using existing mock data: {e}")
        
        # Load test data
        logger.info("Loading test data...")
        data = load_test_data()
        
        # Run backtest
        logger.info("Running backtest...")
        metrics = run_backtest(data)
        
        # Print results
        print_results(metrics)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 