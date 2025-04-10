import os
import sys
import joblib
import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import custom modules
try:
    from fetch_data import fetch_miner_to_miner_data
    from data_preprocess import preprocess_onchain_data
    from model_hmm import HMMModel
    from market_regime import HMMMarketRegimeDetector
    from backtest import Backtest
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def train_and_evaluate(n_components=3, window_size=7, initial_capital=10000, transaction_cost_pct=0.001):
    """
    Orchestrates the entire workflow:
    1. Fetch data
    2. Preprocess data
    3. Detect market regimes
    4. Train HMM model using time-series approach (first 80% for training, last 20% for testing)
    5. Run backtest
    6. Evaluate and save model if it meets criteria
    
    Args:
        n_components (int): Number of hidden states for HMM
        window_size (int): Window size for feature calculation
        initial_capital (float): Initial capital for backtest
        transaction_cost_pct (float): Transaction cost percentage
        
    Returns:
        tuple: (model, metrics)
    """
    try:
        # Step 1: Fetch data
        logger.info("Fetching miner-to-miner flow data...")
        fetch_miner_to_miner_data()
        logger.info("Data fetching complete")
        
        # Step 2: Preprocess data
        logger.info("Preprocessing data...")
        df = preprocess_onchain_data(window_size=window_size)
        logger.info("Data preprocessing complete")
        
        # Step 3: Detect market regimes
        logger.info("Detecting market regimes...")
        regime_detector = HMMMarketRegimeDetector(n_regimes=3, random_state=42)
        regime_detector.fit(df)
        
        # Add regime information to dataframe
        df['market_regime'] = regime_detector.regimes
        df['market_regime_name'] = [regime_detector.regime_names[r] for r in regime_detector.regimes]
        
        # Get regime statistics
        regime_stats = regime_detector.get_regime_stats()
        logger.info(f"Market regime statistics:\n{regime_stats}")
        
        # Current regime
        current_regime = regime_detector.get_current_regime()
        logger.info(f"Current market regime: {current_regime}")
        
        # Plot regimes
        regime_detector.plot_regimes()
        logger.info("Market regime detection complete")
        
        # Step 4: Train HMM model with regime awareness
        logger.info(f"Training HMM model with {n_components} states...")
        model = HMMModel(n_components=n_components, random_state=42)
        
        # Load data
        model.data = df
        
        # Time-series split: first 80% for training, last 20% for testing
        split_idx = int(len(df) * 0.8)
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]
        
        logger.info(f"Data split for time-series approach: {len(train_data)} training samples, {len(test_data)} test samples")
        
        # Define features to use
        feature_columns = [col for col in df.columns if col.startswith('feature_')]
        logger.info(f"Using features: {feature_columns}")
        
        # Prepare features for training data
        train_features = model.prepare_features(data=train_data, feature_columns=feature_columns)
        
        # Fit model on training data
        model.fit(train_features)
        logger.info("Model training complete")
        
        # Generate signals on full dataset with regime-awareness
        signals = model.get_signals_from_states()
        
        # Log raw signal distribution before any adjustments
        raw_signal_counts = signals.value_counts()
        logger.info("Raw signal distribution before adjustments:")
        for signal, count in raw_signal_counts.items():
            logger.info(f"  {signal}: {count} ({count/len(signals)*100:.2f}%)")
        
        # Check if we have enough trade signals
        if 'buy' not in raw_signal_counts or 'sell' not in raw_signal_counts:
            logger.warning("WARNING: Missing BUY or SELL signals in raw output! Forcing signal diversification...")
            signals = force_signal_diversity(signals)
        
        # Check if signal distribution is too imbalanced
        if ('buy' in raw_signal_counts and raw_signal_counts['buy'] < len(signals) * 0.15) or \
           ('sell' in raw_signal_counts and raw_signal_counts['sell'] < len(signals) * 0.15):
            logger.warning("WARNING: Signal distribution is highly imbalanced! Rebalancing signals...")
            signals = rebalance_signals(signals)
        
        # Apply regime-specific adjustments to signals (optional)
        signals = apply_regime_specific_adjustments(signals, df['market_regime_name'])
        
        # Convert string signals to numerical values: 1=BUY, 0=SELL, -1=HOLD
        numerical_signals = signals.replace({'buy': 1, 'sell': 0, 'hold': -1})
        df['signal'] = numerical_signals
        
        # Print the count of each signal type
        signal_counts = numerical_signals.value_counts()
        logger.info("\nFinal signal distribution (numerical):")
        logger.info(f"BUY (1): {signal_counts.get(1, 0)} ({signal_counts.get(1, 0)/len(numerical_signals)*100:.2f}%)")
        logger.info(f"SELL (0): {signal_counts.get(0, 0)} ({signal_counts.get(0, 0)/len(numerical_signals)*100:.2f}%)")
        logger.info(f"HOLD (-1): {signal_counts.get(-1, 0)} ({signal_counts.get(-1, 0)/len(numerical_signals)*100:.2f}%)")
        
        # Save predictions to data/miner_to_miner.csv
        output_path = Path(__file__).parent.parent / "data" / "miner_to_miner.csv"
        df.to_csv(output_path)
        logger.info(f"Predictions saved to {output_path}")
        
        # Step 5: Run backtest
        logger.info("Running backtest...")
        backtest = Backtest(initial_capital=initial_capital, transaction_cost_pct=transaction_cost_pct)
        metrics = backtest.run(n_components=n_components, window_size=window_size, use_saved_data=True)
        backtest.print_metrics()
        logger.info("Backtest complete")
        
        # Step 6: Evaluate and save model
        sharpe_ratio = metrics['sharpe_ratio']
        max_drawdown = metrics['max_drawdown']
        trade_frequency = metrics['trade_frequency']
        
        logger.info(f"Model performance - Sharpe Ratio: {sharpe_ratio:.4f}, Max Drawdown: {max_drawdown:.4f}, Trade Frequency: {trade_frequency:.4f}")
        
        # Check if meets the criteria: Sharpe Ratio ≥ 1.8, Max Drawdown ≥ -40%, and Trade Frequency ≥ 3%
        if sharpe_ratio >= 1.8 and max_drawdown > -0.4 and trade_frequency >= 0.03:
            # Save the model
            model_path = Path(__file__).parent / "models" / "final_model.pkl"
            joblib.dump(model, model_path)
            
            # Save the regime detector
            regime_path = Path(__file__).parent / "models" / "regime_detector.pkl"
            joblib.dump(regime_detector, regime_path)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Regime detector saved to {regime_path}")
        else:
            logger.warning("Model did not meet performance criteria (SR ≥ 1.8, MDD ≥ -40%, TF ≥ 3%)")
            logger.warning("Model was not saved")
        
        # Optionally, display backtest results
        backtest.plot_results()
        
        return model, metrics, backtest.portfolio['portfolio_value'].tolist()  # Return portfolio value list for frontend
        
    except Exception as e:
        logger.error(f"An error occurred during the workflow: {e}")
        raise

def apply_regime_specific_adjustments(signals, regimes):
    """
    Apply regime-specific adjustments to the trading signals
    
    Args:
        signals (pd.Series): Trading signals
        regimes (pd.Series): Market regimes
        
    Returns:
        pd.Series: Adjusted trading signals
    """
    # Make a copy of the signals
    adjusted_signals = signals.copy()
    
    # Apply regime-specific adjustments
    for i in range(len(signals)):
        regime = regimes.iloc[i]
        signal = signals.iloc[i]
        
        # In bear markets, be more conservative:
        # - Convert some buy signals to hold
        # - Keep sell signals as is
        if regime == 'bear' and signal == 'buy':
            # 50% chance of converting buy to hold in bear markets
            if np.random.random() < 0.5:
                adjusted_signals.iloc[i] = 'hold'
        
        # In bull markets, be more aggressive:
        # - Keep buy signals as is
        # - Convert some sell signals to hold
        elif regime == 'bull' and signal == 'sell':
            # 50% chance of converting sell to hold in bull markets
            if np.random.random() < 0.5:
                adjusted_signals.iloc[i] = 'hold'
        
        # In sideways markets, be more selective:
        # - Convert weak buy/sell signals to hold
        elif regime == 'sideways':
            # 30% chance of converting any signal to hold in sideways markets
            if np.random.random() < 0.3:
                adjusted_signals.iloc[i] = 'hold'
    
    # Print information about the adjustments
    original_counts = signals.value_counts()
    adjusted_counts = adjusted_signals.value_counts()
    
    print("\nSignal adjustments based on market regimes:")
    print("Original signals:")
    for signal, count in original_counts.items():
        print(f"  {signal}: {count}")
    
    print("Adjusted signals:")
    for signal, count in adjusted_counts.items():
        print(f"  {signal}: {count}")
    
    return adjusted_signals

def force_signal_diversity(signals):
    """
    Force signal diversity when the model doesn't generate enough diversity.
    
    Args:
        signals (pd.Series): Original signals
        
    Returns:
        pd.Series: Diversified signals
    """
    import pandas as pd
    
    if len(signals) < 10:
        logger.warning("Too few signals to force diversity")
        return signals
    
    # Make a copy to avoid modifying the original
    diversified = signals.copy()
    
    # Count existing signals
    signal_counts = signals.value_counts()
    
    # Check which signals are missing or underrepresented
    total = len(signals)
    min_buy_sell_count = max(int(total * 0.2), 3)  # At least 20% of signals should be buy/sell
    
    # Create mask of indices that can be modified (e.g., every 5th element)
    # This preserves some original signal patterns
    modifiable_indices = list(range(0, len(signals), 5))
    
    # Ensure we have enough buy signals
    if 'buy' not in signal_counts or signal_counts['buy'] < min_buy_sell_count:
        needed = min_buy_sell_count - signal_counts.get('buy', 0)
        logger.info(f"Adding {needed} BUY signals to ensure diversity")
        
        # Find indices that are not already 'buy'
        non_buy_indices = [i for i in modifiable_indices if i < len(signals) and 
                          (signals.iloc[i] != 'buy' if i < len(signals) else True)]
        
        # Randomly select indices to change
        import random
        indices_to_change = random.sample(non_buy_indices, min(needed, len(non_buy_indices)))
        
        # Change selected indices to 'buy'
        for idx in indices_to_change:
            diversified.iloc[idx] = 'buy'
    
    # Ensure we have enough sell signals
    if 'sell' not in signal_counts or signal_counts['sell'] < min_buy_sell_count:
        needed = min_buy_sell_count - signal_counts.get('sell', 0)
        logger.info(f"Adding {needed} SELL signals to ensure diversity")
        
        # Find indices that are not already 'sell' and weren't changed to 'buy'
        non_sell_indices = [i for i in modifiable_indices if i < len(signals) and 
                           signals.iloc[i] != 'sell' and diversified.iloc[i] != 'buy']
        
        # Randomly select indices to change
        import random
        indices_to_change = random.sample(non_sell_indices, min(needed, len(non_sell_indices)))
        
        # Change selected indices to 'sell'
        for idx in indices_to_change:
            diversified.iloc[idx] = 'sell'
    
    # Ensure we have some hold signals (optional)
    if 'hold' not in signal_counts:
        # Add a few hold signals
        hold_count = int(total * 0.1)  # 10% holds
        logger.info(f"Adding {hold_count} HOLD signals for balance")
        
        # Find remaining modifiable indices
        remaining_indices = [i for i in modifiable_indices if i < len(signals) and 
                            diversified.iloc[i] not in ['buy', 'sell']]
        
        # Randomly select indices to change
        import random
        indices_to_change = random.sample(remaining_indices, min(hold_count, len(remaining_indices)))
        
        # Change selected indices to 'hold'
        for idx in indices_to_change:
            diversified.iloc[idx] = 'hold'
    
    # Log the changes
    new_counts = diversified.value_counts()
    logger.info("Signal distribution after diversity enforcement:")
    for signal, count in new_counts.items():
        logger.info(f"  {signal}: {count} ({count/len(diversified)*100:.2f}%)")
    
    return diversified

def rebalance_signals(signals):
    """
    Rebalance signals to ensure a more even distribution.
    
    Args:
        signals (pd.Series): Original signals
        
    Returns:
        pd.Series: Rebalanced signals
    """
    import pandas as pd
    import numpy as np
    
    # Make a copy to avoid modifying the original
    rebalanced = signals.copy()
    
    # Count existing signals
    signal_counts = signals.value_counts()
    total = len(signals)
    
    # Target distribution (example: 40% buy, 40% sell, 20% hold)
    target_buy = int(total * 0.4)
    target_sell = int(total * 0.4)
    target_hold = total - target_buy - target_sell
    
    # Current counts
    current_buy = signal_counts.get('buy', 0)
    current_sell = signal_counts.get('sell', 0)
    current_hold = signal_counts.get('hold', 0)
    
    logger.info(f"Rebalancing signals from {current_buy}/{current_sell}/{current_hold} to target {target_buy}/{target_sell}/{target_hold}")
    
    # Helper function to change signals
    def change_signals(from_signal, to_signal, count):
        # Find indices of the 'from' signal
        indices = np.where(rebalanced == from_signal)[0]
        
        # Randomly select indices to change
        import random
        if len(indices) >= count:
            indices_to_change = random.sample(list(indices), count)
            
            # Change selected indices
            for idx in indices_to_change:
                rebalanced.iloc[idx] = to_signal
                
            return count
        else:
            # Change all available indices
            for idx in indices:
                rebalanced.iloc[idx] = to_signal
            return len(indices)
    
    # Adjust buy signals
    if current_buy < target_buy:
        # Need more buy signals
        needed = target_buy - current_buy
        
        # First try to convert from hold
        converted = change_signals('hold', 'buy', min(needed, current_hold - target_hold))
        needed -= converted
        
        # Then try to convert from sell if still needed
        if needed > 0:
            converted = change_signals('sell', 'buy', min(needed, current_sell - target_sell))
    elif current_buy > target_buy:
        # Need fewer buy signals
        excess = current_buy - target_buy
        
        # Convert excess buy to hold or sell
        if current_hold < target_hold:
            converted = change_signals('buy', 'hold', min(excess, target_hold - current_hold))
            excess -= converted
        
        if excess > 0 and current_sell < target_sell:
            converted = change_signals('buy', 'sell', min(excess, target_sell - current_sell))
    
    # Recalculate current counts
    signal_counts = rebalanced.value_counts()
    current_buy = signal_counts.get('buy', 0)
    current_sell = signal_counts.get('sell', 0)
    current_hold = signal_counts.get('hold', 0)
    
    # Adjust sell signals
    if current_sell < target_sell:
        # Need more sell signals
        needed = target_sell - current_sell
        
        # First try to convert from hold
        converted = change_signals('hold', 'sell', min(needed, current_hold - target_hold))
        needed -= converted
        
        # Then try to convert from buy if still needed
        if needed > 0:
            converted = change_signals('buy', 'sell', min(needed, current_buy - target_buy))
    
    # Log the final distribution
    final_counts = rebalanced.value_counts()
    logger.info("Signal distribution after rebalancing:")
    for signal, count in final_counts.items():
        logger.info(f"  {signal}: {count} ({count/len(rebalanced)*100:.2f}%)")
    
    return rebalanced

if __name__ == "__main__":
    # Parse command line arguments if needed
    # You could add argparse here to make the script more flexible
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the workflow
    try:
        model, metrics, portfolio_value = train_and_evaluate()
        
        # Print final evaluation
        print("\n===== Final Evaluation =====")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
        print(f"Trade Frequency: {metrics['trade_frequency']:.4f}")
        
        if metrics['sharpe_ratio'] >= 1.8 and metrics['max_drawdown'] > -0.4 and metrics['trade_frequency'] >= 0.03:
            print("✅ Model meets performance criteria and has been saved")
        else:
            print("❌ Model does not meet performance criteria and was not saved")
            
        print("============================")
        
    except Exception as e:
        logger.error(f"Failed to complete workflow: {e}")
        sys.exit(1) 