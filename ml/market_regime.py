"""
Market Regime Detection Module

This module provides a class for detecting market regimes (bull, bear, sideways)
using Hidden Markov Models.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from pathlib import Path

class HMMMarketRegimeDetector:
    """
    Hidden Markov Model-based Market Regime Detector
    
    This class identifies different market regimes (bull, bear, sideways)
    by training a Hidden Markov Model on price and volume features.
    """
    
    def __init__(self, n_regimes=3, random_state=42, lookback_window=30):
        """
        Initialize the regime detector
        
        Args:
            n_regimes (int): Number of market regimes to detect (default: 3)
            random_state (int): Random seed for reproducibility
            lookback_window (int): Number of days to look back for regime features
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.lookback_window = lookback_window
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            random_state=random_state,
            n_iter=100
        )
        self.regimes = None
        self.regime_names = {
            0: "bear",   # Typically lowest returns
            1: "sideways", # Medium returns
            2: "bull"    # Highest returns
        }
        self.data = None
        self.regime_features = ["returns", "volatility", "volume_change", "whale_net_movement"]
    
    def prepare_features(self, data):
        """
        Prepare features for regime detection
        
        Args:
            data (pd.DataFrame): DataFrame with price, volume, and whale data
            
        Returns:
            np.ndarray: Feature matrix for HMM
        """
        df = data.copy()
        
        # Calculate returns
        if 'price' in df.columns:
            df['returns'] = df['price'].pct_change()
            
            # Calculate volatility (rolling standard deviation of returns)
            df['volatility'] = df['returns'].rolling(window=self.lookback_window).std()
        else:
            # If no price column, try to use a numeric column as a proxy
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                proxy_col = numeric_cols[0]
                print(f"No price column found. Using {proxy_col} as a proxy for price.")
                df['returns'] = df[proxy_col].pct_change()
                df['volatility'] = df['returns'].rolling(window=self.lookback_window).std()
            else:
                raise ValueError("No numeric columns found to calculate returns and volatility")
        
        # Calculate volume change
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'].pct_change()
        elif 'transaction_count' in df.columns:
            df['volume_change'] = df['transaction_count'].pct_change()
        elif 'flow_count' in df.columns:
            df['volume_change'] = df['flow_count'].pct_change()
        
        # Calculate whale net movement (or equivalent)
        if 'large_inflow' in df.columns and 'large_outflow' in df.columns:
            df['whale_net_movement'] = df['large_inflow'] - df['large_outflow']
        elif 'inflow' in df.columns and 'outflow' in df.columns:
            df['whale_net_movement'] = df['inflow'] - df['outflow']
        elif 'flow_amount' in df.columns:
            # For miner flow data, use the flow amount as a proxy for net movement
            df['whale_net_movement'] = df['flow_amount']
        
        # Fill NaN values instead of dropping them
        df.fillna(method='ffill', inplace=True)  # Forward fill
        df.fillna(method='bfill', inplace=True)  # Backward fill for any remaining NaNs at the beginning
        df.fillna(0, inplace=True)  # Replace any remaining NaNs with zeros
        
        # Select features for regime detection
        features = []
        for feature in self.regime_features:
            if feature in df.columns:
                features.append(feature)
        
        if not features:
            raise ValueError("No valid features found for regime detection")
        
        print(f"Using features for regime detection: {features}")
        
        # Extract feature matrix
        X = df[features].values
        
        # Standardize features
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        # Avoid division by zero
        stds[stds == 0] = 1.0
        X = (X - means) / stds
        
        self.data = df
        return X, features
    
    def fit(self, data):
        """
        Fit the HMM model to detect regimes
        
        Args:
            data (pd.DataFrame): DataFrame with price and volume data
            
        Returns:
            np.ndarray: Detected regimes
        """
        # Prepare features
        X, used_features = self.prepare_features(data)
        
        # Fit the model
        self.model.fit(X)
        
        # Predict regimes
        self.regimes = self.model.predict(X)
        
        # Map regimes to meaningful names based on average returns in each regime
        self._label_regimes()
        
        # Add regime column to data
        self.data['regime'] = self.regimes
        self.data['regime_name'] = [self.regime_names[r] for r in self.regimes]
        
        return self.regimes
    
    def _label_regimes(self):
        """
        Label regimes as bull, bear, or sideways based on average returns
        """
        # Try to use returns if available
        if 'returns' in self.data.columns:
            feature = 'returns'
        # Otherwise, use a proxy feature that might indicate market direction
        elif 'flow_amount' in self.data.columns:
            feature = 'flow_amount'
        else:
            # Use the first numeric column as a proxy
            numeric_cols = [col for col in self.data.columns if col not in ['regime', 'regime_name']]
            if numeric_cols:
                feature = numeric_cols[0]
                print(f"Using {feature} as proxy for labeling regimes")
            else:
                print("Warning: Cannot label regimes without suitable numeric data")
                return
        
        # Calculate average value for each regime
        regime_values = {}
        for regime in range(self.n_regimes):
            mask = self.regimes == regime
            if np.any(mask):
                regime_values[regime] = self.data.loc[mask, feature].mean()
        
        # Sort regimes by value
        sorted_regimes = sorted(regime_values.items(), key=lambda x: x[1])
        
        # Assign labels based on values
        if len(sorted_regimes) == 3:
            # Bear = lowest values, Bull = highest values, Sideways = middle
            self.regime_names = {
                sorted_regimes[0][0]: "bear",
                sorted_regimes[1][0]: "sideways",
                sorted_regimes[2][0]: "bull"
            }
        elif len(sorted_regimes) == 2:
            # Two regimes: bear and bull
            self.regime_names = {
                sorted_regimes[0][0]: "bear",
                sorted_regimes[1][0]: "bull"
            }
        
        print(f"Regimes labeled based on {feature}:")
        for regime, name in self.regime_names.items():
            print(f"  Regime {regime} -> {name.capitalize()}: " 
                  f"Avg {feature} = {regime_values.get(regime, 'N/A'):.6f}")
    
    def predict(self, data=None):
        """
        Predict regimes for new data
        
        Args:
            data (pd.DataFrame, optional): New data to predict regimes for
            
        Returns:
            list: Predicted regime names
        """
        if data is None:
            if self.regimes is None:
                raise ValueError("Model has not been fitted yet")
            return [self.regime_names[r] for r in self.regimes]
        
        # Prepare features for new data
        X, _ = self.prepare_features(data)
        
        # Predict regimes
        regimes = self.model.predict(X)
        
        # Map to regime names
        regime_names = [self.regime_names.get(r, f"unknown_{r}") for r in regimes]
        
        return regime_names
    
    def plot_regimes(self):
        """
        Plot price/value chart with colored regimes
        """
        if self.data is None or self.regimes is None:
            raise ValueError("Model has not been fitted yet")
        
        # Find a suitable value column to plot
        if 'price' in self.data.columns:
            value_col = 'price'
            value_label = 'Price'
        elif 'flow_amount' in self.data.columns:
            value_col = 'flow_amount'
            value_label = 'Flow Amount'
        else:
            # Use first numeric column that's not a feature or regime
            numeric_cols = [col for col in self.data.select_dtypes(include=['number']).columns
                           if not col.startswith('feature_') and col not in ['regime', 'returns', 'volatility']]
            if numeric_cols:
                value_col = numeric_cols[0]
                value_label = value_col.replace('_', ' ').title()
            else:
                raise ValueError("No suitable value column found for plotting")
        
        plt.figure(figsize=(12, 8))
        
        # Plot value
        ax = plt.subplot(2, 1, 1)
        
        # Map regimes to colors
        colors = {
            "bull": "green",
            "bear": "red",
            "sideways": "gray"
        }
        
        # Plot value with regime background colors
        for regime in self.regime_names.values():
            mask = self.data['regime_name'] == regime
            if np.any(mask):
                plt.plot(self.data.index[mask], self.data.loc[mask, value_col], 
                         color=colors[regime], label=f"{regime.capitalize()} Regime")
        
        plt.title(f"{value_label} with Regime Detection")
        plt.ylabel(value_label)
        plt.legend()
        plt.grid(True)
        
        # Plot regime timeline
        ax2 = plt.subplot(2, 1, 2)
        regime_numeric = self.data['regime_name'].map({"bull": 2, "sideways": 1, "bear": 0})
        plt.plot(self.data.index, regime_numeric)
        plt.yticks([0, 1, 2], ["Bear", "Sideways", "Bull"])
        plt.title("Market Regimes Over Time")
        plt.ylabel("Regime")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def get_current_regime(self):
        """
        Get the current market regime
        
        Returns:
            str: Current regime name (bull, bear, or sideways)
        """
        if self.regimes is None:
            raise ValueError("Model has not been fitted yet")
        
        current_regime = self.regimes[-1]
        return self.regime_names[current_regime]
    
    def get_regime_stats(self):
        """
        Get statistics for each regime
        
        Returns:
            pd.DataFrame: Statistics for each regime
        """
        if self.data is None or self.regimes is None:
            raise ValueError("Model has not been fitted yet")
        
        # Columns to include in stats
        stats_columns = {
            'returns': 'Avg Return',
            'volatility': 'Volatility',
            'volume_change': 'Volume Change',
            'whale_net_movement': 'Net Movement',
            'flow_amount': 'Flow Amount',
            'flow_count': 'Flow Count'
        }
        
        stats = []
        for regime_num, regime_name in self.regime_names.items():
            mask = self.regimes == regime_num
            if np.any(mask):
                regime_data = self.data[mask]
                
                # Start with base stats
                stat_row = {
                    "Regime": regime_name.capitalize(),
                    "Count": len(regime_data),
                    "Percentage": len(regime_data) / len(self.data) * 100
                }
                
                # Add available metric stats
                for col, label in stats_columns.items():
                    if col in regime_data:
                        # For percentage-based metrics, multiply by 100
                        multiplier = 100 if col in ['returns', 'volatility', 'volume_change'] else 1
                        stat_row[label] = regime_data[col].mean() * multiplier
                
                stats.append(stat_row)
        
        return pd.DataFrame(stats) 