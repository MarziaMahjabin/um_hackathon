import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from pathlib import Path
from data_preprocess import get_preprocessed_data

class HMMModel:
    """
    Hidden Markov Model for predicting market states and generating trading signals
    """
    def __init__(self, n_components=3, random_state=42):
        """
        Initialize the HMM model
        
        Args:
            n_components (int): Number of hidden states
            random_state (int): Random seed for reproducibility
        """
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components, 
            covariance_type="full", 
            random_state=random_state,
            n_iter=100
        )
        self.states = None
        self.data = None
        self.features = None
        self.random_state = random_state
        
    def load_data(self, window_size=7):
        """
        Load and preprocess data
        
        Args:
            window_size (int): Window size for rolling calculations
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        self.data = get_preprocessed_data(window_size)
        return self.data
        
    def prepare_features(self, feature_columns=None, data=None):
        """
        Prepare features for the HMM model
        
        Args:
            feature_columns (list): List of column names to use as features
            data (pd.DataFrame): Optional data to use instead of self.data
            
        Returns:
            np.ndarray: Feature matrix
        """
        if data is None:
            if self.data is None:
                self.load_data()
            data = self.data
        
        if feature_columns is None:
            # Default to all columns starting with 'feature_'
            feature_columns = [col for col in data.columns if col.startswith('feature_')]
        
        if not feature_columns:
            raise ValueError("No feature columns found in the dataset")
            
        print(f"Using features: {feature_columns}")
        
        # Store feature columns for later use
        self.features = feature_columns
            
        # Extract features
        X = data[feature_columns].values
        
        # Standardize features to have zero mean and unit variance
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        # Replace zero standard deviations with 1 to avoid division by zero
        stds[stds == 0] = 1
        X = (X - means) / stds
        
        return X
        
    def fit(self, X=None, feature_columns=None):
        """
        Fit the HMM model and predict states
        
        Args:
            X (np.ndarray): Optional feature matrix to use
            feature_columns (list): List of column names to use as features
            
        Returns:
            np.ndarray: Predicted states
        """
        # Prepare features if not provided
        if X is None:
            X = self.prepare_features(feature_columns)
        
        try:
            # Fit the model
            self.model.fit(X)
            
            # Predict states for the entire dataset
            full_X = self.prepare_features(feature_columns)
            self.states = self.model.predict(full_X)
            
            print(f"Model fitted with {self.n_components} states")
            print(f"State distribution: {np.bincount(self.states)}")
            
            return self.states
        except Exception as e:
            print(f"Error fitting model: {e}")
            raise
        
    def get_signals_from_states(self):
        """
        Convert hidden states to trading signals (buy/sell/hold)
        
        Returns:
            pd.Series: Trading signals with the same index as the data
        """
        # Ensure states are available
        if self.states is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Calculate mean feature values for each state
        state_features = {}
        
        # Find relevant features for signal generation
        pct_change_cols = [col for col in self.data.columns if 'pct_change' in col]
        if not pct_change_cols:
            pct_change_cols = [col for col in self.data.columns if 'z_score' in col]
            
        for state in range(self.n_components):
            # Get indices where state appears
            state_indices = np.where(self.states == state)[0]
            
            # Skip if no indices for this state
            if len(state_indices) == 0:
                continue
                
            # Calculate mean for relevant features in this state
            state_data = self.data.iloc[state_indices]
            state_features[state] = {
                'mean_values': {},
                'count': len(state_indices)
            }
            
            # Store mean values for all pct_change features
            for col in pct_change_cols:
                if col in state_data:
                    state_features[state]['mean_values'][col] = state_data[col].mean()
        
        # Map states to signals
        # Strategy: 
        # - Buy: Highest mean price % change 
        # - Sell: Lowest mean price % change
        # - Hold: Everyone else
        
        signals = np.full(len(self.states), 'hold', dtype=object)
        
        if state_features and pct_change_cols:
            # Get the primary feature to use for signal generation
            primary_feature = pct_change_cols[0]
            
            # Sort states by primary feature mean value
            sorted_states = sorted(state_features.keys(), 
                                 key=lambda s: state_features[s]['mean_values'].get(primary_feature, 0))
            
            if len(sorted_states) >= 3:
                # Lowest state is sell
                sell_state = sorted_states[0]
                # Highest state is buy
                buy_state = sorted_states[-1]
                
                # Assign signals
                signals[self.states == buy_state] = 'buy'
                signals[self.states == sell_state] = 'sell'
            elif len(sorted_states) == 2:
                # If only two states, assign buy and sell
                buy_state = sorted_states[1]
                sell_state = sorted_states[0]
                
                signals[self.states == buy_state] = 'buy'
                signals[self.states == sell_state] = 'sell'
            else:
                # If only one state, use the sign of primary_feature
                state = sorted_states[0]
                if state_features[state]['mean_values'].get(primary_feature, 0) > 0:
                    signals[self.states == state] = 'buy'
                elif state_features[state]['mean_values'].get(primary_feature, 0) < 0:
                    signals[self.states == state] = 'sell'
        
        # Print state characteristics
        print("\nState characteristics:")
        for state, features in state_features.items():
            signal = 'buy' if any(signals[self.states == state] == 'buy') else \
                     'sell' if any(signals[self.states == state] == 'sell') else 'hold'
            
            feature_info = []
            for feature_name, value in features['mean_values'].items():
                feature_info.append(f"{feature_name.split('feature_')[-1]}: {value:.2f}")
            
            feature_str = ", ".join(feature_info)
            print(f"State {state} ({signal}): {feature_str}, Count: {features['count']}")
        
        # Save the signals to the data DataFrame
        self.data['signal'] = signals
        
        # Save the data with signals
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / "miner_to_miner.csv"
        self.data.to_csv(output_path)
        print(f"Saved data with signals to {output_path}")
        
        # Return signals as a Series with the same index as the data
        return pd.Series(signals, index=self.data.index, name='signal')
        
    def predict(self, new_data=None):
        """
        Predict states for new data
        
        Args:
            new_data (pd.DataFrame): New data to predict states for
            
        Returns:
            np.ndarray: Predicted states
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        if new_data is None:
            return self.states
            
        # Prepare features from new data
        if isinstance(new_data, pd.DataFrame):
            # Use the same features as in training
            if self.features is None:
                raise ValueError("No features stored from training. Call fit() first.")
                
            X = new_data[self.features].values
        else:
            # Assume new_data is already prepared feature matrix
            X = new_data
        
        # Predict states
        predicted_states = self.model.predict(X)
        return predicted_states

if __name__ == "__main__":
    # Example usage
    model = HMMModel(n_components=3)
    model.load_data()
    model.fit()
    signals = model.get_signals_from_states()
    
    # Print signal counts
    signal_counts = signals.value_counts()
    print("\nSignal distribution:")
    for signal, count in signal_counts.items():
        print(f"{signal}: {count} ({count/len(signals)*100:.1f}%)")
    
    # Print a sample of the signals
    print("\nSample of signals:")
    print(signals.head(10)) 