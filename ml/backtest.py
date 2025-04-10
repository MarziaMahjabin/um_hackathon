import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model_hmm import HMMModel

class Backtest:
    """
    Backtests trading signals generated by the HMM model
    """
    def __init__(self, initial_capital=10000, transaction_cost_pct=0.0006):  # Updated to use 0.06% trading fee
        """
        Initialize the backtest
        
        Args:
            initial_capital (float): Initial capital for the backtest
            transaction_cost_pct (float): Transaction cost as a percentage of trade value
        """
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.model = None
        self.data = None
        self.signals = None
        self.positions = None
        self.portfolio = None
        self.metrics = {}
        
    def run(self, n_components=3, window_size=7, use_saved_data=False):
        """
        Run the backtest using the HMM model
        
        Args:
            n_components (int): Number of hidden states for the HMM model
            window_size (int): Window size for the rolling calculations
            use_saved_data (bool): Whether to use saved data with signals from miner_to_miner.csv
            
        Returns:
            dict: Backtest metrics
        """
        if use_saved_data:
            # Load data with signals from miner_to_miner.csv
            data_path = Path(__file__).parent.parent / "data" / "miner_to_miner.csv"
            if data_path.exists():
                self.data = pd.read_csv(data_path)
                
                # Convert string index to datetime if needed
                date_column = next((col for col in self.data.columns if any(date_term in col.lower() for date_term in ['time', 'date', 'timestamp'])), None)
                if date_column:
                    self.data[date_column] = pd.to_datetime(self.data[date_column])
                    self.data.set_index(date_column, inplace=True)
                
                if 'signal' in self.data.columns:
                    self.signals = self.data['signal']
                    
                    # Print the count of each signal type
                    unique_signals = self.signals.value_counts()
                    print("\nSignal distribution in loaded data:")
                    if all(isinstance(s, (int, float)) for s in unique_signals.index):
                        # Numeric signals
                        print(f"BUY (1): {unique_signals.get(1, 0)}")
                        print(f"SELL (0): {unique_signals.get(0, 0)}")
                        print(f"HOLD (-1): {unique_signals.get(-1, 0)}")
                    else:
                        # String signals
                        print(f"BUY: {unique_signals.get('buy', 0)}")
                        print(f"SELL: {unique_signals.get('sell', 0)}")
                        print(f"HOLD: {unique_signals.get('hold', 0)}")
                else:
                    raise ValueError("No 'signal' column found in miner_to_miner.csv")
            else:
                raise FileNotFoundError(f"Data file not found: {data_path}")
        else:
            # Initialize and train the HMM model
            self.model = HMMModel(n_components=n_components, random_state=42)  # Set random_state=42 for reproducibility
            self.data = self.model.load_data(window_size=window_size)
            
            # Fit the model and get signals
            self.model.fit()
            self.signals = self.model.get_signals_from_states()
        
        # Ensure price data is available
        if 'price' not in self.data.columns:
            # If price is not available, use first value column as a proxy
            value_columns = [col for col in self.data.columns if not col.startswith('feature_') and col != 'signal']
            if value_columns:
                print(f"No price column found. Using {value_columns[0]} as price proxy.")
                self.data['price'] = self.data[value_columns[0]]
            else:
                raise ValueError("Price data is required for backtesting")
        
        # Run the backtest
        self._calculate_positions()
        self._calculate_portfolio_value()
        self._calculate_metrics()
        
        return self.metrics
    
    def _calculate_positions(self):
        """
        Calculate positions based on signals
        """
        # Map signals to positions (0 for no position, 1 for long position)
        # This follows the trading rules exactly:
        # - If signal == 1 (BUY) and no position → buy
        # - If signal == 0 (SELL) and in position → sell
        # - If signal == -1 (HOLD) → maintain current position
        
        position = 0  # Start with no position
        positions = []
        position_changes = []
        
        for signal in self.signals:
            prev_position = position
            
            # Convert signal to int if it's a string
            if isinstance(signal, str):
                if signal == 'buy':
                    numeric_signal = 1  # BUY
                elif signal == 'sell':
                    numeric_signal = 0  # SELL
                else:  # 'hold'
                    numeric_signal = -1  # HOLD
            else:
                numeric_signal = signal  # Already numeric
            
            if numeric_signal == 1 and position == 0:
                position = 1  # Enter position
            elif numeric_signal == 0 and position == 1:
                position = 0  # Exit position
            # For HOLD (-1), maintain current position
            
            positions.append(position)
            position_changes.append(abs(position - prev_position))
        
        self.positions = pd.Series(positions, index=self.signals.index)
        self.position_changes = pd.Series(position_changes, index=self.signals.index)
        
    def _calculate_portfolio_value(self):
        """
        Calculate portfolio value based on positions
        """
        # Create a DataFrame for portfolio calculations
        self.portfolio = pd.DataFrame(index=self.data.index)
        self.portfolio['price'] = self.data['price']
        self.portfolio['positions'] = self.positions
        
        # Calculate daily returns of the price
        self.portfolio['price_returns'] = self.portfolio['price'].pct_change()
        
        # Calculate strategy returns based on positions held and price returns
        # If we hold a position of 1 (long), we get the full returns
        # If we hold a position of 0, we get no returns
        self.portfolio['strategy_returns'] = self.portfolio['positions'].shift(1) * self.portfolio['price_returns']
        
        # Calculate transaction costs
        self.portfolio['transaction_costs'] = self.position_changes * self.transaction_cost_pct
        
        # Calculate net returns (strategy returns minus transaction costs)
        self.portfolio['net_returns'] = self.portfolio['strategy_returns'] - self.portfolio['transaction_costs']
        
        # Calculate cumulative returns
        self.portfolio['cumulative_returns'] = (1 + self.portfolio['net_returns']).cumprod()
        
        # Calculate portfolio value
        self.portfolio['portfolio_value'] = self.initial_capital * self.portfolio['cumulative_returns']
        
        # Fill missing values
        self.portfolio.fillna(method='ffill', inplace=True)
        self.portfolio.fillna(0, inplace=True)
    
    def _calculate_metrics(self):
        """
        Calculate backtest metrics
        """
        # Create a dictionary to store metrics
        metrics = {}
        
        # Calculate returns and annualized metrics
        daily_returns = self.portfolio['net_returns']
        
        # Check if we have at least 2 days of data
        if len(daily_returns) < 2:
            raise ValueError("Not enough data for calculating metrics")
        
        # Calculate annualization factor based on data frequency
        # Assuming daily data, use 252 trading days per year
        annualization_factor = 252
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.sqrt(annualization_factor) * daily_returns.mean() / daily_returns.std()
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Calculate Max Drawdown
        cumulative_returns = self.portfolio['cumulative_returns']
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        metrics['max_drawdown'] = max_drawdown
        
        # Calculate trade frequency (as a percentage)
        # Convert signals to numeric if they are strings
        numeric_signals = self.signals.copy()
        if isinstance(self.signals.iloc[0], str):
            numeric_signals = self.signals.replace({'buy': 1, 'sell': 0, 'hold': -1})
        
        # Count non-HOLD signals (buy or sell)
        non_hold_signals = (numeric_signals != -1).sum()
        total_rows = len(numeric_signals)
        trade_frequency = (non_hold_signals / total_rows)
        metrics['trade_frequency'] = trade_frequency
        metrics['total_trades'] = self.position_changes.sum()
        
        # Calculate total return
        total_return = self.portfolio['portfolio_value'].iloc[-1] / self.initial_capital - 1
        metrics['total_return'] = total_return
        
        # Calculate annualized return
        years = len(self.portfolio) / annualization_factor
        annualized_return = (1 + total_return) ** (1 / years) - 1
        metrics['annualized_return'] = annualized_return
        
        # Store metrics
        self.metrics = metrics
    
    def plot_results(self):
        """
        Plot backtest results
        """
        if self.portfolio is None:
            raise ValueError("Backtest has not been run yet")
        
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value
        ax1 = plt.subplot(2, 1, 1)
        self.portfolio['portfolio_value'].plot(ax=ax1, color='blue')
        ax1.set_title('Portfolio Value')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True)
        
        # Plot positions and signals
        ax2 = plt.subplot(2, 1, 2)
        self.portfolio['positions'].plot(ax=ax2, color='green')
        ax2.set_title('Positions (0=Cash, 1=Long)')
        ax2.set_ylabel('Position')
        ax2.grid(True)
        
        # Convert signals to numeric if they are strings
        numeric_signals = self.signals.copy()
        if isinstance(self.signals.iloc[0], str):
            numeric_signals = self.signals.replace({'buy': 1, 'sell': 0, 'hold': -1})
        
        # Add buy/sell markers
        buy_signals = numeric_signals[numeric_signals == 1].index
        sell_signals = numeric_signals[numeric_signals == 0].index
        
        ax1.scatter(buy_signals, self.portfolio.loc[buy_signals, 'portfolio_value'], 
                   marker='^', color='g', s=100, label='Buy')
        ax1.scatter(sell_signals, self.portfolio.loc[sell_signals, 'portfolio_value'], 
                   marker='v', color='r', s=100, label='Sell')
        
        ax1.legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_metrics(self):
        """
        Print backtest metrics
        """
        if not self.metrics:
            raise ValueError("Backtest has not been run yet")
        
        print("\n===== Backtest Results =====")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.4f} ({self.metrics['max_drawdown']*100:.2f}%)")
        print(f"Trade Frequency: {self.metrics['trade_frequency']:.4f} ({self.metrics['trade_frequency']*100:.2f}%)")
        print(f"Total Trades: {self.metrics['total_trades']}")
        print(f"Total Return: {self.metrics['total_return']:.4f} ({self.metrics['total_return']*100:.2f}%)")
        print(f"Annualized Return: {self.metrics['annualized_return']:.4f} ({self.metrics['annualized_return']*100:.2f}%)")
        print("=============================")
        
    def get_portfolio_value(self):
        """
        Get portfolio value over time as a list
        
        Returns:
            list: Portfolio value over time
        """
        if self.portfolio is None:
            raise ValueError("Backtest has not been run yet")
        
        return self.portfolio['portfolio_value'].tolist()

if __name__ == "__main__":
    # Run backtest with saved data
    backtest = Backtest(initial_capital=10000, transaction_cost_pct=0.0006)  # 0.06% trading fee
    try:
        # First, try to use saved data with signals
        metrics = backtest.run(use_saved_data=True)
        print("Running backtest using miner-to-miner data with pre-calculated signals")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error using saved data: {e}")
        print("Running backtest using HMM model on miner-to-miner data")
        # If that fails, run the HMM model
        metrics = backtest.run(n_components=3, window_size=7, use_saved_data=False)
    
    # Print metrics
    backtest.print_metrics()
    
    # Get portfolio value over time
    portfolio_values = backtest.get_portfolio_value()
    
    # Return metrics in a dictionary and portfolio values
    result = {
        'Sharpe Ratio': round(metrics['sharpe_ratio'], 2),
        'Max Drawdown': round(metrics['max_drawdown'] * 100, 2),
        'Trade Frequency': round(metrics['trade_frequency'] * 100, 2)
    }
    
    print("\nResults dictionary:")
    print(result)
    
    # Plot results
    backtest.plot_results() 