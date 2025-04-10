"""
Backtesting Framework Template - Balaena Quant Submission

This framework simulates a trading strategy based on model predictions.
It calculates Sharpe Ratio, Max Drawdown, Trade Frequency and Net Return.
You can plug in any strategy (ML/HMM) that outputs buy/sell signals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestFramework:
    def __init__(self, data: pd.DataFrame, signal_column: str = 'signal', price_column: str = 'price', 
                 initial_cash: float = 10000, fee: float = 0.0006, regime_column: str = 'market_regime_name'):
        self.data = data.copy()
        self.signal_column = signal_column
        self.price_column = price_column
        self.regime_column = regime_column
        self.initial_cash = initial_cash
        self.cash = self.initial_cash
        self.crypto = 0
        self.fee = fee  # 0.06% trading fee
        self.portfolio_history = []
        self.position = False  # Track if we're in a position
        self.regime_aware = self.regime_column in self.data.columns
        self.trades = []  # List to track all trades
        self.prev_signal = None  # Track previous signal to detect changes

    def run(self):
        # First, check if we have reasonable price data
        self._validate_price_data()
        
        # Analyze signal distribution before running backtest
        self._analyze_signals()
        
        # Reset for clean run
        self.cash = self.initial_cash
        self.crypto = 0
        self.position = False
        self.portfolio_history = []
        self.trades = []
        self.prev_signal = None

        for i in range(len(self.data)):
            price = self.data.iloc[i][self.price_column]
            signal = self.data.iloc[i][self.signal_column]
            
            # Get current regime if available
            regime = None
            if self.regime_aware:
                regime = self.data.iloc[i][self.regime_column]

            # Convert signal to int if it's a string
            numeric_signal = signal
            if isinstance(signal, str):
                if signal.lower() == 'buy':
                    numeric_signal = 1  # BUY
                elif signal.lower() == 'sell':
                    numeric_signal = 0  # SELL
                else:  # 'hold'
                    numeric_signal = -1  # HOLD
            
            # Apply regime-specific position sizing (optional)
            position_size = 1.0  # Default position size
            if regime == 'bull':
                position_size = 1.0  # Full position in bull markets
            elif regime == 'bear':
                position_size = 0.5  # Half position in bear markets
            elif regime == 'sideways':
                position_size = 0.75  # 75% position in sideways markets

            # Only trade when signal changes (key improvement)
            signal_changed = self.prev_signal != numeric_signal
            
            # Trading rules with improved logic:
            # If signal is BUY and no position → buy
            # If signal is SELL and in position → sell
            # Only execute trade when signal changes
            # Apply 0.06% trading fee on each transaction
            trade_executed = False
            
            if numeric_signal == 1 and not self.position and self.cash > 0 and (signal_changed or self.prev_signal is None):
                # Apply position sizing based on regime
                cash_to_use = self.cash * position_size
                self.crypto = (cash_to_use * (1 - self.fee)) / price
                self.cash -= cash_to_use
                self.position = True
                trade_executed = True
                self.trades.append({
                    'index': i,
                    'type': 'BUY',
                    'price': price,
                    'amount': self.crypto,
                    'value': cash_to_use,
                    'fee': cash_to_use * self.fee
                })
                logger.info(f"BUY executed at price {price:.2f}, amount: {self.crypto:.6f}, cash remaining: {self.cash:.2f}")
                
            elif numeric_signal == 0 and self.position and self.crypto > 0 and (signal_changed or self.prev_signal is None):
                trade_value = self.crypto * price
                self.cash += trade_value * (1 - self.fee)
                
                self.trades.append({
                    'index': i,
                    'type': 'SELL',
                    'price': price,
                    'amount': self.crypto,
                    'value': trade_value,
                    'fee': trade_value * self.fee
                })
                
                logger.info(f"SELL executed at price {price:.2f}, amount: {self.crypto:.6f}, cash now: {self.cash:.2f}")
                self.crypto = 0
                self.position = False
                trade_executed = True

            # Calculate portfolio value at this step (always update even without a trade)
            portfolio_value = self.cash + self.crypto * price
            self.portfolio_history.append(portfolio_value)
            
            # Update previous signal
            self.prev_signal = numeric_signal
            
            # Log portfolio value at regular intervals
            if i % 20 == 0 or i == len(self.data) - 1:
                logger.info(f"Step {i}, Portfolio value: {portfolio_value:.2f}, Position: {'In' if self.position else 'Out'}")

        # Store portfolio value in the dataframe
        self.data["portfolio_value"] = self.portfolio_history
        
        # Check if portfolio values are all the same
        if len(self.portfolio_history) > 1 and all(x == self.portfolio_history[0] for x in self.portfolio_history):
            logger.warning("WARNING: Portfolio value did not change during backtest! Check your signal generation.")
            logger.warning("This likely means no trades were executed or all signals were HOLD.")

    def _validate_price_data(self):
        """Validate price data to ensure it has enough variation for meaningful backtest"""
        prices = self.data[self.price_column].values
        price_std = np.std(prices)
        price_mean = np.mean(prices)
        
        # Check if prices are too flat (which can lead to 0 Sharpe/Drawdown)
        if price_std / price_mean < 0.01:
            logger.warning("Price data shows very low volatility (<1%). This may result in unrealistic backtest results.")
            
        # Check for missing values
        if self.data[self.price_column].isna().any():
            logger.warning("Price data contains missing values. This may affect backtest results.")

    def _analyze_signals(self):
        """Analyze the distribution of signals before running the backtest"""
        signals = self.data[self.signal_column]
        
        # Convert to common format for analysis
        if isinstance(signals.iloc[0], str):
            signal_counts = signals.str.lower().value_counts()
        else:
            # Map numeric signals to string labels for counting
            signal_map = {1: 'buy', 0: 'sell', -1: 'hold'}
            signal_counts = signals.map(signal_map).value_counts()
        
        # Calculate percentages
        total = len(signals)
        signal_pcts = {signal: count/total*100 for signal, count in signal_counts.items()}
        
        # Log the distribution
        logger.info(f"Signal distribution: {signal_counts.to_dict()}")
        logger.info(f"Signal percentages: {signal_pcts}")
        
        # Check if we have enough trade signals
        buys = signal_counts.get('buy', 0)
        sells = signal_counts.get('sell', 0)
        holds = signal_counts.get('hold', 0)
        
        if buys == 0 or sells == 0:
            logger.warning("WARNING: Missing BUY or SELL signals. Backtest will not generate trades!")
            
        if buys + sells < 0.1 * total:
            logger.warning("WARNING: Very few trade signals (<10%). Backtest may not be meaningful.")

    def metrics(self):
        """Calculate performance metrics with improved Sharpe Ratio calculation"""
        
        # Calculate returns using pct_change as specified
        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()
        
        # Calculate metrics only if we have data
        if len(returns) < 2:
            logger.warning("Insufficient data points for metrics calculation")
            return self._create_zero_metrics()
            
        # Calculate Sharpe Ratio with safeguards against division by zero
        if returns.std() == 0 or np.isnan(returns.std()):
            logger.warning("Standard deviation of returns is zero, indicating no portfolio value changes")
            sharpe = 0
        else:
            # Annualize based on data frequency (assuming daily by default)
            # For 4-hour data, use sqrt(6*252) instead of sqrt(252)
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        # Calculate drawdown
        equity_curve = portfolio_series
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max
        max_drawdown = drawdown.max()
        
        # Count actual trades executed
        trade_count = len(self.trades)
        trade_frequency = trade_count / len(self.data) * 100  # As percentage
        
        # Calculate final return
        final_return = (self.portfolio_history[-1] / self.initial_cash - 1) * 100
        
        # Calculate regime-specific performance if available
        regime_metrics = {}
        if self.regime_aware:
            for regime in self.data[self.regime_column].unique():
                regime_mask = self.data[self.regime_column] == regime
                if regime_mask.sum() > 1:  # Need at least 2 data points for returns
                    regime_returns = self.data.loc[regime_mask, 'portfolio_value'].pct_change().dropna()
                    if len(regime_returns) > 1 and regime_returns.std() > 0:
                        regime_sharpe = (regime_returns.mean() / regime_returns.std()) * np.sqrt(252)
                    else:
                        regime_sharpe = 0
                    
                    regime_metrics[regime] = {
                        'Sharpe': round(float(regime_sharpe), 2),
                        'Count': int(regime_mask.sum()),
                        'Avg Return': round(float(regime_returns.mean() * 100), 2) if len(regime_returns) > 0 else 0
                    }

        # Check if portfolio didn't change at all
        if all(x == self.portfolio_history[0] for x in self.portfolio_history):
            logger.error("ERROR: Portfolio value remained constant throughout the backtest!")
            logger.error("This indicates no trades were executed or they had no effect.")
            return self._create_zero_metrics("No trades executed or strategy had no effect on portfolio value")
        
        # Replace any NaN values with 0 to ensure valid JSON
        if np.isnan(sharpe):
            sharpe = 0
        if np.isnan(max_drawdown):
            max_drawdown = 0
        
        metrics_dict = {
            "Sharpe Ratio": round(float(sharpe), 2),  # Ensure it's a float, not numpy.float
            "Max Drawdown": round(float(max_drawdown * 100), 2),  # Convert to percentage and ensure it's a float
            "Trade Frequency": round(float(trade_frequency), 2),  # Ensure it's a float
            "Final Portfolio Value": round(float(self.portfolio_history[-1]), 2),  # Ensure it's a float
            "Portfolio Equity Curve": [float(x) for x in self.portfolio_history],  # Convert all to float
            "Total Return": round(float(final_return), 2),  # Total return %
            "Number of Trades": trade_count,
            "Trades Per Day": round(float(trade_count / len(self.data)), 4)
        }
        
        # Add regime metrics if available
        if regime_metrics:
            metrics_dict["Regime Performance"] = regime_metrics
            
        # Add win/loss stats if trades were executed
        if trade_count > 0:
            # Calculate win rate and other stats
            buy_trades = [t for t in self.trades if t['type'] == 'BUY']
            sell_trades = [t for t in self.trades if t['type'] == 'SELL']
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                # Match buy/sell pairs to calculate trade P&L
                profit_trades = 0
                loss_trades = 0
                
                # Simple approach assuming buys and sells alternate
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_value = buy_trades[i]['value']
                    sell_value = sell_trades[i]['value']
                    if sell_value > buy_value:
                        profit_trades += 1
                    else:
                        loss_trades += 1
                
                total_paired_trades = profit_trades + loss_trades
                win_rate = (profit_trades / total_paired_trades * 100) if total_paired_trades > 0 else 0
                
                metrics_dict["Win Rate"] = round(float(win_rate), 2)
                metrics_dict["Profitable Trades"] = profit_trades
                metrics_dict["Losing Trades"] = loss_trades
        
        return metrics_dict
    
    def _create_zero_metrics(self, message="Insufficient data for meaningful metrics"):
        """Create a metrics dictionary with zero values when backtest fails"""
        return {
            "Sharpe Ratio": 0.0,
            "Max Drawdown": 0.0,
            "Trade Frequency": 0.0,
            "Final Portfolio Value": round(float(self.initial_cash), 2),
            "Portfolio Equity Curve": [float(self.initial_cash)] * len(self.data),
            "Total Return": 0.0,
            "Number of Trades": 0,
            "Error": message
        }
    
    def plot_results(self):
        """
        Plot backtest results with regime highlighting if available
        """
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value
        ax1 = plt.subplot(2, 1, 1)
        
        # Plot portfolio with regime coloring if available
        if self.regime_aware:
            regime_colors = {
                'bull': 'green',
                'bear': 'red',
                'sideways': 'gray'
            }
            
            # Plot portfolio value with regime coloring
            for regime in self.data[self.regime_column].unique():
                mask = self.data[self.regime_column] == regime
                if np.any(mask):
                    plt.plot(self.data.index[mask], 
                            self.data.loc[mask, 'portfolio_value'],
                            color=regime_colors.get(regime, 'blue'),
                            label=f"{regime.capitalize()} Regime")
        else:
            # Simple portfolio value plot
            plt.plot(self.data.index, self.data['portfolio_value'], color='blue')
        
        # Add a horizontal line at initial cash level
        plt.axhline(y=self.initial_cash, color='r', linestyle='--', alpha=0.3, label='Initial Cash')
        
        plt.title("Portfolio Value")
        plt.ylabel("Value ($)")
        plt.grid(True)
        plt.legend()
        
        # Plot signals and positions
        ax2 = plt.subplot(2, 1, 2)
        
        # Convert signals to numeric if they are strings
        signals = self.data[self.signal_column]
        if isinstance(signals.iloc[0], str):
            signals = signals.replace({'buy': 1, 'sell': 0, 'hold': -1})
        
        # Plot signals
        buy_signals = signals == 1
        sell_signals = signals == 0
        
        # Plot signals where trades were actually executed
        trade_indices = [t['index'] for t in self.trades]
        buy_indices = [t['index'] for t in self.trades if t['type'] == 'BUY']
        sell_indices = [t['index'] for t in self.trades if t['type'] == 'SELL']
        
        # Plot all signals as small markers
        plt.scatter(self.data.index[buy_signals], 
                   [0.3] * buy_signals.sum(), 
                   marker='^', color='g', s=20, alpha=0.3, label='Buy Signal')
        plt.scatter(self.data.index[sell_signals], 
                   [0.3] * sell_signals.sum(), 
                   marker='v', color='r', s=20, alpha=0.3, label='Sell Signal')
        
        # Plot executed trades as larger markers
        if buy_indices:
            plt.scatter(self.data.index[buy_indices], 
                       [0.7] * len(buy_indices), 
                       marker='^', color='g', s=100, label='Buy Executed')
        if sell_indices:
            plt.scatter(self.data.index[sell_indices], 
                       [0.7] * len(sell_indices), 
                       marker='v', color='r', s=100, label='Sell Executed')
        
        # Plot regime background if available
        if self.regime_aware:
            ax2_twin = ax2.twinx()
            regime_numeric = self.data[self.regime_column].map({"bull": 2, "sideways": 1, "bear": 0})
            ax2_twin.plot(self.data.index, regime_numeric, 'k--', alpha=0.5, label='Regime')
            ax2_twin.set_yticks([0, 1, 2])
            ax2_twin.set_yticklabels(["Bear", "Sideways", "Bull"])
            ax2_twin.set_ylabel("Market Regime")
        
        plt.title("Trading Signals")
        plt.ylabel("Signal")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show() 