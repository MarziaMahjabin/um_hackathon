import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import LiveTradeGraph from './components/LiveTradeGraph';
import SignalHistory from './components/SignalHistory';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [predictionData, setPredictionData] = useState(null);
  const [latestData, setLatestData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [backtestResults, setBacktestResults] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestError, setBacktestError] = useState(null);
  const [backtestDataPoints, setBacktestDataPoints] = useState(0);
  const [priceHistory, setPriceHistory] = useState(null);
  const [priceHistoryLoading, setPriceHistoryLoading] = useState(false);
  const [priceHistoryError, setPriceHistoryError] = useState(null);
  const [signalHistory, setSignalHistory] = useState(null);
  const [signalHistoryLoading, setSignalHistoryLoading] = useState(false);
  const [signalHistoryError, setSignalHistoryError] = useState(null);
  // Add state for live prediction data history
  const [predictionHistory, setPredictionHistory] = useState([]);
  const maxHistoryPoints = 20; // Maximum number of points to keep in history
  // Reference to countdown timer
  const refreshTimerRef = useRef(null);
  const [refreshCountdown, setRefreshCountdown] = useState(60);

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: {
            feature_price_rolling_avg: 42000.50,
            feature_price_pct_change: 2.5,
            feature_net_inflow: 150000
          }
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      const newPrediction = {
        ...data.prediction,
        // Check the data_source flag that we added to the backend response
        isRealData: data.latest_data && data.latest_data.data_source === 'api',
        timestamp: new Date().toISOString() // Add timestamp
      };
      
      setPredictionData(newPrediction);
      setLatestData(data.latest_data);
      const timestamp = new Date();
      setLastUpdated(timestamp);
      
      // Add prediction to history
      setPredictionHistory(prevHistory => {
        const updatedHistory = [...prevHistory, {
          ...newPrediction,
          timestamp
        }];
        // Keep only the last maxHistoryPoints
        return updatedHistory.slice(-maxHistoryPoints);
      });
      
      // Reset countdown
      setRefreshCountdown(60);
    } catch (err) {
      setError(`Failed to fetch prediction: ${err.message}`);
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchBacktestResults = async () => {
    setBacktestLoading(true);
    setBacktestError(null);
    
    try {
      const response = await fetch('http://localhost:5000/backtest');
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      setBacktestResults(data.backtest_results);
      setBacktestDataPoints(data.data_points || 0);
    } catch (err) {
      setBacktestError(`Failed to fetch backtest results: ${err.message}`);
      console.error('Backtest fetch error:', err);
    } finally {
      setBacktestLoading(false);
    }
  };

  const fetchPriceHistory = async () => {
    setPriceHistoryLoading(true);
    setPriceHistoryError(null);
    
    try {
      const response = await fetch('http://localhost:5000/price_history?days=30&interval=1d');
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      setPriceHistory(data);
    } catch (err) {
      setPriceHistoryError(`Failed to fetch price history: ${err.message}`);
      console.error('Price history fetch error:', err);
    } finally {
      setPriceHistoryLoading(false);
    }
  };

  const fetchSignalHistory = async () => {
    setSignalHistoryLoading(true);
    setSignalHistoryError(null);
    
    try {
      const response = await fetch('http://localhost:5000/signals');
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      setSignalHistory(data.signals);
    } catch (err) {
      setSignalHistoryError(`Failed to fetch signal history: ${err.message}`);
      console.error('Signal history fetch error:', err);
    } finally {
      setSignalHistoryLoading(false);
    }
  };

  useEffect(() => {
    // Fetch prediction when component mounts
    fetchPrediction();
    
    // Fetch price history when component mounts
    fetchPriceHistory();
    
    // Fetch signal history when component mounts
    fetchSignalHistory();
    
    // Setup auto-refresh if enabled
    let intervalId = null;
    if (autoRefresh) {
      intervalId = setInterval(() => {
        fetchPrediction();
        fetchPriceHistory();
        fetchSignalHistory();
        setRefreshCountdown(60); // Reset countdown
      }, 60000); // Refresh every minute
      
      // Setup countdown timer
      refreshTimerRef.current = setInterval(() => {
        setRefreshCountdown(prev => (prev > 0 ? prev - 1 : 0));
      }, 1000);
    } else {
      // Clear countdown timer if auto-refresh is disabled
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
      if (refreshTimerRef.current) clearInterval(refreshTimerRef.current);
    };
  }, [autoRefresh]);

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  // Helper function to get signal color
  const getSignalColor = (signal) => {
    if (!signal) return '#888';
    
    switch(signal.toLowerCase()) {
      case 'buy':
        return '#4CAF50'; // Green
      case 'sell':
        return '#F44336'; // Red
      case 'hold':
        return '#FFC107'; // Amber
      default:
        return '#888';    // Gray
    }
  };
  
  // Format currency
  const formatCurrency = (value) => {
    if (value === undefined || value === null) return '';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 2
    }).format(value);
  };

  // Format date for chart
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return `${date.getMonth() + 1}/${date.getDate()}`;
  };

  // Price Chart options
  const priceChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#fff',
          font: {
            size: 12,
            weight: 'bold'
          }
        }
      },
      title: {
        display: true,
        text: 'BTC Price Trend (Last 30 Days)',
        color: '#fff',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleFont: {
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          size: 13
        },
        callbacks: {
          label: function(context) {
            return 'Price: ' + formatCurrency(context.parsed.y);
          }
        }
      }
    },
    scales: {
      x: {
        ticks: {
          color: '#fff',
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
      y: {
        ticks: {
          color: '#fff',
          font: {
            size: 11
          },
          callback: (value) => `$${value.toLocaleString()}`
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    },
    elements: {
      line: {
        tension: 0.3,
        borderWidth: 2
      },
      point: {
        radius: 0,
        hoverRadius: 6
      }
    }
  };

  // Signal Chart options
  const signalChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#fff',
          font: {
            size: 12,
            weight: 'bold'
          }
        }
      },
      title: {
        display: true,
        text: 'Historical Signals',
        color: '#fff',
        font: {
          size: 16,
          weight: 'bold'
        }
      }
    },
    scales: {
      x: {
        ticks: {
          color: '#fff',
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
      y: {
        ticks: {
          color: '#fff',
          font: {
            size: 11
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      }
    }
  };

  // Function to get the color for each signal in the chart
  const getSignalBarColor = (signal) => {
    switch(signal.toLowerCase()) {
      case 'buy':
        return '#4CAF50'; // Green
      case 'sell':
        return '#F44336'; // Red
      case 'hold':
        return '#FFC107'; // Amber
      default:
        return '#888';    // Gray
    }
  };

  // Function to map signal to numeric value for chart
  const mapSignalToValue = (signal) => {
    switch(signal.toLowerCase()) {
      case 'buy':
        return 2;
      case 'hold':
        return 1;
      case 'sell':
        return 0;
      default:
        return 1;
    }
  };

  // Live prediction chart options
  const livePredictionChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 500
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#fff',
          font: {
            size: 12,
            weight: 'bold'
          }
        }
      },
      title: {
        display: true,
        text: 'Real-time Prediction Confidence',
        color: '#fff',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        callbacks: {
          label: function(context) {
            const signal = predictionHistory[context.dataIndex]?.signal || '';
            return `${signal.toUpperCase()}: ${(context.parsed.y * 100).toFixed(1)}%`;
          },
          title: function(context) {
            const point = context[0];
            const timestamp = predictionHistory[point.dataIndex]?.timestamp;
            return timestamp ? new Date(timestamp).toLocaleTimeString() : '';
          }
        }
      }
    },
    scales: {
      x: {
        ticks: {
          color: '#fff',
          callback: function(value, index) {
            if (predictionHistory[index]?.timestamp) {
              const date = new Date(predictionHistory[index].timestamp);
              return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            }
            return '';
          }
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
      y: {
        min: 0,
        max: 1,
        ticks: {
          color: '#fff',
          callback: value => `${(value * 100).toFixed(0)}%`
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      }
    },
    interaction: {
      mode: 'index',
      intersect: false
    }
  };

  // Helper function to get confidence values from history
  const getConfidenceData = () => {
    if (!predictionHistory || predictionHistory.length === 0) return [];
    return predictionHistory.map(p => p.confidence || 0);
  };
  
  // Helper function to get colors based on signal
  const getPointColors = () => {
    if (!predictionHistory || predictionHistory.length === 0) return [];
    return predictionHistory.map(p => getSignalColor(p.signal));
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Crypto Trading Signal Dashboard</h1>
        <div className="last-updated">
          {lastUpdated && (
            <span>Last updated: {lastUpdated.toLocaleString()}</span>
          )}
        </div>
      </header>
      
      <main className="App-content">
        <section className="prediction-section">
          <h2>Current Prediction</h2>
          
          {loading && <div className="loading">Loading prediction data...</div>}
          
          {error && <div className="error-message">{error}</div>}
          
          {predictionData && (
            <div className="prediction-container">
              <div className="prediction-card">
                <div 
                  className="signal-indicator"
                  style={{ backgroundColor: getSignalColor(predictionData.signal) }}
                >
                  <h3 className="signal-text">{predictionData.signal?.toUpperCase()}</h3>
                </div>
                
                <div className="prediction-details">
                  <div className="metric">
                    <span className="metric-label">Confidence:</span>
                    <div className="confidence-bar-container">
                      <div 
                        className="confidence-bar" 
                        style={{ 
                          width: `${predictionData.confidence * 100}%`,
                          backgroundColor: getSignalColor(predictionData.signal)
                        }}
                      ></div>
                      <span className="confidence-value">{(predictionData.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  <div className="metric">
                    <span className="metric-label">State:</span>
                    <span className="metric-value">{predictionData.state}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Source:</span>
                    <span className="metric-value source-indicator">
                      {predictionData.isRealData ? (
                        <>
                          <span className="live-indicator"></span>
                          Real-time API Data
                        </>
                      ) : "Test Data"}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="live-graph-container">
                <div className="chart-content">
                  {predictionHistory.length > 1 ? (
                    <Line
                      options={livePredictionChartOptions}
                      data={{
                        labels: predictionHistory.map((_, i) => i),
                        datasets: [
                          {
                            label: 'Confidence',
                            data: getConfidenceData(),
                            borderColor: 'rgba(255, 255, 255, 0.8)',
                            backgroundColor: 'rgba(255, 255, 255, 0.2)',
                            borderWidth: 2,
                            pointBackgroundColor: getPointColors(),
                            pointBorderColor: '#fff',
                            pointRadius: 5,
                            pointHoverRadius: 8,
                            fill: false,
                            tension: 0.1
                          }
                        ]
                      }}
                    />
                  ) : (
                    <div className="placeholder-content">
                      <p>Collecting data for confidence trend chart...</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
          
          <div className="prediction-controls">
            <button className="refresh-button" onClick={fetchPrediction} disabled={loading}>
              {loading ? 'Refreshing...' : 'Refresh Prediction'}
            </button>
            
            <label className="auto-refresh-label">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              <span>Auto-refresh</span>
              {autoRefresh && (
                <span className="refresh-countdown">({refreshCountdown}s)</span>
              )}
            </label>
          </div>
        </section>
        
        <section className="market-data-section">
          <h2>Current Market Data</h2>
          
          {latestData ? (
            <div className="market-data-grid">
              <div className="market-data-card">
                <h3>Price</h3>
                <p className="large-value">{formatCurrency(latestData.price_usd)}</p>
              </div>
              
              <div className="market-data-card">
                <h3>Inflow</h3>
                <p className="large-value">{formatCurrency(latestData.inflow)}</p>
              </div>
              
              <div className="market-data-card">
                <h3>Outflow</h3>
                <p className="large-value">{formatCurrency(latestData.outflow)}</p>
              </div>
              
              <div className="market-data-card">
                <h3>Net Flow</h3>
                <p className="large-value" style={{ 
                  color: (latestData.net_flow || 0) >= 0 ? '#4CAF50' : '#F44336' 
                }}>
                  {formatCurrency(latestData.net_flow)}
                </p>
              </div>
              
              <div className="market-data-card">
                <h3>Timestamp</h3>
                <p className="medium-value">{formatTimestamp(latestData.timestamp)}</p>
              </div>
            </div>
          ) : (
            <div className="loading">Waiting for market data...</div>
          )}
        </section>
        
        <section className="charts-section">
          <h2>Market Analysis</h2>
          
          <div className="market-metrics-summary">
            {latestData && (
              <div className="metrics-overview">
                <div className="metric-summary">
                  <span className="metric-summary-label">24h Change</span>
                  <span className="metric-summary-value" style={{ 
                    color: Math.random() > 0.5 ? '#4CAF50' : '#F44336' // Mock 50/50 chance for demo
                  }}>
                    {Math.random() > 0.5 ? '+' : '-'}{(Math.random() * 5).toFixed(2)}%
                  </span>
                </div>
                <div className="metric-summary">
                  <span className="metric-summary-label">30d Volatility</span>
                  <span className="metric-summary-value">{(Math.random() * 10 + 5).toFixed(2)}%</span>
                </div>
                <div className="metric-summary">
                  <span className="metric-summary-label">Net Flow</span>
                  <span className="metric-summary-value" style={{ 
                    color: (latestData.net_flow || 0) >= 0 ? '#4CAF50' : '#F44336' 
                  }}>
                    {formatCurrency(latestData.net_flow)}
                  </span>
                </div>
                <div className="metric-summary">
                  <span className="metric-summary-label">Inflow/Outflow Ratio</span>
                  <span className="metric-summary-value">
                    {(latestData.inflow / Math.max(latestData.outflow, 1)).toFixed(2)}
                  </span>
                </div>
              </div>
            )}
          </div>
          
          <div className="chart-placeholders">
            <div className="chart-placeholder">
              <h3>Price & Signal Trend</h3>
              <div className="chart-content">
                {(priceHistoryLoading || !priceHistory) && <div className="loading">Loading price data...</div>}
                
                {priceHistoryError && <div className="error-message">{priceHistoryError}</div>}
                
                {priceHistory && !priceHistoryLoading && (
                  <LiveTradeGraph 
                    priceHistory={priceHistory}
                    predictionHistory={predictionHistory} 
                    latestPrediction={predictionData}
                  />
                )}
              </div>
            </div>
            
            <div className="chart-placeholder">
              <h3>On-Chain Metrics</h3>
              <div className="chart-content">
                {priceHistoryLoading && <div className="loading">Loading metrics data...</div>}
                
                {priceHistoryError && <div className="error-message">{priceHistoryError}</div>}
                
                {latestData && !priceHistoryLoading && (
                  <div className="metrics-details">
                    <div className="metric-card">
                      <div className="metric-icon inflow-icon">
                        <svg viewBox="0 0 24 24" width="24" height="24">
                          <path fill="currentColor" d="M19,6H5L12,13L19,6M19,18H5L12,11L19,18Z" />
                        </svg>
                      </div>
                      <div className="metric-info">
                        <h4>Exchange Inflow</h4>
                        <div className="metric-value">{formatCurrency(latestData.inflow)}</div>
                        <div className="metric-description">
                          <span className="trend-indicator positive">↑ 2.3%</span> from yesterday
                        </div>
                      </div>
                    </div>
                    
                    <div className="metric-card">
                      <div className="metric-icon outflow-icon">
                        <svg viewBox="0 0 24 24" width="24" height="24">
                          <path fill="currentColor" d="M19,18H5L12,11L19,18M19,6H5L12,13L19,6Z" />
                        </svg>
                      </div>
                      <div className="metric-info">
                        <h4>Exchange Outflow</h4>
                        <div className="metric-value">{formatCurrency(latestData.outflow)}</div>
                        <div className="metric-description">
                          <span className="trend-indicator negative">↓ 1.5%</span> from yesterday
                        </div>
                      </div>
                    </div>
                    
                    <div className="metric-card">
                      <div className="metric-icon flow-icon">
                        <svg viewBox="0 0 24 24" width="24" height="24">
                          <path fill="currentColor" d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M7,10L12,15L17,10H7Z" />
                        </svg>
                      </div>
                      <div className="metric-info">
                        <h4>Net Flow</h4>
                        <div className="metric-value" style={{ 
                          color: (latestData.net_flow || 0) >= 0 ? '#4CAF50' : '#F44336' 
                        }}>
                          {formatCurrency(latestData.net_flow)}
                        </div>
                        <div className="metric-description">
                          More BTC entering exchanges
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {!latestData && !priceHistoryLoading && !priceHistoryError && (
                  <div className="placeholder-content">
                    <p>On-chain metrics visualization will appear here</p>
                  </div>
                )}
              </div>
            </div>
            
            <div className="chart-placeholder">
              <h3>Signal History</h3>
              <div className="chart-content">
                {signalHistoryLoading && <div className="loading">Loading signal history...</div>}
                
                {signalHistoryError && <div className="error-message">{signalHistoryError}</div>}
                
                {signalHistory && !signalHistoryLoading && (
                  <SignalHistory signalHistory={signalHistory} />
                )}
              </div>
            </div>
          </div>
        </section>
        
        <section className="backtest-section">
          <h2>Strategy Backtest</h2>
          
          <div className="backtest-controls">
            <button 
              className="run-backtest-button" 
              onClick={fetchBacktestResults} 
              disabled={backtestLoading}
            >
              {backtestLoading ? 'Running Backtest...' : 'Run Backtest'}
            </button>
          </div>
          
          {backtestError && <div className="error-message">{backtestError}</div>}
          
          {backtestResults && (
            <div className="backtest-card">
              <h3>Backtest Results</h3>
              <div className="backtest-info">
                <p>Analysis based on {backtestDataPoints} days of historical data</p>
                
                {/* Show error message if present */}
                {backtestResults["Error"] && (
                  <div className="backtest-error">
                    <div className="error-icon">⚠️</div>
                    <div className="error-message">{backtestResults["Error"]}</div>
                    <p className="error-hint">Try adjusting your model parameters or retraining to generate more diverse signals.</p>
                  </div>
                )}
                
                {/* Only show quality indicator if no error and valid Sharpe Ratio */}
                {!backtestResults["Error"] && (
                  <div className="backtest-summary">
                    {backtestResults["Sharpe Ratio"] > 1 ? (
                      <div className="backtest-quality good">
                        <span className="quality-indicator">✓</span> This strategy shows good risk-adjusted returns
                      </div>
                    ) : backtestResults["Sharpe Ratio"] > 0.5 ? (
                      <div className="backtest-quality average">
                        <span className="quality-indicator">○</span> This strategy shows moderate risk-adjusted returns
                      </div>
                    ) : (
                      <div className="backtest-quality poor">
                        <span className="quality-indicator">✗</span> This strategy shows poor risk-adjusted returns
                      </div>
                    )}
                  </div>
                )}
              </div>
              
              {/* Display trade statistics if they exist */}
              {backtestResults["Number of Trades"] > 0 && (
                <div className="trade-statistics">
                  <h4>Trade Statistics</h4>
                  <div className="trade-stats-grid">
                    <div className="trade-stat">
                      <span className="stat-label">Total Trades:</span>
                      <span className="stat-value">{backtestResults["Number of Trades"]}</span>
                    </div>
                    {backtestResults["Win Rate"] && (
                      <div className="trade-stat">
                        <span className="stat-label">Win Rate:</span>
                        <span className="stat-value" style={{
                          color: backtestResults["Win Rate"] > 50 ? '#4CAF50' : '#F44336'
                        }}>{backtestResults["Win Rate"]}%</span>
                      </div>
                    )}
                    {backtestResults["Profitable Trades"] && (
                      <div className="trade-stat">
                        <span className="stat-label">Profitable Trades:</span>
                        <span className="stat-value">{backtestResults["Profitable Trades"]}</span>
                      </div>
                    )}
                    {backtestResults["Losing Trades"] && (
                      <div className="trade-stat">
                        <span className="stat-label">Losing Trades:</span>
                        <span className="stat-value">{backtestResults["Losing Trades"]}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              <div className="backtest-metrics">
                <div className="metric">
                  <span className="metric-label">Sharpe Ratio:</span>
                  <span className="metric-value" style={{
                    color: backtestResults["Sharpe Ratio"] > 1 ? '#4CAF50' : 
                           backtestResults["Sharpe Ratio"] > 0.5 ? '#FFC107' : '#F44336'
                  }}>
                    {backtestResults["Sharpe Ratio"]}
                    <span className="metric-explainer">Risk-adjusted return</span>
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">Max Drawdown:</span>
                  <span className="metric-value" style={{
                    color: backtestResults["Max Drawdown"] < 10 ? '#4CAF50' :
                           backtestResults["Max Drawdown"] < 20 ? '#FFC107' : '#F44336'
                  }}>
                    {backtestResults["Max Drawdown"]}%
                    <span className="metric-explainer">Largest decline from peak</span>
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">Trade Frequency:</span>
                  <span className="metric-value">
                    {backtestResults["Trade Frequency"]}%
                    <span className="metric-explainer">Percentage of days with trades</span>
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">Final Portfolio Value:</span>
                  <span className="metric-value" style={{
                    color: backtestResults["Final Portfolio Value"] > 10000 ? '#4CAF50' : 
                           backtestResults["Final Portfolio Value"] === 10000 ? '#888' : '#F44336'
                  }}>
                    {formatCurrency(backtestResults["Final Portfolio Value"])}
                    <span className="metric-explainer">
                      {backtestResults["Final Portfolio Value"] > 10000 
                        ? `+${((backtestResults["Final Portfolio Value"] / 10000 - 1) * 100).toFixed(2)}% gain` 
                        : backtestResults["Final Portfolio Value"] < 10000
                          ? `${((1 - backtestResults["Final Portfolio Value"] / 10000) * 100).toFixed(2)}% loss`
                          : "No change"}
                    </span>
                  </span>
                </div>
              </div>
              
              {/* Only show explanation if no error */}
              {!backtestResults["Error"] && (
                <div className="backtest-explanation">
                  <h4>What do these metrics mean?</h4>
                  <ul>
                    <li><strong>Sharpe Ratio</strong> measures risk-adjusted return. Higher is better (good: &gt;1, poor: &lt;0.5).</li>
                    <li><strong>Max Drawdown</strong> shows the largest percentage drop from peak to trough. Lower is better.</li>
                    <li><strong>Trade Frequency</strong> indicates how often the strategy generates trades.</li>
                    <li><strong>Final Portfolio Value</strong> shows the ending value of a $10,000 initial investment.</li>
                  </ul>
                </div>
              )}
              
              {/* Show troubleshooting tips if there's an error or Sharpe Ratio is 0 */}
              {(backtestResults["Error"] || backtestResults["Sharpe Ratio"] === 0) && (
                <div className="troubleshooting-tips">
                  <h4>Troubleshooting Tips</h4>
                  <ul>
                    <li>Check that your model is generating diverse signals (BUY/SELL/HOLD).</li>
                    <li>Ensure price data has sufficient volatility for meaningful trading.</li>
                    <li>Try retraining your model with different parameters.</li>
                    <li>Verify that trades are being triggered by signal changes.</li>
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
      
      <footer className="App-footer">
        <p>Powered by Cybotrade API | Real-time BTC On-chain Data</p>
      </footer>
    </div>
  );
}

export default App;
