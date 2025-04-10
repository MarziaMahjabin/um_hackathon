import React, { useState, useMemo } from 'react';

const SignalHistory = ({ signalHistory }) => {
  const [filterType, setFilterType] = useState('all');
  const [showDetails, setShowDetails] = useState(false);
  
  // Function to get signal color
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

  // Format date function
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };

  // Calculate signal counts, trend, performance, and timeline data
  const { 
    signalCounts, 
    trend, 
    filteredHistory, 
    performanceMetrics,
    timelineData 
  } = useMemo(() => {
    if (!signalHistory || signalHistory.length === 0) {
      return { 
        signalCounts: {}, 
        trend: 'neutral', 
        filteredHistory: [],
        performanceMetrics: {
          profitability: 0,
          accuracy: 0,
          averageHoldTime: 0
        },
        timelineData: []
      };
    }
    
    // Count signals by type
    const counts = signalHistory.reduce((acc, item) => {
      const signal = item.signal.toLowerCase();
      acc[signal] = (acc[signal] || 0) + 1;
      return acc;
    }, {});
    
    // Calculate signal trend (last 5 signals)
    const recentSignals = signalHistory.slice(0, 5);
    let buyPoints = 0;
    let sellPoints = 0;
    
    recentSignals.forEach((item, index) => {
      const weight = 5 - index; // More recent signals have higher weight
      if (item.signal.toLowerCase() === 'buy') {
        buyPoints += weight;
      } else if (item.signal.toLowerCase() === 'sell') {
        sellPoints += weight;
      }
    });
    
    let signalTrend = 'neutral';
    if (buyPoints > sellPoints + 3) {
      signalTrend = 'bullish';
    } else if (sellPoints > buyPoints + 3) {
      signalTrend = 'bearish';
    }
    
    // Apply filter
    const filtered = filterType === 'all' 
      ? signalHistory 
      : signalHistory.filter(item => item.signal.toLowerCase() === filterType);
    
    // Calculate mock performance metrics (in a real app, these would come from backend)
    const mockPerformance = {
      buy: {
        profitability: 68,
        accuracy: 72,
        averageHoldTime: 4.2
      },
      sell: {
        profitability: 58,
        accuracy: 61,
        averageHoldTime: 3.5
      },
      hold: {
        profitability: 35,
        accuracy: 55,
        averageHoldTime: 5.8
      }
    };
    
    // Calculate performance metrics based on signal distribution
    const totalSignals = Object.values(counts).reduce((sum, count) => sum + count, 0);
    let avgProfitability = 0;
    let avgAccuracy = 0;
    let avgHoldTime = 0;
    
    if (totalSignals > 0) {
      Object.entries(counts).forEach(([signal, count]) => {
        const metrics = mockPerformance[signal] || { profitability: 50, accuracy: 50, averageHoldTime: 3 };
        avgProfitability += (metrics.profitability * count / totalSignals);
        avgAccuracy += (metrics.accuracy * count / totalSignals);
        avgHoldTime += (metrics.averageHoldTime * count / totalSignals);
      });
    }
    
    const performanceMetrics = {
      profitability: Math.round(avgProfitability),
      accuracy: Math.round(avgAccuracy),
      averageHoldTime: avgHoldTime.toFixed(1)
    };
    
    // Create timeline data for visualization
    const timelineData = signalHistory.slice(0, 14).map(item => ({
      date: new Date(item.timestamp),
      signal: item.signal.toLowerCase()
    })).reverse();
    
    return { 
      signalCounts: counts, 
      trend: signalTrend,
      filteredHistory: filtered,
      performanceMetrics,
      timelineData
    };
  }, [signalHistory, filterType]);

  // Total signal count
  const totalSignals = Object.values(signalCounts).reduce((sum, count) => sum + count, 0);

  // Create distribution chart data
  const renderDistributionChart = () => {
    const total = totalSignals;
    if (total === 0) return null;
    
    const buyPercentage = ((signalCounts.buy || 0) / total) * 100;
    const sellPercentage = ((signalCounts.sell || 0) / total) * 100;
    const holdPercentage = ((signalCounts.hold || 0) / total) * 100;
    
    return (
      <div className="signal-distribution-chart">
        <div className="distribution-bar">
          <div 
            className="distribution-segment buy-segment" 
            style={{ width: `${buyPercentage}%` }}
            title={`Buy: ${buyPercentage.toFixed(1)}%`}
          ></div>
          <div 
            className="distribution-segment hold-segment" 
            style={{ width: `${holdPercentage}%` }}
            title={`Hold: ${holdPercentage.toFixed(1)}%`}
          ></div>
          <div 
            className="distribution-segment sell-segment" 
            style={{ width: `${sellPercentage}%` }}
            title={`Sell: ${sellPercentage.toFixed(1)}%`}
          ></div>
        </div>
        <div className="distribution-labels">
          <div className="distribution-label">
            <span className="distribution-color buy-color"></span>
            <span className="distribution-text">Buy ({buyPercentage.toFixed(1)}%)</span>
          </div>
          <div className="distribution-label">
            <span className="distribution-color hold-color"></span>
            <span className="distribution-text">Hold ({holdPercentage.toFixed(1)}%)</span>
          </div>
          <div className="distribution-label">
            <span className="distribution-color sell-color"></span>
            <span className="distribution-text">Sell ({sellPercentage.toFixed(1)}%)</span>
          </div>
        </div>
      </div>
    );
  };

  // Render signal timeline visualization
  const renderTimeline = () => {
    return (
      <div className="signal-timeline">
        <h4 className="timeline-title">Recent Signals Timeline</h4>
        <div className="timeline-container">
          {timelineData.map((item, index) => (
            <div 
              key={index} 
              className="timeline-item" 
              title={`${formatDate(item.date)}: ${item.signal.toUpperCase()}`}
            >
              <div 
                className={`timeline-dot ${item.signal}-dot`} 
                style={{ backgroundColor: getSignalColor(item.signal) }}
              ></div>
              {index < timelineData.length - 1 && (
                <div className="timeline-line"></div>
              )}
            </div>
          ))}
        </div>
        <div className="timeline-dates">
          <span>{timelineData.length > 0 ? formatDate(timelineData[0].date) : ''}</span>
          <span>{timelineData.length > 0 ? formatDate(timelineData[timelineData.length - 1].date) : ''}</span>
        </div>
      </div>
    );
  };

  if (!signalHistory || signalHistory.length === 0) {
    return <div className="loading">No signal history available</div>;
  }

  return (
    <div className="signal-history-container">
      {/* Signal header with toggle button */}
      <div className="signal-header">
        <h4 className="signal-section-title">Signal Distribution</h4>
        <button 
          className="toggle-details-btn"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? 'Hide Details' : 'Show Details'}
        </button>
      </div>
      
      {/* Distribution chart */}
      {renderDistributionChart()}
      
      <div className="signal-summary-stats">
        <div className="signal-stat-item">
          <div className="signal-count buy-count">
            {signalCounts.buy || 0}
          </div>
          <div className="signal-label">Buy</div>
        </div>
        <div className="signal-stat-item">
          <div className="signal-count hold-count">
            {signalCounts.hold || 0}
          </div>
          <div className="signal-label">Hold</div>
        </div>
        <div className="signal-stat-item">
          <div className="signal-count sell-count">
            {signalCounts.sell || 0}
          </div>
          <div className="signal-label">Sell</div>
        </div>
      </div>
      
      {/* Trend indicator */}
      <div className={`trend-indicator-bar ${trend}`}>
        <div className="trend-icon">
          {trend === 'bullish' ? '↗' : trend === 'bearish' ? '↘' : '→'}
        </div>
        <div className="trend-text">
          {trend === 'bullish' 
            ? 'Bullish trend developing' 
            : trend === 'bearish' 
              ? 'Bearish trend developing' 
              : 'No clear trend'}
        </div>
      </div>
      
      {/* Signal timeline visualization */}
      {renderTimeline()}
      
      {/* Performance metrics */}
      {showDetails && (
        <div className="signal-performance">
          <h4 className="performance-title">Performance Metrics</h4>
          <div className="performance-metrics">
            <div className="performance-metric">
              <div className="metric-value">{performanceMetrics.profitability}%</div>
              <div className="metric-label">Profitability</div>
            </div>
            <div className="performance-metric">
              <div className="metric-value">{performanceMetrics.accuracy}%</div>
              <div className="metric-label">Accuracy</div>
            </div>
            <div className="performance-metric">
              <div className="metric-value">{performanceMetrics.averageHoldTime}d</div>
              <div className="metric-label">Avg Hold</div>
            </div>
          </div>
          <div className="performance-note">
            * Based on historical signal performance
          </div>
        </div>
      )}
      
      {/* Filter tabs */}
      <div className="signal-list-header">
        <h4 className="signal-section-title">Signal History</h4>
        <div className="signal-filter-tabs">
          <button 
            className={`filter-tab ${filterType === 'all' ? 'active' : ''}`}
            onClick={() => setFilterType('all')}
          >
            All
          </button>
          <button 
            className={`filter-tab ${filterType === 'buy' ? 'active' : ''}`}
            onClick={() => setFilterType('buy')}
          >
            Buy
          </button>
          <button 
            className={`filter-tab ${filterType === 'hold' ? 'active' : ''}`}
            onClick={() => setFilterType('hold')}
          >
            Hold
          </button>
          <button 
            className={`filter-tab ${filterType === 'sell' ? 'active' : ''}`}
            onClick={() => setFilterType('sell')}
          >
            Sell
          </button>
        </div>
      </div>
      
      <div className="signal-history-grid">
        <div className="signal-history-header">
          <div className="signal-date">Date</div>
          <div className="signal-type">Signal</div>
        </div>
        <div className="signal-history-body">
          {filteredHistory.map((item, index) => (
            <div key={index} className="signal-history-row">
              <div className="signal-date">{formatDate(item.timestamp)}</div>
              <div 
                className="signal-type"
                style={{ 
                  backgroundColor: getSignalColor(item.signal),
                  color: "#fff",
                  padding: "4px 8px",
                  borderRadius: "4px",
                  fontWeight: "bold"
                }}
              >
                {item.signal.toUpperCase()}
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {filteredHistory.length === 0 && (
        <div className="no-results">No signals match the selected filter</div>
      )}
      
      {/* Signal descriptions */}
      {showDetails && (
        <div className="signal-descriptions">
          <h4 className="description-title">Signal Definitions</h4>
          <div className="description-item">
            <div className="description-label buy">BUY</div>
            <div className="description-text">
              Indicates positive market conditions and suggests entering a long position.
            </div>
          </div>
          <div className="description-item">
            <div className="description-label hold">HOLD</div>
            <div className="description-text">
              Suggests maintaining current positions without taking new action.
            </div>
          </div>
          <div className="description-item">
            <div className="description-label sell">SELL</div>
            <div className="description-text">
              Indicates negative market conditions and suggests closing long positions.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SignalHistory; 