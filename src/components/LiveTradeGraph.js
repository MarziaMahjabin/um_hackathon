import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';

const LiveTradeGraph = ({ predictionHistory, priceHistory, latestPrediction }) => {
  const [candleData, setCandleData] = useState(null);
  const chartRef = useRef(null);
  
  useEffect(() => {
    // Process price history data for the chart when it updates
    if (priceHistory && priceHistory.prices && priceHistory.timestamps) {
      processPriceData();
    }
  }, [priceHistory]);
  
  const processPriceData = () => {
    // Prepare data for the candlestick-like visualization
    const prices = priceHistory.prices;
    const timestamps = priceHistory.timestamps;
    
    // Create gradient colors for "candles" based on price movement
    const colors = [];
    for (let i = 1; i < prices.length; i++) {
      // Green for price increase, red for decrease
      colors.push(prices[i] >= prices[i-1] ? 'rgba(75, 192, 75, 0.8)' : 'rgba(255, 99, 71, 0.8)');
    }
    
    // Add the first color (can't determine movement for first point)
    colors.unshift('rgba(75, 192, 192, 0.8)');
    
    setCandleData({
      labels: timestamps.map(ts => formatDate(ts)),
      prices,
      colors
    });
  };
  
  // Format date for display
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
  };
  
  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
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
            if (context.dataset.label === 'Price') {
              return `Price: $${context.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            } else if (context.dataset.label === 'Signals') {
              // Find matching prediction from history if available
              if (predictionHistory && predictionHistory.length > 0) {
                const matchingIndex = predictionHistory.findIndex(p => 
                  new Date(p.timestamp).toDateString() === new Date(priceHistory.timestamps[context.dataIndex]).toDateString()
                );
                if (matchingIndex >= 0) {
                  const signal = predictionHistory[matchingIndex].signal;
                  return `Signal: ${signal ? signal.toUpperCase() : 'NONE'}`;
                }
              }
              return 'Signal: N/A';
            }
            return context.dataset.label + ': ' + context.parsed.y;
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
          },
          maxRotation: 45,
          minRotation: 45
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Price (USD)',
          color: '#fff'
        },
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
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Signal',
          color: '#fff'
        },
        min: -0.5,
        max: 2.5,
        ticks: {
          color: '#fff',
          font: {
            size: 11
          },
          callback: function(value) {
            if (value === 0) return 'SELL';
            if (value === 1) return 'HOLD';
            if (value === 2) return 'BUY';
            return '';
          }
        },
        grid: {
          drawOnChartArea: false
        }
      }
    },
    animations: {
      tension: {
        duration: 1000,
        easing: 'linear',
        from: 0.8,
        to: 0.2,
        loop: true
      }
    }
  };
  
  // Function to map signal strings to numeric values
  const mapSignalToValue = (signal) => {
    if (!signal) return 1; // Default to HOLD
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
  
  // Get data for signals overlay
  const getSignalData = () => {
    if (!priceHistory || !priceHistory.timestamps || !predictionHistory || predictionHistory.length === 0) {
      return Array(priceHistory?.timestamps?.length || 0).fill(null);
    }
    
    // Map each price timestamp to the nearest prediction signal
    return priceHistory.timestamps.map(ts => {
      const priceDate = new Date(ts);
      // Find closest prediction by date
      let closestPrediction = null;
      let smallestDiff = Infinity;
      
      predictionHistory.forEach(pred => {
        const predDate = new Date(pred.timestamp);
        const diff = Math.abs(priceDate - predDate);
        if (diff < smallestDiff) {
          smallestDiff = diff;
          closestPrediction = pred;
        }
      });
      
      // Only use prediction if it's within 1 day of the price point
      if (closestPrediction && smallestDiff < 24 * 60 * 60 * 1000) {
        return mapSignalToValue(closestPrediction.signal);
      }
      return null;
    });
  };
  
  // Apply current prediction signal to the chart
  const getSignalPoints = () => {
    if (!latestPrediction || !latestPrediction.signal || !priceHistory || !priceHistory.prices) {
      return [];
    }
    
    // Create point for the current prediction
    const signalValue = mapSignalToValue(latestPrediction.signal);
    const lastPrice = priceHistory.prices[priceHistory.prices.length - 1];
    
    // Only show the current signal at the last price point
    const points = Array(priceHistory.prices.length).fill(null);
    points[points.length - 1] = signalValue;
    
    return points;
  };
  
  const getSignalColors = () => {
    if (!predictionHistory || predictionHistory.length === 0) {
      return [];
    }
    
    return predictionHistory.map(pred => {
      switch(pred.signal?.toLowerCase()) {
        case 'buy':
          return '#4CAF50'; // Green
        case 'sell':
          return '#F44336'; // Red
        case 'hold':
          return '#FFC107'; // Amber
        default:
          return '#888';    // Gray
      }
    });
  };
  
  // Render loading state or chart
  if (!candleData || !priceHistory) {
    return (
      <div className="live-graph-container">
        <div className="chart-content">
          <div className="placeholder-content">
            <p>Loading price data...</p>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="live-graph-container">
      <div className="chart-content">
        <Line
          ref={chartRef}
          options={chartOptions}
          data={{
            labels: candleData.labels,
            datasets: [
              {
                label: 'Price',
                data: candleData.prices,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderWidth: 2,
                pointBackgroundColor: candleData.colors,
                pointBorderColor: '#fff',
                pointRadius: 5,
                pointHoverRadius: 7,
                yAxisID: 'y'
              },
              {
                label: 'Signals',
                data: getSignalData(),
                borderColor: 'rgba(255, 159, 64, 0.7)',
                backgroundColor: 'rgba(255, 159, 64, 0.5)', 
                pointBackgroundColor: getSignalColors(),
                pointBorderColor: '#fff',
                pointRadius: 6,
                pointHoverRadius: 8,
                borderDash: [5, 5],
                stepped: 'before',
                yAxisID: 'y1'
              },
              {
                label: 'Current Signal',
                data: getSignalPoints(),
                borderColor: 'transparent',
                backgroundColor: 'transparent',
                pointBackgroundColor: latestPrediction ? 
                  (latestPrediction.signal === 'buy' ? '#4CAF50' : 
                   latestPrediction.signal === 'sell' ? '#F44336' : '#FFC107') : '#888',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 10,
                pointHoverRadius: 12,
                pointStyle: 'star',
                yAxisID: 'y1'
              }
            ]
          }}
        />
      </div>
      <div className="chart-legend-container">
        <div className="chart-legend">
          <div className="legend-item">
            <div className="legend-color price-color"></div>
            <span>Price</span>
          </div>
          <div className="legend-item">
            <div className="legend-color buy-color"></div>
            <span>Buy Signal</span>
          </div>
          <div className="legend-item">
            <div className="legend-color hold-color"></div>
            <span>Hold Signal</span>
          </div>
          <div className="legend-item">
            <div className="legend-color sell-color"></div>
            <span>Sell Signal</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveTradeGraph; 