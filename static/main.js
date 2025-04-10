document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const symbolSelect = document.getElementById('symbol');
    const intervalSelect = document.getElementById('interval');
    const daysInput = document.getElementById('days');
    const runBacktestBtn = document.getElementById('run-backtest');
    const loadingElement = document.getElementById('loading');
    const resultsContainer = document.getElementById('results-container');
    
    // Metrics elements
    const sharpeRatioElement = document.getElementById('sharpe-ratio');
    const maxDrawdownElement = document.getElementById('max-drawdown');
    const tradeFrequencyElement = document.getElementById('trade-frequency');
    const finalValueElement = document.getElementById('final-value');
    
    // Chart
    let priceChart = null;
    
    // Run backtest button click handler
    runBacktestBtn.addEventListener('click', async function() {
        // Show loading, hide results
        loadingElement.style.display = 'flex';
        resultsContainer.style.display = 'none';
        
        try {
            // Get form values
            const symbol = symbolSelect.value;
            const interval = intervalSelect.value;
            const days = parseInt(daysInput.value);
            
            // Run backtest API call
            const response = await fetch('/api/run_backtest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbol: symbol,
                    interval: interval,
                    days: days
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Update metrics
                sharpeRatioElement.textContent = data.metrics['Sharpe Ratio'];
                maxDrawdownElement.textContent = '-' + data.metrics['Max Drawdown'] + '%';
                tradeFrequencyElement.textContent = data.metrics['Trade Frequency'] + '%';
                finalValueElement.textContent = '$' + data.metrics['Final Portfolio Value'].toLocaleString();
                
                // Update chart
                createOrUpdateChart(data.chart_data, data.signals);
                
                // Show results
                resultsContainer.style.display = 'block';
            } else {
                alert('Error running backtest: ' + data.message);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while running the backtest. Please try again.');
        } finally {
            // Hide loading
            loadingElement.style.display = 'none';
        }
    });
    
    // Function to create or update the chart
    function createOrUpdateChart(chartData, signals) {
        // Prepare data for the chart
        const timestamps = chartData.map(item => new Date(item.timestamp));
        const prices = chartData.map(item => item.price);
        const portfolioValues = chartData.map(item => item.portfolio_value);
        
        // Prepare buy/sell signals
        const buySignals = signals.filter(s => s.type === 'buy').map(s => ({
            x: new Date(s.timestamp),
            y: s.price
        }));
        
        const sellSignals = signals.filter(s => s.type === 'sell').map(s => ({
            x: new Date(s.timestamp),
            y: s.price
        }));
        
        // If chart already exists, destroy it
        if (priceChart) {
            priceChart.destroy();
        }
        
        // Get canvas context
        const ctx = document.getElementById('price-chart').getContext('2d');
        
        // Create new chart
        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timestamps,
                datasets: [
                    {
                        label: 'Price',
                        data: prices,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 2,
                        pointRadius: 0,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Portfolio Value',
                        data: portfolioValues,
                        borderColor: 'rgba(153, 102, 255, 1)',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        borderWidth: 2,
                        pointRadius: 0,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Buy Signals',
                        data: buySignals,
                        borderColor: 'rgba(75, 192, 192, 0)',
                        backgroundColor: 'rgba(75, 192, 192, 0)',
                        pointBackgroundColor: 'rgba(0, 200, 0, 1)',
                        pointBorderColor: 'rgba(0, 200, 0, 1)',
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        showLine: false,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Sell Signals',
                        data: sellSignals,
                        borderColor: 'rgba(75, 192, 192, 0)',
                        backgroundColor: 'rgba(75, 192, 192, 0)',
                        pointBackgroundColor: 'rgba(255, 0, 0, 1)',
                        pointBorderColor: 'rgba(255, 0, 0, 1)',
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        showLine: false,
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day'
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Price'
                        }
                    },
                    y1: {
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Portfolio Value'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += '$' + context.parsed.y.toFixed(2);
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Run a backtest automatically on page load
    runBacktestBtn.click();
});