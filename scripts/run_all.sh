#!/bin/bash

# Set the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting the ML pipeline execution..."

# Step 1: Fetch the latest data from Cybotrade API
echo "Fetching the latest BTC on-chain metrics data..."
python ml/fetch_data.py
if [ $? -ne 0 ]; then
    echo "Error fetching data. Aborting pipeline."
    exit 1
fi
echo "Data fetching completed successfully."

# Step 2: Train the ML model and save it
echo "Training the trading model..."
python ml/train_and_save.py
if [ $? -ne 0 ]; then
    echo "Error training model. Aborting pipeline."
    exit 1
fi
echo "Model training completed successfully."

# Step 3: Start the Flask backend server
echo "Starting the Flask backend server..."
python backend/app.py
if [ $? -ne 0 ]; then
    echo "Error starting Flask server."
    exit 1
fi 