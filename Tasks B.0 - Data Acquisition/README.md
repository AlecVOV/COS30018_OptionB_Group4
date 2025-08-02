# Stock Data Acquisition - CBA Stock Prediction

This project contains a Python script for acquiring historical stock data for Commonwealth Bank of Australia (CBA) using the Yahoo Finance API.

## Overview

The `v0_codebase_stock_prediction.py` script is designed to fetch and store historical stock data for CBA (ASX: CBA.AX) for use in stock price prediction models. This is the initial data acquisition phase of a larger stock prediction project.

## Features

- Fetches 10 years of historical stock data for CBA.AX
- Displays the complete dataset in the console
- Saves data to CSV format for further analysis
- Retrieves current stock price information

## Requirements

Before running the script, ensure you have the following dependencies installed:

```bash
pip install yfinance
```

## Installation

1. Clone or download the project files
2. Install the required dependencies:
   ```bash
   pip install yfinance
   ```
3. Run the script:
   ```bash
   python v0_codebase_stock_prediction.py
   ```

## Usage

Simply execute the Python script:

```bash
python v0_codebase_stock_prediction.py
```

## Output

The script will:
1. Display the historical stock data in the console
2. Create a CSV file named `cba_stock_data.csv` containing:
   - Date
   - Open price
   - High price
   - Low price
   - Close price
   - Volume
   - Dividends
   - Stock splits
3. Print the current stock price

## Data Structure

The retrieved data includes the following columns:
- **Date**: Trading date (index)
- **Open**: Opening price for the day
- **High**: Highest price during the day
- **Low**: Lowest price during the day
- **Close**: Closing price for the day
- **Volume**: Number of shares traded
- **Dividends**: Dividend payments (if any)
- **Stock Splits**: Stock split information (if any)

## File Structure

```
Tasks B.0 - Data Acquisition/
├── v0_codebase_stock_prediction.py    # Main data acquisition script
├── cba_stock_data.csv                 # Generated output file (after running)
└── README.md                          # This file
```

## Notes

- The script uses the Yahoo Finance API through the `yfinance` library
- CBA.AX refers to Commonwealth Bank of Australia listed on the Australian Securities Exchange
- The data period is set to 10 years from the current date
- The `.fast_info` method is used for current price retrieval as it's more reliable than `.info`

## Next Steps

This data acquisition script serves as the foundation for:
- Data preprocessing and cleaning
- Feature engineering
- Machine learning model development
- Stock price prediction analysis

## Troubleshooting

If you encounter any issues:
1. Ensure you have a stable internet connection (required for Yahoo Finance API)
2. Verify that the `yfinance` library is properly installed
3. Check if the ticker symbol "CBA.AX" is still valid on Yahoo Finance

## Version

- **Version**: 0.0 (Initial Data Acquisition)
- **Author**: Le Hoang Triet Thong
- **Course**: COS30018 - Intelligent Systems
- **Assignment**: Project Assignment Option B
