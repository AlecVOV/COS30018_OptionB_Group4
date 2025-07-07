import yfinance as yf

# Create ticker object
ticker = yf.Ticker("CBA.AX")

# Get historical data
data = ticker.history(period="10y")

# Print the dataframe
print(data)

# Save to CSV
data.to_csv("cba_stock_data.csv")

# Get current price using `.fast_info` (safer than `.info`)
print("Current Price:", ticker.fast_info['last_price'])
