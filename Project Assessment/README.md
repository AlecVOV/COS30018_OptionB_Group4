# Stock Prediction with Machine Learning and Sentiment Analysis

A comprehensive stock price prediction system that combines multiple machine learning models with sentiment analysis from news and social media data. This project implements various neural networks (LSTM, GRU, RNN), traditional ML models (Random Forest, ARIMA), and ensemble methods to predict stock prices.

## 🚀 Features

### Core Functionality
- **Multiple ML Models**: LSTM, GRU, SimpleRNN, Random Forest, and ARIMA
- **Sentiment Analysis**: Integration of news and social media sentiment using VADER
- **Ensemble Methods**: Simple averaging, weighted ensembles, and stacked meta-learning
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Volatility
- **Interactive Visualizations**: Candlestick charts, prediction comparisons, error analysis
- **Future Predictions**: Multi-step ahead forecasting with confidence bands

### Advanced Features
- **Sentiment Data Collection**: Automated news and social media data gathering
- **Feature Engineering**: Technical indicators and sentiment-based features
- **Model Evaluation**: Comprehensive performance metrics and cross-validation
- **Data Caching**: Efficient sentiment data caching system
- **Flexible Data Splitting**: Sequential, date-based, or random data splitting
- **Export Capabilities**: CSV exports for predictions and model performance

## 📁 Project Structure

```
project/
├── data/                           # Data storage directory
│   ├── sentiment_cache/           # Cached sentiment data
│   ├── {SYMBOL}_data.csv         # Raw stock data
│   ├── {SYMBOL}_cleaned_data.csv # Processed data with features
│   ├── {SYMBOL}_predictions_*.csv # Model predictions
│   └── stock_prediction.log      # Application logs
├── models/                        # Trained model storage
│   ├── {SYMBOL}_lstm_model.keras # LSTM model
│   ├── {SYMBOL}_gru_model.keras  # GRU model
│   ├── {SYMBOL}_rnn_model.keras  # RNN model
│   ├── {SYMBOL}_rf_model.pkl     # Random Forest model
│   └── {SYMBOL}_arima_model.pkl  # ARIMA model
└── v0.7_codebase_stock_prediction.ipynb # Main notebook
```

## 🛠️ Installation

### 1. Clone or Download
Download the notebook file to your desired directory.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional: API Keys (for enhanced sentiment analysis)
Create a `.env` file in the project directory:
```env
NEWS_API_KEY=your_newsapi_key_here
```

## 📊 Usage

### Basic Usage
1. Open the Jupyter notebook:
   ```bash
   jupyter notebook v0.7_codebase_stock_prediction.ipynb
   ```

2. Run cells sequentially, following the prompts for:
   - **Stock Symbol**: Default is 'CBA.AX' (Commonwealth Bank of Australia)
   - **Date Range**: Training period (default: 2020-01-01 to 2023-01-01)
   - **Sentiment Analysis**: Enable/disable sentiment features
   - **Data Splitting**: Choose splitting method (sequential/date-based/random)

### Interactive Features

#### Model Selection
The notebook provides interactive menus for:
- **Visualization**: Choose which model predictions to visualize
- **Candlestick Charts**: Customizable time periods and data aggregation
- **Boxplot Analysis**: Historical data and prediction error analysis
- **Future Predictions**: Select models for multi-day forecasting

#### Customization Options
- **Prediction Window**: Adjust `PREDICTION_DAYS` (default: 60 days)
- **Future Forecast**: Specify number of days to predict ahead
- **Technical Indicators**: Modify moving average windows and RSI periods
- **Model Architecture**: Customize neural network layers and parameters

## 🧠 Models Implemented

### Individual Models
1. **LSTM**: Long Short-Term Memory networks for sequence modeling
2. **GRU**: Gated Recurrent Units for efficient sequence processing
3. **SimpleRNN**: Basic recurrent neural networks
4. **Random Forest**: Ensemble tree-based regression
5. **ARIMA**: Autoregressive Integrated Moving Average

### Ensemble Methods
1. **Simple Averaging**: Equal weight combination of models
2. **Weighted Ensemble**: Performance-based weighting
3. **Stacked Ensemble**: Meta-learner trained on base model predictions

### Technical Indicators
- **Moving Averages**: 5, 20, 50-day periods
- **RSI**: Relative Strength Index (14-day)
- **Bollinger Bands**: 20-day with 2 standard deviations
- **MACD**: Moving Average Convergence Divergence
- **Volatility**: Rolling standard deviation

### Sentiment Features
- **News Sentiment**: Daily aggregated news sentiment scores
- **Social Media**: Twitter/social platform sentiment analysis
- **Sentiment Momentum**: Rate of change in sentiment
- **Volume Metrics**: News and social media mention volumes

## 📈 Output and Results

### Generated Files
- **Model Files**: Trained models saved in `/models/` directory
- **Predictions**: CSV files with detailed predictions and errors
- **Performance Metrics**: Model comparison and ranking
- **Future Forecasts**: Multi-day predictions with confidence levels

### Visualizations
- **Interactive Dashboard**: Comprehensive analysis with multiple subplots
- **Candlestick Charts**: Historical prices with prediction overlays
- **Error Analysis**: Distribution plots and statistical summaries
- **Performance Comparison**: Model ranking and metrics visualization

### Performance Metrics
- **MAE**: Mean Absolute Error (primary metric)
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **Percentage Error**: Relative to current stock price

## 🔧 Configuration Options

### Model Parameters
```python
# LSTM Configuration
lstm_layers = [
    {'type': 'lstm', 'units': 100, 'return_sequences': True},
    {'type': 'dropout', 'rate': 0.2},
    {'type': 'lstm', 'units': 50, 'return_sequences': False},
    {'type': 'dense', 'units': 25, 'activation': 'relu'}
]

# Random Forest Configuration
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
```

### Sentiment Configuration
```python
sentiment_config = SentimentConfig(
    enabled=True,
    sentiment_analyzer='vader',
    cache_duration_hours=24,
    max_articles_per_day=50,
    max_posts_per_day=100
)
```

## 🚨 Important Notes

### Data Requirements
- **Minimum Data**: At least 100 trading days for reliable predictions
- **Feature Availability**: Some technical indicators require sufficient historical data
- **Date Format**: Ensure dates are in 'YYYY-MM-DD' format

### Performance Considerations
- **Training Time**: Neural networks may take several minutes to train
- **Memory Usage**: Large datasets may require significant RAM
- **API Limits**: News API has rate limits for sentiment analysis

### Limitations
- **Market Volatility**: Models may struggle during extreme market events
- **Sentiment Availability**: Limited to English-language sources
- **No Real-time Data**: Uses historical data for training and testing

## 🤝 Contributing

### Extending the Project
1. **New Models**: Add implementations in the model training section
2. **Features**: Extend technical indicators or sentiment sources
3. **Visualizations**: Create additional chart types or analysis views
4. **Data Sources**: Integrate new financial data providers

### Code Structure
- **Modular Design**: Each model type has dedicated training cells
- **Error Handling**: Comprehensive try-catch blocks for robustness
- **Logging**: Detailed logging for debugging and monitoring
- **Caching**: Efficient data caching to reduce API calls

## 📝 License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service when using financial and news APIs.

## ⚠️ Disclaimer

**This software is for educational purposes only. Do not use for actual trading or investment decisions. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.**

## 📞 Support

For questions or issues:
1. Check the notebook comments and documentation
2. Review error messages in the log files
3. Ensure all dependencies are correctly installed
4. Verify API keys are properly configured (if using sentiment analysis)

---

**Happy Analyzing! 📊**
