# Advanced Stock Prediction with Visualization Tools - v0.3

This Jupyter notebook represents an advanced stock price prediction system with comprehensive data processing, LSTM neural networks, and sophisticated visualization capabilities including candlestick charts and statistical analysis tools.

## Overview

The `v0.3_codebase_stock_prediction.ipynb` notebook extends the previous version with powerful visualization and analysis tools. It combines machine learning prediction capabilities with professional-grade financial charting and statistical analysis, making it suitable for both academic research and practical trading analysis.

## Features

### Core Prediction System
- **LSTM Neural Network**: Multi-layer LSTM architecture for time series prediction
- **Interactive Configuration**: User-configurable training periods and parameters
- **Model Persistence**: Automatic saving and loading of trained models
- **Comprehensive Metrics**: MAE, RMSE, MAPE, and accuracy calculations

### Advanced Data Processing
- **Multi-Feature Scaling**: Individual scalers for each feature (Close, High, Low, Open, Volume)
- **Flexible Data Splitting**: Sequential, date-based, and random splitting methods
- **Sequence Generation**: Configurable lookback periods for time series analysis
- **Data Validation**: Robust error handling and data integrity checks

### Professional Visualization Tools

#### Candlestick Charts
- **Interactive Candlestick Plotting**: Professional financial chart visualization
- **Multiple Time Frames**: Daily, weekly, monthly aggregation support
- **Volume Integration**: Synchronized volume bars with price data
- **Prediction Comparison**: Side-by-side actual vs predicted visualizations
- **Customizable Display**: Adjustable timeframes and chart parameters

#### Statistical Analysis
- **Boxplot Analysis**: Comprehensive statistical distribution visualization
- **Multiple Grouping Options**: Daily, weekly, monthly, quarterly, yearly analysis
- **Price Range Analysis**: Volatility and price movement statistics
- **Comparative Analysis**: Multi-metric statistical comparisons
- **Interactive Selection**: User-configurable analysis parameters

### Enhanced User Experience
- **Interactive Mode**: User-friendly parameter selection interfaces
- **Comprehensive Error Handling**: Graceful handling of missing data and errors
- **Detailed Documentation**: Inline help and usage examples
- **Professional Output**: High-quality charts suitable for presentations

## Requirements

### Dependencies
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn yfinance seaborn mplfinance
```

### Detailed Requirements
- **numpy**: Numerical computing and array operations
- **pandas**: Data manipulation and analysis
- **matplotlib**: Basic plotting and chart creation
- **tensorflow**: Deep learning framework for LSTM models
- **scikit-learn**: Machine learning utilities and preprocessing
- **yfinance**: Yahoo Finance API for stock data retrieval
- **seaborn**: Statistical data visualization
- **mplfinance**: Specialized financial charting library

### System Requirements
- Python 3.7 or higher
- Minimum 4GB RAM (8GB recommended)
- GPU support recommended for faster training
- Display capability for interactive charts

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Tasks B.3 - Data Processing 2/v0.3 Codebase"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook v0.3_codebase_stock_prediction.ipynb
   ```

## Usage Guide

### 1. Library Imports and Setup
The notebook begins with comprehensive library imports for all required functionality including visualization tools.

### 2. Data Loading and Configuration
- **Interactive Date Selection**: Configure training periods with validation
- **Automatic Data Management**: Intelligent caching and file management
- **Error Handling**: Robust validation for date formats and ranges

### 3. Data Preprocessing
- **Data Cleaning**: Advanced preprocessing with column mapping and type conversion
- **Flexible Splitting**: Choose from three splitting methodologies:
  - Sequential split (chronological order preserved)
  - Date-based split (specific cutoff dates)
  - Random split (statistical sampling)

### 4. Feature Engineering
- **Advanced Scaling**: Individual MinMaxScalers for optimal normalization
- **Scaler Persistence**: Automatic saving for consistent transformations
- **Feature Selection**: Dynamic feature inclusion based on data availability

### 5. LSTM Model Development
- **Sophisticated Architecture**: 3-layer LSTM with dropout regularization
- **Smart Training**: Automatic model loading or training from scratch
- **Performance Monitoring**: Real-time training metrics and validation

### 6. Prediction and Analysis
- **Comprehensive Evaluation**: Multiple performance metrics calculation
- **Detailed Reporting**: Statistical analysis of prediction accuracy
- **Results Export**: CSV export for further analysis

### 7. Visualization Tools

#### Candlestick Charts
```python
# Interactive candlestick chart display
interactive_chart_display()

# Direct chart display with parameters
display_stock_charts('CBA.AX', n_days=1, limit_days=90)
```

#### Statistical Analysis
```python
# Interactive boxplot analysis
interactive_boxplot_display()

# Direct boxplot display
display_stock_boxplots('CBA.AX', period='monthly')
```

## File Structure

```
v0.3 Codebase/
├── v0.3_codebase_stock_prediction.ipynb  # Main notebook
├── README.md                             # This documentation
├── data/                                 # Generated data directory
│   ├── CBA.AX_data.csv                  # Raw Yahoo Finance data
│   ├── CBA.AX_cleaned_data.csv          # Processed and cleaned data
│   ├── CBA.AX_scalers.pkl               # Feature scaling parameters
│   ├── CBA.AX_training_data.npz         # Preprocessed training sequences
│   ├── CBA.AX_test_predictions.csv      # Model predictions with metrics
│   └── CBA.AX_prediction_results.png    # Visualization outputs
├── models/                               # Generated models directory
│   ├── CBA.AX_lstm_model.h5             # Trained LSTM model
│   └── CBA.AX_training_history.npz      # Training metrics history
└── requirements.txt                      # Project dependencies
```

## Visualization Features

### Candlestick Charts
- **Professional Appearance**: Industry-standard financial chart styling
- **Interactive Parameters**: User-configurable timeframes and aggregation
- **Volume Integration**: Synchronized trading volume visualization
- **Prediction Overlay**: Direct comparison of actual vs predicted prices
- **Export Quality**: High-resolution output suitable for reports

### Statistical Analysis
- **Distribution Analysis**: Boxplots for price and volume distributions
- **Time-based Grouping**: Multiple temporal aggregation options
- **Comparative Metrics**: Side-by-side statistical comparisons
- **Outlier Detection**: Visual identification of unusual market behavior
- **Trend Analysis**: Long-term pattern recognition tools

## Key Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `COMPANY` | 'CBA.AX' | Target stock ticker symbol |
| `PREDICTION_DAYS` | 60 | Lookback period for LSTM sequences |
| `DEFAULT_TRAIN_START` | '2020-01-01' | Default training period start |
| `DEFAULT_TRAIN_END` | '2023-08-01' | Default training period end |
| `LSTM_UNITS` | 50 | LSTM layer units (per layer) |
| `DROPOUT_RATE` | 0.2 | Regularization dropout rate |
| `EPOCHS` | 25 | Training epochs |
| `BATCH_SIZE` | 32 | Training batch size |

### Visualization Parameters
| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `n_days` | 1 | 1, 5, 7, 30 | Days per candlestick |
| `limit_days` | 90 | Any integer | Recent days to display |
| `period` | 'monthly' | daily, weekly, monthly, quarterly, yearly | Boxplot grouping |
| `figsize` | (15, 8) | (width, height) | Chart dimensions |

## Model Performance

### Expected Metrics
- **Accuracy**: 85-95% for stable market conditions
- **MAPE**: 2-8% depending on volatility period
- **Training Time**: 2-10 minutes (hardware dependent)
- **Visualization**: Real-time chart generation

### Features Used
- **Price Data**: Open, High, Low, Close prices
- **Volume**: Trading volume indicators
- **Technical Indicators**: Price ranges and volatility measures
- **Time Features**: Date-based grouping and analysis

## Advanced Features

### Interactive User Interface
- **Parameter Selection**: User-friendly input prompts
- **Error Recovery**: Graceful handling of invalid inputs
- **Default Fallbacks**: Intelligent default value selection
- **Progress Indicators**: Real-time feedback during processing

### Professional Visualization
- **Financial Chart Standards**: Industry-compliant candlestick charts
- **Statistical Analysis**: Comprehensive boxplot and distribution analysis
- **Export Capabilities**: High-quality image generation
- **Customizable Appearance**: Flexible styling and formatting options

### Data Management
- **Intelligent Caching**: Efficient data storage and retrieval
- **Format Validation**: Robust data integrity checking
- **Scalable Architecture**: Handles various data sizes efficiently
- **Export Functionality**: Multiple output format support

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce `limit_days` parameter for visualization
   - Decrease `PREDICTION_DAYS` for model training
   - Use smaller batch sizes

2. **Visualization Issues**:
   - Ensure all required libraries are installed
   - Check display backend configuration
   - Verify data file existence and format

3. **Performance Issues**:
   - Use GPU acceleration if available
   - Reduce chart complexity for large datasets
   - Optimize timeframe selections

### Performance Optimization

1. **Chart Performance**:
   ```python
   # Optimize for large datasets
   display_stock_charts('CBA.AX', n_days=7, limit_days=60)
   ```

2. **Memory Management**:
   ```python
   # Clear plots after display
   plt.close('all')
   ```

3. **Data Caching**:
   ```python
   # Leverage existing cached data
   # Files are automatically reused when available
   ```

## Usage Examples

### Basic Prediction with Visualization
```python
# 1. Run data loading and preprocessing cells
# 2. Train or load model
# 3. Generate predictions
# 4. Display candlestick charts
interactive_chart_display()
```

### Statistical Analysis Workflow
```python
# 1. Complete prediction workflow
# 2. Run statistical analysis
interactive_boxplot_display()
```

### Custom Visualization
```python
# Daily candlesticks for last 30 days
display_stock_charts('CBA.AX', n_days=1, limit_days=30)

# Weekly analysis with boxplots
display_stock_boxplots('CBA.AX', period='weekly')
```

## Next Steps

This notebook provides a foundation for:
- **Advanced Technical Analysis**: Additional indicators and patterns
- **Portfolio Analysis**: Multi-stock prediction and comparison
- **Real-time Trading**: Live data integration and automated trading
- **Risk Management**: Volatility analysis and risk metrics
- **Academic Research**: Statistical validation and hypothesis testing

## Technical Architecture

### Model Design
- **Sequential LSTM**: Time series optimized architecture
- **Regularization**: Dropout layers prevent overfitting
- **Multi-feature Input**: Comprehensive market data utilization
- **Scalable Design**: Easily adaptable to different assets

### Visualization Engine
- **Matplotlib Backend**: Professional chart rendering
- **Seaborn Integration**: Statistical visualization enhancement
- **Interactive Components**: User-driven parameter selection
- **Export Pipeline**: High-quality output generation

## Version Information

- **Version**: 0.3 (Data Processing 2)
- **Framework**: TensorFlow 2.x, Matplotlib 3.x
- **Python**: 3.7+
- **Course**: COS30018 - Intelligent Systems
- **Assignment**: Project Assignment Option B - Tasks B.3

## Contributing

### Development Guidelines
1. Maintain backward compatibility with previous versions
2. Follow PEP 8 coding standards for new functions
3. Include comprehensive documentation for new features
4. Test with multiple stock symbols and time periods
5. Ensure visualization compatibility across platforms

### Extension Opportunities
- Additional chart types (line charts, area charts)
- More statistical analysis tools
- Real-time data streaming capabilities
- Advanced technical indicators
- Portfolio-level analysis tools

## License

This project is part of an academic assignment for COS30018 - Intelligent Systems. The code is provided for educational purposes and research applications.

---

**Note**: This notebook combines machine learning prediction with professional financial visualization tools, making it suitable for both academic study and practical market analysis applications.
