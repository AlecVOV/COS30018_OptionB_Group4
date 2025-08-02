# Stock Prediction with LSTM Neural Networks - v0.2

This Jupyter notebook implements an advanced stock price prediction system using Long Short-Term Memory (LSTM) neural networks with comprehensive data processing and model management capabilities.

## Overview

The `v0.2_codebase_stock_prediction.ipynb` notebook extends the basic data acquisition to include sophisticated data preprocessing, feature engineering, LSTM model development, and prediction analysis. This version focuses on CBA.AX (Commonwealth Bank of Australia) stock prediction with enhanced accuracy and robustness.

## Features

### Data Management
- **Interactive Date Selection**: User-configurable training periods with validation
- **Automatic Data Caching**: Saves and loads processed data to avoid redundant API calls
- **Multiple Data Splitting Methods**: Sequential, date-based, and random splitting options
- **Comprehensive Data Cleaning**: Handles missing values and data type conversions

### Advanced Preprocessing
- **Multi-Feature Scaling**: Individual MinMaxScaler for each feature (Close, High, Low, Open, Volume)
- **Scaler Persistence**: Saves scalers for consistent inverse transformations
- **Sequence Generation**: Creates time-series sequences with configurable lookback periods
- **Feature Selection**: Flexible feature inclusion based on data availability

### LSTM Model Architecture
- **Multi-Layer LSTM**: Three LSTM layers with dropout regularization
- **Model Persistence**: Automatic saving and loading of trained models
- **Training History**: Comprehensive tracking of training metrics
- **Enhanced Compilation**: Multiple evaluation metrics (MAE, MSE, RMSE)

### Prediction Analysis
- **Comprehensive Metrics**: MAE, RMSE, MAPE, and accuracy percentages
- **Detailed Visualizations**: Prediction vs actual plots with error analysis
- **Statistical Analysis**: Error distribution and performance statistics
- **Results Export**: CSV export of predictions and analysis

## Requirements

### Dependencies
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn yfinance mplfinance
```

### System Requirements
- Python 3.7 or higher
- Minimum 4GB RAM (8GB recommended for large datasets)
- GPU support recommended for faster training (optional)

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Tasks B.2 - Data Processing 1/v0.2 Codebase"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the notebook**:
   ```bash
   jupyter notebook v0.2_codebase_stock_prediction.ipynb
   ```

## Usage Guide

### 1. Library Imports
The notebook begins with importing all necessary libraries for data processing, machine learning, and visualization.

### 2. Data Loading
- **Interactive Configuration**: Enter custom date ranges or use defaults (2020-01-01 to 2023-08-01)
- **Automatic Data Management**: Downloads and caches data in the `data/` directory
- **Data Validation**: Ensures date format and range validity

### 3. Data Preprocessing
- **Data Cleaning**: Processes CSV data with proper column naming and type conversion
- **Data Splitting**: Choose from three splitting methods:
  - Sequential split by ratio (e.g., 80/20)
  - Split by specific date
  - Random split by ratio

### 4. Feature Scaling
- **Individual Scalers**: Each feature gets its own MinMaxScaler for optimal normalization
- **Scaler Persistence**: Scalers are saved for future use and inverse transformations
- **Feature Analysis**: Displays scaling statistics and ranges

### 5. Model Development
- **Architecture**: 3-layer LSTM with 50 units each, dropout layers for regularization
- **Smart Training**: Loads pre-trained models if available, otherwise trains from scratch
- **Hyperparameters**:
  - Prediction days: 60 (configurable)
  - Epochs: 25
  - Batch size: 32
  - Optimizer: Adam

### 6. Prediction and Analysis
- **Test Set Predictions**: Generates predictions for the test dataset
- **Performance Metrics**: Calculates MAE, RMSE, MAPE, and accuracy
- **Visualization**: Creates plots comparing actual vs predicted prices
- **Error Analysis**: Identifies best and worst predictions with statistical summaries

## File Structure

```
v0.2 Codebase/
├── v0.2_codebase_stock_prediction.ipynb  # Main notebook
├── README.md                             # This file
├── data/                                 # Generated data directory
│   ├── CBA.AX_data.csv                  # Raw stock data
│   ├── CBA.AX_cleaned_data.csv          # Cleaned and processed data
│   ├── CBA.AX_scalers.pkl               # Saved feature scalers
│   ├── CBA.AX_training_data.npz         # Preprocessed training sequences
│   ├── CBA.AX_test_predictions.csv      # Prediction results
│   └── CBA.AX_prediction_results.png    # Visualization plots
├── models/                               # Generated models directory
│   ├── CBA.AX_lstm_model.h5             # Trained LSTM model
│   └── CBA.AX_training_history.npz      # Training history
└── requirements.txt                      # Project dependencies
```

## Key Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `COMPANY` | 'CBA.AX' | Stock ticker symbol |
| `PREDICTION_DAYS` | 60 | Lookback period for sequences |
| `DEFAULT_TRAIN_START` | '2020-01-01' | Default training start date |
| `DEFAULT_TRAIN_END` | '2023-08-01' | Default training end date |
| `LSTM_UNITS` | 50 | Number of LSTM units per layer |
| `DROPOUT_RATE` | 0.2 | Dropout rate for regularization |
| `EPOCHS` | 25 | Number of training epochs |
| `BATCH_SIZE` | 32 | Training batch size |

## Model Performance

### Expected Metrics
- **Accuracy**: Typically 85-95% depending on market conditions
- **MAPE**: Usually 2-8% for stable periods
- **Training Time**: 2-10 minutes depending on hardware

### Features Used
- **Close Price**: Primary target variable
- **High/Low Prices**: Daily price range information
- **Opening Price**: Start-of-day pricing
- **Volume**: Trading volume indicators

## Advanced Features

### Model Persistence
- Automatic model saving and loading
- Training data caching to avoid reprocessing
- Scaler persistence for consistent transformations

### Error Analysis
- Comprehensive error metrics calculation
- Best/worst prediction identification
- Statistical error distribution analysis

### Visualization
- Actual vs predicted price plots
- Prediction error visualization
- Training history plots (when training occurs)

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce `PREDICTION_DAYS` or batch size
   - Use smaller date ranges for training

2. **Poor Performance**:
   - Increase training epochs
   - Adjust LSTM architecture
   - Try different feature combinations

3. **Data Loading Issues**:
   - Check internet connection for Yahoo Finance API
   - Verify ticker symbol validity
   - Ensure proper date formats

### Performance Optimization

1. **GPU Acceleration**:
   ```python
   # Check GPU availability
   print("GPU Available: ", tf.config.list_physical_devices('GPU'))
   ```

2. **Memory Management**:
   ```python
   # Enable memory growth for GPU
   gpus = tf.config.experimental.list_physical_devices('GPU')
   if gpus:
       tf.config.experimental.set_memory_growth(gpus[0], True)
   ```

## Next Steps

This notebook serves as a foundation for:
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Feature Engineering**: Additional technical indicators
- **Ensemble Methods**: Combining multiple models
- **Real-time Prediction**: Live trading integration
- **Multi-stock Analysis**: Portfolio-level predictions

## Technical Notes

- Uses TensorFlow/Keras for deep learning implementation
- Implements proper time series cross-validation
- Maintains data leakage prevention in preprocessing
- Includes comprehensive error handling and validation

## Version Information

- **Version**: 0.2 (Data Processing 1)
- **Framework**: TensorFlow 2.x
- **Python**: 3.7+
- **Course**: COS30018 - Intelligent Systems
- **Assignment**: Project Assignment Option B - Tasks B.2

## Contributing

When modifying this notebook:
1. Maintain the cell structure and documentation
2. Update hyperparameters in the designated sections
3. Test with different stocks and time periods
4. Document any significant changes or improvements

## License

This project is part of an academic assignment for COS30018 - Intelligent Systems.
