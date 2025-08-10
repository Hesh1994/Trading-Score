# Streamlit Trading Dashboard

A comprehensive technical analysis dashboard built with Streamlit for analyzing stock market data with multiple technical indicators.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🚀 Live Demo

Visit the live app: [Trading Dashboard](https://your-app-name.streamlit.app)

## Features

- **Multiple Technical Indicators**: RSI, SMA, EMA, MFI, Stochastic, Aroon, Bollinger Bands, MACD
- **Configurable Parameters**: Adjust periods and parameters for each indicator
- **Composite Scoring System**: Weighted scoring based on multiple signals
- **Interactive Charts**: Plotly-powered visualizations
- **Stock Analysis**: Individual stock deep-dive analysis
- **Weekly Analysis**: Compare daily vs weekly timeframes
- **S&P 500 Integration**: Built-in S&P 500 stock selection
- **Custom Stock Lists**: Support for custom ticker symbols

## 🚀 Quick Deploy to Streamlit Cloud

1. **Fork this repository** on GitHub
2. **Sign up** for [Streamlit Cloud](https://streamlit.io/cloud)
3. **Connect your GitHub account**
4. **Deploy** by selecting this repository
5. **Set the main file** to `trading_dashboard.py`
6. **Click Deploy!**

## Installation

### Option 1: Using Setup Script (Windows)
1. Run the setup script:
   ```
   setup.bat
   ```

### Option 2: Manual Installation
1. Install Python 3.8 or higher
2. Install required packages:
   ```
   pip install -r requirements-minimal.txt
   ```
   
   Or for full features:
   ```
   pip install -r requirements.txt
   ```

## Running the Dashboard

1. Open command prompt/terminal
2. Navigate to the project directory
3. Run the dashboard:
   ```
   streamlit run trading_dashboard.py
   ```
4. Open your browser to the URL shown (typically http://localhost:8501)

## Usage

### Configuration
1. **Date Range**: Set start and end dates for analysis
2. **Symbols**: Choose from S&P 500 stocks or enter custom symbols
3. **Technical Indicators**: Select which indicators to calculate
4. **Parameters**: Adjust periods and settings for each indicator
5. **Signal Weights**: Configure how much weight each signal carries
6. **Target Score**: Set your target composite score threshold

### Analysis Tabs
- **Overview**: Configuration summary and score distribution
- **Score Changes**: Stocks with recent score changes
- **Target Scores**: Stocks meeting your target criteria
- **Weekly Analysis**: Weekly vs daily comparison
- **Individual Stock Analysis**: Detailed view of specific stocks

## Technical Indicators Included

### Trend Indicators
- **SMA (Simple Moving Average)**: Short and long period averages
- **EMA (Exponential Moving Average)**: Responsive trend indicator
- **MACD**: Moving Average Convergence Divergence

### Momentum Indicators
- **RSI (Relative Strength Index)**: Overbought/oversold conditions
- **Stochastic**: %K and %D oscillators
- **MFI (Money Flow Index)**: Volume-weighted RSI

### Volatility Indicators
- **Bollinger Bands**: Price channels with standard deviation

### Trend Strength Indicators
- **Aroon**: Trend strength and direction

## Scoring System

The dashboard creates a composite score by:
1. Calculating multiple technical indicators
2. Generating buy/sell signals from each indicator
3. Applying user-defined weights to each signal
4. Combining into a single composite score
5. Tracking score changes over time

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- yfinance
- plotly
- beautifulsoup4
- requests

## Troubleshooting

### Common Issues

1. **PyArrow DLL Error**: 
   ```
   pip install --upgrade pyarrow
   ```

2. **Missing packages**:
   ```
   pip install -r requirements.txt
   ```

3. **Data download issues**: Check internet connection and verify ticker symbols

4. **Performance issues**: Reduce date range or number of symbols

### Tips for Better Performance
- Start with fewer symbols (5-10) for testing
- Use shorter date ranges initially
- Gradually increase complexity as needed

## File Structure

```
Trading Dashboard/
├── trading_dashboard.py      # Main Streamlit application
├── requirements.txt          # Full requirements
├── requirements-minimal.txt  # Minimal requirements
├── setup.bat                # Windows setup script
└── README.md                # This file
```

## Contributing

Feel free to enhance the dashboard by:
- Adding new technical indicators
- Improving the UI/UX
- Adding more analysis features
- Optimizing performance

## License

This project is open source and available under the MIT License.
