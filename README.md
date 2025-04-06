# Quantitative-Market-Indicator


This project provides a two-part Python-based pipeline for identifying and tracking high-volume, frequently traded stocks with upward momentum based on key technical indicators. It uses filtering techniques and a combination of popular market indicators like Bollinger Bands, VWAP, EMA, and MACD to determine potential trading opportunities.

---

### 1. `comprehensive_list.py`

**Purpose**:  
Filters through a comprehensive list of stocks, identifying those that are frequently traded and meet a specific volume threshold.

**How it works**:
- Analyzes stock activity and filters by volume and frequency of trades
- Stores qualifying stock symbols in a JSON file: `polygon_ticker.json`

**Run this script**:

python ./comprehensive_list.py


### 2. project_spectator.py

**Purpose**:
Analyzes stocks from the polygon_ticker.json list to detect upward momentum using several key technical indicators.

**Indicators Used**:

Bollinger Bands

VWAP (Volume-Weighted Average Price)

EMA (Exponential Moving Average: 9 vs 21)

MACD (Moving Average Convergence Divergence)

### How it works:

Iterates over each stock in polygon_ticker.json

Applies the algorithm and checks if all momentum conditions are met

Flags passing tickers for further action (e.g., display, alert, etc.)


### Run this script:

python ./project_spectator.py

comprehensive_list.py	Gathers and filters stocks by trading volume and frequency

project_spectator.py	Applies technical analysis to selected stocks

polygon_ticker.json	JSON file storing tickers that pass the initial screen


**Requirements**:
Make sure to install all required packages (e.g., requests, pandas, numpy, matplotlib, TA-Lib, or any APIs like Polygon.io or Alpaca if used).


Integrate real-time price feeds or broker APIs for live trading

Enhance indicator logic with additional parameters (RSI, volume spike, etc.)

Visualization dashboard for real-time signal tracking

### Disclaimer!!!!!:

This project is intended for educational and research purposes only. It is not financial advice. Please do your own research before making investment decisions.
