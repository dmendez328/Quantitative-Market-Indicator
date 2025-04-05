import json
import time
import requests
import datetime
import pandas as pd
import math
import sys
import random 
import time
import pytz
import numpy as np
from datetime import datetime, timedelta, date, timezone

from polygon import RESTClient

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca_trade_api import REST
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass


'''
The following section checking if this is a real deployment does not 
necessarily apply to project spectator, but this section is important 
for different models that actually place trades
'''
# Deployment?
real_deployment = False

if real_deployment == True:
    account_positons_url = 'https://api.alpaca.markets/v2/positions'
    account_url = 'https://api.alpaca.markets/v2/account'
    clock_url = 'https://api.alpaca.markets/v2/clock'
elif real_deployment == False:
    account_positons_url = 'https://paper-api.alpaca.markets/v2/positions'
    account_url = 'https://paper-api.alpaca.markets/v2/account'
    clock_url = 'https://paper-api.alpaca.markets/v2/clock'



# These are the keys for the PAPER TRADING Account
api_key_alpaca = ''
api_secret_alpaca = ''


trading_client = TradingClient(api_key_alpaca, api_secret_alpaca) # Getting specific clients details
account = trading_client.get_account() # Get the account itself
stock_client = StockHistoricalDataClient(api_key_alpaca, api_secret_alpaca)

# These are the keys for Polygon where we get the data from
polygon_api_key = '' # Make sure to use the ********@***.edu Polygon account

client = RESTClient(api_key=polygon_api_key)


'''
Account Information
'''

def is_market_open(api_key, api_secret):

    # Sends a request to check if the market is open

    # Headers to allow acces to our account data
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret
    }
    
    response = requests.get(clock_url, headers=headers)
    response.raise_for_status()
    response = response.json()

    return response['is_open']

'''
Indicators and Indicator Tools
'''

def get_historical_bars(symbol, timespan, start, end, multiplier):
    
    # The following function is for retrieving data from the Polygom API

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
    
    params = {"apiKey": polygon_api_key}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  
        data = response.json()
        
        # Validate response
        if "results" in data:
            return data
        else:
            print(f"Warning: No results found for {symbol} from {start} to {end}.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None

def timestamp_to_unix(time_stamp):

    # Converts a timestamp in milliseconds to a formatted UTC datetime string

    dt = datetime.fromtimestamp(time_stamp / 1000, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S')  # Returns a clean UTC timestamp

def filter_trading_hours(trades):

    # Filters through a set of data, checking if any of the trades are outside of trading hours, 
    # removes the ones that aren't within the window


    # Define timezones
    eastern = pytz.timezone('US/Eastern')
    utc = pytz.utc  # Define UTC explicitly

    # Trading hours in minutes from midnight
    market_open = 9 * 60 + 30  # 9:30 AM ET
    market_close = 16 * 60  # 4:00 PM ET

    filtered_trades = []

    for trade in trades:
        # Convert timestamp (milliseconds) to UTC datetime
        dt_utc = datetime.fromtimestamp(trade['t'] / 1000, tz=utc)
        
        # Convert to Eastern Time (ET)
        dt_et = dt_utc.astimezone(eastern)

        # Convert time to minutes from midnight
        time_in_minutes = dt_et.hour * 60 + dt_et.minute

        # Exclude weekends (Saturday = 5, Sunday = 6)
        if dt_et.weekday() >= 5:
            print(f"Skipping weekend trade: {dt_et}")
            continue  # Skip non-trading days

        # Keep only results within trading hours **INCLUDING 4:00 PM**
        if market_open <= time_in_minutes <= market_close:
            filtered_trades.append(trade)
            #print(f"INCLUDED trade: {dt_et} (within trading hours)")
        #else:
            #print(f"EXCLUDED trade: {dt_et} (outside trading hours)")

    return filtered_trades

def get_price_lists(symbol, timeframe, data_type):

    # The input timeframe can be '15min', '30min', or 'day'
    # The input data_type can be 'o', 'c', 'h', 'l', or 'vw'
    # What this function should return are lists of open, high, low, close, or VWAP

    if timeframe == '15min':

        now = datetime.now(timezone.utc)
        end = now - timedelta(minutes=15)  # Account for Polygon's 15-minute delay
        start = end - timedelta(days=10)  # Ensure enough data (weekends & holidays)

        # Convert to milliseconds
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        data = get_historical_bars(symbol, 'minute', start_ms, end_ms, 15)

        data = data['results'] # Make var equal to the list of dictionaries

        data = filter_trading_hours(data) # Filter candles

    elif timeframe == '30min':

        now = datetime.now(timezone.utc)
        end = now - timedelta(minutes=15)  # Account for Polygon's 15-minute delay
        start = end - timedelta(days=10)  # Ensure enough data (weekends & holidays)

        # Convert to milliseconds
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        data = get_historical_bars(symbol, 'minute', start_ms, end_ms, 30)

        data = data['results'] # Make var equal to the list of dictionaries

        data = filter_trading_hours(data) # Filter candles

    elif timeframe == 'day':

        today = datetime.now(timezone.utc)
        start = today - timedelta(days=65)  # Get 60 days to ensure enough data

        # Convert to milliseconds
        today_ms = int(today.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)

        data = get_historical_bars(symbol, 'day', start_ms, today_ms, 1)

        print(data)

        data = data['results'] # Make var equal to the list of dictionaries

    list_prices = [] # Empty list ot add and return

    for i in range(len(data)):
        list_prices.append(data[i][data_type])

    return list_prices

def sma(symbol, period, timeframe, data_type):

    # The input timeframe can be '15min', '30min', or 'day'
    # The input data_type can be 'o', 'c', 'h', 'l', or 'vw'

    temp_list = get_price_lists(symbol, timeframe, data_type) # Get the type of list of prices of bars

    # Shorten the list to the desired period
    while (len(temp_list) > period):
        temp_list.pop(0)

    # Calculate SMA
    sma = round(sum(temp_list) / len(temp_list), 2)
    
    return sma

def ema(symbol, period, timeframe, data_type):
    
    # The input timeframe can be '15min', '30min', or 'day'
    # The input data_type can be 'o', 'c', 'h', 'l', or 'vw'

    temp_list = get_price_lists(symbol, timeframe, data_type)

    # Ensure we have enough data
    if len(temp_list) < period:
        print("Not enough data to compute EMA.")
        return None

    # Shorten the list to the desired period
    while len(temp_list) > period:
        temp_list.pop(0)

    # Calculate the smoothing factor (alpha)
    alpha = 2 / (period + 1)

    # Initialize EMA with the first SMA
    ema_value = sum(temp_list) / len(temp_list)

    # Apply the EMA formula iteratively
    for price in temp_list:
        ema_value = (price * alpha) + (ema_value * (1 - alpha))

    return round(ema_value, 2)

def vwap_retrieval(symbol, timeframe):

    # The input timeframe can be '15min', '30min', or 'day'
    
    temp_list = get_price_lists(symbol, timeframe, 'vw') # 'vw' argument is for VWAP

    return temp_list[(len(temp_list) - 1)]

def calculate_bollinger_bands(symbol, period, timeframe, data_type, std_dev_multiplier):

    # The input timeframe can be '15min', '30min', or 'day'
    # The input data_type can be 'o', 'c', 'h', or 'l'; typeically 'c'

    price_list = get_price_lists(symbol, timeframe, data_type) # List of prices of the given type

    # Shorten the list to the desired amount or price samples
    while (len(price_list) > period):
        price_list.pop(0)

    mean = sum(price_list) / len(price_list) # SMA or also the middle band

    std = np.std(price_list, ddof=1) # Calculate the Standard Deviation; ddof is the Sample Standard Deviation

    upper = mean + std_dev_multiplier * std # Upper Band
    lower = mean - std_dev_multiplier * std # Lower Band

    bands = {
        'middle': round(mean, 2),
        'upper': round(upper, 2),
        'lower': round(lower, 2)
    }

    return bands

def macd_calc(symbol, fast_ema_period, slow_ema_period, signal_value, timeframe):

    # MACD and Signal line calculations along with their rates of change

    price_list = get_price_lists(symbol, timeframe, data_type='c')

    if len(price_list) < slow_ema_period + signal_value + 2:
        print("Not enough data to compute MACD.")
        return None

    def compute_ema(prices, period):
        k = 2 / (period + 1)
        ema_values = []
        sma = sum(prices[:period]) / period
        ema_values.append(sma)
        for price in prices[period:]:
            prev = ema_values[-1]
            ema_values.append((price - prev) * k + prev)
        return [None] * (period - 1) + ema_values

    # Compute EMAs
    fast_ema = compute_ema(price_list, fast_ema_period)
    slow_ema = compute_ema(price_list, slow_ema_period)

    # Compute MACD line
    macd_line = []
    for f, s in zip(fast_ema, slow_ema):
        if f is None or s is None:
            macd_line.append(None)
        else:
            macd_line.append(f - s)

    # Filter valid MACD values for signal line calculation
    valid_macd = [x for x in macd_line if x is not None]
    signal_line = compute_ema(valid_macd, signal_value)
    signal_line_full = [None] * (len(macd_line) - len(signal_line)) + signal_line

    # Extract most recent values
    t = -1
    if any(val is None for val in [
        macd_line[t], macd_line[t-1], macd_line[t-2],
        signal_line_full[t], signal_line_full[t-1], signal_line_full[t-2]
    ]):
        print("Not enough aligned data.")
        return None

    diff_t = macd_line[t] - signal_line_full[t]
    diff_t1 = macd_line[t - 1] - signal_line_full[t - 1]
    diff_t2 = macd_line[t - 2] - signal_line_full[t - 2]

    roc = diff_t - diff_t1
    acceleration = roc - (diff_t1 - diff_t2)

    return {
        "MACD_t": round(macd_line[t], 4),
        "Signal_t": round(signal_line_full[t], 4),
        "MACD_t-1": round(macd_line[t - 1], 4),
        "Signal_t-1": round(signal_line_full[t - 1], 4),
        "MACD_t-2": round(macd_line[t - 2], 4),
        "Signal_t-2": round(signal_line_full[t - 2], 4),
        "Diff_t": round(diff_t, 4),
        "Diff_t-1": round(diff_t1, 4),
        "Diff_t-2": round(diff_t2, 4),
        "Rate_of_Change": round(roc, 4),
        "Acceleration": round(acceleration, 4)
    }

def atr_calc(symbol, timeframe, period):

    close_list = get_price_lists(symbol, timeframe, data_type='c')
    high_list = get_price_lists(symbol, timeframe, data_type='h')
    low_list = get_price_lists(symbol, timeframe, data_type='l')

    if len(close_list) < period + 2:
        print("Not enough data to calculate ATR.")
        return None

    true_ranges = []

    for i in range(1, len(close_list)):
        high = high_list[i]
        low = low_list[i]
        prev_close = close_list[i - 1]
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    # Start with SMA of first 'period' TR values
    atr_values = []
    sma = sum(true_ranges[:period]) / period
    atr_values.append(sma)

    # Use EMA for the rest
    k = 2 / (period + 1)
    for tr in true_ranges[period:]:
        prev_atr = atr_values[-1]
        atr = (tr - prev_atr) * k + prev_atr
        atr_values.append(atr)

    # Add None padding to align length with original list
    atr_full = [None] * period + atr_values

    # Extract t, t-1, t-2
    if atr_full[-1] is None or atr_full[-2] is None or atr_full[-3] is None:
        print("Not enough computed ATR values for ROC/acceleration.")
        return None

    atr_t = atr_full[-1]
    atr_t1 = atr_full[-2]
    atr_t2 = atr_full[-3]

    roc = atr_t - atr_t1
    acceleration = roc - (atr_t1 - atr_t2)

    return {
        "ATR": round(atr_t, 4),
        "ATR_ROC": round(roc, 4),
        "ATR_Acceleration": round(acceleration, 4)
    }

def calculate_percent_change(current, previous):
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def find_big_movers_with_volume(symbols, threshold=40, volume_spike_ratio=2):
    now = datetime.now(timezone.utc)
    today = now - timedelta(minutes=15)
    yesterday = today - timedelta(days=1)
    last_week = today - timedelta(days=7)

    today_ms = int(today.timestamp() * 1000)
    yesterday_ms = int(yesterday.timestamp() * 1000)
    last_week_ms = int(last_week.timestamp() * 1000)

    results = []

    for symbol in symbols:
        data = get_historical_bars(symbol, 'day', last_week_ms, today_ms, 1)

        if not data or 'results' not in data or len(data['results']) < 6:
            continue

        bars = data['results']

        today_close = bars[-1]['c']
        yesterday_close = bars[-2]['c']
        week_ago_close = bars[-6]['c']

        today_volume = bars[-1]['v']
        recent_volumes = [bar['v'] for bar in bars[-6:-1]]
        avg_recent_volume = sum(recent_volumes) / len(recent_volumes)

        day_change = calculate_percent_change(today_close, yesterday_close)
        week_change = calculate_percent_change(today_close, week_ago_close)
        volume_spike = today_volume > avg_recent_volume * volume_spike_ratio

        if (day_change > threshold or week_change > threshold) and volume_spike:
            results.append({
                'symbol': symbol,
                'today_close': round(today_close, 2),
                'day_change': round(day_change, 2),
                'week_change': round(week_change, 2),
                'today_volume': int(today_volume),
                'avg_recent_volume': int(avg_recent_volume),
                'volume_spike_ratio': round(today_volume / avg_recent_volume, 2)
            })

    return results

def find_momentum_candidates_from_json(json_file="polygon_tickers.json"):

    # Load tickers from JSON file
    try:
        with open(json_file, "r") as f:
            symbols = json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return []
    

    print(f"Loaded {len(symbols)} tickers from {json_file}\n") # Successfully loaded tickers from the file

    candidates = []

    for symbol in symbols:

        try:

            # Bollinger Bands
            bands = calculate_bollinger_bands(symbol, period=14, timeframe='15min', data_type='c', std_dev_multiplier=1.3)
            close_prices = get_price_lists(symbol, '15min', 'c')
            last_close = close_prices[-1]

            # VWAP
            vwap = vwap_retrieval(symbol, '15min')

            # EMA Crossover
            ema_9 = ema(symbol, 9, '15min', 'c')
            ema_21 = ema(symbol, 21, '15min', 'c')

            # MACD Calculation
            macd_info = macd_calc(symbol, 12, 26, 9, '15min')  # Typical MACD values

            if not macd_info:
                print(f"{symbol}: Not enough MACD data\n")
                continue


            if ((last_close > bands['upper']) and 
                (last_close > vwap) and 
                ((ema_9 > ema_21) or ((macd_info['Diff_t'] >= 0) and (macd_info['Rate_of_Change'] > 0) and (macd_info['Acceleration'] >= 0)))):
                
                print(f"{symbol} passed the momentum strategy!\n")
                candidates.append(symbol)
            else:
                print(f"{symbol} failed the strategy check\n")

        except Exception as e:
            print(f"Error checking {symbol}: {e}\n")
            continue

    return candidates






# The following is used to filter through a list of stocks 
momentum_picks = find_momentum_candidates_from_json()

print("Final Momentum Picks:", momentum_picks)

