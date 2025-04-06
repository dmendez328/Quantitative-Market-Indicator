import json
import time
import requests
import sys
from datetime import datetime, timedelta, timezone

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca_trade_api import REST
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

from polygon import RESTClient


# These are the kays for the PAPER TRADING Account
api_key = ''
api_secret = ''


polygon_api_key = ''


trading_client = TradingClient(api_key, api_secret) # Getting specific clients details
account = trading_client.get_account() # Get the account itself
stock_client = StockHistoricalDataClient(api_key, api_secret)



def has_minimum_volume(client, symbol, days=5, min_volume=200000):
    try:
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=days + 3)).strftime('%Y-%m-%d')  # Add buffer for weekends
        end = now.strftime('%Y-%m-%d')

        aggs = client.get_aggs(symbol, 1, "day", start, end, adjusted=True)
        if not aggs or len(aggs) < days:
            return False

        recent_volumes = [bar.volume for bar in aggs[-days:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        return avg_volume >= min_volume

    except Exception as e:
        print(f"Skipping {symbol} — error getting volume: {e}")
        return False
    
def get_historical_bars(symbol, timespan, start, end, multiplier):

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
    
def range_price(symbol, min_price=0.25, max_price=30):

    now = datetime.now(timezone.utc)
    end = now - timedelta(minutes=15)  # Account for delay
    start = end - timedelta(days=4)  # Short window to catch last bar

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    data = get_historical_bars(symbol, 'day', start_ms, end_ms, 1)

    if data and 'results' in data and data['results']:
        last_close = data['results'][-1]['c']
        return min_price <= last_close <= max_price
    return False

def get_filtered_polygon_tickers(api_key, output_file="polygon_tickers.json"):
    client = RESTClient(api_key)
    tickers = []

    print("Getting NASDAQ common stock tickers...")
    
    for t in client.list_tickers(
        market="stocks",
        type="CS",
        active=True,
        order="asc",
        limit=1000,
        sort="ticker"
    ):
        if t.primary_exchange == "XNAS":
            tickers.append(t.ticker)

    print(f"Found {len(tickers)} tickers. Now filtering by volume and price range...")

    valid_tickers = []
    for i, symbol in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Checking {symbol}...")
        try:
            if has_minimum_volume(client, symbol) and range_price(symbol):
                valid_tickers.append(symbol)
        except Exception as e:
            print(f"Error with {symbol}: {e}")
            continue

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(valid_tickers, f, indent=2)

    print(f"Saved {len(valid_tickers)} valid tickers to {output_file}")
    return valid_tickers



tickers = get_filtered_polygon_tickers(polygon_api_key)
