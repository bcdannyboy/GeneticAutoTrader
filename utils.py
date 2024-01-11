
import datetime
import json
import numpy as np
import pandas as pd

import yfinance as yf

from yf.gather import GatherData

def FillInitialNaN(data, reference_data):
    first_valid_index = np.where(~np.isnan(data))[0][0]
    first_valid_value = data[first_valid_index]
    filled_data = np.full(len(reference_data), first_valid_value)
    filled_data[-len(data):] = data
    return filled_data


def get_spy_prices(start_date, end_date):
    spy = yf.Ticker("SPY")
    
    try:
        spy_data = spy.history(start=start_date, end=end_date)
    except Exception as e:
        print("Error downloading SPY data:", e)
        return None

    if "Close" not in spy_data.columns:
        print("SPY data does not contain Close prices")

        # Try falling back to the Adjusted Close
        if "Adj Close" in spy_data.columns:
            print("Using Adjusted Close prices instead")
            spy_prices = spy_data["Adj Close"]
        else:
            print("Unable to get SPY price data") 
            return None
    else: 
        spy_prices = spy_data["Close"]
        
    return spy_prices

def save_weights_to_json(weights, filename):
    """Saves the given weights to a JSON file."""
    with open(filename, 'w') as file:
        json.dump(weights, file, indent=4)
        
def aggregate_weights(all_weights):
    """Aggregate weights from all runs. This function can be modified to use a more sophisticated aggregation method."""
    aggregated = {}
    for weights in all_weights:
        for key, value in weights.items():
            if key in aggregated:
                aggregated[key] += value
            else:
                aggregated[key] = value

    # Averaging the weights
    for key in aggregated:
        aggregated[key] /= len(all_weights)

    return aggregated