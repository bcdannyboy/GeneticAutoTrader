
import numpy as np
import talib

def NormalizeOverlapScore(overlap_indicators):
    normalized_scores = []
    for indicators in overlap_indicators:
        max_abs_value = max(abs(value) for value in indicators.values())
        overlap_sum = sum(indicators.values())
        normalized_score = overlap_sum / max(max_abs_value, 1)  # Avoid division by zero
        normalized_scores.append(normalized_score)
    return normalized_scores

def GetBBANDS(prices):
    # Bollinger Bands
    return talib.BBANDS(prices)

def GetDEMA(prices):
    # Double Exponential Moving Average
    return talib.DEMA(prices)

def GetEMA(prices):
    # Exponential Moving Average
    return talib.EMA(prices)

def GetHT_TRENDLINE(prices):
    # Hilbert Transform - Instantaneous Trendline
    return talib.HT_TRENDLINE(prices)

def GetKAMA(prices):
    # Kaufman Adaptive Moving Average
    return talib.KAMA(prices)

def GetMA(prices):
    # Moving average
    return talib.MA(prices)

def GetMAVP(stock_data):
    # Moving average with variable period
    periods = stock_data.Date
    return talib.MAVP(stock_data.Close, periods, minperiod=2, maxperiod=30, matype=0)

def GetMidPoint(prices):
    # MidPoint over period
    return talib.MIDPOINT(prices)

def GetMidPrice(prices, timeperiod=14):
    # Ensure prices is a numpy array
    prices = np.array(prices)

    # Check if there are enough prices to calculate MIDPRICE
    if len(prices) < timeperiod:
       if timeperiod > 1:
            return GetMidPrice(timeperiod = timeperiod - 1)
       else:
           print("Not enough data to calculate MIDPRICE for the given time period")
           return 0

    # Calculate high and low prices for each sub-period
    high_prices = [max(prices[i:i+timeperiod]) for i in range(len(prices) - timeperiod + 1)]
    low_prices = [min(prices[i:i+timeperiod]) for i in range(len(prices) - timeperiod + 1)]

    # Convert lists to numpy arrays
    high_prices = np.array(high_prices)
    low_prices = np.array(low_prices)

    # Calculate MIDPRICE
    midprice = talib.MIDPRICE(high_prices, low_prices, timeperiod)

    return midprice

def GetSAR(prices, timeperiod=14, acceleration=0.02, maximum=0.2):
    # Ensure prices is a numpy array
    prices = np.array(prices)

    # Check if there are enough prices to calculate SAR
    if len(prices) < timeperiod:
        if timeperiod > 1:
            return GetSAR(timeperiod = timeperiod - 1)
        else:
            print("Not enough data to calculate SAR for the given time period")
            return 0

    # Calculate rolling high and low prices for each sub-period
    high_prices = np.array([max(prices[i:i+timeperiod]) for i in range(len(prices) - timeperiod + 1)])
    low_prices = np.array([min(prices[i:i+timeperiod]) for i in range(len(prices) - timeperiod + 1)])

    # Calculate Parabolic SAR
    sar = talib.SAR(high_prices, low_prices, acceleration, maximum)

    return sar

def GetSAREXT(prices, timeperiod=14, startvalue=0, offsetonreverse=0, accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2, accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2):
    # Ensure prices is a numpy array
    prices = np.array(prices)

    # Check if there are enough prices to calculate SAREXT
    if len(prices) < timeperiod:
        if timeperiod > 1:
            return GetSAREXT(timeperiod = timeperiod - 1)
        else:
            print("Not enough data to calculate SAREXT for the given time period")
            return 0

    # Calculate rolling high and low prices for each sub-period
    high_prices = np.array([max(prices[i:i+timeperiod]) for i in range(len(prices) - timeperiod + 1)])
    low_prices = np.array([min(prices[i:i+timeperiod]) for i in range(len(prices) - timeperiod + 1)])

    # Calculate Extended Parabolic SAR
    sarext = talib.SAREXT(high_prices, low_prices, startvalue, offsetonreverse, accelerationinitlong, accelerationlong, accelerationmaxlong, accelerationinitshort, accelerationshort, accelerationmaxshort)

    return sarext

def GetSMA(prices):
    # Simple Moving Average
    return talib.SMA(prices)

def GetT3(prices):
    # Triple Exponential Moving Average (T3)
    return talib.T3(prices)

def GetTEMA(prices):
    # Triple Exponential Moving Average
    return talib.TEMA(prices)

def GetTRIMA(prices):
    # Triangular Moving Average
    return talib.TRIMA(prices)

def GetWMA(prices):
    # Weighted Moving Average
    return talib.WMA(prices)