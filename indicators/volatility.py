import numpy as np
import talib

def NormalizeVolatilityIndicators(indicators):
    normalized_indicators = []
    for indicator in indicators:
        mean = np.nanmean(indicator)
        std = np.nanstd(indicator)
        if std != 0:
            normalized = np.tanh((indicator - mean) / std)
        else:
            normalized = np.zeros_like(indicator)
        normalized_indicators.append(normalized)
    return normalized_indicators

def GetATR(high_prices, low_prices, close_prices, timeperiod=7):
    """
    Calculate the Average True Range (ATR).
    :param high_prices: numpy array of high prices
    :param low_prices: numpy array of low prices
    :param close_prices: numpy array of close prices
    :param timeperiod: Number of periods to use for ATR calculation
    :return: numpy array of ATR values
    """
    return talib.ATR(high_prices, low_prices, close_prices, timeperiod)

def GetNATR(high_prices, low_prices, close_prices, timeperiod=7):
    """
    Calculate the Normalized Average True Range (NATR).
    :param high_prices: numpy array of high prices
    :param low_prices: numpy array of low prices
    :param close_prices: numpy array of close prices
    :param timeperiod: Number of periods to use for NATR calculation
    :return: numpy array of NATR values
    """
    return talib.NATR(high_prices, low_prices, close_prices, timeperiod)

def GetTRANGE(high_prices, low_prices, close_prices):
    """
    Calculate the True Range (TRANGE).
    :param high_prices: numpy array of high prices
    :param low_prices: numpy array of low prices
    :param close_prices: numpy array of close prices
    :return: numpy array of TRANGE values
    """
    return talib.TRANGE(high_prices, low_prices, close_prices)