
import numpy as np
import talib

def NormalizeIndicator(indicator):
    normalized = np.tanh(indicator)  # Normalize each indicator using tanh
    return normalized

def CalculateNormalizedMomentumScores(MomentumIndicators):
    normalized_momentum_scores = []
    for indicators in MomentumIndicators:
        normalized_indicators = []
        for value in indicators.values():
            normalized_value = NormalizeIndicator(value)
            # Ensure normalized_value is a scalar
            if np.isscalar(normalized_value):
                normalized_indicators.append(normalized_value)
            elif normalized_value is not None:
                # If normalized_value is an array, take its mean
                normalized_indicators.append(np.mean(normalized_value))

        if normalized_indicators:
            average_normalized_indicator = np.mean(normalized_indicators)
            normalized_momentum_scores.append(average_normalized_indicator)
        else:
            # Handle the case where all indicators were None or not valid
            normalized_momentum_scores.append(np.nan)

    return normalized_momentum_scores


def GetADX(high_prices, low_prices, close_prices, timeperiod=14):
    adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod)
    return adx

def GetADXR(high_prices, low_prices, close_prices, timeperiod=14):
    adxr = talib.ADXR(high_prices, low_prices, close_prices, timeperiod)
    return adxr

def GetAPO(close_prices, fastperiod=12, slowperiod=26, matype=0):
    apo = talib.APO(close_prices, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
    return apo

def GetAROON(high_prices, low_prices, timeperiod=14):
    aroon_up, aroon_down = talib.AROON(high_prices, low_prices, timeperiod)
    return aroon_up, aroon_down

def GetAROONOSC(high_prices, low_prices, timeperiod=14):
    aroonosc = talib.AROONOSC(high_prices, low_prices, timeperiod)
    return aroonosc

def GetBOP(open_prices, high_prices, low_prices, close_prices):
    bop = talib.BOP(open_prices, high_prices, low_prices, close_prices)
    return bop

def GetCCI(high_prices, low_prices, close_prices, timeperiod=14):
    cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod)
    return cci

def GetCMO(close_prices, timeperiod=14):
    cmo = talib.CMO(close_prices, timeperiod)
    return cmo

def GetDX(high_prices, low_prices, close_prices, timeperiod=14):
    dx = talib.DX(high_prices, low_prices, close_prices, timeperiod)
    return dx

def GetMACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9):
    macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    return macd, macdsignal, macdhist

def GetMACDEXT(close_prices, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0):
    macd, macdsignal, macdhist = talib.MACDEXT(close_prices, fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype, signalperiod=signalperiod, signalmatype=signalmatype)
    return macd, macdsignal, macdhist

def GetMACDFIX(close_prices, signalperiod=9):
    macd, macdsignal, macdhist = talib.MACDFIX(close_prices, signalperiod=signalperiod)
    return macd, macdsignal, macdhist

def GetMFI(high_prices, low_prices, close_prices, volume, timeperiod=14):
    mfi = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod)
    return mfi

def GetMINUS_DI(high_prices, low_prices, close_prices, timeperiod=14):
    minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod)
    return minus_di

def GetMINUS_DM(high_prices, low_prices, timeperiod=14):
    minus_dm = talib.MINUS_DM(high_prices, low_prices, timeperiod)
    return minus_dm

def GetMOM(close_prices, timeperiod=10):
    mom = talib.MOM(close_prices, timeperiod)
    return mom

def GetPLUS_DI(high_prices, low_prices, close_prices, timeperiod=14):
    plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod)
    return plus_di

def GetPLUS_DM(high_prices, low_prices, timeperiod=14):
    plus_dm = talib.PLUS_DM(high_prices, low_prices, timeperiod)
    return plus_dm

def GetPPO(close_prices, fastperiod=12, slowperiod=26, matype=0):
    ppo = talib.PPO(close_prices, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
    return ppo

def GetROC(close_prices, timeperiod=10):
    roc = talib.ROC(close_prices, timeperiod)
    return roc

def GetROCP(close_prices, timeperiod=10):
    rocp = talib.ROCP(close_prices, timeperiod)
    return rocp

def GetROCR(close_prices, timeperiod=10):
    rocr = talib.ROCR(close_prices, timeperiod)
    return rocr

def GetROCR100(close_prices, timeperiod=10):
    rocr100 = talib.ROCR100(close_prices, timeperiod)
    return rocr100

def GetRSI(close_prices):
    rsi = talib.RSI(close_prices)
    return rsi

def GetSTOCH(high_prices, low_prices, close_prices, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)
    return slowk, slowd

def GetSTOCHF(high_prices, low_prices, close_prices, fastk_period=5, fastd_period=3, fastd_matype=0):
    fastk, fastd = talib.STOCHF(high_prices, low_prices, close_prices, fastk_period, fastd_period, fastd_matype)
    return fastk, fastd

def GetSTOCHRSI(close_prices, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
    fastk, fastd = talib.STOCHRSI(close_prices, timeperiod, fastk_period, fastd_period, fastd_matype)
    return fastk, fastd

def GetTRIX(close_prices, timeperiod=14):
    trix = talib.TRIX(close_prices, timeperiod)
    return trix

def GetULTOSC(high_prices, low_prices, close_prices, timeperiod1=7, timeperiod2=14, timeperiod3=28):
    ultosc = talib.ULTOSC(high_prices, low_prices, close_prices, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
    return ultosc

def GetWILLR(high_prices, low_prices, close_prices, timeperiod=14):
    willr = talib.WILLR(high_prices, low_prices, close_prices, timeperiod)
    return willr