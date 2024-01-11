
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from indicators.momentum import *
from indicators.overlap import *
from indicators.pricetransformers import *
from indicators.volatility import *
from utils import FillInitialNaN

def CalculateOverlapIndicators(prices, stock_data, overlap_weights):
    BBANDS = GetBBANDS(prices)
    DEMA = FillInitialNaN(GetDEMA(prices), prices)
    EMA = FillInitialNaN(GetEMA(prices), prices)
    HT_TRENDLINE = FillInitialNaN(GetHT_TRENDLINE(prices), prices)
    KAMA = FillInitialNaN(GetKAMA(prices), prices)
    MA = FillInitialNaN(GetMA(prices), prices)
    MAVP = FillInitialNaN(GetMAVP(stock_data), prices)
    MIDPOINT = FillInitialNaN(GetMidPoint(prices), prices)
    MIDPRICE = FillInitialNaN(GetMidPrice(prices), prices)
    SAR = FillInitialNaN(GetSAR(prices), prices)
    SAREXT = FillInitialNaN(GetSAREXT(prices), prices)
    SMA = FillInitialNaN(GetSMA(prices), prices)
    T3 = FillInitialNaN(GetT3(prices), prices)
    TEMA = FillInitialNaN(GetTEMA(prices), prices)
    TRIMA = FillInitialNaN(GetTRIMA(prices), prices)
    WMA = FillInitialNaN(GetWMA(prices), prices)

    PriceIndicators = []
    for i in range(len(prices)):
        Indicators = {
            "delta_highband": (BBANDS[0][i] - prices[i]) * overlap_weights["delta_highband"],
            "delta_middleband": (BBANDS[1][i] - prices[i]) * overlap_weights["delta_middleband"],
            "delta_lowerband": (BBANDS[2][i] - prices[i]) * overlap_weights["delta_lowerband"],
            "delta_DEMA": (DEMA[i] - prices[i]) * overlap_weights["delta_DEMA"],
            "delta_EMA": (EMA[i] - prices[i]) * overlap_weights["delta_EMA"],
            "delta_HT_TRENDLINE": (HT_TRENDLINE[i] - prices[i]) * overlap_weights["delta_HT_TRENDLINE"],
            "delta_KAMA": (KAMA[i] - prices[i]) * overlap_weights["delta_KAMA"],
            "delta_MA": (MA[i] - prices[i]) * overlap_weights["delta_MA"],
            "delta_MAVP": (MAVP[i] - prices[i]) * overlap_weights["delta_MAVP"],
            "delta_MIDPOINT": (MIDPOINT[i] - prices[i]) * overlap_weights["delta_MIDPOINT"],
            "delta_MIDPRICE": (prices[i] - MIDPRICE[i]) * overlap_weights["delta_MIDPRICE"],
            "delta_SAR": (SAR[i] - prices[i]) * overlap_weights["delta_SAR"],
            "delta_SAREXT": (SAREXT[i] - prices[i]) * overlap_weights["delta_SAREXT"],
            "delta_SMA": (SMA[i] - prices[i]) * overlap_weights["delta_SMA"],
            "delta_T3": (T3[i] - prices[i]) * overlap_weights["delta_T3"],
            "delta_TEMA": (TEMA[i] - prices[i]) * overlap_weights["delta_TEMA"],
            "delta_TRIMA": (TRIMA[i] - prices[i]) * overlap_weights["delta_TRIMA"],
            "delta_WMA": (WMA[i] - prices[i]) * overlap_weights["delta_WMA"],
        }
        PriceIndicators.append(Indicators)
        
    return PriceIndicators
 
def CalculatePriceTransformerIndicators(prices, open_prices, high_prices, low_prices, close_prices, price_weights):
    avg_prices = GetAVGPrice(open_prices, high_prices, low_prices, close_prices)
    med_prices = GetMedPrice(high_prices, low_prices)
    typ_prices = GetTypPrice(high_prices, low_prices, close_prices)
    wcl_prices = GetWCLPrice(high_prices, low_prices, close_prices)
    
    indicators = []
    for i in range(len(prices)):
        Indicators = {
            "delta_avg_price": (avg_prices[i] - prices[i]) * price_weights["delta_avg_price"],
            "delta_med_price": (med_prices[i] - prices[i]) * price_weights["delta_med_price"],
            "delta_typ_price": (typ_prices[i] - prices[i]) * price_weights["delta_typ_price"],
            "delta_wcl_price": (wcl_prices[i] - prices[i]) * price_weights["delta_wcl_price"],
        }
        indicators.append(Indicators)
        
    return indicators
 
def CalculateVolatilityIndicators(prices, high_prices, low_prices, close_prices, vol_weights):
    ATR = GetATR(high_prices, low_prices, close_prices)
    NATR = GetNATR(high_prices, low_prices, close_prices)
    TRANGE = GetTRANGE(high_prices, low_prices, close_prices)

    # Apply weights to the indicators
    weighted_delta_ATR = vol_weights["delta_ATR"] * (ATR - prices)
    weighted_delta_NATR = vol_weights["delta_NATR"] * (NATR - prices)
    weighted_delta_TRANGE = vol_weights["delta_TRANGE"] * (TRANGE - prices)

    return [weighted_delta_ATR, weighted_delta_NATR, weighted_delta_TRANGE]

def CalculateMomentumIndicators(prices, open_prices, high_prices, low_prices, close_prices, volume, momentum_weights):
    high_prices = high_prices.astype(np.float64)
    low_prices = low_prices.astype(np.float64)
    close_prices = close_prices.astype(np.float64)
    volume = volume.astype(np.float64)
    
    ADX = GetADX(high_prices, low_prices, close_prices)
    ADXR = GetADXR(high_prices, low_prices, close_prices)
    APO = GetAPO(close_prices)
    AROON = GetAROON(high_prices, low_prices)
    AROONOSC = GetAROONOSC(high_prices, low_prices)
    BOP = GetBOP(open_prices, high_prices, low_prices, close_prices)
    CCI = GetCCI(high_prices, low_prices, close_prices)
    CMO = GetCMO(close_prices)
    DX = GetDX(high_prices, low_prices, close_prices)
    MACD = GetMACD(close_prices)
    MACD_HIGH = MACD[0]
    MACD_MID = MACD[1]
    MACD_LOW = MACD[2]
    MACDEXT = GetMACDEXT(close_prices)
    MACDEXT_HIGH = MACDEXT[0]
    MACDEXT_MID = MACDEXT[1]
    MACDEXT_LOW = MACDEXT[2]
    MACDFIX = GetMACDFIX(close_prices)
    MACDFIX_HIGH = MACDFIX[0]
    MACDFIX_MID = MACDFIX[1]
    MACDFIX_LOW = MACDFIX[2]
    MFI = GetMFI(high_prices, low_prices, close_prices, volume)
    MINUS_DI = GetMINUS_DI(high_prices, low_prices, close_prices)
    MINUS_DM = GetMINUS_DM(high_prices, low_prices)
    MOM = GetMOM(close_prices)
    PLUS_DI = GetPLUS_DI(high_prices, low_prices, close_prices)
    PLUS_DM = GetPLUS_DM(high_prices, low_prices)
    PPO = GetPPO(close_prices)
    ROC = GetROC(close_prices)
    ROCP = GetROCP(close_prices)
    ROCR = GetROCR(close_prices)
    ROCR100 = GetROCR100(close_prices)
    RSI = GetRSI(close_prices)
    STOCHS = [GetSTOCH(high_prices, low_prices, close_prices) for _ in range(len(prices))]
    STOCHFS = [GetSTOCHF(high_prices, low_prices, close_prices) for _ in range(len(prices))]
    STOCHRSIS = [GetSTOCHRSI(close_prices) for _ in range(len(prices))]
    TRIX = GetTRIX(close_prices)
    ULTOSC = GetULTOSC(high_prices, low_prices, close_prices)
    WILLR = GetWILLR(high_prices, low_prices, close_prices)    
   
    indicators = []
    for i in range(len(prices)):
        Indicator = {
            "ADX": (ADX[i]) * momentum_weights["ADX"] if not np.isnan(ADX[i]) else 0,
            "ADXR": (ADXR[i]) * momentum_weights["ADXR"] if not np.isnan(ADXR[i]) else 0,
            "APO": (APO[i]) * momentum_weights["APO"] if not np.isnan(APO[i]) else 0,
            "AROON_UP": (AROON[0][i]) * momentum_weights["AROON_UP"] if not np.isnan(AROON[0][i]) else 0,
            "AROON_DOWN": (AROON[1][i]) * momentum_weights["AROON_DOWN"] if not np.isnan(AROON[1][i]) else 0,
            "AROONOSC": (AROONOSC[i]) * momentum_weights["AROONOSC"] if not np.isnan(AROONOSC[i]) else 0,
            "BOP": (BOP[i]) * momentum_weights["BOP"] if not np.isnan(BOP[i]) else 0,
            "CCI": (CCI[i]) * momentum_weights["CCI"] if not np.isnan(CCI[i]) else 0,
            "CMO": (CMO[i]) * momentum_weights["CMO"] if not np.isnan(CMO[i]) else 0,
            "DX": (DX[i]) * momentum_weights["DX"] if not np.isnan(DX[i]) else 0,
            "MACD_HIGH": (MACD_HIGH[i]) * momentum_weights["MACD_HIGH"] if not np.isnan(MACD_HIGH[i]) else 0,
            "MACD_MID": (MACD_MID[i]) * momentum_weights["MACD_MID"] if not np.isnan(MACD_MID[i]) else 0,
            "MACD_LOW": (MACD_LOW[i]) * momentum_weights["MACD_LOW"] if not np.isnan(MACD_LOW[i]) else 0,
            "MACDEXT_HIGH": (MACDEXT_HIGH[i]) * momentum_weights["MACDEXT_HIGH"] if not np.isnan(MACDEXT_HIGH[i]) else 0,
            "MACDEXT_MID": (MACDEXT_MID[i]) * momentum_weights["MACDEXT_MID"] if not np.isnan(MACDEXT_MID[i]) else 0,
            "MACDEXT_LOW": (MACDEXT_LOW[i]) * momentum_weights["MACDEXT_LOW"] if not np.isnan(MACDEXT_LOW[i]) else 0,
            "MACDFIX_HIGH": (MACDFIX_HIGH[i]) * momentum_weights["MACDFIX_HIGH"] if not np.isnan(MACDFIX_HIGH[i]) else 0,
            "MACDFIX_MID": (MACDFIX_MID[i]) * momentum_weights["MACDFIX_MID"] if not np.isnan(MACDFIX_MID[i]) else 0,
            "MACDFIX_LOW": (MACDFIX_LOW[i]) * momentum_weights["MACDFIX_LOW"] if not np.isnan(MACDFIX_LOW[i]) else 0,
            "MFI": (MFI[i]) * momentum_weights["MFI"] if not np.isnan(MFI[i]) else 0,
            "MINUS_DI": (MINUS_DI[i]) * momentum_weights["MINUS_DI"] if not np.isnan(MINUS_DI[i]) else 0,
            "MINUS_DM": (MINUS_DM[i]) * momentum_weights["MINUS_DM"] if not np.isnan(MINUS_DM[i]) else 0,
            "MOM": (MOM[i]) * momentum_weights["MOM"] if not np.isnan(MOM[i]) else 0,
            "PLUS_DI": (PLUS_DI[i]) * momentum_weights["PLUS_DI"] if not np.isnan(PLUS_DI[i]) else 0,
            "PLUS_DM": (PLUS_DM[i]) * momentum_weights["PLUS_DM"] if not np.isnan(PLUS_DM[i]) else 0,
            "PPO": (PPO[i]) * momentum_weights["PPO"] if not np.isnan(PPO[i]) else 0,
            "ROC": (ROC[i]) * momentum_weights["ROC"] if not np.isnan(ROC[i]) else 0,
            "ROCP": (ROCP[i]) * momentum_weights["ROCP"] if not np.isnan(ROCP[i]) else 0,
            "ROCR": (ROCR[i]) * momentum_weights["ROCR"] if not np.isnan(ROCR[i]) else 0,
            "ROCR100": (ROCR100[i]) * momentum_weights["ROCR100"] if not np.isnan(ROCR100[i]) else 0,
            "RSI": (RSI[i]) * momentum_weights["RSI"] if not np.isnan(RSI[i]) else 0,
            "STOCH_SLOWK": np.nan_to_num(STOCHS[i][0]) * momentum_weights["STOCH_SLOWK"],
            "STOCH_SLOWD": np.nan_to_num(STOCHS[i][1]) * momentum_weights["STOCH_SLOWD"],
            "STOCHF_FASTK": np.nan_to_num(STOCHFS[i][0]) * momentum_weights["STOCHF_FASTK"],
            "STOCHF_FASTD": np.nan_to_num(STOCHFS[i][1]) * momentum_weights["STOCHF_FASTD"],
            "STOCHRSI_FASTK": np.nan_to_num(STOCHRSIS[i][0]) * momentum_weights["STOCHRSI_FASTK"],
            "STOCHRSI_FASTD": np.nan_to_num(STOCHRSIS[i][1]) * momentum_weights["STOCHRSI_FASTD"],
            "TRIX": (TRIX[i]) * momentum_weights["TRIX"] if not np.isnan(TRIX[i]) else 0,
            "ULTOSC": (ULTOSC[i]) * momentum_weights["ULTOSC"] if not np.isnan(ULTOSC[i]) else 0,
            "WILLR": (WILLR[i]) * momentum_weights["WILLR"] if not np.isnan(WILLR[i]) else 0,
        }
        
        indicators.append(Indicator)
         
         
    normalized_indicators = []
    for indicator in indicators:
        normalized_indicator = {key: np.tanh(value) for key, value in indicator.items()}
        normalized_indicators.append(normalized_indicator)

    return normalized_indicators                                                        
    