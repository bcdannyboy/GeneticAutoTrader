
import talib

def NormalizePriceTransformerScore(price_transformer_indicators):
    normalized_scores = []
    for indicators in price_transformer_indicators:
        max_abs_value = max(abs(value) for value in indicators.values())
        score_sum = sum(indicators.values())
        normalized_score = score_sum / max(max_abs_value, 1)  # Avoid division by zero
        normalized_scores.append(normalized_score)
    return normalized_scores

def GetAVGPrice(open_prices, high_prices, low_prices, close_prices):
    return talib.AVGPRICE(open_prices, high_prices, low_prices, close_prices)

def GetMedPrice(high_prices, low_prices):
    return talib.MEDPRICE(high_prices, low_prices)

def GetTypPrice(high_prices, low_prices, close_prices):
    return talib.TYPPRICE(high_prices, low_prices, close_prices)

def GetWCLPrice(high_prices, low_prices, close_prices):
    return talib.WCLPRICE(high_prices, low_prices, close_prices)