import numpy as np


def execute_strategy(final_scores, prices, initial_cash=1000, initial_stop_loss_percent=0.95):
    final_scores = np.nan_to_num(final_scores, nan=np.nanmean(final_scores))
    prices = np.nan_to_num(prices, nan=np.nanmean(prices))
    cash = initial_cash
    holdings = 0
    portfolio_values = []
    trailing_stop_loss_level = None

    for i in range(len(final_scores)):
        score = final_scores[i]
        price = prices[i]

        # Adjust buy/sell amount based on score
        if score > 2:
            buy_amount = cash  # Buy with all available cash
        elif score > 1:
            buy_amount = 0.5 * cash
        elif score > 0:
            buy_amount = 0.25 * cash
        else:
            buy_amount = 0

        sell_percent = -0.25 if score < -1 else (-0.5 if score < -2 else 0)

        # Execute buy or sell
        if buy_amount > 0:
            shares_to_buy = buy_amount / price
            holdings += shares_to_buy
            cash -= buy_amount
            trailing_stop_loss_level = max(trailing_stop_loss_level, price * initial_stop_loss_percent) if trailing_stop_loss_level else price * initial_stop_loss_percent

        if holdings > 0 and sell_percent < 0:
            shares_to_sell = holdings * abs(sell_percent)
            holdings -= shares_to_sell
            cash += shares_to_sell * price

        # Trailing stop loss
        if holdings > 0 and price < trailing_stop_loss_level:
            cash += holdings * price
            holdings = 0
            trailing_stop_loss_level = None

        portfolio_value = cash + holdings * price
        portfolio_values.append(portfolio_value)

    return portfolio_values
