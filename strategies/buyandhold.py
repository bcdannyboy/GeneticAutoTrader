
def buy_and_hold(prices, initial_cash=1000):
    shares = initial_cash / prices[0]
    return [shares * price for price in prices]
