
from matplotlib import pyplot as plt


def Plot(stock_data, prices, final_scores, strategy_values, hold_values, spy_data, spy_hold_values, Ticker):
    
    # Plotting
    plt.figure(figsize=(12, 8))

    # Create a subplot for stock prices and final scores
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(stock_data.index, prices, label='Stock Price', color='blue')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Create a second y-axis for final scores
    ax2 = ax1.twinx()
    ax2.plot(stock_data.index, final_scores, label='Final Score', color='green')
    ax2.set_ylabel('Score', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    plt.title(f'Stock Price and Final Score Over Time for {Ticker}')

    # Plot strategy and buy-and-hold portfolio values in the second subplot
    plt.subplot(2, 1, 2)
    plt.plot(stock_data.index, strategy_values, label='Strategy Portfolio Value', color='red')
    plt.plot(stock_data.index, hold_values, label=f'Buy and Hold {Ticker}', color='blue')
    plt.plot(spy_data.index, spy_hold_values, label='Buy and Hold SPY', color='green')  # Plot SPY values
    plt.title(f'Strategy vs. Buy and Hold Portfolio Values for {Ticker} and SPY')
    plt.ylabel('Portfolio Value')
    plt.xlabel('Date')
    plt.legend()

    plt.tight_layout()
    plt.show()