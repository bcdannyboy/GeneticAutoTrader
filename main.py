
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd
from calculators import *
from strategies.buyandhold import buy_and_hold
from strategies.scorebased import execute_strategy
from utils import aggregate_weights, save_weights_to_json
from yf.gather import GatherData
import concurrent.futures

def Test(Weights, stock_data, spy_data):
    # Ensure that stock_data and spy_data are in the correct format
    if isinstance(stock_data, pd.DataFrame) and 'Close' in stock_data.columns:
        prices = stock_data['Close'].values
    else:
        raise ValueError("stock_data must be a pandas DataFrame with a 'Close' column")

    if isinstance(spy_data, pd.DataFrame) and 'Close' in spy_data.columns:
        spy_prices = spy_data['Close'].values
    else:
        raise ValueError("spy_data must be a pandas DataFrame with a 'Close' column")

    # Extract other required data from stock_data
    volume = np.array(stock_data['Volume'])
    open_prices = np.array(stock_data['Open'])
    high_prices = np.array(stock_data['High'])
    low_prices = np.array(stock_data['Low'])
    close_prices = np.array(stock_data['Close'])

    # Calculations based on indicators
    overlap_indicators = CalculateOverlapIndicators(prices, stock_data, Weights["Overlap"])
    normalized_overlap_scores = NormalizeOverlapScore(overlap_indicators)

    price_transformer_indicators = CalculatePriceTransformerIndicators(prices, open_prices, high_prices, low_prices, close_prices, Weights["PriceTransformer"])
    normalized_price_transformer_scores = NormalizePriceTransformerScore(price_transformer_indicators)

    VolatilityIndicators = CalculateVolatilityIndicators(prices, high_prices, low_prices, close_prices, Weights["Volatility"])
    normalized_volatility_scores = NormalizeVolatilityIndicators(VolatilityIndicators)
    
    MomentumIndicators = CalculateMomentumIndicators(prices, open_prices, high_prices, low_prices, close_prices, volume, Weights["Momentum"])
    normalized_momentum_scores = CalculateNormalizedMomentumScores(MomentumIndicators)

    # Calculate final scores
    final_scores = []
    for i in range(len(prices)):
        m_score = normalized_momentum_scores[i]
        o_score = normalized_overlap_scores[i]
        p_score = normalized_price_transformer_scores[i]
        v0_score = normalized_volatility_scores[0][i]
        v1_score = normalized_volatility_scores[1][i]
        v2_score = normalized_volatility_scores[2][i]
        final_score = (
            (m_score * Weights["MomentumCategory"]) + 
            (o_score + Weights["OverlapCategory"]) + 
            (p_score * Weights["PriceTransformerCategory"]) + 
            (((v0_score + v1_score + v2_score)/3) * Weights["VolatilityCategory"])
        ) / 4
        final_scores.append(final_score)
    
    # Buy and hold strategy values
    spy_hold_values = buy_and_hold(spy_prices)
    hold_values = buy_and_hold(prices)
    strategy_values = execute_strategy(final_scores, prices)
    
    # Calculate profit/loss for both strategies
    strategy_profit_loss = strategy_values[-1] - strategy_values[0]
    hold_profit_loss = hold_values[-1] - hold_values[0]
    spy_profit_loss = spy_hold_values[-1] - spy_hold_values[0]
    
    return strategy_profit_loss, hold_profit_loss, spy_profit_loss

def initialize_population(size, weights):
    population = []
    for _ in range(size):
        individual = {}
        for category in weights:
            if isinstance(weights[category], dict):
                individual[category] = {key: random.uniform(0, 1) for key in weights[category]}
            else:
                individual[category] = random.uniform(0, 1)
        population.append(individual)
    return population

def calculate_fitness(individual, stock_data, spy_data):
    strategy_score, _, _ = Test(individual, stock_data, spy_data)
    return strategy_score

def select_parents(population, fitnesses, tournament_size=3):
    selected_parents = []
    for _ in range(2):  # Selecting two parents
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        selected_parents.append(max(tournament, key=lambda item: item[1])[0])
    return selected_parents[0], selected_parents[1]

def crossover(parent1, parent2):
    child1, child2 = {}, {}
    crossover_point = random.choice(list(parent1.keys()))
    for key in parent1:
        if key == crossover_point:
            child1[key], child2[key] = parent2[key], parent1[key]
        else:
            child1[key], child2[key] = parent1[key], parent2[key]
    return child1, child2

def mutate(individual, mutation_rate=0.01):
    for category in individual:
        if isinstance(individual[category], dict):  # Check if the value is a dictionary
            if random.random() < mutation_rate:
                for key in individual[category]:
                    individual[category][key] = random.uniform(0, 1)
        else:  # For float values
            if random.random() < mutation_rate:
                individual[category] = random.uniform(0, 1)
    return individual

def genetic_algorithm(stock_data, spy_data, population_size, num_generations):
    population = initialize_population(population_size, Weights)
    best_individual = None
    best_fitness = float('-inf')

    for generation in range(num_generations):
        print(f"Initiating generation {generation+1}/{num_generations}...")

        # Parallel fitness calculation with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all fitness calculations to the executor
            future_fitness = {executor.submit(calculate_fitness, individual, stock_data, spy_data): individual for individual in population}

            fitnesses = []
            for future in concurrent.futures.as_completed(future_fitness):
                individual = future_fitness[future]
                try:
                    fitness = future.result()
                    fitnesses.append(fitness)
                    print(f"Individual Fitness: {fitness}")  # Debug print for each individual's fitness
                except Exception as exc:
                    print(f"{individual} generated an exception: {exc}")

        # Track the best individual
        for i, fitness in enumerate(fitnesses):
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = population[i]
                print(f"New Best Individual Fitness: {best_fitness}")  # Debug print for new best individual

        new_population = []
        for _ in range(len(population) // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            new_population.extend([offspring1, offspring2])
        population = new_population

        print(f"End of Generation {generation+1}/{num_generations} - Best Fitness: {best_fitness}")

    return best_individual

def optimize_overall_weights(aggregated_weights, stock_tickers, num_days_range, population_size, num_generations):
    """
    Run an optimization process on the aggregated weights to find the best overall weights.
    This function initializes the genetic algorithm with the aggregated weights and runs further optimization.
    """
    # Initialize the population with the aggregated weights
    initial_population = [dict(aggregated_weights) for _ in range(population_size)]

    # Combine data from all tickers and timeframes for the optimization run
    combined_spy_data = pd.DataFrame()
    combined_stock_data = pd.DataFrame()
    for ticker in stock_tickers:
        for days_ago in num_days_range:
            start_date = (datetime.now() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")

            spy_data = GatherData("SPY", start_date, end_date)
            stock_data = GatherData(ticker, start_date, end_date)
            combined_spy_data = pd.concat([combined_spy_data, spy_data])
            combined_stock_data = pd.concat([combined_stock_data, stock_data])

    # Run the genetic algorithm with the initial population
    best_overall_weights = genetic_algorithm(combined_stock_data, combined_spy_data, population_size, num_generations, initial_population)

    return best_overall_weights

def run_genetic_algorithm(ticker, start_date, end_date, population_size, num_generations):
    spy_data = GatherData("SPY", start_date, end_date)
    stock_data = GatherData(ticker, start_date, end_date)
    print(f"Running Genetic Algorithm for {ticker} from {start_date} to {end_date}...")
    best_weights = genetic_algorithm(stock_data, spy_data, population_size, num_generations)
    StrategyScore, _, _ = Test(ticker, start_date, best_weights)
    return ticker, start_date, end_date, best_weights, StrategyScore


if __name__ == '__main__':
    Weights = {
        "OverlapCategory": 0.25,
        "Overlap": {
            "delta_highband": 0.1,
            "delta_middleband": 0.1,
            "delta_lowerband": 0.1,
            "delta_DEMA": 0.05,
            "delta_EMA": 0.05,
            "delta_HT_TRENDLINE": 0.05,
            "delta_KAMA": 0.05,
            "delta_MA": 0.05,
            "delta_MAVP": 0.05,
            "delta_MIDPOINT": 0.05,
            "delta_MIDPRICE": 0.05,
            "delta_SAR": 0.05,
            "delta_SAREXT": 0.05,
            "delta_SMA": 0.05,
            "delta_T3": 0.05,
            "delta_TEMA": 0.05,
            "delta_TRIMA": 0.05,
            "delta_WMA": 0.05,
        },
        "PriceTransformerCategory": 0.25,
        "PriceTransformer": {
            "delta_avg_price": 0.25,
            "delta_med_price": 0.25,
            "delta_typ_price": 0.25,
            "delta_wcl_price": 0.25,
        },
        "VolatilityCategory": 0.25,
        "Volatility": {
            "delta_ATR": 0.4,
            "delta_NATR": 0.3,
            "delta_TRANGE": 0.3,
        },
        "MomentumCategory": 0.25,
        "Momentum": {
            "ADX": 0.2,
            "ADXR": 0.1,
            "APO": 0.05,
            "AROON_UP": 0.1,
            "AROON_DOWN": 0.1,
            "AROONOSC": 0.05,
            "BOP": 0.05,
            "CCI": 0.1,
            "CMO": 0.05,
            "DX": 0.05,
            "MACD_HIGH": 0.15,
            "MACD_MID": 0.15,
            "MACD_LOW": 0.15,
            "MACDEXT_HIGH": 0.05,
            "MACDEXT_MID": 0.05,
            "MACDEXT_LOW": 0.05,
            "MACDFIX_HIGH": 0.05,
            "MACDFIX_MID": 0.05,
            "MACDFIX_LOW": 0.05,
            "MFI": 0.1,
            "MINUS_DI": 0.05,
            "MINUS_DM": 0.05,
            "MOM": 0.1,
            "PLUS_DI": 0.05,
            "PLUS_DM": 0.05,
            "PPO": 0.05,
            "ROC": 0.1,
            "ROCP": 0.05,
            "ROCR": 0.05,
            "ROCR100": 0.05,
            "RSI": 0.1,
            "STOCH_SLOWK": 0.1,
            "STOCH_SLOWD": 0.1,
            "STOCHF_FASTK": 0.05,
            "STOCHF_FASTD": 0.05,
            "STOCHRSI_FASTK": 0.05,
            "STOCHRSI_FASTD": 0.05,
            "TRIX": 0.05,
            "ULTOSC": 0.1,
            "WILLR": 0.05,
        }
    }
    
    stock_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "JNJ", "CSCO", "BYND"]
    num_days_range = range(30, 365*3)
    population_size = 100
    num_generations = 50

    all_best_weights = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_genetic_algorithm, ticker, (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"), population_size, num_generations)
                   for ticker in stock_tickers for days_ago in num_days_range]
        for future in concurrent.futures.as_completed(futures):
            _, _, _, best_weights, _ = future.result()
            all_best_weights.append(best_weights)

    aggregated_weights = aggregate_weights(all_best_weights)
    best_overall_weights = optimize_overall_weights(aggregated_weights, stock_tickers, num_days_range, population_size, num_generations)

    overall_weights_filename = "best_overall_weights.json"
    save_weights_to_json(best_overall_weights, overall_weights_filename)

    # Final testing
    final_results = []
    for ticker in stock_tickers:
        start_date = (datetime.now() - timedelta(days=random.choice(num_days_range))).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        spy_data = GatherData("SPY", start_date, end_date)
        stock_data = GatherData(ticker, start_date, end_date)

        StrategyScore, UnderlyingHoldScore, SP500HoldScore = Test(ticker, start_date, best_overall_weights)
        final_results.append({
            'Ticker': ticker,
            'Start Date': start_date,
            'End Date': end_date,
            'Strategy Score': StrategyScore,
            'Underlying Hold Score': UnderlyingHoldScore,
            'SP500 Hold Score': SP500HoldScore
        })

    for result in final_results:
        print(f"{result['Ticker']} ({result['Start Date']} to {result['End Date']}):")
        print(f"  Strategy Score: {result['Strategy Score']}")
        print(f"  Underlying Hold Score: {result['Underlying Hold Score']}")
        print(f"  SP500 Hold Score: {result['SP500 Hold Score']}")

    print("\nBest Overall Weights from Genetic Algorithm:\n", best_overall_weights)