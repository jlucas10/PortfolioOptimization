import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Fetch stock price data from Yahoo Finance
def fetch_portfolio_data(tickers, start_date=None, end_date=None, period="1y"):
    data = pd.DataFrame()
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        if start_date and end_date:
            data[ticker] = stock.history(start=start_date, end=end_date)["Close"]
        else:
            data[ticker] = stock.history(period=period)["Close"]
    return data.dropna()

# Calculate returns and risk metrics
def calculate_portfolio_metrics(data):
    returns = np.log(data / data.shift(1)).dropna()  # Daily log returns
    expected_returns = returns.mean() * 252  # Annualized returns
    cov_matrix = returns.cov() * 252  # Annualized covariance
    return returns, expected_returns, cov_matrix 

# Optimize portfolio for max Sharpe Ratio
def optimize_portfolio(tickers, train_data):
    returns, expected_returns, cov_matrix = calculate_portfolio_metrics(train_data)  # Compute metrics
    
    num_assets = len(tickers)
    
    # Define negative Sharpe Ratio to minimize
    def negative_sharpe(weights):
        port_return = np.sum(weights * expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_volatility
        return -sharpe
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)  # Weights sum to 1
    bounds = tuple((0, 0.2) for _ in range(num_assets))  # 0-20% per stock
    initial_weights = np.array([1.0 / num_assets] * num_assets)  # Start with equal weights
    
    result = minimize(negative_sharpe, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)  # Run optimization
    
    if not result.success:
        print(f"Optimization Details: {result}")
        raise ValueError("Optimization failed: ", result.message)
    
    weights = result.x  # Optimized weights
    port_return = np.sum(weights * expected_returns)  # Predicted return
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Predicted risk
    sharpe_ratio = port_return / port_volatility  # Sharpe Ratio
    
    return weights, port_return, port_volatility, sharpe_ratio, expected_returns, cov_matrix

# Backtest portfolio on test data
def backtest_portfolio(tickers, weights, test_data):
    returns = np.log(test_data / test_data.shift(1)).dropna()  # Daily returns in test period
    portfolio_returns = np.sum(returns * weights, axis=1)  # Daily portfolio returns
    actual_return = portfolio_returns.mean() * 252  # Annualized actual return
    actual_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized actual volatility
    
    # Calculate portfolio value over time
    portfolio_value = (1 + portfolio_returns).cumprod()  # Cumulative product starting at 1
    return actual_return, actual_volatility, portfolio_value

# Simulate random portfolios
def monte_carlo_simulation(tickers, expected_returns, cov_matrix, num_simulations=5000):
    num_assets = len(tickers)
    returns = []
    volatilities = []
    
    for _ in range(num_simulations):
        weights = np.random.uniform(0, 0.2, num_assets)  # Random weights 0-20%
        weights /= np.sum(weights)  # Normalize to 1
        if np.any(weights > 0.2):
            continue  # Skip if any weight > 20%
        
        port_return = np.sum(weights * expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        returns.append(port_return)
        volatilities.append(port_volatility)
    
    return np.array(returns), np.array(volatilities)

# Plot pie chart of non-zero weights only
def plot_portfolio(tickers, weights):
    filtered_tickers = [t for t, w in zip(tickers, weights) if w > 0.001]  # Keep non-zero tickers
    filtered_weights = [w for w in weights if w > 0.001]  # Keep non-zero weights
    
    if not filtered_weights:
        print("No non-zero weights to plot!")
        return
    
    plt.figure(figsize=(10, 10))
    plt.pie(filtered_weights, labels=filtered_tickers, autopct='%1.1f%%', startangle=90)
    plt.title("Portfolio Allocation (20% Max per Stock)")
    plt.show()

# Plot efficient frontier with simulated portfolios
def plot_efficient_frontier(tickers, weights, port_return, port_volatility, expected_returns, cov_matrix):
    sim_returns, sim_volatilities = monte_carlo_simulation(tickers, expected_returns, cov_matrix)  # Simulate portfolios
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sim_volatilities, sim_returns, c=(sim_returns / sim_volatilities), cmap='viridis', alpha=0.5, label='Simulated Portfolios')  # Simulated points
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(port_volatility, port_return, c='red', marker='*', s=200, label='Optimized Portfolio')  # Optimized point
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier with Optimized Portfolio")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot backtest portfolio value over time
def plot_backtest(portfolio_value):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_value.index, portfolio_value, label='Portfolio Value', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Normalized)")
    plt.title("Backtest: Portfolio Performance Over Test Period")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution block
if __name__ == "__main__":
    tickers = ["TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "DIS", "NFLX"]  # Stock list
    
    # Define dates (1 year total, 75% train, 25% test)
    end_date = "2025-03-06"  # Today’s date
    start_date = "2024-03-07"  # 1 year ago
    train_end = "2024-12-06"  # 9 months from start
    
    try:
        # Fetch full data and split
        full_data = fetch_portfolio_data(tickers, start_date, end_date)
        train_data = full_data[:train_end]  # First 9 months
        test_data = full_data[train_end:]   # Last 3 months
        
        # Optimize on training data
        weights, pred_ret, pred_vol, sharpe, exp_returns, cov_matrix = optimize_portfolio(tickers, train_data)
        cleaned_weights = [float(max(0, round(w, 4))) if w > 0.001 else 0 for w in weights]  # Clean weights
        weights_dict = dict(zip(tickers, cleaned_weights))  # Create weights dictionary
        
        # Backtest on test data
        actual_ret, actual_vol, portfolio_value = backtest_portfolio(tickers, cleaned_weights, test_data)
        
        # Print results
        print(f"Optimal Weights (Training): {weights_dict}")
        print(f"Predicted Return (Training): {pred_ret:.4f}")
        print(f"Predicted Volatility (Training): {pred_vol:.4f}")
        print(f"Sharpe Ratio (Training): {sharpe:.4f}")
        print(f"Actual Return (Test): {actual_ret:.4f}")
        print(f"Actual Volatility (Test): {actual_vol:.4f}")
        
        # Plot results
        plot_portfolio(tickers, cleaned_weights)  # Only non-zero weights
        plot_efficient_frontier(tickers, cleaned_weights, pred_ret, pred_vol, exp_returns, cov_matrix)
        plot_backtest(portfolio_value)
    except Exception as e:
        print(f"Error: {e}")