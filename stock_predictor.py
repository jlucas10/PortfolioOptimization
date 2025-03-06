import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pytz  # Import pytz

def fetch_stock_data(ticker, period="1y"):
    """Fetch historical stock data for a given ticker."""
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)[['Close', 'Volume']]
    return data

def prepare_data(data):
    """Add features and target to the data."""
    data['Prev_Close'] = data['Close'].shift(1)
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['Target'] = data['Close'].shift(-1)
    return data.dropna()

def train_and_predict(ticker, cutoff_date=None):
    """Train a linear regression model and predict the next day's price."""
    data = fetch_stock_data(ticker)
    data = prepare_data(data)
    
    # If cutoff_date is provided, use data up to that point
    if cutoff_date:
        cutoff_date = pd.to_datetime(cutoff_date)
        # Convert cutoff_date to the same timezone as the data index
        cutoff_date = cutoff_date.tz_localize(data.index.tz)
        data = data[data.index <= cutoff_date]
    
    X = data[['Prev_Close', 'Volume', 'MA_5']]
    y = data['Target']
    
    X_train = X[:-1]
    y_train = y[:-1]
    X_predict = X.tail(1)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted_price = model.predict(X_predict)[0]
    
    # Get actual next day price if cutoff_date is used
    actual_price = None
    next_date = X_predict.index[0] + pd.Timedelta(days=1)
    if cutoff_date and next_date in data.index:
        actual_price = data.loc[next_date, 'Close']
    
    return data['Close'].tail(30), predicted_price, actual_price, next_date

def plot_results(ticker, historical_data, predicted_price, actual_price, next_date):
    """Plot historical data, predicted, and actual price with labels."""
    plt.figure(figsize=(10, 6))
    plt.plot(historical_data.index, historical_data, label='Historical Price', color='blue')
    
    # Predicted price
    plt.scatter([next_date], [predicted_price], color='red', label='Predicted Price', zorder=5)
    plt.annotate(f"${predicted_price:.2f}", 
                 (next_date, predicted_price), 
                 textcoords="offset points", 
                 xytext=(5, 5), 
                 ha='left', 
                 fontsize=10, 
                 color='red')
    
    # Actual price (if available)
    if actual_price:
        plt.scatter([next_date], [actual_price], color='green', label='Actual Price', zorder=5)
        plt.annotate(f"${actual_price:.2f}", 
                     (next_date, actual_price), 
                     textcoords="offset points", 
                     xytext=(5, -10),  # Below the point
                     ha='left', 
                     fontsize=10, 
                     color='green')
    
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    ticker = "TSLA"
    cutoff_date = "2025-03-04"  # Predict March 5 using data up to March 4
    try:
        historical_data, predicted_price, actual_price, next_date = train_and_predict(ticker, cutoff_date)
        print(f"Predicted price for {next_date.strftime('%Y-%m-%d')}: ${predicted_price:.2f}")
        if actual_price:
            print(f"Actual price: ${actual_price:.2f}")
        print("\nLast 5 days of historical data:")
        print(historical_data.tail())
        plot_results(ticker, historical_data, predicted_price, actual_price, next_date)
    except Exception as e:
        print(f"Error occurred: {e}")