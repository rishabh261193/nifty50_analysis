import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

def fetch_data(ticker, start_date, end_date):
    """Fetch NIFTY50 data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    data["Adj Close"] = data["Close"]
    stock_info = yf.Ticker(ticker).info
    return data, stock_info

def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def feature_engineering(data):
    """Perform feature engineering on stock data."""
    data["RSI"] = calculate_rsi(data["Adj Close"], 14)
    return data.dropna()

def detect_breakout(data):
    """Detect breakout patterns in stock data."""
    recent_high = data['High'][-50:].max().item()
    last_close = data['Close'].iloc[-1].item()
    price_breakout = "Yes" if last_close > recent_high else "No"

    avg_volume = data['Volume'][-50:].mean().item()
    last_volume = data['Volume'].iloc[-1].item()
    volume_spike = "Yes" if last_volume > 1.5 * avg_volume else "No"

    rsi = data["RSI"].iloc[-1].item()
    rsi_breakout = "Yes" if rsi < 30 else "No"

    return {
        "Price Breakout": price_breakout,
        "Volume Spike": volume_spike,
        "RSI Breakout": rsi_breakout
    }

def predict_closing_price(data):
    """Predict the next day's closing price using linear regression."""
    recent_data = data[-30:]
    X = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data["Adj Close"].values
    model = LinearRegression()
    model.fit(X, y)
    next_day = len(recent_data)
    return model.predict([[next_day]])[0].item()

def evaluate_market(data):
    ma20 = data["Adj Close"].rolling(window=20).mean().iloc[-1].item()
    ma50 = data["Adj Close"].rolling(window=50).mean().iloc[-1].item()
    return "Bullish" if ma20 > ma50 else "Bearish"

# Streamlit UI Setup
st.set_page_config(page_title="NIFTY50 Analysis", layout="wide")
st.title("NIFTY50 Analysis")

# Input Dates
st.sidebar.header("Select Date Range")
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=60))
end_date = st.sidebar.date_input("End Date", value=date.today())

# Analyze Button
if st.sidebar.button("Analyze NIFTY50"):
    ticker = "^NSEI"
    try:
        # Fetch and process data
        data, stock_info = fetch_data(ticker, start_date, end_date)
        data = feature_engineering(data)
        breakout_data = detect_breakout(data)
        trend = evaluate_market(data)
        predicted_price = predict_closing_price(data)
        last_close = data["Close"].iloc[-1].item()

        # Display Results
        st.header("Analysis Results for NIFTY50")
        st.metric(label="Last Close", value=f"₹{last_close:.2f}")
        st.metric(label="Predicted Price", value=f"₹{predicted_price:.2f}")

        st.subheader("Breakout Indicators")
        for key, value in breakout_data.items():
            st.write(f"- **{key}:** {value}")

        st.subheader("Market Trend")
        st.write(f"The market trend is **{trend}**.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
