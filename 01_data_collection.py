import yfinance as yf
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
# 1. Data Downloading
tickers = ["SPY", "GLD", "SHY"]
start_date = "2005-01-01"
end_date = "2024-01-01"

print("Downloading historical data from Yahoo Finance...")
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

clean_data = raw_data.ffill()
clean_data = clean_data.dropna()

# 2. Calculating Returns and Excess Returns
daily_returns = clean_data.pct_change().dropna()

excess_returns = pd.DataFrame(index=daily_returns.index)
excess_returns['SPY'] = daily_returns['SPY'] - daily_returns['SHY']
excess_returns['GLD'] = daily_returns['GLD'] - daily_returns['SHY']

# 3. Setting the look ahead window, it's at 60days by default
window = 60
rolling_sharpe = pd.DataFrame(index=excess_returns.index)

for asset in ['SPY', 'GLD']:
    rolling_mean = excess_returns[asset].rolling(window=window).mean()
    rolling_std = excess_returns[asset].rolling(window=window).std()
    rolling_sharpe[asset] = (rolling_mean / rolling_std) * np.sqrt(252)

rolling_sharpe = rolling_sharpe.dropna()

print("\nEngineering Machine Learning Features and Labels...")
# Combine the base data first
prices = clean_data.add_suffix('_Price')
returns = daily_returns.add_suffix('_Return')
sharpes = rolling_sharpe.add_suffix('_Sharpe')
df = pd.concat([prices, returns, sharpes], axis=1)

# 4. Generate Features (X variables) for both risky assets
for asset in ['SPY', 'GLD']:
    # Volatility Clustering Features
    df[f'{asset}_Vol_20'] = df[f'{asset}_Return'].rolling(window=20).std() * np.sqrt(252)
    df[f'{asset}_Vol_60'] = df[f'{asset}_Return'].rolling(window=60).std() * np.sqrt(252)

    # EWMA Volatility
    df[f'{asset}_EWMA_Vol_20'] = df[f'{asset}_Return'].ewm(span=20).std() * np.sqrt(252)

    # Momentum (21 trading days = ~1 month)
    df[f'{asset}_Mom_21'] = df[f'{asset}_Price'].pct_change(periods=21)

    # Drawdown from recent 252-day (1 year) highs
    rolling_max = df[f'{asset}_Price'].rolling(window=252).max()
    df[f'{asset}_DD_252'] = (df[f'{asset}_Price'] / rolling_max) - 1

# Cross-Asset Feature
df['SPY_GLD_Corr_60'] = df['SPY_Return'].rolling(window=60).corr(df['GLD_Return'])

# 5. Generate the Target Label (y variable)
for asset in ['SPY', 'GLD']:
    forward_return_21 = (df[f'{asset}_Price'].shift(-21) / df[f'{asset}_Price']) - 1
    trailing_vol_21 = df[f'{asset}_Return'].rolling(window=21).std() * np.sqrt(252)
    df[f'{asset}_Target_Label'] = forward_return_21 / trailing_vol_21

# 6. Save Data to Parquet
df_final = df.dropna()
parquet_filename = "ml_features_dataset.parquet"
df_final.to_parquet(parquet_filename)

print(f"Success! Features and labels saved to: {parquet_filename}")