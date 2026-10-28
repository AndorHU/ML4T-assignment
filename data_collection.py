import yfinance as yf
import pandas as pd
import numpy as np

#Data Downloading
tickers = ["SPY", "GLD", "SHY"]
start_date = "2005-01-01"
end_date = "2024-01-01"

print("Downloading historical data from Yahoo Finance...")
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

clean_data = raw_data.ffill()
clean_data = clean_data.dropna()

#Calculating returns and Returns
daily_returns = clean_data.pct_change().dropna()

excess_returns = pd.DataFrame(index=daily_returns.index)
excess_returns['SPY'] = daily_returns['SPY'] - daily_returns['SHY']
excess_returns['GLD'] = daily_returns['GLD'] - daily_returns['SHY']

#Setting the look ahead window, it's at 60days by default
window = 60
rolling_sharpe = pd.DataFrame(index=excess_returns.index)

for asset in ['SPY', 'GLD']:
    rolling_mean = excess_returns[asset].rolling(window=window).mean()
    rolling_std = excess_returns[asset].rolling(window=window).std()
    rolling_sharpe[asset] = (rolling_mean / rolling_std) * np.sqrt(252)

rolling_sharpe = rolling_sharpe.dropna()

# --- 4. LÉPÉS: Adatok kimentése CSV fájlba ---
print("\nAdatok mentése CSV-be...")
prices = clean_data.add_suffix('_Price')
returns = daily_returns.add_suffix('_Return')
sharpes = rolling_sharpe.add_suffix('_Sharpe')

final_dataset = pd.concat([prices, returns, sharpes], axis=1)
final_dataset = final_dataset.dropna()

csv_filename = "market_data.csv"
final_dataset.to_csv(csv_filename)

print(f"Siker! Az összes adat és számított metrika kimentve ide: {csv_filename}")