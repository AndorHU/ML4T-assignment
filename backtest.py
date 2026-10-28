import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Figyelmeztetések elnémítása
warnings.filterwarnings('ignore')

# 1. Adatok betöltése
print("CSV adatok betöltése...")
df = pd.read_csv("market_data.csv", index_col="Date", parse_dates=True)

# 2. KERESKEDÉSI SZABÁLYOK (Signal Generation)
print("Kereskedési jelek (signals) kiszámítása...")

# Alapértelmezett pozíciók beállítása (0%)
df['SPY_Position'] = 0.0
df['GLD_Position'] = 0.0
df['SHY_Position'] = 0.0

# Eltolt Sharpe-ráták a múltbeli adatokhoz (jövőbelátás elkerülése)
prev_spy_sharpe = df['SPY_Sharpe'].shift(1)
prev_gld_sharpe = df['GLD_Sharpe'].shift(1)

# Jelek (signals) generálása
spy_signal = (prev_spy_sharpe > prev_gld_sharpe) & (prev_spy_sharpe > 0)
gld_signal = (prev_gld_sharpe > prev_spy_sharpe) & (prev_gld_sharpe > 0)
shy_signal = (prev_spy_sharpe <= 0) & (prev_gld_sharpe <= 0)

# Pozíciók felvétele (100% allokáció az adott eszközbe)
df.loc[spy_signal, 'SPY_Position'] = 1.0
df.loc[gld_signal, 'GLD_Position'] = 1.0
df.loc[shy_signal, 'SHY_Position'] = 1.0

# Kezdeti üres sorok eldobása (a shift miatt)
df = df.dropna()

# 3. STRATÉGIA HOZAMÁNAK KISZÁMÍTÁSA
# Napi hozam = (melyik eszközben vagyunk) * (annak az eszköznek a napi hozama)
df['Strategy_Return'] = (
    (df['SPY_Position'] * df['SPY_Return']) +
    (df['GLD_Position'] * df['GLD_Return']) +
    (df['SHY_Position'] * df['SHY_Return'])
)

# Kumulált (összegyűlt) hozamok kiszámítása (1 dolláros kezdeti tőkével)
df['Strategy_Cumulative'] = (1 + df['Strategy_Return']).cumprod()
df['SPY_BuyAndHold'] = (1 + df['SPY_Return']).cumprod()

# 4. VÉGEREDMÉNYEK ÉS METRIKÁK
final_strategy_value = df['Strategy_Cumulative'].iloc[-1]
final_spy_value = df['SPY_BuyAndHold'].iloc[-1]

# Maximális visszaesés (Maximum Drawdown) kiszámítása
rolling_max = df['Strategy_Cumulative'].cummax()
drawdown = df['Strategy_Cumulative'] / rolling_max - 1.0
max_drawdown = drawdown.min()

print(f"\n--- BACKTEST EREDMÉNYEK (2005 - 2023) ---")
print(f"Kezdeti tőke: $1.00")
print(f"Stratégia végső értéke: ${final_strategy_value:.2f}")
print(f"S&P 500 (Buy & Hold) végső értéke: ${final_spy_value:.2f}")
print(f"Stratégia maximális visszaesése (Max Drawdown): {max_drawdown * 100:.2f}%")

# 5. GRAFIKON RAJZOLÁSA A BEADANDÓHOZ
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Strategy_Cumulative'], label='Kvantitatív Stratégia (SPY/GLD/SHY)', color='blue')
plt.plot(df.index, df['SPY_BuyAndHold'], label='S&P 500 (Buy & Hold)', color='gray', alpha=0.7)

plt.title('Kvantitatív Kereskedési Modell Teljesítménye')
plt.ylabel('Kumulált Hozam (Kezdőérték = 1.0)')
plt.xlabel('Dátum')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Grafikon megjelenítése
plt.show()