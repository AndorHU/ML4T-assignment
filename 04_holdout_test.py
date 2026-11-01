import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 1. Adatok betöltése és Holdout szétválasztás
print("Adatok betöltése a Parquet fájlból...")
df = pd.read_parquet("ml_features_dataset.parquet")

# Tréning adatok a küszöbök illesztéséhez (2006-2021)
train_val_mask = df.index < '2022-01-01'
df_train = df[train_val_mask].copy()

# Lezárt Holdout adatok (2022-01-01 - 2024-01-01)
holdout_mask = (df.index >= '2022-01-01') & (df.index < '2024-01-01')
df_holdout = df[holdout_mask].copy()

print(f"Tréning minták száma (küszöbszámításhoz): {len(df_train)}")
print(f"Lezárt Holdout minták száma (2022-2024 teszt): {len(df_holdout)}")

# 2. Küszöbök rögzítése a TRÉNING adatok alapján (Nincs adatszivárgás!)
spy_reb_thresh = df_train['SPY_Vol_20'].quantile(0.80)
spy_exit_thresh = df_train['SPY_Vol_20'].quantile(0.875)
spy_reentry_thresh = df_train['SPY_Vol_20'].quantile(0.70)

gld_reb_thresh = df_train['GLD_Vol_20'].quantile(0.80)
gld_exit_thresh = df_train['GLD_Vol_20'].quantile(0.875)
gld_reentry_thresh = df_train['GLD_Vol_20'].quantile(0.70)

print(f"\n--- FIXED THRESHOLDS FROM TRAINING ---")
print(
    f"SPY -> Rebalance (80%): {spy_reb_thresh:.4f} | Exit (87.5%): {spy_exit_thresh:.4f} | Re-entry (70%): {spy_reentry_thresh:.4f}")
print(
    f"GLD -> Rebalance (80%): {gld_reb_thresh:.4f} | Exit (87.5%): {gld_exit_thresh:.4f} | Re-entry (70%): {gld_reentry_thresh:.4f}")

# Pre-shifted volatilities on the holdout set to avoid lookahead bias
prev_spy_vol = df_holdout['SPY_Vol_20'].shift(1)
prev_gld_vol = df_holdout['GLD_Vol_20'].shift(1)

# 3. Simulation Loop on the Locked Holdout Set (2022-2024)
dates = df_holdout.index
spy_returns = df_holdout['SPY_Return'].values
gld_returns = df_holdout['GLD_Return'].values
shy_returns = df_holdout['SHY_Return'].values

spy_state = 0  # 0 = SHY, 1 = Risky (SPY)
gld_state = 0  # 0 = SHY, 1 = Risky (GLD)

# Initial capital at the start of the holdout set (1.0 unit)
spy_capital = 0.5
gld_capital = 0.5

cost_per_trade = 0.0005
strategy_values = []

for i in range(len(dates)):
    if i == 0:
        strategy_values.append(1.0)
        continue

    s_vol = prev_spy_vol.iloc[i]
    g_vol = prev_gld_vol.iloc[i]

    # --- SPY State Machine ---
    old_spy_state = spy_state
    if spy_state == 1:
        if not pd.isna(s_vol) and s_vol > spy_exit_thresh:
            spy_state = 0  # Full exit into SHY
    else:
        if not pd.isna(s_vol) and s_vol < spy_reentry_thresh:
            spy_state = 1  # Re-entry into SPY

    # --- GLD State Machine ---
    old_gld_state = gld_state
    if gld_state == 1:
        if not pd.isna(g_vol) and g_vol > gld_exit_thresh:
            gld_state = 0  # Full exit into SHY
    else:
        if not pd.isna(g_vol) and g_vol < gld_reentry_thresh:
            gld_state = 1  # Re-entry into GLD

    # Transaction costs for state transitions
    if spy_state != old_spy_state:
        spy_capital *= (1.0 - cost_per_trade)
    if gld_state != old_gld_state:
        gld_capital *= (1.0 - cost_per_trade)

    # Applying daily returns
    s_ret = spy_returns[i] if spy_state == 1 else shy_returns[i]
    g_ret = gld_returns[i] if gld_state == 1 else shy_returns[i]

    spy_capital *= (1.0 + s_ret)
    gld_capital *= (1.0 + g_ret)

    # --- 80% Rebalance (Taking profits and rebalancing to 50-50) ---
    rebalance_triggered = False
    if not pd.isna(s_vol) and s_vol > spy_reb_thresh:
        rebalance_triggered = True
    if not pd.isna(g_vol) and g_vol > gld_reb_thresh:
        rebalance_triggered = True

    if rebalance_triggered:
        total_portfolio = spy_capital + gld_capital
        old_spy_cap = spy_capital

        spy_capital = total_portfolio * 0.5
        gld_capital = total_portfolio * 0.5

        moved_amount = abs(spy_capital - old_spy_cap)
        total_portfolio -= moved_amount * cost_per_trade

        spy_capital = total_portfolio * 0.5
        gld_capital = total_portfolio * 0.5

    strategy_values.append(spy_capital + gld_capital)

df_holdout['Strategy_Cumulative'] = strategy_values
df_holdout['SPY_BuyAndHold'] = (1 + df_holdout['SPY_Return']).cumprod()
df_holdout['GLD_BuyAndHold'] = (1 + df_holdout['GLD_Return']).cumprod()

# Normalization to the start of the holdout (starting from $1.00)
df_holdout['Strategy_Cumulative'] /= df_holdout['Strategy_Cumulative'].iloc[0]
df_holdout['SPY_BuyAndHold'] /= df_holdout['SPY_BuyAndHold'].iloc[0]
df_holdout['GLD_BuyAndHold'] /= df_holdout['GLD_BuyAndHold'].iloc[0]

# 4. Results and Stress Test Metrics Calculation (Year 2022)
final_strategy = df_holdout['Strategy_Cumulative'].iloc[-1]
final_spy = df_holdout['SPY_BuyAndHold'].iloc[-1]
final_gld = df_holdout['GLD_BuyAndHold'].iloc[-1]

rolling_max = df_holdout['Strategy_Cumulative'].cummax()
drawdown = df_holdout['Strategy_Cumulative'] / rolling_max - 1.0
max_drawdown = drawdown.min()

print(f"\n--- LOCKED HOLDOUT RESULTS (2022 - 2024) ---")
print(f"Initial capital at the start of holdout: $1.00")
print(f"Strategy final value: ${final_strategy:.2f}")
print(f"S&P 500 (Buy & Hold) final value: ${final_spy:.2f}")
print(f"GLD (Buy & Hold) final value: ${final_gld:.2f}")
print(f"Holdout Max Drawdown (Stress Test): {max_drawdown * 100:.2f}%")

# 5. Visualization
plt.figure(figsize=(12, 6))
plt.plot(df_holdout.index, df_holdout['Strategy_Cumulative'], label='Locked Holdout Strategy (2022-2024)',
         color='crimson')
plt.plot(df_holdout.index, df_holdout['SPY_BuyAndHold'], label='S&P 500 (Buy & Hold)', color='gray', alpha=0.7)
plt.plot(df_holdout.index, df_holdout['GLD_BuyAndHold'], label='GLD (Buy & Hold)', color='goldenrod', alpha=0.8)

plt.title('Locked Holdout & 2022 Market Stress Test Performance')
plt.ylabel('Cumulative Return (Initial Value = 1.0)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()