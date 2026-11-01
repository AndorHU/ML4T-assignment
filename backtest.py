import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppressing warnings
warnings.filterwarnings('ignore')

# 1. Loading data
print("Loading data...")
try:
    df = pd.read_parquet("ml_features_dataset.parquet")
except:
    df = pd.read_csv("market_data.csv", index_col="Date", parse_dates=True)

# Holdout filtering (2006-2021 training/validation period)
train_val_mask = df.index < '2022-01-01'
df_train = df[train_val_mask].copy()

print(f"Number of training data rows: {len(df_train)}")

# 2. Setting thresholds (80% Rebalance, 87.5% Full Exit, 70% Re-entry)
spy_reb_thresh = df_train['SPY_Vol_20'].quantile(0.80)
spy_exit_thresh = df_train['SPY_Vol_20'].quantile(0.875)
spy_reentry_thresh = df_train['SPY_Vol_20'].quantile(0.70)

gld_reb_thresh = df_train['GLD_Vol_20'].quantile(0.80)
gld_exit_thresh = df_train['GLD_Vol_20'].quantile(0.875)
gld_reentry_thresh = df_train['GLD_Vol_20'].quantile(0.70)

print(
    f"SPY Thresholds -> Rebalance (80%): {spy_reb_thresh:.4f} | Exit (87.5%): {spy_exit_thresh:.4f} | Re-entry (70%): {spy_reentry_thresh:.4f}")
print(
    f"GLD Thresholds -> Rebalance (80%): {gld_reb_thresh:.4f} | Exit (87.5%): {gld_exit_thresh:.4f} | Re-entry (70%): {gld_reentry_thresh:.4f}")

# Shifted volatilities to avoid lookahead bias
prev_spy_vol = df_train['SPY_Vol_20'].shift(1)
prev_gld_vol = df_train['GLD_Vol_20'].shift(1)

# 3. Daily simulation loop for Sleeve-based dynamic management
dates = df_train.index
spy_returns = df_train['SPY_Return'].values
gld_returns = df_train['GLD_Return'].values
shy_returns = df_train['SHY_Return'].values

spy_state = 0  # 0 = SHY, 1 = Risky (SPY)
gld_state = 0  # 0 = SHY, 1 = Risky (GLD)

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

    # --- SPY State Machine (Exit at 87.5%, Re-entry at 70%) ---
    old_spy_state = spy_state
    if spy_state == 1:
        if not pd.isna(s_vol) and s_vol > spy_exit_thresh:
            spy_state = 0  # Full exit into SHY
    else:
        if not pd.isna(s_vol) and s_vol < spy_reentry_thresh:
            spy_state = 1  # Re-entry into SPY

    # --- GLD Állapotgép (Exit 87.5%-nál, Re-entry 70%-nál) ---
    old_gld_state = gld_state
    if gld_state == 1:
        if not pd.isna(g_vol) and g_vol > gld_exit_thresh:
            gld_state = 0  # Full exit into SHY
    else:
        if not pd.isna(g_vol) and g_vol < gld_reentry_thresh:
            gld_state = 1  # Re-entry into GLD

    # Transaction cost due to state changes
    if spy_state != old_spy_state:
        spy_capital *= (1.0 - cost_per_trade)
    if gld_state != old_gld_state:
        gld_capital *= (1.0 - cost_per_trade)

    # --- Applying daily returns to sleeve capital ---
    s_ret = spy_returns[i] if spy_state == 1 else shy_returns[i]
    g_ret = gld_returns[i] if gld_state == 1 else shy_returns[i]

    spy_capital *= (1.0 + s_ret)
    gld_capital *= (1.0 + g_ret)

    # --- 80% Rebalance rule (Taking profits and adjusting to 50-50%) ---
    rebalance_triggered = False
    if not pd.isna(s_vol) and s_vol > spy_reb_thresh:
        rebalance_triggered = True
    if not pd.isna(g_vol) and g_vol > gld_reb_thresh:
        rebalance_triggered = True

    if rebalance_triggered:
        total_portfolio = spy_capital + gld_capital
        old_spy_cap = spy_capital

        # Rebalancing to 50-50%
        spy_capital = total_portfolio * 0.5
        gld_capital = total_portfolio * 0.5

        # Transaction cost for capital reallocation (based on turnover)
        moved_amount = abs(spy_capital - old_spy_cap)
        total_portfolio -= moved_amount * cost_per_trade

        # Cost-adjusted rebalancing
        spy_capital = total_portfolio * 0.5
        gld_capital = total_portfolio * 0.5

    # Saving total portfolio value
    total_val = spy_capital + gld_capital
    strategy_values.append(total_val)

df_train['Strategy_Cumulative'] = strategy_values
df_train['SPY_BuyAndHold'] = (1 + df_train['SPY_Return']).cumprod()
df_train['GLD_BuyAndHold'] = (1 + df_train['GLD_Return']).cumprod()

# 4. Final Results and Risk Metrics
final_strategy = df_train['Strategy_Cumulative'].iloc[-1]
final_spy = df_train['SPY_BuyAndHold'].iloc[-1]
final_gld = df_train['GLD_BuyAndHold'].iloc[-1]

rolling_max = df_train['Strategy_Cumulative'].cummax()
drawdown = df_train['Strategy_Cumulative'] / rolling_max - 1.0
max_drawdown = drawdown.min()

print(f"\n--- DYNAMIC REBALANCE BACKTEST RESULTS (2006 - 2021) ---")
print(f"Initial capital: $1.00")
print(f"Strategy final value (Net, 5 bps): ${final_strategy:.2f}")
print(f"S&P 500 (Buy & Hold) final value: ${final_spy:.2f}")
print(f"GLD (Buy & Hold) final value: ${final_gld:.2f}")
print(f"Strategy Max Drawdown: {max_drawdown * 100:.2f}%")

# 5. Plotting the chart
plt.figure(figsize=(12, 6))
plt.plot(df_train.index, df_train['Strategy_Cumulative'], label='Dynamic Rebalance Strategy (80/87.5/70)',
         color='darkviolet')
plt.plot(df_train.index, df_train['SPY_BuyAndHold'], label='S&P 500 (Buy & Hold)', color='gray', alpha=0.7)
plt.plot(df_train.index, df_train['GLD_BuyAndHold'], label='GLD (Buy & Hold)', color='goldenrod', alpha=0.8)

plt.title('Dynamic Rebalance & Exit Threshold (Rebal: 80%, Exit: 87.5%, Re-entry: 70%)')
plt.ylabel('Cumulative Return (Initial value = 1.0)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()