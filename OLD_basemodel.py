import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 1. Load the Parquet Dataset
df = pd.read_parquet("ml_features_dataset.parquet")

# 2. Enforce the Holdout Split
train_val_mask = df.index < '2022-01-01'
df_train = df[train_val_mask].copy()
df_holdout = df[~train_val_mask].copy()

print(f"Training/Validation Samples: {len(df_train)}")
print(f"Sealed Holdout Samples: {len(df_holdout)}")

# 3. Rung 1: Non-ML volatility baseline for SPY and GLD
# Rule: hold an asset only if its prior-day 20d volatility is below
# its own historical 80th percentile. If both are too volatile, hold SHY.
spy_vol_threshold = df_train['SPY_Vol_20'].quantile(0.80)
gld_vol_threshold = df_train['GLD_Vol_20'].quantile(0.80)

print(f"SPY Volatility Threshold (80th Percentile): {spy_vol_threshold:.4f}")
print(f"GLD Volatility Threshold (80th Percentile): {gld_vol_threshold:.4f}")

# Use prior-day volatility only: shift(1) prevents lookahead bias
spy_ok = df_train['SPY_Vol_20'].shift(1) <= spy_vol_threshold
gld_ok = df_train['GLD_Vol_20'].shift(1) <= gld_vol_threshold

# Base allocation: 50% SPY, 50% GLD when both are allowed.
# If only one is allowed, allocate 100% to that asset.
# If neither is allowed, allocate 100% to SHY.
df_train['Signal_SPY'] = np.where(spy_ok & gld_ok, 0.5, np.where(spy_ok, 1.0, 0.0))
df_train['Signal_GLD'] = np.where(spy_ok & gld_ok, 0.5, np.where(gld_ok, 1.0, 0.0))
df_train['Signal_SHY'] = 1.0 - df_train['Signal_SPY'] - df_train['Signal_GLD']

# Drop the first row, where the shifted volatility is NaN
df_train = df_train.dropna(subset=['SPY_Vol_20', 'GLD_Vol_20'])

# 4. Transaction costs: 5 bps per unit of turnover
cost_per_trade = 0.0005

df_train['SPY_Turnover'] = df_train['Signal_SPY'].diff().fillna(0).abs()
df_train['GLD_Turnover'] = df_train['Signal_GLD'].diff().fillna(0).abs()
df_train['SHY_Turnover'] = df_train['Signal_SHY'].diff().fillna(0).abs()

total_turnover = (
    df_train['SPY_Turnover']
    + df_train['GLD_Turnover']
    + df_train['SHY_Turnover']
)

# 5. Gross and net strategy returns
df_train['Gross_Strategy_Return'] = (
    df_train['Signal_SPY'] * df_train['SPY_Return']
    + df_train['Signal_GLD'] * df_train['GLD_Return']
    + df_train['Signal_SHY'] * df_train['SHY_Return']
)

df_train['Net_Strategy_Return'] = (
    df_train['Gross_Strategy_Return']
    - total_turnover * cost_per_trade
)

# 6. Cumulative returns
df_train['Strategy_Cumulative'] = (1 + df_train['Net_Strategy_Return']).cumprod()
df_train['SPY_BuyAndHold'] = (1 + df_train['SPY_Return']).cumprod()
df_train['GLD_BuyAndHold'] = (1 + df_train['GLD_Return']).cumprod()

final_strategy = df_train['Strategy_Cumulative'].iloc[-1]
final_spy = df_train['SPY_BuyAndHold'].iloc[-1]
final_gld = df_train['GLD_BuyAndHold'].iloc[-1]

print("\n--- RUNG 1 BASELINE RESULTS (TRAINING SET: 2006-2021) ---")
print(f"Net Strategy Final Value (after 5 bps cost): ${final_strategy:.2f}")
print(f"SPY Buy & Hold Final Value: ${final_spy:.2f}")
print(f"GLD Buy & Hold Final Value: ${final_gld:.2f}")

# 7. Plot
plt.figure(figsize=(12, 6))
plt.plot(df_train.index, df_train['Strategy_Cumulative'],
         label='Rung 1: SPY/GLD Volatility Timing (Net)', color='blue')
plt.plot(df_train.index, df_train['SPY_BuyAndHold'],
         label='SPY Buy & Hold', color='gray', alpha=0.7)
plt.plot(df_train.index, df_train['GLD_BuyAndHold'],
         label='GLD Buy & Hold', color='goldenrod', alpha=0.8)

plt.title('Rung 1 Baseline vs SPY and GLD (Training Period: 2006-2021)')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()