import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

# 1. Adatok betöltése és Walk-Forward Ridge predikciók előállítása
df = pd.read_parquet("ml_features_dataset.parquet")
df_train = df[df.index < '2022-01-01'].copy()

spy_features = ['SPY_Vol_20', 'SPY_Vol_60', 'SPY_EWMA_Vol_20', 'SPY_Mom_21', 'SPY_DD_252', 'SPY_GLD_Corr_60']
gld_features = ['GLD_Vol_20', 'GLD_Vol_60', 'GLD_EWMA_Vol_20', 'GLD_Mom_21', 'GLD_DD_252', 'SPY_GLD_Corr_60']
df_train = df_train.dropna(subset=spy_features + gld_features + ['SPY_Target_Label', 'GLD_Target_Label'])

tscv = TimeSeriesSplit(n_splits=5, gap=21)
df_train['Ridge_SPY_Pred'] = np.nan
df_train['Ridge_GLD_Pred'] = np.nan

for train_index, test_index in tscv.split(df_train):
    train_df = df_train.iloc[train_index]
    test_df = df_train.iloc[test_index]

    scaler_spy, scaler_gld = StandardScaler(), StandardScaler()
    X_tr_spy = scaler_spy.fit_transform(train_df[spy_features])
    X_te_spy = scaler_spy.transform(test_df[spy_features])
    X_tr_gld = scaler_gld.fit_transform(train_df[gld_features])
    X_te_gld = scaler_gld.transform(test_df[gld_features])

    r_spy = Ridge(alpha=1.0).fit(X_tr_spy, train_df['SPY_Target_Label'])
    r_gld = Ridge(alpha=1.0).fit(X_tr_gld, train_df['GLD_Target_Label'])

    df_train.iloc[test_index, df_train.columns.get_loc('Ridge_SPY_Pred')] = r_spy.predict(X_te_spy)
    df_train.iloc[test_index, df_train.columns.get_loc('Ridge_GLD_Pred')] = r_gld.predict(X_te_gld)

df_backtest = df_train.dropna(subset=['Ridge_SPY_Pred', 'Ridge_GLD_Pred']).copy()

# 2. Szignálok és Turnover számítás
prev_spy_pred = df_backtest['Ridge_SPY_Pred'].shift(1)
prev_gld_pred = df_backtest['Ridge_GLD_Pred'].shift(1)

df_backtest['Signal_SPY'] = np.where(prev_spy_pred > 0, 0.5, 0.0)
df_backtest['Signal_GLD'] = np.where(prev_gld_pred > 0, 0.5, 0.0)
df_backtest['Signal_SHY'] = 1.0 - df_backtest['Signal_SPY'] - df_backtest['Signal_GLD']
df_backtest = df_backtest.dropna(subset=['Signal_SPY'])

spy_turnover = df_backtest['Signal_SPY'].diff().fillna(0).abs()
gld_turnover = df_backtest['Signal_GLD'].diff().fillna(0).abs()
shy_turnover = df_backtest['Signal_SHY'].diff().fillna(0).abs()
total_turnover = spy_turnover + gld_turnover + shy_turnover

# --- TURNOVER ÉS TARTÁSI IDŐ METRIKÁK ---
num_years = len(df_backtest) / 252.0
annualized_turnover = total_turnover.sum() / num_years
trades_count = (spy_turnover > 0).sum() + (gld_turnover > 0).sum()
avg_holding_period_days = (len(df_backtest) * 2) / trades_count if trades_count > 0 else np.inf

print(f"\n--- PORTFÓLIÓ DYNAMIKAI METRIKÁK ---")
print(f"Évesített Turnover (Portfólió forgás): {annualized_turnover * 100:.2f}% / év")
print(f"Összes Tranzakció Száma (Walk-Forward): {trades_count} alkalom")
print(f"Átlagos Pozíciótartási Idő (Holding Period): {avg_holding_period_days:.1f} kereskedési nap")

# 3. Cost Sensitivity Sweep (0 bps - 20 bps)
cost_grid_bps = np.linspace(0, 20, 21)  # 0, 1, 2, ..., 20 bps
gross_return = (
        df_backtest['Signal_SPY'] * df_backtest['SPY_Return'] +
        df_backtest['Signal_GLD'] * df_backtest['GLD_Return'] +
        df_backtest['Signal_SHY'] * df_backtest['SHY_Return']
)

final_values = []
sharpe_ratios = []

for bps in cost_grid_bps:
    cost = bps / 10000.0  # bps átváltása decimális formátumba (5 bps = 0.0005)
    net_return = gross_return - (total_turnover * cost)
    cum_return = (1 + net_return).cumprod()

    final_val = cum_return.iloc[-1]
    final_values.append(final_val)

    # Évesített Sharpe-ráta (kockázatmentes hozam = 0 az egyszerűség kedvéért)
    ann_ret = net_return.mean() * 252
    ann_vol = net_return.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    sharpe_ratios.append(sharpe)

gld_final = (1 + df_backtest['GLD_Return']).cumprod().iloc[-1]

# Break-even pont meghatározása (ahol a hozam leesik a GLD alá)
break_even_bps = np.nan
for i in range(len(cost_grid_bps)):
    if final_values[i] < gld_final:
        break_even_bps = cost_grid_bps[i]
        break

print(f"\n--- COST SWEEP EREDMÉNYEK ---")
print(f"Gross Ending Value (0 bps): ${final_values[0]:.2f} (Sharpe: {sharpe_ratios[0]:.2f})")
print(f"Baseline Ending Value (5 bps): ${final_values[5]:.2f} (Sharpe: {sharpe_ratios[5]:.2f})")
print(f"4x Cost Ending Value (20 bps): ${final_values[-1]:.2f} (Sharpe: {sharpe_ratios[-1]:.2f})")
if not np.isnan(break_even_bps):
    print(f"Break-even Cost Point (vs GLD Benchmark): ~{break_even_bps:.1f} bps")
else:
    print(f"Break-even Cost Point: > 20 bps (A stratégia még 20 bps-nél is veri a GLD benchmarkot)")

# 4. Vizualizáció
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Transaction Cost (basis points, bps)')
ax1.set_ylabel('Strategy Ending Value ($)', color=color)
ax1.plot(cost_grid_bps, final_values, color=color, marker='o', linewidth=2, label='Strategy Ending Value')
ax1.axhline(gld_final, color='goldenrod', linestyle='--', label=f'GLD Buy & Hold (${gld_final:.2f})')
ax1.axvline(5, color='red', linestyle=':', label='Baseline Fee (5 bps)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Sharpe Ratio', color=color)
ax2.plot(cost_grid_bps, sharpe_ratios, color=color, marker='s', linestyle='--', alpha=0.7, label='Sharpe Ratio')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Transaction Cost Sensitivity Analysis & Break-even Sweep (Ridge Model)')
fig.tight_layout()
plt.show()