import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

# 1. Adatok betöltése és Walk-Forward Ridge predikciók
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

# 2. Szignálok és nettó hozamok (5 bps)
prev_spy_pred = df_backtest['Ridge_SPY_Pred'].shift(1)
prev_gld_pred = df_backtest['Ridge_GLD_Pred'].shift(1)

df_backtest['Signal_SPY'] = np.where(prev_spy_pred > 0, 0.5, 0.0)
df_backtest['Signal_GLD'] = np.where(prev_gld_pred > 0, 0.5, 0.0)
df_backtest['Signal_SHY'] = 1.0 - df_backtest['Signal_SPY'] - df_backtest['Signal_GLD']
df_backtest = df_backtest.dropna(subset=['Signal_SPY'])

cost_per_trade = 0.0005
spy_turnover = df_backtest['Signal_SPY'].diff().fillna(0).abs()
gld_turnover = df_backtest['Signal_GLD'].diff().fillna(0).abs()
shy_turnover = df_backtest['Signal_SHY'].diff().fillna(0).abs()
total_turnover = spy_turnover + gld_turnover + shy_turnover

gross_return = (
    df_backtest['Signal_SPY'] * df_backtest['SPY_Return'] +
    df_backtest['Signal_GLD'] * df_backtest['GLD_Return'] +
    df_backtest['Signal_SHY'] * df_backtest['SHY_Return']
)
returns = gross_return - (total_turnover * cost_per_trade)

# 3. Sharpe-ráta és Andrew Lo (2002) 95%-os Konfidenciaintervallum
N = len(returns)
annualization = np.sqrt(252)

mean_ret = returns.mean()
std_ret = returns.std(ddof=1)

daily_sr = mean_ret / std_ret
ann_sr = daily_sr * annualization

skew = stats.skew(returns)
kurt = stats.kurtosis(returns, fisher=True)

# Lo (2002) SE kiszámítása napi szintre
sr_se_daily = np.sqrt((1 + (0.5 * daily_sr**2) - (skew * daily_sr) + ((kurt / 4) * daily_sr**2)) / (N - 1))
sr_se_ann = sr_se_daily * annualization

ci_lower_ann = ann_sr - 1.96 * sr_se_ann
ci_upper_ann = ann_sr + 1.96 * sr_se_ann

# 4. Helyes Deflated Sharpe Ratio (DSR) kiszámítás évesített skálán
N_trials = 10                  # Kipróbált modellek/beállítások száma
var_trials_ann = 0.05          # Próbálkozások évesített Sharpe varianciája
gamma = 0.5772156649

# Évesített szintű elvárt max Sharpe véletlen próbálkozásból
exp_max_sr_ann = np.sqrt(var_trials_ann) * ((1 - gamma) * stats.norm.ppf(1 - 1/N_trials) + gamma * stats.norm.ppf(1 - 1/(N_trials * np.e)))

# DSR Z-score évesített mutatókból
dsr_z = (ann_sr - exp_max_sr_ann) / sr_se_ann
dsr_p_value = stats.norm.cdf(dsr_z)

print(f"--- SHARPE RATIO & STATISZTIKAI BIZONYÍTÉKOK (RIDGE) ---")
print(f"Elemzett napok száma (N): {N} nap (~{N/252:.1f} év)")
print(f"Évesített Sharpe-ráta: {ann_sr:.4f}")
print(f"Hozamok Ferdesége (Skewness): {skew:.4f}")
print(f"Hozamok Csúcsossága (Excess Kurtosis): {kurt:.4f}")
print(f"\nAndrew Lo (2002) 95%-os Sharpe CI: [{ci_lower_ann:.4f}, {ci_upper_ann:.4f}]")
print(f"\n--- DEFLATED SHARPE RATIO (LÓPEZ DE PRADO) ---")
print(f"Próbálkozások száma (Research Ledger N_trials): {N_trials}")
print(f"Elvárt Max Véletlen Sharpe (Annualized SR*): {exp_max_sr_ann:.4f}")
print(f"Deflated Sharpe Ratio (DSR valószínűség): {dsr_p_value * 100:.2f}%")