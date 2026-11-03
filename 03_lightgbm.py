import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from scipy.stats import rankdata
import warnings

warnings.filterwarnings('ignore')


def compute_ic_newey_west(pred, target, max_lags=21):
    """
    Spearman IC és a hozzá tartozó Newey-West (HAC) 95%-os konfidenciaintervallum
    kiszámítása OLS regresszió segítségével rangsorolt adatokon.
    """
    df_temp = pd.DataFrame({'pred': pred, 'target': target}).dropna()

    if len(df_temp) < max_lags:
        return np.nan, np.nan, np.nan, np.nan

    # 1. Rangsorolás (Spearman alap)
    r_pred = rankdata(df_temp['pred'])
    r_target = rankdata(df_temp['target'])

    # 2. Standardizálás (Z-score)
    r_pred_std = (r_pred - np.mean(r_pred)) / np.std(r_pred)
    r_target_std = (r_target - np.mean(r_target)) / np.std(r_target)

    # 3. OLS regresszió konstanssal: Y_rank = alpha + beta * X_rank
    X = sm.add_constant(r_pred_std)
    ols_model = sm.OLS(r_target_std, X)

    # 4. Illesztés Newey-West (HAC) standard hibákkal (maxlags = label horizon)
    results = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})

    ic = results.params[1]  # Béta meredekség = Spearman IC
    se = results.bse[1]     # Newey-West Standard Error

    ci_lower = ic - 1.96 * se
    ci_upper = ic + 1.96 * se
    p_value = results.pvalues[1]

    return ic, ci_lower, ci_upper, p_value


# 1. Loading Data and Holdout Split (Pre-2022 training/validation data)
df = pd.read_parquet("ml_features_dataset.parquet")
train_val_mask = df.index < '2022-01-01'
df_train = df[train_val_mask].copy()

print(f"Full Training/Validation Dataset: {len(df_train)} rows")

# 2. Defining Features and Labels for Both Risky Assets
spy_features = ['SPY_Vol_20', 'SPY_Vol_60', 'SPY_EWMA_Vol_20', 'SPY_Mom_21', 'SPY_DD_252', 'SPY_GLD_Corr_60']
gld_features = ['GLD_Vol_20', 'GLD_Vol_60', 'GLD_EWMA_Vol_20', 'GLD_Mom_21', 'GLD_DD_252', 'SPY_GLD_Corr_60']

df_train = df_train.dropna(subset=spy_features + gld_features + ['SPY_Target_Label', 'GLD_Target_Label'])

# 3. Walk-Forward Validation and Collecting Out-of-Sample Predictions with LightGBM
print("\n--- RUNNING WALK-FORWARD VALIDATION (LightGBM) ---")
tscv = TimeSeriesSplit(n_splits=5, gap=21)

df_train['LGBM_SPY_Pred'] = np.nan
df_train['LGBM_GLD_Pred'] = np.nan

fold = 1
for train_index, test_index in tscv.split(df_train):
    train_df = df_train.iloc[train_index]
    test_df = df_train.iloc[test_index]

    # --- SPY LightGBM Model ---
    scaler_spy = StandardScaler()
    X_train_spy = scaler_spy.fit_transform(train_df[spy_features])
    X_test_spy = scaler_spy.transform(test_df[spy_features])

    lgb_spy = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, num_leaves=7, random_state=42,
                                verbose=-1)
    lgb_spy.fit(X_train_spy, train_df['SPY_Target_Label'])

    # --- GLD LightGBM Model ---
    scaler_gld = StandardScaler()
    X_train_gld = scaler_gld.fit_transform(train_df[gld_features])
    X_test_gld = scaler_gld.transform(test_df[gld_features])

    lgb_gld = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, num_leaves=7, random_state=42,
                                verbose=-1)
    lgb_gld.fit(X_train_gld, train_df['GLD_Target_Label'])

    # Saving predictions
    df_train.iloc[test_index, df_train.columns.get_loc('LGBM_SPY_Pred')] = lgb_spy.predict(X_test_spy)
    df_train.iloc[test_index, df_train.columns.get_loc('LGBM_GLD_Pred')] = lgb_gld.predict(X_test_gld)

    print(f"Fold {fold} processed ({len(test_df)} days).")
    fold += 1

df_backtest = df_train.dropna(subset=['LGBM_SPY_Pred', 'LGBM_GLD_Pred']).copy()

# OVERALL TIME-SERIES IC CALCULATION WITH NEWEY-WEST 95% CI (LightGBM)
ic_spy, spy_ci_lower, spy_ci_upper, spy_p = compute_ic_newey_west(
    df_backtest['LGBM_SPY_Pred'], df_backtest['SPY_Target_Label'], max_lags=21
)
ic_gld, gld_ci_lower, gld_ci_upper, gld_p = compute_ic_newey_west(
    df_backtest['LGBM_GLD_Pred'], df_backtest['GLD_Target_Label'], max_lags=21
)

print(f"\n--- TIME-SERIES IC WITH NEWEY-WEST 95% CONFIDENCE INTERVALS (LightGBM) ---")
print(f"SPY IC: {ic_spy:.4f} | 95% CI: [{spy_ci_lower:.4f}, {spy_ci_upper:.4f}] | p-value: {spy_p:.4f}")
print(f"GLD IC: {ic_gld:.4f} | 95% CI: [{gld_ci_lower:.4f}, {gld_ci_upper:.4f}] | p-value: {gld_p:.4f}")

# 4. Two 50% Sleeve Portfolio Logic
prev_spy_pred = df_backtest['LGBM_SPY_Pred'].shift(1)
prev_gld_pred = df_backtest['LGBM_GLD_Pred'].shift(1)

df_backtest['Signal_SPY'] = np.where(prev_spy_pred > 0, 0.5, 0.0)
df_backtest['Signal_GLD'] = np.where(prev_gld_pred > 0, 0.5, 0.0)
df_backtest['Signal_SHY'] = 1.0 - df_backtest['Signal_SPY'] - df_backtest['Signal_GLD']

df_backtest = df_backtest.dropna(subset=['Signal_SPY'])

# 5. Transaction Costs and Return Calculation (5 bps)
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

df_backtest['Net_Strategy_Return'] = gross_return - (total_turnover * cost_per_trade)

# 6. Cumulative Performance
df_backtest['Strategy_Cumulative'] = (1 + df_backtest['Net_Strategy_Return']).cumprod()
df_backtest['SPY_BuyAndHold'] = (1 + df_backtest['SPY_Return']).cumprod()
df_backtest['GLD_BuyAndHold'] = (1 + df_backtest['GLD_Return']).cumprod()

final_strategy = df_backtest['Strategy_Cumulative'].iloc[-1]
final_spy = df_backtest['SPY_BuyAndHold'].iloc[-1]
final_gld = df_backtest['GLD_BuyAndHold'].iloc[-1]

print(f"\n--- WALK-FORWARD RUNG 3 LIGHTGBM RESULTS ---")
print(f"Strategy Net Ending Value (5 bps): ${final_strategy:.2f}")
print(f"SPY Buy & Hold Ending Value: ${final_spy:.2f}")
print(f"GLD Buy & Hold Ending Value: ${final_gld:.2f}")

# 7. Visualization
plt.figure(figsize=(12, 6))
plt.plot(df_backtest.index, df_backtest['Strategy_Cumulative'], label='Walk-Forward LightGBM v2 (Net)',
         color='forestgreen')
plt.plot(df_backtest.index, df_backtest['SPY_BuyAndHold'], label='SPY Buy & Hold', color='gray', alpha=0.7)
plt.plot(df_backtest.index, df_backtest['GLD_BuyAndHold'], label='GLD Buy & Hold', color='goldenrod', alpha=0.8)

plt.title('Walk-Forward Validated LightGBM (SPY/GLD/SHY Sleeves)')
plt.ylabel('Cumulative Return (Initial Value = 1.0)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()