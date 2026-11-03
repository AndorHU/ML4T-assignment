import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
import warnings

warnings.filterwarnings('ignore')

# 1. Loading data
print("Loading the data from the Parquet file...")
df = pd.read_parquet("ml_features_dataset.parquet")
train_val_mask = df.index < '2022-01-01'
df_train = df[train_val_mask].copy()

# Defining benchmarks and cumulative columns for console output
df_train['SPY_BuyAndHold'] = (1 + df_train['SPY_Return']).cumprod()
df_train['GLD_BuyAndHold'] = (1 + df_train['GLD_Return']).cumprod()
df_train['Strategy_Cumulative'] = df_train['SPY_BuyAndHold'] # Can also be replaced with the strategy return

# 2. Calculating VaR and CVaR
returns = df_train['SPY_Return'].values
confidence_level = 0.95
var_95 = np.percentile(returns, (1 - confidence_level) * 100)
cvar_95 = returns[returns <= var_95].mean()

print(f"\n--- RISK METRICS (95% Confidence, Daily) ---")
print(f"95% Value at Risk (VaR): {var_95 * 100:.2f}%")
print(f"95% Conditional Value at Risk (CVaR): {cvar_95 * 100:.2f}%")

# 3. Printing data to Console
print("\n--- LAST 10 ROWS OF DATA ---")
print(df_train[['Strategy_Cumulative', 'SPY_BuyAndHold', 'GLD_BuyAndHold']].tail(10))

# 4. Training LightGBM Model for SHAP Analysis
features = ['SPY_Vol_20', 'SPY_Vol_60', 'SPY_EWMA_Vol_20', 'SPY_Mom_21', 'SPY_DD_252', 'SPY_GLD_Corr_60']
target = 'SPY_Target_Label'

df_train_ml = df_train.dropna(subset=features + [target])
X = df_train_ml[features]
y = df_train_ml[target]

print("\nTraining LightGBM model for SHAP analysis...")
model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, num_leaves=7, random_state=42, verbose=-1)
model.fit(X, y)

# 5. Calculating and Plotting SHAP Values
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("LightGBM Feature Importance (SHAP Summary Plot)")
plt.tight_layout()
plt.show()