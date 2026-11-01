# How to Use: Financial Backtesting & Machine Learning Pipeline

This guide outlines the steps to set up your environment and execute the financial backtesting, LightGBM modeling, and market stress-testing script.

## 1. Environment Setup & Dependencies

The project requires **Python 3.14 or newer**. Dependencies are defined in the provided `pyproject.toml` file.

The exact required packages specified in the configuration are:
*   `matplotlib>=3.11.2`
*   `scikit-learn>=1.9.1`
*   `shap>=0.54.1`
*   `yfinance>=1.7.0`

*(Note: Since your script utilizes LightGBM, pandas, and numpy, ensure those are also available in your Python environment if you haven't added them to the `.toml` yet).*

**Installation:**
Navigate to the directory containing `pyproject.toml` and run the following command in your terminal:
```bash
pip install pyproject.toml
```

## 2. Pipeline Execution Steps

Execute your Python script. The script is structured to process the data and evaluate the strategy in the following chronological order:

### Step 1: Data Acquisition & Normalization
*   Downloads historical data for **SPY**, **GLD**, and **SHY** via `yfinance`.
*   Normalizes the initial capital to the start of the holdout period (starting from $1.00).

### Step 2: State Machine & Walk-Forward Validation
*   Executes the Walk-Forward validation to generate trading signals without lookahead bias.
*   Runs the State Machine logic for signal execution, factoring in 5 bps transaction costs.
*   Applies the **80% Rebalance** rule (Taking profits and rebalancing to 50-50) when triggered.

### Step 3: Results & Stress Test Metrics (2022 - 2024 Holdout)
*   Calculates the final strategy value compared to Buy & Hold benchmarks (S&P 500 and GLD).
*   Computes daily risk metrics, including **Value at Risk (VaR)** and **Conditional VaR (CVaR)** at 95% confidence.
*   Calculates the **Holdout Max Drawdown** for the 2022 stress test period.
*   Prints the locked holdout results and the last 10 rows of data to the console.

### Step 4: Visualizations
*   Generates a cumulative return chart comparing the strategy to benchmarks using Matplotlib (`Locked Holdout & 2022 Market Stress Test Performance`).

### Step 5: SHAP Analysis
*   Trains a specific LightGBM model instance solely for feature explainability.
*   Calculates and plots SHAP values to interpret the model's decision-making process.