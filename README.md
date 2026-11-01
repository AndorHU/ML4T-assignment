# ML4T-assignment:



**\*\*Midterm Proposal Concept: Risk-Based Rotation Between SPY, GLD and SHY\*\***



\*\*1. Idea and mechanism.\*\* We build a machine learning strategy that moves money away from an asset when its forecast risk is too high. Market volatility clusters, meaning calm periods tend to follow calm periods and turbulent periods follow turbulent ones, so recent risk is informative about near-future risk. The strategy uses this to reduce exposure to a risky asset when risk rises and to hold the safer asset instead. The source of edge is risk compensation: the strategy tries to keep risk-adjusted returns high by avoiding stretches where the risk taken is not rewarded. We will stop and report a null result if the strategy cannot beat its non-ML baseline and a passive benchmark after costs on the validation period, or if its signal's confidence interval includes zero.



\*\*2. Universe and trading setup (setup.yaml, version v1).\*\* Three US-listed ETFs: SPY (S\&P 500), GLD (physical gold) and SHY (short-term US Treasuries). SHY is the safe "parking" asset and the risk-free proxy for the Sharpe ratio. Decisions are made at the market close, and trades are executed at the next open. The strategy is long-only and fully invested, with no leverage. The model produces a score for each risky asset (SPY, GLD), and the rotation rule shifts weight to SHY when the score is weak. There is an explicit no-trade state to avoid constant switching. The cost model includes a per-trade cost in 5 bp.



\*\*3. Data.\*\* Daily adjusted prices (total return) from NYSE Yfinance, covering about 2006 to 2026. This includes the 2008, 2020 and 2022 stress periods. The last 24 months are sealed as a holdout, run only once at the end. The earlier data is used for training and walk-forward validation. We document the download date, the meaning of the close (4 p.m. New York), the adjustment method and ticker stability. The data quality report covers missing days (flagged and excluded, never silently forward-filled), raw versus adjusted prices, OHLC violations and duplicates. Point-in-time check: values pass (only prices known at the close are used), identifiers pass, and the universe is a disclosed fail, because the three funds were chosen knowing they still exist today (a mild survivorship bias). The dataset is rebuilt by a script and stored as Parquet.



\*\*4. Label and features.\*\* The label is the forward return divided by trailing volatility over 21 trading days, measured from the next open to match the execution rule. Overlapping labels reduce the effective sample size, so all confidence intervals use the effective sample. Candidate features (about 10 to 30, each with a stated mechanism): rolling volatility (20 and 60 days), EWMA volatility, momentum, SPY-GLD correlation, and drawdown from recent highs. Each feature passes a triage: timing placebo, IC with a Newey-West confidence interval, shape check, turnover feasibility, and multiple-testing correction. Every candidate gets a PROCEED, REVISE or STOP decision.



\*\*5. Models and validation.\*\* We climb the baseline ladder: (1) a non-ML rule (move to SHY when trailing volatility is above a threshold), (2) Ridge regression, (3) LightGBM. Each model is judged by daily Spearman IC with a Newey-West 95% confidence interval and compared to the previous rung. Validation is walk-forward with purging (a gap equal to the label horizon, counted in trading days). We report SHAP for the final model and conformal prediction intervals.



\*\*6. Backtest, costs and risk.\*\* The strategy is compared against the non-ML baseline and passive benchmarks (SPY buy-and-hold and 60/40), all under identical rules and net of costs. We run a cost sweep from zero to four times the assumed cost and report the break-even cost, turnover and holding period. Risk reporting includes 95% VaR and CVaR, maximum drawdown (depth, duration, recovery), rolling beta to SPY, stress tests on 2020 and 2022 with higher costs, and one volatility-targeting overlay using only prior-day information, reported with and without the overlay. A drawdown-based kill-switch ladder is included.



\*\*7. Evaluation.\*\* Sharpe ratio is reported as an estimate, not one number: with a 95% confidence interval, separately for validation and holdout, plus the Deflated Sharpe Ratio using the trial count from our research ledger. IC is reported for signal quality. The result is classified using Chapter 20's patterns, and we name one next research step.



\*\*8. Monitoring plan.\*\* Rolling Sharpe, drawdown and IC over 30, 60 and 90 days, drift checks on the top features, and written retrain, pause and retire rules.



\*\*What could break this:\*\* survivorship in the hand-picked universe, regime changes, vendor revisions to adjusted prices, and only three assets limiting generality.




**Inputs from the professor:**
---

This is a solid topic, and your proposal follows the guideline structure well. The idea is workable: volatility timing has a real economic basis (see Moreira \& Muir, 2017), and the data are free and clean. 

Two big-picture suggestions before you start. 

First, frame the mechanism as volatility timing rather than risk compensation: your claim is that high-volatility periods are not rewarded with proportionally higher returns, so be explicit about that link. With only three assets, this is a timing problem, not a ranking problem, so evaluate your signals over time for each asset rather than across assets. 

Second, freeze your setup before you look at any results: how often you rebalance, how scores become weights, and when you switch to SHY. Also expect limited statistical power, since 20 years of monthly decisions is a small sample. 

A clean null result with a correct protocol is a perfectly good outcome for this project. Go ahead.


