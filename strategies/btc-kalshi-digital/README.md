# BTC Kalshi digital-option backtest

Kalshi "BTC above $X at 5pm ET" contracts are digital options. This backtest compares the contract price to a model probability
(normal or filtered-historical distribution; rv / EWMA / GARCH(1,1) vol) and trades the gap, with an optional delta hedge.

## Run
```
pip install -r ../../requirements.txt
python btc_pm_data_pull.py              # pulls Kalshi + spot data into output/btc_pm/ (not committed; includes a large cache)
python btc_digital_model_backtest.py    # writes results to output/btc_pm_results/
python btc_digital_model_backtest.py --h 6 --edge 0.04 --model garch --dist fhs --hedge-bps 5
```

## Contents
- `btc_pm_data_pull.py` - data pull (Kalshi hourly prices, markets, BTC spot)
- `btc_digital_model_backtest.py` - model, calibration test (Brier), trading rule, hedge, bootstrap CIs
- `output/btc_pm_results/` - chart, trades, observations, reliability table from the last run
- `output/btc_bt_stdout.txt` - console output from the last run
