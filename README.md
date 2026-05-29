#N— Neural Option Pricing

LSTM volatility forecasting fused with analytical Black–Scholes to price European
call options, benchmarked against a naive Black–Scholes baseline on synthetic
GARCH market data. Ships with a training pipeline, an evaluation/figures script,
and an interactive Streamlit dashboard.

## The idea in one line

> The LSTM forecasts forward-looking volatility from the recent return sequence;
> Black–Scholes prices that forecast exactly.

## Why a hybrid (and not a pure black box)

A first instinct is to train a network to map contract features straight to a
price. For **vanilla European options this does not work well**, and the reason is
fundamental rather than a tuning problem: Black–Scholes is already the *analytical
optimum* for these payoffs given a volatility. A black-box network can only
approximate that closed form, so it tends to land *behind* a well-calibrated BS,
not ahead of it. (The `--head price` option reproduces this for comparison.)

The genuinely hard, data-dependent part of pricing is choosing the right
**volatility**. That is where a sequence model earns its keep. VolSight therefore
uses a hybrid head:

```
return sequence ──▶ LSTM ──▶ hidden ─┐
                                     ├─▶ MLP ─▶ softplus ─▶ volatility
contract features ───────────────────┘                         │
                                              differentiable Black–Scholes layer
                                                               │
                                                          price / spot
```

The volatility head is supervised **directly** on the GARCH-optimal forward
volatility (a well-conditioned regression target), then a differentiable BS layer
converts it to a price. This avoids the vanishing-gradient trap of supervising
volatility *through* a price loss (vega → 0 in the option wings makes that
objective collapse to a constant).

## Results

Trained on 20,000 synthetic contracts (GARCH(1,1), Student-t innovations,
persistence 0.95), evaluated on a held-out set:

| model                    | MAE     | RMSE    | R²      |
|--------------------------|---------|---------|---------|
| **LSTM** (hybrid) vs fair | 0.00014 | 0.00052 | 0.99994 |
| naive BS (trailing vol)  | 0.00043 | 0.00155 | 0.99945 |

*(prices normalized by spot)*

- **~67% lower pricing RMSE** than naive Black–Scholes.
- **Volatility forecast MAE 0.0054 vs 0.0161** for an equal-weighted trailing
  estimator — about 3× better. The LSTM learns the volatility-clustering /
  mean-reversion dynamics instead of lagging them.

The headline takeaway matches the project's goal of weighing the strengths and
limitations of data-driven pricing: **neural models add real value on the
volatility-forecasting sub-problem, while the pricing map itself is best left to
the analytical formula.**

## What the data is

There is no external market feed; the underlying is simulated under a risk-neutral
GARCH(1,1) process with standardized Student-t innovations, so the series shows
realistic volatility clustering and fat-tailed shocks. For each example the
"fair" target price uses the **forward integrated volatility** over the option's
life (what a GARCH-aware desk would use), while the naive baseline uses the
**trailing realized volatility** of the observed window.

## Install

```bash
pip install -r requirements.txt
```

## Usage

Train (writes `models/volsight_lstm.pt` and `models/metrics.json`):

```bash
python -m scripts.train                 # hybrid vol head, sensible defaults
python -m scripts.train --head price     # black-box baseline for comparison
python -m scripts.train --samples 30000 --epochs 80
```

Generate evaluation figures into `figures/`:

```bash
python -m scripts.evaluate
```

Launch the interactive dashboard:

```bash
streamlit run app/dashboard.py
```

The dashboard lets you tune spot, strike, maturity, rate, and the **volatility
regime** (base vol, GARCH persistence, a recent-shock multiplier, scenario seed).
It shows the LSTM price and implied vol next to naive BS, full Greeks, sensitivity
curves (price vs spot / vol / maturity), a BS price surface, an LSTM−BS deviation
heatmap over moneyness × maturity, and a live Monte-Carlo RMSE benchmark.

## Library API

```python
from volsight import (
    bs_price, bs_greeks, implied_vol,        # analytical engine
    GarchParams, fit_garch,                  # volatility estimation
    forward_integrated_vol, trailing_realized_vol,
    SimConfig, build_dataset,                # synthetic data
    ModelConfig, OptionLSTM, train_model, predict, save_model, load_model,
    error_metrics, comparison_table, format_table,
)
```

## Project layout

```
volsight/
  volsight/
    black_scholes.py     analytical price, Greeks, implied vol
    garch_vol.py         GARCH(1,1) fit + forward-vol forecast
    data_generation.py   synthetic GARCH market + dataset builder
    lstm_model.py        LSTM with hybrid vol head & BS layer
    benchmark.py         error metrics and comparison tables
  scripts/
    train.py             training CLI
    evaluate.py          figure generation
  app/
    dashboard.py         Streamlit dashboard (helpers are import-testable)
  models/                saved weights + metrics (after training)
  figures/               evaluation plots (after evaluate)
```

## Notes & limitations

- Synthetic data is a controlled testbed, not real markets; the absolute error
  numbers reflect the simulated process, though the *relative* LSTM-vs-BS story is
  robust across seeds and GARCH settings.
- Scope is European calls; puts follow by parity and the engine already supports
  them. American/exotic payoffs would need a different pricing layer.
- CPU training of the default config takes a few minutes; no GPU required.
