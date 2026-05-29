"""
Benchmarking utilities: error metrics and comparison tables for evaluating the
neural model against Black-Scholes and the GARCH-optimal fair price.
"""
from __future__ import annotations

import numpy as np


def error_metrics(pred, target):
    """Return MAE, RMSE, MAPE and R^2 for two aligned arrays."""
    pred = np.asarray(pred, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(target) < 1e-8, np.nan, target)
    mape = float(np.nanmean(np.abs(err / denom)) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}


def comparison_table(lstm_pred, bs_naive, fair):
    """Compare LSTM and naive-BS predictions against the fair (target) price.

    Returns a dict of {model_name: metrics_dict}. Prices may be in price/spot
    units; metrics are scale-consistent.
    """
    return {
        "LSTM_vs_fair": error_metrics(lstm_pred, fair),
        "NaiveBS_vs_fair": error_metrics(bs_naive, fair),
    }


def format_table(table: dict) -> str:
    """Render a comparison dict as a fixed-width text table."""
    cols = ["MAE", "RMSE", "MAPE_%", "R2"]
    header = f"{'model':<18}" + "".join(f"{c:>12}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, m in table.items():
        row = f"{name:<18}" + "".join(f"{m[c]:>12.5f}" for c in cols)
        lines.append(row)
    return "\n".join(lines)
