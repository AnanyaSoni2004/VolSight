"""
Evaluate a trained VolSight model and write analysis figures to figures/.

Usage
-----
    python -m scripts.evaluate

Produces
--------
    figures/training_curve.png        train/val loss
    figures/pred_vs_fair.png          LSTM & naive-BS predicted vs fair price
    figures/error_vs_moneyness.png    absolute pricing error across moneyness
    figures/error_vs_maturity.png     absolute pricing error across maturity
    figures/vol_forecast.png          LSTM-implied vs trailing vs forward vol
    figures/rmse_comparison.png       headline RMSE bar chart
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volsight import (
    SimConfig, build_dataset, load_model, predict, error_metrics,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")
FIG_DIR = os.path.join(HERE, "figures")

ACCENT = "#2563eb"
ACCENT2 = "#f59e0b"
GREY = "#64748b"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "volsight_lstm.pt")
    if not os.path.exists(model_path):
        raise SystemExit("No trained model found. Run `python -m scripts.train` first.")

    model, scaler = load_model(model_path)
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        meta = json.load(f)
    cfg_args = meta["config"]

    # rebuild a fresh evaluation set with the same market dynamics
    cfg = SimConfig(n_samples=6000, alpha=cfg_args["alpha"], beta=cfg_args["beta"],
                    omega=cfg_args["omega"], nu=cfg_args["nu"], seed=777)
    data = build_dataset(cfg)
    S = data["meta"][:, 0]
    K = data["meta"][:, 1]
    T = data["meta"][:, 2]
    fwd_vol = data["meta"][:, 4]
    trail_vol = data["meta"][:, 5]
    moneyness = S / K

    pred, vol = predict(model, scaler, data["seq"], data["static"], return_vol=True)
    fair = data["y_fair"]
    naive = data["y_bs_naive"]

    # 1. training curve
    hist = meta["history"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["train"], color=ACCENT, label="train")
    ax.plot(hist["val"], color=ACCENT2, label="val")
    ax.set_yscale("log")
    _style(ax, "Training curve", "epoch", "MSE loss")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/training_curve.png", dpi=130); plt.close(fig)

    # 2. predicted vs fair
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(fair.max(), pred.max(), naive.max()) * 1.02
    ax.plot([0, lim], [0, lim], color=GREY, lw=1, ls="--")
    ax.scatter(fair, naive, s=6, alpha=0.3, color=ACCENT2, label="naive BS")
    ax.scatter(fair, pred, s=6, alpha=0.3, color=ACCENT, label="LSTM")
    _style(ax, "Predicted vs fair price (/spot)", "fair price", "model price")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/pred_vs_fair.png", dpi=130); plt.close(fig)

    # 3. error vs moneyness (binned)
    def binned_abs_err(x, err, bins):
        idx = np.digitize(x, bins)
        centers, lstm_e, bs_e = [], [], []
        for b in range(1, len(bins)):
            m = idx == b
            if m.sum() < 5:
                continue
            centers.append(0.5 * (bins[b - 1] + bins[b]))
            lstm_e.append(np.mean(np.abs(pred[m] - fair[m])))
            bs_e.append(np.mean(np.abs(naive[m] - fair[m])))
        return np.array(centers), np.array(lstm_e), np.array(bs_e)

    c, le, be = binned_abs_err(moneyness, None, np.linspace(0.83, 1.25, 16))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(c, be, "-o", color=ACCENT2, ms=4, label="naive BS")
    ax.plot(c, le, "-o", color=ACCENT, ms=4, label="LSTM")
    _style(ax, "Mean abs. pricing error vs moneyness", "S / K", "MAE (/spot)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/error_vs_moneyness.png", dpi=130); plt.close(fig)

    # 4. error vs maturity
    c, le, be = binned_abs_err(T * 252, None, np.linspace(5, 60, 12))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(c, be, "-o", color=ACCENT2, ms=4, label="naive BS")
    ax.plot(c, le, "-o", color=ACCENT, ms=4, label="LSTM")
    _style(ax, "Mean abs. pricing error vs maturity", "days to maturity", "MAE (/spot)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/error_vs_maturity.png", dpi=130); plt.close(fig)

    # 5. volatility forecast quality
    if vol is not None:
        order = np.argsort(fwd_vol)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(fwd_vol, trail_vol, s=6, alpha=0.25, color=ACCENT2, label="trailing realized")
        ax.scatter(fwd_vol, vol, s=6, alpha=0.25, color=ACCENT, label="LSTM-implied")
        lim = fwd_vol.max() * 1.05
        ax.plot([0, lim], [0, lim], color=GREY, lw=1, ls="--")
        _style(ax, "Volatility forecast vs GARCH forward vol", "forward vol", "estimated vol")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(f"{FIG_DIR}/vol_forecast.png", dpi=130); plt.close(fig)

    # 6. headline RMSE comparison
    lstm_rmse = error_metrics(pred, fair)["RMSE"]
    bs_rmse = error_metrics(naive, fair)["RMSE"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["naive BS", "LSTM"], [bs_rmse, lstm_rmse], color=[ACCENT2, ACCENT])
    for b, v in zip(bars, [bs_rmse, lstm_rmse]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)
    _style(ax, "Pricing RMSE vs fair value", "", "RMSE (/spot)")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/rmse_comparison.png", dpi=130); plt.close(fig)

    print(f"Figures written to {FIG_DIR}/")
    print(f"  LSTM RMSE {lstm_rmse:.3e}  |  naive BS RMSE {bs_rmse:.3e}  "
          f"|  improvement {(1 - lstm_rmse / bs_rmse) * 100:.1f}%")


if __name__ == "__main__":
    main()
