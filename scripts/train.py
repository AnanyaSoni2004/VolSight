"""
Train the VolSight LSTM option-pricing model.

Usage
-----
    python -m scripts.train                       # sensible defaults
    python -m scripts.train --head price          # black-box price head
    python -m scripts.train --samples 30000 --epochs 80

Outputs
-------
    models/volsight_lstm.pt     trained weights + scaler + config
    models/metrics.json          held-out benchmark vs naive Black-Scholes
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from volsight import (
    SimConfig, build_dataset, ModelConfig, train_model, predict,
    save_model, comparison_table, error_metrics,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(HERE, "models")


def split(data, n_test, seed=99):
    n = len(data["y_fair"])
    idx = np.random.default_rng(seed).permutation(n)
    te, tr = idx[:n_test], idx[n_test:]
    train = {k: (v[tr] if isinstance(v, np.ndarray) and v.shape[0] == n else v)
             for k, v in data.items()}
    return train, te


def main():
    ap = argparse.ArgumentParser(description="Train VolSight LSTM")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head", choices=["vol", "price"], default="vol")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.11)
    ap.add_argument("--beta", type=float, default=0.84)
    ap.add_argument("--omega", type=float, default=1.2e-6)
    ap.add_argument("--nu", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--test", type=int, default=3000)
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(">> Generating synthetic GARCH market dataset ...")
    cfg = SimConfig(n_samples=args.samples, alpha=args.alpha, beta=args.beta,
                    omega=args.omega, nu=args.nu, seed=args.seed)
    data = build_dataset(cfg)
    print(f"   sequences {data['seq'].shape} | persistence {cfg.params.persistence:.3f} | "
          f"uncond vol {cfg.params.uncond_vol_annual:.3f}")

    train_data, te = split(data, args.test)

    print(f">> Training LSTM (head='{args.head}') ...")
    t0 = time.time()
    model_cfg = ModelConfig(hidden_size=args.hidden, num_layers=args.layers, head=args.head)
    model, scaler, history = train_model(
        train_data, model_cfg, epochs=args.epochs, batch_size=args.batch, lr=args.lr,
    )
    print(f"   done in {time.time()-t0:.1f}s")

    pred, vol = predict(model, scaler, data["seq"][te], data["static"][te], return_vol=True)
    table = comparison_table(pred, data["y_bs_naive"][te], data["y_fair"][te])
    bs_rmse = error_metrics(data["y_bs_naive"][te], data["y_fair"][te])["RMSE"]
    lstm_rmse = error_metrics(pred, data["y_fair"][te])["RMSE"]
    improvement = (1 - lstm_rmse / bs_rmse) * 100

    metrics = {
        "head": args.head,
        "n_train": int(args.samples - args.test),
        "n_test": int(args.test),
        "persistence": float(cfg.params.persistence),
        "table": table,
        "lstm_rmse_improvement_pct_vs_naive_bs": float(improvement),
    }
    if vol is not None:
        fwd, trail = data["meta"][te, 4], data["meta"][te, 5]
        metrics["vol_mae_lstm"] = float(np.mean(np.abs(vol - fwd)))
        metrics["vol_mae_trailing"] = float(np.mean(np.abs(trail - fwd)))

    save_model(os.path.join(MODELS_DIR, "volsight_lstm.pt"), model, scaler)
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "history": history, "config": vars(args)}, f, indent=2)

    from volsight import format_table
    print("\n" + format_table(table))
    print(f"\nLSTM RMSE improvement over naive BS: {improvement:.1f}%")
    if vol is not None:
        print(f"Volatility MAE  -- LSTM: {metrics['vol_mae_lstm']:.4f}  "
              f"trailing: {metrics['vol_mae_trailing']:.4f}")
    print(f"\nSaved model -> {os.path.join(MODELS_DIR, 'volsight_lstm.pt')}")


if __name__ == "__main__":
    main()
