"""
LSTM-based neural option-pricing model (PyTorch), with two output heads.

  head="price"  (black-box)
      return sequence -> LSTM -> hidden -+-> MLP -> softplus -> price/spot
                static features ---------'

  head="vol"    (hybrid, recommended)
      return sequence -> LSTM -> hidden -+-> MLP -> softplus -> volatility
                static features ---------'                         |
                                       differentiable Black-Scholes layer
                                                                   |
                                                              price/spot

The hybrid head asks the network to do the *hard, sequence-dependent* part
(forecast forward-looking volatility from the return history) while a closed-form
Black-Scholes layer handles the *known* price nonlinearity exactly. This makes
the model both more accurate and interpretable: the predicted volatility is a
directly readable "LSTM-implied vol" you can compare against trailing realized
vol and the GARCH forecast.

Everything is normalized by spot, so S = 1 and K = exp(-log_moneyness) inside the
BS layer. Static features are [log_moneyness, T_years, r, trailing_vol].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_SQRT2 = math.sqrt(2.0)


def bs_call_unit_spot(K_rel, T, r, sigma, eps: float = 1e-6):
    """Differentiable Black-Scholes call price with spot fixed at 1.

    All inputs are tensors of equal shape. Returns price / spot.
    """
    sqrtT = torch.sqrt(torch.clamp(T, min=eps))
    vol = torch.clamp(sigma, min=eps) * sqrtT
    d1 = (-torch.log(torch.clamp(K_rel, min=eps)) + (r + 0.5 * sigma ** 2) * T) / vol
    d2 = d1 - vol
    Nd1 = 0.5 * (1.0 + torch.erf(d1 / _SQRT2))
    Nd2 = 0.5 * (1.0 + torch.erf(d2 / _SQRT2))
    return Nd1 - K_rel * torch.exp(-r * T) * Nd2


@dataclass
class ModelConfig:
    seq_features: int = 2
    static_features: int = 4
    hidden_size: int = 64
    num_layers: int = 2
    mlp_hidden: int = 64
    dropout: float = 0.1
    head: str = "vol"          # "vol" (hybrid) or "price" (black-box)


class OptionLSTM(nn.Module):
    # static feature column order produced by data_generation.build_dataset
    LOG_MONEYNESS, T_IDX, R_IDX, TRAIL_VOL = 0, 1, 2, 3

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.seq_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head_net = nn.Sequential(
            nn.Linear(cfg.hidden_size + cfg.static_features, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, seq, static, static_raw=None, return_vol=False):
        """seq:(B,L,F) scaled. static:(B,4) scaled. static_raw:(B,4) unscaled.

        For the hybrid ("vol") head the BS layer needs the *unscaled* contract
        terms, supplied via `static_raw`.
        """
        _, (h_n, _) = self.lstm(seq)
        combined = torch.cat([h_n[-1], static], dim=1)
        out = self.softplus(self.head_net(combined)).squeeze(-1)

        if self.cfg.head == "price":
            return (out, None) if return_vol else out

        # hybrid: `out` is volatility -> analytical BS price
        if static_raw is None:
            raise ValueError("head='vol' requires static_raw for the BS layer")
        log_m = static_raw[:, self.LOG_MONEYNESS]
        T = static_raw[:, self.T_IDX]
        r = static_raw[:, self.R_IDX]
        K_rel = torch.exp(-log_m)            # K / S
        price = bs_call_unit_spot(K_rel, T, r, out)
        return (price, out) if return_vol else price


@dataclass
class StandardScaler:
    mean: list
    std: list

    @classmethod
    def fit(cls, x: np.ndarray):
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean.tolist(), std.tolist())

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - np.asarray(self.mean)) / np.asarray(self.std)


def train_model(data: dict, model_cfg: "ModelConfig | None" = None,
                epochs: int = 60, batch_size: int = 256, lr: float = 1e-3,
                val_frac: float = 0.15, device: "str | None" = None,
                weight_decay: float = 0.0, verbose: bool = True):
    """Train an OptionLSTM. Returns (best_model, scaler, history).

    Keeps the validation-best weights (simple early-stopping memory).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = model_cfg or ModelConfig()

    seq = data["seq"].astype(np.float32)
    static = data["static"].astype(np.float32)
    # The vol head is supervised directly on volatility (well-conditioned),
    # the price head on price/spot. Black-Scholes converts vol->price exactly.
    head = (model_cfg.head if model_cfg else "vol")
    if head == "vol":
        target = data["y_fwd_vol"].astype(np.float32)
    else:
        target = data["y_fair"].astype(np.float32)

    scaler = StandardScaler.fit(static)
    static_s = scaler.transform(static).astype(np.float32)

    n = len(target)
    idx = np.random.default_rng(0).permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def make_loader(ids, shuffle):
        ds = TensorDataset(
            torch.tensor(seq[ids]), torch.tensor(static_s[ids]),
            torch.tensor(static[ids]), torch.tensor(target[ids]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)

    model = OptionLSTM(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss()

    def model_target(sb, stb, strawb):
        """Return the quantity to compare against `target` for the active head."""
        price, vol = model(sb, stb, strawb, return_vol=True)
        return vol if head == "vol" else price

    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        tr = 0.0
        for sb, stb, strawb, yb in train_loader:
            sb, stb, strawb, yb = (t.to(device) for t in (sb, stb, strawb, yb))
            opt.zero_grad()
            loss = loss_fn(model_target(sb, stb, strawb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr += loss.item() * len(yb)
        tr /= len(train_idx)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for sb, stb, strawb, yb in val_loader:
                sb, stb, strawb, yb = (t.to(device) for t in (sb, stb, strawb, yb))
                va += loss_fn(model_target(sb, stb, strawb), yb).item() * len(yb)
        va /= max(len(val_idx), 1)
        sched.step(va)

        history["train"].append(tr)
        history["val"].append(va)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % max(epochs // 12, 1) == 0 or epoch == epochs - 1):
            print(f"epoch {epoch+1:3d}/{epochs} | train {tr:.3e} | val {va:.3e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, history


@torch.no_grad()
def predict(model, scaler, seq, static, device=None, return_vol=False):
    """Predict price/spot (and optionally implied vol) for raw inputs."""
    device = device or next(model.parameters()).device
    model.eval()
    seq = np.asarray(seq, dtype=np.float32)
    static = np.asarray(static, dtype=np.float32)
    if seq.ndim == 2:
        seq = seq[None]
        static = static[None]
    seq_t = torch.tensor(seq).to(device)
    static_s = torch.tensor(scaler.transform(static).astype(np.float32)).to(device)
    static_raw = torch.tensor(static).to(device)
    price, vol = model(seq_t, static_s, static_raw, return_vol=True)
    price = price.cpu().numpy()
    if return_vol:
        vol = None if vol is None else vol.cpu().numpy()
        return price, vol
    return price


def save_model(path, model, scaler):
    torch.save({
        "state_dict": model.state_dict(),
        "model_cfg": asdict(model.cfg),
        "scaler": {"mean": scaler.mean, "std": scaler.std},
    }, path)


def load_model(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = OptionLSTM(ModelConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = StandardScaler(ckpt["scaler"]["mean"], ckpt["scaler"]["std"])
    return model, scaler
