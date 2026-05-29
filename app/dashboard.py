"""
VolSight interactive dashboard.

Run with:
    streamlit run app/dashboard.py

The non-UI helpers (simulate_recent_window, price_point, ...) are importable and
unit-tested separately; all Streamlit UI lives inside main().
"""
from __future__ import annotations

import os
import sys

import numpy as np

# allow `streamlit run app/dashboard.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volsight import (
    bs_price, bs_greeks, load_model, predict,
    GarchParams, forward_integrated_vol, trailing_realized_vol,
)
from volsight.data_generation import _simulate_garch_returns, TRADING_DAYS

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "volsight_lstm.pt"
)


# --------------------------------------------------------------------------- #
# Pure helpers (no Streamlit) -- safe to import and unit-test
# --------------------------------------------------------------------------- #
def simulate_recent_window(sigma_base, persistence, shock_mult, seq_len, r, seed=0):
    """Simulate a recent daily return window from a chosen volatility regime.

    sigma_base   : target annualized base volatility (unconditional)
    persistence  : GARCH alpha+beta in (0,1); higher => stickier vol
    shock_mult   : multiply the final few returns' vol to mimic a recent shock
    Returns (window[seq_len], params, sigma2_next).
    """
    persistence = float(np.clip(persistence, 0.50, 0.995))
    alpha = min(0.15, max(0.02, (1 - persistence) * 2.0))
    beta = persistence - alpha
    daily_var = (sigma_base ** 2) / TRADING_DAYS
    omega = daily_var * (1 - persistence)
    params = GarchParams(omega=omega, alpha=alpha, beta=beta)

    rng = np.random.default_rng(seed)
    rets, _, sigma2_next = _simulate_garch_returns(
        250 + seq_len, params, r / TRADING_DAYS, rng, nu=5.0
    )
    window = rets[-seq_len:].copy()
    if shock_mult and shock_mult != 1.0:
        # amplify the most recent 3 days to emulate a fresh volatility shock
        window[-3:] *= shock_mult
        # reflect the shock in the conditional variance forecast
        sigma2_next = params.omega + params.alpha * window[-1] ** 2 + params.beta * sigma2_next
    return window.astype(np.float32), params, float(sigma2_next)


def make_features(window, S, K, T_years, r):
    """Build the (seq, static) model inputs for one contract."""
    seq = np.stack([window, window ** 2], axis=-1).astype(np.float32)
    trail = trailing_realized_vol(window)
    static = np.array([np.log(S / K), T_years, r, trail], dtype=np.float32)
    return seq, static, trail


def price_point(model, scaler, window, S, K, T_years, r):
    """Return a dict of LSTM and Black-Scholes prices/vols for one contract."""
    seq, static, trail = make_features(window, S, K, T_years, r)
    lstm_price_rel, lstm_vol = predict(model, scaler, seq, static, return_vol=True)
    lstm_price = float(lstm_price_rel[0]) * S
    lstm_vol = None if lstm_vol is None else float(lstm_vol[0])
    bs_at_trail = float(bs_price(S, K, T_years, r, trail, option_type="call"))
    out = {
        "lstm_price": lstm_price,
        "lstm_vol": lstm_vol,
        "trailing_vol": trail,
        "bs_price_trailing_vol": bs_at_trail,
    }
    if lstm_vol is not None:
        out["bs_price_lstm_vol"] = float(bs_price(S, K, T_years, r, lstm_vol, option_type="call"))
    return out


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def main():  # pragma: no cover - exercised only under `streamlit run`
    import pandas as pd
    import streamlit as st
    import matplotlib.pyplot as plt

    ACCENT, ACCENT2, GREY = "#2563eb", "#f59e0b", "#64748b"
    st.set_page_config(page_title="VolSight", layout="wide")

    st.title("VolSight — Neural Option Pricing")
    st.caption("LSTM volatility forecasting + analytical Black-Scholes, benchmarked live.")

    @st.cache_resource
    def _load():
        if not os.path.exists(MODEL_PATH):
            return None, None
        return load_model(MODEL_PATH, device="cpu")

    model, scaler = _load()
    if model is None:
        st.error("No trained model found. Run `python -m scripts.train` first, then reload.")
        st.stop()

    sb = st.sidebar
    sb.header("Contract")
    S = sb.slider("Spot S", 50.0, 150.0, 100.0, 1.0)
    K = sb.slider("Strike K", 50.0, 150.0, 100.0, 1.0)
    T_days = sb.slider("Days to maturity", 5, 60, 30, 1)
    r = sb.slider("Risk-free rate r", 0.0, 0.08, 0.02, 0.005)
    T_years = T_days / TRADING_DAYS

    sb.header("Volatility regime (drives the LSTM input)")
    sigma_base = sb.slider("Base volatility", 0.05, 0.60, 0.20, 0.01)
    persistence = sb.slider("GARCH persistence", 0.50, 0.99, 0.94, 0.01)
    shock_mult = sb.slider("Recent shock multiplier", 1.0, 5.0, 1.0, 0.25)
    seed = sb.number_input("Scenario seed", 0, 9999, 7, 1)

    window, params, _ = simulate_recent_window(
        sigma_base, persistence, shock_mult, 30, r, seed
    )
    res = price_point(model, scaler, window, S, K, T_years, r)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LSTM price", f"${res['lstm_price']:.3f}")
    c2.metric("BS price (trailing vol)", f"${res['bs_price_trailing_vol']:.3f}",
              f"{res['lstm_price'] - res['bs_price_trailing_vol']:+.3f}")
    c3.metric("LSTM-implied vol", f"{res['lstm_vol']*100:.1f}%")
    c4.metric("Trailing realized vol", f"{res['trailing_vol']*100:.1f}%")

    tab1, tab2, tab3 = st.tabs(["Pricing & Greeks", "Heatmaps", "Error analysis"])

    # ---- Tab 1: Greeks + sensitivity curves ----
    with tab1:
        vol = res["lstm_vol"]
        greeks = bs_greeks(S, K, T_years, r, vol, option_type="call")
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Delta", f"{greeks['delta']:.3f}")
        g2.metric("Gamma", f"{greeks['gamma']:.4f}")
        g3.metric("Vega", f"{greeks['vega']/100:.3f}")
        g4.metric("Theta/day", f"{greeks['theta']/365:.4f}")
        g5.metric("Rho", f"{greeks['rho']/100:.4f}")
        st.caption("Greeks computed analytically at the LSTM-implied volatility.")

        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
        S_grid = np.linspace(0.6 * K, 1.4 * K, 60)
        axes[0].plot(S_grid, bs_price(S_grid, K, T_years, r, vol, option_type="call"), color=ACCENT)
        axes[0].axvline(S, color=GREY, ls="--", lw=1)
        axes[0].set_title("Price vs spot"); axes[0].set_xlabel("S")

        vol_grid = np.linspace(0.05, 0.6, 60)
        axes[1].plot(vol_grid, bs_price(S, K, T_years, r, vol_grid, option_type="call"), color=ACCENT)
        axes[1].axvline(vol, color=GREY, ls="--", lw=1)
        axes[1].set_title("Price vs volatility"); axes[1].set_xlabel("sigma")

        T_grid = np.linspace(1, 90, 60) / TRADING_DAYS
        axes[2].plot(T_grid * 252, bs_price(S, K, T_grid, r, vol, option_type="call"), color=ACCENT)
        axes[2].axvline(T_days, color=GREY, ls="--", lw=1)
        axes[2].set_title("Price vs maturity"); axes[2].set_xlabel("days")
        for ax in axes:
            ax.grid(alpha=0.25)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout(); st.pyplot(fig)

    # ---- Tab 2: heatmaps ----
    with tab2:
        st.subheader("Black-Scholes price surface")
        spot_axis = np.linspace(0.7 * K, 1.3 * K, 40)
        vol_axis = np.linspace(0.05, 0.5, 40)
        SS, VV = np.meshgrid(spot_axis, vol_axis)
        price_surface = bs_price(SS, K, T_years, r, VV, option_type="call")
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(price_surface, origin="lower", aspect="auto",
                       extent=[spot_axis[0], spot_axis[-1], vol_axis[0], vol_axis[-1]],
                       cmap="viridis")
        ax.set_xlabel("spot"); ax.set_ylabel("volatility"); fig.colorbar(im, ax=ax, label="call price")
        st.pyplot(fig)

        st.subheader("LSTM − BS price deviation (moneyness x maturity)")
        st.caption("Holding the simulated return window fixed, contract terms vary across the grid.")
        m_axis = np.linspace(0.85, 1.18, 28)         # S/K
        d_axis = np.arange(5, 61, 3)                 # days
        dev = np.zeros((len(d_axis), len(m_axis)))
        for i, dd in enumerate(d_axis):
            for j, mm in enumerate(m_axis):
                Kij = S / mm
                p = price_point(model, scaler, window, S, Kij, dd / TRADING_DAYS, r)
                dev[i, j] = p["lstm_price"] - p["bs_price_trailing_vol"]
        fig, ax = plt.subplots(figsize=(7, 4))
        vmax = np.abs(dev).max() + 1e-9
        im = ax.imshow(dev, origin="lower", aspect="auto",
                       extent=[m_axis[0], m_axis[-1], d_axis[0], d_axis[-1]],
                       cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("moneyness S/K"); ax.set_ylabel("days to maturity")
        fig.colorbar(im, ax=ax, label="LSTM − BS ($)")
        st.pyplot(fig)

    # ---- Tab 3: error analysis ----
    with tab3:
        st.subheader("Monte-Carlo benchmark on fresh scenarios")
        n_mc = st.slider("Scenarios", 200, 3000, 1000, 100)
        rng = np.random.default_rng(int(seed) + 1)
        lstm_p, bs_p, fair_p = [], [], []
        for k in range(n_mc):
            w, prm, s2 = simulate_recent_window(
                sigma_base, persistence, shock_mult, 30, r, seed=int(seed) + 100 + k
            )
            Kk = S * rng.uniform(0.85, 1.18)
            Tk_days = int(rng.integers(5, 61))
            Tk = Tk_days / TRADING_DAYS
            fwd = forward_integrated_vol(s2, prm, Tk_days)
            fair_p.append(bs_price(S, Kk, Tk, r, fwd, option_type="call"))
            p = price_point(model, scaler, w, S, Kk, Tk, r)
            lstm_p.append(p["lstm_price"]); bs_p.append(p["bs_price_trailing_vol"])
        lstm_p, bs_p, fair_p = map(np.array, (lstm_p, bs_p, fair_p))
        lstm_rmse = float(np.sqrt(np.mean((lstm_p - fair_p) ** 2)))
        bs_rmse = float(np.sqrt(np.mean((bs_p - fair_p) ** 2)))

        m1, m2, m3 = st.columns(3)
        m1.metric("LSTM RMSE", f"{lstm_rmse:.4f}")
        m2.metric("naive BS RMSE", f"{bs_rmse:.4f}")
        m3.metric("RMSE improvement", f"{(1 - lstm_rmse / max(bs_rmse,1e-9)) * 100:.1f}%")

        fig, ax = plt.subplots(figsize=(6, 5))
        lim = max(fair_p.max(), lstm_p.max(), bs_p.max()) * 1.03
        ax.plot([0, lim], [0, lim], color=GREY, ls="--", lw=1)
        ax.scatter(fair_p, bs_p, s=8, alpha=0.35, color=ACCENT2, label="naive BS")
        ax.scatter(fair_p, lstm_p, s=8, alpha=0.35, color=ACCENT, label="LSTM")
        ax.set_xlabel("fair price"); ax.set_ylabel("model price"); ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
