"""
GARCH(1,1) volatility estimation and forecasting.

Two responsibilities:
  1. Estimate conditional-variance parameters from an observed return series
     (wraps the `arch` package).
  2. Produce the forward-looking *integrated* volatility over an option's life,
     which is the volatility a GARCH-consistent trader would feed into
     Black-Scholes. This forward vol is what the LSTM is implicitly trying to
     recover from the raw return sequence.

The closed-form multi-step variance forecast for GARCH(1,1) is

    E[sigma^2_{t+k} | F_t] = s2_uncond + (alpha + beta)^(k-1) * (s2_{t+1} - s2_uncond)

with the unconditional variance s2_uncond = omega / (1 - alpha - beta). The
expected average variance over the next N steps integrates this geometric decay.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from arch import arch_model
    _HAS_ARCH = True
except Exception:  # pragma: no cover - arch is an optional heavy dep
    _HAS_ARCH = False

TRADING_DAYS = 252


@dataclass
class GarchParams:
    """Container for GARCH(1,1) parameters (daily scale)."""
    omega: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def uncond_var(self) -> float:
        denom = max(1.0 - self.persistence, 1e-8)
        return self.omega / denom

    @property
    def uncond_vol_annual(self) -> float:
        return float(np.sqrt(self.uncond_var * TRADING_DAYS))


def fit_garch(returns: np.ndarray, rescale: bool = True) -> GarchParams:
    """Estimate GARCH(1,1) parameters from a 1-D array of daily log-returns.

    `arch` works best on returns expressed in percent, so we rescale by 100,
    fit, then convert the variance parameters back to the raw (decimal) scale.
    """
    if not _HAS_ARCH:
        raise ImportError("The 'arch' package is required for fit_garch().")

    r = np.asarray(returns, dtype=float).ravel()
    scale = 100.0 if rescale else 1.0
    model = arch_model(r * scale, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    res = model.fit(disp="off")
    p = res.params
    # Variance parameters live on the (scale)^2 grid; omega rescales by scale^2,
    # while alpha and beta are scale-invariant.
    return GarchParams(
        omega=float(p["omega"]) / (scale ** 2),
        alpha=float(p["alpha[1]"]),
        beta=float(p["beta[1]"]),
    )


def forecast_path_variance(sigma2_next: float, params: GarchParams,
                           horizon: int) -> np.ndarray:
    """Per-step expected variance for k = 1..horizon given next-step variance."""
    s2u = params.uncond_var
    phi = params.persistence
    k = np.arange(horizon)  # 0-indexed; k=0 is the 1-step-ahead forecast
    return s2u + (phi ** k) * (sigma2_next - s2u)


def forward_integrated_vol(sigma2_next: float, params: GarchParams,
                           horizon_days: int) -> float:
    """Annualized volatility of the *average* variance over `horizon_days`.

    This is the GARCH-optimal constant volatility to plug into Black-Scholes for
    an option maturing in `horizon_days` trading days.
    """
    horizon_days = max(int(horizon_days), 1)
    per_step = forecast_path_variance(sigma2_next, params, horizon_days)
    avg_daily_var = float(np.mean(per_step))
    return float(np.sqrt(avg_daily_var * TRADING_DAYS))


def trailing_realized_vol(returns: np.ndarray) -> float:
    """Backward-looking annualized realized volatility of a return window.

    This is the naive point estimate a practitioner without a dynamic model
    would feed into Black-Scholes.
    """
    r = np.asarray(returns, dtype=float).ravel()
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))
