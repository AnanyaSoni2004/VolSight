"""
Synthetic market simulator and supervised-dataset builder.

We simulate risk-neutral underlying paths whose volatility follows a GARCH(1,1)
recursion, so the data exhibits realistic volatility clustering. For each
training example we expose:

  * a window of recent daily log-returns  -> the LSTM's sequential input
  * static contract features (log-moneyness, time-to-maturity, rate, trailing
    realized vol)                          -> concatenated to the LSTM state
  * a target price                         -> the GARCH-optimal Black-Scholes
                                              price using the *forward* integrated
                                              volatility (the "fair" value)
  * a naive benchmark price                -> Black-Scholes using *trailing*
                                              realized vol (the lagging estimate)

The learning task: given only the raw return history and contract terms, predict
the forward-looking fair price. A model that merely echoes trailing vol (like
naive BS) will lag during volatility regime shifts; a good sequence model learns
to anticipate mean reversion / clustering and tracks the fair price more closely.

Prices and strikes are normalized by spot to keep the target scale-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .black_scholes import bs_price
from .garch_vol import GarchParams, forward_integrated_vol, trailing_realized_vol

TRADING_DAYS = 252


@dataclass
class SimConfig:
    n_samples: int = 20000          # number of (sequence, contract) examples
    seq_len: int = 30               # length of the return window fed to the LSTM
    burn_in: int = 250              # steps simulated before sampling to reach stationarity
    # GARCH(1,1) daily parameters (persistence alpha+beta < 1 for stationarity)
    omega: float = 8.0e-7
    alpha: float = 0.08
    beta: float = 0.90
    # contract sampling ranges
    min_T_days: int = 5
    max_T_days: int = 60
    moneyness_lo: float = 0.80      # K / S range
    moneyness_hi: float = 1.20
    r_lo: float = 0.00
    r_hi: float = 0.06
    # Student-t degrees of freedom for innovations (None -> Gaussian). Fat tails
    # create sharp conditional-variance spikes that lag an equal-weighted
    # trailing estimator, giving a sequence model room to add value.
    nu: float | None = 5.0
    seed: int = 7
    params: GarchParams = field(init=False)

    def __post_init__(self):
        self.params = GarchParams(self.omega, self.alpha, self.beta)


def _simulate_garch_returns(n_steps, params: GarchParams, r_daily, rng, nu=None):
    """Simulate `n_steps` risk-neutral daily log-returns with GARCH variance.

    log-return_t = (r_daily - 0.5 * sigma_t^2) + sigma_t * z_t,  z_t ~ unit-var
    sigma^2_{t+1} = omega + alpha * eps_t^2 + beta * sigma^2_t,  eps_t = sigma_t z_t

    If `nu` is given, z_t is a standardized Student-t with `nu` dof (unit variance),
    producing fat tails while preserving E[z^2] = 1 so the GARCH variance
    recursion and the forward-variance forecast formula remain valid.
    """
    sigma2 = params.uncond_var
    returns = np.empty(n_steps)
    sig2_series = np.empty(n_steps)
    t_scale = np.sqrt(nu / (nu - 2.0)) if nu is not None else None
    for t in range(n_steps):
        sig2_series[t] = sigma2
        if nu is None:
            z = rng.standard_normal()
        else:
            z = rng.standard_t(nu) / t_scale
        eps = np.sqrt(sigma2) * z
        returns[t] = (r_daily - 0.5 * sigma2) + eps
        sigma2 = params.omega + params.alpha * eps ** 2 + params.beta * sigma2
    # sigma2 now holds the one-step-ahead variance forecast for the next return
    return returns, sig2_series, sigma2


def build_dataset(cfg: SimConfig | None = None):
    """Return a dict of arrays ready for model training.

    Keys:
      seq        : (N, seq_len, 2)  log-returns and squared log-returns
      static     : (N, 4)           [log_moneyness, T_years, r, trailing_vol]
      y_fair     : (N,)             target price / spot (GARCH-optimal BS)
      y_bs_naive : (N,)             naive trailing-vol BS price / spot
      meta       : (N, 6)           [S, K, T_years, r, fwd_vol, trail_vol] for analysis
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    p = cfg.params

    N = cfg.n_samples
    L = cfg.seq_len

    seq = np.empty((N, L, 2), dtype=np.float32)
    static = np.empty((N, 4), dtype=np.float32)
    y_fair = np.empty(N, dtype=np.float32)
    y_bs = np.empty(N, dtype=np.float32)
    meta = np.empty((N, 6), dtype=np.float32)

    for i in range(N):
        r_annual = rng.uniform(cfg.r_lo, cfg.r_hi)
        r_daily = r_annual / TRADING_DAYS

        # Simulate enough history: burn-in + the observation window.
        n_steps = cfg.burn_in + L
        rets, _, sigma2_next = _simulate_garch_returns(n_steps, p, r_daily, rng, nu=cfg.nu)
        window = rets[-L:]

        S = 100.0  # spot is a numeraire; everything is normalized by it
        K = S * rng.uniform(cfg.moneyness_lo, cfg.moneyness_hi)
        T_days = int(rng.integers(cfg.min_T_days, cfg.max_T_days + 1))
        T_years = T_days / TRADING_DAYS

        # forward-looking (fair) volatility and the lagging trailing estimate
        fwd_vol = forward_integrated_vol(sigma2_next, p, T_days)
        trail_vol = trailing_realized_vol(window)

        fair_price = float(bs_price(S, K, T_years, r_annual, fwd_vol, option_type="call"))
        naive_price = float(bs_price(S, K, T_years, r_annual, trail_vol, option_type="call"))

        seq[i, :, 0] = window
        seq[i, :, 1] = window ** 2
        static[i] = [np.log(S / K), T_years, r_annual, trail_vol]
        y_fair[i] = fair_price / S
        y_bs[i] = naive_price / S
        meta[i] = [S, K, T_years, r_annual, fwd_vol, trail_vol]

    return {
        "seq": seq,
        "static": static,
        "y_fair": y_fair,
        "y_bs_naive": y_bs,
        "y_fwd_vol": meta[:, 4].copy(),    # GARCH-optimal forward vol (vol-head target)
        "y_trail_vol": meta[:, 5].copy(),  # trailing realized vol (naive baseline)
        "meta": meta,
        "config": cfg,
    }
