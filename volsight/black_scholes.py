"""
Black-Scholes analytical pricing engine for European options.

All functions are vectorized (accept scalars or NumPy arrays) and return prices /
Greeks for European calls and puts. This module is the classical benchmark that
the neural model is evaluated against.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

ArrayLike = "float | np.ndarray"


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """Return the Black-Scholes d1 and d2 terms.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to maturity in years
    r : continuously-compounded risk-free rate
    sigma : annualized volatility
    q : continuous dividend yield (default 0)
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Guard against zero maturity / volatility to avoid division warnings.
    eps = 1e-12
    sqrtT = np.sqrt(np.maximum(T, eps))
    vol = np.maximum(sigma, eps) * sqrtT

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol
    d2 = d1 - vol
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Black-Scholes price of a European option.

    Returns the intrinsic value when T <= 0.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-np.asarray(r, dtype=float) * T)
    disc_q = np.exp(-q * T)

    if option_type == "call":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
    elif option_type == "put":
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        intrinsic = np.maximum(K - S, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    price = np.where(T <= 0, intrinsic, price)
    return price


def bs_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Return a dict of the standard Greeks (per-unit conventions).

    delta : dPrice/dS
    gamma : d2Price/dS2
    vega  : dPrice/dSigma (per 1.00 change in vol, i.e. 100 vol points)
    theta : dPrice/dT  (per year; divide by 365 for per-day)
    rho   : dPrice/dr  (per 1.00 change in rate)
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    pdf_d1 = norm.pdf(d1)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def implied_vol(price, S, K, T, r, q=0.0, option_type="call",
                tol=1e-6, max_iter=100):
    """Recover Black-Scholes implied volatility via bisection.

    Robust (no derivative blow-ups) at the cost of a few extra iterations.
    """
    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = bs_price(S, K, T, r, mid, q, option_type) - price
        if abs(val) < tol:
            return mid
        # price is monotonically increasing in sigma
        if val > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
