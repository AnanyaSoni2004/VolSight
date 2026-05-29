"""
Lightweight sanity tests for VolSight.

Run with:  python -m pytest -q   (or just  python tests/test_core.py)
No pytest dependency required for the __main__ path.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from volsight import (
    bs_price, bs_greeks, implied_vol, GarchParams,
    forward_integrated_vol, build_dataset, SimConfig,
)


def test_bs_textbook_value():
    c = float(bs_price(100, 100, 1, 0.05, 0.20, option_type="call"))
    assert abs(c - 10.4506) < 1e-3


def test_put_call_parity():
    c = float(bs_price(100, 110, 0.5, 0.03, 0.25, option_type="call"))
    p = float(bs_price(100, 110, 0.5, 0.03, 0.25, option_type="put"))
    assert abs((c - p) - (100 - 110 * np.exp(-0.03 * 0.5))) < 1e-8


def test_implied_vol_roundtrip():
    price = float(bs_price(100, 95, 0.75, 0.02, 0.30, option_type="call"))
    iv = implied_vol(price, 100, 95, 0.75, 0.02, option_type="call")
    assert abs(iv - 0.30) < 1e-4


def test_greeks_signs():
    g = bs_greeks(100, 100, 1, 0.05, 0.20, option_type="call")
    assert 0 < g["delta"] < 1
    assert g["gamma"] > 0 and g["vega"] > 0


def test_garch_mean_reversion():
    p = GarchParams(8e-7, 0.08, 0.90)
    hi = forward_integrated_vol(p.uncond_var * 4, p, 60)
    lo = forward_integrated_vol(p.uncond_var * 0.25, p, 60)
    assert lo < p.uncond_vol_annual < hi


def test_dataset_shapes():
    d = build_dataset(SimConfig(n_samples=64, seed=1))
    assert d["seq"].shape == (64, 30, 2)
    assert d["static"].shape == (64, 4)
    assert d["y_fair"].shape == (64,) and d["y_fwd_vol"].shape == (64,)
    assert np.all(d["y_fair"] >= 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
