"""VolSight: neural option pricing with GARCH-aware LSTM models."""

from .black_scholes import bs_price, bs_greeks, implied_vol
from .garch_vol import (
    GarchParams,
    fit_garch,
    forward_integrated_vol,
    trailing_realized_vol,
)
from .data_generation import SimConfig, build_dataset
from .lstm_model import (
    ModelConfig,
    OptionLSTM,
    train_model,
    predict,
    save_model,
    load_model,
)
from .benchmark import error_metrics, comparison_table, format_table

__version__ = "0.1.0"

__all__ = [
    "bs_price", "bs_greeks", "implied_vol",
    "GarchParams", "fit_garch", "forward_integrated_vol", "trailing_realized_vol",
    "SimConfig", "build_dataset",
    "ModelConfig", "OptionLSTM", "train_model", "predict", "save_model", "load_model",
    "error_metrics", "comparison_table", "format_table",
]
