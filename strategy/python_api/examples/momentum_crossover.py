"""
Example: SMA Momentum Crossover Strategy

A classic dual-moving-average crossover strategy. Goes long when the fast
SMA crosses above the slow SMA, and short when it crosses below.

compute_signals() is called ONCE with the full tick array - pandas rolling
operations on 1M ticks run in ~100ms, vs 83s for the old per-tick API.

Usage:
    Submit via the platform with strategy_type="Custom" and this file
    as python_source_code, or paste into the in-browser editor.
"""

import numpy as np
import pandas as pd

from trading_platform import BaseStrategy, DataRequirements, RiskLimits


class Strategy(BaseStrategy):
    """SMA Momentum Crossover - long/short with configurable periods."""

    def name(self) -> str:
        return "SMA_Momentum_Crossover"

    def description(self) -> str:
        fp = self.params.get("fast_period", 10) if hasattr(self, "params") else 10
        sp = self.params.get("slow_period", 30) if hasattr(self, "params") else 30
        return f"SMA crossover ({fp}/{sp}) momentum strategy"

    def initialize(self, parameters: dict) -> None:
        self.params = {"fast_period": 10, "slow_period": 30, "risk_per_trade": 0.01}
        self.params.update({k: v for k, v in parameters.items() if v is not None})

    def compute_signals(self, prices, volumes, timestamps):
        fast = int(self.params.get("fast_period", 10)) if hasattr(self, "params") else 10
        slow = int(self.params.get("slow_period", 30)) if hasattr(self, "params") else 30

        s = pd.Series(prices)
        fast_sma = s.rolling(fast).mean()
        slow_sma = s.rolling(slow).mean()

        sig = np.zeros(len(prices), dtype=np.int8)
        # BUY on bullish crossover
        sig[(fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))] = self.BUY
        # SELL on bearish crossover
        sig[(fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))] = self.SELL
        return sig

    def update_parameters(self, parameters: dict) -> None:
        if not hasattr(self, "params"):
            self.params = {}
        self.params.update({k: v for k, v in parameters.items() if v is not None})

    def get_parameters(self) -> dict:
        return dict(self.params) if hasattr(self, "params") else {}

    def parameter_space(self) -> dict:
        return {
            "fast_period":    {"type": "int",   "min": 5,     "max": 50,   "default": 10},
            "slow_period":    {"type": "int",   "min": 20,    "max": 200,  "default": 30},
            "risk_per_trade": {"type": "float", "min": 0.005, "max": 0.05, "default": 0.01},
        }

    def data_requirements(self) -> DataRequirements:
        return DataRequirements.momentum()

    def required_data_window(self) -> int:
        slow = int(self.params.get("slow_period", 30)) if hasattr(self, "params") else 30
        return slow + 10

    def get_risk_limits(self) -> RiskLimits:
        return RiskLimits(max_position_size=1.0, max_drawdown_pct=0.15)