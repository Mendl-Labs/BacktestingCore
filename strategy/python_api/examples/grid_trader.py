"""
Example: Grid Trading Strategy

Places a grid of buy and sell orders around a reference price.
Profits from mean-reverting price action within a range.

Grid trading is inherently stateful, so compute_signals() uses a sequential
loop internally. This is still faster than per-tick PyO3 FFI calls since
the loop runs entirely in Python without Rust round-trips.
"""

import numpy as np

from trading_platform import BaseStrategy, DataRequirements, RiskLimits


class Strategy(BaseStrategy):
    """Grid strategy that places orders at fixed price intervals."""

    def name(self) -> str:
        return "Grid_Trader"

    def description(self) -> str:
        p = self.params if hasattr(self, "params") else {}
        levels  = p.get("grid_levels", 5)
        spacing = p.get("grid_spacing_pct", 0.005)
        return f"Grid trading with {levels} levels at {float(spacing)*100:.1f}% spacing"

    def initialize(self, parameters: dict) -> None:
        self.params = {
            "grid_levels": 5,
            "grid_spacing_pct": 0.005,
            "order_size": 0.01,
        }
        self.params.update({k: v for k, v in parameters.items() if v is not None})

    def compute_signals(self, prices, volumes, timestamps):
        p             = self.params if hasattr(self, "params") else {}
        grid_levels   = int(p.get("grid_levels", 5))
        spacing_pct   = float(p.get("grid_spacing_pct", 0.005))

        sig            = np.zeros(len(prices), dtype=np.int8)
        reference      = prices[0] if len(prices) > 0 else 0.0
        filled_levels: set = set()

        for i, price in enumerate(prices):
            if price <= 0:
                continue

            # BUY at grid levels below reference
            for level in range(-grid_levels, 0):
                grid_price = reference * (1 + level * spacing_pct)
                if price <= grid_price and level not in filled_levels:
                    sig[i] = self.BUY
                    filled_levels.add(level)
                    break

            # SELL at grid levels above reference
            for level in range(1, grid_levels + 1):
                grid_price = reference * (1 + level * spacing_pct)
                if price >= grid_price and level not in filled_levels:
                    sig[i] = self.SELL
                    filled_levels.add(level)
                    break

            # Reset grid when all levels are filled
            if len(filled_levels) >= grid_levels * 2:
                reference = price
                filled_levels.clear()

        return sig

    def update_parameters(self, parameters: dict) -> None:
        if not hasattr(self, "params"):
            self.params = {}
        self.params.update({k: v for k, v in parameters.items() if v is not None})

    def get_parameters(self) -> dict:
        return dict(self.params) if hasattr(self, "params") else {}

    def parameter_space(self) -> dict:
        return {
            "grid_levels":     {"type": "int",   "min": 2,     "max": 20,   "default": 5},
            "grid_spacing_pct":{"type": "float", "min": 0.001, "max": 0.02, "default": 0.005},
            "order_size":      {"type": "float", "min": 0.001, "max": 0.1,  "default": 0.01},
        }

    def data_requirements(self) -> DataRequirements:
        return DataRequirements(needs_trades=True, lookback_window=50)

    def get_risk_limits(self) -> RiskLimits:
        return RiskLimits(max_position_size=1.0, max_drawdown_pct=0.15)