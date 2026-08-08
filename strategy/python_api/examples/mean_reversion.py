"""
Example: Bollinger Band Mean Reversion Strategy

Buys when price drops below the lower Bollinger Band (oversold, RSI confirmed)
and sells when price rises above the upper Bollinger Band (overbought).

compute_signals() is called ONCE with the full tick array, so all indicator
computation is vectorized via pandas — no per-tick Python overhead.
"""

import numpy as np
import pandas as pd

from trading_platform import BaseStrategy, DataRequirements, RiskLimits


class Strategy(BaseStrategy):
    """Bollinger Band mean reversion with RSI confirmation."""

    def name(self) -> str:
        return "BB_Mean_Reversion"

    def description(self) -> str:
        return "Bollinger Band mean reversion with RSI filter"

    def initialize(self, parameters: dict) -> None:
        self.params = {
            "bb_period": 20, "bb_std": 2.0,
            "rsi_period": 14, "rsi_oversold": 30.0, "rsi_overbought": 70.0,
            "risk_per_trade": 0.02,
        }
        self.params.update({k: v for k, v in parameters.items() if v is not None})

    def _rsi(self, s: "pd.Series", period: int) -> "pd.Series":
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    def compute_signals(self, prices, volumes, timestamps):
        p = self.params if hasattr(self, "params") else {}
        bb_period = int(p.get("bb_period", 20))
        bb_std    = float(p.get("bb_std", 2.0))
        rsi_period = int(p.get("rsi_period", 14))
        oversold   = float(p.get("rsi_oversold", 30.0))
        overbought = float(p.get("rsi_overbought", 70.0))

        s = pd.Series(prices)
        sma   = s.rolling(bb_period).mean()
        std   = s.rolling(bb_period).std()
        upper = sma + bb_std * std
        lower = sma - bb_std * std
        rsi   = self._rsi(s, rsi_period)

        sig = np.zeros(len(prices), dtype=np.int8)
        # BUY: price below lower band AND RSI oversold
        sig[(s < lower) & (rsi < oversold)] = self.BUY
        # SELL: price above upper band AND RSI overbought
        sig[(s > upper) & (rsi > overbought)] = self.SELL
        # CLOSE: price returns to middle band while in a position
        sig[((s - sma).abs() / sma.replace(0, float("nan")) < 0.001)] = self.CLOSE
        return sig

    def update_parameters(self, parameters: dict) -> None:
        if not hasattr(self, "params"):
            self.params = {}
        self.params.update({k: v for k, v in parameters.items() if v is not None})

    def get_parameters(self) -> dict:
        return dict(self.params) if hasattr(self, "params") else {}

    def parameter_space(self) -> dict:
        return {
            "bb_period":    {"type": "int",   "min": 10,  "max": 100,  "default": 20},
            "bb_std":       {"type": "float", "min": 1.0, "max": 4.0,  "default": 2.0},
            "rsi_period":   {"type": "int",   "min": 5,   "max": 30,   "default": 14},
            "rsi_oversold": {"type": "float", "min": 20,  "max": 40,   "default": 30.0},
            "rsi_overbought":{"type": "float","min": 60,  "max": 80,   "default": 70.0},
        }

    def data_requirements(self) -> DataRequirements:
        return DataRequirements.momentum()

    def required_data_window(self) -> int:
        p = self.params if hasattr(self, "params") else {}
        return max(int(p.get("bb_period", 20)), int(p.get("rsi_period", 14))) + 10

    def get_risk_limits(self) -> RiskLimits:
        return RiskLimits(max_position_size=0.5, max_drawdown_pct=0.10, max_daily_loss=500.0)