"""
Example: RSI + MACD Confluence Strategy

Enters long when both RSI is oversold AND MACD has a bullish crossover.
Enters short when both RSI is overbought AND MACD has a bearish crossover.
Volume spike acts as an additional confirmation filter.

All indicator computation is vectorized via pandas - compute_signals() is
called ONCE with the full tick array.
"""

import numpy as np
import pandas as pd

from trading_platform import BaseStrategy, DataRequirements, RiskLimits


class Strategy(BaseStrategy):
    """RSI + MACD confluence with volume confirmation."""

    def name(self) -> str:
        return "RSI_MACD_Confluence"

    def description(self) -> str:
        return "RSI + MACD confluence with volume spike confirmation"

    def initialize(self, parameters: dict) -> None:
        self.params = {
            "rsi_period": 14,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "volume_spike_mult": 1.5,
            "atr_period": 14,
        }
        self.params.update({k: v for k, v in parameters.items() if v is not None})

    def _rsi(self, s: "pd.Series", period: int) -> "pd.Series":
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    def _macd(self, s: "pd.Series", fast: int, slow: int, signal: int):
        ema_fast   = s.ewm(span=fast, adjust=False).mean()
        ema_slow   = s.ewm(span=slow, adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    def compute_signals(self, prices, volumes, timestamps):
        p = self.params if hasattr(self, "params") else {}
        rsi_period   = int(p.get("rsi_period", 14))
        oversold     = float(p.get("rsi_oversold", 30.0))
        overbought   = float(p.get("rsi_overbought", 70.0))
        macd_fast    = int(p.get("macd_fast", 12))
        macd_slow    = int(p.get("macd_slow", 26))
        macd_signal  = int(p.get("macd_signal", 9))
        vol_mult     = float(p.get("volume_spike_mult", 1.5))
        atr_period   = int(p.get("atr_period", 14))

        s   = pd.Series(prices)
        vol = pd.Series(volumes)

        rsi = self._rsi(s, rsi_period)
        macd_line, signal_line = self._macd(s, macd_fast, macd_slow, macd_signal)

        # MACD crossover detection
        macd_cross_up   = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        # Volume spike filter
        avg_vol     = vol.rolling(atr_period).mean()
        vol_spike   = vol > avg_vol * vol_mult

        sig = np.zeros(len(prices), dtype=np.int8)
        # BUY: RSI oversold + bullish MACD cross + volume spike
        sig[macd_cross_up  & (rsi < oversold)  & vol_spike] = self.BUY
        # SELL: RSI overbought + bearish MACD cross + volume spike
        sig[macd_cross_down & (rsi > overbought) & vol_spike] = self.SELL
        return sig

    def update_parameters(self, parameters: dict) -> None:
        if not hasattr(self, "params"):
            self.params = {}
        self.params.update({k: v for k, v in parameters.items() if v is not None})

    def get_parameters(self) -> dict:
        return dict(self.params) if hasattr(self, "params") else {}

    def parameter_space(self) -> dict:
        return {
            "rsi_period":        {"type": "int",   "min": 7,   "max": 28,  "default": 14},
            "rsi_oversold":      {"type": "float", "min": 20,  "max": 40,  "default": 30.0},
            "rsi_overbought":    {"type": "float", "min": 60,  "max": 80,  "default": 70.0},
            "macd_fast":         {"type": "int",   "min": 5,   "max": 20,  "default": 12},
            "macd_slow":         {"type": "int",   "min": 15,  "max": 50,  "default": 26},
            "macd_signal":       {"type": "int",   "min": 5,   "max": 15,  "default": 9},
            "volume_spike_mult": {"type": "float", "min": 1.2, "max": 3.0, "default": 1.5},
        }

    def data_requirements(self) -> DataRequirements:
        return DataRequirements.momentum()

    def required_data_window(self) -> int:
        p = self.params if hasattr(self, "params") else {}
        return int(p.get("macd_slow", 26)) + int(p.get("macd_signal", 9)) + 10

    def get_risk_limits(self) -> RiskLimits:
        return RiskLimits(max_position_size=0.8, max_drawdown_pct=0.12)