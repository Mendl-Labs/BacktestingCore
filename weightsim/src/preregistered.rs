//! Pre-registered live-realistic execution defaults (W7.4, council Ruling R10; plan owner decision 12).
//!
//! PRE-REGISTERED 2026-10-02. These numbers are stated BEFORE any live-realistic figure is computed and are
//! changeable ONLY by a written amendment (bump [`LIVE_REALISTIC_CONFIG_VERSION`], record the old and new values and
//! the reason); never by a code change that "tunes" them to a result (Ruling 12: no timing or cost variant is chosen
//! by Sharpe). The delays are R10's (ETF 1 bar, crypto 0 bars). The slippage values below are the INITIAL
//! PLACEHOLDERS awaiting the owner's confirmation (decision 12 asks for the crypto number in particular); confirming
//! them is amendment v1's closing act, changing them is amendment v2.

use crate::execution::ExecutionModel;

/// Version of this table, recorded on every live-realistic result. Bumped by amendment only.
pub const LIVE_REALISTIC_CONFIG_VERSION: &str = "live_realistic_v1_2026-10-02";

/// ETF trend sleeves (monthly, decided at the last close of the month): executed at the NEXT close (R10: 1 bar).
pub const ETF_LIVE_DELAY_BARS: u32 = 1;
/// ETF trend sleeves: slippage on traded notional, bps per side. PLACEHOLDER pending owner confirmation.
pub const ETF_LIVE_SLIPPAGE_BPS: f64 = 5.0;

/// Crypto trend sleeves (daily, 24/7 venues): executed at the same close (R10: 0 bars).
pub const CRYPTO_LIVE_DELAY_BARS: u32 = 0;
/// Crypto trend sleeves: slippage on traded notional, bps per side. PLACEHOLDER pending owner confirmation.
pub const CRYPTO_LIVE_SLIPPAGE_BPS: f64 = 10.0;

/// The sleeve classes the table knows. A rule is mapped to its class by the crate that knows the rules
/// (`weightsim-rules`), never by this one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleeveClass {
    EtfTrend,
    CryptoTrend,
}

impl SleeveClass {
    pub fn label(self) -> &'static str {
        match self {
            SleeveClass::EtfTrend => "etf_trend",
            SleeveClass::CryptoTrend => "crypto_trend",
        }
    }
}

/// The pre-registered live-realistic execution model of a sleeve class.
pub const fn live_realistic_default_for(sleeve: SleeveClass) -> ExecutionModel {
    match sleeve {
        SleeveClass::EtfTrend => ExecutionModel::new(ETF_LIVE_DELAY_BARS, ETF_LIVE_SLIPPAGE_BPS),
        SleeveClass::CryptoTrend => ExecutionModel::new(CRYPTO_LIVE_DELAY_BARS, CRYPTO_LIVE_SLIPPAGE_BPS),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_r10_with_the_placeholder_slippage() {
        let etf = live_realistic_default_for(SleeveClass::EtfTrend);
        assert_eq!((etf.delay_bars, etf.slippage_bps), (1, 5.0));
        let cry = live_realistic_default_for(SleeveClass::CryptoTrend);
        assert_eq!((cry.delay_bars, cry.slippage_bps), (0, 10.0));
        assert!(!etf.is_certification() && !cry.is_certification());
        assert_eq!(LIVE_REALISTIC_CONFIG_VERSION, "live_realistic_v1_2026-10-02");
    }
}
