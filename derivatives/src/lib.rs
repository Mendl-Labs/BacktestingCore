//! Derivative instrument types and market data structures.
//!
//! This crate is the shared foundation for options, futures, and perpetuals
//! across the BacktestingEngine. All other derivative-aware crates depend on
//! this one so that types are defined exactly once.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Instrument classification
// ─────────────────────────────────────────────

/// Classifies a financial instrument by its derivative structure.
///
/// Perpetuals are futures without an expiry that reset via a periodic funding
/// rate mechanism (Deribit, Binance-style).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InstrumentKind {
    /// Plain spot asset (e.g. BTC-SPOT).
    Spot,
    /// Perpetual swap — no expiry, settles via funding rate.
    Perpetual,
    /// Dated futures contract.
    Future {
        expiry: DateTime<Utc>,
    },
    /// European call option.
    Call {
        strike: f64,
        expiry: DateTime<Utc>,
    },
    /// European put option.
    Put {
        strike: f64,
        expiry: DateTime<Utc>,
    },
}

impl InstrumentKind {
    /// Returns the expiry, if the instrument has one.
    pub fn expiry(&self) -> Option<DateTime<Utc>> {
        match self {
            InstrumentKind::Future { expiry }
            | InstrumentKind::Call { expiry, .. }
            | InstrumentKind::Put { expiry, .. } => Some(*expiry),
            _ => None,
        }
    }

    /// Returns the strike price, if applicable.
    pub fn strike(&self) -> Option<f64> {
        match self {
            InstrumentKind::Call { strike, .. } | InstrumentKind::Put { strike, .. } => {
                Some(*strike)
            }
            _ => None,
        }
    }

    /// True for calls or puts.
    pub fn is_option(&self) -> bool {
        matches!(
            self,
            InstrumentKind::Call { .. } | InstrumentKind::Put { .. }
        )
    }

    /// True for dated futures or perpetuals.
    pub fn is_future_or_perp(&self) -> bool {
        matches!(
            self,
            InstrumentKind::Future { .. } | InstrumentKind::Perpetual
        )
    }

    /// Seconds to expiry from `now`. Returns `None` for Spot/Perpetual.
    pub fn time_to_expiry(&self, now: DateTime<Utc>) -> Option<f64> {
        self.expiry().map(|exp| {
            let delta = exp.signed_duration_since(now);
            delta.num_seconds() as f64 / (365.25 * 24.0 * 3600.0)
        })
    }
}

// ─────────────────────────────────────────────
// Derivative contract metadata
// ─────────────────────────────────────────────

/// Static metadata that describes a single derivative contract.
/// Attach this to positions and signals to carry instrument context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivativeMetadata {
    /// Canonical exchange symbol (e.g. `"BTC-25MAR26-100000-C"`).
    pub symbol: String,
    /// Underlying asset (e.g. `"BTC"`).
    pub underlying: String,
    /// Instrument classification.
    pub instrument_kind: InstrumentKind,
    /// How many units of underlying one contract represents.
    /// For Deribit BTC options this is typically 1.0; for futures it may be 10.
    pub contract_multiplier: f64,
    /// Settlement / margin currency (e.g. `"USD"`, `"BTC"`).
    pub settlement_currency: String,
    /// Exchange on which this contract trades (e.g. `"deribit"`).
    pub exchange: String,
}

impl DerivativeMetadata {
    pub fn new(
        symbol: impl Into<String>,
        underlying: impl Into<String>,
        instrument_kind: InstrumentKind,
        contract_multiplier: f64,
        settlement_currency: impl Into<String>,
        exchange: impl Into<String>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            underlying: underlying.into(),
            instrument_kind,
            contract_multiplier,
            settlement_currency: settlement_currency.into(),
            exchange: exchange.into(),
        }
    }
}

// ─────────────────────────────────────────────
// Greeks
// ─────────────────────────────────────────────

/// First- and second-order option price sensitivities (Black-Scholes Greeks).
///
/// All values are per-contract (before multiplier scaling).
/// Use [`Greeks::scale`] when aggregating across a portfolio.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Greeks {
    /// ∂Price/∂Spot  — first-order price sensitivity to the underlying.
    pub delta: f64,
    /// ∂²Price/∂Spot² — rate of change of delta.
    pub gamma: f64,
    /// ∂Price/∂t  — time decay per calendar year (usually negative for long options).
    pub theta: f64,
    /// ∂Price/∂σ  — sensitivity to implied volatility (per 1.0 = 100 vol points).
    pub vega: f64,
    /// ∂Price/∂r  — sensitivity to interest rate (per 1.0 = 100 bps).
    pub rho: f64,
}

impl Greeks {
    pub fn zero() -> Self {
        Self::default()
    }

    /// Scale all sensitivities by a multiplier (e.g. contract size or position quantity).
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            delta: self.delta * factor,
            gamma: self.gamma * factor,
            theta: self.theta * factor,
            vega: self.vega * factor,
            rho: self.rho * factor,
        }
    }

    /// Component-wise addition.
    pub fn add(&self, other: &Greeks) -> Self {
        Self {
            delta: self.delta + other.delta,
            gamma: self.gamma + other.gamma,
            theta: self.theta + other.theta,
            vega: self.vega + other.vega,
            rho: self.rho + other.rho,
        }
    }
}

impl std::ops::Add for Greeks {
    type Output = Greeks;
    fn add(self, rhs: Greeks) -> Greeks {
        Greeks {
            delta: self.delta + rhs.delta,
            gamma: self.gamma + rhs.gamma,
            theta: self.theta + rhs.theta,
            vega: self.vega + rhs.vega,
            rho: self.rho + rhs.rho,
        }
    }
}

impl std::ops::AddAssign for Greeks {
    fn add_assign(&mut self, rhs: Greeks) {
        self.delta += rhs.delta;
        self.gamma += rhs.gamma;
        self.theta += rhs.theta;
        self.vega += rhs.vega;
        self.rho += rhs.rho;
    }
}

// ─────────────────────────────────────────────
// Implied Volatility Surface
// ─────────────────────────────────────────────

/// A single point on the IV surface (one strike × expiry combination).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvPoint {
    pub strike: f64,
    pub expiry: DateTime<Utc>,
    /// Mid implied volatility (annualised, e.g. 0.80 = 80%).
    pub iv: f64,
    /// Bid-side IV.
    pub bid_iv: f64,
    /// Ask-side IV.
    pub ask_iv: f64,
}

/// Snapshot of the full implied volatility surface for one underlying.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvSurface {
    pub underlying: String,
    pub timestamp: DateTime<Utc>,
    pub points: Vec<IvPoint>,
}

impl IvSurface {
    /// Look up the closest IV point for a given strike and expiry (nearest-neighbour).
    pub fn nearest_iv(&self, strike: f64, expiry: DateTime<Utc>) -> Option<&IvPoint> {
        self.points.iter().min_by(|a, b| {
            let da = (a.strike - strike).abs()
                + (a.expiry.signed_duration_since(expiry).num_seconds() as f64 / 86400.0).abs();
            let db = (b.strike - strike).abs()
                + (b.expiry.signed_duration_since(expiry).num_seconds() as f64 / 86400.0).abs();
            da.partial_cmp(&db).unwrap()
        })
    }
}

// ─────────────────────────────────────────────
// Futures Curve
// ─────────────────────────────────────────────

/// A single row in the futures term structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuturesCurvePoint {
    pub expiry: DateTime<Utc>,
    /// Mark price of this futures contract.
    pub mark_price: f64,
    /// Annualised basis relative to spot (futures - spot) / spot.
    pub basis_pct: f64,
    pub open_interest: f64,
}

/// Term structure of futures prices for one underlying.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuturesCurve {
    pub underlying: String,
    pub spot_price: f64,
    pub timestamp: DateTime<Utc>,
    pub points: Vec<FuturesCurvePoint>,
}

impl FuturesCurve {
    /// Returns the front-month (nearest expiry) point, if any.
    pub fn front_month(&self) -> Option<&FuturesCurvePoint> {
        self.points
            .iter()
            .min_by_key(|p| p.expiry.timestamp())
    }
}

// ─────────────────────────────────────────────
// Funding Rate
// ─────────────────────────────────────────────

/// Funding rate for a perpetual swap instrument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    /// Perpetual symbol (e.g. `"BTC-PERPETUAL"`).
    pub symbol: String,
    /// Current period rate as a decimal (e.g. 0.0001 = 0.01%).
    pub rate: f64,
    /// Next settlement timestamp.
    pub next_settlement: DateTime<Utc>,
    /// Predicted rate for the next period (exchange-provided estimate), if available.
    pub predicted_rate: Option<f64>,
    /// Annualised rate (rate × periods_per_year), if calculable.
    pub annualised_rate: Option<f64>,
}

// ─────────────────────────────────────────────
// Derivative market data snapshot
// ─────────────────────────────────────────────

/// Combined real-time snapshot for a single derivative instrument.
/// Produced by the DataEngine Deribit pipeline and fed into StrategyContext.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivativeSnapshot {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub instrument: DerivativeMetadata,
    /// Mid market price.
    pub mark_price: f64,
    /// Underlying index price.
    pub index_price: f64,
    /// Implied volatility (options/perps), if available.
    pub implied_vol: Option<f64>,
    /// Snapshot of Black-Scholes Greeks, if available.
    pub greeks: Option<Greeks>,
    /// Open interest in contract units.
    pub open_interest: f64,
    /// Current funding rate (perpetuals only).
    pub funding_rate: Option<FundingRate>,
    /// Best bid price.
    pub best_bid: Option<f64>,
    /// Best ask price.
    pub best_ask: Option<f64>,
}

// ─────────────────────────────────────────────
// Parsing: Deribit instrument name → InstrumentKind
// ─────────────────────────────────────────────

/// Parse a Deribit instrument name into an [`InstrumentKind`].
///
/// Deribit uses the following naming conventions:
/// - Perpetuals:  `BTC-PERPETUAL`
/// - Futures:     `BTC-25MAR26`
/// - Options:     `BTC-25MAR26-100000-C` / `...-P`
///
/// Returns `None` if the name does not match a known pattern.
pub fn parse_deribit_instrument(name: &str) -> Option<InstrumentKind> {
    let parts: Vec<&str> = name.split('-').collect();
    if parts.len() < 2 {
        return None;
    }

    // Perpetual: two parts, second == "PERPETUAL"
    if parts.len() == 2 && parts[1].eq_ignore_ascii_case("PERPETUAL") {
        return Some(InstrumentKind::Perpetual);
    }

    // Futures or option — second part is the expiry date
    let expiry = parse_deribit_date(parts[1])?;

    match parts.len() {
        // Futures: two parts (UNDERLYING-DDMMMYY)
        2 => Some(InstrumentKind::Future { expiry }),

        // Options: four parts (UNDERLYING-DDMMMYY-STRIKE-C/P)
        4 => {
            let strike: f64 = parts[2].parse().ok()?;
            match parts[3].to_uppercase().as_str() {
                "C" => Some(InstrumentKind::Call { strike, expiry }),
                "P" => Some(InstrumentKind::Put { strike, expiry }),
                _ => None,
            }
        }

        _ => None,
    }
}

/// Parse a Deribit expiry date string (`25MAR26`, `25MAR2026`) into a UTC DateTime.
/// Expiry is at 08:00 UTC, as per Deribit conventions.
fn parse_deribit_date(s: &str) -> Option<DateTime<Utc>> {
    use chrono::{NaiveDate, TimeZone};

    // Normalize: "25MAR26" → length 7, "25MAR2026" → length 9
    let s_upper = s.to_uppercase();
    if s_upper.len() < 5 {
        return None;
    }
    let (day_str, rest) = s_upper.split_at(2);
    let day: u32 = day_str.parse().ok()?;

    // Extract month abbreviation (next 3 chars)
    if rest.len() < 3 {
        return None;
    }
    let (mon_str, year_str) = rest.split_at(3);

    let month: u32 = match mon_str {
        "JAN" => 1,
        "FEB" => 2,
        "MAR" => 3,
        "APR" => 4,
        "MAY" => 5,
        "JUN" => 6,
        "JUL" => 7,
        "AUG" => 8,
        "SEP" => 9,
        "OCT" => 10,
        "NOV" => 11,
        "DEC" => 12,
        _ => return None,
    };

    // Year: "26" → 2026, "2026" → 2026
    let year: i32 = match year_str.len() {
        2 => 2000 + year_str.parse::<i32>().ok()?,
        4 => year_str.parse().ok()?,
        _ => return None,
    };

    let naive = NaiveDate::from_ymd_opt(year, month, day)?
        .and_hms_opt(8, 0, 0)?;
    Some(Utc.from_utc_datetime(&naive))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike;

    #[test]
    fn parse_perpetual() {
        assert_eq!(
            parse_deribit_instrument("BTC-PERPETUAL"),
            Some(InstrumentKind::Perpetual)
        );
    }

    #[test]
    fn parse_future() {
        let kind = parse_deribit_instrument("BTC-25MAR26").unwrap();
        assert!(matches!(kind, InstrumentKind::Future { .. }));
        if let InstrumentKind::Future { expiry } = kind {
            assert_eq!(expiry.year(), 2026);
            assert_eq!(expiry.month(), 3);
            assert_eq!(expiry.day(), 25);
        }
    }

    #[test]
    fn parse_call_option() {
        let kind = parse_deribit_instrument("BTC-25MAR26-100000-C").unwrap();
        assert!(matches!(kind, InstrumentKind::Call { strike, .. } if strike == 100_000.0));
    }

    #[test]
    fn parse_put_option() {
        let kind = parse_deribit_instrument("ETH-28JUN26-3000-P").unwrap();
        assert!(matches!(kind, InstrumentKind::Put { strike, .. } if strike == 3_000.0));
    }

    #[test]
    fn parse_unknown_returns_none() {
        assert!(parse_deribit_instrument("NOT-A-REAL-INSTRUMENT-X").is_none());
    }

    #[test]
    fn time_to_expiry_positive() {
        use chrono::Duration;
        let expiry = Utc::now() + Duration::days(30);
        let kind = InstrumentKind::Call {
            strike: 50000.0,
            expiry,
        };
        let tte = kind.time_to_expiry(Utc::now()).unwrap();
        assert!(tte > 0.0 && tte < 0.1); // ~30/365 years
    }

    #[test]
    fn greeks_add_and_scale() {
        let g = Greeks {
            delta: 0.5,
            gamma: 0.02,
            theta: -10.0,
            vega: 50.0,
            rho: 0.1,
        };
        let doubled = g.scale(2.0);
        assert_eq!(doubled.delta, 1.0);
        assert_eq!(doubled.theta, -20.0);

        let sum = g.add(&doubled);
        assert_eq!(sum.delta, 1.5);
    }
}
