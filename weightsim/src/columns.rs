//! [`SeriesColumns`]: the full-resolution result series of a run as plain, owned columns, exactly the fields the
//! series digest covers (design 3.6 and 3.7, stage T4 slice C1).
//!
//! A consumer that stores a run (the Engine's series codec) serialises these columns and nothing else. Later, from the
//! decoded columns alone, [`SeriesColumns::digest`] recomputes the SHA-256 the simulator stamped on the run
//! ([`crate::SimResult::series_sha256`]), bit for bit: `SimResult::series_sha256` IS `SeriesColumns::from(&res).digest()`
//! (the simulator calls the same function), so a stored run can be checked without trusting any stored number.
//!
//! What the digest covers (choice C13 of the crate documentation): rule id and impl version, symbols, cost-model id,
//! metric-definition label, and per bar the date, nine scalar columns, the decision and refusal flags, and the three
//! per-asset matrices, every float by its bit pattern. What it does NOT cover, on purpose: the counted window, the
//! signal flips, the refusal messages and the exposure statistics. Those are derived from the columns (or from the run
//! configuration) and are recomputed by the verifier, never read from storage.
//!
//! Flat matrices are `[bar * n_assets + asset]`, as in [`crate::SimResult`].

use std::fmt;

use crate::date::Date;
use crate::sim::{digest_columns, SimResult};

/// The digest input of one run, owned. Field names and meanings are those of [`SimResult`].
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesColumns {
    pub rule_id: String,
    pub rule_impl_version: String,
    pub symbols: Vec<String>,
    pub cost_model_id: String,
    pub metric_definitions: String,
    pub dates: Vec<Date>,
    pub ret: Vec<f64>,
    pub ret_pre_cost: Vec<f64>,
    pub equity: Vec<f64>,
    pub cash: Vec<f64>,
    pub cost: Vec<f64>,
    pub traded_notional: Vec<f64>,
    pub financing: Vec<f64>,
    pub gross_exposure: Vec<f64>,
    pub net_exposure: Vec<f64>,
    pub decision: Vec<bool>,
    pub refused: Vec<bool>,
    pub target_weights: Vec<f64>,
    pub held_weights: Vec<f64>,
    pub units: Vec<f64>,
}

/// Why a set of columns cannot be a run's series (checked before anything is indexed or hashed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnsError {
    /// No bars or no symbols.
    Empty,
    /// A column does not have the length the dates and symbols imply.
    LengthMismatch { column: &'static str, expected: usize, found: usize },
    /// The dates are not strictly ascending; `bar` is the first bar that is not after its predecessor.
    DatesNotAscending { bar: usize },
}

impl fmt::Display for ColumnsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnsError::Empty => write!(f, "no bars or no symbols"),
            ColumnsError::LengthMismatch { column, expected, found } => {
                write!(f, "column `{column}` has {found} values, expected {expected}")
            }
            ColumnsError::DatesNotAscending { bar } => write!(f, "dates not strictly ascending at bar {bar}"),
        }
    }
}

impl std::error::Error for ColumnsError {}

impl SeriesColumns {
    pub fn n_bars(&self) -> usize {
        self.dates.len()
    }

    pub fn n_assets(&self) -> usize {
        self.symbols.len()
    }

    /// Row `t` of a flat `[bar * n_assets + asset]` matrix.
    pub fn row<'a>(&self, flat: &'a [f64], t: usize) -> &'a [f64] {
        let k = self.n_assets();
        &flat[t * k..(t + 1) * k]
    }

    /// Every column has the length the dates and symbols imply and the dates ascend. [`SeriesColumns::digest`] calls
    /// this first, so it never indexes out of range on malformed (for instance truncated or hand-edited) input.
    pub fn validate_shape(&self) -> Result<(), ColumnsError> {
        let n = self.n_bars();
        let k = self.n_assets();
        if n == 0 || k == 0 {
            return Err(ColumnsError::Empty);
        }
        let per_bar: [(&'static str, usize); 11] = [
            ("ret", self.ret.len()),
            ("ret_pre_cost", self.ret_pre_cost.len()),
            ("equity", self.equity.len()),
            ("cash", self.cash.len()),
            ("cost", self.cost.len()),
            ("traded_notional", self.traded_notional.len()),
            ("financing", self.financing.len()),
            ("gross_exposure", self.gross_exposure.len()),
            ("net_exposure", self.net_exposure.len()),
            ("decision", self.decision.len()),
            ("refused", self.refused.len()),
        ];
        for (column, found) in per_bar {
            if found != n {
                return Err(ColumnsError::LengthMismatch { column, expected: n, found });
            }
        }
        let per_cell: [(&'static str, usize); 3] = [
            ("target_weights", self.target_weights.len()),
            ("held_weights", self.held_weights.len()),
            ("units", self.units.len()),
        ];
        for (column, found) in per_cell {
            if found != n * k {
                return Err(ColumnsError::LengthMismatch { column, expected: n * k, found });
            }
        }
        for t in 1..n {
            if self.dates[t] <= self.dates[t - 1] {
                return Err(ColumnsError::DatesNotAscending { bar: t });
            }
        }
        Ok(())
    }

    /// The first non-finite float, as `(column, bar)`, scanning columns in a fixed order (per-bar scalars first, then
    /// the matrices). A genuine run never contains one (the simulator refuses non-finite weights and non-positive
    /// equity). Malformed shapes are reported by [`SeriesColumns::validate_shape`], not here; call that first.
    pub fn first_non_finite(&self) -> Option<(&'static str, usize)> {
        let k = self.n_assets().max(1);
        let scalars: [(&'static str, &[f64]); 9] = [
            ("ret", &self.ret),
            ("ret_pre_cost", &self.ret_pre_cost),
            ("equity", &self.equity),
            ("cash", &self.cash),
            ("cost", &self.cost),
            ("traded_notional", &self.traded_notional),
            ("financing", &self.financing),
            ("gross_exposure", &self.gross_exposure),
            ("net_exposure", &self.net_exposure),
        ];
        for (name, col) in scalars {
            if let Some(t) = col.iter().position(|v| !v.is_finite()) {
                return Some((name, t));
            }
        }
        let matrices: [(&'static str, &[f64]); 3] =
            [("target_weights", &self.target_weights), ("held_weights", &self.held_weights), ("units", &self.units)];
        for (name, col) in matrices {
            if let Some(i) = col.iter().position(|v| !v.is_finite()) {
                return Some((name, i / k));
            }
        }
        None
    }

    /// The series digest: SHA-256 (lower-case hex) over the canonical byte layout the simulator uses, so that
    /// `SeriesColumns::from(&run).digest() == Ok(run.series_sha256)` for every run. `Err` for malformed columns.
    pub fn digest(&self) -> Result<String, ColumnsError> {
        self.validate_shape()?;
        Ok(digest_columns(self))
    }
}

impl From<&SimResult> for SeriesColumns {
    fn from(r: &SimResult) -> SeriesColumns {
        SeriesColumns {
            rule_id: r.rule_id.clone(),
            rule_impl_version: r.rule_impl_version.clone(),
            symbols: r.symbols.clone(),
            cost_model_id: r.cost_model_id.to_string(),
            metric_definitions: r.metric_definitions.to_string(),
            dates: r.dates.clone(),
            ret: r.ret.clone(),
            ret_pre_cost: r.ret_pre_cost.clone(),
            equity: r.equity.clone(),
            cash: r.cash.clone(),
            cost: r.cost.clone(),
            traded_notional: r.traded_notional.clone(),
            financing: r.financing.clone(),
            gross_exposure: r.gross_exposure.clone(),
            net_exposure: r.net_exposure.clone(),
            decision: r.decision.clone(),
            refused: r.refused.clone(),
            target_weights: r.target_weights.clone(),
            held_weights: r.held_weights.clone(),
            units: r.units.clone(),
        }
    }
}

impl SimResult {
    /// The digest-covered series of this run as owned columns (what a run store serialises).
    pub fn series_columns(&self) -> SeriesColumns {
        SeriesColumns::from(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> SeriesColumns {
        let d = |n: i64| Date::from_days_since_epoch(18_000 + n);
        SeriesColumns {
            rule_id: "r".into(),
            rule_impl_version: "v".into(),
            symbols: vec!["A".into(), "B".into()],
            cost_model_id: "zero".into(),
            metric_definitions: "answer_key_v1".into(),
            dates: vec![d(0), d(1), d(2)],
            ret: vec![0.0, 0.01, -0.02],
            ret_pre_cost: vec![0.0, 0.01, -0.02],
            equity: vec![1.0, 1.01, 0.99],
            cash: vec![1.0; 3],
            cost: vec![0.0; 3],
            traded_notional: vec![0.0; 3],
            financing: vec![0.0; 3],
            gross_exposure: vec![0.0; 3],
            net_exposure: vec![0.0; 3],
            decision: vec![false, true, true],
            refused: vec![false; 3],
            target_weights: vec![0.0; 6],
            held_weights: vec![0.0; 6],
            units: vec![0.0; 6],
        }
    }

    #[test]
    fn a_well_formed_set_of_columns_has_a_digest_and_every_malformation_is_an_error_not_a_panic() {
        let ok = tiny();
        assert_eq!(ok.digest().unwrap().len(), 64);
        assert_eq!(ok.digest(), ok.clone().digest());

        let mut c = tiny();
        c.ret.pop();
        assert_eq!(c.digest(), Err(ColumnsError::LengthMismatch { column: "ret", expected: 3, found: 2 }));
        let mut c = tiny();
        c.units.push(0.0);
        assert_eq!(c.digest(), Err(ColumnsError::LengthMismatch { column: "units", expected: 6, found: 7 }));
        let mut c = tiny();
        c.dates.swap(1, 2);
        assert_eq!(c.digest(), Err(ColumnsError::DatesNotAscending { bar: 2 }));
        let mut c = tiny();
        c.symbols.clear();
        assert_eq!(c.digest(), Err(ColumnsError::Empty));
    }

    #[test]
    fn non_finite_values_are_located() {
        let mut c = tiny();
        assert_eq!(c.first_non_finite(), None);
        c.held_weights[5] = f64::NAN;
        assert_eq!(c.first_non_finite(), Some(("held_weights", 2)));
        c.equity[1] = f64::INFINITY;
        assert_eq!(c.first_non_finite(), Some(("equity", 1)));
    }
}
