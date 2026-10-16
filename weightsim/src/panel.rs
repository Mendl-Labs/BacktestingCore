//! The price panel the simulator runs on, the causal `HistoryView` handed to rules, and the fixture loader.
//!
//! Design S-1: the simulation calendar is the JOINT calendar (dates on which every instrument of the rule's universe
//! has a bar), strictly ascending, never interpolated or forward-filled. Design 2.4(1): a rule only ever sees a
//! [`HistoryView`], whose slices are cut to `..=t` (the same memory as the panel, not a copy), and whose constructor is
//! `pub(crate)`, so only the simulator can build one.

use crate::date::Date;
use crate::sha256::sha256_hex;
use std::fmt;

/// Errors from building or loading a panel. There is no repair path: bad input is an error.
#[derive(Clone, Debug, PartialEq)]
pub enum PanelError {
    Empty(String),
    Shape(String),
    Ordering { symbol: String, date: Date },
    BadPrice { symbol: String, date: Date, value: f64 },
    Csv { line: usize, message: String },
    MissingSymbol(String),
    NoCommonDates,
    Sha256Mismatch { expected: String, actual: String },
}

impl fmt::Display for PanelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelError::Empty(m) => write!(f, "empty panel: {m}"),
            PanelError::Shape(m) => write!(f, "panel shape error: {m}"),
            PanelError::Ordering { symbol, date } => {
                write!(f, "{symbol}: dates not strictly ascending at {date}")
            }
            PanelError::BadPrice { symbol, date, value } => {
                write!(f, "{symbol} {date}: price {value} is not finite and > 0")
            }
            PanelError::Csv { line, message } => write!(f, "csv line {line}: {message}"),
            PanelError::MissingSymbol(s) => write!(f, "symbol {s} has no rows"),
            PanelError::NoCommonDates => write!(f, "no date is common to all requested symbols"),
            PanelError::Sha256Mismatch { expected, actual } => {
                write!(f, "fixture sha256 mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for PanelError {}

/// Aligned daily closes for a fixed, ordered universe. Column `i` is `symbols[i]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Panel {
    symbols: Vec<String>,
    dates: Vec<Date>,
    closes: Vec<Vec<f64>>, // [asset][bar]
}

impl Panel {
    /// Validated constructor: equal lengths, strictly ascending dates, every close finite and > 0.
    pub fn new(symbols: Vec<String>, dates: Vec<Date>, closes: Vec<Vec<f64>>) -> Result<Panel, PanelError> {
        if symbols.is_empty() {
            return Err(PanelError::Empty("no symbols".into()));
        }
        if dates.is_empty() {
            return Err(PanelError::Empty("no dates".into()));
        }
        if closes.len() != symbols.len() {
            return Err(PanelError::Shape(format!("{} symbols but {} price columns", symbols.len(), closes.len())));
        }
        for (s, c) in symbols.iter().zip(&closes) {
            if c.len() != dates.len() {
                return Err(PanelError::Shape(format!("{s}: {} closes for {} dates", c.len(), dates.len())));
            }
        }
        for w in dates.windows(2) {
            if w[1] <= w[0] {
                return Err(PanelError::Ordering { symbol: "<calendar>".into(), date: w[1] });
            }
        }
        for (s, c) in symbols.iter().zip(&closes) {
            for (i, &v) in c.iter().enumerate() {
                if !(v.is_finite() && v > 0.0) {
                    return Err(PanelError::BadPrice { symbol: s.clone(), date: dates[i], value: v });
                }
            }
        }
        Ok(Panel { symbols, dates, closes })
    }

    /// Inner join of per-symbol (date, close) series onto the joint calendar (design S-1).
    /// Each series must be strictly ascending in date with finite positive closes.
    pub fn inner_join(series: Vec<(String, Vec<(Date, f64)>)>) -> Result<Panel, PanelError> {
        if series.is_empty() {
            return Err(PanelError::Empty("no series".into()));
        }
        for (s, rows) in &series {
            if rows.is_empty() {
                return Err(PanelError::MissingSymbol(s.clone()));
            }
            for (i, &(d, v)) in rows.iter().enumerate() {
                if i > 0 && d <= rows[i - 1].0 {
                    return Err(PanelError::Ordering { symbol: s.clone(), date: d });
                }
                if !(v.is_finite() && v > 0.0) {
                    return Err(PanelError::BadPrice { symbol: s.clone(), date: d, value: v });
                }
            }
        }
        // Sequential k-way intersection of sorted date lists.
        let mut cursors = vec![0usize; series.len()];
        let mut dates = Vec::new();
        let mut closes: Vec<Vec<f64>> = vec![Vec::new(); series.len()];
        'outer: loop {
            let mut target = series[0].1.get(cursors[0]).map(|r| r.0);
            for (k, (_, rows)) in series.iter().enumerate() {
                match rows.get(cursors[k]) {
                    None => break 'outer,
                    Some(&(d, _)) => {
                        if target.is_none_or(|t| d > t) {
                            target = Some(d);
                        }
                    }
                }
            }
            let target = target.expect("series is non-empty");
            let mut all_equal = true;
            for (k, (_, rows)) in series.iter().enumerate() {
                while cursors[k] < rows.len() && rows[cursors[k]].0 < target {
                    cursors[k] += 1;
                }
                match rows.get(cursors[k]) {
                    None => break 'outer,
                    Some(&(d, _)) => {
                        if d != target {
                            all_equal = false;
                        }
                    }
                }
            }
            if all_equal {
                dates.push(target);
                for (k, (_, rows)) in series.iter().enumerate() {
                    closes[k].push(rows[cursors[k]].1);
                    cursors[k] += 1;
                }
            }
        }
        if dates.is_empty() {
            return Err(PanelError::NoCommonDates);
        }
        let symbols = series.into_iter().map(|(s, _)| s).collect();
        Panel::new(symbols, dates, closes)
    }

    /// Parse the long-format fixture (`symbol,date_utc,close`) for exactly `symbols` (in that order) and join them
    /// onto the joint calendar. Rows of other symbols are skipped without being parsed. No silent repair: a
    /// duplicate or out-of-order date, an unparsable field or a non-positive close for a requested symbol is an error.
    pub fn from_long_csv(text: &str, symbols: &[&str]) -> Result<Panel, PanelError> {
        let mut lines = text.split('\n').enumerate();
        match lines.next() {
            Some((_, h)) if h.trim_end_matches('\r') == "symbol,date_utc,close" => {}
            _ => {
                return Err(PanelError::Csv {
                    line: 1,
                    message: "header must be exactly `symbol,date_utc,close`".into(),
                });
            }
        }
        let mut series: Vec<(String, Vec<(Date, f64)>)> =
            symbols.iter().map(|s| ((*s).to_string(), Vec::new())).collect();
        for (idx, raw) in lines {
            let line = raw.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let lineno = idx + 1;
            let mut parts = line.split(',');
            let sym = parts.next().unwrap_or("");
            let slot = match symbols.iter().position(|s| *s == sym) {
                Some(p) => p,
                None => continue,
            };
            let (d, c) = match (parts.next(), parts.next(), parts.next()) {
                (Some(d), Some(c), None) => (d, c),
                _ => return Err(PanelError::Csv { line: lineno, message: "expected exactly 3 fields".into() }),
            };
            let date = Date::parse(d).map_err(|e| PanelError::Csv { line: lineno, message: e.to_string() })?;
            let close: f64 = c
                .parse()
                .map_err(|_| PanelError::Csv { line: lineno, message: format!("cannot parse close `{c}`") })?;
            series[slot].1.push((date, close));
        }
        for (s, rows) in &series {
            if rows.is_empty() {
                return Err(PanelError::MissingSymbol(s.clone()));
            }
        }
        Panel::inner_join(series)
    }

    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }
    pub fn dates(&self) -> &[Date] {
        &self.dates
    }
    pub fn n_bars(&self) -> usize {
        self.dates.len()
    }
    pub fn n_assets(&self) -> usize {
        self.symbols.len()
    }
    /// Full close series of asset `i`.
    pub fn closes(&self, i: usize) -> &[f64] {
        &self.closes[i]
    }

    /// First `keep_bars` bars only.
    pub fn truncated(&self, keep_bars: usize) -> Panel {
        assert!(keep_bars >= 1 && keep_bars <= self.n_bars(), "keep_bars out of range");
        Panel {
            symbols: self.symbols.clone(),
            dates: self.dates[..keep_bars].to_vec(),
            closes: self.closes.iter().map(|c| c[..keep_bars].to_vec()).collect(),
        }
    }

    /// Same panel with the close of every bar at index >= `first_poisoned_bar` replaced by `f(asset, bar, old)`.
    /// Dates are untouched (design 2.4(4): date-only look-ahead is an input, not a leak). `f` should return finite,
    /// positive garbage so the simulation still runs; that is asserted.
    pub fn with_prices_replaced_from(
        &self,
        first_poisoned_bar: usize,
        mut f: impl FnMut(usize, usize, f64) -> f64,
    ) -> Panel {
        let mut out = self.clone();
        for (i, col) in out.closes.iter_mut().enumerate() {
            for (t, v) in col.iter_mut().enumerate().skip(first_poisoned_bar) {
                let nv = f(i, t, *v);
                assert!(nv.is_finite() && nv > 0.0, "poison values must be finite and positive");
                *v = nv;
            }
        }
        out
    }

    /// Same panel with the asset columns permuted: new column `j` is old column `order[j]`.
    pub fn with_asset_order(&self, order: &[usize]) -> Panel {
        assert_eq!(order.len(), self.n_assets());
        Panel {
            symbols: order.iter().map(|&i| self.symbols[i].clone()).collect(),
            dates: self.dates.clone(),
            closes: order.iter().map(|&i| self.closes[i].clone()).collect(),
        }
    }
}

/// Where prices come from. T1 has exactly one source: a pinned fixture. Live vendor sources are Stage T6 and are
/// deliberately absent here (this crate performs no I/O and makes no network calls).
pub enum PriceSource<'a> {
    /// CSV bytes in the long format `symbol,date_utc,close` plus the SHA-256 the bytes must have.
    Fixture { csv: &'a [u8], expected_sha256: &'a str },
}

impl PriceSource<'_> {
    /// Verify the pin, then parse. The hash check happens before any parsing.
    pub fn load(&self, symbols: &[&str]) -> Result<Panel, PanelError> {
        match self {
            PriceSource::Fixture { csv, expected_sha256 } => {
                let actual = sha256_hex(csv);
                if !actual.eq_ignore_ascii_case(expected_sha256) {
                    return Err(PanelError::Sha256Mismatch { expected: (*expected_sha256).to_string(), actual });
                }
                let text = std::str::from_utf8(csv)
                    .map_err(|_| PanelError::Csv { line: 0, message: "fixture is not valid UTF-8".into() })?;
                Panel::from_long_csv(text, symbols)
            }
        }
    }
}

/// Everything a rule may look at when deciding at the close of bar `t`: bars `0..=t` of every asset, and nothing else.
///
/// The slices alias the panel's memory (no copy), cut to `..=t`, so a rule cannot index a future bar because it is
/// not in the slice. The constructor is `pub(crate)`: only the simulator (and this crate's harnesses) can build one.
///
/// ```compile_fail,E0624
/// // Outside the crate a rule (or anything else) cannot construct a view of its own choosing.
/// fn cheat(p: &weightsim::Panel) -> usize {
///     weightsim::HistoryView::new(p, 1_000_000).len()
/// }
/// ```
#[derive(Clone, Copy)]
pub struct HistoryView<'a> {
    panel: &'a Panel,
    len: usize,
}

impl<'a> HistoryView<'a> {
    /// View of bars `0..=t`.
    pub(crate) fn new(panel: &'a Panel, t: usize) -> HistoryView<'a> {
        HistoryView { panel, len: t + 1 }
    }

    /// Number of bars visible (t + 1).
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn n_assets(&self) -> usize {
        self.panel.n_assets()
    }
    pub fn symbols(&self) -> &'a [String] {
        self.panel.symbols()
    }
    /// Dates `0..=t`; the last entry is the decision date.
    pub fn dates(&self) -> &'a [Date] {
        &self.panel.dates[..self.len]
    }
    /// Closes of asset `i` for bars `0..=t`.
    pub fn closes(&self, i: usize) -> &'a [f64] {
        &self.panel.closes[i][..self.len]
    }
    /// Decision date.
    pub fn date(&self) -> Date {
        self.panel.dates[self.len - 1]
    }
    /// Close of asset `i` at the decision bar.
    pub fn last_close(&self, i: usize) -> f64 {
        self.panel.closes[i][self.len - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    #[test]
    fn new_validates_shape_order_and_prices() {
        let s = vec!["A".to_string()];
        assert!(Panel::new(s.clone(), vec![d("2020-01-02"), d("2020-01-01")], vec![vec![1.0, 2.0]]).is_err());
        assert!(Panel::new(s.clone(), vec![d("2020-01-01"), d("2020-01-01")], vec![vec![1.0, 2.0]]).is_err());
        assert!(Panel::new(s.clone(), vec![d("2020-01-01")], vec![vec![1.0, 2.0]]).is_err());
        assert!(Panel::new(s.clone(), vec![d("2020-01-01")], vec![vec![0.0]]).is_err());
        assert!(Panel::new(s.clone(), vec![d("2020-01-01")], vec![vec![f64::NAN]]).is_err());
        assert!(Panel::new(s.clone(), vec![d("2020-01-01")], vec![vec![-1.0]]).is_err());
        assert!(Panel::new(s.clone(), vec![d("2020-01-01")], vec![vec![1.0]]).is_ok());
    }

    #[test]
    fn inner_join_keeps_only_common_dates_and_never_fills() {
        let a = ("A".to_string(), vec![(d("2020-01-01"), 1.0), (d("2020-01-02"), 2.0), (d("2020-01-04"), 4.0)]);
        let b = ("B".to_string(), vec![(d("2020-01-02"), 20.0), (d("2020-01-03"), 30.0), (d("2020-01-04"), 40.0)]);
        let p = Panel::inner_join(vec![a, b]).unwrap();
        assert_eq!(p.dates(), &[d("2020-01-02"), d("2020-01-04")]);
        assert_eq!(p.closes(0), &[2.0, 4.0]);
        assert_eq!(p.closes(1), &[20.0, 40.0]);
    }

    #[test]
    fn inner_join_rejects_disjoint_and_unsorted() {
        let a = ("A".to_string(), vec![(d("2020-01-01"), 1.0)]);
        let b = ("B".to_string(), vec![(d("2020-01-02"), 1.0)]);
        assert_eq!(Panel::inner_join(vec![a, b]).unwrap_err(), PanelError::NoCommonDates);
        let u = ("A".to_string(), vec![(d("2020-01-02"), 1.0), (d("2020-01-01"), 1.0)]);
        assert!(matches!(Panel::inner_join(vec![u]), Err(PanelError::Ordering { .. })));
    }

    #[test]
    fn csv_loader_filters_symbols_joins_and_rejects_bad_rows() {
        let csv = "symbol,date_utc,close\nA,2020-01-01,1.5\nZ,garbage,garbage\nB,2020-01-01,10\nA,2020-01-02,1.6\nB,2020-01-02,11\r\n";
        let p = Panel::from_long_csv(csv, &["A", "B"]).unwrap();
        assert_eq!(p.n_bars(), 2);
        assert_eq!(p.closes(0), &[1.5, 1.6]);
        assert!(Panel::from_long_csv("symbol,date,close\n", &["A"]).is_err());
        assert!(Panel::from_long_csv("symbol,date_utc,close\nA,2020-01-01,x\n", &["A"]).is_err());
        assert!(Panel::from_long_csv("symbol,date_utc,close\nA,2020-01-02,1\nA,2020-01-02,1\n", &["A"]).is_err());
        assert!(Panel::from_long_csv("symbol,date_utc,close\nA,2020-01-01,0\n", &["A"]).is_err());
        assert!(matches!(
            Panel::from_long_csv("symbol,date_utc,close\nA,2020-01-01,1\n", &["A", "B"]),
            Err(PanelError::MissingSymbol(_))
        ));
    }

    #[test]
    fn fixture_source_checks_the_pin_before_parsing() {
        let csv = b"symbol,date_utc,close\nA,2020-01-01,1.5\n";
        let good = sha256_hex(csv);
        assert!(PriceSource::Fixture { csv, expected_sha256: &good }.load(&["A"]).is_ok());
        let bad = "0".repeat(64);
        assert!(matches!(
            PriceSource::Fixture { csv, expected_sha256: &bad }.load(&["A"]),
            Err(PanelError::Sha256Mismatch { .. })
        ));
    }

    #[test]
    fn history_view_is_cut_at_t_and_aliases_panel_memory() {
        let p = Panel::new(
            vec!["A".into()],
            vec![d("2020-01-01"), d("2020-01-02"), d("2020-01-03")],
            vec![vec![1.0, 2.0, 3.0]],
        )
        .unwrap();
        let v = HistoryView::new(&p, 1);
        assert_eq!(v.len(), 2);
        assert_eq!(v.closes(0), &[1.0, 2.0]);
        assert_eq!(v.dates().len(), 2);
        assert_eq!(v.date(), d("2020-01-02"));
        assert_eq!(v.last_close(0), 2.0);
        // Same memory as the panel: the view's slice starts at the panel's slice start.
        assert!(std::ptr::eq(v.closes(0).as_ptr(), p.closes(0).as_ptr()));
    }

    #[test]
    fn poisoning_replaces_only_prices_from_the_cut_and_keeps_dates() {
        let p = Panel::new(
            vec!["A".into(), "B".into()],
            vec![d("2020-01-01"), d("2020-01-02"), d("2020-01-03")],
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        )
        .unwrap();
        let q = p.with_prices_replaced_from(2, |i, t, old| old * 100.0 + (i + t) as f64);
        assert_eq!(q.dates(), p.dates());
        assert_eq!(&q.closes(0)[..2], &p.closes(0)[..2]);
        assert_eq!(q.closes(0)[2], 302.0);
        assert_eq!(q.closes(1)[2], 603.0);
    }
}
