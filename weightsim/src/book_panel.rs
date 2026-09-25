//! [`BookPanel`]: the price data of a BOOK on the UNION account clock (design 3.3), replacing the strictly
//! inner-joined [`Panel`] for books (the `Panel` stays, unchanged, for single rules).
//!
//! * The clock is the union of every instrument's bar times, strictly ascending, [`BarTime`] (ms UTC of the bar close).
//! * `close[instrument][bar]` is `None` where the instrument has no bar on that clock instant. What a missing bar
//!   MEANS is decided by the instrument's [`SessionKind`], never guessed ([`Availability`]):
//!   `Closed` (its market is declared closed that day: the position is carried at the last close, zero return, and the
//!   next real bar earns the full change since the last real close, so no return is lost), `Gap` (the market was
//!   open but the vendor has no bar: a data gap; the book records a named refusal, never a silent fill), or
//!   `Unlisted` (before the instrument's first bar or after its last: it simply is not there yet / any more).
//! * Nothing is ever interpolated, forward-filled into a rule's history or truncated to the shortest history. A
//!   rule sees only the OWN bars of its sleeve ([`BookPanel::sleeve_calendar`]): the bars on which every instrument
//!   of its universe has a price, i.e. its own joint calendar, so estimators never run on carry rows.

use crate::bartime::BarTime;
use crate::date::Date;
use crate::panel::{Panel, PanelError};
use std::fmt;

/// How an instrument's market is open. Used only to classify a missing bar.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionKind {
    /// Trades every day (crypto, and any feed that must have a bar every clock instant): a missing bar is a GAP.
    Continuous,
    /// An exchange session: closed on Saturday and Sunday and on `closed_dates` (holidays). A missing bar on any
    /// other date is a GAP. There is no built-in holiday calendar (design U6): the caller declares the closures.
    Exchange { id: String, closed_dates: Vec<Date> },
}

impl SessionKind {
    /// Exchange session with the given holiday list (sorted and de-duplicated here).
    pub fn exchange(id: impl Into<String>, mut closed_dates: Vec<Date>) -> SessionKind {
        closed_dates.sort();
        closed_dates.dedup();
        SessionKind::Exchange { id: id.into(), closed_dates }
    }

    /// Is the market declared closed on `date`?
    pub fn is_declared_closed(&self, date: Date) -> bool {
        match self {
            SessionKind::Continuous => false,
            SessionKind::Exchange { closed_dates, .. } => {
                date.weekday() >= 5 || closed_dates.binary_search(&date).is_ok()
            }
        }
    }
}

/// Classification of an instrument at one clock instant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Availability {
    /// The instrument has a bar.
    Open,
    /// No bar, and its session declares the market closed: carry.
    Closed,
    /// No bar although the market was open and the instrument has bars on both sides: a data gap.
    Gap,
    /// No bar because the instrument has no history yet (before its first bar) or none any more (after its last).
    Unlisted,
}

/// Errors from building a [`BookPanel`]. There is no repair path: bad input is an error.
#[derive(Clone, Debug, PartialEq)]
pub enum BookPanelError {
    Empty(String),
    Shape(String),
    Ordering {
        time: BarTime,
    },
    BadPrice {
        instrument: String,
        time: BarTime,
        value: f64,
    },
    NoBars(String),
    DuplicateInstrument(String),
    Csv {
        line: usize,
        message: String,
    },
    /// A sleeve panel needs strictly ascending civil dates; sub-daily bars need a time-aware history view (later phase).
    NotDaily {
        time: BarTime,
    },
    Sleeve(PanelError),
}

impl fmt::Display for BookPanelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BookPanelError::Empty(m) => write!(f, "empty book panel: {m}"),
            BookPanelError::Shape(m) => write!(f, "book panel shape error: {m}"),
            BookPanelError::Ordering { time } => write!(f, "clock not strictly ascending at {time}"),
            BookPanelError::BadPrice { instrument, time, value } => {
                write!(f, "{instrument} {time}: price {value} is not finite and > 0")
            }
            BookPanelError::NoBars(s) => write!(f, "instrument {s} has no bar at all"),
            BookPanelError::DuplicateInstrument(s) => write!(f, "instrument {s} listed twice"),
            BookPanelError::Csv { line, message } => write!(f, "csv line {line}: {message}"),
            BookPanelError::NotDaily { time } => {
                write!(f, "two own bars of a sleeve fall on the same UTC date at {time}: sub-daily bars are not supported by weightsim 0.2")
            }
            BookPanelError::Sleeve(e) => write!(f, "sleeve panel: {e}"),
        }
    }
}

impl std::error::Error for BookPanelError {}

/// The own calendar of one sleeve inside a book: a dense [`Panel`] over the bars on which every instrument of the
/// sleeve's universe has a price, plus the map back to the union clock.
#[derive(Clone, Debug)]
pub struct SleeveCalendar {
    /// `union_index[own_bar]` = index on the union clock.
    pub union_index: Vec<usize>,
    /// The sleeve's own joint-calendar panel (the exact object a single-rule `simulate` would run on).
    pub panel: Panel,
}

/// One instrument's `(time, close)` series with its session, the input of [`BookPanel::from_series`].
pub type BarSeries = (String, SessionKind, Vec<(BarTime, f64)>);
/// One instrument's daily `(date, close)` series with its session, the input of [`BookPanel::from_dated_series`].
pub type DatedSeries = (String, SessionKind, Vec<(Date, f64)>);

/// Prices on the union clock. See the module docs.
#[derive(Clone, Debug, PartialEq)]
pub struct BookPanel {
    instruments: Vec<String>,
    sessions: Vec<SessionKind>,
    times: Vec<BarTime>,
    close: Vec<Vec<Option<f64>>>, // [instrument][bar]
    first_bar: Vec<usize>,
    last_bar: Vec<usize>,
}

impl BookPanel {
    /// Validated constructor: equal lengths, strictly ascending clock, every present close finite and > 0, every
    /// instrument has at least one bar, instrument ids unique.
    pub fn new(
        instruments: Vec<String>,
        sessions: Vec<SessionKind>,
        times: Vec<BarTime>,
        close: Vec<Vec<Option<f64>>>,
    ) -> Result<BookPanel, BookPanelError> {
        if instruments.is_empty() {
            return Err(BookPanelError::Empty("no instruments".into()));
        }
        if times.is_empty() {
            return Err(BookPanelError::Empty("no bars".into()));
        }
        if sessions.len() != instruments.len() || close.len() != instruments.len() {
            return Err(BookPanelError::Shape(format!(
                "{} instruments, {} sessions, {} price columns",
                instruments.len(),
                sessions.len(),
                close.len()
            )));
        }
        for (i, name) in instruments.iter().enumerate() {
            if instruments[..i].contains(name) {
                return Err(BookPanelError::DuplicateInstrument(name.clone()));
            }
            if close[i].len() != times.len() {
                return Err(BookPanelError::Shape(format!(
                    "{name}: {} closes for {} bars",
                    close[i].len(),
                    times.len()
                )));
            }
        }
        for w in times.windows(2) {
            if w[1] <= w[0] {
                return Err(BookPanelError::Ordering { time: w[1] });
            }
        }
        let mut first_bar = Vec::new();
        let mut last_bar = Vec::new();
        for (i, name) in instruments.iter().enumerate() {
            let mut first = None;
            let mut last = None;
            for (u, v) in close[i].iter().enumerate() {
                if let Some(p) = v {
                    if !(p.is_finite() && *p > 0.0) {
                        return Err(BookPanelError::BadPrice { instrument: name.clone(), time: times[u], value: *p });
                    }
                    if first.is_none() {
                        first = Some(u);
                    }
                    last = Some(u);
                }
            }
            match (first, last) {
                (Some(a), Some(b)) => {
                    first_bar.push(a);
                    last_bar.push(b);
                }
                _ => return Err(BookPanelError::NoBars(name.clone())),
            }
        }
        Ok(BookPanel { instruments, sessions, times, close, first_bar, last_bar })
    }

    /// Union of per-instrument `(time, close)` series. Each series must be strictly ascending.
    pub fn from_series(series: Vec<BarSeries>) -> Result<BookPanel, BookPanelError> {
        if series.is_empty() {
            return Err(BookPanelError::Empty("no series".into()));
        }
        let mut times: Vec<BarTime> = Vec::new();
        for (_, _, rows) in &series {
            for w in rows.windows(2) {
                if w[1].0 <= w[0].0 {
                    return Err(BookPanelError::Ordering { time: w[1].0 });
                }
            }
            times.extend(rows.iter().map(|r| r.0));
        }
        times.sort();
        times.dedup();
        let mut instruments = Vec::new();
        let mut sessions = Vec::new();
        let mut close = Vec::new();
        for (name, session, rows) in series {
            let mut col = vec![None; times.len()];
            let mut u = 0usize;
            for (t, p) in rows {
                while times[u] < t {
                    u += 1;
                }
                col[u] = Some(p);
            }
            instruments.push(name);
            sessions.push(session);
            close.push(col);
        }
        BookPanel::new(instruments, sessions, times, close)
    }

    /// Union of daily `(date, close)` series (each date becomes midnight UTC, [`BarTime::from_date`]).
    pub fn from_dated_series(series: Vec<DatedSeries>) -> Result<BookPanel, BookPanelError> {
        BookPanel::from_series(
            series
                .into_iter()
                .map(|(n, s, rows)| (n, s, rows.into_iter().map(|(d, p)| (BarTime::from_date(d), p)).collect()))
                .collect(),
        )
    }

    /// Parse the long fixture format (`symbol,date_utc,close`) for exactly `instruments` (in that order, each with
    /// its session) and take the UNION of their dates. Rows of other symbols are skipped without being parsed. No
    /// silent repair: an unparsable field, a non-positive close or an out-of-order date is an error.
    pub fn from_long_csv(text: &str, instruments: &[(&str, SessionKind)]) -> Result<BookPanel, BookPanelError> {
        let mut lines = text.split('\n').enumerate();
        match lines.next() {
            Some((_, h)) if h.trim_end_matches('\r') == "symbol,date_utc,close" => {}
            _ => {
                return Err(BookPanelError::Csv {
                    line: 1,
                    message: "header must be exactly `symbol,date_utc,close`".into(),
                })
            }
        }
        let mut series: Vec<DatedSeries> =
            instruments.iter().map(|(s, k)| ((*s).to_string(), k.clone(), Vec::new())).collect();
        for (idx, raw) in lines {
            let line = raw.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let lineno = idx + 1;
            let mut parts = line.split(',');
            let sym = parts.next().unwrap_or("");
            let slot = match instruments.iter().position(|(s, _)| *s == sym) {
                Some(p) => p,
                None => continue,
            };
            let (d, c) = match (parts.next(), parts.next(), parts.next()) {
                (Some(d), Some(c), None) => (d, c),
                _ => return Err(BookPanelError::Csv { line: lineno, message: "expected exactly 3 fields".into() }),
            };
            let date = Date::parse(d).map_err(|e| BookPanelError::Csv { line: lineno, message: e.to_string() })?;
            let close: f64 = c
                .parse()
                .map_err(|_| BookPanelError::Csv { line: lineno, message: format!("cannot parse close `{c}`") })?;
            series[slot].2.push((date, close));
        }
        for (s, _, rows) in &series {
            if rows.is_empty() {
                return Err(BookPanelError::NoBars(s.clone()));
            }
        }
        BookPanel::from_dated_series(series)
    }

    /// A book panel with every instrument of `panel` present on every bar (the degenerate, one-calendar case; this is
    /// what one-sleeve identity tests use). Sessions are `Continuous`.
    pub fn from_panel(panel: &Panel) -> BookPanel {
        let times: Vec<BarTime> = panel.dates().iter().map(|d| BarTime::from_date(*d)).collect();
        let close: Vec<Vec<Option<f64>>> =
            (0..panel.n_assets()).map(|i| panel.closes(i).iter().map(|p| Some(*p)).collect()).collect();
        BookPanel::new(panel.symbols().to_vec(), vec![SessionKind::Continuous; panel.n_assets()], times, close)
            .expect("a valid Panel is a valid BookPanel")
    }

    pub fn instruments(&self) -> &[String] {
        &self.instruments
    }
    pub fn sessions(&self) -> &[SessionKind] {
        &self.sessions
    }
    pub fn times(&self) -> &[BarTime] {
        &self.times
    }
    pub fn n_bars(&self) -> usize {
        self.times.len()
    }
    pub fn n_instruments(&self) -> usize {
        self.instruments.len()
    }
    /// Closes of instrument `i` on the union clock (`None` = no bar).
    pub fn close(&self, i: usize) -> &[Option<f64>] {
        &self.close[i]
    }
    pub fn instrument_index(&self, id: &str) -> Option<usize> {
        self.instruments.iter().position(|s| s == id)
    }

    /// Classify instrument `i` at clock index `u` (see [`Availability`]).
    pub fn availability(&self, i: usize, u: usize) -> Availability {
        if self.close[i][u].is_some() {
            Availability::Open
        } else if u < self.first_bar[i] || u > self.last_bar[i] {
            Availability::Unlisted
        } else if self.sessions[i].is_declared_closed(self.times[u].date()) {
            Availability::Closed
        } else {
            Availability::Gap
        }
    }

    /// The own calendar of a sleeve over `universe` (instrument indices, in the sleeve's universe order): the bars on
    /// which every one of those instruments has a close. Errors when it is empty or when two own bars fall on the same
    /// civil date (sub-daily data).
    pub fn sleeve_calendar(&self, universe: &[usize]) -> Result<SleeveCalendar, BookPanelError> {
        let mut union_index = Vec::new();
        for u in 0..self.n_bars() {
            if universe.iter().all(|&i| self.close[i][u].is_some()) {
                union_index.push(u);
            }
        }
        if union_index.is_empty() {
            return Err(BookPanelError::Sleeve(PanelError::NoCommonDates));
        }
        for w in union_index.windows(2) {
            if self.times[w[0]].date() >= self.times[w[1]].date() {
                return Err(BookPanelError::NotDaily { time: self.times[w[1]] });
            }
        }
        let dates: Vec<Date> = union_index.iter().map(|&u| self.times[u].date()).collect();
        let closes: Vec<Vec<f64>> = universe
            .iter()
            .map(|&i| union_index.iter().map(|&u| self.close[i][u].expect("own bar has every close")).collect())
            .collect();
        let symbols = universe.iter().map(|&i| self.instruments[i].clone()).collect();
        let panel = Panel::new(symbols, dates, closes).map_err(BookPanelError::Sleeve)?;
        Ok(SleeveCalendar { union_index, panel })
    }

    /// First `keep_bars` clock instants only.
    pub fn truncated(&self, keep_bars: usize) -> BookPanel {
        assert!(keep_bars >= 1 && keep_bars <= self.n_bars(), "keep_bars out of range");
        let close: Vec<Vec<Option<f64>>> = self.close.iter().map(|c| c[..keep_bars].to_vec()).collect();
        // An instrument whose only bars were cut has no bar left: keep the invariant by panicking loudly (test helper).
        BookPanel::new(self.instruments.clone(), self.sessions.clone(), self.times[..keep_bars].to_vec(), close)
            .expect("truncation left an instrument without any bar")
    }

    /// Same panel with the close of every PRESENT bar at clock index >= `first_poisoned_bar` replaced by
    /// `f(instrument, bar, old)`. Times and the availability pattern (which bars are `None`) are untouched: date-only
    /// look-ahead is an input, not a leak (design 2.4(4)). `f` must return finite positive garbage.
    pub fn with_prices_replaced_from(
        &self,
        first_poisoned_bar: usize,
        mut f: impl FnMut(usize, usize, f64) -> f64,
    ) -> BookPanel {
        let mut out = self.clone();
        for (i, col) in out.close.iter_mut().enumerate() {
            for (u, v) in col.iter_mut().enumerate().skip(first_poisoned_bar) {
                if let Some(old) = *v {
                    let nv = f(i, u, old);
                    assert!(nv.is_finite() && nv > 0.0, "poison values must be finite and positive");
                    *v = Some(nv);
                }
            }
        }
        out
    }

    /// Same panel with the instrument columns permuted: new column `j` is old column `order[j]`.
    pub fn with_instrument_order(&self, order: &[usize]) -> BookPanel {
        assert_eq!(order.len(), self.n_instruments());
        BookPanel::new(
            order.iter().map(|&i| self.instruments[i].clone()).collect(),
            order.iter().map(|&i| self.sessions[i].clone()).collect(),
            self.times.clone(),
            order.iter().map(|&i| self.close[i].clone()).collect(),
        )
        .expect("a permutation of a valid panel is valid")
    }

    /// Same panel with the given (instrument, bar) closes removed (set to `None`): builds missing-bar fixtures.
    pub fn with_bars_removed(&self, remove: &[(usize, usize)]) -> BookPanel {
        let mut close = self.close.clone();
        for &(i, u) in remove {
            close[i][u] = None;
        }
        BookPanel::new(self.instruments.clone(), self.sessions.clone(), self.times.clone(), close)
            .expect("removal left an instrument without any bar")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    fn mixed() -> BookPanel {
        // ETF-like weekday series (closed Sat/Sun, holiday on Mon 2020-01-20) and a 7-day crypto-like series.
        let etf = SessionKind::exchange("test_us", vec![d("2020-01-20")]);
        let mut etf_rows = Vec::new();
        let mut cry_rows = Vec::new();
        let mut day = d("2020-01-15");
        for i in 0..10 {
            if day.weekday() < 5 && day != d("2020-01-20") {
                etf_rows.push((day, 100.0 + i as f64));
            }
            cry_rows.push((day, 10.0 + i as f64));
            day = day.add_days(1);
        }
        BookPanel::from_dated_series(vec![
            ("ETF".into(), etf, etf_rows),
            ("CRY".into(), SessionKind::Continuous, cry_rows),
        ])
        .unwrap()
    }

    #[test]
    fn union_clock_and_availability_classes() {
        let p = mixed();
        assert_eq!(p.n_bars(), 10);
        // 2020-01-18 Sat, 19 Sun, 20 Mon(holiday): ETF closed; crypto always open.
        let idx = |s: &str| p.times().iter().position(|t| t.date() == d(s)).unwrap();
        for s in ["2020-01-18", "2020-01-19", "2020-01-20"] {
            assert_eq!(p.availability(0, idx(s)), Availability::Closed, "{s}");
        }
        assert_eq!(p.availability(0, idx("2020-01-21")), Availability::Open);
        assert_eq!(p.availability(1, idx("2020-01-18")), Availability::Open);
    }

    #[test]
    fn a_missing_bar_on_an_open_market_is_a_gap_and_outside_history_is_unlisted() {
        let p = mixed();
        let i = p.times().iter().position(|t| t.date() == d("2020-01-17")).unwrap();
        let q = p.with_bars_removed(&[(0, i)]);
        assert_eq!(q.availability(0, i), Availability::Gap);
        // A crypto bar removed on the very first clock instant leaves the instrument Unlisted there, not a gap.
        let r = p.with_bars_removed(&[(1, 0)]);
        assert_eq!(r.availability(1, 0), Availability::Unlisted);
        let last = p.n_bars() - 1;
        let s = p.with_bars_removed(&[(1, last)]);
        assert_eq!(s.availability(1, last), Availability::Unlisted);
    }

    #[test]
    fn sleeve_calendar_is_the_inner_join_of_its_own_universe_only() {
        let p = mixed();
        let etf = p.sleeve_calendar(&[0]).unwrap();
        let cry = p.sleeve_calendar(&[1]).unwrap();
        assert_eq!(cry.panel.n_bars(), 10);
        assert_eq!(etf.panel.n_bars(), 7); // 15,16,17 (Wed-Fri) and 21,22,23,24 (Tue-Fri) within the 10 clock days
        assert!(etf.union_index.windows(2).all(|w| w[0] < w[1]));
        let both = p.sleeve_calendar(&[0, 1]).unwrap();
        assert_eq!(both.panel.n_bars(), 7);
        assert_eq!(both.panel.symbols(), &["ETF".to_string(), "CRY".to_string()]);
    }

    #[test]
    fn validation_rejects_bad_input() {
        let t = |s: &str| BarTime::from_date(d(s));
        let ok_t = vec![t("2020-01-01"), t("2020-01-02")];
        let c = SessionKind::Continuous;
        assert!(BookPanel::new(vec![], vec![], ok_t.clone(), vec![]).is_err());
        assert!(matches!(
            BookPanel::new(
                vec!["A".into()],
                vec![c.clone()],
                vec![t("2020-01-02"), t("2020-01-01")],
                vec![vec![Some(1.0), Some(2.0)]]
            ),
            Err(BookPanelError::Ordering { .. })
        ));
        assert!(matches!(
            BookPanel::new(vec!["A".into()], vec![c.clone()], ok_t.clone(), vec![vec![Some(1.0), Some(0.0)]]),
            Err(BookPanelError::BadPrice { .. })
        ));
        assert!(matches!(
            BookPanel::new(vec!["A".into()], vec![c.clone()], ok_t.clone(), vec![vec![Some(1.0), Some(f64::NAN)]]),
            Err(BookPanelError::BadPrice { .. })
        ));
        assert!(matches!(
            BookPanel::new(vec!["A".into()], vec![c.clone()], ok_t.clone(), vec![vec![None, None]]),
            Err(BookPanelError::NoBars(_))
        ));
        assert!(matches!(
            BookPanel::new(
                vec!["A".into(), "A".into()],
                vec![c.clone(), c.clone()],
                ok_t.clone(),
                vec![vec![Some(1.0), Some(1.0)]; 2]
            ),
            Err(BookPanelError::DuplicateInstrument(_))
        ));
        assert!(matches!(
            BookPanel::new(vec!["A".into()], vec![c], ok_t, vec![vec![Some(1.0)]]),
            Err(BookPanelError::Shape(_))
        ));
    }

    #[test]
    fn sub_daily_own_bars_are_refused_not_mangled() {
        let base = BarTime::from_date(d("2020-01-01")).ms();
        let times = vec![BarTime::from_ms(base), BarTime::from_ms(base + 3_600_000)];
        let p =
            BookPanel::new(vec!["A".into()], vec![SessionKind::Continuous], times, vec![vec![Some(1.0), Some(2.0)]])
                .unwrap();
        assert!(matches!(p.sleeve_calendar(&[0]), Err(BookPanelError::NotDaily { .. })));
    }

    #[test]
    fn csv_union_loader_keeps_every_date_and_pads_none() {
        let csv = "symbol,date_utc,close\nA,2020-01-01,1.5\nZ,garbage,garbage\nB,2020-01-02,10\nA,2020-01-03,1.6\r\n";
        let p =
            BookPanel::from_long_csv(csv, &[("A", SessionKind::Continuous), ("B", SessionKind::Continuous)]).unwrap();
        assert_eq!(p.n_bars(), 3);
        assert_eq!(p.close(0), &[Some(1.5), None, Some(1.6)]);
        assert_eq!(p.close(1), &[None, Some(10.0), None]);
        assert!(BookPanel::from_long_csv("symbol,date,close\n", &[("A", SessionKind::Continuous)]).is_err());
    }

    #[test]
    fn poisoning_and_permutation_preserve_the_availability_pattern() {
        let p = mixed();
        let q = p.with_prices_replaced_from(4, |i, t, old| old * 7.0 + (i + t) as f64);
        for i in 0..2 {
            for u in 0..p.n_bars() {
                assert_eq!(p.close(i)[u].is_some(), q.close(i)[u].is_some());
                if u < 4 {
                    assert_eq!(p.close(i)[u], q.close(i)[u]);
                }
            }
        }
        let r = p.with_instrument_order(&[1, 0]);
        assert_eq!(r.instruments(), &["CRY".to_string(), "ETF".to_string()]);
        assert_eq!(r.close(1), p.close(0));
    }
}
