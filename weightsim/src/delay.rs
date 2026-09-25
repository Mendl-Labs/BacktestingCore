//! Delay sensitivity (council Ruling 4, work item W1): run a book, or one sleeve of it, at several execution delays and
//! report the metrics table, so a customer (and the library entry's provenance) can see how much of a result depends on
//! the assumption "decided at the close, filled at the same close".
//!
//! The table is a pure function of `(panel, book, cfg, scope, delays)`: every row is one full [`simulate_book_gross_and_net`]
//! (the gross run under zero cost and no financing, the net run under `cfg` as given), so a row is exactly what a user who
//! set that delay by hand would have got. Nothing here selects a delay; the caller pre-registers the delay it will state
//! (ETF 1, crypto 0) and shows this table beside it (Ruling 12: no timing variant is chosen by Sharpe).
//!
//! What `d` means (see [`crate::SleeveSpec::execution_delay`]): a decision at the close of a sleeve's OWN bar `t` is
//! traded at the close of the sleeve's own bar `t + d`.

use crate::book::{Book, BookConfig, BookError};
use crate::book_panel::BookPanel;
use crate::book_result::BookResult;
use crate::book_sim::simulate_book_gross_and_net;
use crate::date::Date;
use crate::metrics::Metrics;
use std::fmt::Write as _;

/// The delays of the council's table (Ruling 4): `d = 0, 1, 2, 3, 5` own bars.
pub const STANDARD_DELAYS: [usize; 5] = [0, 1, 2, 3, 5];

/// Which sleeves a sensitivity row delays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelayScope {
    /// EVERY sleeve runs at `d` (this overrides the book-level value and any per-sleeve delay already set).
    Book,
    /// Only sleeve `s` (index into `Book::sleeves`) runs at `d`; every other sleeve keeps the delay it already has (its own
    /// `execution_delay`, else the book-level `execution_delay_bars`).
    Sleeve(usize),
}

/// One row of the table: the run at `delay`.
#[derive(Clone, Debug)]
pub struct DelayRow {
    pub delay: usize,
    /// Account-clock metrics (`account_clock_v1`) of the gross run (zero cost, no financing); `None` when the counted
    /// window has fewer than two returns.
    pub gross: Option<Metrics>,
    /// The same for the net run (`cfg` as given).
    pub net: Option<Metrics>,
    /// Pearson correlation of the NET counted daily returns of this row with the FIRST row of the table (the baseline,
    /// normally `d = 0`), over the dates both windows count. NaN when fewer than two dates overlap or a series is constant.
    pub corr_net_to_baseline: f64,
    /// Total cost paid in the net run and its traded notional, in units of `initial_equity`.
    pub total_cost: f64,
    pub total_traded_notional: f64,
    /// Series digest of the net run (pins the row).
    pub series_sha256: String,
}

/// `book` with the delay of `scope` set to `d`.
pub fn book_with_delay(book: &Book, scope: DelayScope, d: usize) -> Result<Book, BookError> {
    let mut b = book.clone();
    match scope {
        DelayScope::Book => {
            for sp in &mut b.sleeves {
                sp.execution_delay = Some(d);
            }
        }
        DelayScope::Sleeve(s) => {
            let n = b.sleeves.len();
            let sp = b
                .sleeves
                .get_mut(s)
                .ok_or_else(|| BookError::BadBook(format!("delay scope names sleeve {s} but the book has {n}")))?;
            sp.execution_delay = Some(d);
        }
    }
    Ok(b)
}

/// The metrics table of `book` at every delay in `delays` (rows in the order given; the first is the baseline of
/// `corr_net_to_baseline`). Use [`STANDARD_DELAYS`] for the council's table.
pub fn delay_sensitivity(
    panel: &BookPanel,
    book: &Book,
    cfg: &BookConfig,
    scope: DelayScope,
    delays: &[usize],
) -> Result<Vec<DelayRow>, BookError> {
    if delays.is_empty() {
        return Err(BookError::BadBook("delay sensitivity needs at least one delay".into()));
    }
    let mut rows: Vec<DelayRow> = Vec::with_capacity(delays.len());
    let mut baseline: Option<(Vec<Date>, Vec<f64>)> = None;
    for &d in delays {
        let b = book_with_delay(book, scope, d)?;
        let (g, n) = simulate_book_gross_and_net(panel, &b, cfg)?;
        let dates = n.window_dates();
        let rets = n.window_returns().to_vec();
        let corr = match &baseline {
            None => {
                baseline = Some((dates.clone(), rets.clone()));
                aligned_correlation(&dates, &rets, &dates, &rets)
            }
            Some((bd, br)) => aligned_correlation(bd, br, &dates, &rets),
        };
        rows.push(row(d, &g, &n, corr));
    }
    Ok(rows)
}

fn row(delay: usize, g: &BookResult, n: &BookResult, corr: f64) -> DelayRow {
    DelayRow {
        delay,
        gross: g.metrics(),
        net: n.metrics(),
        corr_net_to_baseline: corr,
        total_cost: n.total_cost(),
        total_traded_notional: n.total_traded_notional(),
        series_sha256: n.series_sha256.clone(),
    }
}

/// Pearson correlation of two dated return series over the dates present in both (both ascending). Sequential sums, so the
/// result is reproducible. NaN for fewer than two common dates or a zero variance.
pub fn aligned_correlation(da: &[Date], ra: &[f64], db: &[Date], rb: &[f64]) -> f64 {
    assert_eq!(da.len(), ra.len());
    assert_eq!(db.len(), rb.len());
    let (mut i, mut j) = (0usize, 0usize);
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    while i < da.len() && j < db.len() {
        if da[i] == db[j] {
            xs.push(ra[i]);
            ys.push(rb[j]);
            i += 1;
            j += 1;
        } else if da[i] < db[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut sx = 0.0;
    let mut sy = 0.0;
    for k in 0..n {
        sx += xs[k];
        sy += ys[k];
    }
    let (mx, my) = (sx / n as f64, sy / n as f64);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for k in 0..n {
        let (dx, dy) = (xs[k] - mx, ys[k] - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if !(sxx > 0.0 && syy > 0.0) {
        return f64::NAN;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// The table as fixed-width text (for reports and test output): net Sharpe, CAGR and max drawdown, the gross Sharpe, the
/// correlation of the net returns to the baseline row, and the cost.
pub fn format_delay_table(rows: &[DelayRow]) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{:>5} {:>10} {:>10} {:>10} {:>13} {:>11} {:>10}",
        "delay", "net_sharpe", "net_cagr", "net_max_dd", "corr_to_base", "gross_sharpe", "cost"
    );
    for r in rows {
        let (ns, nc, nd) =
            r.net.as_ref().map_or((f64::NAN, f64::NAN, f64::NAN), |m| (m.sharpe, m.cagr, m.max_drawdown));
        let gs = r.gross.as_ref().map_or(f64::NAN, |m| m.sharpe);
        let _ = writeln!(
            s,
            "{:>5} {:>10.4} {:>10.4} {:>10.4} {:>13.6} {:>11.4} {:>10.6}",
            r.delay, ns, nc, nd, r.corr_net_to_baseline, gs, r.total_cost
        );
    }
    s
}
