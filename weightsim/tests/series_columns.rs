//! `SeriesColumns` (stage T4, slice C1): the digest-covered series as plain owned columns. The simulator's own
//! `series_sha256` IS `SeriesColumns::from(&run).digest()`, so a stored run can be re-verified from its columns.

// Index loops mirror the layout of the flat matrices.
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

mod common;

use common::*;
use weightsim::*;

fn net(base: SimConfig) -> SimConfig {
    SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..base }
}

/// The goldens of `determinism.rs`, produced BEFORE the digest moved into `SeriesColumns`.
const GOLDEN_S1_GROSS: &str = "a6bae736c2f06218a77a473863f666abcb72d0ffa9d0bc540801f4bc2da4bd88";
const GOLDEN_S3_NET: &str = "2932738e542adbf8ca5097e3079249d03dd676b4e57e1f1a789502875fc97430";

fn runs() -> Vec<SimResult> {
    let mut out = vec![
        simulate(&s1_panel(), &S1TestRule, &s1_config()).unwrap(),
        simulate(&s3_panel(), &S3TestRule, &net(s3_config())).unwrap(),
        simulate(&s3_panel(), &S3TestRule, &SimConfig { execution_delay_bars: 1, ..s3_config() }).unwrap(),
        simulate(&s3_panel(), &S3TestRule, &SimConfig { risk_scale: 0.5, ..net(s3_config()) }).unwrap(),
    ];
    // A levered long/short book with financing, and a rule that refuses on some bars (the run holds its book).
    let panel = synth_panel(&["A", "B", "C"], 300, 42, "2019-01-01");
    let lev = FnRule::new(&["A", "B", "C"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 5, |h| {
        let s = if (h.len() / 9) % 2 == 0 { 1.0 } else { -1.0 };
        Ok(vec![0.6 * s, -0.4, 0.3])
    });
    let cfg = SimConfig {
        financing: Financing::FlatAnnual { long_bps: 50.0, short_bps: 80.0, cash_bps: 100.0 },
        ..net(SimConfig::default())
    };
    out.push(simulate(&panel, &lev, &cfg).unwrap());
    let refusing = FnRule::new(&["A", "B", "C"], DecisionSchedule::Daily, RebalancePolicy::EveryBar, 5, |h| {
        if h.len() % 7 == 0 {
            Err(RuleRefusal::new(RefusalKind::Data, "gap", "no data"))
        } else {
            Ok(vec![0.3, 0.3, 0.3])
        }
    });
    let held =
        simulate(&panel, &refusing, &SimConfig { on_refusal: OnRefusal::HoldPrevious, ..net(SimConfig::default()) })
            .unwrap();
    assert!(held.refused.iter().any(|r| *r), "the refusing run must actually refuse");
    out.push(held);
    out
}

#[test]
fn the_columns_digest_is_the_simulators_own_series_digest_for_every_kind_of_run() {
    let all = runs();
    assert_eq!(all[0].series_sha256, GOLDEN_S1_GROSS);
    assert_eq!(all[1].series_sha256, GOLDEN_S3_NET);
    for (i, r) in all.iter().enumerate() {
        let cols = SeriesColumns::from(r);
        assert_eq!(cols, r.series_columns());
        assert_eq!(cols.digest().unwrap(), r.series_sha256, "run {i}");
        assert_eq!((cols.n_bars(), cols.n_assets()), (r.n_bars(), r.n_assets()));
        assert_eq!(cols.cost_model_id, r.cost_model_id);
        assert_eq!(cols.metric_definitions, METRIC_DEFINITIONS);
        assert_eq!(cols.first_non_finite(), None);
        assert_eq!(cols.validate_shape(), Ok(()));
        assert_eq!(cols.row(&cols.held_weights, 3), r.row(&r.held_weights, 3));
    }
}

fn set_float(c: &mut SeriesColumns, name: &str, i: usize) {
    let col = match name {
        "ret" => &mut c.ret,
        "ret_pre_cost" => &mut c.ret_pre_cost,
        "equity" => &mut c.equity,
        "cash" => &mut c.cash,
        "cost" => &mut c.cost,
        "traded_notional" => &mut c.traded_notional,
        "financing" => &mut c.financing,
        "gross_exposure" => &mut c.gross_exposure,
        "net_exposure" => &mut c.net_exposure,
        "target_weights" => &mut c.target_weights,
        "held_weights" => &mut c.held_weights,
        "units" => &mut c.units,
        _ => panic!("{name}"),
    };
    col[i] = f64::from_bits(col[i].to_bits() ^ 1);
}

#[test]
fn every_field_of_the_columns_is_covered_by_the_digest() {
    let r = &runs()[4];
    let base = SeriesColumns::from(r);
    let want = base.digest().unwrap();
    let k = base.n_assets();
    let n = base.n_bars();
    let differs = |c: &SeriesColumns| c.digest().unwrap() != want;
    for name in [
        "ret",
        "ret_pre_cost",
        "equity",
        "cash",
        "cost",
        "traded_notional",
        "financing",
        "gross_exposure",
        "net_exposure",
    ] {
        for i in [0, n / 2, n - 1] {
            let mut c = base.clone();
            set_float(&mut c, name, i);
            assert!(differs(&c), "{name}[{i}]");
        }
    }
    for name in ["target_weights", "held_weights", "units"] {
        for i in [0, (n / 2) * k + 1, n * k - 1] {
            let mut c = base.clone();
            set_float(&mut c, name, i);
            assert!(differs(&c), "{name}[{i}]");
        }
    }
    let mut c = base.clone();
    c.decision[n / 2] = !c.decision[n / 2];
    assert!(differs(&c));
    let mut c = base.clone();
    c.refused[n / 3] = !c.refused[n / 3];
    assert!(differs(&c));
    // A date moved one day back into a weekend gap stays ascending: only the digest sees it.
    let t = (1..n).find(|&t| base.dates[t - 1].days_until(base.dates[t]) >= 2).expect("a weekend gap");
    let mut c = base.clone();
    c.dates[t] = c.dates[t].add_days(-1);
    assert!(differs(&c));
    for edit in [
        |c: &mut SeriesColumns| c.rule_id.push('x'),
        |c: &mut SeriesColumns| c.rule_impl_version.push('x'),
        |c: &mut SeriesColumns| c.cost_model_id.push('x'),
        |c: &mut SeriesColumns| c.metric_definitions.push('x'),
        |c: &mut SeriesColumns| c.symbols[1].push('x'),
        |c: &mut SeriesColumns| c.symbols.swap(0, 1),
    ] {
        let mut c = base.clone();
        edit(&mut c);
        assert!(differs(&c));
    }
}

#[test]
fn malformed_columns_are_errors_not_panics_and_never_get_a_digest() {
    let base = SeriesColumns::from(&runs()[4]);
    let n = base.n_bars();
    let k = base.n_assets();
    let mut c = base.clone();
    c.units.truncate(n * k - 1);
    assert_eq!(c.digest(), Err(ColumnsError::LengthMismatch { column: "units", expected: n * k, found: n * k - 1 }));
    let mut c = base.clone();
    c.decision.push(true);
    assert_eq!(c.digest(), Err(ColumnsError::LengthMismatch { column: "decision", expected: n, found: n + 1 }));
    let mut c = base.clone();
    c.dates[9] = c.dates[8];
    assert_eq!(c.digest(), Err(ColumnsError::DatesNotAscending { bar: 9 }));
    let mut c = base.clone();
    c.dates.clear();
    assert_eq!(c.digest(), Err(ColumnsError::Empty));
    assert!(ColumnsError::Empty.to_string().contains("no bars"));
    // Non-finite values are located, and are not a shape problem.
    let mut c = base.clone();
    c.cash[12] = f64::NAN;
    assert_eq!(c.first_non_finite(), Some(("cash", 12)));
    assert!(c.digest().is_ok());
    let mut c = base.clone();
    c.target_weights[5 * k + 2] = f64::INFINITY;
    assert_eq!(c.first_non_finite(), Some(("target_weights", 5)));
}
