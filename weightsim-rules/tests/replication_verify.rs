//! Always-on: stage T4 slice C1. Replicate a library rule, package its full-resolution series as plain columns, and
//! RE-VERIFY a stored run from the columns alone. Synthetic fixtures only (the crate's in-memory answer key of
//! `tests/common`); the real pinned data is `tests/replication_verify_real.rs` (env-gated).
//!
//! One test per tamper class (a changed cell by one ulp, a changed or swapped date, a changed fixture byte, the rule id,
//! the implementation version, the cost preset, a dropped row, swapped gross/net columns, a NaN, a wrong claimed digest
//! or summary), each asserting the exact typed error, plus the round trip, bit-for-bit metrics, determinism and the
//! bytes-versus-files fixture loaders.

mod common;

use std::sync::OnceLock;

use common::*;
use weightsim::*;
use weightsim_rules::ladder::fixtures::{Fixtures, LadderError, Pins, F_CANDLES, F_MANIFEST};
use weightsim_rules::ladder::runner::{entry_date, flips_by_key_convention, run_gross_and_net, Basis};
use weightsim_rules::ladder::verify::*;
use weightsim_rules::ladder::{run_ladder_with, LadderOptions};
use weightsim_rules::{CryptoTrendRule, EtfTrendRule, FlatUntil, ADAPTER_VERSION};

const S1: &str = "etf_trend_faber";
const S3: &str = "crypto_trend_100d";
const FAST: VerifyOptions = VerifyOptions { tier4: false };
const NO_CANARIES: LadderOptions = LadderOptions { check_canaries: false };

struct World {
    fx: Fixtures,
    s1: ReplicationRun,
    s3: ReplicationRun,
    /// The ladder's own base-run digests: S1 gross, S1 net, S3 gross, S3 net.
    ladder_digests: Vec<(String, String)>,
}

fn world() -> &'static World {
    static W: OnceLock<World> = OnceLock::new();
    W.get_or_init(|| {
        let fx = fixtures();
        let s1 = replicate(&fx, S1).expect("S1 replicates");
        let s3 = replicate(&fx, S3).expect("S3 replicates");
        let ladder_digests = run_ladder_with(&fx, &NO_CANARIES).expect("ladder runs").series_digests();
        World { fx, s1, s3, ladder_digests }
    })
}

fn verify_fast(
    fx: &Fixtures,
    rule: &str,
    gross: &SeriesColumns,
    net: &SeriesColumns,
) -> Result<VerifiedRun, VerifyError> {
    verify_with(fx, &VerifyRequest::new(rule, gross, net), &FAST)
}

/// Apply `tamper` to copies of the genuine S1 columns and verify them (Tier IV off: it is a property of the fixtures).
fn tampered_s1(tamper: impl FnOnce(&mut SeriesColumns, &mut SeriesColumns)) -> Result<VerifiedRun, VerifyError> {
    let w = world();
    let (mut g, mut n) = (w.s1.gross.clone(), w.s1.net.clone());
    tamper(&mut g, &mut n);
    verify_fast(&w.fx, S1, &g, &n)
}

fn expect_err(r: Result<VerifiedRun, VerifyError>) -> VerifyError {
    match r {
        Ok(v) => panic!("a tampered run was ACCEPTED: {v:?}"),
        Err(e) => e,
    }
}

/// `(basis, first difference)` of a digest mismatch; any other error fails the test.
fn digest_mismatch(e: VerifyError) -> (&'static str, Option<Difference>) {
    match e {
        VerifyError::DigestMismatch { basis, stored, platform, first_difference } => {
            assert_ne!(stored, platform);
            assert_eq!(stored.len(), 64);
            (basis, first_difference)
        }
        other => panic!("expected DigestMismatch, got {other:?}"),
    }
}

fn float_col<'a>(c: &'a mut SeriesColumns, name: &str) -> &'a mut Vec<f64> {
    match name {
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
        other => panic!("no float column {other}"),
    }
}

const SCALAR_COLS: [&str; 9] =
    ["ret", "ret_pre_cost", "equity", "cash", "cost", "traded_notional", "financing", "gross_exposure", "net_exposure"];
const MATRIX_COLS: [&str; 3] = ["target_weights", "held_weights", "units"];

fn next_ulp(v: f64) -> f64 {
    f64::from_bits(v.to_bits() ^ 1)
}

fn drop_last_row(c: &mut SeriesColumns) {
    let k = c.n_assets();
    c.dates.pop();
    for name in SCALAR_COLS {
        float_col(c, name).pop();
    }
    c.decision.pop();
    c.refused.pop();
    for name in MATRIX_COLS {
        let col = float_col(c, name);
        col.truncate(col.len() - k);
    }
}

fn drop_first_row(c: &mut SeriesColumns) {
    let k = c.n_assets();
    c.dates.remove(0);
    for name in SCALAR_COLS {
        float_col(c, name).remove(0);
    }
    c.decision.remove(0);
    c.refused.remove(0);
    for name in MATRIX_COLS {
        float_col(c, name).drain(0..k);
    }
}

// ------------------------------------------------------------------------------------------------ the round trip

#[test]
fn round_trip_replicate_columns_verify_is_ok_with_identical_digests() {
    let w = world();
    for (rule, run, ladder) in [(S1, &w.s1, &w.ladder_digests[0..2]), (S3, &w.s3, &w.ladder_digests[2..4])] {
        // The columns hash to the simulator's own digest, and to the ladder's base-run digests.
        assert_eq!(run.gross.digest().unwrap(), run.gross_sha256, "{rule}");
        assert_eq!(run.net.digest().unwrap(), run.net_sha256, "{rule}");
        assert_eq!((run.gross_sha256.clone(), run.net_sha256.clone()), (ladder[0].1.clone(), ladder[1].1.clone()));
        assert_ne!(run.gross_sha256, run.net_sha256);
        assert_eq!(run.gross.n_bars(), run.net.n_bars());

        // Every tier, Tier IV included, and every claim of the run.
        let req = VerifyRequest { claims: run.claims(), ..VerifyRequest::new(rule, &run.gross, &run.net) };
        let v = verify_with(&w.fx, &req, &VerifyOptions::default()).unwrap_or_else(|e| panic!("{rule}: {e}"));
        assert_eq!(v.gross.report.series_sha256, run.gross_sha256);
        assert_eq!(v.net.report.series_sha256, run.net_sha256);
        assert!(v.tier_failures().is_empty());
        for b in [&v.gross, &v.net] {
            let r = &b.report;
            assert!(r.tier1_pass && r.tier2_pass && r.tier3_pass && r.window_matches_key && r.cmp.covers_key_exactly());
            assert_eq!(r.flips_run, r.flips_key);
        }
        let t4 = v.tier4.as_ref().expect("Tier IV ran");
        assert!(t4.passed() && t4.sleeve_code == v.sleeve_code);
        assert_eq!(v.rule_id, rule);
        assert_eq!(v.rule_impl_version, run.rule_impl_version);
        assert_eq!(
            (v.manifest_sha256.as_str(), v.candles_sha256.as_str()),
            (w.fx.manifest_sha256.as_str(), w.fx.candles_sha256.as_str())
        );
        assert_eq!(v.cost_model_id, DEFAULT_COST_MODEL_ID);

        // The plain form gives the same verdict.
        assert_eq!(verify(&w.fx, rule, &run.gross, &run.net).unwrap(), v);
    }
    assert_eq!(w.s1.gross.n_assets(), 5);
    assert_eq!(w.s3.gross.n_assets(), 2);
}

/// The implementation version is a digest input and contains the crate version: it must stay `0.1.0` (bumping it
/// changes every pinned digest). This is the finding that C1 keeps `weightsim-rules` at 0.1.0.
#[test]
fn the_impl_version_string_carries_the_crate_version_and_is_a_digest_input() {
    let w = world();
    assert_eq!(ADAPTER_VERSION, "weightsim-rules 0.1.0 over reference-rules");
    let entry = entry_date(&w.fx.etf_panel, &w.fx.s1).unwrap();
    assert_eq!(w.s1.rule_impl_version, format!("{ADAPTER_VERSION}, flat until {entry}"));
    assert_eq!(w.s1.gross.rule_impl_version, w.s1.rule_impl_version);
    // Changing only that string changes the digest.
    let mut c = w.s1.gross.clone();
    c.rule_impl_version = c.rule_impl_version.replace("0.1.0", "0.1.1");
    assert_ne!(c.digest().unwrap(), w.s1.gross_sha256);
}

/// Golden digests of the synthetic fixture runs, measured on the tree BEFORE this slice (the `SeriesColumns` move):
/// the digest is byte-identical, so every pinned digest in Core and in the Engine is unchanged.
const SYNTH_S1_GROSS: &str = "640d62c6e3d87acb53cb0643a9bdce458ca2b41bb0dadcb4cf12ed2a7881068e";
const SYNTH_S1_NET: &str = "6e64c47b434ebbe25f73755810a3e61dfcafe190b1bba8f21934d02bdf3e5a55";
const SYNTH_S3_GROSS: &str = "0a2ce28aec227a3084a2786dbec6cb9793e3271cbf72744f25145908d474c59d";
const SYNTH_LADDER_DIGEST: &str = "e21e94b1f9b4a3c40313514e231c0ea0753b066af2c85a59a8d6c6052265c3db";
const SYNTH_S3_NET: &str = "d4cd9a87c3fc4ac5051d3fb457033b4caa7600669db5b30f507202977b37c5da";

#[test]
fn moving_the_digest_into_series_columns_changed_no_digest() {
    let w = world();
    assert_eq!(w.s1.gross_sha256, SYNTH_S1_GROSS);
    assert_eq!(w.s1.net_sha256, SYNTH_S1_NET);
    assert_eq!(w.s3.gross_sha256, SYNTH_S3_GROSS);
    assert_eq!(w.s3.net_sha256, SYNTH_S3_NET);
    // And the ladder's own digest over them, the mutants and every check.
    let ladder = run_ladder_with(&w.fx, &NO_CANARIES).unwrap();
    assert_eq!(ladder.digest, SYNTH_LADDER_DIGEST);
}

/// A store-shaped round trip: serialise the columns to text with the shortest round-trip float rendering, parse them
/// back, and the digest is the original. (The real codec lives in the Engine; this pins that nothing in the digest
/// depends on anything but the columns' values.)
fn encode(c: &SeriesColumns) -> String {
    let f = |v: &[f64]| v.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>().join(" ");
    let b = |v: &[bool]| v.iter().map(|x| if *x { "1" } else { "0" }).collect::<Vec<_>>().join("");
    let dates = c.dates.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" ");
    [
        c.rule_id.clone(),
        c.rule_impl_version.clone(),
        c.symbols.join(","),
        c.cost_model_id.clone(),
        c.metric_definitions.clone(),
        dates,
        f(&c.ret),
        f(&c.ret_pre_cost),
        f(&c.equity),
        f(&c.cash),
        f(&c.cost),
        f(&c.traded_notional),
        f(&c.financing),
        f(&c.gross_exposure),
        f(&c.net_exposure),
        b(&c.decision),
        b(&c.refused),
        f(&c.target_weights),
        f(&c.held_weights),
        f(&c.units),
    ]
    .join("\n")
}

fn decode(text: &str) -> SeriesColumns {
    let l: Vec<&str> = text.split('\n').collect();
    let f = |i: usize| -> Vec<f64> { l[i].split(' ').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect() };
    let b = |i: usize| -> Vec<bool> { l[i].chars().map(|c| c == '1').collect() };
    SeriesColumns {
        rule_id: l[0].to_string(),
        rule_impl_version: l[1].to_string(),
        symbols: l[2].split(',').map(str::to_string).collect(),
        cost_model_id: l[3].to_string(),
        metric_definitions: l[4].to_string(),
        dates: l[5].split(' ').map(|s| Date::parse(s).unwrap()).collect(),
        ret: f(6),
        ret_pre_cost: f(7),
        equity: f(8),
        cash: f(9),
        cost: f(10),
        traded_notional: f(11),
        financing: f(12),
        gross_exposure: f(13),
        net_exposure: f(14),
        decision: b(15),
        refused: b(16),
        target_weights: f(17),
        held_weights: f(18),
        units: f(19),
    }
}

#[test]
fn a_decode_then_recompute_digest_equals_the_original_series_sha256() {
    let w = world();
    for run in [&w.s1, &w.s3] {
        for (cols, want) in [(&run.gross, &run.gross_sha256), (&run.net, &run.net_sha256)] {
            let back = decode(&encode(cols));
            assert_eq!(&back, cols);
            assert_eq!(&back.digest().unwrap(), want);
        }
        let g = decode(&encode(&run.gross));
        let n = decode(&encode(&run.net));
        assert!(verify_fast(&w.fx, &run.rule_id, &g, &n).is_ok());
    }
}

// ----------------------------------------------------------------------------------------- metrics from columns

/// The independent oracle for the key's metric definitions, written out from design 2.8 with no call into `weightsim`.
struct Oracle {
    n: usize,
    sharpe: f64,
    cagr: f64,
    vol: f64,
    max_drawdown: f64,
}

fn oracle(dates: &[Date], r: &[f64]) -> Oracle {
    let n = r.len();
    let years = dates[0].days_until(dates[n - 1]) as f64 / 365.25;
    let ppy = n as f64 / years;
    let mean = r.iter().sum::<f64>() / n as f64;
    let var = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n as f64 - 1.0);
    let sd = var.sqrt();
    let (mut cum, mut peak, mut mdd) = (1.0f64, f64::NEG_INFINITY, f64::INFINITY);
    for x in r {
        cum *= 1.0 + x;
        peak = peak.max(cum);
        mdd = mdd.min(cum / peak - 1.0);
    }
    Oracle {
        n,
        sharpe: mean / sd * ppy.sqrt(),
        cagr: cum.powf(1.0 / years) - 1.0,
        vol: sd * ppy.sqrt(),
        max_drawdown: mdd,
    }
}

#[test]
fn metrics_recomputed_from_the_columns_equal_the_runs_own_metrics_bit_for_bit() {
    let w = world();
    let cases: [(&str, &ReplicationRun, &Panel, &weightsim_rules::ladder::fixtures::SleeveKey, bool); 2] =
        [(S1, &w.s1, &w.fx.etf_panel, &w.fx.s1, true), (S3, &w.s3, &w.fx.crypto_panel, &w.fx.s3, false)];
    for (rule, run, panel, key, is_etf) in cases {
        // The run's own SimResults, produced independently of `replicate`.
        let entry = entry_date(panel, key).unwrap();
        let sims = if is_etf {
            run_gross_and_net(panel, &FlatUntil::new(EtfTrendRule, entry), key).unwrap()
        } else {
            run_gross_and_net(panel, &FlatUntil::new(CryptoTrendRule, entry), key).unwrap()
        };
        let v = verify(&w.fx, rule, &run.gross, &run.net).unwrap();
        let (first, last) = (key.bars[0].date, key.bars[key.bars.len() - 1].date);
        for (sim, vb, cols, summary) in
            [(&sims.gross, &v.gross, &run.gross, &run.gross_summary), (&sims.net, &v.net, &run.net, &run.net_summary)]
        {
            let own = sim.metrics().expect("the run has metrics");
            assert_eq!(vb.metrics, own, "{rule}: metrics recomputed from the columns differ from the run's own");
            for (a, b) in [
                (vb.metrics.sharpe, own.sharpe),
                (vb.metrics.cagr, own.cagr),
                (vb.metrics.vol, own.vol),
                (vb.metrics.max_drawdown, own.max_drawdown),
                (vb.metrics.ppy, own.ppy),
            ] {
                assert_eq!(a.to_bits(), b.to_bits(), "{rule}: not bit-identical");
            }
            assert_eq!(Some(vb.window), sim.window, "{rule}: the derived window differs from the simulator's");
            assert_eq!(vb.report.flips_run, flips_by_key_convention(sim, first, last));
            assert_eq!(&vb.summary, summary);
            assert_eq!(vb.summary.n, own.n);

            // And they follow the key's definitions (ddof 1, ppy = n / years), computed independently from the columns.
            let win = vb.window.first_bar..=vb.window.last_bar;
            let o = oracle(&cols.dates[win.clone()], &cols.ret[win]);
            assert_eq!(o.n, vb.metrics.n);
            for (name, got, want) in [
                ("sharpe", vb.summary.sharpe, o.sharpe),
                ("cagr", vb.summary.cagr, o.cagr),
                ("vol", vb.summary.vol, o.vol),
                ("max_drawdown", vb.summary.max_drawdown, o.max_drawdown),
            ] {
                assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "{rule} {name}: {got} vs oracle {want}");
            }
        }
        // Net is charged: it differs from gross.
        assert!(v.net.summary.cagr < v.gross.summary.cagr && v.net.summary.sharpe < v.gross.summary.sharpe);
        assert_ne!(v.net.summary, v.gross.summary);
    }
}

// ------------------------------------------------------------------------------------------------- tamper classes

#[test]
fn tamper_one_ulp_in_one_return_is_a_digest_mismatch_at_that_cell() {
    let w = world();
    let t = w.s1.net.n_bars() / 2;
    let e = expect_err(tampered_s1(|_, n| n.ret[t] = next_ulp(n.ret[t])));
    assert_eq!(digest_mismatch(e), ("net", Some(Difference { column: "ret", bar: t })));
    let e = expect_err(tampered_s1(|g, _| g.ret[3] = next_ulp(g.ret[3])));
    assert_eq!(digest_mismatch(e), ("gross", Some(Difference { column: "ret", bar: 3 })));
    // One ulp in a weight cell names the bar and the column.
    let k = w.s1.net.n_assets();
    let e = expect_err(tampered_s1(|_, n| n.held_weights[t * k + 2] = next_ulp(n.held_weights[t * k + 2])));
    assert_eq!(digest_mismatch(e), ("net", Some(Difference { column: "held_weights", bar: t })));
    // And a crypto run.
    let mut g = w.s3.gross.clone();
    let t3 = g.n_bars() - 5;
    g.equity[t3] = next_ulp(g.equity[t3]);
    let e = expect_err(verify_fast(&w.fx, S3, &g, &w.s3.net));
    assert_eq!(digest_mismatch(e), ("gross", Some(Difference { column: "equity", bar: t3 })));
}

#[test]
fn tamper_every_column_by_one_ulp_is_caught_with_the_right_locator() {
    let w = world();
    let k = w.s1.net.n_assets();
    let t = 100;
    for name in SCALAR_COLS.into_iter().chain(MATRIX_COLS) {
        let per = if MATRIX_COLS.contains(&name) { k } else { 1 };
        let e = expect_err(tampered_s1(|_, n| {
            let col = float_col(n, name);
            let i = t * per + (per - 1);
            col[i] = next_ulp(col[i]);
        }));
        assert_eq!(digest_mismatch(e), ("net", Some(Difference { column: name, bar: t })), "{name}");
    }
    // The boolean flags.
    let e = expect_err(tampered_s1(|_, n| n.decision[t] = !n.decision[t]));
    assert_eq!(digest_mismatch(e), ("net", Some(Difference { column: "decision", bar: t })));
    let e = expect_err(tampered_s1(|g, _| g.refused[t] = !g.refused[t]));
    assert_eq!(digest_mismatch(e), ("gross", Some(Difference { column: "refused", bar: t })));
}

#[test]
fn tamper_a_date_is_a_digest_mismatch_and_out_of_order_dates_are_a_shape_error() {
    let w = world();
    // A date moved by one day into a gap between two bars stays ascending: only the digest can see it.
    let dates = &w.s1.net.dates;
    let t = (1..dates.len() - 1).find(|&t| dates[t].days_until(dates[t + 1]) >= 2).expect("a weekend gap");
    let e = expect_err(tampered_s1(|_, n| n.dates[t] = n.dates[t].add_days(1)));
    assert_eq!(digest_mismatch(e), ("net", Some(Difference { column: "dates", bar: t })));
    // A repeated date is not strictly ascending either.
    let e = expect_err(tampered_s1(|_, n| n.dates[11] = n.dates[10]));
    assert_eq!(e, VerifyError::Shape { basis: "net", error: ColumnsError::DatesNotAscending { bar: 11 } });
    // Two swapped dates are not ascending.
    let e = expect_err(tampered_s1(|g, _| g.dates.swap(10, 11)));
    // (bar 10 now holds the later date, so the first bar that is not after its predecessor is bar 11.)
    assert_eq!(e, VerifyError::Shape { basis: "gross", error: ColumnsError::DatesNotAscending { bar: 11 } });
}

#[test]
fn tamper_one_fixture_byte_breaks_the_trust_chain_with_a_pin_error() {
    let (files, ms, cs) = ready();
    let table: Vec<(&str, &[u8])> = files.files.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    let pins = Pins { manifest_sha256: ms, candles_sha256: cs };
    assert!(Fixtures::from_files(&table, pins).is_ok());
    // One byte of any listed file (the manifest is untouched, so its own pin still matches).
    for (i, (name, bytes)) in table.iter().enumerate() {
        if *name == F_MANIFEST {
            continue;
        }
        let mut changed = bytes.to_vec();
        let last = changed.len() - 2;
        changed[last] = if changed[last] == b'1' { b'2' } else { b'1' };
        let mut t2 = table.clone();
        t2[i] = (name, changed.as_slice());
        match Fixtures::from_files(&t2, pins) {
            Err(LadderError::Pin(_)) => {}
            other => panic!("{name}: expected a pin error, got {:?}", other.map(|_| "Ok")),
        }
    }
    // The manifest itself changed by one byte.
    let mut m = files.files[F_MANIFEST].clone();
    m.push(b'\n');
    let mut t2 = table.clone();
    let mi = table.iter().position(|(n, _)| *n == F_MANIFEST).unwrap();
    t2[mi] = (F_MANIFEST, m.as_slice());
    assert!(matches!(Fixtures::from_files(&t2, pins), Err(LadderError::Pin(_))));
    // A wrong pin.
    let zeros = "0".repeat(64);
    assert!(matches!(
        Fixtures::from_files(&table, Pins { manifest_sha256: &zeros, candles_sha256: cs }),
        Err(LadderError::Pin(_))
    ));
    assert!(matches!(
        Fixtures::from_files(&table, Pins { manifest_sha256: ms, candles_sha256: &zeros }),
        Err(LadderError::Pin(_))
    ));
}

/// Rescale the ETF closes of the second half of the data by a row-dependent factor and re-pin the set (the attacker
/// controls a whole fixture set, pins included). The manifest trust chain accepts it; only the re-run can object.
fn repinned_altered_fixtures() -> Fixtures {
    let (files, _, _) = ready();
    let mut f = SyntheticFiles { files: files.files.clone() };
    let csv = String::from_utf8(f.files[F_CANDLES].clone()).unwrap();
    let mut out = String::new();
    for (i, line) in csv.lines().enumerate() {
        let p: Vec<&str> = line.split(',').collect();
        if i > 2000 && p.len() == 3 && ETF.contains(&p[0]) {
            let close: f64 = p[2].parse().unwrap();
            out.push_str(&format!("{},{},{}\n", p[0], p[1], close * (1.0 + 0.01 * (i % 5) as f64)));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    f.files.insert(F_CANDLES.into(), out.into_bytes());
    let (f, ms, cs) = with_manifest(f);
    load(&f, &ms, &cs).expect("the re-pinned altered set loads")
}

#[test]
fn tamper_the_fixture_under_a_repinned_manifest_is_a_digest_mismatch_or_a_fixture_mismatch() {
    let w = world();
    let altered = repinned_altered_fixtures();
    assert_ne!(altered.manifest_sha256, w.fx.manifest_sha256);
    assert_ne!(altered.candles_sha256, w.fx.candles_sha256);
    // The genuine S1 run is not what the platform computes on the altered data.
    let e = expect_err(verify_fast(&altered, S1, &w.s1.gross, &w.s1.net));
    let (basis, diff) = digest_mismatch(e);
    assert_eq!(basis, "gross");
    assert!(diff.is_some());
    // If the run says which fixtures it used, that is caught first.
    let req = VerifyRequest { claims: w.s1.claims(), ..VerifyRequest::new(S1, &w.s1.gross, &w.s1.net) };
    match expect_err(verify_with(&altered, &req, &FAST)) {
        VerifyError::FixtureMismatch { what: "manifest_sha256", claimed, actual } => {
            assert_eq!(claimed, w.fx.manifest_sha256);
            assert_eq!(actual, altered.manifest_sha256);
        }
        other => panic!("expected FixtureMismatch, got {other:?}"),
    }
    let only_candles = VerifyRequest {
        claims: Claims { candles_sha256: Some(w.fx.candles_sha256.clone()), ..Claims::default() },
        ..req.clone()
    };
    assert!(matches!(
        expect_err(verify_with(&altered, &only_candles, &FAST)),
        VerifyError::FixtureMismatch { what: "candles_sha256", .. }
    ));
    // A run replicated on the altered data is consistent with it, but the altered data no longer reproduces the key.
    let own = replicate(&altered, S1).unwrap();
    assert_ne!(own.gross_sha256, w.s1.gross_sha256);
    assert!(matches!(expect_err(verify_fast(&altered, S1, &own.gross, &own.net)), VerifyError::TierFailed { .. }));
}

#[test]
fn tamper_the_rule_id_is_a_rule_id_mismatch_and_an_unknown_rule_is_typed() {
    let w = world();
    // The columns claim another rule.
    let e = expect_err(tampered_s1(|g, n| {
        g.rule_id = S3.into();
        n.rule_id = S3.into();
    }));
    assert_eq!(e, VerifyError::RuleIdMismatch { basis: "gross", expected: S1.into(), found: S3.into() });
    let e = expect_err(tampered_s1(|_, n| n.rule_id = "etf_trend_faber_v2".into()));
    assert_eq!(
        e,
        VerifyError::RuleIdMismatch { basis: "net", expected: S1.into(), found: "etf_trend_faber_v2".into() }
    );
    // The entry's rule is another rule than the run's.
    let e = expect_err(verify_fast(&w.fx, S3, &w.s1.gross, &w.s1.net));
    assert_eq!(e, VerifyError::RuleIdMismatch { basis: "gross", expected: S3.into(), found: S1.into() });
    // No such library rule.
    let e = expect_err(verify_fast(&w.fx, "momentum_12_1", &w.s1.gross, &w.s1.net));
    assert_eq!(e, VerifyError::UnknownRule { rule_id: "momentum_12_1".into() });
    assert_eq!(
        replicate(&w.fx, "momentum_12_1").unwrap_err(),
        ReplicateError::UnknownRule { rule_id: "momentum_12_1".into() }
    );
}

#[test]
fn tamper_the_impl_version_is_an_impl_version_mismatch() {
    let w = world();
    let old = w.s1.rule_impl_version.clone();
    let bumped = old.replace("0.1.0", "0.2.0");
    assert_ne!(old, bumped);
    let e = expect_err(tampered_s1(|_, n| n.rule_impl_version = bumped.clone()));
    assert_eq!(e, VerifyError::ImplVersionMismatch { basis: "net", expected: old.clone(), found: bumped.clone() });
    // The flat-until date is part of the version: another entry bar is another implementation.
    let other_entry = old.replace("flat until", "flat until 1999-");
    let e = expect_err(tampered_s1(|g, _| g.rule_impl_version = other_entry.clone()));
    assert!(matches!(e, VerifyError::ImplVersionMismatch { basis: "gross", .. }));
}

#[test]
fn tamper_the_cost_preset_is_a_cost_model_mismatch_or_a_digest_mismatch() {
    let w = world();
    // Verified under another configured preset.
    let req = VerifyRequest {
        config: ReplicationConfig { cost_model_id: "zero".into() },
        ..VerifyRequest::new(S1, &w.s1.gross, &w.s1.net)
    };
    let e = expect_err(verify_with(&w.fx, &req, &FAST));
    assert_eq!(
        e,
        VerifyError::CostModelMismatch { basis: "net", expected: "zero".into(), found: DEFAULT_COST_MODEL_ID.into() }
    );
    // A net series relabelled as zero cost.
    let e = expect_err(tampered_s1(|_, n| n.cost_model_id = "zero".into()));
    assert_eq!(
        e,
        VerifyError::CostModelMismatch { basis: "net", expected: DEFAULT_COST_MODEL_ID.into(), found: "zero".into() }
    );
    // A gross series claiming to be charged.
    let e = expect_err(tampered_s1(|g, _| g.cost_model_id = DEFAULT_COST_MODEL_ID.into()));
    assert_eq!(
        e,
        VerifyError::CostModelMismatch { basis: "gross", expected: "zero".into(), found: DEFAULT_COST_MODEL_ID.into() }
    );
    // The zero-cost series passed off as the charged one under the right label: the numbers give it away.
    let e = expect_err(tampered_s1(|g, n| {
        *n = g.clone();
        n.cost_model_id = DEFAULT_COST_MODEL_ID.into();
    }));
    let (basis, diff) = digest_mismatch(e);
    assert_eq!(basis, "net");
    assert!(diff.is_some());
    // An unknown preset is refused before anything runs.
    let req = VerifyRequest {
        config: ReplicationConfig { cost_model_id: "venue_ibkr_v9".into() },
        ..VerifyRequest::new(S1, &w.s1.gross, &w.s1.net)
    };
    assert_eq!(
        expect_err(verify_with(&w.fx, &req, &FAST)),
        VerifyError::UnknownCostPreset { cost_model_id: "venue_ibkr_v9".into() }
    );
    assert!(matches!(
        replicate_with(&w.fx, S1, &ReplicationConfig { cost_model_id: "nope".into() }),
        Err(ReplicateError::UnknownCostPreset { .. })
    ));
}

#[test]
fn a_run_replicated_at_zero_cost_is_consistent_but_does_not_reproduce_the_charged_key() {
    let w = world();
    let cfg = ReplicationConfig { cost_model_id: "zero".into() };
    let free = replicate_with(&w.fx, S1, &cfg).unwrap();
    // Both runs are zero cost, so they carry the same series.
    assert_eq!(free.gross_sha256, free.net_sha256);
    assert_ne!(free.net_sha256, w.s1.net_sha256);
    let req = VerifyRequest { config: cfg, ..VerifyRequest::new(S1, &free.gross, &free.net) };
    match expect_err(verify_with(&w.fx, &req, &FAST)) {
        VerifyError::TierFailed { failures, run } => {
            assert!(failures.contains(&TierFailure { scope: "net", tier: "tier2_identity" }), "{failures:?}");
            assert!(
                !failures.iter().any(|f| f.scope == "gross"),
                "the gross run still reproduces the key: {failures:?}"
            );
            assert!(run.gross.report.tier2_pass && !run.net.report.tier2_pass);
        }
        other => panic!("expected TierFailed, got {other:?}"),
    }
}

#[test]
fn tamper_dropping_the_last_row_is_a_truncation() {
    let w = world();
    let n = w.s1.gross.n_bars();
    let e = expect_err(tampered_s1(|g, n| {
        drop_last_row(g);
        drop_last_row(n);
    }));
    assert_eq!(e, VerifyError::Truncated { basis: "gross", expected_bars: n, found_bars: n - 1 });
    // Only the net series truncated.
    let e = expect_err(tampered_s1(|_, nn| drop_last_row(nn)));
    assert_eq!(e, VerifyError::Truncated { basis: "net", expected_bars: n, found_bars: n - 1 });
    // The first row, or many rows.
    let e = expect_err(tampered_s1(|g, _| drop_first_row(g)));
    assert_eq!(e, VerifyError::Truncated { basis: "gross", expected_bars: n, found_bars: n - 1 });
    let e = expect_err(tampered_s1(|_, nn| {
        for _ in 0..50 {
            drop_last_row(nn);
        }
    }));
    assert_eq!(e, VerifyError::Truncated { basis: "net", expected_bars: n, found_bars: n - 50 });
    // A padded series is not a truncation but is refused just as typed.
    let e = expect_err(tampered_s1(|g, _| {
        let last = g.n_bars() - 1;
        let k = g.n_assets();
        let d = g.dates[last].add_days(1);
        g.dates.push(d);
        for name in SCALAR_COLS {
            let v = float_col(g, name)[last];
            float_col(g, name).push(v);
        }
        g.decision.push(false);
        g.refused.push(false);
        for name in MATRIX_COLS {
            let row: Vec<f64> = float_col(g, name)[last * k..].to_vec();
            float_col(g, name).extend(row);
        }
    }));
    assert_eq!(e, VerifyError::ExtraBars { basis: "gross", expected_bars: n, found_bars: n + 1 });
    // The crypto run too.
    let mut g = w.s3.gross.clone();
    drop_last_row(&mut g);
    let n3 = w.s3.gross.n_bars();
    assert_eq!(
        expect_err(verify_fast(&w.fx, S3, &g, &w.s3.net)),
        VerifyError::Truncated { basis: "gross", expected_bars: n3, found_bars: n3 - 1 }
    );
}

#[test]
fn tamper_one_short_column_is_a_shape_error_never_a_panic() {
    let w = world();
    let n = w.s1.net.n_bars();
    let k = w.s1.net.n_assets();
    for name in SCALAR_COLS {
        let e = expect_err(tampered_s1(|_, nn| {
            float_col(nn, name).pop();
        }));
        assert_eq!(
            e,
            VerifyError::Shape {
                basis: "net",
                error: ColumnsError::LengthMismatch { column: name, expected: n, found: n - 1 }
            }
        );
    }
    for name in MATRIX_COLS {
        let e = expect_err(tampered_s1(|g, _| {
            float_col(g, name).pop();
        }));
        assert_eq!(
            e,
            VerifyError::Shape {
                basis: "gross",
                error: ColumnsError::LengthMismatch { column: name, expected: n * k, found: n * k - 1 }
            }
        );
    }
    let e = expect_err(tampered_s1(|_, nn| {
        nn.decision.pop();
    }));
    assert!(matches!(
        e,
        VerifyError::Shape { basis: "net", error: ColumnsError::LengthMismatch { column: "decision", .. } }
    ));
    let e = expect_err(tampered_s1(|_, nn| {
        nn.refused.pop();
    }));
    assert!(matches!(
        e,
        VerifyError::Shape { basis: "net", error: ColumnsError::LengthMismatch { column: "refused", .. } }
    ));
    // Empty columns.
    let empty = SeriesColumns { dates: vec![], symbols: vec![], ..w.s1.net.clone() };
    assert_eq!(
        expect_err(verify_fast(&w.fx, S1, &empty, &w.s1.net)),
        VerifyError::Shape { basis: "gross", error: ColumnsError::Empty }
    );
}

#[test]
fn tamper_swapped_gross_and_net_columns_are_caught() {
    let w = world();
    // The two whole series handed over in the wrong slots.
    let e = expect_err(verify_fast(&w.fx, S1, &w.s1.net, &w.s1.gross));
    assert_eq!(
        e,
        VerifyError::CostModelMismatch { basis: "gross", expected: "zero".into(), found: DEFAULT_COST_MODEL_ID.into() }
    );
    // Return columns exchanged between the two series (labels intact).
    let e = expect_err(tampered_s1(|g, n| std::mem::swap(&mut g.ret, &mut n.ret)));
    let (basis, diff) = digest_mismatch(e);
    assert_eq!(basis, "gross");
    assert_eq!(diff.unwrap().column, "ret");
    // Post-cost and pre-cost returns exchanged inside the net series.
    let e = expect_err(tampered_s1(|_, n| std::mem::swap(&mut n.ret, &mut n.ret_pre_cost)));
    let (basis, diff) = digest_mismatch(e);
    assert_eq!(basis, "net");
    assert_eq!(diff.unwrap().column, "ret");
    // Equity exchanged.
    let e = expect_err(tampered_s1(|g, n| std::mem::swap(&mut g.equity, &mut n.equity)));
    assert_eq!(digest_mismatch(e).0, "gross");
}

#[test]
fn tamper_a_nan_or_an_infinity_is_a_non_finite_error() {
    let w = world();
    let k = w.s1.net.n_assets();
    let e = expect_err(tampered_s1(|_, n| n.ret[40] = f64::NAN));
    assert_eq!(e, VerifyError::NonFinite { basis: "net", column: "ret", bar: 40 });
    let e = expect_err(tampered_s1(|g, _| g.equity[7] = f64::INFINITY));
    assert_eq!(e, VerifyError::NonFinite { basis: "gross", column: "equity", bar: 7 });
    let e = expect_err(tampered_s1(|g, _| g.units[70 * k + 3] = f64::NEG_INFINITY));
    assert_eq!(e, VerifyError::NonFinite { basis: "gross", column: "units", bar: 70 });
    let e = expect_err(tampered_s1(|_, n| n.held_weights[5 * k + 1] = f64::NAN));
    assert_eq!(e, VerifyError::NonFinite { basis: "net", column: "held_weights", bar: 5 });
    // Every float column, NaN at the last cell.
    for name in SCALAR_COLS.into_iter().chain(MATRIX_COLS) {
        let e = expect_err(tampered_s1(|_, n| {
            let col = float_col(n, name);
            let last = col.len() - 1;
            col[last] = f64::NAN;
        }));
        assert!(matches!(e, VerifyError::NonFinite { basis: "net", column, .. } if column == name), "{name}: {e:?}");
    }
}

#[test]
fn tamper_a_claimed_digest_is_a_claimed_digest_mismatch() {
    let w = world();
    let claims = w.s1.claims();
    let with = |claims: Claims, g: &SeriesColumns, n: &SeriesColumns| {
        verify_with(&w.fx, &VerifyRequest { claims, ..VerifyRequest::new(S1, g, n) }, &FAST)
    };
    // Genuine columns, a digest that is not theirs.
    let e = expect_err(with(Claims { gross_sha256: Some("0".repeat(64)), ..claims.clone() }, &w.s1.gross, &w.s1.net));
    assert!(matches!(e, VerifyError::ClaimedDigestMismatch { basis: "gross", .. }));
    let mut flipped = w.s1.net_sha256.clone();
    flipped.replace_range(10..11, if &flipped[10..11] == "a" { "b" } else { "a" });
    let e = expect_err(with(Claims { net_sha256: Some(flipped.clone()), ..claims.clone() }, &w.s1.gross, &w.s1.net));
    assert_eq!(
        e,
        VerifyError::ClaimedDigestMismatch { basis: "net", claimed: flipped, recomputed: w.s1.net_sha256.clone() }
    );
    // Columns altered after the digest was taken.
    let mut n = w.s1.net.clone();
    n.ret[9] = next_ulp(n.ret[9]);
    let e = expect_err(with(claims.clone(), &w.s1.gross, &n));
    assert_eq!(
        e,
        VerifyError::ClaimedDigestMismatch {
            basis: "net",
            claimed: w.s1.net_sha256.clone(),
            recomputed: n.digest().unwrap()
        }
    );
    // The forger recomputes the digest over the altered columns: the re-run still refuses.
    let forged = Claims { net_sha256: Some(n.digest().unwrap()), ..claims };
    let e = expect_err(with(forged, &w.s1.gross, &n));
    assert_eq!(digest_mismatch(e), ("net", Some(Difference { column: "ret", bar: 9 })));
}

#[test]
fn tamper_a_claimed_summary_number_is_a_claimed_metrics_mismatch() {
    let w = world();
    let good = w.s1.claims();
    let check = |claims: Claims| {
        verify_with(&w.fx, &VerifyRequest { claims, ..VerifyRequest::new(S1, &w.s1.gross, &w.s1.net) }, &FAST)
    };
    assert!(check(good.clone()).is_ok());
    let net = w.s1.net_summary;
    let gross = w.s1.gross_summary;
    let metric_of = |e: VerifyError| match e {
        VerifyError::ClaimedMetricsMismatch { basis, metric, claimed, recomputed } => {
            assert_ne!(claimed.to_bits(), recomputed.to_bits());
            (basis, metric)
        }
        other => panic!("expected ClaimedMetricsMismatch, got {other:?}"),
    };
    let edits: Vec<(&str, &str, RunSummary)> = vec![
        ("net", "sharpe", RunSummary { sharpe: next_ulp(net.sharpe), ..net }),
        ("net", "sharpe", RunSummary { sharpe: net.sharpe + 0.01, ..net }),
        ("net", "vol", RunSummary { vol: next_ulp(net.vol), ..net }),
        ("net", "max_drawdown", RunSummary { max_drawdown: net.max_drawdown * 0.5, ..net }),
        ("net", "cagr", RunSummary { cagr: net.cagr + 1e-9, ..net }),
        ("net", "flips", RunSummary { flips: net.flips + 1, ..net }),
        ("net", "n", RunSummary { n: net.n - 1, ..net }),
        ("net", "first_date", RunSummary { first_date: net.first_date.add_days(1), ..net }),
        ("net", "last_date", RunSummary { last_date: net.last_date.add_days(-1), ..net }),
    ];
    for (basis, metric, edited) in edits {
        let e = expect_err(check(Claims { net_summary: Some(edited), ..good.clone() }));
        assert_eq!(metric_of(e), (basis, metric));
    }
    // The gross claim is checked against the gross recomputation, the net claim against the net one.
    let e = expect_err(check(Claims {
        gross_summary: Some(RunSummary { sharpe: gross.sharpe * 1.001, ..gross }),
        ..good.clone()
    }));
    assert_eq!(metric_of(e), ("gross", "sharpe"));
    let e = expect_err(check(Claims { net_summary: Some(gross), ..good.clone() }));
    assert_eq!(metric_of(e).0, "net");
    // The CAGR alone is compared with a tolerance (powf); anything inside it is accepted.
    assert!(check(Claims { net_summary: Some(RunSummary { cagr: net.cagr + 1e-13, ..net }), ..good }).is_ok());
    assert_eq!(CLAIMED_CAGR_TOL, 1e-12);
}

#[test]
fn tamper_symbols_or_metric_definitions_are_typed() {
    let w = world();
    let e = expect_err(tampered_s1(|g, n| {
        g.symbols.reverse();
        n.symbols.reverse();
    }));
    match e {
        VerifyError::SymbolsMismatch { basis: "gross", expected, found } => {
            assert_eq!(expected, ETF.to_vec());
            assert_eq!(found, ETF.iter().rev().map(|s| s.to_string()).collect::<Vec<_>>());
        }
        other => panic!("{other:?}"),
    }
    let e = expect_err(tampered_s1(|_, n| n.metric_definitions = "population_var_365".into()));
    assert_eq!(
        e,
        VerifyError::MetricDefinitionsMismatch {
            basis: "net",
            expected: "answer_key_v1".into(),
            found: "population_var_365".into()
        }
    );
    // A wrong-universe run: the crypto columns presented as the ETF rule's, with the labels changed to match.
    let mut g = w.s3.gross.clone();
    let mut n = w.s3.net.clone();
    for c in [&mut g, &mut n] {
        c.rule_id = S1.into();
        c.rule_impl_version = w.s1.rule_impl_version.clone();
    }
    let e = expect_err(verify_fast(&w.fx, S1, &g, &n));
    assert!(matches!(e, VerifyError::SymbolsMismatch { basis: "gross", .. }));
}

#[test]
fn a_genuine_run_that_does_not_reproduce_the_key_is_tier_failed_with_the_numbers() {
    let w = world();
    // Tier II: one key cell off by 1e-7.
    let mut fx = w.fx.clone();
    fx.s1.bars[10].ret_net += 1e-7;
    match expect_err(verify_fast(&fx, S1, &w.s1.gross, &w.s1.net)) {
        VerifyError::TierFailed { failures, run } => {
            assert_eq!(failures, vec![TierFailure { scope: "net", tier: "tier2_identity" }]);
            assert!(run.gross.report.tier2_pass && !run.net.report.tier2_pass);
            assert!(run.net.report.cmp.max_abs_ret_diff > 9e-8);
            assert!(run.net.summary.sharpe.is_finite());
        }
        other => panic!("{other:?}"),
    }
    // Tier I trades band.
    let mut fx = w.fx.clone();
    fx.s1.flips += 40;
    match expect_err(verify_fast(&fx, S1, &w.s1.gross, &w.s1.net)) {
        VerifyError::TierFailed { failures, .. } => {
            assert!(failures.contains(&TierFailure { scope: "gross", tier: "tier1_trades" }), "{failures:?}");
            assert!(failures.contains(&TierFailure { scope: "net", tier: "tier1_trades" }), "{failures:?}");
        }
        other => panic!("{other:?}"),
    }
    // Tier III weights: 20% of the key's target cells are wrong.
    let mut fx = w.fx.clone();
    for b in &mut fx.s1.bars {
        b.w_target[0] += 0.2;
    }
    match expect_err(verify_fast(&fx, S1, &w.s1.gross, &w.s1.net)) {
        VerifyError::TierFailed { failures, .. } => {
            assert!(failures.contains(&TierFailure { scope: "gross", tier: "tier3_weights" }), "{failures:?}");
            assert!(failures.contains(&TierFailure { scope: "net", tier: "tier3_weights" }), "{failures:?}");
        }
        other => panic!("{other:?}"),
    }
    // Tier I bands: the key's net returns shifted enough to move the Sharpe band, so the bands fail on net.
    let mut fx = w.fx.clone();
    for b in &mut fx.s1.bars {
        b.ret_net = b.ret_net * 0.5 + 0.0005;
    }
    match expect_err(verify_fast(&fx, S1, &w.s1.gross, &w.s1.net)) {
        VerifyError::TierFailed { failures, .. } => {
            assert!(failures.contains(&TierFailure { scope: "net", tier: "tier1_bands" }), "{failures:?}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn tier_iv_failure_is_a_tier_failed_and_can_be_switched_off() {
    let w = world();
    // The key says another tier catches a mutant than the one that does: the backtester does not reproduce it.
    let mut fx = w.fx.clone();
    let m = fx.expected_mutants.iter_mut().find(|m| m.name == "s1_one_bar_late").unwrap();
    m.caught_by = vec!["tier1_bands".into()];
    match expect_err(verify(&fx, S1, &w.s1.gross, &w.s1.net)) {
        VerifyError::TierFailed { failures, run } => {
            assert_eq!(failures, vec![TierFailure { scope: "tier4", tier: "matches_mutants_json" }]);
            let t4 = run.tier4.unwrap();
            assert!(t4.all_caught() && !t4.all_match_recorded() && !t4.passed());
            // Only the rule's sleeve is run: three S1 mutants.
            assert_eq!(t4.mutants.len(), 3);
            assert!(t4.mutants.iter().all(|m| m.name.starts_with("s1_")));
        }
        other => panic!("{other:?}"),
    }
    // A mutant with no entry in mutants.json is a failure too.
    let mut fx = w.fx.clone();
    fx.expected_mutants.retain(|m| m.name != "s1_daily_rebalanced_20");
    assert!(matches!(expect_err(verify(&fx, S1, &w.s1.gross, &w.s1.net)), VerifyError::TierFailed { .. }));
    // With Tier IV off the same fixtures verify (the run itself is fine), and the report says it was not run.
    let v = verify_fast(&fx, S1, &w.s1.gross, &w.s1.net).unwrap();
    assert!(v.tier4.is_none());
    // The crypto sleeve has five mutants.
    assert_eq!(tier4(&w.fx, weightsim_rules::ladder::mutants::MutantSleeve::S3).unwrap().mutants.len(), 5);
    assert!(tier4(&w.fx, weightsim_rules::ladder::mutants::MutantSleeve::S1).unwrap().passed());
    // The report predicates.
    let empty = Tier4Report { sleeve_code: "S1", mutants: vec![] };
    assert!(!empty.all_caught() && !empty.all_match_recorded() && !empty.passed());
    let uncaught = Tier4Report {
        sleeve_code: "S1",
        mutants: vec![Tier4Mutant { name: "x", caught_by: vec![], mismatches: vec![] }],
    };
    assert!(!uncaught.all_caught() && uncaught.all_match_recorded() && !uncaught.passed());
}

// ------------------------------------------------------------------------ properties: determinism, no side effects

#[test]
fn verify_is_deterministic_and_has_no_side_effects() {
    let w = world();
    let (g, n) = (w.s1.gross.clone(), w.s1.net.clone());
    let fx_before = format!("{:?}", w.fx);
    let a = verify(&w.fx, S1, &g, &n).unwrap();
    let b = verify(&w.fx, S1, &g, &n).unwrap();
    assert_eq!(a, b);
    assert_eq!(g, w.s1.gross);
    assert_eq!(n, w.s1.net);
    assert_eq!(format!("{:?}", w.fx), fx_before);
    // Errors are deterministic too.
    let mut bad = n.clone();
    bad.cash[3] = next_ulp(bad.cash[3]);
    assert_eq!(verify_fast(&w.fx, S1, &g, &bad).unwrap_err(), verify_fast(&w.fx, S1, &g, &bad).unwrap_err());
    // Replication is deterministic: same digests, same columns.
    let again = replicate(&w.fx, S1).unwrap();
    assert_eq!(
        (again.gross_sha256.as_str(), again.net_sha256.as_str()),
        (w.s1.gross_sha256.as_str(), w.s1.net_sha256.as_str())
    );
    assert_eq!(again.gross, w.s1.gross);
    assert_eq!(again.net, w.s1.net);
    // A fixture set loaded twice, or on other threads, gives the same verdict.
    let verdicts: Vec<Result<VerifiedRun, VerifyError>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..3).map(|_| s.spawn(|| verify_fast(&w.fx, S1, &g, &n))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for v in verdicts {
        assert_eq!(v.unwrap().gross, a.gross);
    }
}

/// A tiny deterministic generator (no dependency).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

#[test]
fn no_single_cell_change_is_ever_accepted_and_the_locator_names_the_cell() {
    let w = world();
    let mut rng = Lcg(0x5EED);
    let k = w.s1.net.n_assets();
    let names: Vec<&str> = SCALAR_COLS.into_iter().chain(MATRIX_COLS).collect();
    for round in 0..300 {
        let name = names[rng.next() as usize % names.len()];
        let gross_side = rng.next() & 1 == 0;
        let per = if MATRIX_COLS.contains(&name) { k } else { 1 };
        let (mut g, mut n) = (w.s1.gross.clone(), w.s1.net.clone());
        let col = float_col(if gross_side { &mut g } else { &mut n }, name);
        let i = rng.next() as usize % col.len();
        col[i] = next_ulp(col[i]);
        let e = expect_err(verify_fast(&w.fx, S1, &g, &n));
        let (basis, diff) = digest_mismatch(e);
        assert_eq!(basis, if gross_side { "gross" } else { "net" }, "round {round}");
        assert_eq!(diff, Some(Difference { column: name, bar: i / per }), "round {round}: {name}[{i}]");
    }
}

// ---------------------------------------------------------------------------- loaders: bytes versus files, rules

#[test]
fn the_bytes_loader_and_the_file_loader_give_identical_fixtures_and_digests() {
    let (files, ms, cs) = ready();
    let pins = Pins { manifest_sha256: ms, candles_sha256: cs };
    let table: Vec<(&str, &[u8])> = files.files.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
    let from_bytes = Fixtures::from_files(&table, pins).unwrap();

    let dir = std::env::temp_dir().join(format!("wsr-verify-{}-{}", std::process::id(), &ms[..8]));
    let _ = std::fs::remove_dir_all(&dir);
    for (name, bytes) in &files.files {
        let p = dir.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }
    let from_dir = Fixtures::from_dir_with_pins(&dir, pins).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let from_reader = load(files, ms, cs).unwrap();

    for other in [&from_dir, &from_reader] {
        assert_eq!(from_bytes.manifest_sha256, other.manifest_sha256);
        assert_eq!(from_bytes.candles_sha256, other.candles_sha256);
        assert_eq!(from_bytes.verified_files, other.verified_files);
        assert_eq!(format!("{from_bytes:?}"), format!("{other:?}"));
    }
    assert_eq!(from_bytes.verified_files.len(), files.files.len() - 1);
    // Identical fixtures give identical runs and identical ladder digests.
    for rule in [S1, S3] {
        let a = replicate(&from_bytes, rule).unwrap();
        let b = replicate(&from_dir, rule).unwrap();
        assert_eq!((&a.gross_sha256, &a.net_sha256), (&b.gross_sha256, &b.net_sha256));
        assert_eq!(a.gross, b.gross);
        assert!(verify_fast(&from_dir, rule, &a.gross, &a.net).is_ok());
    }
    let la = run_ladder_with(&from_bytes, &NO_CANARIES).unwrap();
    let lb = run_ladder_with(&from_dir, &NO_CANARIES).unwrap();
    assert_eq!(la.digest, lb.digest);
    // Every manifest-listed file is really read from the table: dropping any one is an i/o error, not a silent skip.
    for skip in 0..table.len() {
        let t2: Vec<(&str, &[u8])> = table.iter().enumerate().filter(|(i, _)| *i != skip).map(|(_, e)| *e).collect();
        let name = table[skip].0;
        match Fixtures::from_files(&t2, pins) {
            Err(LadderError::Io(_)) | Err(LadderError::Pin(_)) => {}
            other => panic!("without {name}: expected an error, got {:?}", other.map(|_| "Ok")),
        }
    }
    // Two files with their contents swapped are a pin error, and a repeated file name is refused.
    let (i, j) = (
        table.iter().position(|(n, _)| *n == "key/S1_etf_trend_faber_perbar.csv").unwrap(),
        table.iter().position(|(n, _)| *n == "key/S3_crypto_trend_100d_perbar.csv").unwrap(),
    );
    let mut swapped = table.clone();
    swapped[i].1 = table[j].1;
    swapped[j].1 = table[i].1;
    assert!(matches!(Fixtures::from_files(&swapped, pins), Err(LadderError::Pin(_))));
    let mut dup = table.clone();
    dup.push(table[0]);
    assert!(matches!(Fixtures::from_files(&dup, pins), Err(LadderError::Io(_))));
    // An unlisted extra file is ignored, as with the directory loader.
    let mut extra = table.clone();
    extra.push(("notes.txt", b"not in the manifest"));
    assert!(Fixtures::from_files(&extra, pins).is_ok());
}

#[test]
fn rule_facts_are_the_adapters_declared_facts() {
    let e = rule_facts(S1).unwrap();
    assert_eq!((e.id, e.sleeve_code), (S1, "S1"));
    assert_eq!(e.universe, ETF.to_vec());
    assert_eq!(e.decision_schedule, DecisionSchedule::LastBarOfMonth);
    assert_eq!(e.rebalance_policy, RebalancePolicy::OnDecision);
    assert_eq!(e.min_history_bars, 1);
    assert_eq!(e.declared_parameters["sma_month_ends"], "10");
    assert_eq!(e.declared_parameters["weight_per_instrument"], "0.2");
    let c = rule_facts(S3).unwrap();
    assert_eq!((c.id, c.sleeve_code), (S3, "S3"));
    assert_eq!(c.universe, CRY.to_vec());
    assert_eq!(c.decision_schedule, DecisionSchedule::Daily);
    assert_eq!(c.rebalance_policy, RebalancePolicy::EveryBar);
    assert_eq!(c.min_history_bars, 100);
    assert_eq!(c.declared_parameters["sma_days"], "100");
    assert_eq!(c.declared_parameters["schedule"], "\"daily\"");
    assert!(rule_facts("etf_trend").is_none() && rule_facts("").is_none());
    assert_eq!(LIBRARY_RULE_IDS, [S1, S3]);
    for id in LIBRARY_RULE_IDS {
        assert!(rule_facts(id).is_some());
    }
}

#[test]
fn replicate_packages_the_ladders_own_base_runs() {
    let w = world();
    for run in [&w.s1, &w.s3] {
        assert_eq!(run.cost_model_id, DEFAULT_COST_MODEL_ID);
        assert_eq!(run.gross.cost_model_id, GROSS_COST_MODEL_ID);
        assert_eq!(run.net.cost_model_id, DEFAULT_COST_MODEL_ID);
        assert_eq!(run.gross.metric_definitions, "answer_key_v1");
        assert_eq!(run.manifest_sha256, w.fx.manifest_sha256);
        assert_eq!(run.candles_sha256, w.fx.candles_sha256);
        assert_eq!(run.gross.rule_id, run.rule_id);
        // Warm-up refusals only (the ladder aborts on any other); none after the first decision.
        assert!(run.refusals.iter().all(|r| r.kind == RefusalKind::Warmup), "{:?}", run.refusals);
        let first = run.gross.decision.iter().position(|d| *d).unwrap();
        assert!(run.refusals.iter().all(|r| r.bar < first));
        // Costs are really charged in the net series and only there.
        assert!(run.gross.cost.iter().all(|c| *c == 0.0));
        assert!(run.net.cost.iter().any(|c| *c > 0.0));
        assert!(run.gross_summary.flips > 0 && run.gross_summary.flips == run.net_summary.flips);
    }
    // Simulator provenance is constant.
    assert!(SIMULATOR_VERSION.starts_with("weightsim "));
}

// ------------------------------------------------ the analysis half: tiers on stored columns, without the re-run

#[test]
fn analyze_columns_recomputes_tiers_from_the_columns_and_each_edit_fails_the_tier_the_amendment_names() {
    let w = world();
    let n = w.s1.net.n_bars();
    let k = w.s1.net.n_assets();
    let t = n / 2;
    let analyze = |basis: Basis, c: &SeriesColumns| analyze_columns(&w.fx, S1, basis, c);

    // The genuine series pass every tier and reproduce the verified run's numbers.
    let g = analyze(Basis::Gross, &w.s1.gross).unwrap();
    let nn = analyze(Basis::Net, &w.s1.net).unwrap();
    let v = verify(&w.fx, S1, &w.s1.gross, &w.s1.net).unwrap();
    assert_eq!((&g, &nn), (&v.gross, &v.net));
    assert!(nn.report.tier1_pass && nn.report.tier2_pass && nn.report.tier3_pass);

    // One ulp in one return is far inside Tier II (1e-9): the tiers cannot see it, only the digest can.
    let mut c = w.s1.net.clone();
    c.ret[t] = next_ulp(c.ret[t]);
    let a = analyze(Basis::Net, &c).unwrap();
    assert!(a.report.tier1_pass && a.report.tier2_pass && a.report.tier3_pass);
    assert!(matches!(verify_fast(&w.fx, S1, &w.s1.gross, &c), Err(VerifyError::DigestMismatch { .. })));

    // A standing-target cell off by 1e-8: Tier II only (Tier III tolerates 1e-6, the Tier I bands see nothing).
    // (A held asset on a bar that is not a decision bar, so the sign of no decision changes and the flips stay equal.)
    let (bar, asset) = (t..n)
        .flat_map(|bar| (0..k).map(move |asset| (bar, asset)))
        .find(|&(bar, asset)| w.s1.gross.target_weights[bar * k + asset] > 0.1 && !w.s1.gross.decision[bar])
        .expect("a held asset on a non-decision bar");
    let mut c = w.s1.gross.clone();
    c.target_weights[bar * k + asset] += 1e-8;
    let a = analyze(Basis::Gross, &c).unwrap();
    assert!(a.report.tier1_pass && a.report.tier3_pass && !a.report.tier2_pass);
    assert!(a.report.cmp.max_abs_w_target_diff.unwrap() > 9e-9);

    // The return series shifted one bar (every return earned a day late): Tier II fails.
    let mut c = w.s1.net.clone();
    c.ret.rotate_right(1);
    let a = analyze(Basis::Net, &c).unwrap();
    assert!(!a.report.tier2_pass);

    // A truncated series no longer covers the key's bars.
    let mut c = w.s1.net.clone();
    drop_last_row(&mut c);
    let a = analyze(Basis::Net, &c).unwrap();
    assert!(!a.report.cmp.covers_key_exactly() && !a.report.window_matches_key);
    assert_eq!(a.window.last_bar + 2, n);

    // The gross series analysed as the net one fails Tier II (no costs), and vice versa.
    assert!(!analyze(Basis::Net, &w.s1.gross).unwrap().report.tier2_pass);
    assert!(!analyze(Basis::Gross, &w.s1.net).unwrap().report.tier2_pass);

    // Malformed input is a typed error, not a panic.
    let mut c = w.s1.net.clone();
    c.ret[3] = f64::NAN;
    assert_eq!(analyze(Basis::Net, &c).unwrap_err(), VerifyError::NonFinite { basis: "net", column: "ret", bar: 3 });
    let mut c = w.s1.gross.clone();
    c.cash.pop();
    assert!(matches!(analyze(Basis::Gross, &c), Err(VerifyError::Shape { basis: "gross", .. })));
    assert!(matches!(analyze_columns(&w.fx, "nope", Basis::Net, &w.s1.net), Err(VerifyError::UnknownRule { .. })));
}
