//! W7.4 (council R10): the live-realistic execution layer. The certification model must be the identity on the
//! certified path (bit-identical series digests), a delay of 1 shifts the applied weight by exactly one bar, slippage
//! reduces the net result by exactly turnover x bps, and the crypto default (delay 0) equals the certification path
//! except for the slippage.

mod common;

use common::*;
use weightsim::harness::compare_prefix;
use weightsim::*;

const A: [&str; 1] = ["A"];

fn five_bar_panel() -> Panel {
    let dates = weekdays(d("2020-01-27"), 5);
    Panel::new(vec!["A".to_string()], dates, vec![vec![100.0, 110.0, 99.0, 108.9, 100.0]]).unwrap()
}

/// The weight the rule decides at bar `t` (the last visible bar).
fn w_at(t: usize) -> f64 {
    [0.5, 1.0, 0.0, 0.5, 1.0][t]
}

fn five_bar_rule() -> FnRule {
    FnRule::new(&A, DecisionSchedule::Daily, RebalancePolicy::EveryBar, 1, |h| Ok(vec![w_at(h.len() - 1)]))
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn cert_cfg() -> SimConfig {
    SimConfig { cost: CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, ..SimConfig::default() }
}

// ----------------------------------------------------------------------------------------- certification identity

#[test]
fn certification_model_yields_bit_identical_series_digests_on_both_sleeves_and_both_presets() {
    let s1 = synth_panel(&ETF, 2600, 11, "2015-01-02");
    let s3 = synth_panel(&CRY, 2200, 13, "2015-01-01");
    for cost in [CostModel::ZERO, CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE] {
        let cfg = SimConfig { cost, ..SimConfig::default() };
        let under = ExecutionModel::certification().apply_to(&cfg).unwrap();
        let (a, b) = (simulate(&s1, &S1TestRule, &cfg).unwrap(), simulate(&s1, &S1TestRule, &under).unwrap());
        assert_eq!(a.series_sha256, b.series_sha256, "S1 {}", cost.id);
        assert!(compare_prefix(&a, &b, a.n_bars() - 1).is_empty());
        assert_eq!(b.cost_model_id, cost.id);
        let (a, b) = (simulate(&s3, &S3TestRule, &cfg).unwrap(), simulate(&s3, &S3TestRule, &under).unwrap());
        assert_eq!(a.series_sha256, b.series_sha256, "S3 {}", cost.id);
        assert!(compare_prefix(&a, &b, a.n_bars() - 1).is_empty());
    }
    assert_eq!(ExecutionModel::certification().layer(), Layer::Certified);
}

// ------------------------------------------------------------------------------------------------- delay of one bar

#[test]
fn delay_one_shifts_the_applied_weight_by_exactly_one_bar_hand_computed_five_bar_fixture() {
    let panel = five_bar_panel();
    let p = [100.0, 110.0, 99.0, 108.9, 100.0];
    let d0 = simulate(&panel, &five_bar_rule(), &SimConfig::default()).unwrap();
    let live = ExecutionModel::new(1, 0.0).apply_to(&SimConfig::default()).unwrap();
    assert_eq!(live.execution_delay_bars, 1);
    let d1 = simulate(&panel, &five_bar_rule(), &live).unwrap();

    // Standing target: delay 1 is the delay-0 column shifted by exactly one bar (nothing stands at bar 0).
    assert_eq!(d1.row(&d1.target_weights, 0), &[0.0]);
    for t in 1..5 {
        assert_eq!(d1.row(&d1.target_weights, t), d0.row(&d0.target_weights, t - 1), "bar {t}");
        assert_eq!(d1.row(&d1.target_weights, t), &[w_at(t - 1)]);
    }
    // Held weight after the fill at t equals the target decided at t-1 (zero cost: exactly on target).
    for t in 1..5 {
        assert!(close(d1.row(&d1.held_weights, t)[0], w_at(t - 1), 1e-12), "bar {t}");
    }
    // Returns by hand: the position earning bar t was decided at t-2 (filled at t-1).
    //   bar 1: nothing held during (0,1]           -> 0
    //   bar 2: w(0)=0.5 x (99/110 - 1)             -> -0.05
    //   bar 3: w(1)=1.0 x (108.9/99 - 1)           -> +0.10
    //   bar 4: w(2)=0.0                            -> 0
    let want_ret = [0.0, 0.0, -0.05, 0.1, 0.0];
    let want_eq = [1.0, 1.0, 0.95, 1.045, 1.045];
    for t in 0..5 {
        assert!(close(d1.ret[t], want_ret[t], 1e-12), "ret[{t}] = {} want {}", d1.ret[t], want_ret[t]);
        assert!(close(d1.equity[t], want_eq[t], 1e-12), "equity[{t}] = {} want {}", d1.equity[t], want_eq[t]);
        if t >= 2 {
            assert!(close(d1.ret[t], w_at(t - 2) * (p[t] / p[t - 1] - 1.0), 1e-12));
        }
    }
    // And the undelayed run is the same rule one bar earlier: bar 1 earns w(0) x (110/100 - 1) = +0.05.
    assert!(close(d0.ret[1], 0.05, 1e-12) && close(d0.ret[2], -0.1, 1e-12));
    assert_eq!((d0.window.unwrap().first_bar, d1.window.unwrap().first_bar), (1, 2));
    // Decisions are taken on the same bars in both runs: only the execution moved.
    assert_eq!(d0.decision, d1.decision);
}

// ------------------------------------------------------------------------------------------------------ slippage

#[test]
fn slippage_reduces_net_by_exactly_turnover_times_bps_every_bar() {
    let panel = synth_panel(&CRY, 700, 5, "2017-01-01");
    for (base, extra, want_rate_bps) in
        [(CostModel::ZERO, 25.0, 25.0), (CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE, 10.0, 20.0)]
    {
        let cfg = SimConfig { cost: base, ..SimConfig::default() };
        let live_cfg = ExecutionModel::new(0, extra).apply_to(&cfg).unwrap();
        assert_eq!(live_cfg.cost.rate_bps(), want_rate_bps);
        let cert = simulate(&panel, &S3TestRule, &cfg).unwrap();
        let live = simulate(&panel, &S3TestRule, &live_cfg).unwrap();
        let rate = want_rate_bps / 10_000.0;
        let mut total = 0.0;
        for t in 0..live.n_bars() {
            // Bit-exact: the ledger charges `traded x rate` and nothing else.
            assert_eq!(live.cost[t].to_bits(), (live.traded_notional[t] * rate).to_bits(), "bar {t}");
            total += live.cost[t];
        }
        assert_eq!(live.total_cost().to_bits(), total.to_bits());
        // On the first fill both runs have the same pre-cost equity, so the net difference IS the slippage.
        let f = cert.window.unwrap().first_bar - 1;
        assert!(cert.traded_notional[f] > 0.0);
        let want = cert.equity[f] + cert.cost[f] - live.traded_notional[f] * rate;
        assert!(close(live.equity[f], want, 1e-15), "{} vs {}", live.equity[f], want);
        assert!(live.equity[live.n_bars() - 1] < cert.equity[cert.n_bars() - 1]);
        assert_eq!(live.cost_model_id, live_cfg.cost.id);
        assert!(live.cost_model_id.ends_with("+live_realistic"));
    }
}

// ------------------------------------------------------------------------------------------ pre-registered defaults

#[test]
fn crypto_default_is_the_certification_path_except_for_the_slippage() {
    let panel = synth_panel(&CRY, 900, 7, "2016-01-01");
    let cfg = cert_cfg();
    let model = live_realistic_default_for(SleeveClass::CryptoTrend);
    assert_eq!(model.delay_bars, 0);
    let live_cfg = model.apply_to(&cfg).unwrap();
    assert_eq!(live_cfg.execution_delay_bars, 0);
    let cert = simulate(&panel, &S3TestRule, &cfg).unwrap();
    let live = simulate(&panel, &S3TestRule, &live_cfg).unwrap();
    // Same decisions, same standing targets, same dates: the decision stream is untouched by a 0-bar delay.
    assert_eq!(cert.decision, live.decision);
    assert_eq!(cert.target_weights, live.target_weights);
    assert_eq!(cert.dates, live.dates);
    assert_eq!(cert.window, live.window);
    // Only the cost differs: rate 10 + 10 bps instead of 10.
    let rate = (10.0 + CRYPTO_LIVE_SLIPPAGE_BPS) / 10_000.0;
    for t in 0..live.n_bars() {
        assert_eq!(live.cost[t].to_bits(), (live.traded_notional[t] * rate).to_bits());
    }
    assert_ne!(cert.series_sha256, live.series_sha256);
    assert_eq!(live.cost_model_id, "certification_flat_10bps_per_side+live_realistic");
}

#[test]
fn etf_default_delays_every_fill_by_one_own_bar() {
    let panel = synth_panel(&ETF, 1500, 3, "2015-01-02");
    let cfg = cert_cfg();
    let model = live_realistic_default_for(SleeveClass::EtfTrend);
    assert_eq!((model.delay_bars, model.slippage_bps), (1, ETF_LIVE_SLIPPAGE_BPS));
    let cert = simulate(&panel, &S1TestRule, &cfg).unwrap();
    let live = simulate(&panel, &S1TestRule, &model.apply_to(&cfg).unwrap()).unwrap();
    assert_eq!(cert.decision, live.decision, "decisions are taken on the same month-ends");
    for t in 1..live.n_bars() {
        assert_eq!(live.row(&live.target_weights, t), cert.row(&cert.target_weights, t - 1), "bar {t}");
    }
    assert_eq!(live.window.unwrap().first_bar, cert.window.unwrap().first_bar + 1);
    assert_eq!(model.layer().label(), "live_realistic");
    assert_eq!(LIVE_REALISTIC_CONFIG_VERSION, "live_realistic_v1_2026-10-02");
}
