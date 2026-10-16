//! Always-on (synthetic fixtures): the live-realistic layer (W7.4, R10) beside the certified replication.

mod common;

use common::*;
use weightsim::{CostModel, ExecutionModel, Layer, SleeveClass};
use weightsim_rules::ladder::fixtures::Fixtures;
use weightsim_rules::ladder::verify::{replicate, ReplicationConfig};
use weightsim_rules::ladder::{
    live_realistic_model_for, replicate_live_realistic, replicate_live_realistic_with, sleeve_class_of,
    LiveRealisticError, CERTIFIED_LAYER, LIVE_REALISTIC_LAYER,
};

fn fixtures() -> Fixtures {
    let (files, ms, cs) = synthetic_ready();
    load(&files, &ms, &cs).expect("synthetic fixtures load")
}

#[test]
fn library_rules_map_to_their_pre_registered_sleeve_class_and_model() {
    assert_eq!(sleeve_class_of("etf_trend_faber"), Some(SleeveClass::EtfTrend));
    assert_eq!(sleeve_class_of("crypto_trend_100d"), Some(SleeveClass::CryptoTrend));
    assert_eq!(sleeve_class_of("nope"), None);
    let etf = live_realistic_model_for("etf_trend_faber").unwrap();
    let cry = live_realistic_model_for("crypto_trend_100d").unwrap();
    assert_eq!((etf.delay_bars, etf.slippage_bps), (1, weightsim::ETF_LIVE_SLIPPAGE_BPS));
    assert_eq!((cry.delay_bars, cry.slippage_bps), (0, weightsim::CRYPTO_LIVE_SLIPPAGE_BPS));
    assert_eq!((CERTIFIED_LAYER, LIVE_REALISTIC_LAYER), ("certified", "live_realistic"));
}

#[test]
fn the_live_layer_is_labelled_separately_and_never_shares_the_certified_cost_id() {
    let fx = fixtures();
    for rule in ["etf_trend_faber", "crypto_trend_100d"] {
        let cert = replicate(&fx, rule).unwrap();
        let live = replicate_live_realistic(&fx, rule).unwrap();
        assert_eq!(cert.layer(), Layer::Certified);
        assert_eq!(live.layer, Layer::LiveRealistic);
        assert_eq!(live.layer_label(), "live_realistic");
        assert_eq!(live.config_version, weightsim::LIVE_REALISTIC_CONFIG_VERSION);
        assert_eq!(live.base_cost_model_id, cert.cost_model_id);
        assert_eq!(live.cost_model_id, "certification_flat_10bps_per_side+live_realistic");
        assert_eq!(CostModel::by_id(&live.cost_model_id), None, "a live id never resolves to a preset");
        assert_ne!(live.net_sha256, cert.net_sha256);
        assert_eq!(live.net.digest().unwrap(), live.net_sha256);
        assert_eq!(live.rule_id, cert.rule_id);
        assert_eq!(live.rule_impl_version, cert.rule_impl_version);
        assert_eq!(
            (live.manifest_sha256.as_str(), live.candles_sha256.as_str()),
            (fx.manifest_sha256.as_str(), fx.candles_sha256.as_str())
        );
        assert_eq!(live.rows.dates.len(), live.net_summary.n);
        assert!(live.net_summary.sharpe.is_finite());
        println!(
            "{rule} {} vs certified: sharpe {:.4} vs {:.4}, cagr {:.4} vs {:.4}",
            live.execution_model,
            live.net_summary.sharpe,
            cert.net_summary.sharpe,
            live.net_summary.cagr,
            cert.net_summary.cagr
        );
    }
}

#[test]
fn crypto_live_layer_is_the_certified_decision_stream_with_slippage_only() {
    let fx = fixtures();
    let cert = replicate(&fx, "crypto_trend_100d").unwrap();
    let live = replicate_live_realistic(&fx, "crypto_trend_100d").unwrap();
    assert_eq!(live.execution_model.delay_bars, 0);
    assert_eq!(cert.net.decision, live.net.decision);
    assert_eq!(cert.net.target_weights, live.net.target_weights);
    assert_eq!(cert.net.dates, live.net.dates);
    let rate = (10.0 + weightsim::CRYPTO_LIVE_SLIPPAGE_BPS) / 10_000.0;
    for t in 0..live.net.n_bars() {
        assert_eq!(live.net.cost[t].to_bits(), (live.net.traded_notional[t] * rate).to_bits(), "bar {t}");
    }
    let last = live.net.n_bars() - 1;
    assert!(live.net.equity[last] < cert.net.equity[last], "slippage costs money");
}

#[test]
fn etf_live_layer_executes_each_month_end_decision_one_bar_later() {
    let fx = fixtures();
    let cert = replicate(&fx, "etf_trend_faber").unwrap();
    let live = replicate_live_realistic(&fx, "etf_trend_faber").unwrap();
    assert_eq!(live.execution_model.delay_bars, 1);
    assert_eq!(cert.net.decision, live.net.decision, "decided on the same bars");
    let k = live.net.n_assets();
    for t in 1..live.net.n_bars() {
        assert_eq!(
            &live.net.target_weights[t * k..(t + 1) * k],
            &cert.net.target_weights[(t - 1) * k..t * k],
            "bar {t}"
        );
    }
}

#[test]
fn the_certification_model_is_refused_here_and_a_custom_model_is_honoured() {
    let fx = fixtures();
    let base = ReplicationConfig::default();
    assert_eq!(
        replicate_live_realistic_with(&fx, "etf_trend_faber", &base, ExecutionModel::certification()).err(),
        Some(LiveRealisticError::NotLiveRealistic)
    );
    assert!(matches!(replicate_live_realistic(&fx, "nope"), Err(LiveRealisticError::Replicate(_))));
    let custom = replicate_live_realistic_with(&fx, "crypto_trend_100d", &base, ExecutionModel::new(2, 0.0)).unwrap();
    assert_eq!(custom.execution_model, ExecutionModel::new(2, 0.0));
    assert_eq!(custom.cost_model_id, "certification_flat_10bps_per_side+live_realistic");
    let zero_base = ReplicationConfig { cost_model_id: "zero".into() };
    let g = replicate_live_realistic_with(&fx, "crypto_trend_100d", &zero_base, ExecutionModel::new(0, 10.0)).unwrap();
    assert_eq!(g.cost_model_id, "zero+live_realistic");
}
