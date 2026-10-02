//! Env-gated: W7.4 (R10) on the REAL pinned data. The certification execution model must reproduce the four pinned
//! series digests bit for bit (the certified layer is untouched by the live-realistic layer), and the live-realistic
//! runs are produced beside them with their own label and digests. Only digests and headline numbers appear here.
//!
//! Set `WEIGHTSIM_RULES_LADDER_DIR` to the Engine's `program/tests/fixtures/replication_ladder` directory; otherwise
//! every test here prints SKIPPED and passes.

use std::path::PathBuf;

use weightsim::{simulate, CostModel, ExecutionModel, Layer};
use weightsim_rules::ladder::fixtures::Fixtures;
use weightsim_rules::ladder::replicate_live_realistic;
use weightsim_rules::ladder::runner::{entry_date, sleeve_config};
use weightsim_rules::ladder::verify::replicate;
use weightsim_rules::ladder::verify::LIBRARY_RULE_IDS;
use weightsim_rules::{CryptoTrendRule, EtfTrendRule, FlatUntil};

const ENV: &str = "WEIGHTSIM_RULES_LADDER_DIR";

fn dir(test: &str) -> Option<PathBuf> {
    match std::env::var(ENV) {
        Ok(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => {
            println!("SKIPPED {test}: set {ENV} to the replication_ladder fixture directory (vendor data is not in this repository)");
            None
        }
    }
}

/// The four series digests pinned in `ladder_real.rs` (S1 gross, S1 net, S3 gross, S3 net), measured before W7.4.
const PINNED: [&str; 4] = [
    "0a857c83d2558b8e9b4d5ca2175ec9f8bd06b640867575185635c1322bbe0987",
    "2bf54e4607077ca9f0037e7898a0e95de1394781045f597c236742faa73996d7",
    "35e20b47ddd7b28cda5f8d5bf1b9cf6abca689e162e69ce5696e42ff67a35156",
    "20e4b939a33d5c75b5a6b4b6317f5a218ab4416dec41d3dce7ef8a900ff81e2c",
];

#[test]
fn certification_execution_model_reproduces_the_pinned_digests_bit_for_bit() {
    let Some(dir) = dir("certification model digests") else { return };
    let fx = Fixtures::from_dir(&dir).unwrap_or_else(|e| panic!("fixtures: {e}"));
    // `replicate` (the certified path, unchanged code) still pins.
    let s1 = replicate(&fx, "etf_trend_faber").unwrap();
    let s3 = replicate(&fx, "crypto_trend_100d").unwrap();
    assert_eq!([&s1.gross_sha256, &s1.net_sha256, &s3.gross_sha256, &s3.net_sha256].map(String::as_str), PINNED);
    assert_eq!((s1.layer(), s3.layer()), (Layer::Certified, Layer::Certified));
    // And the same runs driven through `ExecutionModel::certification().apply_to` are byte-identical.
    let cert = ExecutionModel::certification();
    let e1 = entry_date(&fx.etf_panel, &fx.s1).unwrap();
    let e3 = entry_date(&fx.crypto_panel, &fx.s3).unwrap();
    let mut got = Vec::new();
    for cost in [CostModel::ZERO, CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE] {
        let cfg = cert.apply_to(&sleeve_config(&fx.s1, cost)).unwrap();
        got.push(simulate(&fx.etf_panel, &FlatUntil::new(EtfTrendRule, e1), &cfg).unwrap().series_sha256);
    }
    for cost in [CostModel::ZERO, CostModel::CERTIFICATION_FLAT_10BPS_PER_SIDE] {
        let cfg = cert.apply_to(&sleeve_config(&fx.s3, cost)).unwrap();
        got.push(simulate(&fx.crypto_panel, &FlatUntil::new(CryptoTrendRule, e3), &cfg).unwrap().series_sha256);
    }
    for (name, (want, g)) in ["S1 gross", "S1 net", "S3 gross", "S3 net"].iter().zip(PINNED.iter().zip(&got)) {
        println!("CERTIFICATION-MODEL DIGEST {name}: {g}");
        assert_eq!(want, g, "{name} changed under ExecutionModel::certification()");
    }
}

#[test]
fn live_realistic_runs_are_produced_beside_the_certified_ones_with_their_own_digests() {
    let Some(dir) = dir("live-realistic layer") else { return };
    let fx = Fixtures::from_dir(&dir).unwrap_or_else(|e| panic!("fixtures: {e}"));
    for rule in LIBRARY_RULE_IDS {
        let cert = replicate(&fx, rule).unwrap();
        let live = replicate_live_realistic(&fx, rule).unwrap();
        assert_eq!(live.layer, Layer::LiveRealistic);
        assert_ne!(live.net_sha256, cert.net_sha256);
        assert!(!PINNED.contains(&live.net_sha256.as_str()));
        assert_eq!(live.net.digest().unwrap(), live.net_sha256);
        assert_eq!(live.cost_model_id, "certification_flat_10bps_per_side+live_realistic");
        println!(
            "LIVE-REALISTIC {rule} [{}] {} {}: net sha256 {} | sharpe {:.4} (certified {:.4}) cagr {:.4} (certified {:.4}) max_dd {:.4} (certified {:.4}) n {}",
            live.layer_label(),
            live.config_version,
            live.execution_model,
            live.net_sha256,
            live.net_summary.sharpe,
            cert.net_summary.sharpe,
            live.net_summary.cagr,
            cert.net_summary.cagr,
            live.net_summary.max_drawdown,
            cert.net_summary.max_drawdown,
            live.net_summary.n
        );
    }
}
