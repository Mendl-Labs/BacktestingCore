//! Monte Carlo of the size and power of the paired marginal-contribution test ([`crate::marginal`]) and the "PF5
//! power table" (design Section 6 "What to measure first (4)", doubt U7).
//!
//! # Data-generating process
//!
//! Two unit-volatility sleeves, the existing book `P` and the candidate `X`, with contemporaneous correlation `rho`
//! and per-period volatility `SIGMA`. `P` has annual Sharpe `base_sharpe`. The candidate is added with capital share
//! `share` (`w`), so the enlarged book is `B = (1 - w) P + w X` (a static-weight, per-period rebalanced blend). The
//! TRUE incremental Sharpe is `effect = SR(B) - SR(P)` (annualised, at equal volatility because Sharpe is scale free);
//! the candidate's mean is solved from it in closed form:
//!
//! ```text
//! var_B = (1-w)^2 + w^2 + 2 w (1-w) rho,   SR(B) = SR(P) + effect,   SR(X) = (SR(B) sqrt(var_B) - (1-w) SR(P)) / w
//! ```
//!
//! `effect = 0` gives the SIZE of the test (false-positive rate when the candidate adds nothing); `effect > 0` gives
//! power. The test under evaluation is exactly the shipped one: [`marginal_contribution`] with the default
//! automatic block length, one-sided level `alpha`.
//!
//! # Reproducibility
//!
//! Every Monte Carlo task `(horizon, rho, rep)` draws its data from `Rng::from_stream(seed, task)`, all effects of a
//! task reuse the same noise and the same bootstrap seed (common random numbers, so the power curve is smooth), tasks
//! may run on any number of threads and results are merged in task order, and every function used is deterministic
//! (`detmath`, integer RNG). The result is therefore bit-identical for any thread count and on any platform.

use crate::detmath;
use crate::error::{invalid, Result};
use crate::marginal::{marginal_contribution, mde_analytic_iid, BlockLength, MarginalConfig};
use crate::rng::Rng;
use crate::sha256::sha256_hex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Per-period volatility of the simulated sleeves (irrelevant to every statistic; fixed for reproducibility).
pub const SIGMA: f64 = 0.01;
/// Extra draws discarded at the start of an autoregressive series.
const BURN_IN: usize = 50;

/// Innovation law of the simulated returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Innovations {
    /// `None` = Gaussian; `Some(df)` = Student-t with `df >= 3` degrees of freedom, rescaled to unit variance.
    pub t_df: Option<u32>,
    /// AR(1) coefficient of each series (`0` = serially independent); the series is rescaled to unit variance.
    pub ar1: f64,
}

impl Innovations {
    pub const GAUSSIAN: Innovations = Innovations { t_df: None, ar1: 0.0 };
}

/// One Monte Carlo experiment (a grid of horizons x correlations x true effects).
#[derive(Clone, Debug, PartialEq)]
pub struct PowerSpec {
    pub horizons_years: Vec<f64>,
    /// Correlations `rho` between `P` and `X`.
    pub corrs: Vec<f64>,
    /// True annualised incremental Sharpe values; include `0.0` for the size column.
    pub effects: Vec<f64>,
    pub reps: usize,
    pub n_boot: usize,
    pub periods_per_year: f64,
    pub base_sharpe: f64,
    /// Capital share `w` of the candidate in the enlarged book.
    pub share: f64,
    pub alpha: f64,
    pub power: f64,
    pub seed: u64,
    pub innovations: Innovations,
}

impl PowerSpec {
    /// Number of observations of a horizon.
    pub fn n_obs(&self, years: f64) -> usize {
        (years * self.periods_per_year).round() as usize
    }

    fn validate(&self) -> Result<()> {
        if self.horizons_years.is_empty() || self.corrs.is_empty() || self.effects.is_empty() {
            return Err(invalid("spec", "horizons, correlations and effects must all be non-empty"));
        }
        if self.reps == 0 {
            return Err(invalid("reps", "must be at least 1"));
        }
        if !(self.share > 0.0 && self.share < 1.0) {
            return Err(invalid("share", "must be in (0, 1)"));
        }
        for r in &self.corrs {
            if !(r.is_finite() && *r > -1.0 && *r < 1.0) {
                return Err(invalid("corrs", "correlations must be in (-1, 1)"));
            }
        }
        if let Some(df) = self.innovations.t_df {
            if df < 3 {
                return Err(invalid("t_df", "Student-t needs at least 3 degrees of freedom"));
            }
        }
        if !(self.innovations.ar1.is_finite() && self.innovations.ar1.abs() < 1.0) {
            return Err(invalid("ar1", "must be in (-1, 1)"));
        }
        Ok(())
    }

    /// Annual Sharpe of the candidate sleeve that produces the true incremental Sharpe `effect` at correlation `rho`.
    pub fn candidate_sharpe(&self, rho: f64, effect: f64) -> f64 {
        let w = self.share;
        let var_b = (1.0 - w) * (1.0 - w) + w * w + 2.0 * w * (1.0 - w) * rho;
        ((self.base_sharpe + effect) * var_b.sqrt() - (1.0 - w) * self.base_sharpe) / w
    }

    /// Correlation between the base book `P` and the enlarged book `B`.
    pub fn corr_books(&self, rho: f64) -> f64 {
        let w = self.share;
        let var_b = (1.0 - w) * (1.0 - w) + w * w + 2.0 * w * (1.0 - w) * rho;
        ((1.0 - w) + w * rho) / var_b.sqrt()
    }

    /// Analytic (iid normal, Memmel) MDE of the Sharpe difference at this horizon and correlation.
    pub fn analytic_mde(&self, years: f64, rho: f64) -> Result<f64> {
        mde_analytic_iid(
            self.n_obs(years),
            self.periods_per_year,
            self.base_sharpe,
            self.corr_books(rho),
            self.alpha,
            self.power,
        )
    }
}

/// Aggregated result of one `(horizon, rho, effect)` cell.
#[derive(Clone, Debug, PartialEq)]
pub struct CellResult {
    pub years: f64,
    pub n: usize,
    pub rho: f64,
    pub effect: f64,
    pub reps: usize,
    pub rejections: usize,
    /// Mean bootstrap standard error of the Sharpe difference over the repetitions.
    pub mean_boot_se: f64,
    /// Mean block length chosen by the automatic rule.
    pub mean_block: f64,
    /// Mean estimated Sharpe difference.
    pub mean_delta: f64,
}

impl CellResult {
    /// Rejection rate.
    pub fn rate(&self) -> f64 {
        self.rejections as f64 / self.reps as f64
    }
}

/// All cells, ordered by horizon, then correlation, then effect.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerResults {
    pub spec: PowerSpec,
    pub cells: Vec<CellResult>,
}

impl PowerResults {
    /// The cell at `(horizon index, correlation index, effect index)`.
    pub fn cell(&self, hi: usize, ci: usize, ei: usize) -> &CellResult {
        let ne = self.spec.effects.len();
        let nc = self.spec.corrs.len();
        &self.cells[(hi * nc + ci) * ne + ei]
    }

    /// Smallest effect at which the Monte Carlo power reaches `spec.power`, by linear interpolation of the power curve
    /// (`None` if it never does within the grid).
    pub fn mc_mde(&self, hi: usize, ci: usize) -> Option<f64> {
        let ne = self.spec.effects.len();
        let mut prev: Option<(f64, f64)> = None;
        for ei in 0..ne {
            let c = self.cell(hi, ci, ei);
            let (e, p) = (c.effect, c.rate());
            if p >= self.spec.power {
                return Some(match prev {
                    Some((pe, pp)) if p > pp => pe + (self.spec.power - pp) / (p - pp) * (e - pe),
                    _ => e,
                });
            }
            prev = Some((e, p));
        }
        None
    }

    /// SHA-256 of the exact counts and float bits of every cell: a fingerprint of the whole experiment.
    pub fn digest(&self) -> String {
        let mut s = String::new();
        for c in &self.cells {
            s.push_str(&format!(
                "{} {} {} {} {} {:016x} {:016x} {:016x}\n",
                c.years,
                c.rho,
                c.effect,
                c.reps,
                c.rejections,
                c.mean_boot_se.to_bits(),
                c.mean_block.to_bits(),
                c.mean_delta.to_bits()
            ));
        }
        sha256_hex(s.as_bytes())
    }
}

fn gen_series(rng: &mut Rng, n: usize, inn: &Innovations) -> Vec<f64> {
    let scale = (1.0 - inn.ar1 * inn.ar1).sqrt();
    let mut out = Vec::with_capacity(n);
    let mut x = 0.0;
    for t in 0..n + BURN_IN {
        let e = match inn.t_df {
            None => rng.normal(),
            Some(df) => {
                let z = rng.normal();
                let mut chi2 = 0.0;
                for _ in 0..df {
                    let g = rng.normal();
                    chi2 += g * g;
                }
                z / (chi2 / df as f64).sqrt() * ((df as f64 - 2.0) / df as f64).sqrt()
            }
        };
        x = inn.ar1 * x + e;
        if t >= BURN_IN {
            out.push(x * scale);
        }
    }
    out
}

struct EffectOut {
    reject: bool,
    se: f64,
    block: f64,
    delta: f64,
}

fn run_task(spec: &PowerSpec, hi: usize, ci: usize, rep: usize) -> Result<Vec<EffectOut>> {
    let n = spec.n_obs(spec.horizons_years[hi]);
    let rho = spec.corrs[ci];
    let stream = ((hi as u64) << 44) | ((ci as u64) << 36) | rep as u64;
    let mut rng = Rng::from_stream(spec.seed, stream);
    let z1 = gen_series(&mut rng, n, &spec.innovations);
    let z2 = gen_series(&mut rng, n, &spec.innovations);
    let boot_seed = rng.next_u64();
    let root_ppy = spec.periods_per_year.sqrt();
    let cross = (1.0 - rho * rho).sqrt();
    let sr_p = spec.base_sharpe / root_ppy;
    let base: Vec<f64> = z1.iter().map(|z| SIGMA * (sr_p + z)).collect();
    let w = spec.share;
    let mut cfg = MarginalConfig::new(spec.periods_per_year);
    cfg.n_boot = spec.n_boot;
    cfg.block = BlockLength::Auto;
    cfg.seed = boot_seed;
    cfg.alpha = spec.alpha;
    cfg.power = spec.power;
    let mut out = Vec::with_capacity(spec.effects.len());
    for &effect in &spec.effects {
        let sr_x = spec.candidate_sharpe(rho, effect) / root_ppy;
        let combined: Vec<f64> = (0..n)
            .map(|t| {
                let x = SIGMA * (sr_x + rho * z1[t] + cross * z2[t]);
                (1.0 - w) * base[t] + w * x
            })
            .collect();
        let r = marginal_contribution(&base, &combined, &cfg)?;
        out.push(EffectOut { reject: r.significant(), se: r.boot_se, block: r.block_length, delta: r.delta_sharpe });
    }
    Ok(out)
}

/// Run the experiment on `threads` worker threads (`>= 1`). The result does not depend on `threads`.
pub fn run_power(spec: &PowerSpec, threads: usize) -> Result<PowerResults> {
    spec.validate()?;
    let threads = threads.max(1);
    let nh = spec.horizons_years.len();
    let nc = spec.corrs.len();
    let total = nh * nc * spec.reps;
    let next = AtomicUsize::new(0);
    let collected: Mutex<Vec<(usize, Result<Vec<EffectOut>>)>> = Mutex::new(Vec::with_capacity(total));
    std::thread::scope(|scope| {
        for _ in 0..threads.min(total) {
            scope.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let task = next.fetch_add(1, Ordering::Relaxed);
                    if task >= total {
                        break;
                    }
                    let rep = task % spec.reps;
                    let rest = task / spec.reps;
                    let (hi, ci) = (rest / nc, rest % nc);
                    local.push((task, run_task(spec, hi, ci, rep)));
                }
                if let Ok(mut g) = collected.lock() {
                    g.extend(local);
                }
            });
        }
    });
    let mut all = collected.into_inner().unwrap_or_default();
    all.sort_by_key(|(t, _)| *t);
    let ne = spec.effects.len();
    let mut cells = Vec::with_capacity(nh * nc * ne);
    for hi in 0..nh {
        for ci in 0..nc {
            for ei in 0..ne {
                cells.push(CellResult {
                    years: spec.horizons_years[hi],
                    n: spec.n_obs(spec.horizons_years[hi]),
                    rho: spec.corrs[ci],
                    effect: spec.effects[ei],
                    reps: spec.reps,
                    rejections: 0,
                    mean_boot_se: 0.0,
                    mean_block: 0.0,
                    mean_delta: 0.0,
                });
            }
        }
    }
    for (task, res) in all {
        let outs = res?;
        let rest = task / spec.reps;
        let (hi, ci) = (rest / nc, rest % nc);
        for (ei, o) in outs.iter().enumerate() {
            let c = &mut cells[(hi * nc + ci) * ne + ei];
            c.rejections += usize::from(o.reject);
            c.mean_boot_se += o.se;
            c.mean_block += o.block;
            c.mean_delta += o.delta;
        }
    }
    for c in &mut cells {
        let r = c.reps as f64;
        c.mean_boot_se /= r;
        c.mean_block /= r;
        c.mean_delta /= r;
    }
    Ok(PowerResults { spec: spec.clone(), cells })
}

/// The experiment behind the committed `POWER_TABLE.md`: 2, 3.5, 5 and 10 years of daily data (252 bars a year),
/// correlations 0, 0.3, 0.6, base Sharpe 0.5, candidate share 0.5, one-sided `alpha = 0.05`.
pub fn committed_main_spec() -> PowerSpec {
    PowerSpec {
        horizons_years: vec![2.0, 3.5, 5.0, 10.0],
        corrs: vec![0.0, 0.3, 0.6],
        effects: vec![0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
        reps: 400,
        n_boot: 199,
        periods_per_year: 252.0,
        base_sharpe: 0.5,
        share: 0.5,
        alpha: 0.05,
        power: 0.8,
        seed: 0x5046_3520_504f_5745,
        innovations: Innovations::GAUSSIAN,
    }
}

/// A labelled single-horizon, single-correlation variation of the main experiment (crypto calendar, fat tails,
/// autocorrelation, a bull-market base Sharpe, a smaller candidate share) for the robustness table.
#[derive(Clone, Debug, PartialEq)]
pub struct Scenario {
    pub name: &'static str,
    pub spec: PowerSpec,
}

/// The robustness scenarios of `POWER_TABLE.md`: all at 3.5 years and `rho = 0.3`.
pub fn committed_scenarios() -> Vec<Scenario> {
    let base = |mutate: &dyn Fn(&mut PowerSpec)| {
        let mut s = committed_main_spec();
        s.horizons_years = vec![3.5];
        s.corrs = vec![0.3];
        s.reps = 300;
        mutate(&mut s);
        s
    };
    vec![
        Scenario { name: "baseline (Gaussian, 252 bars/yr, base SR 0.5, share 50%)", spec: base(&|_| {}) },
        Scenario {
            name: "crypto calendar (365 bars/yr, 3.5y = 1278 bars)",
            spec: base(&|s| s.periods_per_year = 365.0),
        },
        Scenario {
            name: "fat tails (Student-t, 4 df)",
            spec: base(&|s| s.innovations = Innovations { t_df: Some(4), ar1: 0.0 }),
        },
        Scenario {
            name: "serial correlation (AR(1) 0.1)",
            spec: base(&|s| s.innovations = Innovations { t_df: None, ar1: 0.1 }),
        },
        Scenario { name: "bull-market base book (base SR 1.5)", spec: base(&|s| s.base_sharpe = 1.5) },
        Scenario { name: "small candidate (share 20%)", spec: base(&|s| s.share = 0.2) },
    ]
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.2}"),
        None => "> grid".to_string(),
    }
}

fn effect_header(effects: &[f64]) -> String {
    let mut h = String::new();
    for e in effects {
        if *e == 0.0 {
            h.push_str(" 0 (size) |");
        } else {
            h.push_str(&format!(" {e} |"));
        }
    }
    h
}

/// Render the main experiment as markdown tables (one per correlation).
pub fn render_main(res: &PowerResults) -> Result<String> {
    let s = &res.spec;
    let mut out = String::new();
    out.push_str(&format!(
        "Experiment: {} repetitions per cell, {} bootstrap replicates, {} bars/year, base Sharpe {}, candidate share {}, one-sided alpha {}, target power {}, seed 0x{:016x}.\n\n",
        s.reps, s.n_boot, s.periods_per_year, s.base_sharpe, s.share, s.alpha, s.power, s.seed
    ));
    out.push_str("Cells are the rejection rate in percent of the one-sided paired marginal test at each TRUE incremental annual Sharpe (columns); the `0 (size)` column is the false-positive rate. ");
    out.push_str(&format!(
        "Monte Carlo standard error of a rate p is sqrt(p(1-p)/{}), at most {:.1} percentage points.\n",
        s.reps,
        50.0 / (s.reps as f64).sqrt()
    ));
    for (ci, rho) in s.corrs.iter().enumerate() {
        out.push_str(&format!(
            "\n### Correlation rho = {rho} between the existing book's sleeve set and the candidate\n\n"
        ));
        out.push_str(&format!(
            "| years | bars |{} MDE analytic | MDE Monte Carlo | mean bootstrap SE |\n",
            effect_header(&s.effects)
        ));
        out.push_str(&format!("|---|---|{}---|---|---|\n", "---|".repeat(s.effects.len())));
        for (hi, years) in s.horizons_years.iter().enumerate() {
            let n = s.n_obs(*years);
            let mut row = format!("| {years} | {n} |");
            for ei in 0..s.effects.len() {
                row.push_str(&format!(" {:.1} |", 100.0 * res.cell(hi, ci, ei).rate()));
            }
            let mean_se = res.cell(hi, ci, 0).mean_boot_se;
            row.push_str(&format!(
                " {} | {} | {:.3} |",
                fmt_opt(Some(s.analytic_mde(*years, *rho)?)),
                fmt_opt(res.mc_mde(hi, ci)),
                mean_se
            ));
            out.push_str(&row);
            out.push('\n');
        }
    }
    Ok(out)
}

/// Render the robustness scenarios (one row per scenario).
pub fn render_scenarios(results: &[(&'static str, PowerResults)]) -> Result<String> {
    let mut out = String::new();
    if let Some((_, first)) = results.first() {
        let s = &first.spec;
        out.push_str(&format!(
            "All rows: {} years, rho = {}, {} repetitions per cell, {} bootstrap replicates, one-sided alpha {}.\n\n",
            s.horizons_years[0], s.corrs[0], s.reps, s.n_boot, s.alpha
        ));
        out.push_str(&format!("| scenario | bars |{} MDE analytic | MDE Monte Carlo |\n", effect_header(&s.effects)));
        out.push_str(&format!("|---|---|{}---|---|\n", "---|".repeat(s.effects.len())));
    }
    for (name, res) in results {
        let s = &res.spec;
        let mut row = format!("| {name} | {} |", s.n_obs(s.horizons_years[0]));
        for ei in 0..s.effects.len() {
            row.push_str(&format!(" {:.1} |", 100.0 * res.cell(0, 0, ei).rate()));
        }
        row.push_str(&format!(
            " {} | {} |",
            fmt_opt(Some(s.analytic_mde(s.horizons_years[0], s.corrs[0])?)),
            fmt_opt(res.mc_mde(0, 0))
        ));
        out.push_str(&row);
        out.push('\n');
    }
    Ok(out)
}

/// Run and render the complete generated section of `POWER_TABLE.md`.
pub fn render_committed(threads: usize) -> Result<String> {
    let main = run_power(&committed_main_spec(), threads)?;
    let mut scen = Vec::new();
    for sc in committed_scenarios() {
        scen.push((sc.name, run_power(&sc.spec, threads)?));
    }
    let mut out = String::new();
    out.push_str("## Size and power by horizon and correlation\n\n");
    out.push_str(&render_main(&main)?);
    out.push_str("\n## Robustness scenarios\n\n");
    out.push_str(&render_scenarios(&scen)?);
    // tie a numeric guard to the exact ln/erfc paths used by the analytic column
    out.push_str(&format!(
        "\nAnalytic z-multiplier (z_(1-alpha) + z_power) = {:.4}.\n",
        detmath::norm_isf(0.05) + detmath::norm_ppf(0.8)
    ));
    Ok(out)
}

/// Marker lines that delimit the generated section inside `POWER_TABLE.md`.
pub const BEGIN_MARKER: &str = "<!-- BEGIN GENERATED (cargo run --release --bin power_table) -->";
/// See [`BEGIN_MARKER`].
pub const END_MARKER: &str = "<!-- END GENERATED -->";

/// The generated section wrapped in its markers, with its SHA-256 line.
pub fn generated_block(threads: usize) -> Result<String> {
    let body = render_committed(threads)?;
    let digest = sha256_hex(body.as_bytes());
    Ok(format!("{BEGIN_MARKER}\n{body}\nsha256 of the generated section above: `{digest}`\n{END_MARKER}\n"))
}
