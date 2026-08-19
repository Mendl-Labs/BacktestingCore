//! GA post-optimization analysis: landscape heatmaps, sensitivity analysis,
//! overfitting gap detection, and convergence metrics.
//!
//! These reports help users understand whether optimized parameters are robust
//! or fragile (sharp peak = overfitting, flat plateau = robust).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::dynamic_chromosome::{
    DynamicChromosome, Gene, ParameterSchema, ParamType,
};
use crate::{AsyncFitnessFn, Chromosome, FitnessResult};

// ---------------------------------------------------------------------------
// GP math helpers (used by bo_landscape_heatmap)
// ---------------------------------------------------------------------------

/// Cholesky factorization: A = L * L^T. Returns L (lower triangular, row-major).
/// Input must be positive-definite; caller ensures this by adding σ²I + jitter.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
                s.max(0.0).sqrt()
            } else {
                let ljj = l[j * n + j];
                if ljj.abs() > 1e-12 { s / ljj } else { 0.0 }
            };
        }
    }
    l
}

/// Solve L * x = b by forward substitution (L lower triangular, row-major).
fn forward_solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * x[j];
        }
        let lii = l[i * n + i];
        x[i] = if lii.abs() > 1e-12 { s / lii } else { 0.0 };
    }
    x
}

/// Solve L^T * x = b by backward substitution (L lower triangular, row-major).
fn backward_solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j]; // L^T[i][j] = L[j][i]
        }
        let lii = l[i * n + i];
        x[i] = if lii.abs() > 1e-12 { s / lii } else { 0.0 };
    }
    x
}

/// General N-dimensional Gaussian Process surrogate with squared-exponential
/// (RBF) kernel. Inputs are expected to be in normalised [0, 1]^d space.
/// Generalized 2026-08-18 from a 2D-only version (built for `bo_landscape_heatmap`'s
/// 2-parameter visualization slices) so the same GP math can drive a real
/// N-dimensional Bayesian-optimization search -- the RBF kernel and Cholesky
/// solve don't care about dimensionality, only the call sites did.
struct GpSurrogate {
    xs: Vec<Vec<f64>>,
    alpha: Vec<f64>,  // (K + σ²I)^{-1} y
    chol_l: Vec<f64>, // lower Cholesky factor, n×n row-major
    n: usize,
    length_scale: f64, // RBF length scale in normalised space
    noise_var: f64,    // observation noise variance
}

impl GpSurrogate {
    fn new(length_scale: f64, noise_var: f64) -> Self {
        Self { xs: vec![], alpha: vec![], chol_l: vec![], n: 0, length_scale, noise_var }
    }

    /// k(a, b) = exp(-‖a − b‖² / (2 l²)). `a`/`b` must have equal length.
    #[inline]
    fn rbf(&self, a: &[f64], b: &[f64]) -> f64 {
        let sq: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum();
        (-sq / (2.0 * self.length_scale.powi(2))).exp()
    }

    /// Fit the GP to observed (xs, y) pairs.
    fn fit(&mut self, xs: Vec<Vec<f64>>, y: Vec<f64>) {
        let n = xs.len();
        let mut k = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                k[i * n + j] = self.rbf(&xs[i], &xs[j]);
            }
            // σ²I + small jitter to ensure positive-definiteness
            k[i * n + i] += self.noise_var + 1e-6;
        }
        let l = cholesky(&k, n);
        let v = forward_solve(&l, &y, n);
        let alpha = backward_solve(&l, &v, n);
        self.xs = xs;
        self.alpha = alpha;
        self.chol_l = l;
        self.n = n;
    }

    /// Posterior mean at x*.
    fn predict_mean(&self, x: &[f64]) -> f64 {
        if self.n == 0 { return 0.0; }
        self.xs.iter().zip(&self.alpha).map(|(xi, a)| self.rbf(x, xi) * a).sum()
    }

    /// Posterior variance at x*.
    fn predict_var(&self, x: &[f64]) -> f64 {
        if self.n == 0 { return 1.0; }
        let k_star: Vec<f64> = self.xs.iter().map(|xi| self.rbf(x, xi)).collect();
        let v = forward_solve(&self.chol_l, &k_star, self.n);
        // k(x*,x*) = 1 for RBF; subtract ||v||²
        (1.0 - v.iter().map(|vi| vi * vi).sum::<f64>()).max(0.0)
    }

}

/// Latin Hypercube Sample: n points in [0, 1]^dims. Each dimension gets its
/// own independently-shuffled permutation of n strata (standard LHS), so
/// every stratum along every axis is hit exactly once regardless of `dims`.
fn latin_hypercube_sample(n: usize, dims: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
    let per_dim_strata: Vec<Vec<usize>> = (0..dims)
        .map(|_| {
            let mut s: Vec<usize> = (0..n).collect();
            s.shuffle(rng);
            s
        })
        .collect();
    (0..n)
        .map(|i| {
            (0..dims)
                .map(|d| (per_dim_strata[d][i] as f64 + rng.gen::<f64>()) / n as f64)
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bayesian optimization: acquisition function + inner search
// (2026-08-18 -- built to drive a real optimizer, not just visualize one;
// see run_bayesian_optimization below for the outer evaluate-refit loop.)
// ---------------------------------------------------------------------------

/// Upper Confidence Bound acquisition: `mean + kappa * sqrt(variance)`.
/// Balances exploitation (go where the GP predicts high fitness) against
/// exploration (go where the GP is uncertain, since that's where a real
/// evaluation teaches it the most) -- the standard, simplest real acquisition
/// function. Higher `kappa` biases toward exploration.
fn acquisition_ucb(gp: &GpSurrogate, x: &[f64], kappa: f64) -> f64 {
    gp.predict_mean(x) + kappa * gp.predict_var(x).sqrt()
}

/// Propose the next point to evaluate by maximizing the UCB acquisition
/// function over `[0, 1]^dims`, via random-restart local search. Standard,
/// sufficient practice at this dimensionality (the strategies this drives
/// expose 3-6 tunable parameters) -- no need for a heavier global optimizer
/// just to maximize a cheap-to-evaluate (GP-predicted, not real-backtest)
/// surface.
///
/// `n_restarts` independent random starting points are each locally refined
/// via coordinate-wise hill climbing over `n_local_iters` iterations with a
/// shrinking step size (simple, no learning-rate tuning needed at this
/// scale); the best point found across every restart wins.
fn maximize_acquisition(
    gp: &GpSurrogate,
    dims: usize,
    kappa: f64,
    n_restarts: usize,
    n_local_iters: usize,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let mut best_point: Vec<f64> = (0..dims).map(|_| rng.gen::<f64>()).collect();
    let mut best_score = acquisition_ucb(gp, &best_point, kappa);

    for _ in 0..n_restarts {
        let mut point: Vec<f64> = (0..dims).map(|_| rng.gen::<f64>()).collect();
        let mut score = acquisition_ucb(gp, &point, kappa);

        let mut step = 0.25_f64;
        for _ in 0..n_local_iters {
            let dim = rng.gen_range(0..dims);
            let delta = if rng.gen::<bool>() { step } else { -step };
            let mut candidate = point.clone();
            candidate[dim] = (candidate[dim] + delta).clamp(0.0, 1.0);
            let candidate_score = acquisition_ucb(gp, &candidate, kappa);
            if candidate_score > score {
                point = candidate;
                score = candidate_score;
            }
            step *= 0.95; // shrink for finer refinement as iterations proceed
        }

        if score > best_score {
            best_score = score;
            best_point = point;
        }
    }

    best_point
}

/// Above this many total evaluations, refitting a GP from scratch every
/// iteration (Cholesky decomposition, O(n^3) in the number of accumulated
/// points) becomes computationally infeasible -- a naive refit at n=5000
/// (a realistic Deep-tier GA budget: population_size=100 * generations=50)
/// would be ~125 BILLION operations for the last refit alone. This isn't a
/// shortcut: real BO's entire value proposition is sample efficiency at
/// SMALL budgets, so comparing it against GA at a budget this large would
/// defeat the point of using it even if it were computationally free. Any
/// `total_evals` above this cap is silently clamped down to it, spending
/// the same real backtest-evaluation budget a GA run at that size would.
pub const BO_MAX_TOTAL_EVALS: usize = 150;

/// Acquisition-maximization search budget used INSIDE the iterative BO loop
/// (as opposed to the more thorough search the standalone
/// `maximize_acquisition` unit tests use, where the GP is small and fixed).
/// `predict_var` costs O(n) via a forward-solve against the Cholesky factor,
/// so each candidate evaluated during acquisition maximization costs O(n) --
/// and this runs once per real BO iteration, with n growing every iteration.
/// (n_restarts * n_local_iters) candidates per outer iteration, summed
/// across a growing n, must stay cheap relative to n^3 (the GP refit cost)
/// or acquisition search dominates runtime instead of the model it's
/// supposed to be cheap relative to. 6*30=180 candidates/iteration is ample
/// for a 2-6 dimensional space and keeps this firmly a minor cost next to
/// the refit.
const BO_INNER_SEARCH_RESTARTS: usize = 6;
const BO_INNER_SEARCH_LOCAL_ITERS: usize = 30;

/// Run Bayesian optimization as a search mechanism, structurally parallel to
/// `AdaptiveGeneticOptimizer::run()` -- takes a total real-evaluation budget
/// (the BO analogue of `population_size * generations`, so a guided-GA-vs-BO
/// comparison spends the same real backtest cost) and returns the best
/// `(chromosome, fitness)` found, the same shape the GA path returns.
///
/// Two phases:
/// 1. **Initialization**: the first `max(dims + 1, total_evals / 5)` points
///    (capped at the full budget) are Latin Hypercube samples -- the GP has
///    nothing to condition on yet, so uniform coverage is the right thing to
///    spend early evaluations on, same reasoning `bo_landscape_heatmap` uses
///    for visualization (just serving optimization here, not a picture).
/// 2. **Iterative refinement**: for the remaining budget, fit a GP on every
///    point observed so far, propose the next point via UCB acquisition
///    maximization (`maximize_acquisition`), evaluate it for real, fold it
///    back into the observed set, repeat.
///
/// `kappa` is the UCB exploration weight passed straight through to
/// `maximize_acquisition` -- higher explores more, matching that function's
/// own doc comment.
pub async fn run_bayesian_optimization(
    schema: &Arc<ParameterSchema>,
    fitness_fn: &AsyncFitnessFn<DynamicChromosome>,
    total_evals: usize,
    kappa: f64,
    on_eval: &(dyn Fn() + Send + Sync),
) -> (DynamicChromosome, f64) {
    let dims = schema.params.len().max(1);
    let total_evals = total_evals.min(BO_MAX_TOTAL_EVALS).max(1);
    // NOT rand::thread_rng() -- this fn is spawned inside a tokio task and
    // holds `rng` across multiple `fitness_fn(...).await` points (unlike
    // bo_landscape_heatmap's rng, which is scoped to a single non-async
    // block before its own await loop begins). ThreadRng is
    // Rc<UnsafeCell<..>>-backed and not Send, so it can't survive a yield
    // point in a future that must itself be Send. StdRng has no thread-local
    // storage and is Send, safe to hold for the whole function.
    let mut rng = rand::rngs::StdRng::from_entropy();

    let n_init = (total_evals / 5).max(dims + 1).min(total_evals);

    let to_chromosome = |point: &[f64]| -> DynamicChromosome {
        let mut chromo = schema.default_chromosome();
        for (i, def) in schema.params.iter().enumerate() {
            let (min, max) = param_range(&def.param_type);
            let raw = min + point[i] * (max - min).max(1e-9);
            let mut gene = f64_to_gene(&def.param_type, raw);
            def.enforce_bounds(&mut gene);
            chromo.genes[i] = gene;
        }
        chromo
    };

    let mut xs_norm: Vec<Vec<f64>> = Vec::with_capacity(total_evals);
    let mut ys: Vec<f64> = Vec::with_capacity(total_evals);
    let mut best_chromo: Option<DynamicChromosome> = None;
    let mut best_fitness = f64::NEG_INFINITY;

    let record = |chromo: DynamicChromosome, fitness: f64, best_chromo: &mut Option<DynamicChromosome>, best_fitness: &mut f64| {
        if fitness > *best_fitness {
            *best_fitness = fitness;
            *best_chromo = Some(chromo);
        }
    };

    // Phase 1: LHS initialization.
    let init_points = latin_hypercube_sample(n_init, dims, &mut rng);
    for point in init_points {
        let chromo = to_chromosome(&point);
        let result: FitnessResult = fitness_fn(&chromo).await;
        on_eval();
        record(chromo, result.fitness, &mut best_chromo, &mut best_fitness);
        xs_norm.push(point);
        ys.push(result.fitness);
    }

    // Phase 2: iterative acquisition-guided refinement for the rest of the budget.
    let remaining = total_evals.saturating_sub(n_init);
    for _ in 0..remaining {
        let mut gp = GpSurrogate::new(0.3, 1e-3);
        gp.fit(xs_norm.clone(), ys.clone());
        let next_point = maximize_acquisition(
            &gp, dims, kappa, BO_INNER_SEARCH_RESTARTS, BO_INNER_SEARCH_LOCAL_ITERS, &mut rng,
        );
        let chromo = to_chromosome(&next_point);
        let result: FitnessResult = fitness_fn(&chromo).await;
        on_eval();
        record(chromo, result.fitness, &mut best_chromo, &mut best_fitness);
        xs_norm.push(next_point);
        ys.push(result.fitness);
    }

    (
        best_chromo.unwrap_or_else(|| schema.default_chromosome()),
        if best_fitness.is_finite() { best_fitness } else { 0.0 },
    )
}

/// Return (min, max) of a parameter in its native f64 representation.
fn param_range(pt: &ParamType) -> (f64, f64) {
    match pt {
        ParamType::Float { min, max } => (*min, *max),
        ParamType::Int { min, max }   => (*min as f64, *max as f64),
        ParamType::Bool               => (0.0, 1.0),
    }
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

/// Complete robustness report produced after GA optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GAReport {
    /// Best parameter set found by GA.
    pub best_params: HashMap<String, serde_json::Value>,
    /// Best fitness.
    pub best_fitness: f64,
    /// Top N diverse parameter sets.
    pub top_n_params: Vec<TopParamSet>,
    /// 2D landscape heatmaps over the most-sensitive parameter pairs.
    pub landscapes: Vec<LandscapeReport>,
    /// Per-parameter sensitivity analysis.
    pub sensitivity: Vec<SensitivityEntry>,
    /// Convergence curve: (generation, best_fitness, mean_fitness).
    pub convergence_curve: Vec<ConvergencePoint>,
    /// Overfitting gap: train vs test Sharpe comparison.
    pub overfitting: Option<OverfitReport>,
    /// Pareto front solutions (multi-objective mode only).
    pub pareto_front: Option<Vec<crate::nsga2::ParetoSolution>>,
    /// Pareto hypervolume indicator (2-D only).
    pub pareto_hypervolume: Option<f64>,
    /// Objective definitions used in multi-objective mode.
    pub pareto_objectives: Option<Vec<crate::nsga2::ObjectiveDef>>,
    /// Dual-seed stability comparison (only populated when
    /// `GeneticConfig.check_seed_stability` is enabled and a `random_seed`
    /// was configured).
    pub seed_stability: Option<SeedStabilityReport>,
    /// The authoritative count of real (non-cached) fitness evaluations this
    /// validation run actually performed -- across the main GA/NSGA-II
    /// search, the sensitivity/landscape sweeps, AND every per-window mini-GA
    /// rerun from GA-driven walk-forward, all of which are genuine additional
    /// searches against the same underlying data and therefore genuine
    /// multiple-comparisons trials.
    ///
    /// This exists because `population_size * convergence_curve.len()` (the
    /// figure callers used to reconstruct themselves) is wrong in both
    /// directions at once: it OVERcounts, since fitness memoization means
    /// carried-over elites and duplicate chromosomes are scored from a cache
    /// rather than re-run, yet still counted as if fresh; and it
    /// UNDERcounts, since multi-objective (NSGA-II) runs don't populate
    /// `convergence_curve` at all (so `gens` silently floors to 1 regardless
    /// of how many generations actually ran) and GA-driven walk-forward's
    /// per-window mini-GA reruns -- each a fully independent
    /// population x generations search -- were never counted anywhere.
    /// Callers (`worker.rs`) should read this field directly instead of
    /// re-deriving an approximation from `convergence_curve.len()`.
    #[serde(default)]
    pub total_fresh_evaluations: Option<usize>,
}

/// A top-N parameter set from the final population.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopParamSet {
    pub params: HashMap<String, serde_json::Value>,
    pub fitness: f64,
    /// Normalized distance from the best solution.
    pub distance_from_best: f64,
}

/// A single point in the 2D landscape heatmap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandscapePoint {
    pub param_a_val: f64,
    pub param_b_val: f64,
    pub fitness: f64,
}

/// 2D fitness landscape over two parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandscapeReport {
    /// Name of the X-axis parameter.
    pub param_a_name: String,
    /// Name of the Y-axis parameter.
    pub param_b_name: String,
    /// Grid resolution (param_a_steps × param_b_steps).
    pub grid_size: usize,
    /// Grid points with fitness values.
    pub grid: Vec<LandscapePoint>,
    /// Metrics derived from the landscape.
    pub flatness_score: f64,  // 0 = sharp peak (fragile), 1 = flat plateau (robust)
}

/// Per-parameter sensitivity: how much fitness changes when this parameter is perturbed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityEntry {
    pub param_name: String,
    /// Fitness at -25% perturbation of range.
    pub fitness_minus_25: f64,
    /// Fitness at -10% perturbation.
    pub fitness_minus_10: f64,
    /// Fitness at optimal value.
    pub fitness_at_optimal: f64,
    /// Fitness at +10% perturbation.
    pub fitness_plus_10: f64,
    /// Fitness at +25% perturbation.
    pub fitness_plus_25: f64,
    /// Sensitivity score: higher = more fragile.
    pub sensitivity_score: f64,
}

/// Convergence curve data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergencePoint {
    pub generation: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
}

/// Discount ratio applied to a GA's raw evaluation count (`GAReport::
/// total_fresh_evaluations`) before it feeds DSR's multiple-comparisons
/// trial count -- an approximation of "how many of these evaluations were
/// genuinely independent trials" using data already collected by every GA
/// run, without requiring new per-evaluation instrumentation.
///
/// The problem this addresses: DSR treats every GA evaluation as an
/// independent random draw, but a GA doesn't sample parameter space
/// uniformly -- selection pressure concentrates later generations around
/// already-promising regions, so many evaluations are refinements of a
/// known solution rather than independent bets. Counting them at full
/// weight over-penalizes a genuinely GA-optimized strategy's DSR relative
/// to what a correlation-aware trial count would show. A fully rigorous
/// fix would track each population member's own return series and apply
/// an `effective_breadth`-style (Grinold) correlation adjustment across
/// all of them -- that data isn't currently collected (would need new
/// per-evaluation instrumentation in the GA's hot loop), so this is a
/// principled, conservative proxy instead, not a replacement for that
/// eventual, more rigorous version.
///
/// Proxy: the fraction of generations whose best_fitness meaningfully
/// improved over the previous generation's. A generation whose incumbent
/// didn't improve found nothing better than what the GA already knew --
/// its evaluations still happened and still carry real overfitting risk
/// against the same data, but they didn't expand the set of genuinely
/// distinct candidate solutions the way an improving generation did, so
/// weighting them at full strength toward the multiple-comparisons penalty
/// is likely too harsh. The first generation always counts as informative
/// (it improves over "nothing"). Floored at 0.1 -- even a fully plateaued
/// search still represents real evaluations against real data with real
/// overfitting risk, never discounted by more than 10x. Returns 1.0 (no
/// discount) when there's no usable convergence curve to measure from
/// (e.g. NSGA-II multi-objective runs, which don't populate it) --
/// conservative in the "don't under-penalize" direction when the signal
/// isn't available.
pub fn effective_trial_ratio(convergence_curve: &[ConvergencePoint]) -> f64 {
    if convergence_curve.len() < 2 {
        return 1.0;
    }
    let improved = convergence_curve
        .windows(2)
        .filter(|w| w[1].best_fitness > w[0].best_fitness + 1e-9)
        .count();
    // +1 credits the first generation as informative by construction.
    let informative = improved + 1;
    let ratio = informative as f64 / convergence_curve.len() as f64;
    ratio.max(0.1)
}

/// Overfitting gap report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverfitReport {
    /// Sharpe on training (in-sample) data.
    pub train_sharpe: f64,
    /// Sharpe on test (out-of-sample) data.
    pub test_sharpe: f64,
    /// Degradation: (train - test) / train × 100%.
    pub degradation_pct: f64,
    /// Whether the gap exceeds the 30% threshold.
    pub is_overfit: bool,
}

/// Relative fitness difference (as a fraction of the larger magnitude) below
/// which two GA runs' best fitness are considered "comparably good" rather
/// than one run simply finding a worse optimum.
pub const SEED_STABILITY_FITNESS_PARITY: f64 = 0.15;
/// Normalized parameter-space distance (via `Chromosome::distance()`) above
/// which two comparably-fit runs are flagged as fragile/overfit rather than
/// converging on the same stable optimum.
pub const SEED_STABILITY_DISTANCE_THRESHOLD: f64 = 0.3;

/// Result of running the GA twice with two different fixed seeds and
/// comparing the two runs' best parameters. A large parameter-space
/// divergence between runs that found comparably good fitness is a concrete,
/// data-driven signal that the "best" parameters are an artifact of one
/// particular random initialization/evaluation order (overfit to noise in
/// the fitness landscape) rather than a real, stable optimum -- unlike the
/// single-run `OverfitReport`, which only measures in-sample vs. out-of-
/// sample degradation for ONE parameter set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedStabilityReport {
    pub seed_a: u64,
    pub seed_b: u64,
    pub best_fitness_a: f64,
    pub best_fitness_b: f64,
    /// Normalized parameter-space distance between the two runs' best
    /// chromosomes (`Chromosome::distance()`). 0.0 = identical parameters.
    pub param_distance: f64,
    /// `true` when both runs found comparably good fitness (within
    /// `SEED_STABILITY_FITNESS_PARITY`) but their best parameters diverged
    /// beyond `SEED_STABILITY_DISTANCE_THRESHOLD` -- treat the reported
    /// "best" parameters with extra skepticism when this is set.
    pub fragile: bool,
}

impl SeedStabilityReport {
    /// Builds the report and computes `fragile` from the fitness-parity and
    /// distance thresholds above. `param_distance` should already be
    /// normalized (e.g. from `Chromosome::distance()`).
    pub fn new(seed_a: u64, seed_b: u64, best_fitness_a: f64, best_fitness_b: f64, param_distance: f64) -> Self {
        let max_abs = best_fitness_a.abs().max(best_fitness_b.abs()).max(1e-9);
        let fitness_delta_frac = (best_fitness_a - best_fitness_b).abs() / max_abs;
        let fragile = fitness_delta_frac <= SEED_STABILITY_FITNESS_PARITY
            && param_distance > SEED_STABILITY_DISTANCE_THRESHOLD;
        Self { seed_a, seed_b, best_fitness_a, best_fitness_b, param_distance, fragile }
    }
}


// ---------------------------------------------------------------------------
// Landscape heatmap
// ---------------------------------------------------------------------------

/// Generate a 2D fitness landscape by sweeping two parameters while holding
/// all others at the best chromosome's values.
///
/// `grid_size` controls resolution per axis (total evals = grid_size²).
pub async fn generate_landscape_heatmap(
    schema: &Arc<ParameterSchema>,
    fitness_fn: &AsyncFitnessFn<DynamicChromosome>,
    best: &DynamicChromosome,
    param_a_idx: usize,
    param_b_idx: usize,
    grid_size: usize,
    on_eval: &(dyn Fn() + Send + Sync),
) -> LandscapeReport {
    let start = Instant::now();

    let param_a = &schema.params[param_a_idx];
    let param_b = &schema.params[param_b_idx];

    let a_steps = generate_steps(&param_a.param_type, grid_size);
    let b_steps = generate_steps(&param_b.param_type, grid_size);

    let mut grid = Vec::with_capacity(a_steps.len() * b_steps.len());

    for &a_val in &a_steps {
        for &b_val in &b_steps {
            // Clone best and override the two swept parameters
            let mut probe = best.clone();
            probe.genes[param_a_idx] = f64_to_gene(&param_a.param_type, a_val);
            probe.genes[param_b_idx] = f64_to_gene(&param_b.param_type, b_val);

            let result: FitnessResult = fitness_fn(&probe).await;
            on_eval();
            grid.push(LandscapePoint {
                param_a_val: a_val,
                param_b_val: b_val,
                fitness: result.fitness,
            });
        }
    }

    // Compute flatness score: 1 - (stddev / range) of fitness values.
    // High flatness = robust (plateau). Low = sharp peak (fragile).
    let flatness_score = compute_flatness(&grid);

    let elapsed = start.elapsed().as_millis();
    let points = grid.len();
    log_info_structured!(crate::GENETIC_LOGGER, "LANDSCAPE_ANALYSIS_COMPLETED",
        "grid_points" => points,
        "elapsed_ms" => elapsed,
    );

    LandscapeReport {
        param_a_name: param_a.name.clone(),
        param_b_name: param_b.name.clone(),
        grid_size,
        grid,
        flatness_score,
    }
}

/// Generate a 2D fitness landscape using LHS sampling + GP posterior for visualization.
///
/// `n_init` LHS points are evaluated (actual backtest calls). A GP is then fitted on
/// those points and queried over a dense `vis_grid × vis_grid` grid (zero extra backtest
/// calls) to produce a smooth heatmap.
///
/// UCB acquisition is intentionally absent: it is an optimizer objective (minimises regret
/// by oversampling near the current best), not a visualization objective. Using UCB here
/// would bias `flatness_score` upward for fragile strategies by concentrating samples near
/// the GA optimum. LHS gives uniform spatial coverage, which is what an honest landscape
/// visualization requires.
pub async fn bo_landscape_heatmap(
    schema: &Arc<ParameterSchema>,
    fitness_fn: &AsyncFitnessFn<DynamicChromosome>,
    best: &DynamicChromosome,
    param_a_idx: usize,
    param_b_idx: usize,
    n_init: usize,
    vis_grid: usize,
    on_eval: &(dyn Fn() + Send + Sync),
) -> LandscapeReport {
    let start = Instant::now();

    let param_a = &schema.params[param_a_idx];
    let param_b = &schema.params[param_b_idx];
    let (a_min, a_max) = param_range(&param_a.param_type);
    let (b_min, b_max) = param_range(&param_b.param_type);
    let a_span = (a_max - a_min).max(1e-9);
    let b_span = (b_max - b_min).max(1e-9);

    let lhs_pts = {
        let mut rng = rand::thread_rng();
        latin_hypercube_sample(n_init, 2, &mut rng)
    };

    let mut xs_norm: Vec<Vec<f64>> = Vec::with_capacity(n_init);
    let mut ys: Vec<f64> = Vec::with_capacity(n_init);

    // Evaluate LHS points — uniform spatial coverage is the correct objective for visualization.
    for pt in &lhs_pts {
        let a_raw = a_min + pt[0] * a_span;
        let b_raw = b_min + pt[1] * b_span;
        let mut probe = best.clone();
        probe.genes[param_a_idx] = f64_to_gene(&param_a.param_type, a_raw);
        probe.genes[param_b_idx] = f64_to_gene(&param_b.param_type, b_raw);
        let result: FitnessResult = fitness_fn(&probe).await;
        on_eval();
        xs_norm.push(pt.clone());
        ys.push(result.fitness);
    }

    // Fit GP on the LHS observations and generate dense vis grid (zero extra backtest calls).
    let mut gp = GpSurrogate::new(0.5, 0.01);
    gp.fit(xs_norm, ys);
    let mut grid = Vec::with_capacity(vis_grid * vis_grid);
    for i in 0..vis_grid {
        for j in 0..vis_grid {
            let a_t = i as f64 / (vis_grid - 1).max(1) as f64;
            let b_t = j as f64 / (vis_grid - 1).max(1) as f64;
            grid.push(LandscapePoint {
                param_a_val: a_min + a_t * a_span,
                param_b_val: b_min + b_t * b_span,
                fitness: gp.predict_mean(&[a_t, b_t]),
            });
        }
    }

    let flatness_score = compute_flatness(&grid);
    let elapsed = start.elapsed().as_millis();
    log_info_structured!(crate::GENETIC_LOGGER, "BO_LANDSCAPE_COMPLETED",
        "param_a"   => param_a.name.as_str(),
        "param_b"   => param_b.name.as_str(),
        "n_evals"   => n_init,
        "vis_grid"  => vis_grid * vis_grid,
        "elapsed_ms" => elapsed,
    );

    LandscapeReport {
        param_a_name: param_a.name.clone(),
        param_b_name: param_b.name.clone(),
        grid_size: vis_grid,
        grid,
        flatness_score,
    }
}

/// Generate evenly spaced steps for a parameter.
fn generate_steps(param_type: &ParamType, n: usize) -> Vec<f64> {
    match param_type {
        ParamType::Float { min, max } => {
            (0..n)
                .map(|i| *min + (max - min) * (i as f64 / (n - 1).max(1) as f64))
                .collect()
        }
        ParamType::Int { min, max } => {
            let range = (max - min) as f64;
            (0..n)
                .map(|i| (*min as f64 + range * (i as f64 / (n - 1).max(1) as f64)).round())
                .collect()
        }
        ParamType::Bool => vec![0.0, 1.0],
    }
}

/// Convert an f64 back to a Gene of the appropriate type.
fn f64_to_gene(param_type: &ParamType, val: f64) -> Gene {
    match param_type {
        ParamType::Float { min, max } => Gene::Float(val.clamp(*min, *max)),
        ParamType::Int { min, max } => Gene::Int((val.round() as i64).clamp(*min, *max)),
        ParamType::Bool => Gene::Bool(val > 0.5),
    }
}

/// Compute flatness: stddev of fitness normalized to range.
fn compute_flatness(grid: &[LandscapePoint]) -> f64 {
    if grid.is_empty() {
        return 0.5;
    }
    let fitnesses: Vec<f64> = grid.iter().map(|p| p.fitness).collect();
    let max_f = fitnesses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_f = fitnesses.iter().copied().fold(f64::INFINITY, f64::min);
    let range = max_f - min_f;

    if range < 1e-12 {
        return 1.0; // perfectly flat
    }

    let mean = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
    let variance = fitnesses.iter().map(|f| (f - mean).powi(2)).sum::<f64>()
        / fitnesses.len() as f64;
    let stddev = variance.sqrt();

    (1.0 - (stddev / range)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Sensitivity analysis
// ---------------------------------------------------------------------------

/// Perturb each parameter individually at ±10% and ±25% of range from the
/// optimal value, re-evaluate fitness, and measure sensitivity.
pub async fn sensitivity_analysis(
    schema: &Arc<ParameterSchema>,
    fitness_fn: &AsyncFitnessFn<DynamicChromosome>,
    best: &DynamicChromosome,
    optimal_fitness: f64,
    on_eval: &(dyn Fn() + Send + Sync),
) -> Vec<SensitivityEntry> {
    let mut entries = Vec::with_capacity(schema.params.len());

    for (i, def) in schema.params.iter().enumerate() {
        let range = match &def.param_type {
            ParamType::Float { min, max } => max - min,
            ParamType::Int { min, max } => (max - min) as f64,
            ParamType::Bool => {
                // For bools, just test both values
                let mut probe_t = best.clone();
                probe_t.genes[i] = Gene::Bool(true);
                let ft = fitness_fn(&probe_t).await.fitness;
                on_eval();

                let mut probe_f = best.clone();
                probe_f.genes[i] = Gene::Bool(false);
                let ff = fitness_fn(&probe_f).await.fitness;
                on_eval();

                entries.push(SensitivityEntry {
                    param_name: def.name.clone(),
                    fitness_minus_25: ff,
                    fitness_minus_10: ff,
                    fitness_at_optimal: optimal_fitness,
                    fitness_plus_10: ft,
                    fitness_plus_25: ft,
                    sensitivity_score: (ft - ff).abs() / optimal_fitness.abs().max(1e-9),
                });
                continue;
            }
        };

        let optimal_val = best.genes[i].as_f64();
        let perturbations = [-0.25, -0.10, 0.10, 0.25];
        let mut fitnesses = Vec::with_capacity(4);

        for &pct in &perturbations {
            let perturbed = optimal_val + range * pct;
            let mut probe = best.clone();
            probe.genes[i] = f64_to_gene(&def.param_type, perturbed);
            def.enforce_bounds(&mut probe.genes[i]);
            let result = fitness_fn(&probe).await;
            on_eval();
            fitnesses.push(result.fitness);
        }

        // Sensitivity = max deviation from optimal / optimal
        let max_dev = fitnesses
            .iter()
            .map(|f| (f - optimal_fitness).abs())
            .fold(0.0_f64, f64::max);
        let sensitivity_score = max_dev / optimal_fitness.abs().max(1e-9);

        entries.push(SensitivityEntry {
            param_name: def.name.clone(),
            fitness_minus_25: fitnesses[0],
            fitness_minus_10: fitnesses[1],
            fitness_at_optimal: optimal_fitness,
            fitness_plus_10: fitnesses[2],
            fitness_plus_25: fitnesses[3],
            sensitivity_score,
        });
    }

    entries
}

// ---------------------------------------------------------------------------
// Overfitting gap
// ---------------------------------------------------------------------------

/// Compare GA-optimized fitness on training data vs fitness on held-out test data.
/// Degradation > 30% is flagged as overfit.
pub fn overfitting_gap(train_sharpe: f64, test_sharpe: f64) -> OverfitReport {
    let degradation_pct = if train_sharpe.abs() > 1e-9 {
        ((train_sharpe - test_sharpe) / train_sharpe * 100.0).max(0.0)
    } else {
        0.0
    };

    OverfitReport {
        train_sharpe,
        test_sharpe,
        degradation_pct,
        is_overfit: degradation_pct > 30.0,
    }
}

// ---------------------------------------------------------------------------
// Top-N extraction
// ---------------------------------------------------------------------------

/// Extract top N diverse parameter sets from a final population.
///
/// Takes the best `2*n` by fitness, then greedily selects `n` that maximize
/// minimum pairwise distance (diversity).
pub fn extract_top_n(
    population: &[(DynamicChromosome, f64)],
    best: &DynamicChromosome,
    n: usize,
) -> Vec<TopParamSet> {
    if population.is_empty() || n == 0 {
        return vec![];
    }

    // Sort by fitness descending
    let mut sorted: Vec<_> = population.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top 2N candidates
    let candidates: Vec<_> = sorted.into_iter().take(n * 2).collect();

    // Greedy: always include the best, then add the candidate with max min-distance
    let mut selected: Vec<(DynamicChromosome, f64)> = vec![candidates[0].clone()];

    while selected.len() < n && selected.len() < candidates.len() {
        let mut best_idx = None;
        let mut best_min_dist = f64::NEG_INFINITY;

        for (i, (cand, _)) in candidates.iter().enumerate() {
            if selected.iter().any(|(s, _)| s.distance(cand) < 1e-9) {
                continue; // already selected (or duplicate)
            }
            let min_dist = selected
                .iter()
                .map(|(s, _)| s.distance(cand))
                .fold(f64::INFINITY, f64::min);
            if min_dist > best_min_dist {
                best_min_dist = min_dist;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            selected.push(candidates[idx].clone());
        } else {
            break;
        }
    }

    selected
        .into_iter()
        .map(|(chr, fitness)| TopParamSet {
            distance_from_best: chr.distance(best),
            params: chr.to_param_map(),
            fitness,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamic_chromosome::{Gene, ParamType, ParameterDef};

    fn simple_schema() -> Arc<ParameterSchema> {
        Arc::new(ParameterSchema {
            params: vec![
                ParameterDef {
                    name: "x".into(),
                    param_type: ParamType::Float { min: 0.0, max: 10.0 },
                    default: Gene::Float(5.0),
                },
                ParameterDef {
                    name: "y".into(),
                    param_type: ParamType::Float { min: 0.0, max: 10.0 },
                    default: Gene::Float(5.0),
                },
            ],
        })
    }

    #[test]
    fn test_ga_report_total_fresh_evaluations_serde_roundtrip() {
        let report = GAReport {
            best_params: HashMap::new(),
            best_fitness: 1.5,
            top_n_params: vec![],
            landscapes: vec![],
            sensitivity: vec![],
            convergence_curve: vec![],
            overfitting: None,
            pareto_front: None,
            pareto_hypervolume: None,
            pareto_objectives: None,
            seed_stability: None,
            total_fresh_evaluations: Some(1234),
        };
        let json = serde_json::to_string(&report).unwrap();
        let deserialized: GAReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_fresh_evaluations, Some(1234));
    }

    #[test]
    fn test_ga_report_total_fresh_evaluations_defaults_to_none_when_absent() {
        // Deserializing a pre-fix persisted GAReport (missing the new field
        // entirely) must not fail -- #[serde(default)] makes it None, and
        // callers (worker.rs) fall back to the old approximation in that case.
        let json = r#"{
            "best_params": {},
            "best_fitness": 1.5,
            "top_n_params": [],
            "landscapes": [],
            "sensitivity": [],
            "convergence_curve": [],
            "overfitting": null,
            "pareto_front": null,
            "pareto_hypervolume": null,
            "pareto_objectives": null,
            "seed_stability": null
        }"#;
        let deserialized: GAReport = serde_json::from_str(json).unwrap();
        assert_eq!(deserialized.total_fresh_evaluations, None);
    }

    #[test]
    fn test_overfitting_gap_pass() {
        let report = overfitting_gap(1.5, 1.2);
        assert!(!report.is_overfit);
        assert!(report.degradation_pct < 30.0);
    }

    #[test]
    fn test_overfitting_gap_fail() {
        let report = overfitting_gap(2.0, 0.5);
        assert!(report.is_overfit);
        assert!(report.degradation_pct > 30.0);
    }

    #[test]
    fn test_flatness_uniform() {
        let grid = vec![
            LandscapePoint { param_a_val: 0.0, param_b_val: 0.0, fitness: 1.0 },
            LandscapePoint { param_a_val: 1.0, param_b_val: 0.0, fitness: 1.0 },
            LandscapePoint { param_a_val: 0.0, param_b_val: 1.0, fitness: 1.0 },
            LandscapePoint { param_a_val: 1.0, param_b_val: 1.0, fitness: 1.0 },
        ];
        let flat = compute_flatness(&grid);
        assert!((flat - 1.0).abs() < 1e-6); // perfectly flat
    }

    #[test]
    fn test_extract_top_n_empty() {
        let schema = simple_schema();
        let best = schema.default_chromosome();
        let result = extract_top_n(&[], &best, 5);
        assert!(result.is_empty());
    }

    // GP math tests

    #[test]
    fn gp_predicts_observed_values_approximately() {
        // A GP with low noise should interpolate observed points closely.
        let mut gp = GpSurrogate::new(0.5, 1e-4);
        let xs = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let ys = vec![0.0, 1.0, 1.0, 2.0];
        gp.fit(xs.clone(), ys.clone());
        for (x, &y) in xs.iter().zip(&ys) {
            let pred = gp.predict_mean(x);
            assert!((pred - y).abs() < 0.1, "GP mean {pred:.3} far from observed {y} at {x:?}");
        }
    }

    #[test]
    fn gp_variance_is_small_at_observed_and_large_elsewhere() {
        let mut gp = GpSurrogate::new(0.5, 1e-4);
        gp.fit(vec![vec![0.0, 0.0]], vec![1.0]);
        let near_var = gp.predict_var(&[0.0, 0.0]);
        let far_var = gp.predict_var(&[1.0, 1.0]);
        assert!(near_var < 0.01, "variance near observation should be small, got {near_var:.4}");
        assert!(far_var > 0.1, "variance far from observation should be large, got {far_var:.4}");
    }

    // --- N-D generalization coverage (2026-08-18): the two tests above only
    // prove the 2D call shape still works: these prove the GP is genuinely
    // dimension-agnostic, not just compiling with a wider signature.

    #[test]
    fn gp_interpolates_in_3d_not_just_2d() {
        // Same interpolation property as gp_predicts_observed_values_approximately,
        // but in a dimensionality bo_landscape_heatmap never exercises (it's
        // permanently fixed at 2 swept parameters) -- a real BO search over a
        // 3+ parameter strategy needs this to hold.
        let mut gp = GpSurrogate::new(0.5, 1e-4);
        let xs = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        let ys = vec![0.0, 1.0, 1.0, 1.0, 3.0];
        gp.fit(xs.clone(), ys.clone());
        for (x, &y) in xs.iter().zip(&ys) {
            let pred = gp.predict_mean(x);
            assert!((pred - y).abs() < 0.1, "GP mean {pred:.3} far from observed {y} at {x:?}");
        }
    }

    #[test]
    fn gp_variance_behaves_correctly_in_5d() {
        // Same near/far variance property, at a dimensionality closer to a
        // real strategy's parameter_space() (most tested strategies expose
        // 3-6 tunable parameters).
        let mut gp = GpSurrogate::new(0.5, 1e-4);
        gp.fit(vec![vec![0.0, 0.0, 0.0, 0.0, 0.0]], vec![1.0]);
        let near_var = gp.predict_var(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let far_var = gp.predict_var(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        assert!(near_var < 0.01, "variance near observation should be small, got {near_var:.4}");
        assert!(far_var > 0.1, "variance far from observation should be large, got {far_var:.4}");
    }

    #[test]
    fn cholesky_identity_roundtrip() {
        // L * L^T of identity should give identity.
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let l = cholesky(&a, 2);
        // L should be identity for identity input
        assert!((l[0] - 1.0).abs() < 1e-9);
        assert!(l[1].abs() < 1e-9);
        assert!(l[2].abs() < 1e-9);
        assert!((l[3] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn forward_backward_solve_roundtrip() {
        // Solve (L L^T) x = b and verify A x ≈ b.
        let a = vec![4.0, 2.0, 2.0, 3.0]; // 2×2 SPD
        let b = vec![8.0, 6.0];
        let l = cholesky(&a, 2);
        let v = forward_solve(&l, &b, 2);
        let x = backward_solve(&l, &v, 2);
        // Verify A x = b: [4*x0 + 2*x1, 2*x0 + 3*x1]
        let r0 = 4.0 * x[0] + 2.0 * x[1];
        let r1 = 2.0 * x[0] + 3.0 * x[1];
        assert!((r0 - b[0]).abs() < 1e-9, "r0={r0}");
        assert!((r1 - b[1]).abs() < 1e-9, "r1={r1}");
    }

    #[test]
    fn lhs_produces_n_points_in_unit_square() {
        let mut rng = rand::thread_rng();
        let pts = latin_hypercube_sample(10, 2, &mut rng);
        assert_eq!(pts.len(), 10);
        for pt in &pts {
            assert_eq!(pt.len(), 2);
            assert!(pt[0] >= 0.0 && pt[0] <= 1.0, "x out of [0,1]: {}", pt[0]);
            assert!(pt[1] >= 0.0 && pt[1] <= 1.0, "y out of [0,1]: {}", pt[1]);
        }
    }

    #[test]
    fn lhs_produces_n_points_in_unit_hypercube_at_5d() {
        // Proves the per-dimension shuffle generalizes past 2D -- a real BO
        // search's initial batch needs this for a 3-6 parameter strategy.
        let mut rng = rand::thread_rng();
        let pts = latin_hypercube_sample(8, 5, &mut rng);
        assert_eq!(pts.len(), 8);
        for pt in &pts {
            assert_eq!(pt.len(), 5);
            for &v in pt {
                assert!((0.0..=1.0).contains(&v), "coordinate out of [0,1]: {v}");
            }
        }
    }

    // --- run_bayesian_optimization end-to-end (2026-08-18) -- exercises the
    // real ParameterSchema/DynamicChromosome plumbing (not just the GP math
    // in isolation above) against a synthetic fitness function with a known
    // peak, still no real backtest pipeline involved.

    #[tokio::test]
    async fn run_bayesian_optimization_converges_toward_a_known_peak() {
        // simple_schema(): two float params "x","y", each range [0, 10].
        // Fitness peaks at (7, 3) -- negative squared distance, maximized at zero.
        let schema = simple_schema();
        let fitness_fn: AsyncFitnessFn<DynamicChromosome> = Arc::new(|chromo: &DynamicChromosome| {
            let x = chromo.genes[0].as_f64();
            let y = chromo.genes[1].as_f64();
            Box::pin(async move {
                let dist_sq = (x - 7.0).powi(2) + (y - 3.0).powi(2);
                FitnessResult::new(-dist_sq, vec![])
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = FitnessResult> + Send>>
        });

        let (best, best_fitness) =
            run_bayesian_optimization(&schema, &fitness_fn, 60, 1.0, &|| {}).await;

        let x = best.genes[0].as_f64();
        let y = best.genes[1].as_f64();
        let dist = ((x - 7.0).powi(2) + (y - 3.0).powi(2)).sqrt();
        assert!(
            dist < 2.0,
            "BO found ({x:.2}, {y:.2}) with fitness {best_fitness:.3}, too far from the true \
             peak (7, 3): dist={dist:.2}"
        );
    }

    #[tokio::test]
    async fn run_bayesian_optimization_respects_the_max_evals_safety_cap() {
        // A budget far above BO_MAX_TOTAL_EVALS must not attempt a GP refit
        // on anywhere near that many points -- clamped down silently rather
        // than hanging or blowing up. Cheap fitness fn + small clamp target
        // keeps this test itself fast.
        let schema = simple_schema();
        let eval_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter_for_fn = eval_count.clone();
        let fitness_fn: AsyncFitnessFn<DynamicChromosome> = Arc::new(move |chromo: &DynamicChromosome| {
            let x = chromo.genes[0].as_f64();
            counter_for_fn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Box::pin(async move { FitnessResult::new(-x, vec![]) })
                as std::pin::Pin<Box<dyn std::future::Future<Output = FitnessResult> + Send>>
        });

        let _ = run_bayesian_optimization(&schema, &fitness_fn, 50_000, 1.0, &|| {}).await;
        let actual_evals = eval_count.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            actual_evals <= BO_MAX_TOTAL_EVALS,
            "expected at most {BO_MAX_TOTAL_EVALS} real evaluations, got {actual_evals}"
        );
    }

    // --- Acquisition function / inner search (2026-08-18) -- no backtest
    // pipeline involved, pure math against known synthetic objectives.

    #[test]
    fn maximize_acquisition_finds_a_near_optimal_point_on_a_known_peak() {
        // Fit a GP to samples of a simple, single-peaked 2D function with a
        // known analytical maximum at (0.7, 0.3), then verify pure-exploitation
        // acquisition maximization (kappa=0) converges close to it.
        let peak = [0.7_f64, 0.3_f64];
        let f = |x: &[f64]| -((x[0] - peak[0]).powi(2) + (x[1] - peak[1]).powi(2));

        let mut rng = rand::thread_rng();
        let samples = latin_hypercube_sample(20, 2, &mut rng);
        let ys: Vec<f64> = samples.iter().map(|s| f(s)).collect();

        let mut gp = GpSurrogate::new(0.3, 1e-4);
        gp.fit(samples, ys);

        let proposed = maximize_acquisition(&gp, 2, 0.0, 20, 200, &mut rng);
        let dist = ((proposed[0] - peak[0]).powi(2) + (proposed[1] - peak[1]).powi(2)).sqrt();
        assert!(
            dist < 0.15,
            "proposed point {proposed:?} too far from true peak {peak:?} (dist={dist:.3})"
        );
    }

    #[test]
    fn maximize_acquisition_does_not_lose_to_plain_random_sampling() {
        // The proposed point's acquisition score should not be worse than the
        // best of a comparable number of purely random samples -- a sanity
        // floor proving the local search is actually doing something smarter
        // than chance, not a strong optimality claim.
        let peak = [0.5_f64, 0.5_f64, 0.5_f64];
        let f = |x: &[f64]| -x.iter().zip(peak.iter()).map(|(xi, pi)| (xi - pi).powi(2)).sum::<f64>();

        let mut rng = rand::thread_rng();
        let samples = latin_hypercube_sample(15, 3, &mut rng);
        let ys: Vec<f64> = samples.iter().map(|s| f(s)).collect();
        let mut gp = GpSurrogate::new(0.3, 1e-4);
        gp.fit(samples, ys);

        let proposed = maximize_acquisition(&gp, 3, 0.0, 20, 200, &mut rng);
        let proposed_score = acquisition_ucb(&gp, &proposed, 0.0);

        let random_best = (0..50)
            .map(|_| {
                let pt: Vec<f64> = (0..3).map(|_| rng.gen::<f64>()).collect();
                acquisition_ucb(&gp, &pt, 0.0)
            })
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            proposed_score >= random_best - 1e-6,
            "maximize_acquisition ({proposed_score:.4}) did worse than the best of 50 random samples ({random_best:.4})"
        );
    }

    #[test]
    fn higher_kappa_explores_more_than_lower_kappa() {
        // With one observed point, a kappa=0 (pure exploitation) search
        // should propose a point close to that observation (highest
        // predicted mean is right there); a large kappa (exploration-heavy)
        // should propose something farther away, chasing the higher
        // uncertainty in unobserved regions instead.
        let mut rng = rand::thread_rng();
        let mut gp = GpSurrogate::new(0.3, 1e-4);
        gp.fit(vec![vec![0.5, 0.5]], vec![1.0]);

        let exploit_pt = maximize_acquisition(&gp, 2, 0.0, 20, 200, &mut rng);
        let explore_pt = maximize_acquisition(&gp, 2, 5.0, 20, 200, &mut rng);

        let dist = |p: &[f64]| ((p[0] - 0.5_f64).powi(2) + (p[1] - 0.5_f64).powi(2)).sqrt();
        assert!(
            dist(&explore_pt) > dist(&exploit_pt),
            "high-kappa point (dist={:.3}) should be farther from the sole observation than \
             low-kappa point (dist={:.3})",
            dist(&explore_pt),
            dist(&exploit_pt)
        );
    }

    #[test]
    fn lhs_covers_every_stratum_exactly_once_per_dimension() {
        // The defining LHS property: for each dimension independently, the n
        // points fall into n distinct strata (no two points share a stratum
        // on that axis) -- what actually distinguishes LHS from plain random
        // sampling and gives it better space-filling coverage.
        let mut rng = rand::thread_rng();
        let n = 12;
        let dims = 4;
        let pts = latin_hypercube_sample(n, dims, &mut rng);
        for d in 0..dims {
            let mut strata: Vec<usize> = pts.iter().map(|p| (p[d] * n as f64).floor() as usize).collect();
            strata.sort_unstable();
            strata.dedup();
            assert_eq!(strata.len(), n, "dimension {d} did not hit all {n} strata exactly once");
        }
    }

    fn cp(generation: usize, best_fitness: f64) -> ConvergencePoint {
        ConvergencePoint { generation, best_fitness, mean_fitness: best_fitness }
    }

    #[test]
    fn effective_trial_ratio_is_1_when_no_curve_or_single_point() {
        assert_eq!(effective_trial_ratio(&[]), 1.0);
        assert_eq!(effective_trial_ratio(&[cp(0, 1.0)]), 1.0);
    }

    #[test]
    fn effective_trial_ratio_is_1_when_every_generation_improves() {
        let curve: Vec<ConvergencePoint> = (0..10).map(|g| cp(g, g as f64)).collect();
        assert_eq!(effective_trial_ratio(&curve), 1.0);
    }

    #[test]
    fn effective_trial_ratio_discounts_a_fully_plateaued_search_to_the_floor() {
        // First generation finds fitness=1.0; the other 9 never improve on it.
        let mut curve = vec![cp(0, 1.0)];
        curve.extend((1..10).map(|g| cp(g, 1.0)));
        // 1 informative generation (the first) out of 10 = 0.1, exactly the floor.
        assert!((effective_trial_ratio(&curve) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn effective_trial_ratio_never_discounts_below_the_floor() {
        // Even a single-generation improvement out of 1000 stays >= 0.1,
        // not 1/1000 -- the floor caps how aggressive the discount can be.
        let mut curve = vec![cp(0, 1.0)];
        curve.extend((1..1000).map(|g| cp(g, 1.0)));
        assert_eq!(effective_trial_ratio(&curve), 0.1);
    }

    #[test]
    fn effective_trial_ratio_reflects_partial_improvement() {
        // Generations: 0 (base, informative), 1 (improves), 2 (plateau),
        // 3 (improves), 4 (plateau) -- 3 informative out of 5 = 0.6.
        let curve = vec![cp(0, 1.0), cp(1, 2.0), cp(2, 2.0), cp(3, 3.0), cp(4, 3.0)];
        assert!((effective_trial_ratio(&curve) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn effective_trial_ratio_ignores_floating_point_noise_as_non_improvement() {
        // An "improvement" of 1e-12 is numerical noise, not a real improving
        // generation -- only generation 0 counts as informative, 1 of 3 = 0.333
        // (above the 0.1 floor, so not floored).
        let curve = vec![cp(0, 1.0), cp(1, 1.0 + 1e-12), cp(2, 1.0 + 2e-12)];
        assert!((effective_trial_ratio(&curve) - (1.0 / 3.0)).abs() < 1e-9);
    }
}
