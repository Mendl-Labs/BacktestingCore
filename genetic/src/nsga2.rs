//! NSGA-II multi-objective Pareto optimizer.
//!
//! Implements Deb et al. (2002) "A Fast and Elitist Multiobjective Genetic
//! Algorithm: NSGA-II" with:
//! - Fast non-dominated sorting  — O(MN²)
//! - Crowding distance assignment — preserves diversity along fronts
//! - Tournament selection by (rank, then crowding distance)
//! - Knee-point selection for backward-compatible single-best output

use std::sync::Arc;
use rand::Rng;
use rand::SeedableRng;
use serde::{Serialize, Deserialize};

use crate::{
    Chromosome, FitnessResult, FitnessContext,
    AsyncContextFitnessFn, AdaptiveSamplingConfig,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Direction of an objective.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Direction {
    Maximize,
    Minimize,
}

/// Definition of a single objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveDef {
    /// Field name matching [`FitnessResult::get_objective`].
    pub field: String,
    /// Whether to maximise or minimise this objective.
    pub direction: Direction,
}

/// A solution on the Pareto front with its rank and crowding distance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoSolution {
    /// Index in the final population.
    pub index: usize,
    /// Non-domination rank (0 = first front).
    pub rank: usize,
    /// Crowding distance.
    pub crowding_distance: f64,
    /// Objective values in the same order as the `ObjectiveDef` vec.
    pub objective_values: Vec<f64>,
    /// The corresponding FitnessResult.
    #[serde(skip)]
    pub fitness_result: Option<FitnessResult>,
}

/// Result of NSGA-II optimisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NsgaIIResult {
    /// All solutions on the first Pareto front.
    pub pareto_front: Vec<ParetoSolution>,
    /// The knee-point solution (best trade-off) index within `pareto_front`.
    pub knee_index: usize,
    /// Hypervolume indicator of the first front (2-D only; 0.0 otherwise).
    pub hypervolume: f64,
    /// Objective definitions used.
    pub objectives: Vec<ObjectiveDef>,
}

/// NSGA-II optimizer, generic over chromosome type.
pub struct NsgaIIOptimizer<C: Chromosome> {
    pub population_size: usize,
    pub generations: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub objectives: Vec<ObjectiveDef>,
    pub fitness_fn: AsyncContextFitnessFn<C>,
    pub sampling_config: AdaptiveSamplingConfig,
    pub force_sequential: bool,
}

// ---------------------------------------------------------------------------
// Core NSGA-II algorithms (pub for testing)
// ---------------------------------------------------------------------------

/// Fast non-dominated sorting.
/// Returns a vector of fronts; each front is a vector of indices into `pop_objectives`.
pub fn fast_non_dominated_sort(
    pop_objectives: &[Vec<f64>],
    objectives: &[ObjectiveDef],
) -> Vec<Vec<usize>> {
    let n = pop_objectives.len();
    let mut domination_count = vec![0usize; n]; // how many dominate p
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n]; // whom p dominates
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut front0: Vec<usize> = Vec::new();

    for p in 0..n {
        for q in 0..n {
            if p == q { continue; }
            if dominates(&pop_objectives[p], &pop_objectives[q], objectives) {
                dominated_set[p].push(q);
            } else if dominates(&pop_objectives[q], &pop_objectives[p], objectives) {
                domination_count[p] += 1;
            }
        }
        if domination_count[p] == 0 {
            front0.push(p);
        }
    }

    fronts.push(front0);

    let mut i = 0;
    while !fronts[i].is_empty() {
        let mut next_front = Vec::new();
        for &p in &fronts[i] {
            for &q in &dominated_set[p] {
                domination_count[q] -= 1;
                if domination_count[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        if next_front.is_empty() { break; }
        fronts.push(next_front);
        i += 1;
    }

    fronts
}

/// Crowding distance assignment for one front.
pub fn crowding_distance_assignment(
    front: &[usize],
    pop_objectives: &[Vec<f64>],
    num_objectives: usize,
) -> Vec<f64> {
    let l = front.len();
    if l <= 2 {
        return vec![f64::INFINITY; l];
    }

    let mut distances = vec![0.0f64; l];

    for m in 0..num_objectives {
        // Sort front members by objective m
        let mut sorted_indices: Vec<usize> = (0..l).collect();
        sorted_indices.sort_by(|&a, &b| {
            pop_objectives[front[a]][m]
                .partial_cmp(&pop_objectives[front[b]][m])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary points get infinite distance
        distances[sorted_indices[0]] = f64::INFINITY;
        distances[sorted_indices[l - 1]] = f64::INFINITY;

        let f_min = pop_objectives[front[sorted_indices[0]]][m];
        let f_max = pop_objectives[front[sorted_indices[l - 1]]][m];
        let range = f_max - f_min;
        if range < 1e-14 { continue; }

        for k in 1..l - 1 {
            let prev = pop_objectives[front[sorted_indices[k - 1]]][m];
            let next = pop_objectives[front[sorted_indices[k + 1]]][m];
            distances[sorted_indices[k]] += (next - prev) / range;
        }
    }

    distances
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// p dominates q iff p is no worse on all objectives and strictly better on at least one.
fn dominates(p: &[f64], q: &[f64], objectives: &[ObjectiveDef]) -> bool {
    let mut dominated_in_some = false;
    for (i, obj) in objectives.iter().enumerate() {
        let (pv, qv) = (p[i], q[i]);
        match obj.direction {
            Direction::Maximize => {
                if pv < qv { return false; }
                if pv > qv { dominated_in_some = true; }
            }
            Direction::Minimize => {
                if pv > qv { return false; }
                if pv < qv { dominated_in_some = true; }
            }
        }
    }
    dominated_in_some
}

/// Select the knee-point: the solution with the largest perpendicular distance
/// from the line connecting the two extreme solutions on the first front.
fn select_knee_point(front: &[ParetoSolution], objectives: &[ObjectiveDef]) -> usize {
    if front.len() <= 2 { return 0; }
    let m = objectives.len();

    // Normalise objectives to [0, 1]
    let mut mins = vec![f64::INFINITY; m];
    let mut maxs = vec![f64::NEG_INFINITY; m];
    for s in front {
        for (j, &v) in s.objective_values.iter().enumerate() {
            if v < mins[j] { mins[j] = v; }
            if v > maxs[j] { maxs[j] = v; }
        }
    }
    let ranges: Vec<f64> = mins.iter().zip(&maxs).map(|(&lo, &hi)| (hi - lo).max(1e-12)).collect();

    // Normalise and flip minimise objectives so larger = better
    let normalised: Vec<Vec<f64>> = front.iter().map(|s| {
        s.objective_values.iter().enumerate().map(|(j, &v)| {
            let n = (v - mins[j]) / ranges[j];
            match objectives[j].direction {
                Direction::Maximize => n,
                Direction::Minimize => 1.0 - n,
            }
        }).collect()
    }).collect();

    // Line from ideal (1,1,...) to nadir (0,0,...)
    // Distance of point from that line = ||cross|| / ||line||
    // For n-D, use projection-based distance.
    let mut best_idx = 0;
    let mut best_dist = f64::NEG_INFINITY;
    let line_len = (m as f64).sqrt(); // ||(1,1,...) - (0,0,...)|| = sqrt(m)

    for (i, pt) in normalised.iter().enumerate() {
        // Project onto line direction (1/√m, 1/√m, ...)
        let proj: f64 = pt.iter().sum::<f64>() / line_len;
        // Distance = ||pt - proj * direction||
        let _dist_sq: f64 = pt.iter().map(|&v| {
            let on_line = proj / (m as f64).sqrt();
            (v - on_line).powi(2)
        }).sum();
        // We actually want the point closest to the ideal direction (highest sum)
        // with maximum perpendicular distance from nadir-ideal line → most balanced trade-off.
        // Simpler: maximise geometric mean of normalised objectives (Nash solution).
        let geo_mean = pt.iter().map(|&v| v.max(1e-12)).product::<f64>().powf(1.0 / m as f64);
        if geo_mean > best_dist {
            best_dist = geo_mean;
            best_idx = i;
        }
    }

    best_idx
}

/// Compute 2-D hypervolume relative to the origin (0, 0).
fn hypervolume_2d(front: &[ParetoSolution], objectives: &[ObjectiveDef]) -> f64 {
    if objectives.len() != 2 || front.is_empty() { return 0.0; }

    // Convert all values so that larger = better
    let mut pts: Vec<(f64, f64)> = front.iter().map(|s| {
        let x = match objectives[0].direction {
            Direction::Maximize => s.objective_values[0],
            Direction::Minimize => -s.objective_values[0],
        };
        let y = match objectives[1].direction {
            Direction::Maximize => s.objective_values[1],
            Direction::Minimize => -s.objective_values[1],
        };
        (x, y)
    }).collect();

    // Sort by first objective descending
    pts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Sweep
    let ref_x = 0.0;
    let ref_y = 0.0;
    let mut hv = 0.0;
    let mut prev_y = ref_y;
    for &(x, y) in &pts {
        if y > prev_y {
            hv += (x - ref_x) * (y - prev_y);
            prev_y = y;
        }
    }
    hv
}

// ---------------------------------------------------------------------------
// Optimizer implementation
// ---------------------------------------------------------------------------

impl<C: Chromosome + 'static> NsgaIIOptimizer<C> {
    /// Run NSGA-II optimisation. Returns the Pareto front and the knee-point best.
    pub async fn run(&self) -> (NsgaIIResult, C, f64) {
        let mut rng = rand::rngs::StdRng::from_entropy();
        let pop_size = self.population_size;
        let mut population: Vec<C> = (0..pop_size).map(|_| C::random(&mut rng)).collect();

        log_info_structured!(crate::GENETIC_LOGGER, "NSGA2_STARTED",
            "pop_size" => pop_size,
            "generations" => self.generations,
        );

        for generation in 0..self.generations {
            let ctx_gen = generation;
            let ctx_total = self.generations;

            // Evaluate fitness
            let fitness_results: Vec<FitnessResult> = if self.force_sequential {
                let mut results = Vec::with_capacity(population.len());
                for chromo in &population {
                    let ctx = FitnessContext::new(ctx_gen, ctx_total, &self.sampling_config);
                    results.push((self.fitness_fn)(chromo, ctx).await);
                }
                results
            } else {
                use rayon::prelude::*;
                let fitness_fn = Arc::clone(&self.fitness_fn);
                let sampling_config = self.sampling_config.clone();
                let handle = tokio::runtime::Handle::current();
                population.par_iter().map(|chromo| {
                    let ctx = FitnessContext::new(ctx_gen, ctx_total, &sampling_config);
                    handle.block_on(fitness_fn(chromo, ctx))
                }).collect()
            };

            // Extract objective values
            let pop_objectives: Vec<Vec<f64>> = fitness_results.iter().map(|r| {
                self.objectives.iter().map(|o| r.get_objective(&o.field)).collect()
            }).collect();

            // Non-dominated sorting
            let fronts = fast_non_dominated_sort(&pop_objectives, &self.objectives);

            // Assign rank + crowding
            let mut rank = vec![0usize; population.len()];
            let mut crowd = vec![0.0f64; population.len()];
            for (r, front) in fronts.iter().enumerate() {
                let cd = crowding_distance_assignment(front, &pop_objectives, self.objectives.len());
                for (fi, &idx) in front.iter().enumerate() {
                    rank[idx] = r;
                    crowd[idx] = cd[fi];
                }
            }

            // Build sorted index by (rank asc, crowding desc)
            let mut sorted: Vec<usize> = (0..population.len()).collect();
            sorted.sort_by(|&a, &b| {
                rank[a].cmp(&rank[b])
                    .then(crowd[b].partial_cmp(&crowd[a]).unwrap_or(std::cmp::Ordering::Equal))
            });

            // Per-generation summary
            let front0_size = fronts.first().map(|f| f.len()).unwrap_or(0);
            log_info_structured!(crate::GENETIC_LOGGER, "NSGA2_GENERATION",
                "generation" => generation,
                "pareto_front_size" => front0_size,
            );

            // Last generation — extract Pareto front and return
            if generation == self.generations - 1 {
                let pareto_indices: Vec<usize> = fronts.first().cloned().unwrap_or_default();
                let pareto_front: Vec<ParetoSolution> = pareto_indices.iter().enumerate().map(|(_fi, &idx)| {
                    ParetoSolution {
                        index: idx,
                        rank: 0,
                        crowding_distance: crowd[idx],
                        objective_values: pop_objectives[idx].clone(),
                        fitness_result: Some(fitness_results[idx].clone()),
                    }
                }).collect();

                let knee = select_knee_point(&pareto_front, &self.objectives);
                let hv = hypervolume_2d(&pareto_front, &self.objectives);

                let knee_chromo = population[pareto_front[knee].index].clone();
                let knee_fitness = fitness_results[pareto_front[knee].index].fitness;

                log_info_structured!(crate::GENETIC_LOGGER, "NSGA2_COMPLETED",
                    "total_generations" => generation + 1,
                    "pareto_size" => pareto_front.len(),
                );

                return (
                    NsgaIIResult {
                        pareto_front,
                        knee_index: knee,
                        hypervolume: hv,
                        objectives: self.objectives.clone(),
                    },
                    knee_chromo,
                    knee_fitness,
                );
            }

            // Selection + reproduction
            let elite_count = (pop_size / 10).max(2);
            let mut new_pop: Vec<C> = sorted.iter().take(elite_count).map(|&i| population[i].clone()).collect();

            while new_pop.len() < pop_size {
                let a = tournament_select(&sorted, &rank, &crowd, &mut rng);
                let b = tournament_select(&sorted, &rank, &crowd, &mut rng);
                let mut child = population[a].clone();
                if rng.gen::<f64>() < self.crossover_rate {
                    child = child.crossover(&population[b], &mut rng);
                }
                child.mutate(&mut rng, self.mutation_rate);
                new_pop.push(child);
            }

            population = new_pop;
        }

        // Fallback (shouldn't reach here)
        let ctx = FitnessContext::full_resolution(0, self.generations);
        let fr = (self.fitness_fn)(&population[0], ctx).await;
        (
            NsgaIIResult {
                pareto_front: vec![],
                knee_index: 0,
                hypervolume: 0.0,
                objectives: self.objectives.clone(),
            },
            population[0].clone(),
            fr.fitness,
        )
    }
}

fn tournament_select(
    sorted: &[usize],
    rank: &[usize],
    crowd: &[f64],
    rng: &mut impl Rng,
) -> usize {
    let n = sorted.len();
    let a = sorted[rng.gen_range(0..n)];
    let b = sorted[rng.gen_range(0..n)];
    // Prefer lower rank, then higher crowding
    if rank[a] < rank[b] {
        a
    } else if rank[b] < rank[a] {
        b
    } else if crowd[a] > crowd[b] {
        a
    } else {
        b
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominates_maximize() {
        let objs = vec![
            ObjectiveDef { field: "a".into(), direction: Direction::Maximize },
            ObjectiveDef { field: "b".into(), direction: Direction::Maximize },
        ];
        assert!(dominates(&[3.0, 4.0], &[2.0, 3.0], &objs));   // strictly better
        assert!(!dominates(&[3.0, 3.0], &[2.0, 4.0], &objs));   // neither dominates
        assert!(!dominates(&[2.0, 3.0], &[3.0, 4.0], &objs));   // worse
    }

    #[test]
    fn test_dominates_minimize() {
        let objs = vec![
            ObjectiveDef { field: "a".into(), direction: Direction::Minimize },
            ObjectiveDef { field: "b".into(), direction: Direction::Minimize },
        ];
        assert!(dominates(&[1.0, 2.0], &[3.0, 4.0], &objs));
        assert!(!dominates(&[3.0, 4.0], &[1.0, 2.0], &objs));
    }

    #[test]
    fn test_non_dominated_sort_simple() {
        let objs = vec![
            ObjectiveDef { field: "a".into(), direction: Direction::Maximize },
            ObjectiveDef { field: "b".into(), direction: Direction::Maximize },
        ];
        // 3 solutions: (3,1) (1,3) (2,2) — all non-dominated with each other
        let pop = vec![vec![3.0, 1.0], vec![1.0, 3.0], vec![2.0, 2.0]];
        let fronts = fast_non_dominated_sort(&pop, &objs);
        assert_eq!(fronts.len(), 1);
        assert_eq!(fronts[0].len(), 3);
    }

    #[test]
    fn test_non_dominated_sort_two_fronts() {
        let objs = vec![
            ObjectiveDef { field: "a".into(), direction: Direction::Maximize },
            ObjectiveDef { field: "b".into(), direction: Direction::Maximize },
        ];
        let pop = vec![
            vec![3.0, 3.0], // front 0 — dominates everything
            vec![2.0, 2.0], // front 1
            vec![1.0, 1.0], // front 2
        ];
        let fronts = fast_non_dominated_sort(&pop, &objs);
        assert_eq!(fronts.len(), 3);
        assert_eq!(fronts[0], vec![0]);
        assert_eq!(fronts[1], vec![1]);
        assert_eq!(fronts[2], vec![2]);
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let pop = vec![vec![1.0, 3.0], vec![2.0, 2.0], vec![3.0, 1.0]];
        let front = vec![0, 1, 2];
        let cd = crowding_distance_assignment(&front, &pop, 2);
        assert!(cd[0].is_infinite());
        assert!(cd[2].is_infinite());
        assert!(cd[1].is_finite());
    }

    #[test]
    fn test_knee_point_selection() {
        let objs = vec![
            ObjectiveDef { field: "a".into(), direction: Direction::Maximize },
            ObjectiveDef { field: "b".into(), direction: Direction::Maximize },
        ];
        let front = vec![
            ParetoSolution { index: 0, rank: 0, crowding_distance: f64::INFINITY, objective_values: vec![10.0, 0.1], fitness_result: None },
            ParetoSolution { index: 1, rank: 0, crowding_distance: 1.0, objective_values: vec![5.0, 5.0], fitness_result: None },
            ParetoSolution { index: 2, rank: 0, crowding_distance: f64::INFINITY, objective_values: vec![0.1, 10.0], fitness_result: None },
        ];
        let knee = select_knee_point(&front, &objs);
        assert_eq!(knee, 1, "balanced (5,5) should be the knee point");
    }
}
