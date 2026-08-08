// Diversity-preserving mechanisms for unique strategy generation
// Prevents strategy crowding and alpha decay when user base grows

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Behavioral signature of a strategy for similarity detection
/// Unlike parameter distance, this captures *how* the strategy trades
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BehavioralSignature {
    /// Average trade duration bucket (0: <1min, 1: 1-10min, 2: 10min-1hr, 3: 1hr+)
    pub hold_time_bucket: u8,
    /// Trading frequency bucket (0: <1/hr, 1: 1-10/hr, 2: 10-60/hr, 3: >60/hr)
    pub frequency_bucket: u8,
    /// Primary entry signal type (market_making, breakout, arbitrage, custom)
    pub signal_type: SignalType,
    /// Position sizing style (0: fixed, 1: kelly, 2: volatility-scaled)
    pub sizing_style: u8,
    /// Active trading hours bitmask (24 bits for each hour)
    pub active_hours: u32,
    /// Preferred assets (hashed set of symbols)
    pub asset_hash: u64,
    /// Risk profile (0: conservative, 1: moderate, 2: aggressive)
    pub risk_profile: u8,
    /// TP/SL ratio bucket (0: <1, 1: 1-2, 2: 2-3, 3: >3)
    pub reward_risk_bucket: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SignalType {
    #[default]
    MarketMaking,
    Breakout,
    Arbitrage,
    Custom,
}

impl BehavioralSignature {
    /// Calculate behavioral distance (0.0 = identical, 1.0 = completely different)
    pub fn distance(&self, other: &Self) -> f64 {
        let mut diff_count = 0u32;
        const TOTAL_FEATURES: u32 = 8;
        
        if self.hold_time_bucket != other.hold_time_bucket { diff_count += 1; }
        if self.frequency_bucket != other.frequency_bucket { diff_count += 1; }
        if self.signal_type != other.signal_type { diff_count += 2; } // Signal type weighted 2x
        if self.sizing_style != other.sizing_style { diff_count += 1; }
        if self.risk_profile != other.risk_profile { diff_count += 1; }
        if self.reward_risk_bucket != other.reward_risk_bucket { diff_count += 1; }
        
        // Active hours: Jaccard distance (different hours = more different)
        let intersection = (self.active_hours & other.active_hours).count_ones();
        let union = (self.active_hours | other.active_hours).count_ones();
        let hours_similarity = if union > 0 { intersection as f64 / union as f64 } else { 1.0 };
        if hours_similarity < 0.5 { diff_count += 1; } // Different active hours
        
        // Asset overlap (different assets = more different)
        if self.asset_hash != other.asset_hash { diff_count += 1; }
        
        diff_count as f64 / TOTAL_FEATURES as f64
    }
}

/// Global registry of deployed strategies for crowding detection
#[derive(Debug, Clone)]
pub struct StrategyRegistry {
    /// Map of strategy_id -> (behavioral_signature, parameter_hash, user_id)
    strategies: Arc<RwLock<HashMap<String, RegisteredStrategy>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredStrategy {
    pub strategy_id: String,
    pub user_id: String,
    pub behavioral_sig: BehavioralSignature,
    pub parameter_hash: u64,
    pub deployed_at: i64,
    /// Current AUM allocated to this strategy (for capacity weighting)
    pub aum_usd: f64,
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyRegistry {
    pub fn new() -> Self {
        Self {
            strategies: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a new deployed strategy
    pub fn register(&self, strategy: RegisteredStrategy) {
        self.strategies.write().insert(strategy.strategy_id.clone(), strategy);
    }
    
    /// Remove a strategy (when stopped)
    pub fn unregister(&self, strategy_id: &str) {
        self.strategies.write().remove(strategy_id);
    }
    
    /// Calculate crowding penalty for a candidate strategy
    /// Returns 0.0 (no penalty) to 1.0 (fully crowded - discard strategy)
    pub fn calculate_crowding_penalty(
        &self,
        candidate_sig: &BehavioralSignature,
        candidate_param_hash: u64,
        user_id: &str,
    ) -> f64 {
        let strategies = self.strategies.read();
        
        if strategies.is_empty() {
            return 0.0; // No crowding possible
        }
        
        let mut similar_count = 0;
        let mut aum_weighted_similarity = 0.0;
        
        for (_, registered) in strategies.iter() {
            // Skip own strategies (user can have similar strategies)
            if registered.user_id == user_id {
                continue;
            }
            
            // Behavioral distance
            let behavioral_dist = candidate_sig.distance(&registered.behavioral_sig);
            
            // Parameter distance (simple hash comparison for now)
            let param_dist = if candidate_param_hash == registered.parameter_hash {
                0.0
            } else {
                // XOR and count differing bits for approximate distance
                let diff_bits = (candidate_param_hash ^ registered.parameter_hash).count_ones();
                (diff_bits as f64 / 64.0).min(1.0)
            };
            
            // Combined similarity (0 = identical, 1 = completely different)
            let similarity = 1.0 - (behavioral_dist * 0.6 + param_dist * 0.4);
            
            // Only count as crowded if very similar (similarity > 0.7)
            if similarity > 0.7 {
                similar_count += 1;
                // Weight by AUM - larger strategies cause more crowding
                let aum_weight = (registered.aum_usd / 100_000.0).min(10.0).max(1.0);
                aum_weighted_similarity += similarity * aum_weight;
            }
        }
        
        if similar_count == 0 {
            return 0.0;
        }
        
        // Crowding penalty scales with number of similar strategies and their AUM
        // Formula: penalty = 1 - 1/(1 + k*similar_count*avg_similarity)
        // where k controls how aggressively we penalize
        const CROWDING_SENSITIVITY: f64 = 0.5;
        let avg_similarity = aum_weighted_similarity / similar_count as f64;
        let penalty = 1.0 - 1.0 / (1.0 + CROWDING_SENSITIVITY * similar_count as f64 * avg_similarity);
        
        penalty.min(0.9) // Cap at 90% penalty - always leave some chance
    }
    
    /// Get count of similar strategies (for UI display)
    pub fn count_similar(&self, sig: &BehavioralSignature, threshold: f64) -> usize {
        self.strategies.read()
            .values()
            .filter(|s| sig.distance(&s.behavioral_sig) < threshold)
            .count()
    }
    
    /// Get total AUM in similar strategies
    pub fn similar_strategies_aum(&self, sig: &BehavioralSignature, threshold: f64) -> f64 {
        self.strategies.read()
            .values()
            .filter(|s| sig.distance(&s.behavioral_sig) < threshold)
            .map(|s| s.aum_usd)
            .sum()
    }
    
    /// Get total count of strategies in registry
    pub fn len(&self) -> usize {
        self.strategies.read().len()
    }
    
    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.strategies.read().is_empty()
    }
}

/// Database-backed registry that loads strategies from PostgreSQL
/// Uses caching with TTL to avoid hitting DB on every fitness evaluation
#[cfg(feature = "postgres")]
pub mod db_registry {
    use super::*;
    use diesel::prelude::*;
    use diesel::r2d2::{ConnectionManager, Pool};
    use diesel::PgConnection;
    use std::time::{Duration, Instant};
    use tokio::sync::RwLock as TokioRwLock;
    use bigdecimal::ToPrimitive;
    
    type DbPool = Pool<ConnectionManager<PgConnection>>;
    
    /// Database-backed strategy registry with caching
    pub struct DatabaseBackedRegistry {
        pool: DbPool,
        cache: Arc<TokioRwLock<CachedStrategies>>,
        cache_ttl: Duration,
    }
    
    struct CachedStrategies {
        strategies: HashMap<String, RegisteredStrategy>,
        last_refresh: Instant,
    }
    
    impl DatabaseBackedRegistry {
        /// Create new registry with database connection pool
        pub fn new(pool: DbPool) -> Self {
            Self {
                pool,
                cache: Arc::new(TokioRwLock::new(CachedStrategies {
                    strategies: HashMap::new(),
                    last_refresh: Instant::now() - Duration::from_secs(3600), // Force initial refresh
                })),
                cache_ttl: Duration::from_secs(300), // 5 minute cache
            }
        }
        
        /// Create with custom TTL
        pub fn with_ttl(pool: DbPool, ttl_secs: u64) -> Self {
            Self {
                pool,
                cache: Arc::new(TokioRwLock::new(CachedStrategies {
                    strategies: HashMap::new(),
                    last_refresh: Instant::now() - Duration::from_secs(3600),
                })),
                cache_ttl: Duration::from_secs(ttl_secs),
            }
        }
        
        /// Refresh cache from database if TTL expired
        pub async fn refresh_if_needed(&self) -> Result<(), diesel::result::Error> {
            let needs_refresh = {
                let cache = self.cache.read().await;
                cache.last_refresh.elapsed() > self.cache_ttl
            };
            
            if needs_refresh {
                self.force_refresh().await?;
            }
            
            Ok(())
        }
        
        /// Force refresh from database
        pub async fn force_refresh(&self) -> Result<(), diesel::result::Error> {
            use databaseschema::schema::deployed_strategies::dsl::*;
            
            let pool = self.pool.clone();
            let strategies_data = tokio::task::spawn_blocking(move || {
                let mut conn = pool.get().map_err(|e| {
                    diesel::result::Error::DatabaseError(
                        diesel::result::DatabaseErrorKind::Unknown,
                        Box::new(e.to_string()),
                    )
                })?;
                
                // Load active strategies with behavioral signatures
                deployed_strategies
                    .filter(is_active.eq(true))
                    .filter(behavioral_signature.is_not_null())
                    .select((
                        id,
                        deployed_by,
                        behavioral_signature,
                        parameter_hash,
                        current_aum,
                        deployed_at,
                    ))
                    .load::<(
                        uuid::Uuid,
                        Option<String>,
                        Option<serde_json::Value>,
                        Option<i64>,
                        Option<bigdecimal::BigDecimal>,
                        chrono::DateTime<chrono::Utc>,
                    )>(&mut conn)
            }).await.map_err(|e| {
                diesel::result::Error::DatabaseError(
                    diesel::result::DatabaseErrorKind::Unknown,
                    Box::new(e.to_string()),
                )
            })??;
            
            // Convert to RegisteredStrategy
            let mut new_strategies = HashMap::new();
            for (strat_id, deployer, sig_json, param_hash, aum, deployed) in strategies_data {
                if let Some(sig_value) = sig_json {
                    if let Ok(sig) = serde_json::from_value::<BehavioralSignature>(sig_value) {
                        new_strategies.insert(strat_id.to_string(), RegisteredStrategy {
                            strategy_id: strat_id.to_string(),
                            user_id: deployer.unwrap_or_default(),
                            behavioral_sig: sig,
                            parameter_hash: param_hash.unwrap_or(0) as u64,
                            deployed_at: deployed.timestamp(),
                            aum_usd: aum.and_then(|a| a.to_f64()).unwrap_or(0.0),
                        });
                    }
                }
            }
            
            // Update cache
            let mut cache = self.cache.write().await;
            cache.strategies = new_strategies;
            cache.last_refresh = Instant::now();
            
            Ok(())
        }
        
        /// Get an in-memory snapshot for fast access during GA
        pub async fn to_in_memory_registry(&self) -> StrategyRegistry {
            let _ = self.refresh_if_needed().await;
            
            let cache = self.cache.read().await;
            let registry = StrategyRegistry::new();
            
            for (_, strat) in cache.strategies.iter() {
                registry.register(strat.clone());
            }
            
            registry
        }
        
        /// Calculate crowding penalty (delegates to cached in-memory registry)
        pub async fn calculate_crowding_penalty(
            &self,
            candidate_sig: &BehavioralSignature,
            candidate_param_hash: u64,
            user_id: &str,
        ) -> f64 {
            let _ = self.refresh_if_needed().await;
            
            let cache = self.cache.read().await;
            
            if cache.strategies.is_empty() {
                return 0.0;
            }
            
            let mut similar_count = 0;
            let mut aum_weighted_similarity = 0.0;
            
            for (_, registered) in cache.strategies.iter() {
                if registered.user_id == user_id {
                    continue;
                }
                
                let behavioral_dist = candidate_sig.distance(&registered.behavioral_sig);
                let param_dist = if candidate_param_hash == registered.parameter_hash {
                    0.0
                } else {
                    let diff_bits = (candidate_param_hash ^ registered.parameter_hash).count_ones();
                    (diff_bits as f64 / 64.0).min(1.0)
                };
                
                let similarity = 1.0 - (behavioral_dist * 0.6 + param_dist * 0.4);
                
                if similarity > 0.7 {
                    similar_count += 1;
                    let aum_weight = (registered.aum_usd / 100_000.0).min(10.0).max(1.0);
                    aum_weighted_similarity += similarity * aum_weight;
                }
            }
            
            if similar_count == 0 {
                return 0.0;
            }
            
            const CROWDING_SENSITIVITY: f64 = 0.5;
            let avg_similarity = aum_weighted_similarity / similar_count as f64;
            let penalty = 1.0 - 1.0 / (1.0 + CROWDING_SENSITIVITY * similar_count as f64 * avg_similarity);
            
            penalty.min(0.9)
        }
        
        /// Get count of strategies in cache
        pub async fn len(&self) -> usize {
            self.cache.read().await.strategies.len()
        }
    }
}

/// Configuration for diversity-preserving GA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityConfig {
    /// Enable fitness sharing (penalize similar individuals within population)
    pub enable_fitness_sharing: bool,
    /// Sharing radius - individuals closer than this share fitness
    pub sharing_radius: f64,
    /// Enable crowding penalty (penalize similarity to deployed strategies)
    pub enable_crowding_penalty: bool,
    /// Weight of crowding penalty (0.0-1.0)
    pub crowding_weight: f64,
    /// Enable behavioral diversity bonus
    pub enable_behavioral_bonus: bool,
    /// Weight of behavioral novelty bonus
    pub behavioral_bonus_weight: f64,
    /// Minimum required behavioral distance from any deployed strategy
    pub min_behavioral_distance: f64,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            enable_fitness_sharing: true,
            sharing_radius: 0.15,
            enable_crowding_penalty: true,
            crowding_weight: 0.3, // 30% of fitness can be penalized
            enable_behavioral_bonus: true,
            behavioral_bonus_weight: 0.1, // 10% bonus for unique strategies
            min_behavioral_distance: 0.3, // Reject strategies closer than 30%
        }
    }
}

/// Fitness sharing for within-population diversity
/// Reduces fitness of individuals that are too similar to each other
/// Modifies fitnesses in-place
pub fn apply_fitness_sharing<C: crate::Chromosome>(
    fitnesses: &mut [f64],
    population: &[C],
    sharing_radius: f64,
) {
    let n = population.len();
    let raw_fitnesses: Vec<f64> = fitnesses.to_vec();

    for i in 0..n {
        let mut niche_count = 0.0;

        for j in 0..n {
            let distance = population[i].distance(&population[j]);
            if distance < sharing_radius {
                // Triangular sharing function
                niche_count += 1.0 - (distance / sharing_radius);
            }
        }

        // Shared fitness: crowding must always be a PENALTY regardless of
        // sign. Dividing a NEGATIVE fitness by the niche count moves it
        // toward zero, i.e. REWARDS crowded bad chromosomes — a converged
        // cluster of zero-trade chromosomes (sentinel fitness) was boosted
        // from -1.0 to ~-0.15 and won the GA (job 64e39acf). For negative
        // fitness, multiply by niche count instead so crowding makes it worse.
        let niche = niche_count.max(1.0);
        fitnesses[i] = if raw_fitnesses[i] >= 0.0 {
            raw_fitnesses[i] / niche
        } else {
            raw_fitnesses[i] * niche
        };
    }

    // Log summary after fitness sharing is applied
    let avg_penalty: f64 = raw_fitnesses.iter().zip(fitnesses.iter())
        .map(|(raw, shared)| raw - shared)
        .sum::<f64>() / n as f64;
    log_info!(
        "FITNESS_SHARING_APPLIED pop_size={} sharing_radius={:.3} avg_penalty={:.4}",
        n, sharing_radius, avg_penalty
    );
}

/// Calculate the final adjusted fitness incorporating all diversity mechanisms
pub fn calculate_adjusted_fitness(
    raw_fitness: f64,
    candidate_sig: &BehavioralSignature,
    candidate_param_hash: u64,
    user_id: &str,
    registry: &StrategyRegistry,
    config: &DiversityConfig,
) -> (f64, DiversityMetrics) {
    let mut adjusted = raw_fitness;
    let mut metrics = DiversityMetrics::default();
    
    // 1. Crowding penalty from deployed strategies
    if config.enable_crowding_penalty {
        let crowding_penalty = registry.calculate_crowding_penalty(
            candidate_sig,
            candidate_param_hash,
            user_id,
        );
        metrics.crowding_penalty = crowding_penalty;
        adjusted *= 1.0 - (crowding_penalty * config.crowding_weight);
    }
    
    // 2. Behavioral novelty bonus
    if config.enable_behavioral_bonus {
        let similar_count = registry.count_similar(candidate_sig, 0.5);
        if similar_count == 0 {
            // Unique strategy - give bonus
            metrics.novelty_bonus = config.behavioral_bonus_weight;
            adjusted *= 1.0 + config.behavioral_bonus_weight;
        }
    }
    
    // 3. Check minimum distance requirement
    let min_distance = registry.strategies.read()
        .values()
        .filter(|s| s.user_id != user_id)
        .map(|s| candidate_sig.distance(&s.behavioral_sig))
        .fold(f64::INFINITY, f64::min);
    
    metrics.min_distance_to_deployed = if min_distance.is_infinite() { 1.0 } else { min_distance };
    
    if min_distance < config.min_behavioral_distance && !min_distance.is_infinite() {
        // Too similar to deployed strategy - heavy penalty
        metrics.rejected_too_similar = true;
        adjusted *= 0.1; // 90% penalty
    }
    
    metrics.final_fitness = adjusted;
    (adjusted, metrics)
}

/// Metrics for debugging/monitoring diversity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiversityMetrics {
    pub crowding_penalty: f64,
    pub novelty_bonus: f64,
    pub min_distance_to_deployed: f64,
    pub rejected_too_similar: bool,
    pub final_fitness: f64,
}

// NOTE: behavioral-signature/parameter-hash extraction now lives directly on
// `Chromosome::behavioral_signature()` / `Chromosome::parameter_hash()`
// (see genetic/src/lib.rs), with a real override on `DynamicChromosome`.
// The generic stub extractors that used to live here always returned
// `BehavioralSignature::default()` / `0` regardless of the
// concrete chromosome type, silently neutralizing crowding-penalty detection
// for every chromosome type. They were removed in favor of per-type trait
// overrides.

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_behavioral_distance() {
        let sig1 = BehavioralSignature {
            hold_time_bucket: 1,
            frequency_bucket: 2,
            signal_type: SignalType::MarketMaking,
            sizing_style: 0,
            active_hours: 0xFF00, // Europe
            asset_hash: 12345,
            risk_profile: 1,
            reward_risk_bucket: 2,
        };
        
        // Identical
        assert_eq!(sig1.distance(&sig1), 0.0);
        
        // Different signal type (weighted 2x)
        let sig2 = BehavioralSignature {
            signal_type: SignalType::Breakout,
            ..sig1.clone()
        };
        assert!(sig2.distance(&sig1) > 0.2);
        
        // Completely different
        let sig3 = BehavioralSignature {
            hold_time_bucket: 3,
            frequency_bucket: 0,
            signal_type: SignalType::MarketMaking,
            sizing_style: 2,
            active_hours: 0xFF, // Asia only
            asset_hash: 99999,
            risk_profile: 2,
            reward_risk_bucket: 0,
        };
        assert!(sig3.distance(&sig1) > 0.7);
    }
    
    #[test]
    fn test_crowding_penalty() {
        let registry = StrategyRegistry::new();
        
        let sig = BehavioralSignature {
            hold_time_bucket: 1,
            frequency_bucket: 2,
            signal_type: SignalType::MarketMaking,
            sizing_style: 0,
            active_hours: 0xFF00,
            asset_hash: 12345,
            risk_profile: 1,
            reward_risk_bucket: 2,
        };
        
        // Empty registry = no penalty
        assert_eq!(registry.calculate_crowding_penalty(&sig, 12345, "user1"), 0.0);
        
        // Add similar strategy from different user
        registry.register(RegisteredStrategy {
            strategy_id: "strat1".into(),
            user_id: "user2".into(),
            behavioral_sig: sig.clone(),
            parameter_hash: 12345,
            deployed_at: 0,
            aum_usd: 100_000.0,
        });
        
        // Should have penalty now
        let penalty = registry.calculate_crowding_penalty(&sig, 12345, "user1");
        assert!(penalty > 0.3);
        
        // Own strategy = no penalty
        let own_penalty = registry.calculate_crowding_penalty(&sig, 12345, "user2");
        assert_eq!(own_penalty, 0.0);
    }
    
    #[test]
    fn test_fitness_sharing() {
        use crate::{Gene, ParamType, ParameterDef, ParameterSchema};
        use crate::dynamic_chromosome::DynamicChromosome;
        use std::sync::Arc;

        let schema = Arc::new(ParameterSchema {
            params: vec![
                ParameterDef {
                    name: "risk_aversion".into(),
                    param_type: ParamType::Float { min: 0.0001, max: 1.0 },
                    default: Gene::Float(0.3),
                },
                ParameterDef {
                    name: "order_size_pct".into(),
                    param_type: ParamType::Float { min: 0.005, max: 0.25 },
                    default: Gene::Float(0.01),
                },
                ParameterDef {
                    name: "inventory_target".into(),
                    param_type: ParamType::Float { min: 0.0, max: 1.0 },
                    default: Gene::Float(0.0),
                },
            ],
        });

        // Create a deterministic base chromosome with enforced bounds
        let mut base = DynamicChromosome {
            genes: vec![Gene::Float(0.3), Gene::Float(0.01), Gene::Float(0.0)],
            schema: schema.clone(),
        };
        base.enforce_bounds();
        let mut population = vec![base.clone()];

        // Add near-identical copies with tiny perturbations on key distance dimensions.
        // Perturb only one small field so distance stays well under sharing_radius.
        for i in 1..=3 {
            let mut similar = base.clone();
            // Tiny shift in risk_aversion
            if let Gene::Float(v) = &mut similar.genes[0] {
                *v += 0.01 * i as f64;
            }
            similar.enforce_bounds();
            population.push(similar);
        }

        // Add chromosomes that differ significantly on MULTIPLE distance dimensions.
        // Each one is far from base AND far from the others.
        for i in 0..3 {
            let mut diff = base.clone();
            // Large shifts across several dimensions to guarantee distance >> sharing_radius
            if let Gene::Float(v) = &mut diff.genes[0] {
                *v += 2.0 + 1.0 * i as f64;
            }
            if let Gene::Float(v) = &mut diff.genes[1] {
                *v += 0.05 + 0.02 * i as f64;
            }
            if let Gene::Float(v) = &mut diff.genes[2] {
                *v = (*v + 0.3 + 0.1 * i as f64).min(1.0);
            }
            diff.enforce_bounds();
            population.push(diff);
        }
        
        // Sharing radius large enough to capture the similar cluster
        let sharing_radius = 0.5;
        let mut fitnesses: Vec<f64> = (0..population.len()).map(|_| 1.0).collect();
        apply_fitness_sharing(&mut fitnesses, &population, sharing_radius);
        
        // Similar individuals share a niche → lower shared fitness
        // Different individuals are spread out → retain more fitness
        let similar_avg: f64 = fitnesses[0..4].iter().sum::<f64>() / 4.0;
        let different_avg: f64 = fitnesses[4..7].iter().sum::<f64>() / 3.0;
        
        assert!(similar_avg < different_avg, 
            "Similar group avg ({:.4}) should be < different group avg ({:.4}). \
             Fitnesses: {:?}", 
            similar_avg, different_avg, fitnesses);
    }

    /// Regression test (job 64e39acf): a crowded cluster of NEGATIVE-fitness
    /// chromosomes (e.g. zero-trade sentinels) must NOT be boosted toward 0
    /// by fitness sharing. Dividing a negative fitness by niche_count > 1
    /// previously turned -1.0 into ~-0.15, letting no-trade chromosomes
    /// outrank genuinely-trading (but losing) chromosomes.
    #[test]
    fn test_fitness_sharing_never_rewards_negative_fitness() {
        use crate::{Gene, ParamType, ParameterDef, ParameterSchema};
        use crate::dynamic_chromosome::DynamicChromosome;
        use std::sync::Arc;

        let schema = Arc::new(ParameterSchema {
            params: vec![
                ParameterDef {
                    name: "p".into(),
                    param_type: ParamType::Float { min: 0.0, max: 10.0 },
                    default: Gene::Float(1.0),
                },
            ],
        });

        // 6 identical chromosomes (distance 0 → large niche count) with the
        // zero-trade sentinel fitness, plus 1 distant chromosome that trades
        // but loses slightly.
        let clone_chromo = DynamicChromosome {
            genes: vec![Gene::Float(1.0)],
            schema: schema.clone(),
        };
        let mut population = vec![clone_chromo; 6];
        population.push(DynamicChromosome {
            genes: vec![Gene::Float(9.0)],
            schema: schema.clone(),
        });

        let mut fitnesses = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5];
        apply_fitness_sharing(&mut fitnesses, &population, 0.15);

        // Crowded negatives must get WORSE (or stay equal), never improve.
        for f in &fitnesses[0..6] {
            assert!(*f <= -1.0,
                "Crowded negative fitness was boosted by sharing: {:?}", fitnesses);
        }
        // The trading loser must still outrank the no-trade cluster.
        assert!(fitnesses[6] > fitnesses[0],
            "Trading chromosome should outrank crowded no-trade cluster: {:?}", fitnesses);
    }
}
