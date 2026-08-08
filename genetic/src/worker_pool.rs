//! Process-pool parallel GA evaluator for Python strategies.
//!
//! The Python GIL prevents in-process parallelism (rayon threads contend for
//! the GIL with zero speedup).  This module sidesteps the GIL entirely by
//! spawning **N worker child processes**, each with its own Python interpreter.
//!
//! # Protocol (JSON-lines over stdio)
//!
//! 1. **Coordinator → Worker** (one JSON object per line on stdin):
//!    ```json
//!    {"chromosome": {...}, "context": {"generation":0, "total_generations":5, "sample_rate":1}}
//!    ```
//!    A line containing just `"SHUTDOWN"` tells the worker to exit.
//!
//! 2. **Worker → Coordinator** (one JSON object per line on stdout):
//!    ```json
//!    {"Ok": { ...FitnessResult fields... }}
//!    ```
//!    or on error:
//!    ```json
//!    {"Err": "error message"}
//!    ```

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::{Serialize, Deserialize, de::DeserializeOwned};

use crate::{FitnessContext, FitnessResult};
use crate::dynamic_chromosome::DynamicChromosome;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// A single evaluation request sent to a worker process.
#[derive(Serialize, Deserialize)]
pub struct EvalRequest {
    pub chromosome: DynamicChromosome,
    pub context: FitnessContext,
}

/// Worker-side static configuration written to a temp file once.
/// Workers load this at startup so we don't repeat it per chromosome.
#[derive(Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Python strategy source code
    pub python_source: String,
    /// Serialised `BacktestConfig`
    pub backtest_config: serde_json::Value,
    /// Serialised `ExchangeFeeConfig`
    pub fee_config: Option<serde_json::Value>,
    /// Supplementary data (key → value)
    pub supplementary_data: std::collections::HashMap<String, f64>,
    /// Fitness weights
    pub fitness_weights: serde_json::Value,
    /// Initial capital
    pub initial_capital: f64,
    /// Path to the market data binary (bincode-serialised `Vec<MarketData>` or SimBin format)
    pub market_data_path: String,
    /// Path to orderbook snapshots binary (optional)
    pub orderbook_data_path: Option<String>,
    /// If true, market_data_path points to a SimBin file (mmap'd SimulationTick[])
    /// instead of bincode-serialised Vec<MarketData>.
    #[serde(default)]
    pub simbin_format: bool,
    /// Strategy type identifier for Rust-native fast path (e.g., "sma_crossover")
    #[serde(default)]
    pub strategy_type: Option<String>,
    /// Path to bincode-serialized precomputed indicators (optional)
    #[serde(default)]
    pub precomputed_path: Option<String>,
    /// Hard drawdown disqualification threshold (fraction of initial capital,
    /// e.g. 0.40 = 40%), already resolved (default-applied + clamped) by the
    /// caller. Candidates whose max drawdown exceeds this get fitness 0.0
    /// regardless of Sharpe -- mirrors the built-in engine's hard gate.
    /// Defaults to 0.40 for older configs serialized before this field
    /// existed.
    #[serde(default = "default_max_drawdown_hard_cap")]
    pub max_drawdown_hard_cap: f64,
}

fn default_max_drawdown_hard_cap() -> f64 {
    0.40
}

// ---------------------------------------------------------------------------
// Bincode framing helpers (length-prefixed binary protocol)
// ---------------------------------------------------------------------------

/// Write a value as a length-prefixed bincode frame.
pub fn write_bincode_frame<W: Write, T: Serialize>(w: &mut W, val: &T) -> std::io::Result<()> {
    let bytes = bincode::serialize(val).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(&bytes)
}

/// Read a length-prefixed bincode frame.
pub fn read_bincode_frame<R: Read, T: DeserializeOwned>(r: &mut R) -> std::io::Result<T> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    bincode::deserialize(&buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ---------------------------------------------------------------------------
// WorkerPool
// ---------------------------------------------------------------------------

/// Manages N child processes for parallel GA chromosome evaluation.
///
/// Each child is an instance of `backtesting-engine ga-eval-worker` which
/// loads Python + market data once, then evaluates chromosomes streamed over
/// stdin.
pub struct WorkerPool {
    workers: Vec<WorkerHandle>,
    config_path: String,
    _data_path: String,
    use_bincode: bool,
}

struct WorkerHandle {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl WorkerPool {
    /// Spawn `n` worker processes.
    ///
    /// - `exe_path`: path to the `backtesting-engine` binary (usually `/usr/local/bin/backtesting-engine`
    ///   inside Docker, or discovered via `std::env::current_exe()`).
    /// - `config`: the shared configuration (written to a temp file).
    /// - `market_data`: the subsampled GA market data (serialised to a temp file).
    pub fn spawn(
        n: usize,
        config: WorkerConfig,
    ) -> Result<Self, String> {
        Self::spawn_with_protocol(n, config, true)
    }

    /// Spawn workers with explicit protocol choice.
    pub fn spawn_with_protocol(
        n: usize,
        config: WorkerConfig,
        use_bincode: bool,
    ) -> Result<Self, String> {
        let exe = std::env::current_exe()
            .map_err(|e| format!("cannot determine exe path: {e}"))?;

        // Write config to a temp file
        let config_path = std::env::temp_dir().join("ga_worker_config.json");
        let config_json = serde_json::to_string(&config)
            .map_err(|e| format!("serialise config: {e}"))?;
        std::fs::write(&config_path, &config_json)
            .map_err(|e| format!("write config file: {e}"))?;

        let config_path_str = config_path.to_string_lossy().to_string();

        let mut workers = Vec::with_capacity(n);
        for i in 0..n {
            let mut cmd = Command::new(&exe);
            cmd.arg("ga-eval-worker")
                .arg("--config-path")
                .arg(&config_path_str)
                .arg("--worker-id")
                .arg(i.to_string());
            if use_bincode {
                cmd.arg("--bincode");
            }
            let mut child = cmd
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|e| format!("spawn worker {i}: {e}"))?;

            let stdin = child.stdin.take()
                .ok_or_else(|| format!("worker {i}: no stdin"))?;
            let stdout = BufReader::new(
                child.stdout.take()
                    .ok_or_else(|| format!("worker {i}: no stdout"))?,
            );

            workers.push(WorkerHandle { child, stdin, stdout });
        }

        log_info_structured!(crate::GENETIC_LOGGER, "WORKER_POOL_SPAWNED",
            "num_workers" => n,
        );

        Ok(Self {
            workers,
            config_path: config_path_str,
            _data_path: config.market_data_path,
            use_bincode,
        })
    }

    /// Number of worker processes.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Evaluate a batch of chromosomes in parallel across workers.
    ///
    /// Distributes chromosomes round-robin to workers and collects results
    /// in the original order.
    pub fn evaluate_batch(
        &mut self,
        chromosomes: &[DynamicChromosome],
        ctx: &FitnessContext,
        job_tag: &str,
    ) -> Vec<FitnessResult> {
        let n = self.workers.len();
        let total = chromosomes.len();

        // Assign chromosomes to workers round-robin
        let mut assignments: Vec<Vec<(usize, &DynamicChromosome)>> = vec![vec![]; n];
        for (idx, chromo) in chromosomes.iter().enumerate() {
            assignments[idx % n].push((idx, chromo));
        }

        let start = Instant::now();
        let completed = Arc::new(AtomicUsize::new(0));

        // Per-chromosome durations (ms) for percentile reporting at end of batch.
        // Each entry = wall-clock between two consecutive responses on a single
        // worker pipe. Because the worker pipelines requests, this approximates
        // per-chromosome eval time once the pipe is primed (first response on
        // each worker also includes any setup latency).
        let durations: Arc<std::sync::Mutex<Vec<f64>>> =
            Arc::new(std::sync::Mutex::new(Vec::with_capacity(total)));

        // Collect results in original order
        let mut results: Vec<Option<FitnessResult>> = vec![None; total];
        let results_mutex = std::sync::Mutex::new(&mut results);

        // Split workers into individual mutable references for safe parallel access
        let worker_refs: Vec<&mut WorkerHandle> = self.workers.iter_mut().collect();

        // Process each worker's batch sequentially per worker, but workers
        // run in parallel across the OS scheduler
        // Use scoped threads so we can borrow &mut self.workers
        std::thread::scope(|s| {
            let mut handles = Vec::new();

            // Consume worker_refs to give each thread exclusive ownership of one &mut WorkerHandle
            let mut worker_iter = worker_refs.into_iter().enumerate();

            for (worker_idx, batch) in assignments.iter().enumerate() {
                if batch.is_empty() {
                    // Still consume the worker ref to keep indices aligned
                    let _ = worker_iter.next();
                    continue;
                }
                let (_, worker) = worker_iter.next().expect("worker index mismatch");
                let completed = completed.clone();
                let _job_tag = job_tag.to_string();
                let results_mutex = &results_mutex;
                let durations = durations.clone();
                let use_bincode = self.use_bincode;

                handles.push(s.spawn(move || {
                    // --- Batch pipe I/O: send all chromosomes, then read all responses ---
                    // This eliminates per-chromosome flush overhead and lets the worker
                    // pipeline evaluation (next request is already in the pipe buffer).
                    let mut pending: Vec<usize> = Vec::with_capacity(batch.len());

                    // Phase 1: Write all requests without flushing
                    for &(orig_idx, chromo) in batch {
                        let req = EvalRequest {
                            chromosome: chromo.clone(),
                            context: ctx.clone(),
                        };
                        let write_result = if use_bincode {
                            write_bincode_frame(&mut worker.stdin, &req)
                        } else {
                            match serde_json::to_string(&req) {
                                Ok(j) => writeln!(worker.stdin, "{j}"),
                                Err(e) => {
                                    log_error_structured!(crate::GENETIC_LOGGER, "WORKER_SERIALISE_ERROR",
                                        "worker_idx" => worker_idx,
                                        "error" => e,
                                    );
                                    let mut guard = results_mutex.lock().unwrap();
                                    guard[orig_idx] = Some(FitnessResult::failure());
                                    continue;
                                }
                            }
                        };
                        if let Err(e) = write_result {
                            log_error_structured!(crate::GENETIC_LOGGER, "WORKER_STDIN_WRITE_ERROR",
                                "worker_idx" => worker_idx,
                                "error" => e,
                            );
                            let mut guard = results_mutex.lock().unwrap();
                            guard[orig_idx] = Some(FitnessResult::failure());
                            continue;
                        }
                        pending.push(orig_idx);
                    }

                    // Phase 2: Single flush for all queued requests
                    if let Err(e) = worker.stdin.flush() {
                        log_error_structured!(crate::GENETIC_LOGGER, "WORKER_STDIN_FLUSH_ERROR",
                            "worker_idx" => worker_idx,
                            "error" => e,
                        );
                        let mut guard = results_mutex.lock().unwrap();
                        for &orig_idx in &pending {
                            guard[orig_idx] = Some(FitnessResult::failure());
                        }
                        let done = completed.fetch_add(pending.len(), Ordering::Relaxed) + pending.len();
                        if done % 20 == 0 || done == total {
                            let elapsed = start.elapsed().as_secs_f64();
                            log_warn_structured!(crate::GENETIC_LOGGER, "WORKER_BATCH_PROGRESS_FLUSH_FAIL",
                                "done" => done,
                                "total" => total,
                                "rate" => format!("{:.1}", done as f64 / elapsed),
                            );
                        }
                        return;
                    }

                    // Phase 3: Read all responses in order
                    let mut line = String::new();
                    let mut last_response = Instant::now();
                    for &orig_idx in &pending {
                        let result = if use_bincode {
                            match read_bincode_frame::<_, Result<FitnessResult, String>>(&mut worker.stdout) {
                                Ok(Ok(fr)) => fr,
                                Ok(Err(e)) => {
                                    log_error_structured!(crate::GENETIC_LOGGER, "WORKER_EVAL_ERROR",
                                        "worker_idx" => worker_idx,
                                        "error" => e,
                                    );
                                    FitnessResult::failure()
                                }
                                Err(e) => {
                                    log_error_structured!(crate::GENETIC_LOGGER, "WORKER_READ_ERROR",
                                        "worker_idx" => worker_idx,
                                        "error" => e,
                                    );
                                    FitnessResult::failure()
                                }
                            }
                        } else {
                            line.clear();
                            match worker.stdout.read_line(&mut line) {
                                Ok(0) => {
                                    log_error_structured!(crate::GENETIC_LOGGER, "WORKER_EOF",
                                        "worker_idx" => worker_idx,
                                    );
                                    FitnessResult::failure()
                                }
                                Ok(_) => {
                                    let parsed: Result<Result<FitnessResult, String>, _> =
                                        serde_json::from_str(line.trim());
                                    match parsed {
                                        Ok(Ok(fr)) => fr,
                                        Ok(Err(e)) => {
                                            log_error_structured!(crate::GENETIC_LOGGER, "WORKER_EVAL_ERROR",
                                                "worker_idx" => worker_idx,
                                                "error" => e,
                                            );
                                            FitnessResult::failure()
                                        }
                                        Err(e) => {
                                            log_error_structured!(crate::GENETIC_LOGGER, "WORKER_PARSE_ERROR",
                                                "worker_idx" => worker_idx,
                                                "error" => e,
                                            );
                                            FitnessResult::failure()
                                        }
                                    }
                                }
                                Err(e) => {
                                    log_error_structured!(crate::GENETIC_LOGGER, "WORKER_READ_ERROR",
                                        "worker_idx" => worker_idx,
                                        "error" => e,
                                    );
                                    FitnessResult::failure()
                                }
                            }
                        };

                        // Capture per-chromosome duration BEFORE updating shared state
                        // so lock contention isn't counted toward the eval cost.
                        let now = Instant::now();
                        let dur_ms = now.duration_since(last_response).as_secs_f64() * 1000.0;
                        last_response = now;
                        durations.lock().unwrap().push(dur_ms);

                        {
                            let mut guard = results_mutex.lock().unwrap();
                            guard[orig_idx] = Some(result);
                        }

                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 10 == 0 || done == total {
                            let elapsed = start.elapsed().as_secs_f64();
                            let rate = done as f64 / elapsed.max(0.001);
                            log_info_structured!(crate::GENETIC_LOGGER, "WORKER_BATCH_PROGRESS",
                                "done" => done,
                                "total" => total,
                                "elapsed_s" => format!("{:.1}", elapsed),
                                "rate" => format!("{:.2}", rate),
                            );
                        }
                    }
                }));
            }

            // Wait for all worker threads to complete
            for h in handles {
                let _ = h.join();
            }
        });

        // Drop the mutex to release the borrow on results
        drop(results_mutex);

        // Per-chromosome timing summary: p50/p95/p99 + min/max.
        // Helps decide whether to invest in horizontal scaling vs strategy
        // vectorization (Phase 2 vs Phase 3 of the perf plan).
        let mut ds = durations.lock().unwrap().clone();
        if !ds.is_empty() {
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let pct = |p: f64| -> f64 {
                let idx = ((ds.len() as f64 - 1.0) * p).round() as usize;
                ds[idx.min(ds.len() - 1)]
            };
            let mean: f64 = ds.iter().sum::<f64>() / ds.len() as f64;
            let total_ms = start.elapsed().as_secs_f64() * 1000.0;
            log_info_structured!(crate::GENETIC_LOGGER, "BATCH_EVAL_COMPLETED",
                "batch_size" => ds.len(),
                "elapsed_ms" => format!("{:.0}", total_ms),
                "p50_ms" => format!("{:.0}", pct(0.50)),
                "p95_ms" => format!("{:.0}", pct(0.95)),
                "p99_ms" => format!("{:.0}", pct(0.99)),
                "min_ms" => format!("{:.0}", ds.first().copied().unwrap_or(0.0)),
                "max_ms" => format!("{:.0}", ds.last().copied().unwrap_or(0.0)),
                "mean_ms" => format!("{:.0}", mean),
                "workers" => n,
            );
        }

        // Convert Option<FitnessResult> → FitnessResult, filling gaps with failure
        results
            .into_iter()
            .map(|opt| opt.unwrap_or_else(FitnessResult::failure))
            .collect()
    }

    /// Gracefully shut down all workers.
    pub fn shutdown(mut self) {
        for (i, worker) in self.workers.iter_mut().enumerate() {
            // Send shutdown signal
            let _ = writeln!(worker.stdin, "\"SHUTDOWN\"");
            let _ = worker.stdin.flush();

            // Wait for worker to exit (with timeout)
            match worker.child.try_wait() {
                Ok(Some(_)) => {} // already exited
                Ok(None) => {
                    // Give it 2 seconds
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    match worker.child.try_wait() {
                        Ok(Some(_)) => {}
                        _ => {
                            log_warn_structured!(crate::GENETIC_LOGGER, "WORKER_KILLED",
                                "worker_idx" => i,
                            );
                            let _ = worker.child.kill();
                        }
                    }
                }
                Err(_) => {
                    let _ = worker.child.kill();
                }
            }
        }

        // Clean up temp config file
        let _ = std::fs::remove_file(&self.config_path);
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        // Best-effort cleanup: send shutdown + kill
        for worker in self.workers.iter_mut() {
            let _ = writeln!(worker.stdin, "\"SHUTDOWN\"");
            let _ = worker.stdin.flush();
            let _ = worker.child.kill();
        }
    }
}
