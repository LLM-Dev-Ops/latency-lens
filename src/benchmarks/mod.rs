//! Canonical benchmarks module for LLM Latency Lens
//!
//! This module provides the standardized benchmark interface required
//! across all 25 benchmark-target repositories, including:
//! - `run_all_benchmarks()` entrypoint returning `Vec<BenchmarkResult>`
//! - Standardized `BenchmarkResult` struct
//! - Markdown summary generation
//! - File I/O utilities for benchmark output

pub mod io;
pub mod markdown;
pub mod result;

pub use io::{
    cleanup_old_results, ensure_output_dirs, output_dir, raw_output_dir, read_all_results,
    read_result, read_summary, summary_path, write_result, write_results, write_summary,
    DEFAULT_OUTPUT_DIR, RAW_OUTPUT_DIR, SUMMARY_FILENAME,
};
pub use markdown::{generate_summary, one_line_summary};
pub use result::BenchmarkResult;

use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::info;

use super::adapters::{all_targets, BenchTarget};
use super::config::Config;

/// Configuration for running all benchmarks
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of requests per target
    pub requests: u32,
    /// Concurrency level
    pub concurrency: u32,
    /// Rate limit (requests per second, 0 = unlimited)
    pub rate_limit: u32,
    /// Warmup requests before actual benchmark
    pub warmup: u32,
    /// Maximum tokens to generate per request
    pub max_tokens: u32,
    /// Temperature for generation
    pub temperature: f32,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Show progress bars
    pub show_progress: bool,
    /// Default prompt to use for benchmarks
    pub default_prompt: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            requests: 10,
            concurrency: 1,
            rate_limit: 0,
            warmup: 0,
            max_tokens: 1024,
            temperature: 0.7,
            timeout_secs: 120,
            show_progress: true,
            default_prompt: "Explain the concept of machine learning in simple terms.".to_string(),
        }
    }
}

/// Run all benchmarks and return standardized results
///
/// This is the canonical entrypoint required for compatibility with
/// the benchmark-target interface across all 25 repositories.
///
/// # Arguments
///
/// * `config` - Application configuration with provider credentials
/// * `bench_config` - Benchmark-specific configuration
///
/// # Returns
///
/// A vector of `BenchmarkResult` containing metrics for each target
pub async fn run_all_benchmarks(
    config: Config,
    bench_config: BenchmarkConfig,
) -> Result<Vec<BenchmarkResult>> {
    info!("Starting run_all_benchmarks with {} targets", all_targets(&config).len());

    // Ensure output directories exist
    ensure_output_dirs()?;

    let shutdown_signal = Arc::new(tokio::sync::Notify::new());
    let mut results = Vec::new();

    // Get all configured benchmark targets
    let targets = all_targets(&config);

    if targets.is_empty() {
        tracing::warn!("No benchmark targets configured");
        return Ok(results);
    }

    info!("Running benchmarks for {} targets", targets.len());

    for target in targets {
        let target_id = target.id();
        info!("Benchmarking target: {}", target_id);

        match target
            .run(
                &config,
                &bench_config,
                Arc::clone(&shutdown_signal),
            )
            .await
        {
            Ok(result) => {
                // Write result to file
                if let Err(e) = write_result(&result) {
                    tracing::warn!("Failed to write result for {}: {}", target_id, e);
                }

                info!(
                    "Completed benchmark for {}: {}",
                    target_id,
                    one_line_summary(&result)
                );
                results.push(result);
            }
            Err(e) => {
                tracing::error!("Benchmark failed for {}: {}", target_id, e);

                // Create a failure result
                let failure_result = BenchmarkResult::new(
                    target_id.clone(),
                    serde_json::json!({
                        "error": e.to_string(),
                        "total_requests": 0,
                        "successful_requests": 0,
                        "failed_requests": 0
                    }),
                );

                if let Err(we) = write_result(&failure_result) {
                    tracing::warn!("Failed to write failure result for {}: {}", target_id, we);
                }

                results.push(failure_result);
            }
        }
    }

    // Generate and write summary
    let summary = generate_summary(&results);
    write_summary(&summary).context("Failed to write summary")?;

    info!(
        "Completed all benchmarks: {}/{} successful",
        results.iter().filter(|r| r.is_success()).count(),
        results.len()
    );

    Ok(results)
}

/// Run benchmarks for specific targets only
pub async fn run_benchmarks_for_targets(
    config: Config,
    bench_config: BenchmarkConfig,
    target_ids: &[String],
) -> Result<Vec<BenchmarkResult>> {
    info!("Running benchmarks for {} specific targets", target_ids.len());

    ensure_output_dirs()?;

    let shutdown_signal = Arc::new(tokio::sync::Notify::new());
    let mut results = Vec::new();

    let all = all_targets(&config);

    for target in all {
        if target_ids.contains(&target.id()) {
            let target_id = target.id();
            info!("Benchmarking target: {}", target_id);

            match target
                .run(&config, &bench_config, Arc::clone(&shutdown_signal))
                .await
            {
                Ok(result) => {
                    if let Err(e) = write_result(&result) {
                        tracing::warn!("Failed to write result for {}: {}", target_id, e);
                    }
                    results.push(result);
                }
                Err(e) => {
                    tracing::error!("Benchmark failed for {}: {}", target_id, e);
                    let failure_result = BenchmarkResult::new(
                        target_id.clone(),
                        serde_json::json!({
                            "error": e.to_string(),
                            "total_requests": 0,
                            "successful_requests": 0,
                            "failed_requests": 0
                        }),
                    );
                    results.push(failure_result);
                }
            }
        }
    }

    // Generate and write summary
    let summary = generate_summary(&results);
    write_summary(&summary)?;

    Ok(results)
}

/// Get a summary of all available benchmark results
pub fn get_results_summary() -> Result<String> {
    let results = read_all_results()?;
    Ok(generate_summary(&results))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();

        assert_eq!(config.requests, 10);
        assert_eq!(config.concurrency, 1);
        assert_eq!(config.rate_limit, 0);
        assert_eq!(config.warmup, 0);
        assert_eq!(config.max_tokens, 1024);
        assert!(config.show_progress);
    }

    #[test]
    fn test_ensure_output_dirs() {
        // This test would need a temp directory setup
        // Skipping actual filesystem operations in unit tests
    }
}
