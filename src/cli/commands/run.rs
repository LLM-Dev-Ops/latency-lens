//! Run command implementation - Canonical benchmark interface
//!
//! This command provides the canonical run_all_benchmarks() entrypoint
//! required for compatibility across all 25 benchmark-target repositories.

use anyhow::{Context, Result};
use colored::Colorize;
use tracing::info;

use crate::cli::RunArgs;
use crate::config::Config;

// Import from the sibling modules (relative to main.rs)
use crate::benchmarks::{
    self, cleanup_old_results, ensure_output_dirs, get_results_summary,
    read_all_results, run_all_benchmarks, run_benchmarks_for_targets,
    BenchmarkConfig, BenchmarkResult,
};

#[allow(unused_imports)]
use super::read_prompt;

/// Run the run command (canonical benchmark interface)
pub async fn run(args: RunArgs, config: Config, json_output: bool, quiet: bool) -> Result<()> {
    info!("Starting run command (canonical benchmark interface)");

    // Handle cleanup if requested
    if let Some(keep) = args.cleanup {
        let deleted = cleanup_old_results(keep)?;
        if !quiet {
            println!(
                "{} Cleaned up {} old result files (keeping {} per target)",
                "=>".bright_cyan(),
                deleted,
                keep
            );
        }
        return Ok(());
    }

    // Handle summary-only mode
    if args.summary_only {
        let summary = get_results_summary()?;
        if json_output {
            let results = read_all_results()?;
            let json = serde_json::to_string_pretty(&results)?;
            println!("{}", json);
        } else {
            println!("{}", summary);
        }
        return Ok(());
    }

    // Build benchmark configuration
    let prompt = if let Some(ref p) = args.prompt {
        p.clone()
    } else if let Some(ref path) = args.prompt_file {
        std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read prompt file: {}", path.display()))?
    } else {
        "Explain the concept of machine learning in simple terms.".to_string()
    };

    let bench_config = BenchmarkConfig {
        requests: args.requests,
        concurrency: args.concurrency,
        rate_limit: args.rate_limit,
        warmup: args.warmup,
        max_tokens: args.max_tokens,
        temperature: args.temperature.unwrap_or(0.7),
        timeout_secs: args.timeout,
        show_progress: args.progress && !quiet && !json_output,
        default_prompt: prompt,
    };

    // Ensure output directories exist
    ensure_output_dirs()?;

    if !quiet {
        println!(
            "{}",
            "LLM Latency Lens - Canonical Benchmark Interface"
                .bright_cyan()
                .bold()
        );
        println!();
    }

    // Run benchmarks
    let results: Vec<BenchmarkResult> = if args.targets.is_empty() {
        if !quiet {
            println!(
                "{} Running benchmarks for all configured targets...",
                "=>".bright_cyan()
            );
            println!();
        }
        run_all_benchmarks(config, bench_config).await?
    } else {
        if !quiet {
            println!(
                "{} Running benchmarks for {} specific targets...",
                "=>".bright_cyan(),
                args.targets.len()
            );
            println!();
        }
        run_benchmarks_for_targets(config, bench_config, &args.targets).await?
    };

    // Output results
    if json_output {
        let json = serde_json::to_string_pretty(&results)?;
        println!("{}", json);
    } else if !quiet {
        // Print summary table
        println!();
        println!(
            "{}",
            "Benchmark Results Summary"
                .bright_cyan()
                .bold()
                .underline()
        );
        println!();

        println!(
            "{:40} {:10} {:12} {:15} {:15}",
            "Target".bright_white().bold(),
            "Status".bright_white().bold(),
            "Success %".bright_white().bold(),
            "TTFT (mean)".bright_white().bold(),
            "Throughput".bright_white().bold()
        );
        println!("{}", "-".repeat(92));

        for result in &results {
            let status = if result.is_success() {
                "PASS".bright_green()
            } else {
                "FAIL".bright_red()
            };

            let success_rate = result
                .success_rate()
                .map(|r| format!("{:.1}%", r))
                .unwrap_or_else(|| "N/A".to_string());

            let ttft = result
                .get_metric("ttft_distribution.mean")
                .and_then(|v| v.as_u64())
                .map(|ns| format!("{:.2}ms", ns as f64 / 1_000_000.0))
                .unwrap_or_else(|| "N/A".to_string());

            let throughput = result
                .get_metric("throughput.mean_tokens_per_second")
                .and_then(|v| v.as_f64())
                .map(|t| format!("{:.1} tok/s", t))
                .unwrap_or_else(|| "N/A".to_string());

            println!(
                "{:40} {:10} {:12} {:15} {:15}",
                result.target_id(),
                status,
                success_rate,
                ttft,
                throughput
            );
        }

        println!();

        // Overall summary
        let total = results.len();
        let passed = results.iter().filter(|r| r.is_success()).count();
        let failed = total - passed;

        println!(
            "{} Total: {} | {} | {}",
            "=>".bright_cyan(),
            format!("{} targets", total).bright_white(),
            format!("{} passed", passed).bright_green(),
            format!("{} failed", failed).bright_red()
        );

        // Print output location
        println!();
        println!(
            "{} Results written to: {}",
            "=>".bright_cyan(),
            benchmarks::output_dir().display()
        );
        println!(
            "{} Summary available at: {}",
            "=>".bright_cyan(),
            benchmarks::summary_path().display()
        );

        println!();
        println!("{} Benchmark run complete!", "✓".bright_green().bold());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_from_args() {
        // Basic test to ensure config can be built
        let config = BenchmarkConfig::default();
        assert_eq!(config.requests, 10);
        assert_eq!(config.concurrency, 1);
    }
}
