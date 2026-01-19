//! Compare command implementation

use anyhow::{Context, Result};
use colored::Colorize;
use std::sync::Arc;
use tabled::{Table, Tabled};
use tracing::info;

use crate::cli::CompareArgs;
use crate::config::Config;
use crate::orchestrator::{Orchestrator, OrchestratorConfig};
use llm_latency_lens_metrics::{AggregatedMetrics, MetricsAggregator, MetricsCollector};
use llm_latency_lens_providers::{create_provider, MessageRole, StreamingRequest};

use super::{read_prompt, write_output};

/// Run the compare command
pub async fn run(
    args: CompareArgs,
    mut config: Config,
    json_output: bool,
    quiet: bool,
    shutdown_signal: Arc<tokio::sync::Notify>,
) -> Result<()> {
    info!("Starting compare command");

    // Validate configuration
    config.validate().with_context(|| "Configuration validation failed")?;

    // Parse targets (format: provider:model)
    let targets: Result<Vec<(String, String)>> = args
        .targets
        .iter()
        .map(|t| {
            let parts: Vec<&str> = t.split(':').collect();
            if parts.len() != 2 {
                anyhow::bail!(
                    "Invalid target format '{}'. Expected 'provider:model' (e.g., 'openai:gpt-4o')",
                    t
                );
            }
            Ok((parts[0].to_string(), parts[1].to_string()))
        })
        .collect();

    let targets = targets?;

    // Read prompt
    let prompt = read_prompt(&args.prompt, &args.prompt_file)
        .context("Failed to read prompt")?;

    if !quiet {
        println!(
            "{} Comparing {} configurations",
            "=>".bright_cyan().bold(),
            targets.len().to_string().bright_white().bold()
        );
        for (provider, model) in &targets {
            println!(
                "   - {} ({})",
                model.bright_green(),
                provider.bright_yellow()
            );
        }
        println!("   {} requests per configuration", args.requests);
        println!();
    }

    // Results for each target
    let mut results: Vec<(String, String, AggregatedMetrics)> = Vec::new();

    // Run benchmarks for each target
    for (provider_name, model) in &targets {
        if !quiet {
            println!(
                "{} Benchmarking {} - {}...",
                "=>".bright_cyan(),
                provider_name.bright_yellow(),
                model.bright_green()
            );
        }

        // Get provider configuration
        let provider_config = config.get_provider(provider_name)
            .with_context(|| format!("Provider '{}' not configured", provider_name))?;

        let api_key = provider_config
            .api_key
            .as_ref()
            .context("API key not found for provider")?;

        // Create provider
        let provider = Arc::new(
            create_provider(provider_name, api_key.clone())
                .with_context(|| format!("Failed to create provider: {}", provider_name))?
        );

        // Build request template
        let request_template = StreamingRequest::builder()
            .model(model.clone())
            .message(MessageRole::User, prompt.clone())
            .max_tokens(args.max_tokens)
            .temperature(args.temperature.unwrap_or(0.7))
            .top_p(args.top_p.unwrap_or(1.0))
            .timeout_secs(args.timeout)
            .build();

        // Create orchestrator
        let orchestrator_config = OrchestratorConfig {
            concurrency: 1, // Sequential for fair comparison
            total_requests: args.requests,
            rate_limit: 0,
            show_progress: !quiet && !json_output,
            shutdown_timeout: std::time::Duration::from_secs(30),
        };

        let orchestrator = Orchestrator::new(orchestrator_config, Arc::clone(&shutdown_signal));
        let session_id = orchestrator.session_id();

        // Create metrics collector
        let collector = Arc::new(
            MetricsCollector::with_defaults(session_id)
                .context("Failed to create metrics collector")?
        );

        // Execute benchmark
        let _summary = orchestrator
            .execute(provider, request_template, Arc::clone(&collector))
            .await?;

        // Aggregate metrics
        let aggregated = MetricsAggregator::aggregate(&collector)
            .context("Failed to aggregate metrics")?;

        results.push((provider_name.clone(), model.clone(), aggregated));

        if !quiet {
            println!("{} Complete\n", "✓".bright_green());
        }
    }

    // Output comparison
    if json_output {
        let json_data: Vec<_> = results
            .iter()
            .map(|(provider, model, metrics)| {
                serde_json::json!({
                    "provider": provider,
                    "model": model,
                    "metrics": {
                        "ttft": {
                            "mean_ms": metrics.ttft_distribution.mean.as_secs_f64() * 1000.0,
                            "p50_ms": metrics.ttft_distribution.p50.as_secs_f64() * 1000.0,
                            "p95_ms": metrics.ttft_distribution.p95.as_secs_f64() * 1000.0,
                            "p99_ms": metrics.ttft_distribution.p99.as_secs_f64() * 1000.0,
                        },
                        "total_latency": {
                            "mean_ms": metrics.total_latency_distribution.mean.as_secs_f64() * 1000.0,
                            "p50_ms": metrics.total_latency_distribution.p50.as_secs_f64() * 1000.0,
                            "p95_ms": metrics.total_latency_distribution.p95.as_secs_f64() * 1000.0,
                        },
                        "throughput": {
                            "mean_tokens_per_sec": metrics.throughput.mean_tokens_per_second,
                            "p50_tokens_per_sec": metrics.throughput.p50_tokens_per_second,
                        },
                        "cost_usd": metrics.total_cost_usd,
                        "success_rate": metrics.success_rate(),
                    }
                })
            })
            .collect();

        let output = if quiet {
            serde_json::to_string(&json_data)?
        } else {
            serde_json::to_string_pretty(&json_data)?
        };

        write_output(&output, &args.output)?;
    } else {
        if !quiet {
            println!();
            println!("{}", "Comparison Results".bright_cyan().bold().underline());
            println!();

            // TTFT Comparison
            if args.metrics.contains(&"ttft".to_string()) {
                println!("{}", "Time to First Token (TTFT)".bright_white().bold());
                println!();

                #[derive(Tabled)]
                struct TtftRow {
                    #[tabled(rename = "Provider:Model")]
                    target: String,
                    #[tabled(rename = "Mean (ms)")]
                    mean: String,
                    #[tabled(rename = "P50 (ms)")]
                    p50: String,
                    #[tabled(rename = "P95 (ms)")]
                    p95: String,
                    #[tabled(rename = "P99 (ms)")]
                    p99: String,
                }

                let ttft_rows: Vec<_> = results
                    .iter()
                    .map(|(provider, model, metrics)| TtftRow {
                        target: format!("{}:{}", provider, model),
                        mean: format!("{:.2}", metrics.ttft_distribution.mean.as_secs_f64() * 1000.0),
                        p50: format!("{:.2}", metrics.ttft_distribution.p50.as_secs_f64() * 1000.0),
                        p95: format!("{:.2}", metrics.ttft_distribution.p95.as_secs_f64() * 1000.0),
                        p99: format!("{:.2}", metrics.ttft_distribution.p99.as_secs_f64() * 1000.0),
                    })
                    .collect();

                println!("{}", Table::new(ttft_rows));
                println!();
            }

            // Total Latency Comparison
            if args.metrics.contains(&"total".to_string()) {
                println!("{}", "Total Latency".bright_white().bold());
                println!();

                #[derive(Tabled)]
                struct LatencyRow {
                    #[tabled(rename = "Provider:Model")]
                    target: String,
                    #[tabled(rename = "Mean (ms)")]
                    mean: String,
                    #[tabled(rename = "P50 (ms)")]
                    p50: String,
                    #[tabled(rename = "P95 (ms)")]
                    p95: String,
                }

                let latency_rows: Vec<_> = results
                    .iter()
                    .map(|(provider, model, metrics)| LatencyRow {
                        target: format!("{}:{}", provider, model),
                        mean: format!("{:.2}", metrics.total_latency_distribution.mean.as_secs_f64() * 1000.0),
                        p50: format!("{:.2}", metrics.total_latency_distribution.p50.as_secs_f64() * 1000.0),
                        p95: format!("{:.2}", metrics.total_latency_distribution.p95.as_secs_f64() * 1000.0),
                    })
                    .collect();

                println!("{}", Table::new(latency_rows));
                println!();
            }

            // Throughput Comparison
            if args.metrics.contains(&"throughput".to_string()) {
                println!("{}", "Throughput (tokens/sec)".bright_white().bold());
                println!();

                #[derive(Tabled)]
                struct ThroughputRow {
                    #[tabled(rename = "Provider:Model")]
                    target: String,
                    #[tabled(rename = "Mean")]
                    mean: String,
                    #[tabled(rename = "P50")]
                    p50: String,
                    #[tabled(rename = "P95")]
                    p95: String,
                }

                let throughput_rows: Vec<_> = results
                    .iter()
                    .map(|(provider, model, metrics)| ThroughputRow {
                        target: format!("{}:{}", provider, model),
                        mean: format!("{:.2}", metrics.throughput.mean_tokens_per_second),
                        p50: format!("{:.2}", metrics.throughput.p50_tokens_per_second),
                        p95: format!("{:.2}", metrics.throughput.p95_tokens_per_second),
                    })
                    .collect();

                println!("{}", Table::new(throughput_rows));
                println!();
            }

            // Cost Comparison
            if args.metrics.contains(&"cost".to_string()) {
                println!("{}", "Cost".bright_white().bold());
                println!();

                #[derive(Tabled)]
                struct CostRow {
                    #[tabled(rename = "Provider:Model")]
                    target: String,
                    #[tabled(rename = "Total Cost")]
                    total: String,
                    #[tabled(rename = "Per Request")]
                    per_request: String,
                }

                let cost_rows: Vec<_> = results
                    .iter()
                    .map(|(provider, model, metrics)| CostRow {
                        target: format!("{}:{}", provider, model),
                        total: metrics
                            .total_cost_usd
                            .map(|c| format!("${:.6}", c))
                            .unwrap_or_else(|| "N/A".to_string()),
                        per_request: metrics
                            .avg_cost_per_request()
                            .map(|c| format!("${:.6}", c))
                            .unwrap_or_else(|| "N/A".to_string()),
                    })
                    .collect();

                println!("{}", Table::new(cost_rows));
                println!();
            }

            // Winner analysis
            let fastest_ttft = results
                .iter()
                .min_by(|(_, _, a), (_, _, b)| {
                    a.ttft_distribution.mean.cmp(&b.ttft_distribution.mean)
                });

            let highest_throughput = results
                .iter()
                .max_by(|(_, _, a), (_, _, b)| {
                    a.throughput.mean_tokens_per_second
                        .partial_cmp(&b.throughput.mean_tokens_per_second)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some((provider, model, _)) = fastest_ttft {
                println!(
                    "{} Fastest TTFT: {} ({})",
                    "🏆".bright_yellow(),
                    model.bright_green().bold(),
                    provider.bright_yellow()
                );
            }

            if let Some((provider, model, _)) = highest_throughput {
                println!(
                    "{} Highest throughput: {} ({})",
                    "🏆".bright_yellow(),
                    model.bright_green().bold(),
                    provider.bright_yellow()
                );
            }

            println!();
            println!("{} Comparison complete!", "✓".bright_green().bold());
        }

        // Save to file if requested
        if let Some(ref output_path) = args.output {
            let json_data: Vec<_> = results
                .iter()
                .map(|(provider, model, metrics)| {
                    serde_json::json!({
                        "provider": provider,
                        "model": model,
                        "metrics": metrics,
                    })
                })
                .collect();

            std::fs::write(output_path, serde_json::to_string_pretty(&json_data)?)?;

            if !quiet {
                println!("Results saved to: {}", output_path.display());
            }
        }
    }

    Ok(())
}
