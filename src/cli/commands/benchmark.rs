//! Benchmark command implementation

use anyhow::{Context, Result};
use colored::Colorize;
use std::sync::Arc;
use tabled::{Table, Tabled};
use tracing::info;

use crate::cli::BenchmarkArgs;
use crate::config::Config;
use crate::orchestrator::{Orchestrator, OrchestratorConfig};
use llm_latency_lens_exporters::{Exporter, JsonExporter};
use llm_latency_lens_metrics::{MetricsAggregator, MetricsCollector};
use llm_latency_lens_providers::{create_provider, MessageRole, StreamingRequest};

use super::{read_prompt, write_output};

/// Run the benchmark command
pub async fn run(
    args: BenchmarkArgs,
    mut config: Config,
    json_output: bool,
    quiet: bool,
    shutdown_signal: Arc<tokio::sync::Notify>,
) -> Result<()> {
    info!("Starting benchmark command");

    // Merge CLI overrides
    config.merge_cli_overrides(&args.provider, args.api_key.clone(), args.endpoint.clone());

    // Validate configuration
    config.validate().with_context(|| "Configuration validation failed")?;

    // Get provider configuration
    let provider_config = config.get_provider(&args.provider)?;

    let api_key = provider_config
        .api_key
        .as_ref()
        .context("API key not found for provider")?;

    // Create provider
    let provider = Arc::new(
        create_provider(&args.provider, api_key.clone())
            .with_context(|| format!("Failed to create provider: {}", args.provider))?
    );

    // Read prompt
    let prompt = read_prompt(&args.prompt, &args.prompt_file)
        .context("Failed to read prompt")?;

    if !quiet {
        println!(
            "{} Benchmarking {} with model {}",
            "=>".bright_cyan().bold(),
            args.provider.bright_yellow(),
            args.model.bright_green()
        );
        println!(
            "   {} requests with concurrency {}",
            args.requests.to_string().bright_white().bold(),
            args.concurrency.to_string().bright_white().bold()
        );
        if args.rate_limit > 0 {
            println!("   Rate limit: {} req/s", args.rate_limit);
        }
        println!();
    }

    // Build request template
    let request_template = StreamingRequest::builder()
        .model(args.model.clone())
        .message(MessageRole::User, prompt)
        .max_tokens(args.max_tokens)
        .temperature(args.temperature.unwrap_or(0.7))
        .top_p(args.top_p.unwrap_or(1.0))
        .timeout_secs(args.timeout)
        .build();

    // Create orchestrator
    let orchestrator_config = OrchestratorConfig {
        concurrency: args.concurrency,
        total_requests: args.requests,
        rate_limit: args.rate_limit,
        show_progress: args.progress && !quiet && !json_output,
        shutdown_timeout: std::time::Duration::from_secs(30),
    };

    let orchestrator = Orchestrator::new(orchestrator_config, shutdown_signal);
    let session_id = orchestrator.session_id();

    // Create metrics collector
    let collector = Arc::new(
        MetricsCollector::with_defaults(session_id)
            .context("Failed to create metrics collector")?
    );

    // Run warmup if requested
    if args.warmup > 0 && !quiet {
        println!(
            "{} Running {} warmup requests...",
            "=>".bright_cyan(),
            args.warmup
        );

        let warmup_config = OrchestratorConfig {
            concurrency: args.concurrency,
            total_requests: args.warmup,
            rate_limit: args.rate_limit,
            show_progress: false,
            shutdown_timeout: std::time::Duration::from_secs(30),
        };

        let warmup_orchestrator = Orchestrator::new(
            warmup_config,
            orchestrator.shutdown_signal(),
        );

        let warmup_collector = Arc::new(
            MetricsCollector::with_defaults(warmup_orchestrator.session_id())?
        );

        let _ = warmup_orchestrator
            .execute(
                Arc::clone(&provider),
                request_template.clone(),
                warmup_collector,
            )
            .await?;

        println!("{} Warmup complete\n", "✓".bright_green());
    }

    // Execute benchmark
    let summary = orchestrator
        .execute(provider, request_template, Arc::clone(&collector))
        .await?;

    // Aggregate metrics
    let aggregated = MetricsAggregator::aggregate(&collector)
        .context("Failed to aggregate metrics")?;

    // Output results
    if json_output {
        let json_exporter = JsonExporter::new(!quiet);
        let output = json_exporter.export(&aggregated)?;
        write_output(&output, &args.output)?;
    } else {
        // Print summary
        if !quiet {
            println!();
            println!("{}", "Benchmark Summary".bright_cyan().bold().underline());
            println!();

            #[derive(Tabled)]
            struct SummaryRow {
                #[tabled(rename = "Metric")]
                metric: String,
                #[tabled(rename = "Value")]
                value: String,
            }

            let rows = vec![
                SummaryRow {
                    metric: "Total Requests".to_string(),
                    value: summary.total_requests.to_string(),
                },
                SummaryRow {
                    metric: "Successful".to_string(),
                    value: format!("{} ({:.1}%)", summary.successful_requests, summary.success_rate()),
                },
                SummaryRow {
                    metric: "Failed".to_string(),
                    value: summary.failed_requests.to_string(),
                },
                SummaryRow {
                    metric: "Duration".to_string(),
                    value: format!("{:.2}s", summary.total_duration.as_secs_f64()),
                },
                SummaryRow {
                    metric: "Requests/sec".to_string(),
                    value: format!("{:.2}", summary.requests_per_second),
                },
            ];

            println!("{}", Table::new(rows));
            println!();

            // TTFT metrics
            println!("{}", "Time to First Token (TTFT)".bright_cyan().bold().underline());
            println!();

            #[derive(Tabled)]
            struct LatencyRow {
                #[tabled(rename = "Metric")]
                metric: String,
                #[tabled(rename = "Value")]
                value: String,
            }

            let ttft_rows = vec![
                LatencyRow {
                    metric: "Min".to_string(),
                    value: format!("{:.2}ms", aggregated.ttft_distribution.min.as_secs_f64() * 1000.0),
                },
                LatencyRow {
                    metric: "Mean".to_string(),
                    value: format!("{:.2}ms", aggregated.ttft_distribution.mean.as_secs_f64() * 1000.0),
                },
                LatencyRow {
                    metric: "P50 (Median)".to_string(),
                    value: format!("{:.2}ms", aggregated.ttft_distribution.p50.as_secs_f64() * 1000.0),
                },
                LatencyRow {
                    metric: "P90".to_string(),
                    value: format!("{:.2}ms", aggregated.ttft_distribution.p90.as_secs_f64() * 1000.0),
                },
                LatencyRow {
                    metric: "P95".to_string(),
                    value: format!("{:.2}ms", aggregated.ttft_distribution.p95.as_secs_f64() * 1000.0),
                },
                LatencyRow {
                    metric: "P99".to_string(),
                    value: format!("{:.2}ms", aggregated.ttft_distribution.p99.as_secs_f64() * 1000.0),
                },
                LatencyRow {
                    metric: "Max".to_string(),
                    value: format!("{:.2}ms", aggregated.ttft_distribution.max.as_secs_f64() * 1000.0),
                },
            ];

            println!("{}", Table::new(ttft_rows));
            println!();

            // Throughput
            println!("{}", "Throughput (tokens/sec)".bright_cyan().bold().underline());
            println!();

            let throughput_rows = vec![
                LatencyRow {
                    metric: "Mean".to_string(),
                    value: format!("{:.2}", aggregated.throughput.mean_tokens_per_second),
                },
                LatencyRow {
                    metric: "Min".to_string(),
                    value: format!("{:.2}", aggregated.throughput.min_tokens_per_second),
                },
                LatencyRow {
                    metric: "Max".to_string(),
                    value: format!("{:.2}", aggregated.throughput.max_tokens_per_second),
                },
                LatencyRow {
                    metric: "P50".to_string(),
                    value: format!("{:.2}", aggregated.throughput.p50_tokens_per_second),
                },
                LatencyRow {
                    metric: "P95".to_string(),
                    value: format!("{:.2}", aggregated.throughput.p95_tokens_per_second),
                },
            ];

            println!("{}", Table::new(throughput_rows));
            println!();

            // Cost summary
            if let Some(total_cost) = aggregated.total_cost_usd {
                println!(
                    "{} Total cost: {} (avg: {} per request)",
                    "=>".bright_cyan(),
                    format!("${:.6}", total_cost).bright_green().bold(),
                    format!("${:.6}", aggregated.avg_cost_per_request().unwrap_or(0.0)).bright_white()
                );
            }

            println!();
            println!("{} Benchmark complete!", "✓".bright_green().bold());
        }

        // Save to file if requested
        if let Some(ref output_path) = args.output {
            let json_exporter = JsonExporter::new(true);
            let output = json_exporter.export(&aggregated)?;
            std::fs::write(output_path, output)?;

            if !quiet {
                println!("Results saved to: {}", output_path.display());
            }
        }
    }

    Ok(())
}
