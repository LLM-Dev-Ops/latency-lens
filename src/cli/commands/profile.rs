//! Profile command implementation

use anyhow::{Context, Result};
use colored::Colorize;
use std::sync::Arc;
use tabled::{Table, Tabled};
use tracing::info;

use crate::cli::ProfileArgs;
use crate::config::Config;
use llm_latency_lens_core::TimingEngine;
use llm_latency_lens_exporters::{Exporter, JsonExporter};
use llm_latency_lens_providers::{create_provider, MessageRole, StreamingRequest};

use super::{read_prompt, write_output};

/// Run the profile command
pub async fn run(
    args: ProfileArgs,
    mut config: Config,
    json_output: bool,
    quiet: bool,
    _shutdown_signal: Arc<tokio::sync::Notify>,
) -> Result<()> {
    info!("Starting profile command");

    // Merge CLI overrides into config
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
    let provider = create_provider(&args.provider, api_key.clone())
        .with_context(|| format!("Failed to create provider: {}", args.provider))?;

    // Read prompt
    let prompt = read_prompt(&args.prompt, &args.prompt_file)
        .context("Failed to read prompt")?;

    if !quiet {
        println!(
            "{} Profiling {} with model {}...",
            "=>".bright_cyan().bold(),
            args.provider.bright_yellow(),
            args.model.bright_green()
        );
    }

    // Build request
    let request = StreamingRequest::builder()
        .model(args.model.clone())
        .message(MessageRole::User, prompt)
        .max_tokens(args.max_tokens)
        .temperature(args.temperature.unwrap_or(0.7))
        .top_p(args.top_p.unwrap_or(1.0))
        .timeout_secs(args.timeout)
        .build();

    // Create timing engine
    let timing_engine = TimingEngine::new();

    // Execute request
    let start = std::time::Instant::now();
    let result = provider.complete(request, &timing_engine).await
        .context("Request failed")?;

    let duration = start.elapsed();

    // Calculate metrics
    let ttft = result.ttft().unwrap_or_default();
    let avg_inter_token = result.avg_inter_token_latency().unwrap_or_default();
    let median_inter_token = result.median_inter_token_latency().unwrap_or_default();
    let p95_inter_token = result.p95_inter_token_latency().unwrap_or_default();
    let tokens_per_second = result.tokens_per_second().unwrap_or(0.0);

    // Prepare output
    if json_output {
        let json_data = serde_json::json!({
            "provider": args.provider,
            "model": args.model,
            "request_id": result.request_id.to_string(),
            "ttft_ms": ttft.as_millis(),
            "total_duration_ms": duration.as_millis(),
            "token_count": result.token_events.len(),
            "input_tokens": result.metadata.input_tokens,
            "output_tokens": result.metadata.output_tokens,
            "thinking_tokens": result.metadata.thinking_tokens,
            "avg_inter_token_latency_ms": avg_inter_token.as_millis(),
            "median_inter_token_latency_ms": median_inter_token.as_millis(),
            "p95_inter_token_latency_ms": p95_inter_token.as_millis(),
            "tokens_per_second": tokens_per_second,
            "cost_usd": result.metadata.estimated_cost,
            "content": result.content,
        });

        let output = if quiet {
            serde_json::to_string(&json_data)?
        } else {
            serde_json::to_string_pretty(&json_data)?
        };

        write_output(&output, &args.output)?;
    } else {
        // Print results in a beautiful table
        if !quiet {
            println!("\n{}", "Results".bright_cyan().bold().underline());
            println!();

            #[derive(Tabled)]
            struct MetricRow {
                #[tabled(rename = "Metric")]
                metric: String,
                #[tabled(rename = "Value")]
                value: String,
            }

            let rows = vec![
                MetricRow {
                    metric: "Provider".to_string(),
                    value: args.provider.clone(),
                },
                MetricRow {
                    metric: "Model".to_string(),
                    value: args.model.clone(),
                },
                MetricRow {
                    metric: "Request ID".to_string(),
                    value: result.request_id.to_string(),
                },
                MetricRow {
                    metric: "Time to First Token (TTFT)".to_string(),
                    value: format!("{:.2}ms", ttft.as_secs_f64() * 1000.0),
                },
                MetricRow {
                    metric: "Total Duration".to_string(),
                    value: format!("{:.2}ms", duration.as_secs_f64() * 1000.0),
                },
                MetricRow {
                    metric: "Token Count".to_string(),
                    value: result.token_events.len().to_string(),
                },
                MetricRow {
                    metric: "Input Tokens".to_string(),
                    value: result.metadata.input_tokens.map(|t| t.to_string()).unwrap_or_else(|| "N/A".to_string()),
                },
                MetricRow {
                    metric: "Output Tokens".to_string(),
                    value: result.metadata.output_tokens.map(|t| t.to_string()).unwrap_or_else(|| "N/A".to_string()),
                },
                MetricRow {
                    metric: "Avg Inter-Token Latency".to_string(),
                    value: format!("{:.2}ms", avg_inter_token.as_secs_f64() * 1000.0),
                },
                MetricRow {
                    metric: "Median Inter-Token Latency".to_string(),
                    value: format!("{:.2}ms", median_inter_token.as_secs_f64() * 1000.0),
                },
                MetricRow {
                    metric: "P95 Inter-Token Latency".to_string(),
                    value: format!("{:.2}ms", p95_inter_token.as_secs_f64() * 1000.0),
                },
                MetricRow {
                    metric: "Tokens per Second".to_string(),
                    value: format!("{:.2}", tokens_per_second),
                },
            ];

            let mut table = Table::new(rows);
            println!("{}", table);
            println!();

            if let Some(cost) = result.metadata.estimated_cost {
                println!(
                    "{} Estimated cost: {}",
                    "=>".bright_cyan(),
                    format!("${:.6}", cost).bright_green().bold()
                );
            }

            if args.stream {
                println!("\n{}", "Generated Content".bright_cyan().bold().underline());
                println!();
                println!("{}", result.content);
            }

            println!();
            println!("{} Profile complete!", "✓".bright_green().bold());
        }

        // Save to file if requested
        if let Some(ref output_path) = args.output {
            let json_data = serde_json::json!({
                "provider": args.provider,
                "model": args.model,
                "request_id": result.request_id.to_string(),
                "ttft_ms": ttft.as_millis(),
                "total_duration_ms": duration.as_millis(),
                "metrics": {
                    "ttft": ttft.as_millis(),
                    "avg_inter_token": avg_inter_token.as_millis(),
                    "median_inter_token": median_inter_token.as_millis(),
                    "p95_inter_token": p95_inter_token.as_millis(),
                    "tokens_per_second": tokens_per_second,
                },
                "content": result.content,
            });

            std::fs::write(output_path, serde_json::to_string_pretty(&json_data)?)?;

            if !quiet {
                println!("Results saved to: {}", output_path.display());
            }
        }
    }

    Ok(())
}
