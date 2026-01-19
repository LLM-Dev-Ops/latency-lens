//! Analyze command implementation
//!
//! Performs latency analysis on collected metrics using the Latency Analysis Agent.
//! This command invokes the agent through its CLI interface.
//!
//! # CLI Invocation Shape
//!
//! ```bash
//! llm-latency-lens analyze --input metrics.json [--config config.yaml] [--output analysis.json]
//! ```
//!
//! # What This Command Does
//!
//! 1. Loads RequestMetrics from input file
//! 2. Validates input against agentics-contracts schema
//! 3. Executes Latency Analysis Agent
//! 4. Persists DecisionEvent to ruvector-service
//! 5. Outputs analysis results
//!
//! # What This Command NEVER Does
//!
//! - Modify system behavior
//! - Trigger remediation
//! - Apply optimizations

use crate::agents::{
    contracts::{LatencyAnalysisConfig, LatencyAnalysisInput, LatencyAnalysisOutput},
    latency_analysis::LatencyAnalysisAgent,
    ruvector::{RuVectorClient, RuVectorConfig},
};
use crate::cli::AnalyzeArgs;
use anyhow::{Context, Result};
use llm_latency_lens_metrics::RequestMetrics;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};

/// Execute the analyze command
pub async fn execute(args: &AnalyzeArgs, json_output: bool) -> Result<()> {
    info!(
        input = %args.input.display(),
        "Starting latency analysis"
    );

    // Load metrics from input file
    let metrics = load_metrics(&args.input)?;
    info!(metrics_count = metrics.len(), "Loaded metrics from file");

    // Build analysis configuration
    let config = build_config(args)?;

    // Create analysis input
    let input = LatencyAnalysisInput::new(metrics).with_config(config);

    // Validate input
    input
        .validate()
        .context("Input validation failed against agentics-contracts schema")?;

    // Create RuVector client
    let ruvector_config = if let Some(ref endpoint) = args.ruvector_endpoint {
        RuVectorConfig {
            endpoint: endpoint.clone(),
            api_key: args.ruvector_api_key.clone(),
            ..Default::default()
        }
    } else {
        RuVectorConfig::from_env()
    };

    let ruvector_client = Arc::new(
        RuVectorClient::new(ruvector_config).context("Failed to create RuVector client")?,
    );

    // Create and execute agent
    let mut agent = LatencyAnalysisAgent::new(ruvector_client);

    if let Some(ref exec_ref) = args.execution_ref {
        agent = agent.with_execution_ref(exec_ref.clone());
    }

    // Execute analysis
    let (output, decision_event) = agent
        .execute(input)
        .await
        .context("Latency analysis failed")?;

    // Output results
    if json_output {
        output_json(&output)?;
    } else {
        output_console(&output, &decision_event.event_id.to_string())?;
    }

    // Write to output file if specified
    if let Some(ref output_path) = args.output {
        let json = serde_json::to_string_pretty(&output)?;
        fs::write(output_path, json)?;
        info!(output = %output_path.display(), "Analysis written to file");
    }

    info!(
        decision_event_id = %decision_event.event_id,
        "Analysis complete, DecisionEvent persisted"
    );

    Ok(())
}

/// Load metrics from input file
fn load_metrics(path: &Path) -> Result<Vec<RequestMetrics>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read metrics file: {}", path.display()))?;

    // Try to parse as array of metrics or single AggregatedMetrics
    let metrics: Vec<RequestMetrics> = serde_json::from_str(&content)
        .with_context(|| "Failed to parse metrics JSON. Expected array of RequestMetrics.")?;

    if metrics.is_empty() {
        anyhow::bail!("No metrics found in input file");
    }

    Ok(metrics)
}

/// Build analysis configuration from CLI args
fn build_config(args: &AnalyzeArgs) -> Result<LatencyAnalysisConfig> {
    // Load base config from file if provided
    let mut config = if let Some(ref config_path) = args.config {
        let content = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
        serde_json::from_str(&content).with_context(|| "Failed to parse config JSON")?
    } else {
        LatencyAnalysisConfig::default()
    };

    // Override with CLI args
    if let Some(min_samples) = args.min_samples {
        config.min_samples = min_samples;
    }

    if let Some(warmup) = args.warmup {
        config.warmup_count = warmup;
    }

    if args.no_outlier_removal {
        config.remove_outliers = false;
    }

    if let Some(sigma) = args.outlier_sigma {
        config.outlier_sigma = sigma;
    }

    if let Some(ref provider) = args.provider_filter {
        config.provider_filter = Some(parse_provider(provider)?);
    }

    if let Some(ref model) = args.model_filter {
        config.model_filter = Some(model.clone());
    }

    Ok(config)
}

/// Parse provider string to Provider enum
fn parse_provider(s: &str) -> Result<llm_latency_lens_core::Provider> {
    match s.to_lowercase().as_str() {
        "openai" => Ok(llm_latency_lens_core::Provider::OpenAI),
        "anthropic" => Ok(llm_latency_lens_core::Provider::Anthropic),
        "google" => Ok(llm_latency_lens_core::Provider::Google),
        "aws-bedrock" | "bedrock" => Ok(llm_latency_lens_core::Provider::AwsBedrock),
        "azure-openai" | "azure" => Ok(llm_latency_lens_core::Provider::AzureOpenAI),
        "generic" => Ok(llm_latency_lens_core::Provider::Generic),
        _ => anyhow::bail!("Unknown provider: {}", s),
    }
}

/// Output results as JSON
fn output_json(output: &LatencyAnalysisOutput) -> Result<()> {
    let json = serde_json::to_string_pretty(output)?;
    println!("{}", json);
    Ok(())
}

/// Output results to console
fn output_console(output: &LatencyAnalysisOutput, event_id: &str) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "=== Latency Analysis Results ===".bold().cyan());
    println!();

    // Summary
    println!("{}", "Summary:".bold());
    println!(
        "  Total Requests:    {}",
        output.summary.total_requests.to_string().green()
    );
    println!(
        "  Included:          {}",
        output.summary.included_requests.to_string().green()
    );
    println!(
        "  Excluded:          {}",
        output.summary.excluded_requests.to_string().yellow()
    );
    println!(
        "  Success Rate:      {:.1}%",
        output.summary.success_rate
    );
    println!();

    // TTFT Analysis
    println!("{}", "Time to First Token (TTFT):".bold());
    print_distribution(&output.ttft_analysis.distribution);
    println!();

    // Total Latency
    println!("{}", "Total Request Latency:".bold());
    print_distribution(&output.total_latency_analysis.distribution);
    println!();

    // Throughput
    println!("{}", "Throughput:".bold());
    println!(
        "  Mean:              {:.1} tokens/sec",
        output.throughput_analysis.stats.mean_tokens_per_second
    );
    println!(
        "  Peak:              {:.1} tokens/sec",
        output.throughput_analysis.peak_tokens_per_second
    );
    println!(
        "  P95:               {:.1} tokens/sec",
        output.throughput_analysis.stats.p95_tokens_per_second
    );
    println!();

    // Anomalies
    if !output.anomalies.is_empty() {
        println!(
            "{} {}",
            "Anomalies Detected:".bold().red(),
            output.anomalies.len()
        );
        for anomaly in output.anomalies.iter().take(5) {
            println!(
                "  - {:?}: severity={:.2}, deviation={:.1}σ",
                anomaly.anomaly_type, anomaly.severity, anomaly.deviation_sigma
            );
        }
        if output.anomalies.len() > 5 {
            println!("  ... and {} more", output.anomalies.len() - 5);
        }
        println!();
    }

    // DecisionEvent
    println!("{}", "DecisionEvent:".bold());
    println!("  ID: {}", event_id.dimmed());
    println!(
        "  Persisted to ruvector-service: {}",
        "✓".green()
    );
    println!();

    Ok(())
}

/// Print distribution statistics
fn print_distribution(dist: &llm_latency_lens_metrics::LatencyDistribution) {
    println!(
        "  Min:               {:.2}ms",
        dist.min.as_secs_f64() * 1000.0
    );
    println!(
        "  Max:               {:.2}ms",
        dist.max.as_secs_f64() * 1000.0
    );
    println!(
        "  Mean:              {:.2}ms",
        dist.mean.as_secs_f64() * 1000.0
    );
    println!(
        "  P50:               {:.2}ms",
        dist.p50.as_secs_f64() * 1000.0
    );
    println!(
        "  P95:               {:.2}ms",
        dist.p95.as_secs_f64() * 1000.0
    );
    println!(
        "  P99:               {:.2}ms",
        dist.p99.as_secs_f64() * 1000.0
    );
    println!("  Samples:           {}", dist.sample_count);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_provider() {
        assert!(parse_provider("openai").is_ok());
        assert!(parse_provider("ANTHROPIC").is_ok());
        assert!(parse_provider("Google").is_ok());
        assert!(parse_provider("unknown").is_err());
    }
}
