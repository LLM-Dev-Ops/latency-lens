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
//! 3. Creates execution context for Agentics execution graph
//! 4. Executes Latency Analysis Agent within tracked spans
//! 5. Persists DecisionEvent to ruvector-service
//! 6. Outputs analysis results with execution spans
//!
//! # What This Command NEVER Does
//!
//! - Modify system behavior
//! - Trigger remediation
//! - Apply optimizations

use crate::agents::{
    contracts::{LatencyAnalysisConfig, LatencyAnalysisInput, LatencyAnalysisOutput},
    execution_graph::{Artifact, ExecutionContext, ExecutionResult, RepoExecution},
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
use uuid::Uuid;

/// Execute the analyze command
pub async fn execute(args: &AnalyzeArgs, json_output: bool) -> Result<()> {
    info!(
        input = %args.input.display(),
        "Starting latency analysis"
    );

    // Build execution context from CLI args or auto-generate
    let execution_ctx = build_execution_context(args);
    let mut repo_exec = RepoExecution::begin(execution_ctx.clone());

    // Begin agent span
    let agent_span_id = repo_exec.begin_agent("latency-analysis-agent", "analysis");

    // Run the analysis within the execution context
    let result = run_analysis(args, &execution_ctx, json_output).await;

    match result {
        Ok((output, event_id)) => {
            // Build artifacts
            let artifacts = vec![
                Artifact::from_json_output(
                    "analysis_output",
                    &output.analysis_id.to_string(),
                    &output,
                ),
            ];
            repo_exec.complete_agent(agent_span_id, artifacts);

            // Finalize execution with result
            let dummy_ctx = ExecutionContext {
                execution_id: Uuid::nil(),
                parent_span_id: Uuid::nil(),
                trace_id: None,
            };
            let taken = std::mem::replace(&mut repo_exec, RepoExecution::begin(dummy_ctx));
            let exec_result = taken.finalize(Some(serde_json::to_value(&output)?));

            // Output execution spans if json
            if json_output {
                let json = serde_json::to_string_pretty(&exec_result)?;
                println!("{}", json);
            } else {
                output_console(&output, &event_id)?;
                // Print execution span summary
                print_span_summary(&exec_result);
            }

            // Write to output file if specified
            if let Some(ref output_path) = args.output {
                let json = serde_json::to_string_pretty(&exec_result)?;
                fs::write(output_path, json)?;
                info!(output = %output_path.display(), "Analysis written to file");
            }

            Ok(())
        }
        Err(e) => {
            repo_exec.fail_agent(agent_span_id, vec![e.to_string()]);

            let dummy_ctx = ExecutionContext {
                execution_id: Uuid::nil(),
                parent_span_id: Uuid::nil(),
                trace_id: None,
            };
            let taken = std::mem::replace(&mut repo_exec, RepoExecution::begin(dummy_ctx));
            let exec_result = taken.finalize(None);

            if json_output {
                let json = serde_json::to_string_pretty(&exec_result)?;
                println!("{}", json);
            }

            Err(e)
        }
    }
}

/// Build execution context from CLI args or auto-generate for standalone use
fn build_execution_context(args: &AnalyzeArgs) -> ExecutionContext {
    match (&args.execution_id, &args.parent_span_id) {
        (Some(eid), Some(psid)) => {
            let eid = Uuid::parse_str(eid).unwrap_or_else(|_| {
                warn!("Invalid execution_id, generating new one");
                Uuid::new_v4()
            });
            let psid = Uuid::parse_str(psid).unwrap_or_else(|_| {
                warn!("Invalid parent_span_id, generating new one");
                Uuid::new_v4()
            });
            ExecutionContext {
                execution_id: eid,
                parent_span_id: psid,
                trace_id: None,
            }
        }
        _ => {
            // CLI acts as its own Core for standalone use
            let eid = Uuid::new_v4();
            ExecutionContext {
                execution_id: eid,
                parent_span_id: eid,
                trace_id: None,
            }
        }
    }
}

/// Run the analysis, returning the output and event ID
async fn run_analysis(
    args: &AnalyzeArgs,
    execution_ctx: &ExecutionContext,
    _json_output: bool,
) -> Result<(LatencyAnalysisOutput, String)> {
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

    // Create and execute agent with execution context
    let mut agent = LatencyAnalysisAgent::new(ruvector_client);
    agent = agent.with_execution_context(execution_ctx);

    if let Some(ref exec_ref) = args.execution_ref {
        agent = agent.with_execution_ref(exec_ref.clone());
    }

    // Execute analysis
    let (output, decision_event) = agent
        .execute(input)
        .await
        .context("Latency analysis failed")?;

    info!(
        decision_event_id = %decision_event.event_id,
        "Analysis complete, DecisionEvent persisted"
    );

    Ok((output, decision_event.event_id.to_string()))
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

/// Print execution span summary for console output
fn print_span_summary(exec_result: &ExecutionResult) {
    use colored::Colorize;

    println!("{}", "Execution Spans:".bold());
    println!(
        "  Execution ID:      {}",
        exec_result.execution_id.to_string().dimmed()
    );
    println!(
        "  Repo Span:         {} ({})",
        exec_result.repo_span.span_id.to_string().dimmed(),
        format!("{:?}", exec_result.repo_span.status).green()
    );
    for span in &exec_result.agent_spans {
        println!(
            "  Agent Span:        {} ({}, {})",
            span.agent_name.as_deref().unwrap_or("unknown").cyan(),
            span.span_id.to_string().dimmed(),
            format!("{:?}", span.status).green()
        );
        for artifact in &span.artifacts {
            println!(
                "    Artifact:        {} [{}]",
                artifact.artifact_type.dimmed(),
                artifact.reference_id.dimmed()
            );
        }
    }
    println!();
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
