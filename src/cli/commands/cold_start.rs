//! Cold Start command implementation
//!
//! CLI commands for the Cold Start Mitigation Agent.
//! Supports: profile, inspect, replay

use anyhow::{Context, Result};
use colored::Colorize;
use futures::FutureExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tabled::{Table, Tabled};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::cli::{ColdStartProfileArgs, ColdStartInspectArgs, ColdStartReplayArgs};
use crate::config::Config;
use crate::agents::cold_start_mitigation::{
    ColdStartMitigationAgent, ColdStartMeasurementConfig, ColdStartMeasurementInput,
    ColdStartDetectionAlgorithm, ColdStartMeasurementOutput,
};
use crate::agents::execution_graph::{Artifact, ExecutionContext, RepoExecution};
use crate::agents::ruvector::{RuVectorClient, RuVectorConfig};
use crate::orchestrator::{Orchestrator, OrchestratorConfig};
use llm_latency_lens_core::SessionId;
use llm_latency_lens_metrics::{MetricsCollector, RequestMetrics};
use llm_latency_lens_providers::{create_provider, MessageRole, StreamingRequest};

use super::read_prompt;

/// Build execution context from cold start CLI args or auto-generate
fn build_cold_start_execution_context(
    execution_id: &Option<String>,
    parent_span_id: &Option<String>,
) -> ExecutionContext {
    match (execution_id, parent_span_id) {
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

/// Run cold start profile command
pub async fn run_profile(
    args: ColdStartProfileArgs,
    mut config: Config,
    json_output: bool,
    quiet: bool,
    shutdown_signal: Arc<tokio::sync::Notify>,
) -> Result<()> {
    info!("Starting cold start profiling");

    // Build execution context
    let execution_ctx = build_cold_start_execution_context(
        &args.execution_id,
        &args.parent_span_id,
    );
    let mut repo_exec = RepoExecution::begin(execution_ctx.clone());
    let agent_span_id = repo_exec.begin_agent("cold-start-mitigation-agent", "measurement");

    // Merge CLI overrides
    config.merge_cli_overrides(&args.provider, args.api_key.clone(), None);
    config.validate()?;

    let provider_config = config.get_provider(&args.provider)?;
    let api_key = provider_config
        .api_key
        .as_ref()
        .context("API key not found")?;

    let provider = create_provider(&args.provider, api_key.clone())?;

    let prompt = read_prompt(&args.prompt, &args.prompt_file)?;

    if !quiet {
        println!(
            "{} Running cold start profiling for {} with model {}...",
            "=>".bright_cyan().bold(),
            args.provider.bright_yellow(),
            args.model.bright_green()
        );
        println!(
            "   {} requests with {}ms delay",
            args.requests.to_string().bright_yellow(),
            args.delay_ms.to_string().bright_yellow()
        );
    }

    // Collect metrics from multiple requests
    let session_id = SessionId::new();
    let collector = Arc::new(MetricsCollector::with_defaults(session_id)?);
    let orchestrator_config = OrchestratorConfig {
        concurrency: 1, // Sequential for cold start measurement
        total_requests: args.requests,
        rate_limit: 0,
        show_progress: !quiet,
        shutdown_timeout: Duration::from_secs(30),
    };
    let orchestrator = Orchestrator::new(orchestrator_config, shutdown_signal.clone());

    // Build request template
    let request = StreamingRequest::builder()
        .model(args.model.clone())
        .message(MessageRole::User, prompt)
        .max_tokens(args.max_tokens)
        .temperature(0.7)
        .timeout_secs(args.timeout)
        .build();

    // Execute with delays
    let mut all_metrics: Vec<RequestMetrics> = Vec::new();

    for i in 0..args.requests {
        // Check shutdown
        if shutdown_signal.notified().now_or_never().is_some() {
            break;
        }

        let single_config = OrchestratorConfig {
            concurrency: 1,
            total_requests: 1,
            rate_limit: 0,
            show_progress: false,
            shutdown_timeout: Duration::from_secs(30),
        };
        let single_orchestrator = Orchestrator::new(single_config, shutdown_signal.clone());

        let result = single_orchestrator
            .execute_single(provider.as_ref(), request.clone())
            .await;

        match result {
            Ok(metrics) => {
                debug!("Request {} completed: TTFT={:?}", i, metrics.ttft);
                all_metrics.push(metrics);
            }
            Err(e) => {
                if !quiet {
                    println!("  {} Request {} failed: {}", "!".bright_red(), i, e);
                }
            }
        }

        // Add delay between requests
        if i < args.requests - 1 && args.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(args.delay_ms)).await;
        }
    }

    if all_metrics.is_empty() {
        anyhow::bail!("No successful requests to analyze");
    }

    // Run cold start detection
    let algorithm = parse_algorithm(&args.algorithm)?;
    let measurement_config = ColdStartMeasurementConfig {
        cold_start_threshold_multiplier: args.threshold,
        min_samples: 3,
        baseline_warmup_count: 3,
        include_first_request: true,
        confidence_level: 0.95,
        provider_filter: None,
        model_filter: None,
        max_idle_time_ms: None,
        min_idle_time_ms: args.delay_ms,
    };

    let mut agent = ColdStartMitigationAgent::new(measurement_config)
        .with_algorithm(algorithm)
        .with_execution_context(&execution_ctx);

    // Set up RuVector client if persistence requested
    if args.persist {
        let ruvector_config = RuVectorConfig {
            endpoint: args.ruvector_endpoint.unwrap_or_default(),
            ..Default::default()
        };
        if let Ok(client) = RuVectorClient::new(ruvector_config) {
            agent = agent.with_ruvector_client(client);
        }
    }

    let input = ColdStartMeasurementInput::new(session_id, all_metrics);
    match agent.execute(input).await {
        Ok(output) => {
            // Complete agent span with artifacts
            let artifacts = vec![Artifact::from_json_output(
                "cold_start_output",
                &output.measurement_id.to_string(),
                &output,
            )];
            repo_exec.complete_agent(agent_span_id, artifacts);

            // Finalize execution
            let dummy_ctx = ExecutionContext {
                execution_id: Uuid::nil(),
                parent_span_id: Uuid::nil(),
                trace_id: None,
            };
            let taken = std::mem::replace(&mut repo_exec, RepoExecution::begin(dummy_ctx));
            let exec_result = taken.finalize(Some(serde_json::to_value(&output)?));

            // Output results
            if json_output {
                let json = serde_json::to_string_pretty(&exec_result)?;
                println!("{}", json);
            } else {
                output_table(&output, &args.output, quiet)?;
            }

            // Save to file if requested (with full execution result)
            if let Some(ref output_path) = args.output {
                let json = serde_json::to_string_pretty(&exec_result)?;
                std::fs::write(output_path, json)?;
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

            Err(anyhow::anyhow!("Cold start analysis failed: {}", e))
        }
    }
}

/// Run cold start inspect command
pub async fn run_inspect(
    args: ColdStartInspectArgs,
    _config: Config,
    json_output: bool,
    quiet: bool,
) -> Result<()> {
    info!("Running cold start inspection");

    // Load metrics from file
    let metrics: Vec<RequestMetrics> = if let Some(ref path) = args.input {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)?
    } else {
        anyhow::bail!("Input file required for inspect command");
    };

    let session_id = if let Some(ref id) = args.session_id {
        // Parse UUID
        SessionId::new() // Simplified - would parse from string
    } else if let Some(first) = metrics.first() {
        first.session_id
    } else {
        anyhow::bail!("No session ID available");
    };

    let algorithm = parse_algorithm(&args.algorithm)?;
    let config = ColdStartMeasurementConfig {
        cold_start_threshold_multiplier: args.threshold,
        ..Default::default()
    };

    let agent = ColdStartMitigationAgent::new(config)
        .with_algorithm(algorithm)
        .without_telemetry();

    let input = ColdStartMeasurementInput::new(session_id, metrics);
    let output = agent.execute(input).await?;

    if json_output || args.format == "json" {
        output_json(&output, &args.output, quiet)?;
    } else {
        output_table(&output, &args.output, quiet)?;
    }

    Ok(())
}

/// Run cold start replay command
pub async fn run_replay(
    args: ColdStartReplayArgs,
    _config: Config,
    json_output: bool,
    quiet: bool,
) -> Result<()> {
    info!("Running cold start replay for trace: {}", args.trace_id);

    let ruvector_config = RuVectorConfig {
        endpoint: args.ruvector_endpoint.unwrap_or_default(),
        ..Default::default()
    };

    let client = RuVectorClient::new(ruvector_config)?;

    // Query events by trace ID
    let event_id = uuid::Uuid::parse_str(&args.trace_id)
        .context("Invalid trace ID format - must be a valid UUID")?;
    let event = client.get_event(event_id).await
        .context("Failed to retrieve decision event")?
        .ok_or_else(|| anyhow::anyhow!("Decision event not found for trace ID: {}", args.trace_id))?;

    if json_output || args.format == "json" {
        let json = serde_json::to_string_pretty(&event)?;
        println!("{}", json);
    } else {
        // Display event as formatted JSON (event is serde_json::Value)
        println!("{}", "Decision Event".bright_cyan().bold().underline());
        println!();
        if let Some(event_id) = event.get("event_id") {
            println!("Event ID: {}", event_id.to_string().trim_matches('"').bright_yellow());
        }
        if let Some(agent_id) = event.get("agent_id") {
            let version = event.get("agent_version").map(|v| v.to_string()).unwrap_or_default();
            println!("Agent: {} v{}", agent_id.to_string().trim_matches('"').bright_green(), version.trim_matches('"'));
        }
        if let Some(decision_type) = event.get("decision_type") {
            println!("Type: {}", decision_type);
        }
        if let Some(classification) = event.get("classification") {
            println!("Classification: {}", classification);
        }
        if let Some(timestamp) = event.get("timestamp") {
            println!("Timestamp: {}", timestamp);
        }
        if let Some(confidence) = event.get("confidence") {
            if let Some(score) = confidence.get("score").and_then(|s| s.as_f64()) {
                println!("Confidence: {:.2}%", score * 100.0);
            }
        }
        println!();
        println!("{}", "Outputs".bright_cyan().bold());
        if let Some(outputs) = event.get("outputs") {
            println!("{}", serde_json::to_string_pretty(outputs)?);
        }
    }

    Ok(())
}

/// Parse detection algorithm from string
fn parse_algorithm(s: &str) -> Result<ColdStartDetectionAlgorithm> {
    match s.to_lowercase().as_str() {
        "threshold" | "threshold_based" => Ok(ColdStartDetectionAlgorithm::ThresholdBased),
        "zscore" | "z_score" | "outlier" => Ok(ColdStartDetectionAlgorithm::ZScoreOutlier),
        "moving_average" | "movingaverage" | "ma" => Ok(ColdStartDetectionAlgorithm::MovingAverageBaseline),
        "inter_arrival" | "interarrival" | "idle" => Ok(ColdStartDetectionAlgorithm::InterArrivalTime),
        _ => anyhow::bail!("Unknown algorithm: {}. Use: threshold, zscore, moving_average, inter_arrival", s),
    }
}

/// Output results as JSON
fn output_json(
    output: &ColdStartMeasurementOutput,
    path: &Option<PathBuf>,
    quiet: bool,
) -> Result<()> {
    let json = if quiet {
        serde_json::to_string(output)?
    } else {
        serde_json::to_string_pretty(output)?
    };

    if let Some(ref p) = path {
        std::fs::write(p, &json)?;
        if !quiet {
            println!("Results saved to: {}", p.display());
        }
    } else {
        println!("{}", json);
    }

    Ok(())
}

/// Output results as formatted table
fn output_table(
    output: &ColdStartMeasurementOutput,
    path: &Option<PathBuf>,
    quiet: bool,
) -> Result<()> {
    if !quiet {
        println!();
        println!("{}", "Cold Start Analysis Results".bright_cyan().bold().underline());
        println!();

        // Summary table
        #[derive(Tabled)]
        struct SummaryRow {
            #[tabled(rename = "Metric")]
            metric: String,
            #[tabled(rename = "Value")]
            value: String,
        }

        let rows = vec![
            SummaryRow {
                metric: "Session ID".to_string(),
                value: output.session_id.to_string(),
            },
            SummaryRow {
                metric: "Total Requests".to_string(),
                value: output.summary.total_requests.to_string(),
            },
            SummaryRow {
                metric: "Cold Starts Detected".to_string(),
                value: format!("{}", output.summary.cold_start_count).bright_red().to_string(),
            },
            SummaryRow {
                metric: "Warm Requests".to_string(),
                value: format!("{}", output.summary.warm_request_count).bright_green().to_string(),
            },
            SummaryRow {
                metric: "Cold Start Frequency".to_string(),
                value: format!("{:.1}%", output.summary.cold_start_frequency),
            },
            SummaryRow {
                metric: "Avg Cold Start Ratio".to_string(),
                value: format!("{:.2}x", output.summary.avg_cold_start_ratio),
            },
            SummaryRow {
                metric: "Avg Overhead".to_string(),
                value: format!("{:.2}ms", output.summary.avg_cold_start_overhead.as_secs_f64() * 1000.0),
            },
            SummaryRow {
                metric: "Max Overhead".to_string(),
                value: format!("{:.2}ms", output.summary.max_cold_start_overhead.as_secs_f64() * 1000.0),
            },
            SummaryRow {
                metric: "Impact on Total Latency".to_string(),
                value: format!("{:.1}%", output.summary.cold_start_impact_percent),
            },
            SummaryRow {
                metric: "Detection Algorithm".to_string(),
                value: format!("{:?}", output.metadata.detection_algorithm),
            },
            SummaryRow {
                metric: "Confidence".to_string(),
                value: format!("{:.1}%", output.metadata.confidence_achieved * 100.0),
            },
        ];

        let table = Table::new(rows);
        println!("{}", table);

        // Cold start events
        if !output.cold_start_events.is_empty() {
            println!();
            println!("{}", "Cold Start Events".bright_cyan().bold());

            #[derive(Tabled)]
            struct EventRow {
                #[tabled(rename = "#")]
                index: usize,
                #[tabled(rename = "Model")]
                model: String,
                #[tabled(rename = "Cold Start TTFT")]
                cold_ttft: String,
                #[tabled(rename = "Baseline TTFT")]
                baseline: String,
                #[tabled(rename = "Ratio")]
                ratio: String,
                #[tabled(rename = "First Req")]
                is_first: String,
            }

            let event_rows: Vec<EventRow> = output
                .cold_start_events
                .iter()
                .enumerate()
                .map(|(i, e)| EventRow {
                    index: i + 1,
                    model: e.model.clone(),
                    cold_ttft: format!("{:.2}ms", e.cold_start_latency.as_secs_f64() * 1000.0),
                    baseline: format!("{:.2}ms", e.baseline_latency.as_secs_f64() * 1000.0),
                    ratio: format!("{:.2}x", e.cold_start_ratio),
                    is_first: if e.is_first_request { "Yes".to_string() } else { "No".to_string() },
                })
                .collect();

            let table = Table::new(event_rows);
            println!("{}", table);
        }

        // Baseline statistics
        println!();
        println!("{}", "Baseline Latency (Warm Requests)".bright_cyan().bold());
        println!(
            "  Mean: {:.2}ms | P50: {:.2}ms | P95: {:.2}ms | P99: {:.2}ms",
            output.baseline_latency.mean.as_secs_f64() * 1000.0,
            output.baseline_latency.p50.as_secs_f64() * 1000.0,
            output.baseline_latency.p95.as_secs_f64() * 1000.0,
            output.baseline_latency.p99.as_secs_f64() * 1000.0,
        );

        println!();
        println!("{} Cold start analysis complete!", "✓".bright_green().bold());
    }

    // Save to file if requested
    if let Some(ref p) = path {
        let json = serde_json::to_string_pretty(output)?;
        std::fs::write(p, json)?;
        if !quiet {
            println!("Results saved to: {}", p.display());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_algorithm() {
        assert_eq!(parse_algorithm("threshold").unwrap(), ColdStartDetectionAlgorithm::ThresholdBased);
        assert_eq!(parse_algorithm("zscore").unwrap(), ColdStartDetectionAlgorithm::ZScoreOutlier);
        assert_eq!(parse_algorithm("moving_average").unwrap(), ColdStartDetectionAlgorithm::MovingAverageBaseline);
        assert_eq!(parse_algorithm("inter_arrival").unwrap(), ColdStartDetectionAlgorithm::InterArrivalTime);
        assert!(parse_algorithm("invalid").is_err());
    }
}
