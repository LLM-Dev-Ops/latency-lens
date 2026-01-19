//! Replay command implementation
//!
//! Replays a previous analysis with different configuration.
//!
//! # CLI Invocation Shape
//!
//! ```bash
//! llm-latency-lens replay --original-id <UUID> [--config new-config.yaml]
//! ```
//!
//! # Note
//!
//! Replay requires the original input metrics to be stored along with the DecisionEvent.
//! If metrics are not stored, replay will fail with an appropriate error message.

use crate::agents::ruvector::{EventQuery, RuVectorClient, RuVectorConfig};
use crate::cli::ReplayArgs;
use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::{info, warn};

/// Execute the replay command
pub async fn execute(args: &ReplayArgs, json_output: bool) -> Result<()> {
    info!(
        original_id = %args.original_id,
        "Starting replay"
    );

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

    let client = Arc::new(
        RuVectorClient::new(ruvector_config).context("Failed to create RuVector client")?,
    );

    // Parse original ID
    let original_uuid = uuid::Uuid::parse_str(&args.original_id)
        .with_context(|| format!("Invalid original ID: {}", args.original_id))?;

    // Query for original event
    let query = EventQuery::new()
        .agent_id("latency-analysis-agent")
        .decision_type("latency_analysis")
        .limit(100);

    let result = client
        .query_events(query)
        .await
        .context("Failed to query events")?;

    // Find the original event by checking metadata
    let original_event = result.events.into_iter().find(|e| {
        e.get("metadata")
            .and_then(|m| m.get("analysis_id"))
            .and_then(|a| a.as_str())
            .map(|a| a == original_uuid.to_string())
            .unwrap_or(false)
    });

    match original_event {
        Some(event) => {
            warn!(
                "Replay requested but original input metrics are not stored with DecisionEvent"
            );

            if json_output {
                let response = serde_json::json!({
                    "success": false,
                    "error": "REPLAY_NOT_AVAILABLE",
                    "message": "Replay requires original input metrics to be stored. Original analysis found but inputs are not persisted.",
                    "original_event": event,
                    "hint": "To enable replay, persist input metrics along with DecisionEvent"
                });
                println!("{}", serde_json::to_string_pretty(&response)?);
            } else {
                println!("\n{}", "=== Replay Not Available ===");
                println!();
                println!("The original analysis was found, but replay is not possible because");
                println!("the original input metrics were not stored with the DecisionEvent.");
                println!();
                println!("Original Event ID: {}", event.get("event_id").and_then(|v| v.as_str()).unwrap_or("unknown"));
                println!();
                println!("To enable replay functionality, input metrics must be persisted");
                println!("along with the DecisionEvent. This is an advisory output only.");
                println!();
                println!("Hint: Use the --store-inputs flag when running analysis to enable replay.");
            }

            Ok(())
        }
        None => {
            anyhow::bail!(
                "Original analysis {} not found in ruvector-service",
                args.original_id
            );
        }
    }
}
