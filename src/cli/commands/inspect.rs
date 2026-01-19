//! Inspect command implementation
//!
//! Inspects previous analysis results and DecisionEvents.
//!
//! # CLI Invocation Shape
//!
//! ```bash
//! llm-latency-lens inspect --event-id <UUID>
//! llm-latency-lens inspect --analysis-id <UUID>
//! llm-latency-lens inspect --query "latency_analysis" --from 2024-01-01 --to 2024-01-31
//! ```

use crate::agents::ruvector::{EventQuery, RuVectorClient, RuVectorConfig};
use crate::cli::InspectArgs;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tracing::info;

/// Execute the inspect command
pub async fn execute(args: &InspectArgs, json_output: bool) -> Result<()> {
    info!("Starting inspection");

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

    // Inspect by event ID
    if let Some(ref event_id) = args.event_id {
        let uuid = uuid::Uuid::parse_str(event_id)
            .with_context(|| format!("Invalid event ID: {}", event_id))?;

        let event = client
            .get_event(uuid)
            .await
            .context("Failed to fetch event")?;

        match event {
            Some(e) => {
                if json_output {
                    println!("{}", serde_json::to_string_pretty(&e)?);
                } else {
                    print_event(&e)?;
                }
            }
            None => {
                anyhow::bail!("Event {} not found", event_id);
            }
        }

        return Ok(());
    }

    // Build query from args
    let mut query = EventQuery::new()
        .agent_id("latency-analysis-agent")
        .decision_type("latency_analysis");

    if let Some(ref from) = args.from_time {
        let from_dt: DateTime<Utc> = from
            .parse()
            .with_context(|| format!("Invalid from time: {}", from))?;
        if let Some(ref to) = args.to_time {
            let to_dt: DateTime<Utc> = to
                .parse()
                .with_context(|| format!("Invalid to time: {}", to))?;
            query = query.time_range(from_dt, to_dt);
        }
    }

    if let Some(limit) = args.limit {
        query = query.limit(limit);
    }

    // Execute query
    let result = client
        .query_events(query)
        .await
        .context("Failed to query events")?;

    if json_output {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("\n=== Inspection Results ===\n");
        println!("Total Events: {}", result.total_count);
        println!("Has More: {}", result.has_more);
        println!();

        for (i, event) in result.events.iter().enumerate() {
            println!("--- Event {} ---", i + 1);
            print_event(event)?;
            println!();
        }
    }

    Ok(())
}

/// Print event details
fn print_event(event: &serde_json::Value) -> Result<()> {
    use colored::Colorize;

    if let Some(event_id) = event.get("event_id").and_then(|v| v.as_str()) {
        println!("  Event ID:      {}", event_id.cyan());
    }

    if let Some(agent_id) = event.get("agent_id").and_then(|v| v.as_str()) {
        println!("  Agent:         {}", agent_id);
    }

    if let Some(version) = event.get("agent_version").and_then(|v| v.as_str()) {
        println!("  Version:       {}", version);
    }

    if let Some(decision_type) = event.get("decision_type").and_then(|v| v.as_str()) {
        println!("  Decision Type: {}", decision_type.green());
    }

    if let Some(timestamp) = event.get("timestamp").and_then(|v| v.as_str()) {
        println!("  Timestamp:     {}", timestamp);
    }

    if let Some(confidence) = event.get("confidence") {
        if let Some(score) = confidence.get("score").and_then(|v| v.as_f64()) {
            println!("  Confidence:    {:.2}%", score * 100.0);
        }
    }

    if let Some(outputs) = event.get("outputs") {
        if let Some(summary) = outputs.get("summary") {
            println!("  Summary:");
            if let Some(total) = summary.get("total_requests").and_then(|v| v.as_u64()) {
                println!("    Total Requests: {}", total);
            }
            if let Some(included) = summary.get("included_requests").and_then(|v| v.as_u64()) {
                println!("    Included:       {}", included);
            }
            if let Some(rate) = summary.get("success_rate").and_then(|v| v.as_f64()) {
                println!("    Success Rate:   {:.1}%", rate);
            }
        }
    }

    Ok(())
}
