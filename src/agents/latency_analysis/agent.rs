//! Latency Analysis Agent Implementation
//!
//! This is the main agent runtime that:
//! 1. Validates inputs against agentics-contracts
//! 2. Performs latency analysis
//! 3. Emits exactly ONE DecisionEvent
//! 4. Persists to ruvector-service
//! 5. Emits telemetry compatible with LLM-Observatory
//!
//! # Classification: ANALYSIS
//!
//! # Explicit Non-Responsibilities (MUST NEVER do)
//!
//! - Modify system behavior
//! - Trigger remediation
//! - Trigger retries
//! - Change routing
//! - Apply optimizations
//! - Enforce policies
//! - Perform orchestration
//! - Connect directly to Google SQL
//! - Execute SQL queries

use super::{analyzer::LatencyAnalyzer, telemetry::AgentTelemetry, AGENT_ID, AGENT_VERSION};
use crate::agents::{
    contracts::{
        ConfidenceMetadata, DecisionEvent, DecisionType, ErrorBounds, InputValidationError,
        LatencyAnalysisConfig, LatencyAnalysisInput, LatencyAnalysisOutput, MeasurementConstraints,
    },
    execution_graph::ExecutionContext,
    ruvector::{RuVectorClient, RuVectorError},
};
use std::sync::Arc;
use thiserror::Error;
use tracing::{error, info, instrument, warn};

/// Errors from Latency Analysis Agent
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Input validation failed: {0}")]
    InputValidation(#[from] InputValidationError),

    #[error("Persistence failed: {0}")]
    Persistence(#[from] RuVectorError),

    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),

    #[error("DecisionEvent validation failed: {0}")]
    DecisionEventInvalid(String),
}

/// Latency Analysis Agent
///
/// Measures and analyzes end-to-end and token-level latency characteristics
/// of LLM requests.
///
/// # Classification: ANALYSIS
///
/// # Scope
///
/// - Measure time-to-first-token (TTFT)
/// - Measure per-token streaming latency
/// - Measure end-to-end request duration
/// - Produce latency distributions and percentiles
///
/// # Contract Compliance
///
/// - Uses schemas from agentics-contracts ONLY
/// - Validates all inputs/outputs
/// - Emits exactly ONE DecisionEvent per invocation
/// - Persists to ruvector-service only (NEVER direct SQL)
/// - Telemetry compatible with LLM-Observatory
pub struct LatencyAnalysisAgent {
    /// RuVector client for persistence
    ruvector_client: Arc<RuVectorClient>,
    /// Telemetry emitter
    telemetry: AgentTelemetry,
    /// Execution reference for correlation
    execution_ref: Option<String>,
    /// Trace ID for distributed tracing
    trace_id: Option<String>,
    /// Span ID for distributed tracing
    span_id: Option<String>,
}

impl LatencyAnalysisAgent {
    /// Create a new agent with RuVector client
    pub fn new(ruvector_client: Arc<RuVectorClient>) -> Self {
        Self {
            ruvector_client,
            telemetry: AgentTelemetry::default(),
            execution_ref: None,
            trace_id: None,
            span_id: None,
        }
    }

    /// Set execution reference for correlation
    pub fn with_execution_ref(mut self, exec_ref: impl Into<String>) -> Self {
        self.execution_ref = Some(exec_ref.into());
        self
    }

    /// Set trace context for distributed tracing
    pub fn with_trace_context(
        mut self,
        trace_id: impl Into<String>,
        span_id: impl Into<String>,
    ) -> Self {
        self.trace_id = Some(trace_id.into());
        self.span_id = Some(span_id.into());
        self
    }

    /// Set execution context from the Agentics execution graph.
    ///
    /// Propagates trace_id and execution_ref from the execution context
    /// into the agent for DecisionEvent correlation.
    pub fn with_execution_context(mut self, ctx: &ExecutionContext) -> Self {
        self.trace_id = Some(ctx.effective_trace_id().to_string());
        self.execution_ref = Some(ctx.execution_id.to_string());
        self
    }

    /// Execute the agent
    ///
    /// This is the main entry point. It:
    /// 1. Validates input against contract
    /// 2. Performs analysis
    /// 3. Creates and persists DecisionEvent
    /// 4. Emits telemetry
    ///
    /// Returns the analysis output and the persisted DecisionEvent.
    #[instrument(skip(self, input), fields(analysis_id = %input.analysis_id))]
    pub async fn execute(
        &self,
        input: LatencyAnalysisInput,
    ) -> Result<(LatencyAnalysisOutput, DecisionEvent), AgentError> {
        info!(
            analysis_id = %input.analysis_id,
            metrics_count = input.metrics.len(),
            "Starting latency analysis"
        );

        // Step 1: Validate input against contract
        input.validate()?;

        // Step 2: Perform analysis
        let analyzer = LatencyAnalyzer::new(input.config.clone());
        let output = analyzer.analyze(&input);

        // Step 3: Create DecisionEvent
        let decision_event = self.create_decision_event(&input, &output)?;

        // Step 4: Persist DecisionEvent to ruvector-service
        // This is the ONLY persistence path - NEVER direct SQL
        match self.ruvector_client.persist_event(&decision_event).await {
            Ok(persist_response) => {
                info!(
                    event_id = %decision_event.event_id,
                    storage_ref = %persist_response.storage_ref,
                    latency_ms = persist_response.latency_ms,
                    "DecisionEvent persisted successfully"
                );
            }
            Err(e) => {
                error!(
                    error = %e,
                    event_id = %decision_event.event_id,
                    "Failed to persist DecisionEvent"
                );
                return Err(AgentError::Persistence(e));
            }
        }

        // Step 5: Emit telemetry (LLM-Observatory compatible)
        self.telemetry
            .emit_analysis_span(&input, &output, DecisionType::LatencyAnalysis);
        self.telemetry.emit_metrics(&output);

        info!(
            analysis_id = %input.analysis_id,
            included_requests = output.summary.included_requests,
            anomalies = output.anomalies.len(),
            "Latency analysis complete"
        );

        Ok((output, decision_event))
    }

    /// Create DecisionEvent from analysis results
    fn create_decision_event(
        &self,
        input: &LatencyAnalysisInput,
        output: &LatencyAnalysisOutput,
    ) -> Result<DecisionEvent, AgentError> {
        // Calculate confidence based on sample size and data quality
        let confidence = self.calculate_confidence(input, output);

        // Build measurement constraints
        let constraints = MeasurementConstraints {
            min_sample_size: Some(input.config.min_samples),
            max_latency_threshold_ns: input.config.max_latency_ns,
            outlier_removal: input.config.remove_outliers,
            outlier_threshold_sigma: if input.config.remove_outliers {
                Some(input.config.outlier_sigma)
            } else {
                None
            },
            warmup_excluded: input.config.warmup_count > 0,
            warmup_count: if input.config.warmup_count > 0 {
                Some(input.config.warmup_count)
            } else {
                None
            },
        };

        // Serialize output to JSON
        let outputs = serde_json::to_value(output).map_err(|e| {
            AgentError::DecisionEventInvalid(format!("Failed to serialize output: {}", e))
        })?;

        // Build DecisionEvent
        let mut builder = DecisionEvent::builder()
            .agent_id(AGENT_ID)
            .agent_version(AGENT_VERSION)
            .decision_type(DecisionType::LatencyAnalysis)
            .inputs(input)
            .outputs(outputs)
            .confidence(confidence)
            .constraints(constraints)
            .metadata(
                "analysis_id",
                serde_json::Value::String(input.analysis_id.to_string()),
            );

        if let Some(ref exec_ref) = self.execution_ref {
            builder = builder.execution_ref(exec_ref.clone());
        }

        if let Some(ref trace_id) = self.trace_id {
            builder = builder.trace_id(trace_id.clone());
        }

        if let Some(ref span_id) = self.span_id {
            builder = builder.span_id(span_id.clone());
        }

        let event = builder.build().map_err(|e| {
            AgentError::DecisionEventInvalid(format!("Failed to build DecisionEvent: {}", e))
        })?;

        // Validate the event
        event.validate().map_err(|e| {
            AgentError::DecisionEventInvalid(format!("DecisionEvent validation failed: {}", e))
        })?;

        Ok(event)
    }

    /// Calculate confidence metadata based on analysis quality
    fn calculate_confidence(
        &self,
        input: &LatencyAnalysisInput,
        output: &LatencyAnalysisOutput,
    ) -> ConfidenceMetadata {
        let sample_size = output.summary.included_requests;

        // Base confidence from sample size (more samples = higher confidence)
        let sample_confidence = if sample_size >= 1000 {
            0.99
        } else if sample_size >= 100 {
            0.95
        } else if sample_size >= 30 {
            0.90
        } else if sample_size >= 10 {
            0.80
        } else {
            0.60
        };

        // Adjust for data quality
        let success_rate_factor = output.summary.success_rate / 100.0;
        let cv_penalty = if output.total_latency_analysis.coefficient_of_variation > 1.0 {
            0.1 // High variance reduces confidence
        } else {
            0.0
        };

        let final_score = (sample_confidence * success_rate_factor - cv_penalty).max(0.0).min(1.0);

        // Calculate error bounds based on standard error
        let std_error = if sample_size > 1 {
            output.total_latency_analysis.distribution.std_dev.as_nanos() as f64
                / (sample_size as f64).sqrt()
        } else {
            0.0
        };

        let mean = output.total_latency_analysis.distribution.mean.as_nanos() as f64;
        let error_percent = if mean > 0.0 {
            (std_error / mean * 1.96 * 100.0).min(100.0) // 95% CI
        } else {
            0.0
        };

        ConfidenceMetadata {
            score: final_score,
            precision_unit: "nanoseconds".to_string(),
            error_bounds: Some(ErrorBounds {
                lower: error_percent,
                upper: error_percent,
                is_percentage: true,
            }),
            sample_size,
            confidence_interval_percent: Some(95.0),
        }
    }

    /// Get agent ID
    pub fn agent_id(&self) -> &'static str {
        AGENT_ID
    }

    /// Get agent version
    pub fn agent_version(&self) -> &'static str {
        AGENT_VERSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::ruvector::RuVectorConfig;
    use chrono::Utc;
    use llm_latency_lens_core::{Provider, RequestId, SessionId};
    use llm_latency_lens_metrics::RequestMetrics;
    use std::time::Duration;

    fn create_test_metrics(count: usize) -> Vec<RequestMetrics> {
        (0..count)
            .map(|i| RequestMetrics {
                request_id: RequestId::new(),
                session_id: SessionId::new(),
                provider: Provider::OpenAI,
                model: "gpt-4".to_string(),
                timestamp: Utc::now(),
                ttft: Duration::from_millis(100 + (i as u64 * 10)),
                total_latency: Duration::from_secs(1) + Duration::from_millis(i as u64 * 50),
                inter_token_latencies: vec![
                    Duration::from_millis(20),
                    Duration::from_millis(25),
                ],
                input_tokens: 100,
                output_tokens: 50,
                thinking_tokens: None,
                tokens_per_second: 50.0,
                cost_usd: Some(0.05),
                success: true,
                error: None,
            })
            .collect()
    }

    #[test]
    fn test_agent_creation() {
        let config = RuVectorConfig::default();
        let client = Arc::new(RuVectorClient::new(config).unwrap());
        let agent = LatencyAnalysisAgent::new(client);

        assert_eq!(agent.agent_id(), AGENT_ID);
        assert_eq!(agent.agent_version(), AGENT_VERSION);
    }

    #[test]
    fn test_confidence_calculation() {
        let config = RuVectorConfig::default();
        let client = Arc::new(RuVectorClient::new(config).unwrap());
        let agent = LatencyAnalysisAgent::new(client);

        let metrics = create_test_metrics(100);
        let input = LatencyAnalysisInput::new(metrics).with_config(LatencyAnalysisConfig {
            warmup_count: 0,
            min_samples: 1,
            ..Default::default()
        });

        let analyzer = LatencyAnalyzer::new(input.config.clone());
        let output = analyzer.analyze(&input);

        let confidence = agent.calculate_confidence(&input, &output);

        assert!(confidence.score > 0.0 && confidence.score <= 1.0);
        assert_eq!(confidence.precision_unit, "nanoseconds");
        assert!(confidence.error_bounds.is_some());
    }

    #[test]
    fn test_decision_event_creation() {
        let config = RuVectorConfig::default();
        let client = Arc::new(RuVectorClient::new(config).unwrap());
        let agent = LatencyAnalysisAgent::new(client)
            .with_execution_ref("test-exec-123")
            .with_trace_context("trace-abc", "span-xyz");

        let metrics = create_test_metrics(20);
        let input = LatencyAnalysisInput::new(metrics).with_config(LatencyAnalysisConfig {
            warmup_count: 0,
            min_samples: 1,
            ..Default::default()
        });

        let analyzer = LatencyAnalyzer::new(input.config.clone());
        let output = analyzer.analyze(&input);

        let event = agent.create_decision_event(&input, &output).unwrap();

        assert_eq!(event.agent_id, AGENT_ID);
        assert_eq!(event.agent_version, AGENT_VERSION);
        assert_eq!(event.decision_type, DecisionType::LatencyAnalysis);
        assert_eq!(event.execution_ref, Some("test-exec-123".to_string()));
        assert_eq!(event.trace_id, Some("trace-abc".to_string()));
        assert!(event.validate().is_ok());
    }
}
