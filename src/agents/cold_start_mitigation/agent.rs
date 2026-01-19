//! Cold Start Mitigation Agent Implementation
//!
//! This agent measures and characterizes cold start behavior.
//! Classification: MEASUREMENT
//!
//! # Google Cloud Edge Function Deployment
//!
//! This agent is designed to be deployed as a stateless Google Cloud Edge Function.
//! All persistence is handled via ruvector-service client calls.

use super::detector::{ColdStartDetector, DetectionResult};
use super::schemas::*;
use crate::agents::contracts::{
    ConfidenceMetadata, DecisionEvent, DecisionType, ErrorBounds, MeasurementConstraints,
};
use crate::agents::ruvector::RuVectorClient;
use async_trait::async_trait;
use chrono::Utc;
use llm_latency_lens_core::SessionId;
use llm_latency_lens_metrics::{LatencyDistribution, RequestMetrics};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};

/// Agent version (semver)
pub const AGENT_VERSION: &str = "0.1.0";
/// Agent identifier
pub const AGENT_ID: &str = "cold-start-mitigation-agent";

/// Errors from the Cold Start Mitigation Agent
#[derive(Debug, Error)]
pub enum ColdStartAgentError {
    #[error("Input validation failed: {0}")]
    ValidationError(#[from] ColdStartInputValidationError),

    #[error("Detection failed: {0}")]
    DetectionError(String),

    #[error("RuVector persistence failed: {0}")]
    PersistenceError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for Cold Start Agent operations
pub type ColdStartAgentResult<T> = Result<T, ColdStartAgentError>;

/// Cold Start Mitigation Agent
///
/// MEASUREMENT agent for detecting and characterizing cold start behavior.
///
/// # What This Agent DOES:
/// - Detect cold start vs warm request execution
/// - Measure initialization and startup delays
/// - Quantify cold start frequency and latency impact
/// - Emit telemetry compatible with LLM-Observatory
/// - Emit DecisionEvents to ruvector-service
///
/// # What This Agent MUST NEVER DO:
/// - Trigger warm-up logic
/// - Apply mitigations
/// - Modify deployment behavior
pub struct ColdStartMitigationAgent {
    detector: ColdStartDetector,
    ruvector_client: Option<RuVectorClient>,
    emit_telemetry: bool,
}

impl ColdStartMitigationAgent {
    /// Create a new Cold Start Mitigation Agent
    pub fn new(config: ColdStartMeasurementConfig) -> Self {
        Self {
            detector: ColdStartDetector::new(config),
            ruvector_client: None,
            emit_telemetry: true,
        }
    }

    /// Set the RuVector client for persistence
    pub fn with_ruvector_client(mut self, client: RuVectorClient) -> Self {
        self.ruvector_client = Some(client);
        self
    }

    /// Set the detection algorithm
    pub fn with_algorithm(mut self, algorithm: ColdStartDetectionAlgorithm) -> Self {
        self.detector = self.detector.with_algorithm(algorithm);
        self
    }

    /// Disable telemetry emission
    pub fn without_telemetry(mut self) -> Self {
        self.emit_telemetry = false;
        self
    }

    /// Execute cold start measurement
    ///
    /// This is the main entry point for the agent. It:
    /// 1. Validates inputs against agentics-contracts schemas
    /// 2. Runs cold start detection
    /// 3. Builds the output following agentics-contracts
    /// 4. Emits exactly ONE DecisionEvent to ruvector-service
    /// 5. Returns deterministic, machine-readable output
    #[instrument(skip(self, input), fields(measurement_id = %input.measurement_id))]
    pub async fn execute(
        &self,
        input: ColdStartMeasurementInput,
    ) -> ColdStartAgentResult<ColdStartMeasurementOutput> {
        let start_time = Instant::now();
        info!(
            "Starting cold start measurement for session {}",
            input.session_id
        );

        // Step 1: Validate inputs against contract
        input.validate()?;
        debug!("Input validation passed");

        // Step 2: Run detection
        let detection_result = self
            .detector
            .detect(input.session_id, &input.metrics);
        debug!(
            "Detection complete: {} cold starts found",
            detection_result.cold_start_events.len()
        );

        // Step 3: Build output
        let output = self.build_output(&input, detection_result, start_time.elapsed())?;

        // Step 4: Emit DecisionEvent to ruvector-service
        if let Some(ref client) = self.ruvector_client {
            self.emit_decision_event(client, &input, &output).await?;
        }

        // Step 5: Emit telemetry (LLM-Observatory compatible)
        if self.emit_telemetry {
            self.emit_telemetry(&output);
        }

        info!(
            "Cold start measurement complete: {} cold starts in {} requests",
            output.summary.cold_start_count, output.summary.total_requests
        );

        Ok(output)
    }

    /// Build output from detection results
    fn build_output(
        &self,
        input: &ColdStartMeasurementInput,
        result: DetectionResult,
        duration: Duration,
    ) -> ColdStartAgentResult<ColdStartMeasurementOutput> {
        let cold_start_count = result
            .classifications
            .iter()
            .filter(|c| c.is_cold_start)
            .count() as u64;
        let warm_count = result.classifications.len() as u64 - cold_start_count;

        // Calculate summary statistics
        let cold_start_frequency = if result.classifications.is_empty() {
            0.0
        } else {
            cold_start_count as f64 / result.classifications.len() as f64 * 100.0
        };

        let cold_start_overheads: Vec<Duration> = result
            .classifications
            .iter()
            .filter(|c| c.is_cold_start)
            .map(|c| c.observed_ttft.saturating_sub(c.expected_ttft))
            .collect();

        let avg_cold_start_overhead = if cold_start_overheads.is_empty() {
            Duration::ZERO
        } else {
            let sum: u128 = cold_start_overheads.iter().map(|d| d.as_nanos()).sum();
            Duration::from_nanos((sum / cold_start_overheads.len() as u128) as u64)
        };

        let max_cold_start_overhead = cold_start_overheads
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        let avg_cold_start_ratio = if cold_start_count == 0 {
            0.0
        } else {
            result
                .classifications
                .iter()
                .filter(|c| c.is_cold_start)
                .map(|c| c.cold_start_ratio)
                .sum::<f64>()
                / cold_start_count as f64
        };

        // Calculate cold start impact
        let total_latency: u128 = input.metrics.iter().map(|m| m.ttft.as_nanos()).sum();
        let cold_start_latency: u128 = cold_start_overheads.iter().map(|d| d.as_nanos()).sum();
        let cold_start_impact_percent = if total_latency > 0 {
            cold_start_latency as f64 / total_latency as f64 * 100.0
        } else {
            0.0
        };

        // Get time window
        let timestamps: Vec<_> = input.metrics.iter().map(|m| m.timestamp).collect();
        let time_window_start = timestamps.iter().min().copied().unwrap_or_else(Utc::now);
        let time_window_end = timestamps.iter().max().copied().unwrap_or_else(Utc::now);

        let summary = ColdStartSummary {
            total_requests: input.metrics.len() as u64,
            cold_start_count,
            warm_request_count: warm_count,
            cold_start_frequency,
            avg_cold_start_overhead,
            max_cold_start_overhead,
            avg_cold_start_ratio,
            cold_start_impact_percent,
            time_window_start,
            time_window_end,
        };

        // Calculate baseline latency distribution
        let baseline_latency = self.calculate_latency_distribution(
            &result
                .classifications
                .iter()
                .filter(|c| !c.is_cold_start)
                .map(|c| c.observed_ttft)
                .collect::<Vec<_>>(),
        );

        // Calculate cold start latency distribution
        let cold_start_latency = if cold_start_count > 0 {
            Some(self.calculate_latency_distribution(
                &result
                    .classifications
                    .iter()
                    .filter(|c| c.is_cold_start)
                    .map(|c| c.observed_ttft)
                    .collect::<Vec<_>>(),
            ))
        } else {
            None
        };

        // Build provider analysis
        let provider_analysis = self.build_provider_analysis(input, &result);

        // Build model analysis
        let model_analysis = self.build_model_analysis(input, &result);

        let metadata = ColdStartMeasurementMetadata {
            agent_version: AGENT_VERSION.to_string(),
            measurement_duration_ms: duration.as_millis() as u64,
            config: input.config.clone(),
            detection_algorithm: self.detector.algorithm(),
            confidence_achieved: result.confidence,
        };

        Ok(ColdStartMeasurementOutput {
            measurement_id: input.measurement_id,
            session_id: input.session_id,
            summary,
            cold_start_events: result.cold_start_events,
            baseline_latency,
            cold_start_latency,
            provider_analysis,
            model_analysis,
            metadata,
        })
    }

    /// Calculate latency distribution from durations
    fn calculate_latency_distribution(&self, durations: &[Duration]) -> LatencyDistribution {
        if durations.is_empty() {
            return LatencyDistribution::empty();
        }

        let mut sorted: Vec<u64> = durations.iter().map(|d| d.as_nanos() as u64).collect();
        sorted.sort();

        let count = sorted.len();
        let sum: u128 = sorted.iter().map(|&d| d as u128).sum();
        let mean_nanos = (sum / count as u128) as u64;

        // Calculate std_dev
        let variance: f64 = sorted
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean_nanos as f64;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev_nanos = variance.sqrt() as u64;

        // Percentile helper
        let percentile = |p: f64| -> u64 {
            let idx = ((p / 100.0) * (count - 1) as f64).round() as usize;
            sorted[idx.min(count - 1)]
        };

        LatencyDistribution {
            min: Duration::from_nanos(sorted[0]),
            max: Duration::from_nanos(sorted[count - 1]),
            mean: Duration::from_nanos(mean_nanos),
            std_dev: Duration::from_nanos(std_dev_nanos),
            p50: Duration::from_nanos(percentile(50.0)),
            p90: Duration::from_nanos(percentile(90.0)),
            p95: Duration::from_nanos(percentile(95.0)),
            p99: Duration::from_nanos(percentile(99.0)),
            p99_9: Duration::from_nanos(percentile(99.9)),
            sample_count: count as u64,
        }
    }

    /// Build provider-specific analysis
    fn build_provider_analysis(
        &self,
        input: &ColdStartMeasurementInput,
        result: &DetectionResult,
    ) -> Vec<ProviderColdStartAnalysis> {
        let mut provider_stats: HashMap<String, (u64, u64, f64, Vec<Duration>)> = HashMap::new();

        for (classification, metric) in result.classifications.iter().zip(input.metrics.iter()) {
            let key = metric.provider.as_str().to_string();
            let entry = provider_stats.entry(key).or_insert((0, 0, 0.0, Vec::new()));

            entry.0 += 1; // total
            if classification.is_cold_start {
                entry.1 += 1; // cold starts
                entry.2 += classification.cold_start_ratio; // sum of ratios
            }
            entry.3.push(metric.ttft); // ttfts
        }

        provider_stats
            .into_iter()
            .map(|(provider_str, (total, cold, ratio_sum, ttfts))| {
                let provider = match provider_str.as_str() {
                    "openai" => llm_latency_lens_core::Provider::OpenAI,
                    "anthropic" => llm_latency_lens_core::Provider::Anthropic,
                    "google" => llm_latency_lens_core::Provider::Google,
                    "aws-bedrock" => llm_latency_lens_core::Provider::AwsBedrock,
                    "azure-openai" => llm_latency_lens_core::Provider::AzureOpenAI,
                    _ => llm_latency_lens_core::Provider::Generic,
                };

                let warm_ttfts: Vec<Duration> = result
                    .classifications
                    .iter()
                    .zip(input.metrics.iter())
                    .filter(|(c, m)| m.provider == provider && !c.is_cold_start)
                    .map(|(_, m)| m.ttft)
                    .collect();

                ProviderColdStartAnalysis {
                    provider,
                    total_requests: total,
                    cold_start_count: cold,
                    cold_start_frequency: if total > 0 {
                        cold as f64 / total as f64 * 100.0
                    } else {
                        0.0
                    },
                    avg_cold_start_ratio: if cold > 0 {
                        ratio_sum / cold as f64
                    } else {
                        0.0
                    },
                    baseline_ttft: self.calculate_latency_distribution(&warm_ttfts),
                }
            })
            .collect()
    }

    /// Build model-specific analysis
    fn build_model_analysis(
        &self,
        input: &ColdStartMeasurementInput,
        result: &DetectionResult,
    ) -> Vec<ModelColdStartAnalysis> {
        let mut model_stats: HashMap<String, (llm_latency_lens_core::Provider, u64, u64, f64, Vec<Duration>)> =
            HashMap::new();

        for (classification, metric) in result.classifications.iter().zip(input.metrics.iter()) {
            let key = format!("{}:{}", metric.provider.as_str(), metric.model);
            let entry = model_stats
                .entry(key)
                .or_insert((metric.provider, 0, 0, 0.0, Vec::new()));

            entry.1 += 1; // total
            if classification.is_cold_start {
                entry.2 += 1; // cold starts
                entry.3 += classification.cold_start_ratio; // sum of ratios
            }
            entry.4.push(metric.ttft); // ttfts
        }

        model_stats
            .into_iter()
            .map(|(key, (provider, total, cold, ratio_sum, _))| {
                let model = key.split(':').nth(1).unwrap_or("unknown").to_string();

                let warm_ttfts: Vec<Duration> = result
                    .classifications
                    .iter()
                    .zip(input.metrics.iter())
                    .filter(|(c, m)| {
                        m.provider == provider && m.model == model && !c.is_cold_start
                    })
                    .map(|(_, m)| m.ttft)
                    .collect();

                ModelColdStartAnalysis {
                    provider,
                    model,
                    total_requests: total,
                    cold_start_count: cold,
                    cold_start_frequency: if total > 0 {
                        cold as f64 / total as f64 * 100.0
                    } else {
                        0.0
                    },
                    avg_cold_start_ratio: if cold > 0 {
                        ratio_sum / cold as f64
                    } else {
                        0.0
                    },
                    baseline_ttft: self.calculate_latency_distribution(&warm_ttfts),
                }
            })
            .collect()
    }

    /// Emit DecisionEvent to ruvector-service
    async fn emit_decision_event(
        &self,
        client: &RuVectorClient,
        input: &ColdStartMeasurementInput,
        output: &ColdStartMeasurementOutput,
    ) -> ColdStartAgentResult<()> {
        let confidence = ConfidenceMetadata {
            score: output.metadata.confidence_achieved,
            precision_unit: "nanoseconds".to_string(),
            error_bounds: Some(ErrorBounds {
                lower: 0.01,
                upper: 0.01,
                is_percentage: true,
            }),
            sample_size: output.summary.total_requests,
            confidence_interval_percent: Some(input.config.confidence_level * 100.0),
        };

        let constraints = MeasurementConstraints {
            min_sample_size: Some(input.config.min_samples),
            max_latency_threshold_ns: None,
            outlier_removal: false,
            outlier_threshold_sigma: None,
            warmup_excluded: input.config.baseline_warmup_count > 0,
            warmup_count: Some(input.config.baseline_warmup_count),
        };

        let event = DecisionEvent::builder()
            .agent_id(AGENT_ID)
            .agent_version(AGENT_VERSION)
            .decision_type(DecisionType::ColdStartMeasurement)
            .inputs(input)
            .outputs(serde_json::to_value(output).unwrap_or_default())
            .confidence(confidence)
            .constraints(constraints)
            .execution_ref(input.measurement_id.to_string())
            .build()
            .map_err(|e| ColdStartAgentError::PersistenceError(e.to_string()))?;

        // Validate event before persisting
        event
            .validate()
            .map_err(|e| ColdStartAgentError::PersistenceError(e.to_string()))?;

        // Persist to ruvector-service
        client
            .persist_event(&event)
            .await
            .map_err(|e: crate::agents::ruvector::RuVectorError| ColdStartAgentError::PersistenceError(e.to_string()))?;

        debug!("DecisionEvent persisted: {}", event.event_id);
        Ok(())
    }

    /// Emit telemetry compatible with LLM-Observatory
    fn emit_telemetry(&self, output: &ColdStartMeasurementOutput) {
        // Emit structured telemetry using tracing
        // This is picked up by OpenTelemetry exporters
        tracing::info!(
            target: "llm_observatory",
            measurement_id = %output.measurement_id,
            session_id = %output.session_id,
            cold_start_count = output.summary.cold_start_count,
            warm_request_count = output.summary.warm_request_count,
            cold_start_frequency = output.summary.cold_start_frequency,
            cold_start_impact_percent = output.summary.cold_start_impact_percent,
            avg_cold_start_ratio = output.summary.avg_cold_start_ratio,
            "cold_start_measurement_complete"
        );
    }
}

/// Trait for agent execution (Edge Function interface)
#[async_trait]
pub trait MeasurementAgent: Send + Sync {
    /// Agent name
    fn name(&self) -> &'static str;

    /// Agent version
    fn version(&self) -> &'static str;

    /// Health check
    async fn health_check(&self) -> Result<(), String>;
}

#[async_trait]
impl MeasurementAgent for ColdStartMitigationAgent {
    fn name(&self) -> &'static str {
        AGENT_ID
    }

    fn version(&self) -> &'static str {
        AGENT_VERSION
    }

    async fn health_check(&self) -> Result<(), String> {
        // Check RuVector client if configured
        if let Some(ref client) = self.ruvector_client {
            client.health_check().await.map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use llm_latency_lens_core::{Provider, RequestId};

    fn create_test_metrics(count: usize, session_id: SessionId) -> Vec<RequestMetrics> {
        let mut metrics = Vec::new();

        for i in 0..count {
            // First request has higher TTFT (simulating cold start)
            let ttft_ms = if i == 0 { 300 } else { 100 + (i as u64 % 20) };

            let metric = RequestMetrics {
                request_id: RequestId::new(),
                session_id,
                provider: Provider::OpenAI,
                model: "gpt-4".to_string(),
                timestamp: Utc::now() + chrono::Duration::milliseconds(i as i64 * 100),
                ttft: Duration::from_millis(ttft_ms),
                total_latency: Duration::from_secs(1),
                inter_token_latencies: vec![Duration::from_millis(20)],
                input_tokens: 100,
                output_tokens: 50,
                thinking_tokens: None,
                tokens_per_second: 50.0,
                cost_usd: Some(0.05),
                success: true,
                error: None,
            };
            metrics.push(metric);
        }

        metrics
    }

    #[tokio::test]
    async fn test_agent_execution() {
        let config = ColdStartMeasurementConfig::default();
        let agent = ColdStartMitigationAgent::new(config).without_telemetry();

        let session_id = SessionId::new();
        let metrics = create_test_metrics(10, session_id);
        let input = ColdStartMeasurementInput::new(session_id, metrics);

        let result = agent.execute(input).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.summary.cold_start_count >= 1);
        assert!(output.summary.cold_start_frequency > 0.0);
    }

    #[tokio::test]
    async fn test_agent_validation_error() {
        let config = ColdStartMeasurementConfig::default();
        let agent = ColdStartMitigationAgent::new(config).without_telemetry();

        let session_id = SessionId::new();
        let input = ColdStartMeasurementInput::new(session_id, vec![]);

        let result = agent.execute(input).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ColdStartAgentError::ValidationError(_)
        ));
    }

    #[test]
    fn test_agent_name_and_version() {
        let config = ColdStartMeasurementConfig::default();
        let agent = ColdStartMitigationAgent::new(config);

        assert_eq!(agent.name(), AGENT_ID);
        assert_eq!(agent.version(), AGENT_VERSION);
    }
}
