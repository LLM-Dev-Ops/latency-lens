//! Cold Start Mitigation Agent schemas from agentics-contracts
//!
//! Input and output schemas for the Cold Start Mitigation Agent.
//! These define the contract for the agent's API.

use crate::agents::contracts::{
    ColdStartEvent, ConfidenceMetadata, DecisionEvent, DecisionType, ErrorBounds,
    MeasurementConstraints,
};
use llm_latency_lens_core::{Provider, RequestId, SessionId};
use llm_latency_lens_metrics::{LatencyDistribution, RequestMetrics};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

/// Input schema for Cold Start Mitigation Agent
///
/// Defines what data the agent accepts for cold start measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartMeasurementInput {
    /// Unique measurement request ID
    pub measurement_id: Uuid,

    /// Session to analyze
    pub session_id: SessionId,

    /// Raw request metrics for cold start detection
    pub metrics: Vec<RequestMetrics>,

    /// Measurement configuration
    pub config: ColdStartMeasurementConfig,
}

impl ColdStartMeasurementInput {
    /// Create a new input with default config
    pub fn new(session_id: SessionId, metrics: Vec<RequestMetrics>) -> Self {
        Self {
            measurement_id: Uuid::new_v4(),
            session_id,
            metrics,
            config: ColdStartMeasurementConfig::default(),
        }
    }

    /// Set measurement config
    pub fn with_config(mut self, config: ColdStartMeasurementConfig) -> Self {
        self.config = config;
        self
    }

    /// Validate input against schema requirements
    pub fn validate(&self) -> Result<(), ColdStartInputValidationError> {
        if self.metrics.is_empty() {
            return Err(ColdStartInputValidationError::EmptyMetrics);
        }

        if self.metrics.len() < self.config.min_samples as usize {
            return Err(ColdStartInputValidationError::InsufficientSamples {
                required: self.config.min_samples,
                provided: self.metrics.len() as u64,
            });
        }

        // Validate cold start threshold
        if self.config.cold_start_threshold_multiplier <= 1.0 {
            return Err(ColdStartInputValidationError::InvalidConfig(
                "cold_start_threshold_multiplier must be > 1.0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Configuration for cold start measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartMeasurementConfig {
    /// Minimum number of samples required
    pub min_samples: u64,

    /// Cold start detection threshold multiplier
    /// (cold_start if TTFT > baseline * threshold)
    pub cold_start_threshold_multiplier: f64,

    /// Number of warmup requests to establish baseline
    pub baseline_warmup_count: u32,

    /// Whether to include first request as potential cold start
    pub include_first_request: bool,

    /// Statistical confidence level for detection
    pub confidence_level: f64,

    /// Provider filter (optional)
    pub provider_filter: Option<Provider>,

    /// Model filter (optional)
    pub model_filter: Option<String>,

    /// Maximum idle time to consider for cold start (milliseconds)
    pub max_idle_time_ms: Option<u64>,

    /// Minimum idle time to suspect cold start (milliseconds)
    pub min_idle_time_ms: u64,
}

impl Default for ColdStartMeasurementConfig {
    fn default() -> Self {
        Self {
            min_samples: 5,
            cold_start_threshold_multiplier: 2.0,
            baseline_warmup_count: 3,
            include_first_request: true,
            confidence_level: 0.95,
            provider_filter: None,
            model_filter: None,
            max_idle_time_ms: None,
            min_idle_time_ms: 60000, // 1 minute default
        }
    }
}

/// Output schema for Cold Start Mitigation Agent
///
/// Defines the measurement results returned by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartMeasurementOutput {
    /// Measurement request ID (mirrors input)
    pub measurement_id: Uuid,

    /// Session analyzed
    pub session_id: SessionId,

    /// Summary of cold start measurements
    pub summary: ColdStartSummary,

    /// Detected cold start events
    pub cold_start_events: Vec<ColdStartEvent>,

    /// Baseline latency statistics (warm requests)
    pub baseline_latency: LatencyDistribution,

    /// Cold start latency statistics
    pub cold_start_latency: Option<LatencyDistribution>,

    /// Provider-specific cold start analysis
    pub provider_analysis: Vec<ProviderColdStartAnalysis>,

    /// Model-specific cold start analysis
    pub model_analysis: Vec<ModelColdStartAnalysis>,

    /// Measurement metadata
    pub metadata: ColdStartMeasurementMetadata,
}

/// Summary of cold start measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartSummary {
    /// Total requests analyzed
    pub total_requests: u64,

    /// Number of detected cold starts
    pub cold_start_count: u64,

    /// Number of warm requests
    pub warm_request_count: u64,

    /// Cold start frequency (percentage)
    pub cold_start_frequency: f64,

    /// Average cold start latency overhead (nanoseconds)
    #[serde(with = "duration_nanos")]
    pub avg_cold_start_overhead: Duration,

    /// Maximum cold start latency overhead (nanoseconds)
    #[serde(with = "duration_nanos")]
    pub max_cold_start_overhead: Duration,

    /// Average cold start ratio (cold / warm latency)
    pub avg_cold_start_ratio: f64,

    /// Estimated cold start impact (percentage of total latency)
    pub cold_start_impact_percent: f64,

    /// Time window analyzed
    pub time_window_start: chrono::DateTime<chrono::Utc>,
    pub time_window_end: chrono::DateTime<chrono::Utc>,
}

/// Provider-specific cold start analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderColdStartAnalysis {
    /// Provider
    pub provider: Provider,

    /// Total requests for this provider
    pub total_requests: u64,

    /// Cold starts detected for this provider
    pub cold_start_count: u64,

    /// Cold start frequency for this provider
    pub cold_start_frequency: f64,

    /// Average cold start ratio for this provider
    pub avg_cold_start_ratio: f64,

    /// Baseline TTFT for this provider
    pub baseline_ttft: LatencyDistribution,
}

/// Model-specific cold start analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelColdStartAnalysis {
    /// Provider
    pub provider: Provider,

    /// Model name
    pub model: String,

    /// Total requests for this model
    pub total_requests: u64,

    /// Cold starts detected for this model
    pub cold_start_count: u64,

    /// Cold start frequency for this model
    pub cold_start_frequency: f64,

    /// Average cold start ratio for this model
    pub avg_cold_start_ratio: f64,

    /// Baseline TTFT for this model
    pub baseline_ttft: LatencyDistribution,
}

/// Measurement metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartMeasurementMetadata {
    /// Agent version that performed measurement
    pub agent_version: String,

    /// Measurement duration (milliseconds)
    pub measurement_duration_ms: u64,

    /// Configuration used
    pub config: ColdStartMeasurementConfig,

    /// Detection algorithm used
    pub detection_algorithm: ColdStartDetectionAlgorithm,

    /// Statistical confidence achieved
    pub confidence_achieved: f64,
}

/// Cold start detection algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColdStartDetectionAlgorithm {
    /// Simple threshold-based detection
    ThresholdBased,
    /// Statistical outlier detection (Z-score)
    ZScoreOutlier,
    /// Moving average baseline comparison
    MovingAverageBaseline,
    /// Inter-arrival time based detection
    InterArrivalTime,
}

/// Input validation errors for cold start measurement
#[derive(Debug, Clone, thiserror::Error)]
pub enum ColdStartInputValidationError {
    #[error("No metrics provided for measurement")]
    EmptyMetrics,

    #[error("Insufficient samples: required {required}, provided {provided}")]
    InsufficientSamples { required: u64, provided: u64 },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result of cold start classification for a single request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartClassification {
    /// Request ID
    pub request_id: RequestId,

    /// Whether this request was classified as a cold start
    pub is_cold_start: bool,

    /// Observed TTFT
    #[serde(with = "duration_nanos")]
    pub observed_ttft: Duration,

    /// Expected baseline TTFT
    #[serde(with = "duration_nanos")]
    pub expected_ttft: Duration,

    /// Cold start ratio (observed / expected)
    pub cold_start_ratio: f64,

    /// Detection confidence (0.0 - 1.0)
    pub confidence: f64,

    /// Contributing factors
    pub factors: Vec<ColdStartFactor>,
}

/// Factors contributing to cold start classification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColdStartFactor {
    /// First request in session
    FirstRequest,
    /// Long idle time before request
    IdleTime { idle_ms: u64 },
    /// TTFT exceeded threshold
    HighTtft { ratio: f64 },
    /// Network connection re-establishment
    ConnectionReset,
    /// Provider-specific initialization
    ProviderInitialization,
}

/// Serde module for Duration serialization to nanoseconds
mod duration_nanos {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_nanos() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nanos = u64::deserialize(deserializer)?;
        Ok(Duration::from_nanos(nanos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_metric() -> RequestMetrics {
        RequestMetrics {
            request_id: RequestId::new(),
            session_id: SessionId::new(),
            provider: Provider::OpenAI,
            model: "gpt-4".to_string(),
            timestamp: Utc::now(),
            ttft: Duration::from_millis(100),
            total_latency: Duration::from_secs(1),
            inter_token_latencies: vec![Duration::from_millis(20)],
            input_tokens: 100,
            output_tokens: 50,
            thinking_tokens: None,
            tokens_per_second: 50.0,
            cost_usd: Some(0.05),
            success: true,
            error: None,
        }
    }

    #[test]
    fn test_input_validation() {
        let session_id = SessionId::new();
        let metrics: Vec<RequestMetrics> = (0..10).map(|_| create_test_metric()).collect();
        let input = ColdStartMeasurementInput::new(session_id, metrics);

        assert!(input.validate().is_ok());
    }

    #[test]
    fn test_input_validation_empty() {
        let session_id = SessionId::new();
        let input = ColdStartMeasurementInput::new(session_id, vec![]);

        assert!(matches!(
            input.validate(),
            Err(ColdStartInputValidationError::EmptyMetrics)
        ));
    }

    #[test]
    fn test_config_serialization() {
        let config = ColdStartMeasurementConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ColdStartMeasurementConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.min_samples, deserialized.min_samples);
        assert!((config.cold_start_threshold_multiplier - deserialized.cold_start_threshold_multiplier).abs() < 0.001);
    }
}
