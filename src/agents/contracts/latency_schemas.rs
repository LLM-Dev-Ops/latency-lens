//! Latency analysis schemas from agentics-contracts
//!
//! Input and output schemas for the Latency Analysis Agent.
//! These define the contract for the agent's API.

use llm_latency_lens_core::{Provider, RequestId, SessionId};
use llm_latency_lens_metrics::{AggregatedMetrics, LatencyDistribution, RequestMetrics, ThroughputStats};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

/// Input schema for Latency Analysis Agent
///
/// Defines what data the agent accepts for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyAnalysisInput {
    /// Unique analysis request ID
    pub analysis_id: Uuid,

    /// Session to analyze (optional, for filtering)
    pub session_id: Option<SessionId>,

    /// Raw request metrics to analyze
    pub metrics: Vec<RequestMetrics>,

    /// Analysis configuration
    pub config: LatencyAnalysisConfig,
}

impl LatencyAnalysisInput {
    /// Create a new input with default config
    pub fn new(metrics: Vec<RequestMetrics>) -> Self {
        Self {
            analysis_id: Uuid::new_v4(),
            session_id: None,
            metrics,
            config: LatencyAnalysisConfig::default(),
        }
    }

    /// Set session filter
    pub fn with_session(mut self, session_id: SessionId) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Set analysis config
    pub fn with_config(mut self, config: LatencyAnalysisConfig) -> Self {
        self.config = config;
        self
    }

    /// Validate input against schema requirements
    pub fn validate(&self) -> Result<(), InputValidationError> {
        if self.metrics.is_empty() {
            return Err(InputValidationError::EmptyMetrics);
        }

        if self.metrics.len() < self.config.min_samples as usize {
            return Err(InputValidationError::InsufficientSamples {
                required: self.config.min_samples,
                provided: self.metrics.len() as u64,
            });
        }

        // Validate each metric has required fields
        for (i, metric) in self.metrics.iter().enumerate() {
            if metric.ttft.is_zero() && metric.success {
                return Err(InputValidationError::InvalidMetric {
                    index: i,
                    reason: "successful request has zero TTFT".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Configuration for latency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyAnalysisConfig {
    /// Minimum number of samples required
    pub min_samples: u64,

    /// Whether to remove outliers
    pub remove_outliers: bool,

    /// Outlier threshold (sigma)
    pub outlier_sigma: f64,

    /// Number of warmup requests to exclude
    pub warmup_count: u32,

    /// Percentiles to calculate
    pub percentiles: Vec<f64>,

    /// Whether to include per-token analysis
    pub include_token_analysis: bool,

    /// Provider filter (optional)
    pub provider_filter: Option<Provider>,

    /// Model filter (optional)
    pub model_filter: Option<String>,

    /// Maximum latency threshold for validity (nanoseconds)
    pub max_latency_ns: Option<u64>,
}

impl Default for LatencyAnalysisConfig {
    fn default() -> Self {
        Self {
            min_samples: 10,
            remove_outliers: true,
            outlier_sigma: 3.0,
            warmup_count: 2,
            percentiles: vec![50.0, 90.0, 95.0, 99.0, 99.9],
            include_token_analysis: true,
            provider_filter: None,
            model_filter: None,
            max_latency_ns: None,
        }
    }
}

/// Output schema for Latency Analysis Agent
///
/// Defines the analysis results returned by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyAnalysisOutput {
    /// Analysis request ID (mirrors input)
    pub analysis_id: Uuid,

    /// Session analyzed (if filtered)
    pub session_id: Option<SessionId>,

    /// Analysis summary
    pub summary: AnalysisSummary,

    /// TTFT (Time to First Token) analysis
    pub ttft_analysis: LatencyDistributionAnalysis,

    /// Per-token latency analysis
    pub inter_token_analysis: LatencyDistributionAnalysis,

    /// End-to-end latency analysis
    pub total_latency_analysis: LatencyDistributionAnalysis,

    /// Throughput analysis
    pub throughput_analysis: ThroughputAnalysis,

    /// Provider-specific breakdowns
    pub provider_analysis: Vec<ProviderAnalysis>,

    /// Model-specific breakdowns
    pub model_analysis: Vec<ModelAnalysis>,

    /// Anomaly detection results
    pub anomalies: Vec<LatencyAnomaly>,

    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Analysis summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSummary {
    /// Total requests analyzed
    pub total_requests: u64,

    /// Requests included after filtering
    pub included_requests: u64,

    /// Requests excluded (warmup + outliers)
    pub excluded_requests: u64,

    /// Success rate (percentage)
    pub success_rate: f64,

    /// Analysis time window start
    pub time_window_start: chrono::DateTime<chrono::Utc>,

    /// Analysis time window end
    pub time_window_end: chrono::DateTime<chrono::Utc>,

    /// Total tokens analyzed
    pub total_tokens: u64,

    /// Average tokens per request
    pub avg_tokens_per_request: f64,
}

/// Detailed latency distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyDistributionAnalysis {
    /// Base distribution statistics
    pub distribution: LatencyDistribution,

    /// Variance (nanoseconds squared)
    pub variance_ns2: f64,

    /// Coefficient of variation
    pub coefficient_of_variation: f64,

    /// Skewness of the distribution
    pub skewness: f64,

    /// Kurtosis of the distribution
    pub kurtosis: f64,

    /// Histogram buckets (for visualization)
    pub histogram: Vec<HistogramBucket>,

    /// Trend analysis (if time-series)
    pub trend: Option<TrendAnalysis>,
}

/// Histogram bucket for distribution visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    /// Bucket lower bound (nanoseconds)
    pub lower_ns: u64,
    /// Bucket upper bound (nanoseconds)
    pub upper_ns: u64,
    /// Count in this bucket
    pub count: u64,
    /// Percentage of total
    pub percentage: f64,
}

/// Trend analysis for time-series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction
    pub direction: TrendDirection,
    /// Slope (change per request)
    pub slope_ns_per_request: f64,
    /// R-squared (fit quality)
    pub r_squared: f64,
    /// Whether trend is statistically significant
    pub is_significant: bool,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Throughput analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputAnalysis {
    /// Base throughput statistics
    pub stats: ThroughputStats,

    /// Peak throughput observed
    pub peak_tokens_per_second: f64,

    /// Sustained throughput (average over full duration)
    pub sustained_tokens_per_second: f64,

    /// Throughput variance
    pub variance: f64,

    /// Time to reach peak throughput
    pub time_to_peak_ms: Option<u64>,
}

/// Provider-specific analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderAnalysis {
    /// Provider
    pub provider: Provider,
    /// Number of requests
    pub request_count: u64,
    /// TTFT statistics for this provider
    pub ttft: LatencyDistribution,
    /// Total latency statistics
    pub total_latency: LatencyDistribution,
    /// Throughput statistics
    pub throughput: ThroughputStats,
    /// Success rate
    pub success_rate: f64,
}

/// Model-specific analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAnalysis {
    /// Provider
    pub provider: Provider,
    /// Model name
    pub model: String,
    /// Number of requests
    pub request_count: u64,
    /// TTFT statistics for this model
    pub ttft: LatencyDistribution,
    /// Total latency statistics
    pub total_latency: LatencyDistribution,
    /// Throughput statistics
    pub throughput: ThroughputStats,
    /// Average cost per request
    pub avg_cost_usd: Option<f64>,
}

/// Detected latency anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyAnomaly {
    /// Anomaly identifier
    pub anomaly_id: Uuid,
    /// Request that triggered anomaly
    pub request_id: RequestId,
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// Severity (0.0 - 1.0)
    pub severity: f64,
    /// Observed value (nanoseconds)
    pub observed_value_ns: u64,
    /// Expected value (nanoseconds)
    pub expected_value_ns: u64,
    /// Deviation from expected (sigma)
    pub deviation_sigma: f64,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of latency anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnomalyType {
    /// Unusually high TTFT
    HighTtft,
    /// Unusually high total latency
    HighTotalLatency,
    /// Unusually high inter-token latency
    HighInterTokenLatency,
    /// Possible cold start detected
    ColdStart,
    /// Throughput degradation
    ThroughputDegradation,
    /// Request timeout
    Timeout,
}

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Agent version that performed analysis
    pub agent_version: String,
    /// Analysis duration (milliseconds)
    pub analysis_duration_ms: u64,
    /// Configuration used
    pub config: LatencyAnalysisConfig,
    /// Number of outliers removed
    pub outliers_removed: u64,
    /// Warmup requests excluded
    pub warmup_excluded: u64,
}

/// Input validation errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum InputValidationError {
    #[error("No metrics provided for analysis")]
    EmptyMetrics,

    #[error("Insufficient samples: required {required}, provided {provided}")]
    InsufficientSamples { required: u64, provided: u64 },

    #[error("Invalid metric at index {index}: {reason}")]
    InvalidMetric { index: usize, reason: String },
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
        let metrics: Vec<RequestMetrics> = (0..10).map(|_| create_test_metric()).collect();
        let input = LatencyAnalysisInput::new(metrics);

        assert!(input.validate().is_ok());
    }

    #[test]
    fn test_input_validation_empty() {
        let input = LatencyAnalysisInput::new(vec![]);

        assert!(matches!(
            input.validate(),
            Err(InputValidationError::EmptyMetrics)
        ));
    }

    #[test]
    fn test_input_validation_insufficient_samples() {
        let metrics = vec![create_test_metric()]; // Only 1 sample
        let input = LatencyAnalysisInput::new(metrics);

        assert!(matches!(
            input.validate(),
            Err(InputValidationError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_config_serialization() {
        let config = LatencyAnalysisConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: LatencyAnalysisConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.min_samples, deserialized.min_samples);
        assert_eq!(config.percentiles, deserialized.percentiles);
    }
}
