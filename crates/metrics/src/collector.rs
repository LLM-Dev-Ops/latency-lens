//! Metrics collector using HDR Histogram for accurate percentile calculation
//!
//! Provides thread-safe collection of metrics with high-precision histogram tracking
//! for TTFT, inter-token latency, total request latency, and token throughput.

use crate::types::RequestMetrics;
use hdrhistogram::Histogram;
use llm_latency_lens_core::{Provider, RequestId, SessionId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::debug;

/// Configuration for the metrics collector
#[derive(Debug, Clone)]
pub struct CollectorConfig {
    /// Maximum value to track in histograms (in nanoseconds)
    /// Default: 60 seconds (60_000_000_000 ns)
    pub max_value_nanos: u64,

    /// Number of significant digits for histogram precision (1-5)
    /// Default: 3 (provides ~0.1% precision)
    pub significant_digits: u8,

    /// Whether to track per-provider metrics separately
    pub track_per_provider: bool,

    /// Whether to track per-model metrics separately
    pub track_per_model: bool,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            max_value_nanos: 60_000_000_000, // 60 seconds
            significant_digits: 3,
            track_per_provider: true,
            track_per_model: true,
        }
    }
}

impl CollectorConfig {
    /// Create a new collector configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum value to track (in seconds)
    pub fn with_max_value_seconds(mut self, seconds: u64) -> Self {
        self.max_value_nanos = seconds * 1_000_000_000;
        self
    }

    /// Set the number of significant digits for histogram precision
    pub fn with_significant_digits(mut self, digits: u8) -> Self {
        self.significant_digits = digits.clamp(1, 5);
        self
    }

    /// Enable or disable per-provider tracking
    pub fn with_per_provider_tracking(mut self, enabled: bool) -> Self {
        self.track_per_provider = enabled;
        self
    }

    /// Enable or disable per-model tracking
    pub fn with_per_model_tracking(mut self, enabled: bool) -> Self {
        self.track_per_model = enabled;
        self
    }
}

/// Internal histogram set for tracking latency metrics
#[derive(Clone)]
pub struct HistogramSet {
    /// Time to first token histogram
    pub(crate) ttft: Histogram<u64>,

    /// Inter-token latency histogram
    pub(crate) inter_token: Histogram<u64>,

    /// Total request latency histogram
    pub(crate) total_latency: Histogram<u64>,

    /// Token throughput histogram (stored as tokens/sec * 1000 for precision)
    pub(crate) throughput: Histogram<u64>,
}

impl HistogramSet {
    /// Create a new histogram set with the given configuration
    fn new(config: &CollectorConfig) -> Result<Self, MetricsError> {
        let create_histogram = || {
            Histogram::new_with_max(config.max_value_nanos, config.significant_digits)
                .map_err(|e| MetricsError::HistogramCreation(e.to_string()))
        };

        Ok(Self {
            ttft: create_histogram()?,
            inter_token: create_histogram()?,
            total_latency: create_histogram()?,
            // For throughput, we track up to 1M tokens/sec
            throughput: Histogram::new_with_max(1_000_000_000, config.significant_digits)
                .map_err(|e| MetricsError::HistogramCreation(e.to_string()))?,
        })
    }

    /// Record a request's metrics into this histogram set
    fn record(&mut self, metrics: &RequestMetrics) -> Result<(), MetricsError> {
        // Record TTFT
        self.ttft
            .record(metrics.ttft.as_nanos() as u64)
            .map_err(|e| MetricsError::HistogramRecord(e.to_string()))?;

        // Record total latency
        self.total_latency
            .record(metrics.total_latency.as_nanos() as u64)
            .map_err(|e| MetricsError::HistogramRecord(e.to_string()))?;

        // Record inter-token latencies
        for latency in &metrics.inter_token_latencies {
            self.inter_token
                .record(latency.as_nanos() as u64)
                .map_err(|e| MetricsError::HistogramRecord(e.to_string()))?;
        }

        // Record throughput (tokens/sec * 1000 for precision)
        let throughput_scaled = (metrics.tokens_per_second * 1000.0) as u64;
        self.throughput
            .record(throughput_scaled)
            .map_err(|e| MetricsError::HistogramRecord(e.to_string()))?;

        Ok(())
    }
}

/// Internal state for the metrics collector
struct CollectorState {
    /// Session ID for this collection
    session_id: SessionId,

    /// Configuration
    config: CollectorConfig,

    /// Global histogram set
    global_histograms: HistogramSet,

    /// Per-provider histogram sets
    provider_histograms: HashMap<Provider, HistogramSet>,

    /// Per-model histogram sets
    model_histograms: HashMap<String, HistogramSet>,

    /// All collected request metrics
    request_metrics: Vec<RequestMetrics>,

    /// Provider request counts
    provider_counts: HashMap<Provider, u64>,

    /// Model request counts
    model_counts: HashMap<String, u64>,

    /// Total number of successful requests
    successful_requests: u64,

    /// Total number of failed requests
    failed_requests: u64,

    /// Total input tokens
    total_input_tokens: u64,

    /// Total output tokens
    total_output_tokens: u64,

    /// Total thinking tokens
    total_thinking_tokens: u64,

    /// Total cost in USD
    total_cost_usd: f64,
}

impl CollectorState {
    /// Create a new collector state
    fn new(session_id: SessionId, config: CollectorConfig) -> Result<Self, MetricsError> {
        Ok(Self {
            session_id,
            global_histograms: HistogramSet::new(&config)?,
            config,
            provider_histograms: HashMap::new(),
            model_histograms: HashMap::new(),
            request_metrics: Vec::new(),
            provider_counts: HashMap::new(),
            model_counts: HashMap::new(),
            successful_requests: 0,
            failed_requests: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_thinking_tokens: 0,
            total_cost_usd: 0.0,
        })
    }

    /// Record a new request's metrics
    fn record(&mut self, metrics: RequestMetrics) -> Result<(), MetricsError> {
        // Update success/failure counters
        if metrics.success {
            self.successful_requests += 1;

            // Record into global histograms
            self.global_histograms.record(&metrics)?;

            // Record per-provider if enabled
            if self.config.track_per_provider {
                let provider_hist = self
                    .provider_histograms
                    .entry(metrics.provider)
                    .or_insert_with(|| HistogramSet::new(&self.config).unwrap());
                provider_hist.record(&metrics)?;

                *self.provider_counts.entry(metrics.provider).or_insert(0) += 1;
            }

            // Record per-model if enabled
            if self.config.track_per_model {
                let model_hist = self
                    .model_histograms
                    .entry(metrics.model.clone())
                    .or_insert_with(|| HistogramSet::new(&self.config).unwrap());
                model_hist.record(&metrics)?;

                *self.model_counts.entry(metrics.model.clone()).or_insert(0) += 1;
            }

            // Update token counts
            self.total_input_tokens += metrics.input_tokens;
            self.total_output_tokens += metrics.output_tokens;
            self.total_thinking_tokens += metrics.thinking_tokens.unwrap_or(0);

            // Update cost
            if let Some(cost) = metrics.cost_usd {
                self.total_cost_usd += cost;
            }
        } else {
            self.failed_requests += 1;
        }

        // Store the raw metrics
        self.request_metrics.push(metrics);

        Ok(())
    }

    /// Get the number of collected metrics
    fn len(&self) -> usize {
        self.request_metrics.len()
    }

    /// Check if the collector is empty
    fn is_empty(&self) -> bool {
        self.request_metrics.is_empty()
    }

    /// Clear all collected metrics
    fn clear(&mut self) -> Result<(), MetricsError> {
        self.request_metrics.clear();
        self.provider_counts.clear();
        self.model_counts.clear();
        self.successful_requests = 0;
        self.failed_requests = 0;
        self.total_input_tokens = 0;
        self.total_output_tokens = 0;
        self.total_thinking_tokens = 0;
        self.total_cost_usd = 0.0;

        // Reset histograms
        self.global_histograms = HistogramSet::new(&self.config)?;
        self.provider_histograms.clear();
        self.model_histograms.clear();

        Ok(())
    }
}

/// Thread-safe metrics collector
///
/// Collects metrics from multiple requests and provides aggregation
/// functionality using HDR Histogram for accurate percentile calculations.
///
/// # Thread Safety
///
/// This collector is thread-safe and can be safely shared across multiple
/// threads using `Arc<MetricsCollector>`.
///
/// # Example
///
/// ```no_run
/// use llm_latency_lens_metrics::{MetricsCollector, CollectorConfig};
/// use llm_latency_lens_core::SessionId;
///
/// let collector = MetricsCollector::new(
///     SessionId::new(),
///     CollectorConfig::default()
/// ).unwrap();
///
/// // Collect metrics from requests...
/// // let metrics = ...;
/// // collector.record(metrics).unwrap();
///
/// // Get aggregated results
/// let aggregated = collector.aggregate().unwrap();
/// println!("Mean TTFT: {:?}", aggregated.ttft_distribution.mean);
/// ```
#[derive(Clone)]
pub struct MetricsCollector {
    state: Arc<Mutex<CollectorState>>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    ///
    /// # Arguments
    ///
    /// * `session_id` - Unique identifier for this collection session
    /// * `config` - Collector configuration
    ///
    /// # Errors
    ///
    /// Returns an error if histogram initialization fails
    pub fn new(session_id: SessionId, config: CollectorConfig) -> Result<Self, MetricsError> {
        let state = CollectorState::new(session_id, config)?;
        Ok(Self {
            state: Arc::new(Mutex::new(state)),
        })
    }

    /// Create a new metrics collector with default configuration
    pub fn with_defaults(session_id: SessionId) -> Result<Self, MetricsError> {
        Self::new(session_id, CollectorConfig::default())
    }

    /// Record a new request's metrics
    ///
    /// # Arguments
    ///
    /// * `metrics` - The request metrics to record
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The lock cannot be acquired
    /// - Recording into histograms fails
    pub fn record(&self, metrics: RequestMetrics) -> Result<(), MetricsError> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| MetricsError::LockError(e.to_string()))?;

        debug!(
            request_id = %metrics.request_id,
            provider = %metrics.provider,
            model = %metrics.model,
            success = metrics.success,
            "Recording request metrics"
        );

        state.record(metrics)?;
        Ok(())
    }

    /// Get a specific request's metrics by ID
    ///
    /// # Arguments
    ///
    /// * `request_id` - The ID of the request to retrieve
    ///
    /// # Returns
    ///
    /// The request metrics if found, None otherwise
    pub fn get_request(&self, request_id: RequestId) -> Result<Option<RequestMetrics>, MetricsError> {
        let state = self
            .state
            .lock()
            .map_err(|e| MetricsError::LockError(e.to_string()))?;

        Ok(state
            .request_metrics
            .iter()
            .find(|m| m.request_id == request_id)
            .cloned())
    }

    /// Get all collected request metrics
    ///
    /// # Returns
    ///
    /// A vector of all collected request metrics
    pub fn get_all_requests(&self) -> Result<Vec<RequestMetrics>, MetricsError> {
        let state = self
            .state
            .lock()
            .map_err(|e| MetricsError::LockError(e.to_string()))?;

        Ok(state.request_metrics.clone())
    }

    /// Get the number of collected metrics
    pub fn len(&self) -> Result<usize, MetricsError> {
        let state = self
            .state
            .lock()
            .map_err(|e| MetricsError::LockError(e.to_string()))?;

        Ok(state.len())
    }

    /// Check if the collector is empty
    pub fn is_empty(&self) -> Result<bool, MetricsError> {
        let state = self
            .state
            .lock()
            .map_err(|e| MetricsError::LockError(e.to_string()))?;

        Ok(state.is_empty())
    }

    /// Get the session ID
    pub fn session_id(&self) -> Result<SessionId, MetricsError> {
        let state = self
            .state
            .lock()
            .map_err(|e| MetricsError::LockError(e.to_string()))?;

        Ok(state.session_id)
    }

    /// Clear all collected metrics
    ///
    /// This resets all histograms and clears all stored request metrics
    pub fn clear(&self) -> Result<(), MetricsError> {
        let mut state = self
            .state
            .lock()
            .map_err(|e| MetricsError::LockError(e.to_string()))?;

        debug!("Clearing all collected metrics");
        state.clear()
    }

    /// Get internal state snapshot for aggregation
    ///
    /// This is an internal method used by the aggregator
    #[doc(hidden)]
    pub fn get_state_snapshot(&self) -> Result<CollectorStateSnapshot, MetricsError> {
        let state = self
            .state
            .lock()
            .map_err(|e| MetricsError::LockError(e.to_string()))?;

        Ok(CollectorStateSnapshot {
            session_id: state.session_id,
            request_metrics: state.request_metrics.clone(),
            provider_counts: state.provider_counts.clone(),
            model_counts: state.model_counts.clone(),
            successful_requests: state.successful_requests,
            failed_requests: state.failed_requests,
            total_input_tokens: state.total_input_tokens,
            total_output_tokens: state.total_output_tokens,
            total_thinking_tokens: state.total_thinking_tokens,
            total_cost_usd: state.total_cost_usd,
            global_histograms: state.global_histograms.clone(),
        })
    }
}

/// Snapshot of collector state for aggregation
#[doc(hidden)]
#[derive(Clone)]
pub struct CollectorStateSnapshot {
    pub session_id: SessionId,
    pub request_metrics: Vec<RequestMetrics>,
    pub provider_counts: HashMap<Provider, u64>,
    pub model_counts: HashMap<String, u64>,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_thinking_tokens: u64,
    pub total_cost_usd: f64,
    pub global_histograms: HistogramSet,
}

/// Errors that can occur during metrics collection
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    /// Failed to create histogram
    #[error("Failed to create histogram: {0}")]
    HistogramCreation(String),

    /// Failed to record value in histogram
    #[error("Failed to record value in histogram: {0}")]
    HistogramRecord(String),

    /// Lock acquisition failed
    #[error("Failed to acquire lock: {0}")]
    LockError(String),

    /// No metrics available for aggregation
    #[error("No metrics available for aggregation")]
    NoMetrics,

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use llm_latency_lens_core::{Provider, RequestId, SessionId};
    use std::time::Duration;

    fn create_test_metrics(
        provider: Provider,
        model: &str,
        ttft_ms: u64,
        total_ms: u64,
        success: bool,
    ) -> RequestMetrics {
        RequestMetrics {
            request_id: RequestId::new(),
            session_id: SessionId::new(),
            provider,
            model: model.to_string(),
            timestamp: Utc::now(),
            ttft: Duration::from_millis(ttft_ms),
            total_latency: Duration::from_millis(total_ms),
            inter_token_latencies: vec![
                Duration::from_millis(10),
                Duration::from_millis(15),
                Duration::from_millis(12),
            ],
            input_tokens: 100,
            output_tokens: 50,
            thinking_tokens: None,
            tokens_per_second: 50.0,
            cost_usd: Some(0.05),
            success,
            error: if success { None } else { Some("Test error".to_string()) },
        }
    }

    #[test]
    fn test_collector_creation() {
        let collector = MetricsCollector::with_defaults(SessionId::new()).unwrap();
        assert!(collector.is_empty().unwrap());
        assert_eq!(collector.len().unwrap(), 0);
    }

    #[test]
    fn test_record_metrics() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        let metrics = create_test_metrics(Provider::OpenAI, "gpt-4", 100, 1000, true);
        collector.record(metrics).unwrap();

        assert_eq!(collector.len().unwrap(), 1);
        assert!(!collector.is_empty().unwrap());
    }

    #[test]
    fn test_record_multiple_metrics() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        for i in 0..10 {
            let metrics = create_test_metrics(Provider::OpenAI, "gpt-4", 100 + i, 1000 + i, true);
            collector.record(metrics).unwrap();
        }

        assert_eq!(collector.len().unwrap(), 10);
    }

    #[test]
    fn test_record_failed_request() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        let metrics = create_test_metrics(Provider::Anthropic, "claude-3-opus", 100, 1000, false);
        collector.record(metrics).unwrap();

        assert_eq!(collector.len().unwrap(), 1);
    }

    #[test]
    fn test_get_request() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        let metrics = create_test_metrics(Provider::Google, "gemini-pro", 100, 1000, true);
        let request_id = metrics.request_id;

        collector.record(metrics).unwrap();

        let retrieved = collector.get_request(request_id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().request_id, request_id);
    }

    #[test]
    fn test_get_all_requests() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        for i in 0..5 {
            let metrics = create_test_metrics(Provider::OpenAI, "gpt-4", 100 + i, 1000 + i, true);
            collector.record(metrics).unwrap();
        }

        let all_metrics = collector.get_all_requests().unwrap();
        assert_eq!(all_metrics.len(), 5);
    }

    #[test]
    fn test_clear_metrics() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        for i in 0..5 {
            let metrics = create_test_metrics(Provider::OpenAI, "gpt-4", 100 + i, 1000 + i, true);
            collector.record(metrics).unwrap();
        }

        assert_eq!(collector.len().unwrap(), 5);

        collector.clear().unwrap();
        assert_eq!(collector.len().unwrap(), 0);
        assert!(collector.is_empty().unwrap());
    }

    #[test]
    fn test_collector_config() {
        let config = CollectorConfig::new()
            .with_max_value_seconds(120)
            .with_significant_digits(4)
            .with_per_provider_tracking(true)
            .with_per_model_tracking(false);

        assert_eq!(config.max_value_nanos, 120_000_000_000);
        assert_eq!(config.significant_digits, 4);
        assert!(config.track_per_provider);
        assert!(!config.track_per_model);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let session_id = SessionId::new();
        let collector = Arc::new(MetricsCollector::with_defaults(session_id).unwrap());

        let mut handles = vec![];

        for i in 0..10 {
            let collector_clone = Arc::clone(&collector);
            let handle = thread::spawn(move || {
                let metrics = create_test_metrics(
                    Provider::OpenAI,
                    "gpt-4",
                    100 + i,
                    1000 + i,
                    true,
                );
                collector_clone.record(metrics).unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(collector.len().unwrap(), 10);
    }
}
