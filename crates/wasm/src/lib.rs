//! WebAssembly bindings for LLM Latency Lens
//!
//! This crate provides WASM bindings for the LLM Latency Lens profiler,
//! allowing JavaScript/TypeScript applications to measure LLM performance
//! metrics with high precision.

use chrono::Utc;
use llm_latency_lens_core::{Provider, RequestId, SessionId};
use llm_latency_lens_metrics::{
    AggregatedMetrics, CollectorConfig, LatencyDistribution, MetricsCollector, RequestMetrics,
    ThroughputStats,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use wasm_bindgen::prelude::*;

// ============================================================================
// Error Handling
// ============================================================================

/// WASM-compatible error type
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmError {
    message: String,
}

#[wasm_bindgen]
impl WasmError {
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }
}

impl From<String> for WasmError {
    fn from(msg: String) -> Self {
        Self { message: msg }
    }
}

impl From<&str> for WasmError {
    fn from(msg: &str) -> Self {
        Self {
            message: msg.to_string(),
        }
    }
}

// ============================================================================
// Core Types
// ============================================================================

/// JavaScript-friendly provider enum
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum JsProvider {
    OpenAI,
    Anthropic,
    Google,
    AwsBedrock,
    AzureOpenAI,
    Generic,
}

impl From<JsProvider> for Provider {
    fn from(p: JsProvider) -> Self {
        match p {
            JsProvider::OpenAI => Provider::OpenAI,
            JsProvider::Anthropic => Provider::Anthropic,
            JsProvider::Google => Provider::Google,
            JsProvider::AwsBedrock => Provider::AwsBedrock,
            JsProvider::AzureOpenAI => Provider::AzureOpenAI,
            JsProvider::Generic => Provider::Generic,
        }
    }
}

impl From<Provider> for JsProvider {
    fn from(p: Provider) -> Self {
        match p {
            Provider::OpenAI => JsProvider::OpenAI,
            Provider::Anthropic => JsProvider::Anthropic,
            Provider::Google => JsProvider::Google,
            Provider::AwsBedrock => JsProvider::AwsBedrock,
            Provider::AzureOpenAI => JsProvider::AzureOpenAI,
            Provider::Generic => JsProvider::Generic,
        }
    }
}

// ============================================================================
// Metrics Collector
// ============================================================================

/// High-precision metrics collector for LLM requests
///
/// This is the main interface for collecting and aggregating latency metrics
/// in JavaScript/TypeScript applications.
#[wasm_bindgen]
pub struct LatencyCollector {
    inner: Arc<Mutex<MetricsCollector>>,
    session_id: SessionId,
}

#[wasm_bindgen]
impl LatencyCollector {
    /// Create a new metrics collector with default configuration
    ///
    /// # Example (JavaScript)
    /// ```js
    /// const collector = LatencyCollector.new();
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<LatencyCollector, WasmError> {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id)
            .map_err(|e| WasmError::from(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(Mutex::new(collector)),
            session_id,
        })
    }

    /// Create a new metrics collector with custom configuration
    ///
    /// # Parameters
    /// - `max_value_seconds`: Maximum latency value to track (default: 60)
    /// - `significant_digits`: Precision for percentile calculations (1-5, default: 3)
    ///
    /// # Example (JavaScript)
    /// ```js
    /// const collector = LatencyCollector.with_config(120, 3);
    /// ```
    #[wasm_bindgen]
    pub fn with_config(
        max_value_seconds: u64,
        significant_digits: u8,
    ) -> Result<LatencyCollector, WasmError> {
        let session_id = SessionId::new();
        let config = CollectorConfig::new()
            .with_max_value_seconds(max_value_seconds)
            .with_significant_digits(significant_digits);

        let collector = MetricsCollector::new(session_id, config)
            .map_err(|e| WasmError::from(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(Mutex::new(collector)),
            session_id,
        })
    }

    /// Get the session ID for this collector
    ///
    /// # Example (JavaScript)
    /// ```js
    /// const sessionId = collector.session_id();
    /// console.log('Session:', sessionId);
    /// ```
    #[wasm_bindgen]
    pub fn session_id(&self) -> String {
        self.session_id.to_string()
    }

    /// Record metrics from a completed request
    ///
    /// # Parameters
    /// - `metrics`: JavaScript object containing request metrics
    ///
    /// # Example (JavaScript)
    /// ```js
    /// collector.record({
    ///   provider: 'openai',
    ///   model: 'gpt-4',
    ///   ttft_ms: 150,
    ///   total_latency_ms: 2000,
    ///   inter_token_latencies_ms: [10, 15, 12, 11],
    ///   input_tokens: 100,
    ///   output_tokens: 50,
    ///   tokens_per_second: 25.0,
    ///   cost_usd: 0.05,
    ///   success: true
    /// });
    /// ```
    #[wasm_bindgen]
    pub fn record(&self, metrics: JsValue) -> Result<(), WasmError> {
        let js_metrics: JsRequestMetrics = serde_wasm_bindgen::from_value(metrics)
            .map_err(|e| WasmError::from(format!("Failed to parse metrics: {}", e)))?;

        let request_metrics = js_metrics.to_request_metrics(self.session_id)?;

        self.inner
            .lock()
            .map_err(|e| WasmError::from(format!("Lock error: {}", e)))?
            .record(request_metrics)
            .map_err(|e| WasmError::from(e.to_string()))?;

        Ok(())
    }

    /// Get the number of metrics recorded
    ///
    /// # Example (JavaScript)
    /// ```js
    /// const count = collector.len();
    /// console.log(`Recorded ${count} requests`);
    /// ```
    #[wasm_bindgen]
    pub fn len(&self) -> Result<usize, WasmError> {
        self.inner
            .lock()
            .map_err(|e| WasmError::from(format!("Lock error: {}", e)))?
            .len()
            .map_err(|e| WasmError::from(e.to_string()))
    }

    /// Check if the collector is empty
    #[wasm_bindgen]
    pub fn is_empty(&self) -> Result<bool, WasmError> {
        self.inner
            .lock()
            .map_err(|e| WasmError::from(format!("Lock error: {}", e)))?
            .is_empty()
            .map_err(|e| WasmError::from(e.to_string()))
    }

    /// Clear all collected metrics
    ///
    /// # Example (JavaScript)
    /// ```js
    /// collector.clear();
    /// ```
    #[wasm_bindgen]
    pub fn clear(&self) -> Result<(), WasmError> {
        self.inner
            .lock()
            .map_err(|e| WasmError::from(format!("Lock error: {}", e)))?
            .clear()
            .map_err(|e| WasmError::from(e.to_string()))
    }

    /// Aggregate all metrics and return summary statistics
    ///
    /// Returns a JavaScript object with percentile distributions and summary stats.
    ///
    /// # Example (JavaScript)
    /// ```js
    /// const stats = collector.aggregate();
    /// console.log('TTFT p99:', stats.ttft_distribution.p99, 'ms');
    /// console.log('Success rate:', stats.success_rate, '%');
    /// ```
    #[wasm_bindgen]
    pub fn aggregate(&self) -> Result<JsValue, WasmError> {
        use llm_latency_lens_metrics::MetricsAggregator;

        let collector = self
            .inner
            .lock()
            .map_err(|e| WasmError::from(format!("Lock error: {}", e)))?;

        let aggregated = MetricsAggregator::aggregate(&*collector)
            .map_err(|e| WasmError::from(e.to_string()))?;

        let js_aggregated = JsAggregatedMetrics::from(aggregated);

        serde_wasm_bindgen::to_value(&js_aggregated)
            .map_err(|e| WasmError::from(format!("Serialization error: {}", e)))
    }

    /// Aggregate metrics for a specific provider
    ///
    /// # Example (JavaScript)
    /// ```js
    /// const openaiStats = collector.aggregate_by_provider('openai');
    /// ```
    #[wasm_bindgen]
    pub fn aggregate_by_provider(&self, provider: &str) -> Result<JsValue, WasmError> {
        use llm_latency_lens_metrics::MetricsAggregator;

        let provider_enum = match provider.to_lowercase().as_str() {
            "openai" => Provider::OpenAI,
            "anthropic" => Provider::Anthropic,
            "google" => Provider::Google,
            "aws-bedrock" | "awsbedrock" => Provider::AwsBedrock,
            "azure-openai" | "azureopenai" => Provider::AzureOpenAI,
            "generic" => Provider::Generic,
            _ => return Err(WasmError::from("Invalid provider name")),
        };

        let collector = self
            .inner
            .lock()
            .map_err(|e| WasmError::from(format!("Lock error: {}", e)))?;

        let aggregated = MetricsAggregator::aggregate_by_provider(&*collector, provider_enum)
            .map_err(|e| WasmError::from(e.to_string()))?;

        let js_aggregated = JsAggregatedMetrics::from(aggregated);

        serde_wasm_bindgen::to_value(&js_aggregated)
            .map_err(|e| WasmError::from(format!("Serialization error: {}", e)))
    }

    /// Aggregate metrics for a specific model
    ///
    /// # Example (JavaScript)
    /// ```js
    /// const gpt4Stats = collector.aggregate_by_model('gpt-4');
    /// ```
    #[wasm_bindgen]
    pub fn aggregate_by_model(&self, model: &str) -> Result<JsValue, WasmError> {
        use llm_latency_lens_metrics::MetricsAggregator;

        let collector = self
            .inner
            .lock()
            .map_err(|e| WasmError::from(format!("Lock error: {}", e)))?;

        let aggregated = MetricsAggregator::aggregate_by_model(&*collector, model)
            .map_err(|e| WasmError::from(e.to_string()))?;

        let js_aggregated = JsAggregatedMetrics::from(aggregated);

        serde_wasm_bindgen::to_value(&js_aggregated)
            .map_err(|e| WasmError::from(format!("Serialization error: {}", e)))
    }
}

// ============================================================================
// JavaScript-friendly data structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsRequestMetrics {
    provider: String,
    model: String,
    ttft_ms: f64,
    total_latency_ms: f64,
    inter_token_latencies_ms: Vec<f64>,
    input_tokens: u64,
    output_tokens: u64,
    thinking_tokens: Option<u64>,
    tokens_per_second: f64,
    cost_usd: Option<f64>,
    success: bool,
    error: Option<String>,
}

impl JsRequestMetrics {
    fn to_request_metrics(&self, session_id: SessionId) -> Result<RequestMetrics, WasmError> {
        let provider = match self.provider.to_lowercase().as_str() {
            "openai" => Provider::OpenAI,
            "anthropic" => Provider::Anthropic,
            "google" => Provider::Google,
            "aws-bedrock" | "awsbedrock" => Provider::AwsBedrock,
            "azure-openai" | "azureopenai" => Provider::AzureOpenAI,
            "generic" => Provider::Generic,
            _ => return Err(WasmError::from("Invalid provider name")),
        };

        Ok(RequestMetrics {
            request_id: RequestId::new(),
            session_id,
            provider,
            model: self.model.clone(),
            timestamp: Utc::now(),
            ttft: Duration::from_millis(self.ttft_ms as u64),
            total_latency: Duration::from_millis(self.total_latency_ms as u64),
            inter_token_latencies: self
                .inter_token_latencies_ms
                .iter()
                .map(|&ms| Duration::from_millis(ms as u64))
                .collect(),
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            thinking_tokens: self.thinking_tokens,
            tokens_per_second: self.tokens_per_second,
            cost_usd: self.cost_usd,
            success: self.success,
            error: self.error.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsAggregatedMetrics {
    session_id: String,
    start_time: String,
    end_time: String,
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    success_rate: f64,
    ttft_distribution: JsLatencyDistribution,
    inter_token_distribution: JsLatencyDistribution,
    total_latency_distribution: JsLatencyDistribution,
    throughput: JsThroughputStats,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_thinking_tokens: Option<u64>,
    total_cost_usd: Option<f64>,
    avg_cost_per_request: Option<f64>,
    provider_breakdown: Vec<(String, u64)>,
    model_breakdown: Vec<(String, u64)>,
}

impl From<AggregatedMetrics> for JsAggregatedMetrics {
    fn from(m: AggregatedMetrics) -> Self {
        // Calculate values that need borrowing before moving fields
        let success_rate = m.success_rate();
        let avg_cost_per_request = m.avg_cost_per_request();

        Self {
            session_id: m.session_id.to_string(),
            start_time: m.start_time.to_rfc3339(),
            end_time: m.end_time.to_rfc3339(),
            total_requests: m.total_requests,
            successful_requests: m.successful_requests,
            failed_requests: m.failed_requests,
            success_rate,
            ttft_distribution: JsLatencyDistribution::from(m.ttft_distribution),
            inter_token_distribution: JsLatencyDistribution::from(m.inter_token_distribution),
            total_latency_distribution: JsLatencyDistribution::from(m.total_latency_distribution),
            throughput: JsThroughputStats::from(m.throughput),
            total_input_tokens: m.total_input_tokens,
            total_output_tokens: m.total_output_tokens,
            total_thinking_tokens: m.total_thinking_tokens,
            total_cost_usd: m.total_cost_usd,
            avg_cost_per_request,
            provider_breakdown: m
                .provider_breakdown
                .into_iter()
                .map(|(p, c)| (p.to_string(), c))
                .collect(),
            model_breakdown: m.model_breakdown,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsLatencyDistribution {
    min_ms: f64,
    max_ms: f64,
    mean_ms: f64,
    std_dev_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    p99_9_ms: f64,
    sample_count: u64,
}

impl From<LatencyDistribution> for JsLatencyDistribution {
    fn from(d: LatencyDistribution) -> Self {
        Self {
            min_ms: d.min.as_secs_f64() * 1000.0,
            max_ms: d.max.as_secs_f64() * 1000.0,
            mean_ms: d.mean.as_secs_f64() * 1000.0,
            std_dev_ms: d.std_dev.as_secs_f64() * 1000.0,
            p50_ms: d.p50.as_secs_f64() * 1000.0,
            p90_ms: d.p90.as_secs_f64() * 1000.0,
            p95_ms: d.p95.as_secs_f64() * 1000.0,
            p99_ms: d.p99.as_secs_f64() * 1000.0,
            p99_9_ms: d.p99_9.as_secs_f64() * 1000.0,
            sample_count: d.sample_count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsThroughputStats {
    mean_tokens_per_second: f64,
    min_tokens_per_second: f64,
    max_tokens_per_second: f64,
    std_dev_tokens_per_second: f64,
    p50_tokens_per_second: f64,
    p95_tokens_per_second: f64,
    p99_tokens_per_second: f64,
}

impl From<ThroughputStats> for JsThroughputStats {
    fn from(t: ThroughputStats) -> Self {
        Self {
            mean_tokens_per_second: t.mean_tokens_per_second,
            min_tokens_per_second: t.min_tokens_per_second,
            max_tokens_per_second: t.max_tokens_per_second,
            std_dev_tokens_per_second: t.std_dev_tokens_per_second,
            p50_tokens_per_second: t.p50_tokens_per_second,
            p95_tokens_per_second: t.p95_tokens_per_second,
            p99_tokens_per_second: t.p99_tokens_per_second,
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Initialize the WASM module (sets up panic hook for better error messages)
///
/// Call this once when your application starts.
///
/// # Example (JavaScript)
/// ```js
/// import { init_wasm } from '@llm-devops/latency-lens';
/// init_wasm();
/// ```
#[wasm_bindgen]
pub fn init_wasm() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get the library version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_collector_creation() {
        let collector = LatencyCollector::new().unwrap();
        assert!(collector.is_empty().unwrap());
        assert_eq!(collector.len().unwrap(), 0);
    }

    #[wasm_bindgen_test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
    }
}
