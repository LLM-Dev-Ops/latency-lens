//! Upstream Data Consumers for LLM Latency Lens
//!
//! This module provides thin adapter interfaces for consuming data from upstream
//! LLM-Dev-Ops ecosystem repositories. These adapters read and merge external data
//! sources into Latency-Lens's profiling pipeline without altering existing APIs.
//!
//! # Supported Data Sources
//!
//! - **LLM-Observatory**: Telemetry streams, timing spans, request/response traces
//! - **LLM-Analytics-Hub**: Historical baselines, p95/p99 summaries, throughput aggregates
//! - **LLM-Test-Bench** (optional file reader): Benchmark output files (JSON/CSV)
//!
//! # Architecture
//!
//! All consumers implement the [`DataConsumer`] trait, which provides a unified
//! interface for reading external data and converting it to Latency-Lens metrics.
//!
//! ```text
//! ┌─────────────────────┐    ┌─────────────────────┐
//! │  LLM-Observatory    │    │  LLM-Analytics-Hub  │
//! │  (telemetry spans)  │    │  (baselines, p95)   │
//! └──────────┬──────────┘    └──────────┬──────────┘
//!            │                          │
//!            ▼                          ▼
//! ┌──────────────────────────────────────────────────┐
//! │              DataConsumer Trait                  │
//! │  - consume_spans() -> RequestMetrics            │
//! │  - consume_baselines() -> AggregatedMetrics     │
//! └──────────────────────────────────────────────────┘
//!                        │
//!                        ▼
//! ┌──────────────────────────────────────────────────┐
//! │          Latency-Lens Pipeline                   │
//! │  MetricsCollector → MetricsAggregator → Export   │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! # Example Usage
//!
//! ```no_run
//! use llm_latency_lens::consumers::{
//!     ObservatoryConsumer, AnalyticsHubConsumer, TestBenchReader,
//! };
//! use llm_latency_lens::MetricsCollector;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Consume telemetry from LLM-Observatory
//! let observatory = ObservatoryConsumer::new();
//! let spans = observatory.consume_latest_spans(100).await?;
//!
//! // Consume baselines from LLM-Analytics-Hub
//! let analytics = AnalyticsHubConsumer::new();
//! let baselines = analytics.get_historical_baseline("openai", "gpt-4").await?;
//!
//! // Read benchmark files from LLM-Test-Bench (no dependency required)
//! let reader = TestBenchReader::new();
//! let metrics = reader.read_json_file("benchmarks.json")?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "upstream")]
pub mod analytics_hub;
#[cfg(feature = "upstream")]
pub mod observatory;
pub mod testbench;

// Re-export consumer types
#[cfg(feature = "upstream")]
pub use analytics_hub::{AnalyticsHubConsumer, AnalyticsHubConfig, BaselineComparison, HistoricalBaseline, RollingWindow, TimeWindow};
#[cfg(feature = "upstream")]
pub use observatory::{ObservatoryConsumer, ObservatoryConfig, TelemetrySpan, TracedRequest};
pub use testbench::{TestBenchReader, TestBenchFormat, TestBenchMetrics};

use crate::RequestMetrics;
use async_trait::async_trait;
use thiserror::Error;

/// Errors that can occur during data consumption
#[derive(Debug, Error)]
pub enum ConsumerError {
    /// Connection to upstream service failed
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// Data parsing or conversion error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Upstream service returned an error
    #[error("Upstream error: {0}")]
    UpstreamError(String),

    /// File I/O error
    #[error("File I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Timeout waiting for data
    #[error("Timeout waiting for data: {0}")]
    Timeout(String),
}

/// Result type for consumer operations
pub type ConsumerResult<T> = Result<T, ConsumerError>;

/// Trait for consuming data from upstream sources
///
/// This trait provides a unified interface for all upstream data consumers.
/// Implementations convert external data formats to Latency-Lens metrics.
#[async_trait]
pub trait DataConsumer: Send + Sync {
    /// Get the name of this consumer
    fn name(&self) -> &'static str;

    /// Check if the upstream source is available
    async fn health_check(&self) -> ConsumerResult<bool>;

    /// Consume metrics from the upstream source
    ///
    /// Returns a vector of RequestMetrics that can be fed into
    /// the Latency-Lens MetricsCollector.
    async fn consume(&self, limit: usize) -> ConsumerResult<Vec<RequestMetrics>>;
}

/// Configuration for upstream consumer retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds
    pub max_backoff_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Merge multiple data sources into a unified metrics stream
pub struct MergedConsumer {
    consumers: Vec<Box<dyn DataConsumer>>,
}

impl MergedConsumer {
    /// Create a new merged consumer
    pub fn new() -> Self {
        Self {
            consumers: Vec::new(),
        }
    }

    /// Add a consumer to the merge pipeline
    pub fn add_consumer(mut self, consumer: Box<dyn DataConsumer>) -> Self {
        self.consumers.push(consumer);
        self
    }

    /// Consume from all sources and merge results
    pub async fn consume_all(&self, limit_per_source: usize) -> ConsumerResult<Vec<RequestMetrics>> {
        let mut all_metrics = Vec::new();

        for consumer in &self.consumers {
            match consumer.consume(limit_per_source).await {
                Ok(metrics) => all_metrics.extend(metrics),
                Err(e) => {
                    tracing::warn!(
                        consumer = consumer.name(),
                        error = %e,
                        "Failed to consume from source, continuing with others"
                    );
                }
            }
        }

        // Sort by timestamp for consistent ordering
        all_metrics.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        Ok(all_metrics)
    }

    /// Get health status of all consumers
    pub async fn health_check_all(&self) -> Vec<(&'static str, bool)> {
        let mut results = Vec::new();

        for consumer in &self.consumers {
            let healthy = consumer.health_check().await.unwrap_or(false);
            results.push((consumer.name(), healthy));
        }

        results
    }
}

impl Default for MergedConsumer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_defaults() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_backoff_ms, 100);
    }

    #[test]
    fn test_merged_consumer_creation() {
        let merged = MergedConsumer::new();
        assert!(merged.consumers.is_empty());
    }
}
