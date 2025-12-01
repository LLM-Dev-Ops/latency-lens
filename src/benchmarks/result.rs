//! Canonical BenchmarkResult struct for standardized benchmark output
//!
//! This module provides the standardized BenchmarkResult type required for
//! compatibility with the canonical benchmark interface across all 25
//! benchmark-target repositories.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Canonical benchmark result structure
///
/// This struct follows the standardized interface required across all
/// benchmark-target repositories, containing:
/// - `target_id`: Unique identifier for the benchmark target
/// - `metrics`: JSON object containing benchmark metrics
/// - `timestamp`: UTC timestamp when the benchmark was executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Unique identifier for the benchmark target
    /// Format: "{provider}:{model}" or custom identifier
    pub target_id: String,

    /// JSON object containing benchmark metrics
    /// Includes TTFT, latency distributions, throughput, etc.
    pub metrics: Value,

    /// UTC timestamp when the benchmark was executed
    pub timestamp: DateTime<Utc>,
}

impl BenchmarkResult {
    /// Create a new BenchmarkResult
    pub fn new(target_id: impl Into<String>, metrics: Value) -> Self {
        Self {
            target_id: target_id.into(),
            metrics,
            timestamp: Utc::now(),
        }
    }

    /// Create a BenchmarkResult with a specific timestamp
    pub fn with_timestamp(
        target_id: impl Into<String>,
        metrics: Value,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            target_id: target_id.into(),
            metrics,
            timestamp,
        }
    }

    /// Convert from AggregatedMetrics to BenchmarkResult
    pub fn from_aggregated_metrics(
        target_id: impl Into<String>,
        metrics: &llm_latency_lens_metrics::AggregatedMetrics,
    ) -> Result<Self, serde_json::Error> {
        let metrics_value = serde_json::to_value(metrics)?;
        Ok(Self {
            target_id: target_id.into(),
            metrics: metrics_value,
            timestamp: metrics.end_time,
        })
    }

    /// Get the target ID
    pub fn target_id(&self) -> &str {
        &self.target_id
    }

    /// Get the metrics as a reference
    pub fn metrics(&self) -> &Value {
        &self.metrics
    }

    /// Get the timestamp
    pub fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    /// Extract a specific metric value by path (e.g., "ttft_distribution.mean")
    pub fn get_metric(&self, path: &str) -> Option<&Value> {
        let mut current = &self.metrics;
        for key in path.split('.') {
            current = current.get(key)?;
        }
        Some(current)
    }

    /// Check if the benchmark was successful
    pub fn is_success(&self) -> bool {
        self.metrics
            .get("successful_requests")
            .and_then(|v| v.as_u64())
            .map(|s| s > 0)
            .unwrap_or(false)
    }

    /// Get success rate as percentage
    pub fn success_rate(&self) -> Option<f64> {
        let total = self.metrics.get("total_requests")?.as_u64()?;
        let successful = self.metrics.get("successful_requests")?.as_u64()?;
        if total == 0 {
            return Some(0.0);
        }
        Some((successful as f64 / total as f64) * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_benchmark_result_creation() {
        let metrics = json!({
            "total_requests": 100,
            "successful_requests": 95,
            "ttft_distribution": {
                "mean": 150000000,
                "p50": 140000000,
                "p95": 200000000,
                "p99": 250000000
            }
        });

        let result = BenchmarkResult::new("openai:gpt-4o", metrics);

        assert_eq!(result.target_id(), "openai:gpt-4o");
        assert!(result.is_success());
        assert_eq!(result.success_rate(), Some(95.0));
    }

    #[test]
    fn test_get_metric() {
        let metrics = json!({
            "ttft_distribution": {
                "mean": 150000000,
                "p95": 200000000
            }
        });

        let result = BenchmarkResult::new("test", metrics);

        assert_eq!(
            result.get_metric("ttft_distribution.mean"),
            Some(&json!(150000000))
        );
        assert_eq!(
            result.get_metric("ttft_distribution.p95"),
            Some(&json!(200000000))
        );
        assert!(result.get_metric("nonexistent").is_none());
    }

    #[test]
    fn test_serialization() {
        let metrics = json!({"test": "value"});
        let result = BenchmarkResult::new("test:target", metrics);

        let json_str = serde_json::to_string(&result).unwrap();
        let deserialized: BenchmarkResult = serde_json::from_str(&json_str).unwrap();

        assert_eq!(deserialized.target_id(), "test:target");
    }
}
