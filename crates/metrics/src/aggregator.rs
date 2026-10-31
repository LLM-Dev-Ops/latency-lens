//! Statistical aggregation of collected metrics
//!
//! Provides functionality to aggregate metrics from a collector into
//! statistical distributions with percentile calculations.

use crate::collector::{MetricsCollector, MetricsError};
use crate::types::{AggregatedMetrics, LatencyDistribution, ThroughputStats};
use hdrhistogram::Histogram;
use llm_latency_lens_core::Provider;
use std::collections::HashMap;
use std::time::Duration;
use tracing::debug;

/// Aggregates metrics from a collector into statistical distributions
///
/// Uses the collected histograms to calculate accurate percentiles and
/// other statistical measures.
pub struct MetricsAggregator;

impl MetricsAggregator {
    /// Aggregate metrics from a collector
    ///
    /// # Arguments
    ///
    /// * `collector` - The metrics collector to aggregate from
    ///
    /// # Returns
    ///
    /// An `AggregatedMetrics` struct containing all statistical aggregations
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The collector has no metrics
    /// - Lock acquisition fails
    /// - Histogram calculations fail
    pub fn aggregate(collector: &MetricsCollector) -> Result<AggregatedMetrics, MetricsError> {
        let snapshot = collector.get_state_snapshot()?;

        if snapshot.request_metrics.is_empty() {
            return Err(MetricsError::NoMetrics);
        }

        debug!(
            total_requests = snapshot.request_metrics.len(),
            successful = snapshot.successful_requests,
            failed = snapshot.failed_requests,
            "Aggregating metrics"
        );

        // Calculate time range
        let start_time = snapshot
            .request_metrics
            .iter()
            .map(|m| m.timestamp)
            .min()
            .unwrap();

        let end_time = snapshot
            .request_metrics
            .iter()
            .map(|m| m.timestamp)
            .max()
            .unwrap();

        // Calculate distributions from histograms
        let ttft_distribution = Self::calculate_latency_distribution(&snapshot.global_histograms.ttft)?;
        let inter_token_distribution =
            Self::calculate_latency_distribution(&snapshot.global_histograms.inter_token)?;
        let total_latency_distribution =
            Self::calculate_latency_distribution(&snapshot.global_histograms.total_latency)?;

        // Calculate throughput statistics
        let throughput = Self::calculate_throughput_stats(&snapshot.global_histograms.throughput)?;

        // Build provider and model breakdowns
        let mut provider_breakdown: Vec<(Provider, u64)> =
            snapshot.provider_counts.into_iter().collect();
        let mut model_breakdown: Vec<(String, u64)> = snapshot.model_counts.into_iter().collect();
        provider_breakdown.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));
        model_breakdown.sort_by(|(a, _), (b, _)| a.cmp(b));

        // Calculate total cost
        let total_cost_usd = if snapshot.total_cost_usd > 0.0 {
            Some(snapshot.total_cost_usd)
        } else {
            None
        };

        // Calculate total thinking tokens
        let total_thinking_tokens = if snapshot.total_thinking_tokens > 0 {
            Some(snapshot.total_thinking_tokens)
        } else {
            None
        };

        Ok(AggregatedMetrics {
            session_id: snapshot.session_id,
            start_time,
            end_time,
            total_requests: snapshot.request_metrics.len() as u64,
            successful_requests: snapshot.successful_requests,
            failed_requests: snapshot.failed_requests,
            ttft_distribution,
            inter_token_distribution,
            total_latency_distribution,
            throughput,
            total_input_tokens: snapshot.total_input_tokens,
            total_output_tokens: snapshot.total_output_tokens,
            total_thinking_tokens,
            total_cost_usd,
            provider_breakdown,
            model_breakdown,
        })
    }

    /// Calculate latency distribution from a histogram
    fn calculate_latency_distribution(
        histogram: &Histogram<u64>,
    ) -> Result<LatencyDistribution, MetricsError> {
        if histogram.is_empty() {
            return Ok(LatencyDistribution::empty());
        }

        let min = Duration::from_nanos(histogram.min());
        let max = Duration::from_nanos(histogram.max());
        let mean = Duration::from_nanos(histogram.mean() as u64);
        let std_dev = Duration::from_nanos(histogram.stdev() as u64);

        let p50 = Duration::from_nanos(histogram.value_at_quantile(0.50));
        let p90 = Duration::from_nanos(histogram.value_at_quantile(0.90));
        let p95 = Duration::from_nanos(histogram.value_at_quantile(0.95));
        let p99 = Duration::from_nanos(histogram.value_at_quantile(0.99));
        let p99_9 = Duration::from_nanos(histogram.value_at_quantile(0.999));

        Ok(LatencyDistribution {
            min,
            max,
            mean,
            std_dev,
            p50,
            p90,
            p95,
            p99,
            p99_9,
            sample_count: histogram.len(),
        })
    }

    /// Calculate throughput statistics from a histogram
    ///
    /// The throughput histogram stores values as tokens/sec * 1000 for precision
    fn calculate_throughput_stats(
        histogram: &Histogram<u64>,
    ) -> Result<ThroughputStats, MetricsError> {
        if histogram.is_empty() {
            return Ok(ThroughputStats::empty());
        }

        // Convert back from scaled values (divide by 1000)
        let mean_tokens_per_second = histogram.mean() / 1000.0;
        let min_tokens_per_second = histogram.min() as f64 / 1000.0;
        let max_tokens_per_second = histogram.max() as f64 / 1000.0;
        let std_dev_tokens_per_second = histogram.stdev() / 1000.0;
        let p50_tokens_per_second = histogram.value_at_quantile(0.50) as f64 / 1000.0;
        let p95_tokens_per_second = histogram.value_at_quantile(0.95) as f64 / 1000.0;
        let p99_tokens_per_second = histogram.value_at_quantile(0.99) as f64 / 1000.0;

        Ok(ThroughputStats {
            mean_tokens_per_second,
            min_tokens_per_second,
            max_tokens_per_second,
            std_dev_tokens_per_second,
            p50_tokens_per_second,
            p95_tokens_per_second,
            p99_tokens_per_second,
        })
    }

    /// Aggregate metrics for a specific provider
    ///
    /// This filters the collector's metrics to only include those from the specified provider
    pub fn aggregate_by_provider(
        collector: &MetricsCollector,
        provider: Provider,
    ) -> Result<AggregatedMetrics, MetricsError> {
        let all_metrics = collector.get_all_requests()?;
        let filtered: Vec<_> = all_metrics
            .into_iter()
            .filter(|m| m.provider == provider)
            .collect();

        if filtered.is_empty() {
            return Err(MetricsError::NoMetrics);
        }

        Self::aggregate_from_metrics(&filtered)
    }

    /// Aggregate metrics for a specific model
    ///
    /// This filters the collector's metrics to only include those from the specified model
    pub fn aggregate_by_model(
        collector: &MetricsCollector,
        model: &str,
    ) -> Result<AggregatedMetrics, MetricsError> {
        let all_metrics = collector.get_all_requests()?;
        let filtered: Vec<_> = all_metrics
            .into_iter()
            .filter(|m| m.model == model)
            .collect();

        if filtered.is_empty() {
            return Err(MetricsError::NoMetrics);
        }

        Self::aggregate_from_metrics(&filtered)
    }

    /// Aggregate metrics from a slice of request metrics
    ///
    /// This is useful for custom filtering scenarios
    fn aggregate_from_metrics(
        metrics: &[crate::types::RequestMetrics],
    ) -> Result<AggregatedMetrics, MetricsError> {
        if metrics.is_empty() {
            return Err(MetricsError::NoMetrics);
        }

        // Create histograms for this subset
        let mut ttft_hist = Histogram::<u64>::new(3)
            .map_err(|e| MetricsError::HistogramCreation(e.to_string()))?;
        let mut inter_token_hist = Histogram::<u64>::new(3)
            .map_err(|e| MetricsError::HistogramCreation(e.to_string()))?;
        let mut total_latency_hist = Histogram::<u64>::new(3)
            .map_err(|e| MetricsError::HistogramCreation(e.to_string()))?;
        let mut throughput_hist = Histogram::<u64>::new(3)
            .map_err(|e| MetricsError::HistogramCreation(e.to_string()))?;

        let mut successful_requests = 0u64;
        let mut failed_requests = 0u64;
        let mut total_input_tokens = 0u64;
        let mut total_output_tokens = 0u64;
        let mut total_thinking_tokens = 0u64;
        let mut total_cost_usd = 0.0f64;

        let mut provider_counts: HashMap<Provider, u64> = HashMap::new();
        let mut model_counts: HashMap<String, u64> = HashMap::new();

        for metric in metrics {
            if metric.success {
                successful_requests += 1;

                // Record into histograms
                ttft_hist
                    .record(metric.ttft.as_nanos() as u64)
                    .map_err(|e| MetricsError::HistogramRecord(e.to_string()))?;

                total_latency_hist
                    .record(metric.total_latency.as_nanos() as u64)
                    .map_err(|e| MetricsError::HistogramRecord(e.to_string()))?;

                for latency in &metric.inter_token_latencies {
                    inter_token_hist
                        .record(latency.as_nanos() as u64)
                        .map_err(|e| MetricsError::HistogramRecord(e.to_string()))?;
                }

                let throughput_scaled = (metric.tokens_per_second * 1000.0) as u64;
                throughput_hist
                    .record(throughput_scaled)
                    .map_err(|e| MetricsError::HistogramRecord(e.to_string()))?;

                // Accumulate tokens
                total_input_tokens += metric.input_tokens;
                total_output_tokens += metric.output_tokens;
                total_thinking_tokens += metric.thinking_tokens.unwrap_or(0);

                // Accumulate cost
                if let Some(cost) = metric.cost_usd {
                    total_cost_usd += cost;
                }
            } else {
                failed_requests += 1;
            }

            // Count providers and models
            *provider_counts.entry(metric.provider).or_insert(0) += 1;
            *model_counts.entry(metric.model.clone()).or_insert(0) += 1;
        }

        // Calculate distributions
        let ttft_distribution = Self::calculate_latency_distribution(&ttft_hist)?;
        let inter_token_distribution = Self::calculate_latency_distribution(&inter_token_hist)?;
        let total_latency_distribution = Self::calculate_latency_distribution(&total_latency_hist)?;
        let throughput = Self::calculate_throughput_stats(&throughput_hist)?;

        // Calculate time range
        let start_time = metrics.iter().map(|m| m.timestamp).min().unwrap();
        let end_time = metrics.iter().map(|m| m.timestamp).max().unwrap();

        // Get session ID from first metric (assumes all have the same session)
        let session_id = metrics[0].session_id;

        let mut provider_breakdown: Vec<(Provider, u64)> = provider_counts.into_iter().collect();
        let mut model_breakdown: Vec<(String, u64)> = model_counts.into_iter().collect();
        provider_breakdown.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));
        model_breakdown.sort_by(|(a, _), (b, _)| a.cmp(b));

        let total_cost_usd_opt = if total_cost_usd > 0.0 {
            Some(total_cost_usd)
        } else {
            None
        };

        let total_thinking_tokens_opt = if total_thinking_tokens > 0 {
            Some(total_thinking_tokens)
        } else {
            None
        };

        Ok(AggregatedMetrics {
            session_id,
            start_time,
            end_time,
            total_requests: metrics.len() as u64,
            successful_requests,
            failed_requests,
            ttft_distribution,
            inter_token_distribution,
            total_latency_distribution,
            throughput,
            total_input_tokens,
            total_output_tokens,
            total_thinking_tokens: total_thinking_tokens_opt,
            total_cost_usd: total_cost_usd_opt,
            provider_breakdown,
            model_breakdown,
        })
    }

    /// Compare two aggregated metrics
    ///
    /// Returns a comparison showing the differences between two metric sets
    /// This is useful for A/B testing or comparing different time periods
    pub fn compare(
        baseline: &AggregatedMetrics,
        comparison: &AggregatedMetrics,
    ) -> MetricsComparison {
        MetricsComparison {
            baseline_session: baseline.session_id,
            comparison_session: comparison.session_id,
            ttft_change: Self::calculate_distribution_change(
                &baseline.ttft_distribution,
                &comparison.ttft_distribution,
            ),
            inter_token_change: Self::calculate_distribution_change(
                &baseline.inter_token_distribution,
                &comparison.inter_token_distribution,
            ),
            total_latency_change: Self::calculate_distribution_change(
                &baseline.total_latency_distribution,
                &comparison.total_latency_distribution,
            ),
            throughput_change: Self::calculate_percentage_change(
                baseline.throughput.mean_tokens_per_second,
                comparison.throughput.mean_tokens_per_second,
            ),
            success_rate_change: Self::calculate_percentage_change(
                baseline.success_rate(),
                comparison.success_rate(),
            ),
            cost_change: match (baseline.total_cost_usd, comparison.total_cost_usd) {
                (Some(b), Some(c)) => Some(Self::calculate_percentage_change(b, c)),
                _ => None,
            },
        }
    }

    /// Calculate percentage change for latency distributions
    fn calculate_distribution_change(
        baseline: &LatencyDistribution,
        comparison: &LatencyDistribution,
    ) -> DistributionChange {
        DistributionChange {
            mean_change: Self::calculate_percentage_change(
                baseline.mean.as_nanos() as f64,
                comparison.mean.as_nanos() as f64,
            ),
            p50_change: Self::calculate_percentage_change(
                baseline.p50.as_nanos() as f64,
                comparison.p50.as_nanos() as f64,
            ),
            p95_change: Self::calculate_percentage_change(
                baseline.p95.as_nanos() as f64,
                comparison.p95.as_nanos() as f64,
            ),
            p99_change: Self::calculate_percentage_change(
                baseline.p99.as_nanos() as f64,
                comparison.p99.as_nanos() as f64,
            ),
        }
    }

    /// Calculate percentage change between two values
    fn calculate_percentage_change(baseline: f64, comparison: f64) -> f64 {
        if baseline == 0.0 {
            return 0.0;
        }
        ((comparison - baseline) / baseline) * 100.0
    }
}

/// Comparison between two sets of aggregated metrics
#[derive(Debug, Clone)]
pub struct MetricsComparison {
    /// Baseline session ID
    pub baseline_session: llm_latency_lens_core::SessionId,

    /// Comparison session ID
    pub comparison_session: llm_latency_lens_core::SessionId,

    /// TTFT distribution change
    pub ttft_change: DistributionChange,

    /// Inter-token latency distribution change
    pub inter_token_change: DistributionChange,

    /// Total latency distribution change
    pub total_latency_change: DistributionChange,

    /// Throughput change (percentage)
    pub throughput_change: f64,

    /// Success rate change (percentage)
    pub success_rate_change: f64,

    /// Cost change (percentage, if available)
    pub cost_change: Option<f64>,
}

/// Distribution change between baseline and comparison
#[derive(Debug, Clone)]
pub struct DistributionChange {
    /// Mean change (percentage)
    pub mean_change: f64,

    /// P50 change (percentage)
    pub p50_change: f64,

    /// P95 change (percentage)
    pub p95_change: f64,

    /// P99 change (percentage)
    pub p99_change: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RequestMetrics;
    use chrono::Utc;
    use llm_latency_lens_core::{Provider, RequestId, SessionId};

    fn create_test_metrics(ttft_ms: u64, total_ms: u64, tokens_per_sec: f64) -> RequestMetrics {
        RequestMetrics {
            request_id: RequestId::new(),
            session_id: SessionId::new(),
            provider: Provider::OpenAI,
            model: "gpt-4".to_string(),
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
            tokens_per_second: tokens_per_sec,
            cost_usd: Some(0.05),
            success: true,
            error: None,
        }
    }

    #[test]
    fn test_aggregate_empty_collector() {
        let collector = MetricsCollector::with_defaults(SessionId::new()).unwrap();
        let result = MetricsAggregator::aggregate(&collector);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MetricsError::NoMetrics));
    }

    #[test]
    fn test_aggregate_single_metric() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        let metrics = create_test_metrics(100, 1000, 50.0);
        collector.record(metrics).unwrap();

        let aggregated = MetricsAggregator::aggregate(&collector).unwrap();

        assert_eq!(aggregated.total_requests, 1);
        assert_eq!(aggregated.successful_requests, 1);
        assert_eq!(aggregated.failed_requests, 0);
        assert_eq!(aggregated.ttft_distribution.sample_count, 1);
    }

    #[test]
    fn test_aggregate_multiple_metrics() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        for i in 0..100 {
            let metrics = create_test_metrics(100 + i, 1000 + i, 50.0);
            collector.record(metrics).unwrap();
        }

        let aggregated = MetricsAggregator::aggregate(&collector).unwrap();

        assert_eq!(aggregated.total_requests, 100);
        assert_eq!(aggregated.successful_requests, 100);
        assert_eq!(aggregated.ttft_distribution.sample_count, 100);
        assert!(aggregated.ttft_distribution.mean > Duration::ZERO);
        assert!(aggregated.ttft_distribution.p99 > aggregated.ttft_distribution.p50);
    }

    #[test]
    fn test_aggregate_with_failures() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        // Add successful requests
        for i in 0..90 {
            let metrics = create_test_metrics(100 + i, 1000 + i, 50.0);
            collector.record(metrics).unwrap();
        }

        // Add failed requests
        for _ in 0..10 {
            let mut metrics = create_test_metrics(100, 1000, 50.0);
            metrics.success = false;
            metrics.error = Some("Test error".to_string());
            collector.record(metrics).unwrap();
        }

        let aggregated = MetricsAggregator::aggregate(&collector).unwrap();

        assert_eq!(aggregated.total_requests, 100);
        assert_eq!(aggregated.successful_requests, 90);
        assert_eq!(aggregated.failed_requests, 10);
        assert_eq!(aggregated.success_rate(), 90.0);
    }

    #[test]
    fn test_aggregate_by_provider() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        // Add OpenAI metrics
        for i in 0..50 {
            let mut metrics = create_test_metrics(100 + i, 1000 + i, 50.0);
            metrics.provider = Provider::OpenAI;
            collector.record(metrics).unwrap();
        }

        // Add Anthropic metrics
        for i in 0..30 {
            let mut metrics = create_test_metrics(200 + i, 2000 + i, 40.0);
            metrics.provider = Provider::Anthropic;
            collector.record(metrics).unwrap();
        }

        let openai_aggregated =
            MetricsAggregator::aggregate_by_provider(&collector, Provider::OpenAI).unwrap();
        assert_eq!(openai_aggregated.total_requests, 50);

        let anthropic_aggregated =
            MetricsAggregator::aggregate_by_provider(&collector, Provider::Anthropic).unwrap();
        assert_eq!(anthropic_aggregated.total_requests, 30);
    }

    #[test]
    fn test_aggregate_by_model() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        // Add GPT-4 metrics
        for i in 0..40 {
            let mut metrics = create_test_metrics(100 + i, 1000 + i, 50.0);
            metrics.model = "gpt-4".to_string();
            collector.record(metrics).unwrap();
        }

        // Add GPT-3.5 metrics
        for i in 0..60 {
            let mut metrics = create_test_metrics(50 + i, 500 + i, 100.0);
            metrics.model = "gpt-3.5-turbo".to_string();
            collector.record(metrics).unwrap();
        }

        let gpt4_aggregated =
            MetricsAggregator::aggregate_by_model(&collector, "gpt-4").unwrap();
        assert_eq!(gpt4_aggregated.total_requests, 40);

        let gpt35_aggregated =
            MetricsAggregator::aggregate_by_model(&collector, "gpt-3.5-turbo").unwrap();
        assert_eq!(gpt35_aggregated.total_requests, 60);
    }

    #[test]
    fn test_metrics_comparison() {
        // Create baseline metrics
        let session1 = SessionId::new();
        let collector1 = MetricsCollector::with_defaults(session1).unwrap();
        for i in 0..100 {
            let metrics = create_test_metrics(100 + i, 1000 + i, 50.0);
            collector1.record(metrics).unwrap();
        }
        let baseline = MetricsAggregator::aggregate(&collector1).unwrap();

        // Create comparison metrics (faster)
        let session2 = SessionId::new();
        let collector2 = MetricsCollector::with_defaults(session2).unwrap();
        for i in 0..100 {
            let metrics = create_test_metrics(80 + i, 800 + i, 60.0);
            collector2.record(metrics).unwrap();
        }
        let comparison = MetricsAggregator::aggregate(&collector2).unwrap();

        let comp = MetricsAggregator::compare(&baseline, &comparison);

        // Comparison should be faster (negative percentage change for latency)
        assert!(comp.ttft_change.mean_change < 0.0);
        assert!(comp.total_latency_change.mean_change < 0.0);
        // Throughput should be higher (positive percentage change)
        assert!(comp.throughput_change > 0.0);
    }

    #[test]
    fn test_latency_distribution_calculations() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        // Add metrics with known distribution
        let values = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        for &val in &values {
            let metrics = create_test_metrics(val, val * 10, 50.0);
            collector.record(metrics).unwrap();
        }

        let aggregated = MetricsAggregator::aggregate(&collector).unwrap();

        // Check that percentiles make sense
        assert!(aggregated.ttft_distribution.p50 <= aggregated.ttft_distribution.p95);
        assert!(aggregated.ttft_distribution.p95 <= aggregated.ttft_distribution.p99);
        assert!(aggregated.ttft_distribution.min <= aggregated.ttft_distribution.mean);
        assert!(aggregated.ttft_distribution.mean <= aggregated.ttft_distribution.max);
    }

    #[test]
    fn test_throughput_statistics() {
        let session_id = SessionId::new();
        let collector = MetricsCollector::with_defaults(session_id).unwrap();

        for i in 0..100 {
            let metrics = create_test_metrics(100, 1000, 50.0 + i as f64);
            collector.record(metrics).unwrap();
        }

        let aggregated = MetricsAggregator::aggregate(&collector).unwrap();

        assert!(aggregated.throughput.mean_tokens_per_second > 0.0);
        assert!(aggregated.throughput.min_tokens_per_second <= aggregated.throughput.mean_tokens_per_second);
        assert!(aggregated.throughput.mean_tokens_per_second <= aggregated.throughput.max_tokens_per_second);
        assert!(aggregated.throughput.p50_tokens_per_second <= aggregated.throughput.p99_tokens_per_second);
    }
}
