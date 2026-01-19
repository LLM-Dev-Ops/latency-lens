//! Cold Start Detection Logic
//!
//! Core detection algorithms for identifying cold start behavior.
//! This module is MEASUREMENT classified - it only observes and measures.

use super::schemas::{
    ColdStartClassification, ColdStartDetectionAlgorithm, ColdStartFactor,
    ColdStartMeasurementConfig,
};
use crate::agents::contracts::ColdStartEvent;
use chrono::Utc;
use llm_latency_lens_core::{Provider, RequestId, SessionId};
use llm_latency_lens_metrics::RequestMetrics;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, trace};
use uuid::Uuid;

/// Cold start detector for analyzing request patterns
pub struct ColdStartDetector {
    config: ColdStartMeasurementConfig,
    algorithm: ColdStartDetectionAlgorithm,
}

/// Result of cold start detection
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Classified requests
    pub classifications: Vec<ColdStartClassification>,
    /// Detected cold start events
    pub cold_start_events: Vec<ColdStartEvent>,
    /// Baseline statistics per provider/model
    pub baseline_stats: HashMap<String, BaselineStats>,
    /// Overall detection confidence
    pub confidence: f64,
}

/// Baseline statistics for a provider/model combination
#[derive(Debug, Clone)]
pub struct BaselineStats {
    /// Provider
    pub provider: Provider,
    /// Model
    pub model: String,
    /// Mean TTFT (warm requests)
    pub mean_ttft: Duration,
    /// Standard deviation of TTFT
    pub std_dev_ttft: Duration,
    /// Sample count
    pub sample_count: u64,
}

impl ColdStartDetector {
    /// Create a new cold start detector
    pub fn new(config: ColdStartMeasurementConfig) -> Self {
        Self {
            config,
            algorithm: ColdStartDetectionAlgorithm::ThresholdBased,
        }
    }

    /// Set the detection algorithm
    pub fn with_algorithm(mut self, algorithm: ColdStartDetectionAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Get the algorithm being used
    pub fn algorithm(&self) -> ColdStartDetectionAlgorithm {
        self.algorithm
    }

    /// Detect cold starts in a series of request metrics
    pub fn detect(&self, session_id: SessionId, metrics: &[RequestMetrics]) -> DetectionResult {
        debug!(
            "Running cold start detection with algorithm {:?} on {} metrics",
            self.algorithm,
            metrics.len()
        );

        match self.algorithm {
            ColdStartDetectionAlgorithm::ThresholdBased => {
                self.detect_threshold_based(session_id, metrics)
            }
            ColdStartDetectionAlgorithm::ZScoreOutlier => {
                self.detect_zscore_based(session_id, metrics)
            }
            ColdStartDetectionAlgorithm::MovingAverageBaseline => {
                self.detect_moving_average_based(session_id, metrics)
            }
            ColdStartDetectionAlgorithm::InterArrivalTime => {
                self.detect_inter_arrival_based(session_id, metrics)
            }
        }
    }

    /// Threshold-based cold start detection
    fn detect_threshold_based(
        &self,
        session_id: SessionId,
        metrics: &[RequestMetrics],
    ) -> DetectionResult {
        let mut classifications = Vec::new();
        let mut cold_start_events = Vec::new();
        let mut baseline_stats = HashMap::new();

        // Group by provider:model
        let grouped = self.group_by_provider_model(metrics);

        for (key, group_metrics) in grouped {
            let (provider, model) = parse_key(&key);

            // Calculate baseline from warmup requests
            let baseline = self.calculate_baseline(&group_metrics);
            baseline_stats.insert(key.clone(), baseline.clone());

            let threshold_ttft = Duration::from_nanos(
                (baseline.mean_ttft.as_nanos() as f64 * self.config.cold_start_threshold_multiplier)
                    as u64,
            );

            for (i, metric) in group_metrics.iter().enumerate() {
                let is_first = i == 0 && self.config.include_first_request;
                let exceeds_threshold = metric.ttft > threshold_ttft;
                let is_cold_start = is_first || exceeds_threshold;

                let ratio = if baseline.mean_ttft.as_nanos() > 0 {
                    metric.ttft.as_nanos() as f64 / baseline.mean_ttft.as_nanos() as f64
                } else {
                    1.0
                };

                let mut factors = Vec::new();
                if is_first {
                    factors.push(ColdStartFactor::FirstRequest);
                }
                if exceeds_threshold {
                    factors.push(ColdStartFactor::HighTtft { ratio });
                }

                let confidence = self.calculate_confidence(
                    metric.ttft,
                    baseline.mean_ttft,
                    baseline.std_dev_ttft,
                );

                let classification = ColdStartClassification {
                    request_id: metric.request_id,
                    is_cold_start,
                    observed_ttft: metric.ttft,
                    expected_ttft: baseline.mean_ttft,
                    cold_start_ratio: ratio,
                    confidence,
                    factors,
                };

                if is_cold_start {
                    let event = ColdStartEvent {
                        event_id: Uuid::new_v4(),
                        session_id,
                        provider,
                        model: model.clone(),
                        cold_start_latency: metric.ttft,
                        baseline_latency: baseline.mean_ttft,
                        cold_start_ratio: ratio,
                        timestamp: metric.timestamp,
                        is_first_request: is_first,
                    };
                    cold_start_events.push(event);
                }

                classifications.push(classification);
            }
        }

        let overall_confidence = self.calculate_overall_confidence(&classifications);

        DetectionResult {
            classifications,
            cold_start_events,
            baseline_stats,
            confidence: overall_confidence,
        }
    }

    /// Z-score based outlier detection for cold starts
    fn detect_zscore_based(
        &self,
        session_id: SessionId,
        metrics: &[RequestMetrics],
    ) -> DetectionResult {
        let mut classifications = Vec::new();
        let mut cold_start_events = Vec::new();
        let mut baseline_stats = HashMap::new();

        let grouped = self.group_by_provider_model(metrics);

        for (key, group_metrics) in grouped {
            let (provider, model) = parse_key(&key);
            let baseline = self.calculate_baseline(&group_metrics);
            baseline_stats.insert(key.clone(), baseline.clone());

            // Z-score threshold (typically 2.0 for 95% or 3.0 for 99.7%)
            let z_threshold = 2.0;

            for (i, metric) in group_metrics.iter().enumerate() {
                let is_first = i == 0 && self.config.include_first_request;

                // Calculate Z-score
                let z_score = if baseline.std_dev_ttft.as_nanos() > 0 {
                    (metric.ttft.as_nanos() as f64 - baseline.mean_ttft.as_nanos() as f64)
                        / baseline.std_dev_ttft.as_nanos() as f64
                } else {
                    0.0
                };

                let is_cold_start = is_first || z_score > z_threshold;
                let ratio = if baseline.mean_ttft.as_nanos() > 0 {
                    metric.ttft.as_nanos() as f64 / baseline.mean_ttft.as_nanos() as f64
                } else {
                    1.0
                };

                let mut factors = Vec::new();
                if is_first {
                    factors.push(ColdStartFactor::FirstRequest);
                }
                if z_score > z_threshold {
                    factors.push(ColdStartFactor::HighTtft { ratio });
                }

                let confidence = if z_score > z_threshold {
                    // Higher Z-score = higher confidence
                    (1.0 - 2.0 / z_score.abs()).max(0.0).min(1.0)
                } else {
                    1.0 - z_score.abs() / z_threshold
                };

                let classification = ColdStartClassification {
                    request_id: metric.request_id,
                    is_cold_start,
                    observed_ttft: metric.ttft,
                    expected_ttft: baseline.mean_ttft,
                    cold_start_ratio: ratio,
                    confidence,
                    factors,
                };

                if is_cold_start {
                    let event = ColdStartEvent {
                        event_id: Uuid::new_v4(),
                        session_id,
                        provider,
                        model: model.clone(),
                        cold_start_latency: metric.ttft,
                        baseline_latency: baseline.mean_ttft,
                        cold_start_ratio: ratio,
                        timestamp: metric.timestamp,
                        is_first_request: is_first,
                    };
                    cold_start_events.push(event);
                }

                classifications.push(classification);
            }
        }

        let overall_confidence = self.calculate_overall_confidence(&classifications);

        DetectionResult {
            classifications,
            cold_start_events,
            baseline_stats,
            confidence: overall_confidence,
        }
    }

    /// Moving average baseline comparison
    fn detect_moving_average_based(
        &self,
        session_id: SessionId,
        metrics: &[RequestMetrics],
    ) -> DetectionResult {
        let mut classifications = Vec::new();
        let mut cold_start_events = Vec::new();
        let mut baseline_stats = HashMap::new();

        let grouped = self.group_by_provider_model(metrics);
        let window_size = self.config.baseline_warmup_count.max(3) as usize;

        for (key, group_metrics) in grouped {
            let (provider, model) = parse_key(&key);
            let baseline = self.calculate_baseline(&group_metrics);
            baseline_stats.insert(key.clone(), baseline.clone());

            let mut moving_avg_window: Vec<Duration> = Vec::with_capacity(window_size);

            for (i, metric) in group_metrics.iter().enumerate() {
                let is_first = i == 0 && self.config.include_first_request;

                // Calculate moving average baseline
                let moving_avg = if moving_avg_window.is_empty() {
                    baseline.mean_ttft
                } else {
                    let sum: u128 = moving_avg_window.iter().map(|d| d.as_nanos()).sum();
                    Duration::from_nanos((sum / moving_avg_window.len() as u128) as u64)
                };

                let threshold = Duration::from_nanos(
                    (moving_avg.as_nanos() as f64 * self.config.cold_start_threshold_multiplier)
                        as u64,
                );

                let exceeds_threshold = metric.ttft > threshold;
                let is_cold_start = is_first || exceeds_threshold;

                let ratio = if moving_avg.as_nanos() > 0 {
                    metric.ttft.as_nanos() as f64 / moving_avg.as_nanos() as f64
                } else {
                    1.0
                };

                let mut factors = Vec::new();
                if is_first {
                    factors.push(ColdStartFactor::FirstRequest);
                }
                if exceeds_threshold {
                    factors.push(ColdStartFactor::HighTtft { ratio });
                }

                let confidence =
                    self.calculate_confidence(metric.ttft, moving_avg, baseline.std_dev_ttft);

                let classification = ColdStartClassification {
                    request_id: metric.request_id,
                    is_cold_start,
                    observed_ttft: metric.ttft,
                    expected_ttft: moving_avg,
                    cold_start_ratio: ratio,
                    confidence,
                    factors,
                };

                if is_cold_start {
                    let event = ColdStartEvent {
                        event_id: Uuid::new_v4(),
                        session_id,
                        provider,
                        model: model.clone(),
                        cold_start_latency: metric.ttft,
                        baseline_latency: moving_avg,
                        cold_start_ratio: ratio,
                        timestamp: metric.timestamp,
                        is_first_request: is_first,
                    };
                    cold_start_events.push(event);
                }

                classifications.push(classification);

                // Update moving average window (only include warm requests)
                if !is_cold_start {
                    if moving_avg_window.len() >= window_size {
                        moving_avg_window.remove(0);
                    }
                    moving_avg_window.push(metric.ttft);
                }
            }
        }

        let overall_confidence = self.calculate_overall_confidence(&classifications);

        DetectionResult {
            classifications,
            cold_start_events,
            baseline_stats,
            confidence: overall_confidence,
        }
    }

    /// Inter-arrival time based detection
    fn detect_inter_arrival_based(
        &self,
        session_id: SessionId,
        metrics: &[RequestMetrics],
    ) -> DetectionResult {
        let mut classifications = Vec::new();
        let mut cold_start_events = Vec::new();
        let mut baseline_stats = HashMap::new();

        let grouped = self.group_by_provider_model(metrics);

        for (key, group_metrics) in grouped {
            let (provider, model) = parse_key(&key);
            let baseline = self.calculate_baseline(&group_metrics);
            baseline_stats.insert(key.clone(), baseline.clone());

            let mut prev_timestamp: Option<chrono::DateTime<chrono::Utc>> = None;

            for (i, metric) in group_metrics.iter().enumerate() {
                let is_first = i == 0 && self.config.include_first_request;

                // Calculate idle time since last request
                let idle_time_ms = if let Some(prev) = prev_timestamp {
                    let diff = metric.timestamp - prev;
                    diff.num_milliseconds().max(0) as u64
                } else {
                    u64::MAX // First request
                };

                // Check if idle time exceeds threshold
                let long_idle = idle_time_ms >= self.config.min_idle_time_ms;

                // Also check TTFT threshold
                let threshold = Duration::from_nanos(
                    (baseline.mean_ttft.as_nanos() as f64
                        * self.config.cold_start_threshold_multiplier) as u64,
                );
                let high_ttft = metric.ttft > threshold;

                let is_cold_start = is_first || (long_idle && high_ttft);

                let ratio = if baseline.mean_ttft.as_nanos() > 0 {
                    metric.ttft.as_nanos() as f64 / baseline.mean_ttft.as_nanos() as f64
                } else {
                    1.0
                };

                let mut factors = Vec::new();
                if is_first {
                    factors.push(ColdStartFactor::FirstRequest);
                }
                if long_idle && idle_time_ms < u64::MAX {
                    factors.push(ColdStartFactor::IdleTime {
                        idle_ms: idle_time_ms,
                    });
                }
                if high_ttft {
                    factors.push(ColdStartFactor::HighTtft { ratio });
                }

                let confidence =
                    self.calculate_confidence(metric.ttft, baseline.mean_ttft, baseline.std_dev_ttft);

                let classification = ColdStartClassification {
                    request_id: metric.request_id,
                    is_cold_start,
                    observed_ttft: metric.ttft,
                    expected_ttft: baseline.mean_ttft,
                    cold_start_ratio: ratio,
                    confidence,
                    factors,
                };

                if is_cold_start {
                    let event = ColdStartEvent {
                        event_id: Uuid::new_v4(),
                        session_id,
                        provider,
                        model: model.clone(),
                        cold_start_latency: metric.ttft,
                        baseline_latency: baseline.mean_ttft,
                        cold_start_ratio: ratio,
                        timestamp: metric.timestamp,
                        is_first_request: is_first,
                    };
                    cold_start_events.push(event);
                }

                classifications.push(classification);
                prev_timestamp = Some(metric.timestamp);
            }
        }

        let overall_confidence = self.calculate_overall_confidence(&classifications);

        DetectionResult {
            classifications,
            cold_start_events,
            baseline_stats,
            confidence: overall_confidence,
        }
    }

    /// Group metrics by provider:model key
    fn group_by_provider_model<'a>(
        &self,
        metrics: &'a [RequestMetrics],
    ) -> HashMap<String, Vec<&'a RequestMetrics>> {
        let mut grouped: HashMap<String, Vec<&RequestMetrics>> = HashMap::new();

        for metric in metrics {
            // Apply filters
            if let Some(ref provider_filter) = self.config.provider_filter {
                if metric.provider != *provider_filter {
                    continue;
                }
            }
            if let Some(ref model_filter) = self.config.model_filter {
                if !metric.model.contains(model_filter) {
                    continue;
                }
            }

            let key = format!("{}:{}", metric.provider.as_str(), metric.model);
            grouped.entry(key).or_default().push(metric);
        }

        // Sort each group by timestamp
        for group in grouped.values_mut() {
            group.sort_by_key(|m| m.timestamp);
        }

        grouped
    }

    /// Calculate baseline statistics from warmup requests
    fn calculate_baseline(&self, metrics: &[&RequestMetrics]) -> BaselineStats {
        // Skip first N requests as warmup, then calculate from remaining
        let warmup_count = self.config.baseline_warmup_count as usize;
        let baseline_metrics: Vec<_> = metrics
            .iter()
            .skip(if self.config.include_first_request {
                1
            } else {
                0
            })
            .take(warmup_count)
            .collect();

        let (provider, model) = if let Some(first) = metrics.first() {
            (first.provider, first.model.clone())
        } else {
            (Provider::Generic, "unknown".to_string())
        };

        if baseline_metrics.is_empty() {
            return BaselineStats {
                provider,
                model,
                mean_ttft: Duration::ZERO,
                std_dev_ttft: Duration::ZERO,
                sample_count: 0,
            };
        }

        let ttfts: Vec<Duration> = baseline_metrics.iter().map(|m| m.ttft).collect();
        let sum_nanos: u128 = ttfts.iter().map(|d| d.as_nanos()).sum();
        let count = ttfts.len();
        let mean_nanos = sum_nanos / count as u128;
        let mean_ttft = Duration::from_nanos(mean_nanos as u64);

        // Calculate standard deviation
        let variance: f64 = ttfts
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos as f64;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev_ttft = Duration::from_nanos(variance.sqrt() as u64);

        BaselineStats {
            provider,
            model,
            mean_ttft,
            std_dev_ttft,
            sample_count: count as u64,
        }
    }

    /// Calculate detection confidence for a single request
    fn calculate_confidence(
        &self,
        observed: Duration,
        expected: Duration,
        std_dev: Duration,
    ) -> f64 {
        if std_dev.is_zero() {
            return if observed == expected { 1.0 } else { 0.5 };
        }

        let z_score = (observed.as_nanos() as f64 - expected.as_nanos() as f64).abs()
            / std_dev.as_nanos() as f64;

        // Convert Z-score to confidence using normal distribution CDF approximation
        // Higher Z-score = more confident in cold start detection
        if z_score > 3.0 {
            0.99
        } else if z_score > 2.0 {
            0.95
        } else if z_score > 1.0 {
            0.84
        } else {
            0.5 + z_score * 0.34
        }
    }

    /// Calculate overall confidence for all classifications
    fn calculate_overall_confidence(&self, classifications: &[ColdStartClassification]) -> f64 {
        if classifications.is_empty() {
            return 0.0;
        }

        let sum: f64 = classifications.iter().map(|c| c.confidence).sum();
        sum / classifications.len() as f64
    }
}

/// Parse provider:model key back to components
fn parse_key(key: &str) -> (Provider, String) {
    let parts: Vec<&str> = key.splitn(2, ':').collect();
    let provider = match parts.first().unwrap_or(&"generic") {
        &"openai" => Provider::OpenAI,
        &"anthropic" => Provider::Anthropic,
        &"google" => Provider::Google,
        &"aws-bedrock" => Provider::AwsBedrock,
        &"azure-openai" => Provider::AzureOpenAI,
        _ => Provider::Generic,
    };
    let model = parts.get(1).unwrap_or(&"unknown").to_string();
    (provider, model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_metrics(count: usize, base_ttft_ms: u64) -> Vec<RequestMetrics> {
        let session_id = SessionId::new();
        let mut metrics = Vec::new();

        for i in 0..count {
            // First request has higher TTFT (simulating cold start)
            let ttft_ms = if i == 0 {
                base_ttft_ms * 3 // 3x for cold start
            } else {
                base_ttft_ms + (i as u64 % 20) // Small variation
            };

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

    #[test]
    fn test_threshold_based_detection() {
        let config = ColdStartMeasurementConfig::default();
        let detector = ColdStartDetector::new(config);
        let metrics = create_test_metrics(10, 100);
        let session_id = metrics[0].session_id;

        let result = detector.detect(session_id, &metrics);

        // First request should be detected as cold start
        assert!(!result.classifications.is_empty());
        assert!(result.classifications[0].is_cold_start);
        assert!(!result.cold_start_events.is_empty());
    }

    #[test]
    fn test_zscore_detection() {
        let config = ColdStartMeasurementConfig::default();
        let detector =
            ColdStartDetector::new(config).with_algorithm(ColdStartDetectionAlgorithm::ZScoreOutlier);
        let metrics = create_test_metrics(10, 100);
        let session_id = metrics[0].session_id;

        let result = detector.detect(session_id, &metrics);

        assert!(!result.classifications.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_moving_average_detection() {
        let config = ColdStartMeasurementConfig::default();
        let detector = ColdStartDetector::new(config)
            .with_algorithm(ColdStartDetectionAlgorithm::MovingAverageBaseline);
        let metrics = create_test_metrics(10, 100);
        let session_id = metrics[0].session_id;

        let result = detector.detect(session_id, &metrics);

        assert!(!result.classifications.is_empty());
    }

    #[test]
    fn test_baseline_calculation() {
        let config = ColdStartMeasurementConfig::default();
        let detector = ColdStartDetector::new(config);
        let metrics = create_test_metrics(10, 100);
        let refs: Vec<&RequestMetrics> = metrics.iter().collect();

        let baseline = detector.calculate_baseline(&refs);

        assert!(baseline.mean_ttft > Duration::ZERO);
        assert!(baseline.sample_count > 0);
    }
}
