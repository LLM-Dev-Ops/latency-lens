//! Core latency analysis logic
//!
//! Implements statistical analysis of latency data including:
//! - Distribution calculations (percentiles, variance, skewness, kurtosis)
//! - Anomaly detection
//! - Trend analysis
//! - Provider/model breakdown

use crate::agents::contracts::{
    AnalysisMetadata, AnalysisSummary, AnomalyType, HistogramBucket, LatencyAnalysisConfig,
    LatencyAnalysisInput, LatencyAnalysisOutput, LatencyAnomaly, LatencyDistributionAnalysis,
    ModelAnalysis, ProviderAnalysis, ThroughputAnalysis, TrendAnalysis, TrendDirection,
};
use llm_latency_lens_core::Provider;
use llm_latency_lens_metrics::{LatencyDistribution, RequestMetrics, ThroughputStats};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Core latency analyzer
pub struct LatencyAnalyzer {
    config: LatencyAnalysisConfig,
}

impl LatencyAnalyzer {
    /// Create a new analyzer with configuration
    pub fn new(config: LatencyAnalysisConfig) -> Self {
        Self { config }
    }

    /// Perform full latency analysis
    pub fn analyze(&self, input: &LatencyAnalysisInput) -> LatencyAnalysisOutput {
        let start_time = Instant::now();

        // Filter and preprocess metrics
        let (filtered_metrics, excluded_count) = self.filter_metrics(&input.metrics);

        // Calculate distributions
        let ttft_analysis = self.analyze_ttft(&filtered_metrics);
        let inter_token_analysis = self.analyze_inter_token(&filtered_metrics);
        let total_latency_analysis = self.analyze_total_latency(&filtered_metrics);

        // Calculate throughput
        let throughput_analysis = self.analyze_throughput(&filtered_metrics);

        // Provider and model breakdowns
        let provider_analysis = self.analyze_by_provider(&filtered_metrics);
        let model_analysis = self.analyze_by_model(&filtered_metrics);

        // Detect anomalies
        let anomalies = self.detect_anomalies(&filtered_metrics, &ttft_analysis, &total_latency_analysis);

        // Build summary
        let summary = self.build_summary(&input.metrics, &filtered_metrics, excluded_count);

        let analysis_duration_ms = start_time.elapsed().as_millis() as u64;

        LatencyAnalysisOutput {
            analysis_id: input.analysis_id,
            session_id: input.session_id,
            summary,
            ttft_analysis,
            inter_token_analysis,
            total_latency_analysis,
            throughput_analysis,
            provider_analysis,
            model_analysis,
            anomalies,
            metadata: AnalysisMetadata {
                agent_version: super::AGENT_VERSION.to_string(),
                analysis_duration_ms,
                config: self.config.clone(),
                outliers_removed: excluded_count.outliers,
                warmup_excluded: excluded_count.warmup,
            },
        }
    }

    /// Filter metrics based on configuration
    fn filter_metrics<'a>(&self, metrics: &'a [RequestMetrics]) -> (Vec<&'a RequestMetrics>, ExcludedCount) {
        let mut excluded = ExcludedCount::default();
        let mut filtered: Vec<&RequestMetrics> = Vec::new();

        for (i, metric) in metrics.iter().enumerate() {
            // Skip warmup requests
            if i < self.config.warmup_count as usize {
                excluded.warmup += 1;
                continue;
            }

            // Skip failed requests for timing analysis
            if !metric.success {
                excluded.failed += 1;
                continue;
            }

            // Apply provider filter
            if let Some(ref provider) = self.config.provider_filter {
                if metric.provider != *provider {
                    excluded.filtered += 1;
                    continue;
                }
            }

            // Apply model filter
            if let Some(ref model) = self.config.model_filter {
                if metric.model != *model {
                    excluded.filtered += 1;
                    continue;
                }
            }

            // Apply max latency threshold
            if let Some(max_ns) = self.config.max_latency_ns {
                if metric.total_latency.as_nanos() as u64 > max_ns {
                    excluded.threshold += 1;
                    continue;
                }
            }

            filtered.push(metric);
        }

        // Remove outliers if configured
        if self.config.remove_outliers && filtered.len() > 3 {
            let outlier_count = self.remove_outliers(&mut filtered);
            excluded.outliers = outlier_count;
        }

        (filtered, excluded)
    }

    /// Remove outliers using sigma threshold
    fn remove_outliers(&self, metrics: &mut Vec<&RequestMetrics>) -> u64 {
        if metrics.len() < 4 {
            return 0;
        }

        // Calculate mean and std_dev of total latency
        let latencies: Vec<f64> = metrics
            .iter()
            .map(|m| m.total_latency.as_nanos() as f64)
            .collect();

        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let variance =
            latencies.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / latencies.len() as f64;
        let std_dev = variance.sqrt();

        let threshold = self.config.outlier_sigma * std_dev;
        let original_len = metrics.len();

        metrics.retain(|m| {
            let latency = m.total_latency.as_nanos() as f64;
            (latency - mean).abs() <= threshold
        });

        (original_len - metrics.len()) as u64
    }

    /// Analyze TTFT (Time to First Token)
    fn analyze_ttft(&self, metrics: &[&RequestMetrics]) -> LatencyDistributionAnalysis {
        let values: Vec<Duration> = metrics.iter().map(|m| m.ttft).collect();
        self.create_distribution_analysis(&values)
    }

    /// Analyze inter-token latencies
    fn analyze_inter_token(&self, metrics: &[&RequestMetrics]) -> LatencyDistributionAnalysis {
        let values: Vec<Duration> = metrics
            .iter()
            .flat_map(|m| m.inter_token_latencies.iter().copied())
            .collect();
        self.create_distribution_analysis(&values)
    }

    /// Analyze total request latency
    fn analyze_total_latency(&self, metrics: &[&RequestMetrics]) -> LatencyDistributionAnalysis {
        let values: Vec<Duration> = metrics.iter().map(|m| m.total_latency).collect();
        self.create_distribution_analysis(&values)
    }

    /// Create distribution analysis from duration values
    fn create_distribution_analysis(&self, values: &[Duration]) -> LatencyDistributionAnalysis {
        if values.is_empty() {
            return LatencyDistributionAnalysis {
                distribution: LatencyDistribution::empty(),
                variance_ns2: 0.0,
                coefficient_of_variation: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                histogram: Vec::new(),
                trend: None,
            };
        }

        let nanos: Vec<f64> = values.iter().map(|d| d.as_nanos() as f64).collect();
        let n = nanos.len() as f64;

        // Calculate basic statistics
        let min = nanos.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = nanos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = nanos.iter().sum::<f64>() / n;

        let variance = nanos.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // Calculate percentiles
        let mut sorted = nanos.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&sorted, 50.0);
        let p90 = percentile(&sorted, 90.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);
        let p99_9 = percentile(&sorted, 99.9);

        // Calculate skewness and kurtosis
        let skewness = if std_dev > 0.0 {
            let m3 = nanos.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n;
            m3
        } else {
            0.0
        };

        let kurtosis = if std_dev > 0.0 {
            let m4 = nanos.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n;
            m4 - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        // Coefficient of variation
        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

        // Build histogram
        let histogram = self.build_histogram(&sorted, min, max);

        // Trend analysis (if we have enough samples)
        let trend = if values.len() >= 10 {
            Some(self.analyze_trend(&nanos))
        } else {
            None
        };

        LatencyDistributionAnalysis {
            distribution: LatencyDistribution {
                min: Duration::from_nanos(min as u64),
                max: Duration::from_nanos(max as u64),
                mean: Duration::from_nanos(mean as u64),
                std_dev: Duration::from_nanos(std_dev as u64),
                p50: Duration::from_nanos(p50 as u64),
                p90: Duration::from_nanos(p90 as u64),
                p95: Duration::from_nanos(p95 as u64),
                p99: Duration::from_nanos(p99 as u64),
                p99_9: Duration::from_nanos(p99_9 as u64),
                sample_count: values.len() as u64,
            },
            variance_ns2: variance,
            coefficient_of_variation: cv,
            skewness,
            kurtosis,
            histogram,
            trend,
        }
    }

    /// Build histogram buckets
    fn build_histogram(&self, sorted: &[f64], min: f64, max: f64) -> Vec<HistogramBucket> {
        if sorted.is_empty() || min == max {
            return Vec::new();
        }

        let bucket_count = 20;
        let range = max - min;
        let bucket_width = range / bucket_count as f64;

        let mut buckets = Vec::with_capacity(bucket_count);
        let total = sorted.len() as f64;

        for i in 0..bucket_count {
            let lower = min + (i as f64 * bucket_width);
            let upper = if i == bucket_count - 1 {
                max + 1.0 // Include max value
            } else {
                min + ((i + 1) as f64 * bucket_width)
            };

            let count = sorted.iter().filter(|&&v| v >= lower && v < upper).count() as u64;

            buckets.push(HistogramBucket {
                lower_ns: lower as u64,
                upper_ns: upper as u64,
                count,
                percentage: (count as f64 / total) * 100.0,
            });
        }

        buckets
    }

    /// Analyze trend using linear regression
    fn analyze_trend(&self, values: &[f64]) -> TrendAnalysis {
        let n = values.len() as f64;

        // Simple linear regression: y = mx + b
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        let slope = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        // Calculate R-squared
        let intercept = y_mean - slope * x_mean;
        let ss_tot: f64 = values.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = values
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let predicted = slope * i as f64 + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Determine direction and significance
        let relative_slope = slope / y_mean.abs().max(1.0);
        let direction = if relative_slope > 0.01 {
            TrendDirection::Increasing
        } else if relative_slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        // Consider significant if R² > 0.5 and slope is meaningful
        let is_significant = r_squared > 0.5 && relative_slope.abs() > 0.01;

        TrendAnalysis {
            direction,
            slope_ns_per_request: slope,
            r_squared,
            is_significant,
        }
    }

    /// Analyze throughput
    fn analyze_throughput(&self, metrics: &[&RequestMetrics]) -> ThroughputAnalysis {
        if metrics.is_empty() {
            return ThroughputAnalysis {
                stats: ThroughputStats::empty(),
                peak_tokens_per_second: 0.0,
                sustained_tokens_per_second: 0.0,
                variance: 0.0,
                time_to_peak_ms: None,
            };
        }

        let tps_values: Vec<f64> = metrics.iter().map(|m| m.tokens_per_second).collect();
        let n = tps_values.len() as f64;

        let mean = tps_values.iter().sum::<f64>() / n;
        let min = tps_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = tps_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance = tps_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = tps_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        // Find time to peak
        let peak_index = tps_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i);

        let time_to_peak_ms = peak_index.and_then(|i| {
            if i > 0 {
                let cumulative_ms: u64 = metrics[..=i]
                    .iter()
                    .map(|m| m.total_latency.as_millis() as u64)
                    .sum();
                Some(cumulative_ms)
            } else {
                None
            }
        });

        // Calculate sustained throughput (total tokens / total time)
        let total_tokens: u64 = metrics.iter().map(|m| m.output_tokens).sum();
        let total_time_secs: f64 = metrics
            .iter()
            .map(|m| m.total_latency.as_secs_f64())
            .sum();
        let sustained = if total_time_secs > 0.0 {
            total_tokens as f64 / total_time_secs
        } else {
            0.0
        };

        ThroughputAnalysis {
            stats: ThroughputStats {
                mean_tokens_per_second: mean,
                min_tokens_per_second: min,
                max_tokens_per_second: max,
                std_dev_tokens_per_second: std_dev,
                p50_tokens_per_second: p50,
                p95_tokens_per_second: p95,
                p99_tokens_per_second: p99,
            },
            peak_tokens_per_second: max,
            sustained_tokens_per_second: sustained,
            variance,
            time_to_peak_ms,
        }
    }

    /// Analyze metrics by provider
    fn analyze_by_provider(&self, metrics: &[&RequestMetrics]) -> Vec<ProviderAnalysis> {
        let mut by_provider: HashMap<Provider, Vec<&RequestMetrics>> = HashMap::new();

        for metric in metrics {
            by_provider.entry(metric.provider).or_default().push(metric);
        }

        by_provider
            .into_iter()
            .map(|(provider, provider_metrics)| {
                let ttft_values: Vec<Duration> = provider_metrics.iter().map(|m| m.ttft).collect();
                let latency_values: Vec<Duration> =
                    provider_metrics.iter().map(|m| m.total_latency).collect();
                let success_count = provider_metrics.iter().filter(|m| m.success).count();

                ProviderAnalysis {
                    provider,
                    request_count: provider_metrics.len() as u64,
                    ttft: self.calculate_distribution(&ttft_values),
                    total_latency: self.calculate_distribution(&latency_values),
                    throughput: self.calculate_throughput_stats(&provider_metrics),
                    success_rate: (success_count as f64 / provider_metrics.len() as f64) * 100.0,
                }
            })
            .collect()
    }

    /// Analyze metrics by model
    fn analyze_by_model(&self, metrics: &[&RequestMetrics]) -> Vec<ModelAnalysis> {
        let mut by_model: HashMap<(Provider, String), Vec<&RequestMetrics>> = HashMap::new();

        for metric in metrics {
            by_model
                .entry((metric.provider, metric.model.clone()))
                .or_default()
                .push(metric);
        }

        by_model
            .into_iter()
            .map(|((provider, model), model_metrics)| {
                let ttft_values: Vec<Duration> = model_metrics.iter().map(|m| m.ttft).collect();
                let latency_values: Vec<Duration> =
                    model_metrics.iter().map(|m| m.total_latency).collect();
                let total_cost: f64 = model_metrics.iter().filter_map(|m| m.cost_usd).sum();

                ModelAnalysis {
                    provider,
                    model,
                    request_count: model_metrics.len() as u64,
                    ttft: self.calculate_distribution(&ttft_values),
                    total_latency: self.calculate_distribution(&latency_values),
                    throughput: self.calculate_throughput_stats(&model_metrics),
                    avg_cost_usd: if total_cost > 0.0 {
                        Some(total_cost / model_metrics.len() as f64)
                    } else {
                        None
                    },
                }
            })
            .collect()
    }

    /// Detect anomalies in the data
    fn detect_anomalies(
        &self,
        metrics: &[&RequestMetrics],
        ttft_analysis: &LatencyDistributionAnalysis,
        latency_analysis: &LatencyDistributionAnalysis,
    ) -> Vec<LatencyAnomaly> {
        let mut anomalies = Vec::new();

        let ttft_mean = ttft_analysis.distribution.mean.as_nanos() as f64;
        let ttft_std = ttft_analysis.distribution.std_dev.as_nanos() as f64;
        let latency_mean = latency_analysis.distribution.mean.as_nanos() as f64;
        let latency_std = latency_analysis.distribution.std_dev.as_nanos() as f64;

        for (i, metric) in metrics.iter().enumerate() {
            let ttft_ns = metric.ttft.as_nanos() as f64;
            let latency_ns = metric.total_latency.as_nanos() as f64;

            // Check for high TTFT
            if ttft_std > 0.0 {
                let ttft_deviation = (ttft_ns - ttft_mean) / ttft_std;
                if ttft_deviation > 3.0 {
                    anomalies.push(LatencyAnomaly {
                        anomaly_id: Uuid::new_v4(),
                        request_id: metric.request_id,
                        anomaly_type: if i == 0 {
                            AnomalyType::ColdStart
                        } else {
                            AnomalyType::HighTtft
                        },
                        severity: (ttft_deviation / 5.0).min(1.0),
                        observed_value_ns: ttft_ns as u64,
                        expected_value_ns: ttft_mean as u64,
                        deviation_sigma: ttft_deviation,
                        timestamp: metric.timestamp,
                    });
                }
            }

            // Check for high total latency
            if latency_std > 0.0 {
                let latency_deviation = (latency_ns - latency_mean) / latency_std;
                if latency_deviation > 3.0 {
                    anomalies.push(LatencyAnomaly {
                        anomaly_id: Uuid::new_v4(),
                        request_id: metric.request_id,
                        anomaly_type: AnomalyType::HighTotalLatency,
                        severity: (latency_deviation / 5.0).min(1.0),
                        observed_value_ns: latency_ns as u64,
                        expected_value_ns: latency_mean as u64,
                        deviation_sigma: latency_deviation,
                        timestamp: metric.timestamp,
                    });
                }
            }

            // Check for throughput degradation
            let avg_tps = metrics.iter().map(|m| m.tokens_per_second).sum::<f64>()
                / metrics.len() as f64;
            if metric.tokens_per_second < avg_tps * 0.5 {
                anomalies.push(LatencyAnomaly {
                    anomaly_id: Uuid::new_v4(),
                    request_id: metric.request_id,
                    anomaly_type: AnomalyType::ThroughputDegradation,
                    severity: 1.0 - (metric.tokens_per_second / avg_tps),
                    observed_value_ns: (metric.tokens_per_second * 1_000_000.0) as u64,
                    expected_value_ns: (avg_tps * 1_000_000.0) as u64,
                    deviation_sigma: (avg_tps - metric.tokens_per_second) / avg_tps,
                    timestamp: metric.timestamp,
                });
            }
        }

        anomalies
    }

    /// Build analysis summary
    fn build_summary(
        &self,
        all_metrics: &[RequestMetrics],
        filtered_metrics: &[&RequestMetrics],
        excluded: ExcludedCount,
    ) -> AnalysisSummary {
        let success_count = filtered_metrics.iter().filter(|m| m.success).count();

        let (start, end) = if let (Some(first), Some(last)) = (
            all_metrics.iter().map(|m| m.timestamp).min(),
            all_metrics.iter().map(|m| m.timestamp).max(),
        ) {
            (first, last)
        } else {
            let now = chrono::Utc::now();
            (now, now)
        };

        let total_tokens: u64 = filtered_metrics
            .iter()
            .map(|m| m.input_tokens + m.output_tokens)
            .sum();

        AnalysisSummary {
            total_requests: all_metrics.len() as u64,
            included_requests: filtered_metrics.len() as u64,
            excluded_requests: excluded.total(),
            success_rate: if !filtered_metrics.is_empty() {
                (success_count as f64 / filtered_metrics.len() as f64) * 100.0
            } else {
                0.0
            },
            time_window_start: start,
            time_window_end: end,
            total_tokens,
            avg_tokens_per_request: if !filtered_metrics.is_empty() {
                total_tokens as f64 / filtered_metrics.len() as f64
            } else {
                0.0
            },
        }
    }

    /// Calculate simple distribution
    fn calculate_distribution(&self, values: &[Duration]) -> LatencyDistribution {
        if values.is_empty() {
            return LatencyDistribution::empty();
        }

        let nanos: Vec<f64> = values.iter().map(|d| d.as_nanos() as f64).collect();
        let n = nanos.len() as f64;

        let min = nanos.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = nanos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = nanos.iter().sum::<f64>() / n;
        let variance = nanos.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = nanos.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        LatencyDistribution {
            min: Duration::from_nanos(min as u64),
            max: Duration::from_nanos(max as u64),
            mean: Duration::from_nanos(mean as u64),
            std_dev: Duration::from_nanos(std_dev as u64),
            p50: Duration::from_nanos(percentile(&sorted, 50.0) as u64),
            p90: Duration::from_nanos(percentile(&sorted, 90.0) as u64),
            p95: Duration::from_nanos(percentile(&sorted, 95.0) as u64),
            p99: Duration::from_nanos(percentile(&sorted, 99.0) as u64),
            p99_9: Duration::from_nanos(percentile(&sorted, 99.9) as u64),
            sample_count: values.len() as u64,
        }
    }

    /// Calculate throughput stats
    fn calculate_throughput_stats(&self, metrics: &[&RequestMetrics]) -> ThroughputStats {
        if metrics.is_empty() {
            return ThroughputStats::empty();
        }

        let tps_values: Vec<f64> = metrics.iter().map(|m| m.tokens_per_second).collect();
        let n = tps_values.len() as f64;

        let mean = tps_values.iter().sum::<f64>() / n;
        let min = tps_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = tps_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance = tps_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = tps_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        ThroughputStats {
            mean_tokens_per_second: mean,
            min_tokens_per_second: min,
            max_tokens_per_second: max,
            std_dev_tokens_per_second: std_dev,
            p50_tokens_per_second: percentile(&sorted, 50.0),
            p95_tokens_per_second: percentile(&sorted, 95.0),
            p99_tokens_per_second: percentile(&sorted, 99.0),
        }
    }
}

/// Calculate percentile from sorted values
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }

    let index = (p / 100.0) * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

/// Counts of excluded metrics
#[derive(Debug, Default, Clone, Copy)]
struct ExcludedCount {
    warmup: u64,
    outliers: u64,
    failed: u64,
    filtered: u64,
    threshold: u64,
}

impl ExcludedCount {
    fn total(&self) -> u64 {
        self.warmup + self.outliers + self.failed + self.filtered + self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use llm_latency_lens_core::{RequestId, SessionId};

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
                    Duration::from_millis(22),
                ],
                input_tokens: 100,
                output_tokens: 50,
                thinking_tokens: None,
                tokens_per_second: 50.0 - (i as f64),
                cost_usd: Some(0.05),
                success: true,
                error: None,
            })
            .collect()
    }

    #[test]
    fn test_analyzer_basic() {
        let config = LatencyAnalysisConfig {
            warmup_count: 0,
            min_samples: 1,
            ..Default::default()
        };
        let analyzer = LatencyAnalyzer::new(config);
        let metrics = create_test_metrics(10);
        let input = LatencyAnalysisInput::new(metrics);

        let output = analyzer.analyze(&input);

        assert_eq!(output.summary.included_requests, 10);
        assert!(output.ttft_analysis.distribution.sample_count > 0);
    }

    #[test]
    fn test_percentile_calculation() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&sorted, 50.0) - 5.5).abs() < 0.1);
        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 0.1);
        assert!((percentile(&sorted, 100.0) - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_trend_analysis() {
        let config = LatencyAnalysisConfig::default();
        let analyzer = LatencyAnalyzer::new(config);

        // Increasing trend
        let increasing: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 10.0).collect();
        let trend = analyzer.analyze_trend(&increasing);
        assert_eq!(trend.direction, TrendDirection::Increasing);

        // Decreasing trend
        let decreasing: Vec<f64> = (0..20).map(|i| 300.0 - i as f64 * 10.0).collect();
        let trend = analyzer.analyze_trend(&decreasing);
        assert_eq!(trend.direction, TrendDirection::Decreasing);
    }
}
