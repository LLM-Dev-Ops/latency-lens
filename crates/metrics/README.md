# llm-latency-lens-metrics

High-precision metrics collection and statistical aggregation for LLM performance measurement.

## Overview

This crate provides production-ready metrics collection infrastructure using HDR Histogram for accurate percentile calculations. It's designed for measuring LLM API latency with thread-safe collectors and comprehensive statistical analysis.

## Features

- **High-precision percentiles** using HDR Histogram
- **Thread-safe collectors** for concurrent metric recording
- **Efficient memory usage** with configurable histogram parameters
- **Comprehensive metrics**:
  - TTFT (Time to First Token)
  - Inter-token latency distribution
  - Total request latency
  - Token throughput statistics
  - Cost tracking per request/session
- **Statistical aggregation**: p50, p90, p95, p99, p99.9
- **Comparison utilities**: Compare metrics across sessions
- **Full Serde support**: JSON serialization for all types

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
llm-latency-lens-metrics = "0.1.0"
llm-latency-lens-core = "0.1.0"
```

### Basic Example

```rust
use llm_latency_lens_metrics::{MetricsCollector, MetricsAggregator, RequestMetrics};
use llm_latency_lens_core::{SessionId, RequestId, Provider};
use std::time::Duration;
use chrono::Utc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create a metrics collector
    let session_id = SessionId::new();
    let collector = MetricsCollector::with_defaults(session_id)?;

    // Record metrics from LLM requests
    let metrics = RequestMetrics {
        request_id: RequestId::new(),
        session_id,
        provider: Provider::OpenAI,
        model: "gpt-4o".to_string(),
        timestamp: Utc::now(),
        ttft: Duration::from_millis(150),
        total_latency: Duration::from_millis(2000),
        inter_token_latencies: vec![
            Duration::from_millis(10),
            Duration::from_millis(15),
            Duration::from_millis(12),
        ],
        input_tokens: 100,
        output_tokens: 50,
        thinking_tokens: None,
        tokens_per_second: 25.0,
        cost_usd: Some(0.0015),
        success: true,
        error: None,
    };

    collector.record(metrics)?;

    // Aggregate statistics
    let aggregated = MetricsAggregator::aggregate(&collector)?;

    // Print results
    println!("Total requests: {}", aggregated.total_requests);
    println!("Success rate: {:.2}%", aggregated.success_rate());
    println!("TTFT p50: {:?}", aggregated.ttft_distribution.p50);
    println!("TTFT p99: {:?}", aggregated.ttft_distribution.p99);
    println!("Mean throughput: {:.2} tokens/sec",
             aggregated.throughput.mean_tokens_per_second);

    Ok(())
}
```

### Thread-Safe Collection

The collector is thread-safe and can be shared across multiple threads:

```rust
use std::sync::Arc;
use std::thread;

let collector = Arc::new(MetricsCollector::with_defaults(SessionId::new())?);

let mut handles = vec![];
for _ in 0..10 {
    let collector_clone = Arc::clone(&collector);
    let handle = thread::spawn(move || {
        // Each thread records its own metrics
        // collector_clone.record(metrics)?;
    });
    handles.push(handle);
}

for handle in handles {
    handle.join().unwrap();
}

// Aggregate all recorded metrics
let aggregated = MetricsAggregator::aggregate(&collector)?;
```

### Advanced Configuration

```rust
use llm_latency_lens_metrics::CollectorConfig;

let config = CollectorConfig::new()
    .with_max_value_seconds(120)      // Track up to 120 seconds
    .with_significant_digits(3);      // 0.1% percentile accuracy

let collector = MetricsCollector::new(session_id, config)?;
```

## Core Components

### MetricsCollector

Thread-safe collector for recording individual request metrics:

```rust
pub struct MetricsCollector {
    // Internal state with Arc<Mutex<>>
}

impl MetricsCollector {
    pub fn new(session_id: SessionId, config: CollectorConfig) -> Result<Self>;
    pub fn with_defaults(session_id: SessionId) -> Result<Self>;
    pub fn record(&self, metrics: RequestMetrics) -> Result<()>;
    pub fn len(&self) -> Result<usize>;
    pub fn clear(&self) -> Result<()>;
}
```

### MetricsAggregator

Statistical aggregation engine:

```rust
impl MetricsAggregator {
    // Aggregate all metrics
    pub fn aggregate(collector: &MetricsCollector) -> Result<AggregatedMetrics>;

    // Aggregate by provider
    pub fn aggregate_by_provider(
        collector: &MetricsCollector,
        provider: Provider
    ) -> Result<AggregatedMetrics>;

    // Aggregate by model
    pub fn aggregate_by_model(
        collector: &MetricsCollector,
        model: &str
    ) -> Result<AggregatedMetrics>;

    // Compare two aggregated results
    pub fn compare(
        baseline: &AggregatedMetrics,
        comparison: &AggregatedMetrics
    ) -> MetricsComparison;
}
```

### AggregatedMetrics

Complete statistical summary:

```rust
pub struct AggregatedMetrics {
    pub session_id: SessionId,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,

    // Latency distributions
    pub ttft_distribution: LatencyDistribution,
    pub inter_token_distribution: LatencyDistribution,
    pub total_latency_distribution: LatencyDistribution,

    // Throughput
    pub throughput: ThroughputStats,

    // Costs
    pub total_cost_usd: Option<f64>,

    // Token counts
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_thinking_tokens: Option<u64>,

    // Breakdowns
    pub provider_breakdown: HashMap<Provider, ProviderStats>,
    pub model_breakdown: HashMap<String, ModelStats>,
}
```

### LatencyDistribution

Percentile-based latency distribution:

```rust
pub struct LatencyDistribution {
    pub sample_count: u64,
    pub min: Duration,
    pub max: Duration,
    pub mean: Duration,
    pub stddev: Duration,
    pub p50: Duration,  // Median
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub p99_9: Duration,
}
```

### ThroughputStats

Token generation throughput statistics:

```rust
pub struct ThroughputStats {
    pub mean_tokens_per_second: f64,
    pub min_tokens_per_second: f64,
    pub max_tokens_per_second: f64,
    pub total_tokens: u64,
    pub total_duration: Duration,
}
```

## Provider and Model Breakdowns

Aggregate metrics by provider:

```rust
let openai_metrics = MetricsAggregator::aggregate_by_provider(
    &collector,
    Provider::OpenAI
)?;

println!("OpenAI TTFT p50: {:?}", openai_metrics.ttft_distribution.p50);
```

Aggregate metrics by model:

```rust
let gpt4_metrics = MetricsAggregator::aggregate_by_model(
    &collector,
    "gpt-4o"
)?;

println!("GPT-4o requests: {}", gpt4_metrics.total_requests);
```

Access breakdowns in aggregated metrics:

```rust
let aggregated = MetricsAggregator::aggregate(&collector)?;

// Per-provider breakdown
for (provider, stats) in &aggregated.provider_breakdown {
    println!("{:?}: {} requests, {:.2}% success",
        provider, stats.request_count, stats.success_rate());
}

// Per-model breakdown
for (model, stats) in &aggregated.model_breakdown {
    println!("{}: {:.0}ms mean TTFT",
        model, stats.mean_ttft.as_millis());
}
```

## Metrics Comparison

Compare two measurement sessions:

```rust
// Baseline session
let baseline_collector = MetricsCollector::with_defaults(SessionId::new())?;
// ... record baseline metrics ...
let baseline = MetricsAggregator::aggregate(&baseline_collector)?;

// Improved session
let improved_collector = MetricsCollector::with_defaults(SessionId::new())?;
// ... record improved metrics ...
let improved = MetricsAggregator::aggregate(&improved_collector)?;

// Compare
let comparison = MetricsAggregator::compare(&baseline, &improved);

println!("TTFT change: {:.2}%", comparison.ttft_change.mean_change);
println!("Total latency change: {:.2}%", comparison.total_latency_change.mean_change);
println!("Throughput change: {:.2}%", comparison.throughput_change);

if comparison.ttft_change.mean_change < 0.0 {
    println!("Improved by {:.2}%!", comparison.ttft_change.mean_change.abs());
}
```

## Cost Tracking

Track costs across sessions:

```rust
let aggregated = MetricsAggregator::aggregate(&collector)?;

if let Some(total_cost) = aggregated.total_cost_usd {
    println!("Total session cost: ${:.4}", total_cost);

    if let Some(avg_cost) = aggregated.avg_cost_per_request() {
        println!("Average cost per request: ${:.6}", avg_cost);
    }
}

// Per-model cost breakdown
for (model, stats) in &aggregated.model_breakdown {
    if let Some(cost) = stats.total_cost {
        println!("{}: ${:.4}", model, cost);
    }
}
```

## JSON Serialization

All types support Serde serialization:

```rust
use serde_json;

let aggregated = MetricsAggregator::aggregate(&collector)?;

// Serialize to JSON
let json = serde_json::to_string_pretty(&aggregated)?;
println!("{}", json);

// Deserialize from JSON
let restored: AggregatedMetrics = serde_json::from_str(&json)?;
```

## Performance Characteristics

- **Recording overhead**: ~1-2µs per metric
- **Memory usage**: ~100KB per 10,000 samples (default config)
- **Aggregation time**: ~100µs for 10,000 samples
- **Percentile accuracy**: 0.1% (with 3 significant digits)
- **Thread safety**: Lock-based with minimal contention

## Configuration Options

```rust
pub struct CollectorConfig {
    pub max_value_seconds: u64,      // Default: 60
    pub significant_digits: u8,      // Default: 3 (range: 1-5)
}
```

- **max_value_seconds**: Maximum latency value to track (seconds)
  - Higher values use more memory
  - Should be set to expected maximum latency + buffer

- **significant_digits**: Percentile precision (1-5)
  - 1 = 10% accuracy, 3 = 0.1% accuracy, 5 = 0.001% accuracy
  - Higher precision uses more memory

## HDR Histogram Details

This crate uses the `hdrhistogram` crate for percentile calculations:

- Constant-time recording and percentile queries
- Configurable precision and range
- Memory-efficient representation
- Accurate percentiles even at extreme ranges (p99.9+)

## Testing

Comprehensive test coverage including:

```bash
cargo test --package llm-latency-lens-metrics
```

Tests cover:
- Concurrent recording
- Provider/model breakdowns
- Cost tracking
- Percentile accuracy
- Serialization
- Comparison utilities

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Contributing

This crate is part of the LLM Latency Lens project. See the main repository for contribution guidelines.
