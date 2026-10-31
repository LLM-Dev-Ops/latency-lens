use chrono::Utc;
use llm_latency_lens_core::{Provider, RequestId, SessionId};
use llm_latency_lens_metrics::{
    aggregator::MetricsAggregator,
    collector::MetricsCollector,
    types::{AggregatedMetrics, RequestMetrics},
};
use std::time::Duration;

fn collector() -> MetricsCollector {
    let session = SessionId::new();
    let collector = MetricsCollector::with_defaults(session).unwrap();
    for provider in [
        Provider::OpenAI,
        Provider::Google,
        Provider::Anthropic,
        Provider::Generic,
        Provider::AzureOpenAI,
        Provider::AwsBedrock,
    ] {
        for model in ["zeta", "delta", "alpha", "omega", "beta", "gamma"] {
            collector
                .record(RequestMetrics {
                    request_id: RequestId::new(),
                    session_id: session,
                    provider,
                    model: model.into(),
                    timestamp: Utc::now(),
                    ttft: Duration::from_millis(10),
                    total_latency: Duration::from_millis(100),
                    inter_token_latencies: vec![Duration::from_millis(5)],
                    input_tokens: 100,
                    output_tokens: 50,
                    thinking_tokens: None,
                    tokens_per_second: 500.0,
                    cost_usd: Some(0.05),
                    success: true,
                    error: None,
                })
                .unwrap();
        }
    }
    collector
}

fn assert_ordered(metrics: &AggregatedMetrics, count: u64) {
    let providers: Vec<_> = metrics
        .provider_breakdown
        .iter()
        .map(|(p, _)| p.as_str())
        .collect();
    let models: Vec<_> = metrics
        .model_breakdown
        .iter()
        .map(|(m, _)| m.as_str())
        .collect();
    assert!(
        providers.windows(2).all(|p| p[0] < p[1]),
        "provider order: {providers:?}"
    );
    assert!(
        models.windows(2).all(|p| p[0] < p[1]),
        "model order: {models:?}"
    );
    assert_eq!(metrics.total_requests, count);
    assert_eq!(metrics.successful_requests, count);
    assert_eq!(metrics.failed_requests, 0);
    assert_eq!(metrics.total_input_tokens, count * 100);
    assert_eq!(metrics.total_output_tokens, count * 50);
    assert_eq!(
        metrics
            .provider_breakdown
            .iter()
            .map(|(_, n)| n)
            .sum::<u64>(),
        count
    );
    assert_eq!(
        metrics.model_breakdown.iter().map(|(_, n)| n).sum::<u64>(),
        count
    );
}

#[test]
fn full_breakdowns_are_sorted() {
    let collector = collector();
    for _ in 0..8 {
        assert_ordered(&MetricsAggregator::aggregate(&collector).unwrap(), 36);
    }
}

#[test]
fn provider_subset_breakdowns_are_sorted() {
    let collector = collector();
    for _ in 0..8 {
        assert_ordered(
            &MetricsAggregator::aggregate_by_provider(&collector, Provider::OpenAI).unwrap(),
            6,
        );
    }
}

#[test]
fn model_subset_breakdowns_are_sorted() {
    let collector = collector();
    for _ in 0..8 {
        assert_ordered(
            &MetricsAggregator::aggregate_by_model(&collector, "alpha").unwrap(),
            6,
        );
    }
}
