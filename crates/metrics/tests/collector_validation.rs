use chrono::Utc;
use llm_latency_lens_core::{Provider, RequestId, SessionId};
use llm_latency_lens_metrics::{CollectorConfig, MetricsAggregator, MetricsCollector, RequestMetrics};
use std::time::Duration;

fn sample() -> RequestMetrics {
    RequestMetrics {
        request_id: RequestId::new(),
        session_id: SessionId::new(),
        provider: Provider::OpenAI,
        model: "fixture".into(),
        timestamp: Utc::now(),
        ttft: Duration::from_millis(10),
        total_latency: Duration::from_millis(100),
        inter_token_latencies: vec![Duration::from_millis(5)],
        input_tokens: 10,
        output_tokens: 5,
        thinking_tokens: None,
        tokens_per_second: 50.0,
        cost_usd: Some(0.01),
        success: true,
        error: None,
    }
}

fn collector() -> MetricsCollector {
    MetricsCollector::new(SessionId::new(), CollectorConfig::default()).unwrap()
}

fn rejects_without_mutation(case: usize) {
    let c = collector();
    c.record(sample()).unwrap();
    let before = serde_json::to_value(MetricsAggregator::aggregate(&c).unwrap()).unwrap();
    let mut invalid = sample();
    match case {
        0 => invalid.tokens_per_second = f64::NAN,
        1 => invalid.tokens_per_second = f64::INFINITY,
        2 => invalid.tokens_per_second = -1.0,
        3 => invalid.tokens_per_second = 1_000_001.0,
        4 => invalid.ttft = Duration::from_secs(120),
        5 => invalid.total_latency = Duration::from_secs(120),
        6 => invalid.inter_token_latencies.push(Duration::from_secs(120)),
        7 => invalid.total_latency = Duration::from_secs(u64::MAX),
        _ => unreachable!(),
    }
    let id = invalid.request_id;
    assert!(c.record(invalid).is_err(), "invalid input must be rejected");
    assert_eq!(c.len().unwrap(), 1, "rejected request was retained");
    assert!(c.get_request(id).unwrap().is_none());
    let after = serde_json::to_value(MetricsAggregator::aggregate(&c).unwrap()).unwrap();
    assert_eq!(after, before, "rejected input changed collector state");
}

macro_rules! rejection {
    ($name:ident, $case:expr) => {
        #[test]
        fn $name() { rejects_without_mutation($case); }
    };
}
rejection!(reject_nan_throughput, 0);
rejection!(reject_infinite_throughput, 1);
rejection!(reject_negative_throughput, 2);
rejection!(reject_above_max_throughput, 3);
rejection!(reject_above_max_ttft, 4);
rejection!(reject_above_max_total, 5);
rejection!(reject_above_max_inter_token, 6);
rejection!(reject_overflow_duration, 7);

#[test]
fn anchor_normal() {
    let c = collector(); c.record(sample()).unwrap();
    let m = MetricsAggregator::aggregate(&c).unwrap();
    assert_eq!(m.total_requests, 1); assert_eq!(m.successful_requests, 1);
    assert_eq!(m.total_input_tokens, 10); assert_eq!(m.total_output_tokens, 5);
}
#[test]
fn anchor_zero_throughput() {
    let c = collector(); let mut m = sample(); m.tokens_per_second = 0.0;
    c.record(m).unwrap(); assert_eq!(c.len().unwrap(), 1);
}
#[test]
fn anchor_max_throughput() {
    let c = collector(); let mut m = sample(); m.tokens_per_second = 1_000_000.0;
    c.record(m).unwrap(); assert_eq!(c.len().unwrap(), 1);
}
#[test]
fn anchor_max_latency() {
    let c = collector(); let mut m = sample(); m.total_latency = Duration::from_secs(60);
    c.record(m).unwrap(); assert_eq!(c.len().unwrap(), 1);
}
#[test]
fn anchor_failure_retained() {
    let c = collector(); let mut m = sample(); m.success = false; m.tokens_per_second = f64::NAN;
    m.total_latency = Duration::from_secs(120); c.record(m).unwrap();
    let s = c.get_state_snapshot().unwrap(); assert_eq!(s.failed_requests, 1);
    assert_eq!(s.successful_requests, 0); assert_eq!(c.len().unwrap(), 1);
}
#[test]
fn anchor_lookup() {
    let c = collector(); let m = sample(); let id = m.request_id; c.record(m).unwrap();
    assert_eq!(c.get_request(id).unwrap().unwrap().output_tokens, 5);
}
#[test]
fn anchor_clear() {
    let c = collector(); c.record(sample()).unwrap(); c.clear().unwrap();
    assert!(c.is_empty().unwrap()); assert_eq!(c.get_state_snapshot().unwrap().successful_requests, 0);
}
#[test]
fn anchor_shared_clone() {
    let c = collector(); c.clone().record(sample()).unwrap(); assert_eq!(c.len().unwrap(), 1);
}
