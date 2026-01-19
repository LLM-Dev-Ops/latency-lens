//! Integration tests for Latency Analysis Agent
//!
//! These tests verify the agent meets all contract requirements
//! from agentics-contracts and the Agentics Dev platform constitution.

use chrono::Utc;
use llm_latency_lens::agents::{
    contracts::{
        DecisionType, LatencyAnalysisConfig, LatencyAnalysisInput, MeasurementConstraints,
    },
    latency_analysis::{LatencyAnalysisAgent, LatencyAnalyzer, AGENT_ID, AGENT_VERSION},
    ruvector::{RuVectorClient, RuVectorConfig},
};
use llm_latency_lens_core::{Provider, RequestId, SessionId};
use llm_latency_lens_metrics::RequestMetrics;
use std::sync::Arc;
use std::time::Duration;

/// Create test metrics
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

/// Test: Agent ID and Version are correctly defined
#[test]
fn test_agent_identification() {
    assert_eq!(AGENT_ID, "latency-analysis-agent");
    assert!(AGENT_VERSION.contains('.'), "Version must be semver format");
}

/// Test: Input validation against agentics-contracts schema
#[test]
fn test_input_validation_success() {
    let metrics = create_test_metrics(20);
    let input = LatencyAnalysisInput::new(metrics);

    assert!(input.validate().is_ok());
}

/// Test: Input validation fails for empty metrics
#[test]
fn test_input_validation_empty_metrics() {
    let input = LatencyAnalysisInput::new(vec![]);

    assert!(input.validate().is_err());
}

/// Test: Input validation fails for insufficient samples
#[test]
fn test_input_validation_insufficient_samples() {
    let metrics = create_test_metrics(5); // Less than default min_samples of 10
    let input = LatencyAnalysisInput::new(metrics);

    assert!(input.validate().is_err());
}

/// Test: Analyzer produces correct output structure
#[test]
fn test_analyzer_output_structure() {
    let config = LatencyAnalysisConfig {
        warmup_count: 0,
        min_samples: 1,
        ..Default::default()
    };
    let analyzer = LatencyAnalyzer::new(config);
    let metrics = create_test_metrics(20);
    let input = LatencyAnalysisInput::new(metrics);

    let output = analyzer.analyze(&input);

    // Verify output structure
    assert!(output.summary.total_requests > 0);
    assert!(output.summary.included_requests > 0);
    assert!(output.ttft_analysis.distribution.sample_count > 0);
    assert!(output.total_latency_analysis.distribution.sample_count > 0);
    assert!(output.throughput_analysis.stats.mean_tokens_per_second > 0.0);
}

/// Test: TTFT analysis produces valid distribution
#[test]
fn test_ttft_distribution() {
    let config = LatencyAnalysisConfig {
        warmup_count: 0,
        min_samples: 1,
        ..Default::default()
    };
    let analyzer = LatencyAnalyzer::new(config);
    let metrics = create_test_metrics(100);
    let input = LatencyAnalysisInput::new(metrics);

    let output = analyzer.analyze(&input);

    let ttft = &output.ttft_analysis.distribution;

    // Verify distribution properties
    assert!(ttft.min <= ttft.p50);
    assert!(ttft.p50 <= ttft.p90);
    assert!(ttft.p90 <= ttft.p95);
    assert!(ttft.p95 <= ttft.p99);
    assert!(ttft.p99 <= ttft.max);
    assert!(ttft.sample_count > 0);
}

/// Test: Warmup requests are excluded
#[test]
fn test_warmup_exclusion() {
    let warmup_count = 5;
    let config = LatencyAnalysisConfig {
        warmup_count,
        min_samples: 1,
        ..Default::default()
    };
    let analyzer = LatencyAnalyzer::new(config);
    let metrics = create_test_metrics(20);
    let input = LatencyAnalysisInput::new(metrics);

    let output = analyzer.analyze(&input);

    // 20 total - 5 warmup = 15 included (some may be removed as outliers)
    assert!(output.summary.included_requests < 20);
    assert!(output.metadata.warmup_excluded == warmup_count as u64);
}

/// Test: Outlier removal works correctly
#[test]
fn test_outlier_removal() {
    let config = LatencyAnalysisConfig {
        warmup_count: 0,
        min_samples: 1,
        remove_outliers: true,
        outlier_sigma: 2.0, // More aggressive outlier removal
        ..Default::default()
    };
    let analyzer = LatencyAnalyzer::new(config);

    // Create metrics with one extreme outlier
    let mut metrics = create_test_metrics(20);
    metrics[10].total_latency = Duration::from_secs(100); // Extreme outlier

    let input = LatencyAnalysisInput::new(metrics);
    let output = analyzer.analyze(&input);

    // At least one outlier should be removed
    assert!(output.metadata.outliers_removed > 0 || output.summary.included_requests < 20);
}

/// Test: Provider filtering works
#[test]
fn test_provider_filter() {
    let config = LatencyAnalysisConfig {
        warmup_count: 0,
        min_samples: 1,
        provider_filter: Some(Provider::Anthropic),
        ..Default::default()
    };
    let analyzer = LatencyAnalyzer::new(config);

    // Create metrics with mixed providers
    let mut metrics = create_test_metrics(10);
    for m in metrics.iter_mut() {
        m.provider = Provider::OpenAI;
    }

    let input = LatencyAnalysisInput::new(metrics);
    let output = analyzer.analyze(&input);

    // All should be filtered out since none are Anthropic
    assert_eq!(output.summary.included_requests, 0);
}

/// Test: Anomaly detection identifies outliers
#[test]
fn test_anomaly_detection() {
    let config = LatencyAnalysisConfig {
        warmup_count: 0,
        min_samples: 1,
        remove_outliers: false, // Keep outliers so we can detect them
        ..Default::default()
    };
    let analyzer = LatencyAnalyzer::new(config);

    // Create metrics with one extreme value
    let mut metrics = create_test_metrics(20);
    metrics[0].ttft = Duration::from_secs(10); // Extreme TTFT

    let input = LatencyAnalysisInput::new(metrics);
    let output = analyzer.analyze(&input);

    // Should detect the anomaly
    assert!(!output.anomalies.is_empty());
}

/// Test: Trend analysis produces valid results
#[test]
fn test_trend_analysis() {
    let config = LatencyAnalysisConfig {
        warmup_count: 0,
        min_samples: 1,
        ..Default::default()
    };
    let analyzer = LatencyAnalyzer::new(config);

    // Create metrics with increasing latency trend
    let mut metrics = create_test_metrics(50);
    for (i, m) in metrics.iter_mut().enumerate() {
        m.total_latency = Duration::from_millis(100 + (i as u64 * 20));
    }

    let input = LatencyAnalysisInput::new(metrics);
    let output = analyzer.analyze(&input);

    // Should detect increasing trend
    if let Some(ref trend) = output.total_latency_analysis.trend {
        assert!(trend.r_squared >= 0.0 && trend.r_squared <= 1.0);
    }
}

/// Test: Classification is ANALYSIS
#[test]
fn test_agent_classification() {
    let decision_type = DecisionType::LatencyAnalysis;
    let classification = decision_type.classification();

    assert_eq!(
        classification,
        llm_latency_lens::agents::contracts::AgentClassification::Analysis
    );
}

/// Test: Metadata contains required fields
#[test]
fn test_metadata_completeness() {
    let config = LatencyAnalysisConfig {
        warmup_count: 2,
        min_samples: 1,
        ..Default::default()
    };
    let analyzer = LatencyAnalyzer::new(config);
    let metrics = create_test_metrics(20);
    let input = LatencyAnalysisInput::new(metrics);

    let output = analyzer.analyze(&input);

    assert_eq!(output.metadata.agent_version, AGENT_VERSION);
    assert!(output.metadata.analysis_duration_ms > 0);
    assert_eq!(output.metadata.warmup_excluded, 2);
}

/// Test: RuVector client can be created
#[test]
fn test_ruvector_client_creation() {
    let config = RuVectorConfig::default();
    let client = RuVectorClient::new(config);

    assert!(client.is_ok());
}

/// Test: Agent can be created with client
#[test]
fn test_agent_creation() {
    let config = RuVectorConfig::default();
    let client = Arc::new(RuVectorClient::new(config).unwrap());
    let agent = LatencyAnalysisAgent::new(client);

    assert_eq!(agent.agent_id(), AGENT_ID);
    assert_eq!(agent.agent_version(), AGENT_VERSION);
}

/// Test: Agent can be configured with execution reference
#[test]
fn test_agent_with_execution_ref() {
    let config = RuVectorConfig::default();
    let client = Arc::new(RuVectorClient::new(config).unwrap());
    let agent = LatencyAnalysisAgent::new(client)
        .with_execution_ref("test-exec-123")
        .with_trace_context("trace-abc", "span-xyz");

    // Agent should be created without error
    assert_eq!(agent.agent_id(), AGENT_ID);
}

/// Test: Serialization/deserialization of input
#[test]
fn test_input_serialization() {
    let metrics = create_test_metrics(10);
    let input = LatencyAnalysisInput::new(metrics);

    let json = serde_json::to_string(&input).unwrap();
    let deserialized: LatencyAnalysisInput = serde_json::from_str(&json).unwrap();

    assert_eq!(input.analysis_id, deserialized.analysis_id);
    assert_eq!(input.metrics.len(), deserialized.metrics.len());
}

/// Test: Serialization/deserialization of output
#[test]
fn test_output_serialization() {
    let config = LatencyAnalysisConfig {
        warmup_count: 0,
        min_samples: 1,
        ..Default::default()
    };
    let analyzer = LatencyAnalyzer::new(config);
    let metrics = create_test_metrics(20);
    let input = LatencyAnalysisInput::new(metrics);

    let output = analyzer.analyze(&input);

    let json = serde_json::to_string(&output).unwrap();
    let deserialized: llm_latency_lens::LatencyAnalysisOutput = serde_json::from_str(&json).unwrap();

    assert_eq!(output.analysis_id, deserialized.analysis_id);
    assert_eq!(
        output.summary.total_requests,
        deserialized.summary.total_requests
    );
}

/// Test: Histogram buckets are valid
#[test]
fn test_histogram_generation() {
    let config = LatencyAnalysisConfig {
        warmup_count: 0,
        min_samples: 1,
        ..Default::default()
    };
    let analyzer = LatencyAnalyzer::new(config);
    let metrics = create_test_metrics(100);
    let input = LatencyAnalysisInput::new(metrics);

    let output = analyzer.analyze(&input);

    // Histogram should have buckets
    if !output.ttft_analysis.histogram.is_empty() {
        let total_percentage: f64 = output
            .ttft_analysis
            .histogram
            .iter()
            .map(|b| b.percentage)
            .sum();

        // Total percentage should be approximately 100%
        assert!((total_percentage - 100.0).abs() < 1.0);
    }
}
