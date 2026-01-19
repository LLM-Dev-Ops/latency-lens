//! Telemetry emission for Latency Analysis Agent
//!
//! Emits telemetry compatible with LLM-Observatory.

use opentelemetry::trace::{Span, SpanKind, Tracer};
use opentelemetry::{global, KeyValue};
use tracing::{debug, info, instrument};

use crate::agents::contracts::{DecisionType, LatencyAnalysisInput, LatencyAnalysisOutput};

/// Agent telemetry emitter
pub struct AgentTelemetry {
    /// Tracer name
    tracer_name: String,
}

impl AgentTelemetry {
    /// Create a new telemetry emitter
    pub fn new(tracer_name: impl Into<String>) -> Self {
        Self {
            tracer_name: tracer_name.into(),
        }
    }

    /// Create default telemetry emitter for latency analysis agent
    pub fn default_latency_analysis() -> Self {
        Self::new("latency-analysis-agent")
    }

    /// Emit span for agent invocation
    #[instrument(skip(self, input, output))]
    pub fn emit_analysis_span(
        &self,
        input: &LatencyAnalysisInput,
        output: &LatencyAnalysisOutput,
        decision_type: DecisionType,
    ) {
        let tracer_name = self.tracer_name.clone();
        let tracer = global::tracer(tracer_name);

        let mut span = tracer
            .span_builder("latency_analysis")
            .with_kind(SpanKind::Internal)
            .start(&tracer);

        // Add span attributes
        span.set_attribute(KeyValue::new("agent.id", super::AGENT_ID));
        span.set_attribute(KeyValue::new("agent.version", super::AGENT_VERSION));
        span.set_attribute(KeyValue::new(
            "decision.type",
            format!("{:?}", decision_type),
        ));
        span.set_attribute(KeyValue::new(
            "analysis.id",
            input.analysis_id.to_string(),
        ));
        span.set_attribute(KeyValue::new(
            "input.metrics_count",
            input.metrics.len() as i64,
        ));
        span.set_attribute(KeyValue::new(
            "output.included_requests",
            output.summary.included_requests as i64,
        ));
        span.set_attribute(KeyValue::new(
            "output.excluded_requests",
            output.summary.excluded_requests as i64,
        ));
        span.set_attribute(KeyValue::new(
            "output.success_rate",
            output.summary.success_rate,
        ));
        span.set_attribute(KeyValue::new(
            "output.anomalies_count",
            output.anomalies.len() as i64,
        ));
        span.set_attribute(KeyValue::new(
            "analysis.duration_ms",
            output.metadata.analysis_duration_ms as i64,
        ));

        // Add distribution summary attributes
        span.set_attribute(KeyValue::new(
            "ttft.p50_ns",
            output.ttft_analysis.distribution.p50.as_nanos() as i64,
        ));
        span.set_attribute(KeyValue::new(
            "ttft.p99_ns",
            output.ttft_analysis.distribution.p99.as_nanos() as i64,
        ));
        span.set_attribute(KeyValue::new(
            "total_latency.p50_ns",
            output.total_latency_analysis.distribution.p50.as_nanos() as i64,
        ));
        span.set_attribute(KeyValue::new(
            "total_latency.p99_ns",
            output.total_latency_analysis.distribution.p99.as_nanos() as i64,
        ));
        span.set_attribute(KeyValue::new(
            "throughput.mean_tps",
            output.throughput_analysis.stats.mean_tokens_per_second,
        ));

        span.end();

        info!(
            analysis_id = %input.analysis_id,
            metrics_count = input.metrics.len(),
            included = output.summary.included_requests,
            anomalies = output.anomalies.len(),
            "Telemetry emitted for latency analysis"
        );
    }

    /// Emit metrics for the analysis
    pub fn emit_metrics(&self, output: &LatencyAnalysisOutput) {
        // Log key metrics
        debug!(
            ttft_p50_ms = output.ttft_analysis.distribution.p50.as_millis(),
            ttft_p99_ms = output.ttft_analysis.distribution.p99.as_millis(),
            total_latency_p50_ms = output.total_latency_analysis.distribution.p50.as_millis(),
            total_latency_p99_ms = output.total_latency_analysis.distribution.p99.as_millis(),
            throughput_mean = output.throughput_analysis.stats.mean_tokens_per_second,
            anomalies = output.anomalies.len(),
            "Latency analysis metrics"
        );
    }
}

impl Default for AgentTelemetry {
    fn default() -> Self {
        Self::default_latency_analysis()
    }
}
