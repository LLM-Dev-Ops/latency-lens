//! Phase 2 Signal Emission
//!
//! Atomic signals for operational intelligence including:
//! - Anomaly signals
//! - Drift signals
//! - Memory lineage signals
//! - Latency signals
//!
//! # Signal Rules
//!
//! All signals MUST:
//! - Be atomic (single observation, no aggregation)
//! - Include confidence score (0.0 - 1.0)
//! - Avoid conclusions (raw observations only)
//! - Include source agent and timestamp

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;

/// Signal types for Phase 2 operational intelligence
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    /// Anomaly detection signal
    Anomaly,
    /// Drift detection signal
    Drift,
    /// Memory lineage signal
    Lineage,
    /// Latency observation signal
    Latency,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Anomaly => write!(f, "anomaly"),
            SignalType::Drift => write!(f, "drift"),
            SignalType::Lineage => write!(f, "lineage"),
            SignalType::Latency => write!(f, "latency"),
        }
    }
}

/// Common metadata for all signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalMetadata {
    /// Unique signal identifier
    pub signal_id: Uuid,
    /// Signal type
    pub signal_type: SignalType,
    /// Source agent ID
    pub agent_id: String,
    /// Agent phase (phase2)
    pub agent_phase: String,
    /// Agent layer (layer1)
    pub agent_layer: String,
    /// Timestamp when signal was generated
    pub timestamp: DateTime<Utc>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Optional trace ID for distributed tracing
    pub trace_id: Option<String>,
    /// Optional span ID for distributed tracing
    pub span_id: Option<String>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SignalMetadata {
    /// Create new signal metadata
    pub fn new(signal_type: SignalType, agent_id: impl Into<String>, confidence: f64) -> Self {
        Self {
            signal_id: Uuid::now_v7(),
            signal_type,
            agent_id: agent_id.into(),
            agent_phase: "phase2".to_string(),
            agent_layer: "layer1".to_string(),
            timestamp: Utc::now(),
            confidence: confidence.clamp(0.0, 1.0),
            trace_id: None,
            span_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Add trace context
    pub fn with_trace(mut self, trace_id: impl Into<String>, span_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self.span_id = Some(span_id.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Base trait for all signals
pub trait Signal: Serialize + Send + Sync {
    /// Get signal metadata
    fn metadata(&self) -> &SignalMetadata;

    /// Get signal type
    fn signal_type(&self) -> SignalType {
        self.metadata().signal_type.clone()
    }

    /// Serialize signal to JSON
    fn to_json(&self) -> serde_json::Result<serde_json::Value>
    where
        Self: Serialize,
    {
        serde_json::to_value(self)
    }
}

/// Anomaly signal - reports observed anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalySignal {
    /// Signal metadata
    pub metadata: SignalMetadata,
    /// Metric that triggered the anomaly
    pub metric_name: String,
    /// Observed value
    pub observed_value: f64,
    /// Expected value or baseline
    pub expected_value: f64,
    /// Deviation from expected (percentage)
    pub deviation_percent: f64,
    /// Anomaly category
    pub category: AnomalyCategory,
}

/// Categories of anomalies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnomalyCategory {
    /// Latency exceeded threshold
    LatencySpike,
    /// Throughput dropped
    ThroughputDrop,
    /// Error rate increased
    ErrorSpike,
    /// Token usage anomaly
    TokenAnomaly,
    /// Cold start detected
    ColdStart,
    /// Other anomaly
    Other,
}

impl AnomalySignal {
    /// Create a new anomaly signal
    pub fn new(
        agent_id: impl Into<String>,
        metric_name: impl Into<String>,
        observed: f64,
        expected: f64,
        confidence: f64,
        category: AnomalyCategory,
    ) -> Self {
        let deviation = if expected != 0.0 {
            ((observed - expected) / expected * 100.0).abs()
        } else {
            0.0
        };

        Self {
            metadata: SignalMetadata::new(SignalType::Anomaly, agent_id, confidence),
            metric_name: metric_name.into(),
            observed_value: observed,
            expected_value: expected,
            deviation_percent: deviation,
            category,
        }
    }
}

impl Signal for AnomalySignal {
    fn metadata(&self) -> &SignalMetadata {
        &self.metadata
    }
}

/// Drift signal - reports observed drift from baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftSignal {
    /// Signal metadata
    pub metadata: SignalMetadata,
    /// Metric exhibiting drift
    pub metric_name: String,
    /// Baseline period start
    pub baseline_start: DateTime<Utc>,
    /// Baseline period end
    pub baseline_end: DateTime<Utc>,
    /// Baseline value
    pub baseline_value: f64,
    /// Current observed value
    pub current_value: f64,
    /// Drift amount (absolute)
    pub drift_amount: f64,
    /// Drift direction
    pub drift_direction: DriftDirection,
}

/// Direction of drift
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DriftDirection {
    /// Values increasing
    Increasing,
    /// Values decreasing
    Decreasing,
    /// No significant drift
    Stable,
}

impl DriftSignal {
    /// Create a new drift signal
    pub fn new(
        agent_id: impl Into<String>,
        metric_name: impl Into<String>,
        baseline_start: DateTime<Utc>,
        baseline_end: DateTime<Utc>,
        baseline_value: f64,
        current_value: f64,
        confidence: f64,
    ) -> Self {
        let drift_amount = current_value - baseline_value;
        let drift_direction = if drift_amount > 0.0 {
            DriftDirection::Increasing
        } else if drift_amount < 0.0 {
            DriftDirection::Decreasing
        } else {
            DriftDirection::Stable
        };

        Self {
            metadata: SignalMetadata::new(SignalType::Drift, agent_id, confidence),
            metric_name: metric_name.into(),
            baseline_start,
            baseline_end,
            baseline_value,
            current_value,
            drift_amount: drift_amount.abs(),
            drift_direction,
        }
    }
}

impl Signal for DriftSignal {
    fn metadata(&self) -> &SignalMetadata {
        &self.metadata
    }
}

/// Lineage signal - reports memory/event lineage deltas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageSignal {
    /// Signal metadata
    pub metadata: SignalMetadata,
    /// Source event ID
    pub source_event_id: Uuid,
    /// Lineage operation type
    pub operation: LineageOperation,
    /// Related event IDs
    pub related_events: Vec<Uuid>,
    /// Lineage path (agent chain)
    pub lineage_path: Vec<String>,
}

/// Lineage operation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LineageOperation {
    /// New event created
    Create,
    /// Event references another
    Reference,
    /// Event derived from another
    Derive,
    /// Event supersedes another
    Supersede,
}

impl LineageSignal {
    /// Create a new lineage signal
    pub fn new(
        agent_id: impl Into<String>,
        source_event_id: Uuid,
        operation: LineageOperation,
        confidence: f64,
    ) -> Self {
        Self {
            metadata: SignalMetadata::new(SignalType::Lineage, agent_id, confidence),
            source_event_id,
            operation,
            related_events: Vec::new(),
            lineage_path: Vec::new(),
        }
    }

    /// Add related events
    pub fn with_related_events(mut self, events: Vec<Uuid>) -> Self {
        self.related_events = events;
        self
    }

    /// Add lineage path
    pub fn with_lineage_path(mut self, path: Vec<String>) -> Self {
        self.lineage_path = path;
        self
    }
}

impl Signal for LineageSignal {
    fn metadata(&self) -> &SignalMetadata {
        &self.metadata
    }
}

/// Latency signal - reports observed latency measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySignal {
    /// Signal metadata
    pub metadata: SignalMetadata,
    /// Latency type
    pub latency_type: LatencyType,
    /// Observed latency in milliseconds
    pub latency_ms: f64,
    /// Provider name
    pub provider: String,
    /// Model name
    pub model: String,
    /// Request ID
    pub request_id: Option<String>,
}

/// Types of latency measurements
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LatencyType {
    /// Time to first token
    Ttft,
    /// Inter-token latency
    InterToken,
    /// End-to-end latency
    EndToEnd,
    /// Cold start latency
    ColdStart,
}

impl LatencySignal {
    /// Create a new latency signal
    pub fn new(
        agent_id: impl Into<String>,
        latency_type: LatencyType,
        latency_ms: f64,
        provider: impl Into<String>,
        model: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            metadata: SignalMetadata::new(SignalType::Latency, agent_id, confidence),
            latency_type,
            latency_ms,
            provider: provider.into(),
            model: model.into(),
            request_id: None,
        }
    }

    /// Add request ID
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }
}

impl Signal for LatencySignal {
    fn metadata(&self) -> &SignalMetadata {
        &self.metadata
    }
}

/// Signal emitter for Phase 2 agents
pub struct SignalEmitter {
    /// Agent ID for signals
    agent_id: String,
    /// Whether signal emission is enabled
    enabled: bool,
    /// RuVector endpoint for signal persistence
    ruvector_url: Option<String>,
}

impl SignalEmitter {
    /// Create a new signal emitter
    pub fn new(agent_id: impl Into<String>, enabled: bool) -> Self {
        Self {
            agent_id: agent_id.into(),
            enabled,
            ruvector_url: None,
        }
    }

    /// Configure RuVector endpoint
    pub fn with_ruvector(mut self, url: impl Into<String>) -> Self {
        self.ruvector_url = Some(url.into());
        self
    }

    /// Check if emission is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Emit an anomaly signal
    pub fn emit_anomaly(&self, signal: &AnomalySignal) {
        if !self.enabled {
            return;
        }

        info!(
            target: "phase2_signals",
            signal_id = %signal.metadata.signal_id,
            signal_type = "anomaly",
            agent_id = %signal.metadata.agent_id,
            metric = %signal.metric_name,
            observed = signal.observed_value,
            expected = signal.expected_value,
            deviation_percent = signal.deviation_percent,
            confidence = signal.metadata.confidence,
            "Anomaly signal emitted"
        );
    }

    /// Emit a drift signal
    pub fn emit_drift(&self, signal: &DriftSignal) {
        if !self.enabled {
            return;
        }

        info!(
            target: "phase2_signals",
            signal_id = %signal.metadata.signal_id,
            signal_type = "drift",
            agent_id = %signal.metadata.agent_id,
            metric = %signal.metric_name,
            baseline = signal.baseline_value,
            current = signal.current_value,
            drift = signal.drift_amount,
            direction = ?signal.drift_direction,
            confidence = signal.metadata.confidence,
            "Drift signal emitted"
        );
    }

    /// Emit a lineage signal
    pub fn emit_lineage(&self, signal: &LineageSignal) {
        if !self.enabled {
            return;
        }

        debug!(
            target: "phase2_signals",
            signal_id = %signal.metadata.signal_id,
            signal_type = "lineage",
            agent_id = %signal.metadata.agent_id,
            source_event = %signal.source_event_id,
            operation = ?signal.operation,
            related_count = signal.related_events.len(),
            confidence = signal.metadata.confidence,
            "Lineage signal emitted"
        );
    }

    /// Emit a latency signal
    pub fn emit_latency(&self, signal: &LatencySignal) {
        if !self.enabled {
            return;
        }

        info!(
            target: "phase2_signals",
            signal_id = %signal.metadata.signal_id,
            signal_type = "latency",
            agent_id = %signal.metadata.agent_id,
            latency_type = ?signal.latency_type,
            latency_ms = signal.latency_ms,
            provider = %signal.provider,
            model = %signal.model,
            confidence = signal.metadata.confidence,
            "Latency signal emitted"
        );
    }

    /// Emit any signal type
    pub fn emit<S: Signal>(&self, signal: &S) {
        if !self.enabled {
            return;
        }

        let metadata = signal.metadata();
        info!(
            target: "phase2_signals",
            signal_id = %metadata.signal_id,
            signal_type = %metadata.signal_type,
            agent_id = %metadata.agent_id,
            confidence = metadata.confidence,
            "Signal emitted"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_signal_creation() {
        let signal = AnomalySignal::new(
            "test-agent",
            "latency_ms",
            150.0,
            100.0,
            0.95,
            AnomalyCategory::LatencySpike,
        );

        assert_eq!(signal.metadata.signal_type, SignalType::Anomaly);
        assert_eq!(signal.deviation_percent, 50.0);
        assert_eq!(signal.metadata.confidence, 0.95);
    }

    #[test]
    fn test_drift_signal_direction() {
        let now = Utc::now();
        let earlier = now - chrono::Duration::hours(1);

        let increasing = DriftSignal::new(
            "test-agent",
            "throughput",
            earlier,
            now,
            100.0,
            150.0,
            0.9,
        );
        assert_eq!(increasing.drift_direction, DriftDirection::Increasing);

        let decreasing = DriftSignal::new(
            "test-agent",
            "throughput",
            earlier,
            now,
            100.0,
            50.0,
            0.9,
        );
        assert_eq!(decreasing.drift_direction, DriftDirection::Decreasing);
    }

    #[test]
    fn test_confidence_clamping() {
        let meta = SignalMetadata::new(SignalType::Anomaly, "test", 1.5);
        assert_eq!(meta.confidence, 1.0);

        let meta = SignalMetadata::new(SignalType::Anomaly, "test", -0.5);
        assert_eq!(meta.confidence, 0.0);
    }

    #[test]
    fn test_latency_signal() {
        let signal = LatencySignal::new(
            "latency-agent",
            LatencyType::Ttft,
            125.5,
            "openai",
            "gpt-4o",
            0.99,
        )
        .with_request_id("req-123");

        assert_eq!(signal.latency_type, LatencyType::Ttft);
        assert_eq!(signal.request_id, Some("req-123".to_string()));
    }
}
