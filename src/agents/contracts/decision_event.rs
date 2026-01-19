//! DecisionEvent schema from agentics-contracts
//!
//! Every agent invocation MUST emit exactly ONE DecisionEvent.
//! This is persisted to ruvector-service for audit and analysis.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

/// Decision types for latency analysis agents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionType {
    /// Latency analysis decision (ANALYSIS classification)
    LatencyAnalysis,
    /// Profiling decision (MEASUREMENT classification)
    Profiling,
    /// Distribution analysis decision (ANALYSIS classification)
    DistributionAnalysis,
    /// Cold start measurement (MEASUREMENT classification)
    ColdStartMeasurement,
    /// Throughput analysis (ANALYSIS classification)
    ThroughputAnalysis,
}

impl DecisionType {
    /// Get the agent classification for this decision type
    pub fn classification(&self) -> AgentClassification {
        match self {
            DecisionType::LatencyAnalysis => AgentClassification::Analysis,
            DecisionType::Profiling => AgentClassification::Measurement,
            DecisionType::DistributionAnalysis => AgentClassification::Analysis,
            DecisionType::ColdStartMeasurement => AgentClassification::Measurement,
            DecisionType::ThroughputAnalysis => AgentClassification::Analysis,
        }
    }
}

/// Agent classification (MEASUREMENT or ANALYSIS)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentClassification {
    /// Measurement agents capture raw timing data
    Measurement,
    /// Analysis agents process and interpret measurement data
    Analysis,
}

/// Confidence metadata for measurement/analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceMetadata {
    /// Confidence score (0.0 - 1.0)
    pub score: f64,

    /// Precision of measurements (e.g., nanoseconds)
    pub precision_unit: String,

    /// Error bounds (if applicable)
    pub error_bounds: Option<ErrorBounds>,

    /// Sample size used for analysis
    pub sample_size: u64,

    /// Statistical confidence interval (e.g., 95%)
    pub confidence_interval_percent: Option<f64>,
}

/// Error bounds for measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBounds {
    /// Lower bound (percentage or absolute)
    pub lower: f64,
    /// Upper bound (percentage or absolute)
    pub upper: f64,
    /// Whether bounds are percentage or absolute
    pub is_percentage: bool,
}

/// Constraints applied during measurement/analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementConstraints {
    /// Minimum sample size required
    pub min_sample_size: Option<u64>,
    /// Maximum latency threshold (nanoseconds)
    pub max_latency_threshold_ns: Option<u64>,
    /// Outlier removal applied
    pub outlier_removal: bool,
    /// Outlier threshold (if applied)
    pub outlier_threshold_sigma: Option<f64>,
    /// Warmup requests excluded
    pub warmup_excluded: bool,
    /// Number of warmup requests excluded
    pub warmup_count: Option<u32>,
}

/// DecisionEvent schema from agentics-contracts
///
/// This is the canonical event emitted by every agent invocation.
/// It is persisted to ruvector-service (backed by Google SQL/Postgres).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionEvent {
    /// Unique event identifier
    pub event_id: Uuid,

    /// Agent identifier (e.g., "latency-analysis-agent")
    pub agent_id: String,

    /// Agent version (semver format)
    pub agent_version: String,

    /// Type of decision made
    pub decision_type: DecisionType,

    /// Agent classification
    pub classification: AgentClassification,

    /// SHA-256 hash of inputs for reproducibility
    pub inputs_hash: String,

    /// Structured outputs from the agent
    pub outputs: serde_json::Value,

    /// Confidence metadata
    pub confidence: ConfidenceMetadata,

    /// Measurement constraints applied
    pub constraints_applied: MeasurementConstraints,

    /// External execution reference (for correlation)
    pub execution_ref: Option<String>,

    /// Trace ID for distributed tracing
    pub trace_id: Option<String>,

    /// Span ID for distributed tracing
    pub span_id: Option<String>,

    /// Timestamp (UTC)
    pub timestamp: DateTime<Utc>,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl DecisionEvent {
    /// Create a new DecisionEvent builder
    pub fn builder() -> DecisionEventBuilder {
        DecisionEventBuilder::new()
    }

    /// Compute inputs hash from arbitrary serializable data
    pub fn compute_inputs_hash<T: Serialize>(inputs: &T) -> String {
        let json = serde_json::to_vec(inputs).unwrap_or_default();
        let hash = Sha256::digest(&json);
        format!("{:x}", hash)
    }

    /// Validate the event against schema requirements
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Validate agent_id format
        if self.agent_id.is_empty() {
            return Err(ValidationError::EmptyField("agent_id"));
        }

        // Validate agent_version is semver-like
        if !self.agent_version.contains('.') {
            return Err(ValidationError::InvalidFormat("agent_version must be semver format"));
        }

        // Validate inputs_hash is valid hex
        if self.inputs_hash.len() != 64 || !self.inputs_hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(ValidationError::InvalidFormat("inputs_hash must be 64-char hex"));
        }

        // Validate confidence score
        if !(0.0..=1.0).contains(&self.confidence.score) {
            return Err(ValidationError::OutOfRange("confidence.score must be 0.0-1.0"));
        }

        Ok(())
    }
}

/// Validation errors for DecisionEvent
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    #[error("Empty required field: {0}")]
    EmptyField(&'static str),

    #[error("Invalid format: {0}")]
    InvalidFormat(&'static str),

    #[error("Value out of range: {0}")]
    OutOfRange(&'static str),
}

/// Builder for DecisionEvent
pub struct DecisionEventBuilder {
    agent_id: Option<String>,
    agent_version: Option<String>,
    decision_type: Option<DecisionType>,
    inputs_hash: Option<String>,
    outputs: Option<serde_json::Value>,
    confidence: Option<ConfidenceMetadata>,
    constraints: Option<MeasurementConstraints>,
    execution_ref: Option<String>,
    trace_id: Option<String>,
    span_id: Option<String>,
    metadata: HashMap<String, serde_json::Value>,
}

impl DecisionEventBuilder {
    pub fn new() -> Self {
        Self {
            agent_id: None,
            agent_version: None,
            decision_type: None,
            inputs_hash: None,
            outputs: None,
            confidence: None,
            constraints: None,
            execution_ref: None,
            trace_id: None,
            span_id: None,
            metadata: HashMap::new(),
        }
    }

    pub fn agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    pub fn agent_version(mut self, version: impl Into<String>) -> Self {
        self.agent_version = Some(version.into());
        self
    }

    pub fn decision_type(mut self, dt: DecisionType) -> Self {
        self.decision_type = Some(dt);
        self
    }

    pub fn inputs_hash(mut self, hash: impl Into<String>) -> Self {
        self.inputs_hash = Some(hash.into());
        self
    }

    pub fn inputs<T: Serialize>(mut self, inputs: &T) -> Self {
        self.inputs_hash = Some(DecisionEvent::compute_inputs_hash(inputs));
        self
    }

    pub fn outputs(mut self, outputs: serde_json::Value) -> Self {
        self.outputs = Some(outputs);
        self
    }

    pub fn confidence(mut self, confidence: ConfidenceMetadata) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn constraints(mut self, constraints: MeasurementConstraints) -> Self {
        self.constraints = Some(constraints);
        self
    }

    pub fn execution_ref(mut self, exec_ref: impl Into<String>) -> Self {
        self.execution_ref = Some(exec_ref.into());
        self
    }

    pub fn trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    pub fn span_id(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = Some(span_id.into());
        self
    }

    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn build(self) -> Result<DecisionEvent, BuilderError> {
        let decision_type = self.decision_type.ok_or(BuilderError::MissingField("decision_type"))?;

        Ok(DecisionEvent {
            event_id: Uuid::new_v4(),
            agent_id: self.agent_id.ok_or(BuilderError::MissingField("agent_id"))?,
            agent_version: self.agent_version.ok_or(BuilderError::MissingField("agent_version"))?,
            decision_type,
            classification: decision_type.classification(),
            inputs_hash: self.inputs_hash.ok_or(BuilderError::MissingField("inputs_hash"))?,
            outputs: self.outputs.unwrap_or(serde_json::Value::Null),
            confidence: self.confidence.ok_or(BuilderError::MissingField("confidence"))?,
            constraints_applied: self.constraints.unwrap_or(MeasurementConstraints {
                min_sample_size: None,
                max_latency_threshold_ns: None,
                outlier_removal: false,
                outlier_threshold_sigma: None,
                warmup_excluded: false,
                warmup_count: None,
            }),
            execution_ref: self.execution_ref,
            trace_id: self.trace_id,
            span_id: self.span_id,
            timestamp: Utc::now(),
            metadata: self.metadata,
        })
    }
}

impl Default for DecisionEventBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum BuilderError {
    #[error("Missing required field: {0}")]
    MissingField(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_event_builder() {
        let confidence = ConfidenceMetadata {
            score: 0.95,
            precision_unit: "nanoseconds".to_string(),
            error_bounds: Some(ErrorBounds {
                lower: 0.01,
                upper: 0.01,
                is_percentage: true,
            }),
            sample_size: 1000,
            confidence_interval_percent: Some(95.0),
        };

        let event = DecisionEvent::builder()
            .agent_id("latency-analysis-agent")
            .agent_version("0.1.0")
            .decision_type(DecisionType::LatencyAnalysis)
            .inputs(&serde_json::json!({"test": "input"}))
            .outputs(serde_json::json!({"result": "output"}))
            .confidence(confidence)
            .build()
            .unwrap();

        assert_eq!(event.agent_id, "latency-analysis-agent");
        assert_eq!(event.classification, AgentClassification::Analysis);
        assert!(event.validate().is_ok());
    }

    #[test]
    fn test_inputs_hash() {
        let inputs = serde_json::json!({"prompt": "test", "model": "gpt-4"});
        let hash = DecisionEvent::compute_inputs_hash(&inputs);

        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
