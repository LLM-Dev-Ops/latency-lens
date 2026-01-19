//! Timing event schemas from agentics-contracts
//!
//! Defines the timing events captured during LLM request profiling.
//! These are compatible with LLM-Observatory telemetry.

use chrono::{DateTime, Utc};
use llm_latency_lens_core::{Provider, RequestId, SessionId};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

/// Request lifecycle phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestPhase {
    /// DNS resolution phase
    DnsLookup,
    /// TCP connection establishment
    TcpConnect,
    /// TLS handshake
    TlsHandshake,
    /// Request sent, waiting for response
    RequestSend,
    /// Time to first byte from server
    TimeToFirstByte,
    /// Time to first token (LLM specific)
    TimeToFirstToken,
    /// Token streaming phase
    TokenStreaming,
    /// Request complete
    Complete,
    /// Request failed
    Failed,
}

/// Timing checkpoint during request lifecycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingCheckpoint {
    /// Checkpoint identifier
    pub id: Uuid,
    /// Request this checkpoint belongs to
    pub request_id: RequestId,
    /// Phase this checkpoint represents
    pub phase: RequestPhase,
    /// Timestamp (nanoseconds since epoch)
    pub timestamp_nanos: u64,
    /// Duration since request start
    #[serde(with = "duration_nanos")]
    pub elapsed: Duration,
    /// Additional context
    pub context: Option<serde_json::Value>,
}

/// Complete request timing trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTimingTrace {
    /// Unique trace identifier
    pub trace_id: Uuid,
    /// Session identifier
    pub session_id: SessionId,
    /// Request identifier
    pub request_id: RequestId,
    /// Provider used
    pub provider: Provider,
    /// Model used
    pub model: String,
    /// All timing checkpoints
    pub checkpoints: Vec<TimingCheckpoint>,
    /// Token events (for streaming requests)
    pub token_events: Vec<TokenTimingEvent>,
    /// Trace start time
    pub start_time: DateTime<Utc>,
    /// Trace end time (if complete)
    pub end_time: Option<DateTime<Utc>>,
    /// Final status
    pub status: TraceStatus,
    /// Error message (if failed)
    pub error: Option<String>,
}

impl RequestTimingTrace {
    /// Create a new trace
    pub fn new(session_id: SessionId, request_id: RequestId, provider: Provider, model: String) -> Self {
        Self {
            trace_id: Uuid::new_v4(),
            session_id,
            request_id,
            provider,
            model,
            checkpoints: Vec::new(),
            token_events: Vec::new(),
            start_time: Utc::now(),
            end_time: None,
            status: TraceStatus::InProgress,
            error: None,
        }
    }

    /// Add a checkpoint
    pub fn add_checkpoint(&mut self, phase: RequestPhase, timestamp_nanos: u64, elapsed: Duration) {
        self.checkpoints.push(TimingCheckpoint {
            id: Uuid::new_v4(),
            request_id: self.request_id,
            phase,
            timestamp_nanos,
            elapsed,
            context: None,
        });
    }

    /// Add a token event
    pub fn add_token(&mut self, sequence: u64, timestamp_nanos: u64, inter_token_latency: Option<Duration>) {
        self.token_events.push(TokenTimingEvent {
            request_id: self.request_id,
            sequence,
            timestamp_nanos,
            inter_token_latency,
        });
    }

    /// Mark trace as complete
    pub fn complete(&mut self) {
        self.end_time = Some(Utc::now());
        self.status = TraceStatus::Complete;
    }

    /// Mark trace as failed
    pub fn fail(&mut self, error: String) {
        self.end_time = Some(Utc::now());
        self.status = TraceStatus::Failed;
        self.error = Some(error);
    }

    /// Get time to first token (if available)
    pub fn ttft(&self) -> Option<Duration> {
        self.checkpoints
            .iter()
            .find(|cp| cp.phase == RequestPhase::TimeToFirstToken)
            .map(|cp| cp.elapsed)
    }

    /// Get total duration
    pub fn total_duration(&self) -> Option<Duration> {
        self.checkpoints
            .iter()
            .find(|cp| cp.phase == RequestPhase::Complete)
            .map(|cp| cp.elapsed)
    }

    /// Get inter-token latencies
    pub fn inter_token_latencies(&self) -> Vec<Duration> {
        self.token_events
            .iter()
            .filter_map(|te| te.inter_token_latency)
            .collect()
    }
}

/// Trace completion status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceStatus {
    /// Trace in progress
    InProgress,
    /// Trace completed successfully
    Complete,
    /// Trace failed
    Failed,
    /// Trace timed out
    TimedOut,
}

/// Token timing event for streaming requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTimingEvent {
    /// Request this token belongs to
    pub request_id: RequestId,
    /// Token sequence number (0 = first token)
    pub sequence: u64,
    /// Timestamp (nanoseconds since epoch)
    pub timestamp_nanos: u64,
    /// Time since previous token
    #[serde(with = "option_duration_nanos")]
    pub inter_token_latency: Option<Duration>,
}

/// Cold start detection event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartEvent {
    /// Event identifier
    pub event_id: Uuid,
    /// Session identifier
    pub session_id: SessionId,
    /// Provider that experienced cold start
    pub provider: Provider,
    /// Model that experienced cold start
    pub model: String,
    /// Detected cold start latency
    #[serde(with = "duration_nanos")]
    pub cold_start_latency: Duration,
    /// Baseline latency (warm)
    #[serde(with = "duration_nanos")]
    pub baseline_latency: Duration,
    /// Cold start ratio (cold / warm)
    pub cold_start_ratio: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Whether this was the first request in session
    pub is_first_request: bool,
}

/// Serde module for Duration serialization to nanoseconds
mod duration_nanos {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_nanos() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nanos = u64::deserialize(deserializer)?;
        Ok(Duration::from_nanos(nanos))
    }
}

/// Serde module for Option<Duration> serialization
mod option_duration_nanos {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => serializer.serialize_some(&(d.as_nanos() as u64)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nanos = Option::<u64>::deserialize(deserializer)?;
        Ok(nanos.map(Duration::from_nanos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_timing_trace() {
        let mut trace = RequestTimingTrace::new(
            SessionId::new(),
            RequestId::new(),
            Provider::OpenAI,
            "gpt-4".to_string(),
        );

        trace.add_checkpoint(RequestPhase::DnsLookup, 1000000, Duration::from_micros(1));
        trace.add_checkpoint(RequestPhase::TimeToFirstToken, 100000000, Duration::from_millis(100));
        trace.add_token(0, 100000000, None);
        trace.add_token(1, 120000000, Some(Duration::from_millis(20)));
        trace.complete();

        assert_eq!(trace.ttft(), Some(Duration::from_millis(100)));
        assert_eq!(trace.status, TraceStatus::Complete);
        assert_eq!(trace.inter_token_latencies().len(), 1);
    }

    #[test]
    fn test_trace_serialization() {
        let trace = RequestTimingTrace::new(
            SessionId::new(),
            RequestId::new(),
            Provider::Anthropic,
            "claude-3-opus".to_string(),
        );

        let json = serde_json::to_string(&trace).unwrap();
        let deserialized: RequestTimingTrace = serde_json::from_str(&json).unwrap();

        assert_eq!(trace.trace_id, deserialized.trace_id);
        assert_eq!(trace.provider, deserialized.provider);
    }
}
