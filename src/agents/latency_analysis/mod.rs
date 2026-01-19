//! Latency Analysis Agent
//!
//! Measures and analyzes end-to-end and token-level latency characteristics
//! of LLM requests.
//!
//! # Classification: ANALYSIS
//!
//! This agent processes measurement data and produces analytical insights.
//! It does NOT modify system behavior, trigger remediation, or apply optimizations.
//!
//! # Scope
//!
//! - Measure time-to-first-token (TTFT)
//! - Measure per-token streaming latency
//! - Measure end-to-end request duration
//! - Produce latency distributions and percentiles
//!
//! # decision_type: "latency_analysis"
//!
//! # Contract Compliance
//!
//! - Input schema: [`LatencyAnalysisInput`] from agentics-contracts
//! - Output schema: [`LatencyAnalysisOutput`] from agentics-contracts
//! - Emits exactly ONE [`DecisionEvent`] per invocation
//! - Persists to ruvector-service only
//! - Telemetry compatible with LLM-Observatory

mod agent;
mod analyzer;
mod telemetry;

pub use agent::LatencyAnalysisAgent;
pub use analyzer::LatencyAnalyzer;
pub use telemetry::AgentTelemetry;

/// Agent version (semver)
pub const AGENT_VERSION: &str = "0.1.0";

/// Agent identifier
pub const AGENT_ID: &str = "latency-analysis-agent";
