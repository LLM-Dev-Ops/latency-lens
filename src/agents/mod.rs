//! LLM-Latency-Lens Agent Infrastructure
//!
//! This module contains agents for the LLM-Latency-Lens diagnostic layer.
//! All agents are MEASUREMENT or ANALYSIS classified and follow the
//! Agentics Dev platform constitution.
//!
//! # Agent Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Edge Function Entry                       │
//! │              (Google Cloud Edge Function)                    │
//! └─────────────────────────┬───────────────────────────────────┘
//!                           │
//!                           ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Agent Runtime                             │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │  Contracts  │  │   Agent     │  │  RuVector Client    │  │
//! │  │ (agentics-  │  │   Logic     │  │  (persistence)      │  │
//! │  │ contracts)  │  │             │  │                     │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//!                           │
//!                           ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    DecisionEvent                             │
//! │              (persisted to ruvector-service)                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Available Agents
//!
//! - [`cold_start_mitigation`]: Measure and characterize cold start behavior (MEASUREMENT)
//! - [`latency_analysis`]: Measure and analyze end-to-end and token-level latency (ANALYSIS)
//!
//! # Agent Classifications
//!
//! All agents are classified as either:
//! - **MEASUREMENT**: Captures raw timing data, does not interpret
//! - **ANALYSIS**: Processes and interprets measurement data
//!
//! # What Agents MUST NEVER DO
//!
//! - Modify system behavior
//! - Trigger remediation or retries
//! - Change routing or apply optimizations
//! - Enforce policies or orchestrate workflows
//! - Connect directly to Google SQL or execute SQL
//!
//! # Edge Function Deployment
//!
//! The [`edge_function`] module provides HTTP handlers for Google Cloud Edge Functions.

pub mod cold_start_mitigation;
pub mod contracts;
pub mod edge_function;
pub mod execution_graph;
pub mod latency_analysis;
pub mod ruvector;

pub use cold_start_mitigation::ColdStartMitigationAgent;
pub use contracts::*;
pub use edge_function::{EdgeFunctionHandler, EdgeFunctionRequest, EdgeFunctionResponse, EdgeOperation};
pub use execution_graph::{
    Artifact, ExecutionContext, ExecutionResult, ExecutionSpan, RepoExecution, SpanStatus, SpanType,
    REPO_NAME,
};
pub use latency_analysis::LatencyAnalysisAgent;
pub use ruvector::RuVectorClient;
