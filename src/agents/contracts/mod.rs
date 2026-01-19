//! Agentics Contracts for LLM-Latency-Lens
//!
//! This module provides schemas imported from agentics-contracts.
//! All types are validated against the contract specifications.
//!
//! # Contract Compliance
//!
//! - All inputs/outputs must be validated against these schemas
//! - DecisionEvent must be emitted for every invocation
//! - Telemetry must be LLM-Observatory compatible

mod decision_event;
mod latency_schemas;
mod timing_events;

pub use decision_event::*;
pub use latency_schemas::*;
pub use timing_events::*;
