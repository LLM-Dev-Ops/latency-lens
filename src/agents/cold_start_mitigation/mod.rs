//! Cold Start Mitigation Agent for LLM-Latency-Lens
//!
//! This agent measures and characterizes cold start behavior and startup latency
//! of LLM services. It is classified as a MEASUREMENT agent.
//!
//! # Agent Contract
//!
//! - **Classification**: MEASUREMENT
//! - **Decision Type**: `cold_start_measurement`
//! - **Purpose**: Detect cold vs warm execution paths, measure initialization delays,
//!   quantify cold start frequency and impact
//!
//! # What This Agent DOES:
//!
//! - Detect cold start vs warm request execution
//! - Measure initialization and startup delays
//! - Quantify cold start frequency and latency impact
//! - Produce diagnostic outputs for analysis
//! - Emit telemetry compatible with LLM-Observatory
//! - Emit DecisionEvents to ruvector-service
//!
//! # What This Agent MUST NEVER DO:
//!
//! - Trigger warm-up logic
//! - Apply mitigations
//! - Modify deployment behavior
//! - Route traffic
//! - Execute SQL directly
//! - Orchestrate workflows
//!
//! # CLI Invocation
//!
//! ```bash
//! llm-latency-lens cold-start profile --provider openai --model gpt-4
//! llm-latency-lens cold-start inspect --session-id <uuid>
//! llm-latency-lens cold-start replay --trace-id <uuid>
//! ```

mod agent;
mod detector;
mod schemas;

pub use agent::ColdStartMitigationAgent;
pub use detector::{ColdStartDetector, DetectionResult};
pub use schemas::*;
