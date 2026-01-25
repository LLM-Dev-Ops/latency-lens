//! Phase 2 - Operational Intelligence (Layer 1)
//!
//! This module provides the Phase 2 runtime infrastructure for the LLM-Latency-Lens
//! operational intelligence layer.
//!
//! # Key Components
//!
//! - **Startup Hardening**: Fail-fast validation of required configuration
//! - **Signal Emission**: Atomic signals for anomaly, drift, lineage, latency
//! - **Caching**: TTL-based cache for historical reads and lineage lookups
//! - **Performance Budgets**: Enforced limits on tokens, latency, and API calls
//!
//! # Environment Requirements
//!
//! The following environment variables are REQUIRED:
//! - `RUVECTOR_SERVICE_URL`: RuVector service endpoint
//! - `RUVECTOR_API_KEY`: API key for RuVector (from Google Secret Manager)
//! - `AGENT_NAME`: Name of this agent instance
//! - `AGENT_DOMAIN`: Domain classification for the agent
//! - `AGENT_PHASE`: Must be "phase2"
//! - `AGENT_LAYER`: Must be "layer1"
//!
//! # Startup Behavior
//!
//! On startup, Phase 2 agents:
//! 1. Validate all required environment variables
//! 2. Initialize and verify RuVector connection
//! 3. Fail fast if RuVector is unavailable
//! 4. Initialize caching layer
//! 5. Register signal emitters

pub mod agent_config;
pub mod cache;
pub mod runtime;
pub mod signals;

pub use agent_config::{AgentConfig, PerformanceBudget, Phase2Config};
pub use cache::{CacheConfig, LineageCache};
pub use runtime::{Phase2Runtime, RuntimeError, StartupValidation};
pub use signals::{
    AnomalySignal, DriftSignal, LatencySignal, LineageSignal, Signal, SignalEmitter,
    SignalMetadata,
};
