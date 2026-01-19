//! RuVector Service Client
//!
//! Client for persisting data to ruvector-service.
//! This is the ONLY way LLM-Latency-Lens persists data.
//!
//! # Architecture
//!
//! ```text
//! LLM-Latency-Lens ──► RuVector Client ──► ruvector-service ──► Google SQL (Postgres)
//! ```
//!
//! LLM-Latency-Lens NEVER connects directly to Google SQL.
//! All persistence occurs via ruvector-service client calls.

mod client;
mod error;
mod types;

pub use client::RuVectorClient;
pub use error::RuVectorError;
pub use types::*;
