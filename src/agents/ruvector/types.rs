//! RuVector service types

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Response from persisting a DecisionEvent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistResponse {
    /// Event ID that was persisted
    pub event_id: Uuid,
    /// Storage reference (for retrieval)
    pub storage_ref: String,
    /// Timestamp when persisted
    pub persisted_at: chrono::DateTime<chrono::Utc>,
    /// Storage latency in milliseconds
    pub latency_ms: u64,
}

/// Query for retrieving DecisionEvents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventQuery {
    /// Filter by agent ID
    pub agent_id: Option<String>,
    /// Filter by decision type
    pub decision_type: Option<String>,
    /// Filter by time range start
    pub from_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Filter by time range end
    pub to_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Filter by execution reference
    pub execution_ref: Option<String>,
    /// Maximum results to return
    pub limit: Option<u32>,
    /// Offset for pagination
    pub offset: Option<u32>,
}

impl Default for EventQuery {
    fn default() -> Self {
        Self {
            agent_id: None,
            decision_type: None,
            from_time: None,
            to_time: None,
            execution_ref: None,
            limit: Some(100),
            offset: None,
        }
    }
}

impl EventQuery {
    /// Create a new empty query
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by agent ID
    pub fn agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    /// Filter by decision type
    pub fn decision_type(mut self, dt: impl Into<String>) -> Self {
        self.decision_type = Some(dt.into());
        self
    }

    /// Filter by time range
    pub fn time_range(
        mut self,
        from: chrono::DateTime<chrono::Utc>,
        to: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        self.from_time = Some(from);
        self.to_time = Some(to);
        self
    }

    /// Filter by execution reference
    pub fn execution_ref(mut self, exec_ref: impl Into<String>) -> Self {
        self.execution_ref = Some(exec_ref.into());
        self
    }

    /// Set result limit
    pub fn limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set pagination offset
    pub fn offset(mut self, offset: u32) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// Query result containing events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Events matching query
    pub events: Vec<serde_json::Value>,
    /// Total count (for pagination)
    pub total_count: u64,
    /// Whether there are more results
    pub has_more: bool,
}

/// Health status of ruvector-service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Whether service is healthy
    pub healthy: bool,
    /// Service version
    pub version: String,
    /// Database connection status
    pub database_connected: bool,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Current request rate (per second)
    pub request_rate: f64,
}

/// Configuration for RuVector client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuVectorConfig {
    /// Service endpoint URL
    pub endpoint: String,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Enable request compression
    pub compression: bool,
    /// Batch size for bulk operations
    pub batch_size: u32,
}

impl Default for RuVectorConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8080".to_string(),
            api_key: None,
            timeout_ms: 30000, // 30 seconds
            max_retries: 3,
            compression: true,
            batch_size: 100,
        }
    }
}

impl RuVectorConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        Self {
            endpoint: std::env::var("RUVECTOR_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:8080".to_string()),
            api_key: std::env::var("RUVECTOR_API_KEY").ok(),
            timeout_ms: std::env::var("RUVECTOR_TIMEOUT_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30000),
            max_retries: std::env::var("RUVECTOR_MAX_RETRIES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
            compression: std::env::var("RUVECTOR_COMPRESSION")
                .map(|s| s.to_lowercase() == "true")
                .unwrap_or(true),
            batch_size: std::env::var("RUVECTOR_BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100),
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.endpoint.is_empty() {
            return Err("endpoint cannot be empty".to_string());
        }

        if !self.endpoint.starts_with("http://") && !self.endpoint.starts_with("https://") {
            return Err("endpoint must start with http:// or https://".to_string());
        }

        if self.timeout_ms == 0 {
            return Err("timeout_ms must be greater than 0".to_string());
        }

        Ok(())
    }
}
