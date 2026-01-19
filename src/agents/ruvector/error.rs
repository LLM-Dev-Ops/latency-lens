//! RuVector client errors

use thiserror::Error;

/// Errors from RuVector service client
#[derive(Debug, Error)]
pub enum RuVectorError {
    /// Network/HTTP error
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Service returned error response
    #[error("Service error: {status} - {message}")]
    ServiceError { status: u16, message: String },

    /// Request timeout
    #[error("Request timeout after {0}ms")]
    Timeout(u64),

    /// Rate limited
    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Event validation failed
    #[error("Event validation failed: {0}")]
    ValidationFailed(String),

    /// Client not connected
    #[error("Client not connected to ruvector-service")]
    NotConnected,
}

impl RuVectorError {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            RuVectorError::Network(_)
                | RuVectorError::Timeout(_)
                | RuVectorError::RateLimited { .. }
        )
    }

    /// Get suggested retry delay in milliseconds
    pub fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            RuVectorError::RateLimited { retry_after_ms } => Some(*retry_after_ms),
            RuVectorError::Network(_) => Some(1000), // 1 second default
            RuVectorError::Timeout(_) => Some(2000), // 2 seconds for timeout
            _ => None,
        }
    }
}
