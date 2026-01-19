//! RuVector Service Client Implementation
//!
//! Async HTTP client for ruvector-service persistence.
//! Handles retries, rate limiting, and batch operations.

use super::{
    error::RuVectorError,
    types::{EventQuery, HealthStatus, PersistResponse, QueryResult, RuVectorConfig},
};
use crate::agents::contracts::DecisionEvent;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// RuVector service client
///
/// Provides async interface for persisting DecisionEvents to ruvector-service.
/// This is the ONLY persistence mechanism for LLM-Latency-Lens agents.
pub struct RuVectorClient {
    /// HTTP client
    client: Client,
    /// Configuration
    config: RuVectorConfig,
    /// Connection state
    connected: Arc<RwLock<bool>>,
}

impl RuVectorClient {
    /// Create a new RuVector client with configuration
    pub fn new(config: RuVectorConfig) -> Result<Self, RuVectorError> {
        config
            .validate()
            .map_err(RuVectorError::InvalidConfig)?;

        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .gzip(config.compression)
            .build()?;

        Ok(Self {
            client,
            config,
            connected: Arc::new(RwLock::new(false)),
        })
    }

    /// Create client from environment variables
    pub fn from_env() -> Result<Self, RuVectorError> {
        Self::new(RuVectorConfig::from_env())
    }

    /// Check connection to ruvector-service
    pub async fn connect(&self) -> Result<(), RuVectorError> {
        let health = self.health_check().await?;

        if !health.healthy {
            return Err(RuVectorError::ServiceError {
                status: 503,
                message: "Service unhealthy".to_string(),
            });
        }

        let mut connected = self.connected.write().await;
        *connected = true;

        info!(
            endpoint = %self.config.endpoint,
            version = %health.version,
            "Connected to ruvector-service"
        );

        Ok(())
    }

    /// Check if client is connected
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    /// Health check endpoint
    pub async fn health_check(&self) -> Result<HealthStatus, RuVectorError> {
        let url = format!("{}/health", self.config.endpoint);

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(RuVectorError::ServiceError {
                status: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        let health: HealthStatus = response.json().await?;
        Ok(health)
    }

    /// Persist a single DecisionEvent
    ///
    /// This is the primary method for agent persistence.
    /// Every agent invocation MUST call this exactly once.
    pub async fn persist_event(&self, event: &DecisionEvent) -> Result<PersistResponse, RuVectorError> {
        // Validate event before sending
        event
            .validate()
            .map_err(|e| RuVectorError::ValidationFailed(e.to_string()))?;

        let url = format!("{}/api/v1/events", self.config.endpoint);

        let mut attempts = 0;
        loop {
            attempts += 1;

            let result = self.send_persist_request(&url, event).await;

            match result {
                Ok(response) => {
                    debug!(
                        event_id = %event.event_id,
                        storage_ref = %response.storage_ref,
                        latency_ms = response.latency_ms,
                        "DecisionEvent persisted"
                    );
                    return Ok(response);
                }
                Err(e) if e.is_retryable() && attempts < self.config.max_retries => {
                    let delay = e.retry_delay_ms().unwrap_or(1000);
                    warn!(
                        error = %e,
                        attempt = attempts,
                        max_retries = self.config.max_retries,
                        retry_delay_ms = delay,
                        "Retrying persist operation"
                    );
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
                Err(e) => {
                    error!(
                        error = %e,
                        event_id = %event.event_id,
                        "Failed to persist DecisionEvent"
                    );
                    return Err(e);
                }
            }
        }
    }

    /// Internal method to send persist request
    async fn send_persist_request(
        &self,
        url: &str,
        event: &DecisionEvent,
    ) -> Result<PersistResponse, RuVectorError> {
        let mut request = self.client.post(url).json(event);

        if let Some(ref api_key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.send().await?;

        if response.status().as_u16() == 429 {
            // Rate limited
            let retry_after = response
                .headers()
                .get("Retry-After")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(1000);

            return Err(RuVectorError::RateLimited {
                retry_after_ms: retry_after,
            });
        }

        if !response.status().is_success() {
            return Err(RuVectorError::ServiceError {
                status: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        let persist_response: PersistResponse = response.json().await?;
        Ok(persist_response)
    }

    /// Persist multiple DecisionEvents in batch
    pub async fn persist_batch(
        &self,
        events: &[DecisionEvent],
    ) -> Result<Vec<PersistResponse>, RuVectorError> {
        // Validate all events first
        for event in events {
            event
                .validate()
                .map_err(|e| RuVectorError::ValidationFailed(e.to_string()))?;
        }

        let url = format!("{}/api/v1/events/batch", self.config.endpoint);

        let mut request = self.client.post(&url).json(events);

        if let Some(ref api_key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(RuVectorError::ServiceError {
                status: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        let responses: Vec<PersistResponse> = response.json().await?;

        debug!(
            count = events.len(),
            "Batch persisted {} DecisionEvents",
            events.len()
        );

        Ok(responses)
    }

    /// Query DecisionEvents
    pub async fn query_events(&self, query: EventQuery) -> Result<QueryResult, RuVectorError> {
        let url = format!("{}/api/v1/events/query", self.config.endpoint);

        let mut request = self.client.post(&url).json(&query);

        if let Some(ref api_key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(RuVectorError::ServiceError {
                status: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        let result: QueryResult = response.json().await?;
        Ok(result)
    }

    /// Get a single event by ID
    pub async fn get_event(&self, event_id: uuid::Uuid) -> Result<Option<serde_json::Value>, RuVectorError> {
        let url = format!("{}/api/v1/events/{}", self.config.endpoint, event_id);

        let mut request = self.client.get(&url);

        if let Some(ref api_key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        let response = request.send().await?;

        if response.status().as_u16() == 404 {
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(RuVectorError::ServiceError {
                status: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        let event: serde_json::Value = response.json().await?;
        Ok(Some(event))
    }

    /// Get configuration
    pub fn config(&self) -> &RuVectorConfig {
        &self.config
    }
}

impl Clone for RuVectorClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            config: self.config.clone(),
            connected: Arc::clone(&self.connected),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = RuVectorConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = RuVectorConfig {
            endpoint: "".to_string(),
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());

        let invalid_config2 = RuVectorConfig {
            endpoint: "ftp://invalid".to_string(),
            ..Default::default()
        };
        assert!(invalid_config2.validate().is_err());
    }

    #[test]
    fn test_client_creation() {
        let config = RuVectorConfig::default();
        let client = RuVectorClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_event_query_builder() {
        let query = EventQuery::new()
            .agent_id("latency-analysis-agent")
            .decision_type("latency_analysis")
            .limit(50);

        assert_eq!(query.agent_id, Some("latency-analysis-agent".to_string()));
        assert_eq!(query.decision_type, Some("latency_analysis".to_string()));
        assert_eq!(query.limit, Some(50));
    }
}
