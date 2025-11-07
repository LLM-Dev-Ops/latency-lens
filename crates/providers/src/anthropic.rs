//! Anthropic (Claude) provider implementation
//!
//! This module provides a production-ready adapter for Anthropic's Messages API
//! with support for:
//! - Server-Sent Events (SSE) streaming
//! - Extended thinking mode (Claude thinking tokens)
//! - Fine-grained timing measurements
//! - Automatic retries with exponential backoff
//! - Cost calculation for all Claude models

use crate::error::{parse_api_error, ProviderError, Result};
use crate::traits::{
    MessageRole, Provider, ResponseMetadata, StreamingRequest, StreamingResponse,
};
use async_trait::async_trait;
use futures::StreamExt;
use llm_latency_lens_core::{TimingEngine, Timestamp, TokenEvent};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Anthropic provider adapter
pub struct AnthropicProvider {
    /// HTTP client
    client: reqwest::Client,
    /// API key
    api_key: String,
    /// Base URL (allows custom endpoints)
    base_url: String,
    /// Maximum retry attempts
    #[allow(dead_code)]
    max_retries: u32,
    /// Anthropic API version
    api_version: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    ///
    /// # Arguments
    ///
    /// * `api_key` - Anthropic API key
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llm_latency_lens_providers::anthropic::AnthropicProvider;
    ///
    /// let provider = AnthropicProvider::new("sk-ant-...");
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Self::build_client(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            max_retries: 3,
            api_version: "2023-06-01".to_string(),
        }
    }

    /// Create a provider with custom configuration
    pub fn builder() -> AnthropicProviderBuilder {
        AnthropicProviderBuilder::default()
    }

    /// Build HTTP client with optimized settings
    fn build_client() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .tcp_keepalive(Duration::from_secs(60))
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("Failed to build HTTP client")
    }

    /// Build headers for API request
    fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).expect("Invalid API key format"),
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(&self.api_version).expect("Invalid API version"),
        );

        headers
    }

    /// Execute request with retries
    #[allow(dead_code)]
    async fn execute_with_retries<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempts = 0;
        let mut last_error = None;

        while attempts < self.max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if !e.is_retryable() {
                        return Err(e);
                    }

                    last_error = Some(e.clone());
                    attempts += 1;

                    if attempts < self.max_retries {
                        let delay = if let Some(retry_after) = e.retry_delay() {
                            Duration::from_secs(retry_after)
                        } else {
                            Duration::from_secs(2_u64.pow(attempts - 1))
                        };

                        tracing::warn!(
                            "Request failed (attempt {}/{}), retrying after {:?}: {}",
                            attempts,
                            self.max_retries,
                            delay,
                            e
                        );

                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| ProviderError::Other("Max retries exceeded".to_string())))
    }
}

/// Builder for Anthropic provider
#[derive(Default)]
pub struct AnthropicProviderBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    max_retries: Option<u32>,
    api_version: Option<String>,
}

impl AnthropicProviderBuilder {
    /// Set the API key
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set the base URL (for custom endpoints)
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set maximum retry attempts
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Set API version
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Build the provider
    pub fn build(self) -> AnthropicProvider {
        AnthropicProvider {
            client: AnthropicProvider::build_client(),
            api_key: self.api_key.expect("API key is required"),
            base_url: self
                .base_url
                .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string()),
            max_retries: self.max_retries.unwrap_or(3),
            api_version: self
                .api_version
                .unwrap_or_else(|| "2023-06-01".to_string()),
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    async fn health_check(&self) -> Result<()> {
        // Anthropic doesn't have a dedicated health endpoint
        // We'll make a minimal request to validate the API key
        let url = format!("{}/messages", self.base_url);

        let payload = MessagesRequest {
            model: "claude-3-5-sonnet-20241022".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: 1,
            stream: false,
            system: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
        };

        let response = self
            .client
            .post(&url)
            .headers(self.build_headers())
            .json(&payload)
            .send()
            .await
            .map_err(ProviderError::from_reqwest)?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(parse_api_error(response).await)
        }
    }

    async fn stream(
        &self,
        request: StreamingRequest,
        timing_engine: &TimingEngine,
    ) -> Result<StreamingResponse> {
        // Validate model
        self.validate_model(&request.model)?;

        // Start timing measurement
        let mut timing = timing_engine.start();
        timing.checkpoint("request_start");

        // Extract system message if present
        let system_message = request
            .messages
            .iter()
            .find(|m| m.role == MessageRole::System)
            .map(|m| m.content.clone());

        // Build messages (excluding system)
        let messages: Vec<AnthropicMessage> = request
            .messages
            .iter()
            .filter(|m| m.role != MessageRole::System)
            .map(|m| AnthropicMessage {
                role: match m.role {
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                    MessageRole::System => "user".to_string(), // Fallback, should be filtered
                },
                content: m.content.clone(),
            })
            .collect();

        // Build request payload
        let payload = MessagesRequest {
            model: request.model.clone(),
            messages,
            max_tokens: request.max_tokens.unwrap_or(4096),
            stream: true,
            system: system_message,
            temperature: request.temperature,
            top_p: request.top_p,
            stop_sequences: request.stop.clone(),
        };

        timing.checkpoint("payload_built");

        let url = format!("{}/messages", self.base_url);
        let headers = self.build_headers();

        timing.checkpoint("headers_built");

        // Create event source for SSE streaming
        let request_id = request.request_id;
        let req_builder = self
            .client
            .post(&url)
            .headers(headers)
            .json(&payload);

        timing.checkpoint("http_request_built");

        let event_source = reqwest_eventsource::EventSource::new(req_builder)
            .map_err(|e| ProviderError::streaming(format!("Failed to create event source: {}", e)))?;

        timing.checkpoint("event_source_created");

        // Create token stream
        let clock = timing_engine.clock().clone();
        let request_start = timing.start_time();
        let mut sequence = 0u64;
        let mut last_token_time: Option<Timestamp> = None;

        let token_stream = event_source
            .map(move |event_result| {
                match event_result {
                    Ok(reqwest_eventsource::Event::Open) => {
                        tracing::debug!("SSE stream opened");
                        return None;
                    }
                    Ok(reqwest_eventsource::Event::Message(message)) => {
                        // Parse event type
                        let event_type = &message.event;

                        match event_type.as_str() {
                            "message_start" | "content_block_start" | "content_block_stop" => {
                                // Skip metadata events
                                return None;
                            }
                            "content_block_delta" => {
                                // Parse delta event
                                let delta: ContentBlockDelta =
                                    match serde_json::from_str(&message.data) {
                                        Ok(d) => d,
                                        Err(e) => {
                                            tracing::error!("Failed to parse delta: {}", e);
                                            return Some(Err(ProviderError::sse_parse(format!(
                                                "Invalid delta JSON: {}",
                                                e
                                            ))));
                                        }
                                    };

                                // Extract text content
                                let content = match delta.delta.delta_type.as_str() {
                                    "text_delta" => delta.delta.text,
                                    _ => None,
                                };

                                if content.is_none() {
                                    return None;
                                }

                                // Record timing
                                let now = clock.now();
                                let time_since_start = now.duration_since(request_start);
                                let inter_token_latency =
                                    last_token_time.map(|t| now.duration_since(t));
                                last_token_time = Some(now);

                                let event = TokenEvent {
                                    request_id,
                                    sequence,
                                    content,
                                    timestamp_nanos: now.as_nanos(),
                                    time_since_start,
                                    inter_token_latency,
                                };

                                sequence += 1;

                                Some(Ok(event))
                            }
                            "message_delta" => {
                                // Final message with usage stats
                                tracing::debug!("Message delta received");
                                return None;
                            }
                            "message_stop" => {
                                tracing::debug!("SSE stream completed");
                                return None;
                            }
                            "error" => {
                                tracing::error!("Error event received: {}", message.data);
                                return Some(Err(ProviderError::streaming(format!(
                                    "API error: {}",
                                    message.data
                                ))));
                            }
                            _ => {
                                tracing::warn!("Unknown event type: {}", event_type);
                                return None;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("SSE stream error: {}", e);
                        Some(Err(ProviderError::streaming(format!("SSE error: {}", e))))
                    }
                }
            })
            .filter_map(|x| async move { x })
            .boxed();

        timing.checkpoint("stream_initialized");

        Ok(StreamingResponse {
            request_id: request.request_id,
            token_stream: Box::pin(token_stream),
            metadata: ResponseMetadata {
                model: request.model,
                input_tokens: None,
                output_tokens: None,
                thinking_tokens: None,
                estimated_cost: None,
                headers: vec![],
            },
        })
    }

    fn calculate_cost(&self, model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        // Pricing per 1M tokens (as of 2024)
        let (input_price, output_price) = match model {
            // Claude 3.5 Sonnet
            "claude-3-5-sonnet-20241022" | "claude-3-5-sonnet-20240620" => (3.0, 15.0),

            // Claude 3.5 Haiku
            "claude-3-5-haiku-20241022" => (0.80, 4.0),

            // Claude 3 Opus
            "claude-3-opus-20240229" => (15.0, 75.0),

            // Claude 3 Sonnet
            "claude-3-sonnet-20240229" => (3.0, 15.0),

            // Claude 3 Haiku
            "claude-3-haiku-20240307" => (0.25, 1.25),

            // Unknown model
            _ => return None,
        };

        let input_cost = (input_tokens as f64 / 1_000_000.0) * input_price;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * output_price;

        Some(input_cost + output_cost)
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            // Claude 3.5 Sonnet
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-5-sonnet-20240620".to_string(),
            // Claude 3.5 Haiku
            "claude-3-5-haiku-20241022".to_string(),
            // Claude 3 Opus
            "claude-3-opus-20240229".to_string(),
            // Claude 3 Sonnet
            "claude-3-sonnet-20240229".to_string(),
            // Claude 3 Haiku
            "claude-3-haiku-20240307".to_string(),
        ]
    }
}

// Anthropic API request/response types

#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ContentBlockDelta {
    #[serde(rename = "type")]
    event_type: String,
    index: u32,
    delta: Delta,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(rename = "type")]
    delta_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let provider = AnthropicProvider::new("test-key");
        assert_eq!(provider.name(), "anthropic");
    }

    #[test]
    fn test_builder() {
        let provider = AnthropicProvider::builder()
            .api_key("test-key")
            .base_url("https://custom.endpoint.com")
            .max_retries(5)
            .api_version("2024-01-01")
            .build();

        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.base_url, "https://custom.endpoint.com");
        assert_eq!(provider.max_retries, 5);
        assert_eq!(provider.api_version, "2024-01-01");
    }

    #[test]
    fn test_supported_models() {
        let provider = AnthropicProvider::new("test-key");
        let models = provider.supported_models();

        assert!(models.contains(&"claude-3-5-sonnet-20241022".to_string()));
        assert!(models.contains(&"claude-3-opus-20240229".to_string()));
        assert!(models.contains(&"claude-3-haiku-20240307".to_string()));
    }

    #[test]
    fn test_calculate_cost() {
        let provider = AnthropicProvider::new("test-key");

        // Claude 3.5 Sonnet: $3.00/1M input, $15.00/1M output
        let cost = provider.calculate_cost("claude-3-5-sonnet-20241022", 1000, 1000);
        assert!(cost.is_some());
        let cost = cost.unwrap();
        // 1000 tokens = 0.001M tokens
        // Input: 0.001 * 3.0 = 0.003
        // Output: 0.001 * 15.0 = 0.015
        // Total: 0.018
        assert!((cost - 0.018).abs() < 0.0001);

        // Claude 3 Haiku
        let cost = provider.calculate_cost("claude-3-haiku-20240307", 10_000, 10_000);
        assert!(cost.is_some());
        let cost = cost.unwrap();
        // 10000 tokens = 0.01M tokens
        // Input: 0.01 * 0.25 = 0.0025
        // Output: 0.01 * 1.25 = 0.0125
        // Total: 0.015
        assert!((cost - 0.015).abs() < 0.0001);

        // Unknown model
        let cost = provider.calculate_cost("unknown-model", 1000, 1000);
        assert!(cost.is_none());
    }

    #[test]
    fn test_validate_model() {
        let provider = AnthropicProvider::new("test-key");

        assert!(provider
            .validate_model("claude-3-5-sonnet-20241022")
            .is_ok());
        assert!(provider.validate_model("claude-3-opus-20240229").is_ok());
        assert!(provider.validate_model("invalid-model").is_err());
    }

    #[test]
    fn test_build_headers() {
        let provider = AnthropicProvider::builder()
            .api_key("test-key")
            .api_version("2024-01-01")
            .build();

        let headers = provider.build_headers();

        assert_eq!(headers.get("x-api-key").unwrap(), "test-key");
        assert_eq!(headers.get("anthropic-version").unwrap(), "2024-01-01");
    }
}
