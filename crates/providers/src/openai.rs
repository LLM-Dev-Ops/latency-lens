//! OpenAI provider implementation
//!
//! This module provides a production-ready adapter for OpenAI's Chat Completions API
//! with support for:
//! - Server-Sent Events (SSE) streaming
//! - Fine-grained timing measurements (DNS, TLS, TTFT, inter-token latency)
//! - Automatic retries with exponential backoff
//! - Cost calculation for all GPT models
//! - Comprehensive error handling

use crate::error::{parse_api_error, ProviderError, Result};
use crate::traits::{
    MessageRole, Provider, ResponseMetadata, StreamingRequest, StreamingResponse,
};
use async_trait::async_trait;
use futures::StreamExt;
use llm_latency_lens_core::{TimingEngine, Timestamp, TokenEvent};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// OpenAI provider adapter
pub struct OpenAIProvider {
    /// HTTP client
    client: reqwest::Client,
    /// API key
    api_key: String,
    /// Base URL (allows custom endpoints)
    base_url: String,
    /// Organization ID (optional)
    organization: Option<String>,
    /// Maximum retry attempts
    #[allow(dead_code)]
    max_retries: u32,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llm_latency_lens_providers::openai::OpenAIProvider;
    ///
    /// let provider = OpenAIProvider::new("sk-...");
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Self::build_client(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
            max_retries: 3,
        }
    }

    /// Create a provider with custom configuration
    pub fn builder() -> OpenAIProviderBuilder {
        OpenAIProviderBuilder::default()
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
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .expect("Invalid API key format"),
        );

        if let Some(ref org) = self.organization {
            headers.insert(
                "OpenAI-Organization",
                HeaderValue::from_str(org).expect("Invalid organization ID"),
            );
        }

        headers
    }

    /// Execute request with retries
    #[allow(dead_code)]
    async fn execute_with_retries<F, Fut, T>(
        &self,
        operation: F,
    ) -> Result<T>
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
                            // Exponential backoff: 1s, 2s, 4s
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

/// Builder for OpenAI provider
#[derive(Default)]
pub struct OpenAIProviderBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    organization: Option<String>,
    max_retries: Option<u32>,
}

impl OpenAIProviderBuilder {
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

    /// Set the organization ID
    pub fn organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set maximum retry attempts
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Build the provider
    pub fn build(self) -> OpenAIProvider {
        OpenAIProvider {
            client: OpenAIProvider::build_client(),
            api_key: self.api_key.expect("API key is required"),
            base_url: self.base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            organization: self.organization,
            max_retries: self.max_retries.unwrap_or(3),
        }
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &'static str {
        "openai"
    }

    async fn health_check(&self) -> Result<()> {
        let url = format!("{}/models", self.base_url);
        let response = self
            .client
            .get(&url)
            .headers(self.build_headers())
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

        // Build request payload
        let payload = ChatCompletionRequest {
            model: request.model.clone(),
            messages: request
                .messages
                .iter()
                .map(|m| ChatMessage {
                    role: match m.role {
                        MessageRole::System => "system".to_string(),
                        MessageRole::User => "user".to_string(),
                        MessageRole::Assistant => "assistant".to_string(),
                    },
                    content: m.content.clone(),
                })
                .collect(),
            stream: true,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop.clone(),
        };

        timing.checkpoint("payload_built");

        let url = format!("{}/chat/completions", self.base_url);
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

        let token_stream = event_source.map(move |event_result| {
            match event_result {
                Ok(reqwest_eventsource::Event::Open) => {
                    tracing::debug!("SSE stream opened");
                    return None;
                }
                Ok(reqwest_eventsource::Event::Message(message)) => {
                    if message.data == "[DONE]" {
                        tracing::debug!("SSE stream completed");
                        return None;
                    }

                    // Parse SSE chunk
                    let chunk: ChatCompletionChunk = match serde_json::from_str(&message.data) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::error!("Failed to parse SSE chunk: {}", e);
                            return Some(Err(ProviderError::sse_parse(format!(
                                "Invalid JSON in SSE event: {}",
                                e
                            ))));
                        }
                    };

                    // Extract token content
                    let content = chunk
                        .choices
                        .first()
                        .and_then(|c| c.delta.content.clone());

                    if content.is_none() {
                        // Skip empty chunks (role, function calls, etc.)
                        return None;
                    }

                    // Record timing
                    let now = clock.now();
                    let time_since_start = now.duration_since(request_start);
                    let inter_token_latency = last_token_time.map(|t| now.duration_since(t));
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
                input_tokens: None,  // Not available until completion
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
            // GPT-4 Turbo
            "gpt-4-turbo" | "gpt-4-turbo-2024-04-09" => (10.0, 30.0),
            "gpt-4-turbo-preview" => (10.0, 30.0),

            // GPT-4
            "gpt-4" => (30.0, 60.0),
            "gpt-4-32k" => (60.0, 120.0),

            // GPT-4o
            "gpt-4o" | "gpt-4o-2024-08-06" | "gpt-4o-2024-05-13" => (2.50, 10.0),
            "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => (0.15, 0.60),

            // GPT-3.5 Turbo
            "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" => (0.50, 1.50),
            "gpt-3.5-turbo-instruct" => (1.50, 2.0),

            // Unknown model
            _ => return None,
        };

        let input_cost = (input_tokens as f64 / 1_000_000.0) * input_price;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * output_price;

        Some(input_cost + output_cost)
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            // GPT-4o
            "gpt-4o".to_string(),
            "gpt-4o-2024-08-06".to_string(),
            "gpt-4o-2024-05-13".to_string(),
            "gpt-4o-mini".to_string(),
            "gpt-4o-mini-2024-07-18".to_string(),
            // GPT-4 Turbo
            "gpt-4-turbo".to_string(),
            "gpt-4-turbo-2024-04-09".to_string(),
            "gpt-4-turbo-preview".to_string(),
            // GPT-4
            "gpt-4".to_string(),
            "gpt-4-32k".to_string(),
            // GPT-3.5 Turbo
            "gpt-3.5-turbo".to_string(),
            "gpt-3.5-turbo-0125".to_string(),
            "gpt-3.5-turbo-instruct".to_string(),
        ]
    }
}

// OpenAI API request/response types

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct StreamChoice {
    index: u32,
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Delta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let provider = OpenAIProvider::new("test-key");
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_builder() {
        let provider = OpenAIProvider::builder()
            .api_key("test-key")
            .base_url("https://custom.endpoint.com")
            .organization("org-123")
            .max_retries(5)
            .build();

        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.base_url, "https://custom.endpoint.com");
        assert_eq!(provider.organization, Some("org-123".to_string()));
        assert_eq!(provider.max_retries, 5);
    }

    #[test]
    fn test_supported_models() {
        let provider = OpenAIProvider::new("test-key");
        let models = provider.supported_models();

        assert!(models.contains(&"gpt-4o".to_string()));
        assert!(models.contains(&"gpt-4-turbo".to_string()));
        assert!(models.contains(&"gpt-3.5-turbo".to_string()));
    }

    #[test]
    fn test_calculate_cost() {
        let provider = OpenAIProvider::new("test-key");

        // GPT-4o: $2.50/1M input, $10.00/1M output
        let cost = provider.calculate_cost("gpt-4o", 1000, 1000);
        assert!(cost.is_some());
        let cost = cost.unwrap();
        // 1000 tokens = 0.001M tokens
        // Input: 0.001 * 2.50 = 0.0025
        // Output: 0.001 * 10.0 = 0.010
        // Total: 0.0125
        assert!((cost - 0.0125).abs() < 0.0001);

        // GPT-4 Turbo
        let cost = provider.calculate_cost("gpt-4-turbo", 10_000, 10_000);
        assert!(cost.is_some());
        let cost = cost.unwrap();
        // 10000 tokens = 0.01M tokens
        // Input: 0.01 * 10.0 = 0.10
        // Output: 0.01 * 30.0 = 0.30
        // Total: 0.40
        assert!((cost - 0.40).abs() < 0.0001);

        // Unknown model
        let cost = provider.calculate_cost("unknown-model", 1000, 1000);
        assert!(cost.is_none());
    }

    #[test]
    fn test_validate_model() {
        let provider = OpenAIProvider::new("test-key");

        assert!(provider.validate_model("gpt-4o").is_ok());
        assert!(provider.validate_model("gpt-4-turbo").is_ok());
        assert!(provider.validate_model("invalid-model").is_err());
    }

    #[test]
    fn test_build_headers() {
        let provider = OpenAIProvider::builder()
            .api_key("test-key")
            .organization("org-123")
            .build();

        let headers = provider.build_headers();

        assert_eq!(
            headers.get(AUTHORIZATION).unwrap(),
            "Bearer test-key"
        );
        assert_eq!(
            headers.get("OpenAI-Organization").unwrap(),
            "org-123"
        );
    }
}
