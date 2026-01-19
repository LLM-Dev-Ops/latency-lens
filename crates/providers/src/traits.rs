//! Provider trait definitions
//!
//! This module defines the core trait that all LLM provider adapters must implement.
//! The trait is designed to support streaming responses with fine-grained timing
//! measurements for comprehensive latency analysis.

use crate::error::Result;
use async_trait::async_trait;
use futures::Stream;
use llm_latency_lens_core::{RequestId, SessionId, TimingEngine, TokenEvent};
use std::pin::Pin;

/// Configuration for a streaming request
#[derive(Debug, Clone)]
pub struct StreamingRequest {
    /// Unique identifier for this request
    pub request_id: RequestId,
    /// Session this request belongs to
    pub session_id: SessionId,
    /// Model to use for generation
    pub model: String,
    /// Input messages/prompt
    pub messages: Vec<Message>,
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for sampling (0.0 to 2.0)
    pub temperature: Option<f32>,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Request timeout in seconds
    pub timeout_secs: Option<u64>,
}

/// A message in the conversation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    /// Role of the message sender
    pub role: MessageRole,
    /// Content of the message
    pub content: String,
}

/// Role of a message sender
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message (instructions)
    System,
    /// User message (human input)
    User,
    /// Assistant message (AI response)
    Assistant,
}

/// Response from a streaming request
pub struct StreamingResponse {
    /// Request ID
    pub request_id: RequestId,
    /// Stream of token events
    pub token_stream: Pin<Box<dyn Stream<Item = Result<TokenEvent>> + Send>>,
    /// Request metadata
    pub metadata: ResponseMetadata,
}

/// Metadata about the response
#[derive(Debug, Clone)]
pub struct ResponseMetadata {
    /// Model used for generation
    pub model: String,
    /// Input token count (if available)
    pub input_tokens: Option<u64>,
    /// Output token count (updated as tokens arrive)
    pub output_tokens: Option<u64>,
    /// Thinking tokens (Claude extended thinking)
    pub thinking_tokens: Option<u64>,
    /// Estimated cost in USD (if available)
    pub estimated_cost: Option<f64>,
    /// Raw HTTP headers for debugging
    pub headers: Vec<(String, String)>,
}

/// Result of a completed request with timing information
#[derive(Debug, Clone)]
pub struct CompletionResult {
    /// Request ID
    pub request_id: RequestId,
    /// Complete generated text
    pub content: String,
    /// All token events in sequence
    pub token_events: Vec<TokenEvent>,
    /// Response metadata
    pub metadata: ResponseMetadata,
    /// Timing checkpoints
    pub timing_checkpoints: Vec<(String, std::time::Duration)>,
}

impl CompletionResult {
    /// Calculate time to first token (TTFT)
    pub fn ttft(&self) -> Option<std::time::Duration> {
        self.token_events.first().map(|e| e.time_since_start)
    }

    /// Calculate average inter-token latency
    pub fn avg_inter_token_latency(&self) -> Option<std::time::Duration> {
        if self.token_events.len() < 2 {
            return None;
        }

        let sum: std::time::Duration = self
            .token_events
            .iter()
            .filter_map(|e| e.inter_token_latency)
            .sum();

        let count = self.token_events.iter().filter(|e| e.inter_token_latency.is_some()).count();

        if count > 0 {
            Some(sum / count as u32)
        } else {
            None
        }
    }

    /// Calculate median inter-token latency
    pub fn median_inter_token_latency(&self) -> Option<std::time::Duration> {
        let mut latencies: Vec<std::time::Duration> = self
            .token_events
            .iter()
            .filter_map(|e| e.inter_token_latency)
            .collect();

        if latencies.is_empty() {
            return None;
        }

        latencies.sort();
        let mid = latencies.len() / 2;

        if latencies.len() % 2 == 0 {
            Some((latencies[mid - 1] + latencies[mid]) / 2)
        } else {
            Some(latencies[mid])
        }
    }

    /// Calculate p95 inter-token latency
    pub fn p95_inter_token_latency(&self) -> Option<std::time::Duration> {
        let mut latencies: Vec<std::time::Duration> = self
            .token_events
            .iter()
            .filter_map(|e| e.inter_token_latency)
            .collect();

        if latencies.is_empty() {
            return None;
        }

        latencies.sort();
        let idx = (latencies.len() as f64 * 0.95).ceil() as usize - 1;
        Some(latencies[idx.min(latencies.len() - 1)])
    }

    /// Calculate total generation time
    pub fn total_generation_time(&self) -> Option<std::time::Duration> {
        self.token_events.last().map(|e| e.time_since_start)
    }

    /// Calculate tokens per second
    pub fn tokens_per_second(&self) -> Option<f64> {
        if let Some(duration) = self.total_generation_time() {
            let secs = duration.as_secs_f64();
            if secs > 0.0 {
                return Some(self.token_events.len() as f64 / secs);
            }
        }
        None
    }
}

/// Core trait that all LLM provider adapters must implement
#[async_trait]
pub trait Provider: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &'static str;

    /// Check if the provider is properly configured
    async fn health_check(&self) -> Result<()>;

    /// Execute a streaming request
    ///
    /// This method initiates a streaming request and returns a stream of token events.
    /// Each token event includes precise timing information for latency analysis.
    ///
    /// # Arguments
    ///
    /// * `request` - The streaming request configuration
    /// * `timing_engine` - Timing engine for high-precision measurements
    ///
    /// # Returns
    ///
    /// A `StreamingResponse` containing the token stream and metadata
    async fn stream(
        &self,
        request: StreamingRequest,
        timing_engine: &TimingEngine,
    ) -> Result<StreamingResponse>;

    /// Execute a complete request and return all tokens
    ///
    /// This is a convenience method that collects the entire stream into a single result.
    /// Use this when you want to wait for the complete response before processing.
    ///
    /// # Arguments
    ///
    /// * `request` - The streaming request configuration
    /// * `timing_engine` - Timing engine for high-precision measurements
    ///
    /// # Returns
    ///
    /// A `CompletionResult` with the complete response and timing data
    async fn complete(
        &self,
        request: StreamingRequest,
        timing_engine: &TimingEngine,
    ) -> Result<CompletionResult> {
        use futures::StreamExt;

        let request_id = request.request_id;
        let mut response = self.stream(request, timing_engine).await?;

        let mut token_events = Vec::new();
        let mut content = String::new();

        while let Some(event_result) = response.token_stream.next().await {
            let event = event_result?;
            if let Some(ref text) = event.content {
                content.push_str(text);
            }
            token_events.push(event);
        }

        Ok(CompletionResult {
            request_id,
            content,
            token_events,
            metadata: response.metadata,
            timing_checkpoints: Vec::new(), // Will be populated by provider
        })
    }

    /// Calculate the cost of a request
    ///
    /// # Arguments
    ///
    /// * `model` - Model name
    /// * `input_tokens` - Number of input tokens
    /// * `output_tokens` - Number of output tokens
    ///
    /// # Returns
    ///
    /// Estimated cost in USD, or None if pricing is unavailable
    fn calculate_cost(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Option<f64>;

    /// Get supported models for this provider
    fn supported_models(&self) -> Vec<String>;

    /// Validate a model name
    fn validate_model(&self, model: &str) -> Result<()> {
        let supported = self.supported_models();
        if supported.is_empty() || supported.contains(&model.to_string()) {
            Ok(())
        } else {
            Err(crate::error::ProviderError::InvalidModel(format!(
                "Model '{}' is not supported by {}. Supported models: {}",
                model,
                self.name(),
                supported.join(", ")
            )))
        }
    }
}

// Blanket implementation for Box<dyn Provider> to support trait objects
#[async_trait]
impl Provider for Box<dyn Provider> {
    fn name(&self) -> &'static str {
        (**self).name()
    }

    async fn health_check(&self) -> Result<()> {
        (**self).health_check().await
    }

    async fn stream(
        &self,
        request: StreamingRequest,
        timing_engine: &TimingEngine,
    ) -> Result<StreamingResponse> {
        (**self).stream(request, timing_engine).await
    }

    fn calculate_cost(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Option<f64> {
        (**self).calculate_cost(model, input_tokens, output_tokens)
    }

    fn supported_models(&self) -> Vec<String> {
        (**self).supported_models()
    }
}

/// Helper to build a streaming request
impl StreamingRequest {
    /// Create a new streaming request builder
    pub fn builder() -> StreamingRequestBuilder {
        StreamingRequestBuilder::default()
    }
}

/// Builder for streaming requests
#[derive(Default)]
pub struct StreamingRequestBuilder {
    request_id: Option<RequestId>,
    session_id: Option<SessionId>,
    model: Option<String>,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
    timeout_secs: Option<u64>,
}

impl StreamingRequestBuilder {
    /// Set the request ID
    pub fn request_id(mut self, id: RequestId) -> Self {
        self.request_id = Some(id);
        self
    }

    /// Set the session ID
    pub fn session_id(mut self, id: SessionId) -> Self {
        self.session_id = Some(id);
        self
    }

    /// Set the model
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add a message
    pub fn message(mut self, role: MessageRole, content: impl Into<String>) -> Self {
        self.messages.push(Message {
            role,
            content: content.into(),
        });
        self
    }

    /// Add multiple messages
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set top_p
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set stop sequences
    pub fn stop(mut self, sequences: Vec<String>) -> Self {
        self.stop = Some(sequences);
        self
    }

    /// Set timeout in seconds
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Build the request
    pub fn build(self) -> StreamingRequest {
        StreamingRequest {
            request_id: self.request_id.unwrap_or_default(),
            session_id: self.session_id.unwrap_or_default(),
            model: self.model.expect("model is required"),
            messages: self.messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop: self.stop,
            timeout_secs: self.timeout_secs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_request_builder() {
        let request = StreamingRequest::builder()
            .model("gpt-4")
            .message(MessageRole::User, "Hello")
            .max_tokens(100)
            .temperature(0.7)
            .build();

        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_completion_result_ttft() {
        use std::time::Duration;

        let result = CompletionResult {
            request_id: RequestId::new(),
            content: "test".to_string(),
            token_events: vec![
                TokenEvent {
                    request_id: RequestId::new(),
                    sequence: 0,
                    content: Some("Hello".to_string()),
                    timestamp_nanos: 1000000,
                    time_since_start: Duration::from_millis(100),
                    inter_token_latency: None,
                },
                TokenEvent {
                    request_id: RequestId::new(),
                    sequence: 1,
                    content: Some("World".to_string()),
                    timestamp_nanos: 2000000,
                    time_since_start: Duration::from_millis(150),
                    inter_token_latency: Some(Duration::from_millis(50)),
                },
            ],
            metadata: ResponseMetadata {
                model: "test-model".to_string(),
                input_tokens: Some(10),
                output_tokens: Some(2),
                thinking_tokens: None,
                estimated_cost: None,
                headers: vec![],
            },
            timing_checkpoints: vec![],
        };

        assert_eq!(result.ttft(), Some(Duration::from_millis(100)));
        assert_eq!(result.total_generation_time(), Some(Duration::from_millis(150)));
        assert_eq!(result.avg_inter_token_latency(), Some(Duration::from_millis(50)));
    }
}
