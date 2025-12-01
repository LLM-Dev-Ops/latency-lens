//! LLM Latency Lens - Library for programmatic use
//!
//! This library provides high-level APIs for LLM performance measurement
//! that can be used in other Rust applications.
//!
//! # Example
//!
//! ```no_run
//! use llm_latency_lens::{ProfileBuilder, BenchmarkBuilder};
//! use llm_latency_lens_providers::OpenAIProvider;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create a provider
//!     let provider = OpenAIProvider::new("sk-...");
//!
//!     // Profile a single request
//!     let metrics = ProfileBuilder::new(provider.clone())
//!         .model("gpt-4o")
//!         .prompt("Explain quantum computing")
//!         .max_tokens(500)
//!         .execute()
//!         .await?;
//!
//!     println!("TTFT: {:?}", metrics.ttft);
//!     println!("Total: {:?}", metrics.total_latency);
//!
//!     // Run a benchmark
//!     let results = BenchmarkBuilder::new(provider)
//!         .model("gpt-4o")
//!         .prompt("Hello, world!")
//!         .requests(10)
//!         .concurrency(2)
//!         .execute()
//!         .await?;
//!
//!     println!("Mean TTFT: {:?}", results.ttft_distribution.mean);
//!     println!("P95 TTFT: {:?}", results.ttft_distribution.p95);
//!
//!     Ok(())
//! }
//! ```

pub mod adapters;
pub mod benchmarks;
pub mod config;
pub mod orchestrator;

// Re-export core types for convenience
pub use llm_latency_lens_core::{
    Provider as CoreProvider, RequestId, SessionId, TimingEngine,
};
pub use llm_latency_lens_exporters::{
    ConsoleExporter, CsvExporter, Exporter, JsonExporter, PrometheusExporter,
};
pub use llm_latency_lens_metrics::{
    AggregatedMetrics, CollectorConfig, LatencyDistribution, MetricsAggregator,
    MetricsCollector, RequestMetrics, ThroughputStats,
};
pub use llm_latency_lens_providers::{
    AnthropicProvider, CompletionResult, GoogleProvider, Message, MessageRole,
    OpenAIProvider, Provider, ResponseMetadata, StreamingRequest, StreamingResponse,
};

// Re-export canonical benchmark interface
pub use adapters::{all_targets, create_target, create_targets, BenchTarget, LLMTarget};
pub use benchmarks::{
    run_all_benchmarks, BenchmarkConfig, BenchmarkResult,
    ensure_output_dirs, generate_summary, output_dir, raw_output_dir,
    read_all_results, write_result, write_results, write_summary,
};

use anyhow::Result;
use std::sync::Arc;

use orchestrator::{ExecutionSummary, Orchestrator, OrchestratorConfig};

/// Builder for profiling a single request
pub struct ProfileBuilder<P: Provider> {
    provider: Arc<P>,
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    timeout_secs: Option<u64>,
}

impl<P: Provider + 'static> ProfileBuilder<P> {
    /// Create a new profile builder
    pub fn new(provider: P) -> Self {
        Self {
            provider: Arc::new(provider),
            model: String::new(),
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            timeout_secs: None,
        }
    }

    /// Set the model
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set a simple prompt
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.messages = vec![Message {
            role: MessageRole::User,
            content: prompt.into(),
        }];
        self
    }

    /// Set messages
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

    /// Set timeout in seconds
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Execute the profile
    pub async fn execute(self) -> Result<RequestMetrics> {
        if self.model.is_empty() {
            anyhow::bail!("Model is required");
        }

        if self.messages.is_empty() {
            anyhow::bail!("Messages/prompt is required");
        }

        // Build request
        let mut request_builder = StreamingRequest::builder()
            .model(self.model)
            .messages(self.messages);

        if let Some(max_tokens) = self.max_tokens {
            request_builder = request_builder.max_tokens(max_tokens);
        }

        if let Some(temperature) = self.temperature {
            request_builder = request_builder.temperature(temperature);
        }

        if let Some(top_p) = self.top_p {
            request_builder = request_builder.top_p(top_p);
        }

        if let Some(timeout) = self.timeout_secs {
            request_builder = request_builder.timeout_secs(timeout);
        }

        let request = request_builder.build();

        // Create orchestrator
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let config = OrchestratorConfig::default();
        let orchestrator = Orchestrator::new(config, shutdown);

        // Execute single request
        orchestrator.execute_single(&*self.provider, request).await
    }
}

/// Builder for running benchmarks
pub struct BenchmarkBuilder<P: Provider> {
    provider: Arc<P>,
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    timeout_secs: Option<u64>,
    requests: u32,
    concurrency: u32,
    rate_limit: u32,
    show_progress: bool,
}

impl<P: Provider + 'static> BenchmarkBuilder<P> {
    /// Create a new benchmark builder
    pub fn new(provider: P) -> Self {
        Self {
            provider: Arc::new(provider),
            model: String::new(),
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            timeout_secs: None,
            requests: 10,
            concurrency: 1,
            rate_limit: 0,
            show_progress: true,
        }
    }

    /// Set the model
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set a simple prompt
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.messages = vec![Message {
            role: MessageRole::User,
            content: prompt.into(),
        }];
        self
    }

    /// Set messages
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

    /// Set timeout in seconds
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Set number of requests
    pub fn requests(mut self, n: u32) -> Self {
        self.requests = n;
        self
    }

    /// Set concurrency level
    pub fn concurrency(mut self, n: u32) -> Self {
        self.concurrency = n;
        self
    }

    /// Set rate limit (requests per second)
    pub fn rate_limit(mut self, rps: u32) -> Self {
        self.rate_limit = rps;
        self
    }

    /// Show progress bars
    pub fn show_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// Execute the benchmark
    pub async fn execute(self) -> Result<BenchmarkResults> {
        if self.model.is_empty() {
            anyhow::bail!("Model is required");
        }

        if self.messages.is_empty() {
            anyhow::bail!("Messages/prompt is required");
        }

        // Build request template
        let mut request_builder = StreamingRequest::builder()
            .model(self.model)
            .messages(self.messages);

        if let Some(max_tokens) = self.max_tokens {
            request_builder = request_builder.max_tokens(max_tokens);
        }

        if let Some(temperature) = self.temperature {
            request_builder = request_builder.temperature(temperature);
        }

        if let Some(top_p) = self.top_p {
            request_builder = request_builder.top_p(top_p);
        }

        if let Some(timeout) = self.timeout_secs {
            request_builder = request_builder.timeout_secs(timeout);
        }

        let request = request_builder.build();

        // Create orchestrator
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let config = OrchestratorConfig {
            concurrency: self.concurrency,
            total_requests: self.requests,
            rate_limit: self.rate_limit,
            show_progress: self.show_progress,
            shutdown_timeout: std::time::Duration::from_secs(30),
        };
        let orchestrator = Orchestrator::new(config, shutdown);

        // Create metrics collector
        let session_id = orchestrator.session_id();
        let collector = Arc::new(MetricsCollector::with_defaults(session_id)?);

        // Execute benchmark
        let summary = orchestrator
            .execute(self.provider, request, Arc::clone(&collector))
            .await?;

        // Aggregate metrics
        let aggregated = MetricsAggregator::aggregate(&collector)?;

        Ok(BenchmarkResults {
            summary,
            metrics: aggregated,
        })
    }
}

/// Results from a benchmark run
pub struct BenchmarkResults {
    pub summary: ExecutionSummary,
    pub metrics: AggregatedMetrics,
}

impl BenchmarkResults {
    /// Get TTFT distribution
    pub fn ttft_distribution(&self) -> &LatencyDistribution {
        &self.metrics.ttft_distribution
    }

    /// Get inter-token latency distribution
    pub fn inter_token_distribution(&self) -> &LatencyDistribution {
        &self.metrics.inter_token_distribution
    }

    /// Get total latency distribution
    pub fn total_latency_distribution(&self) -> &LatencyDistribution {
        &self.metrics.total_latency_distribution
    }

    /// Get throughput stats
    pub fn throughput(&self) -> &ThroughputStats {
        &self.metrics.throughput
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        self.metrics.success_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_builder() {
        // This is a compile-time test to ensure the builder API works
        // We can't actually execute it without a real provider
    }

    #[test]
    fn test_benchmark_builder() {
        // This is a compile-time test to ensure the builder API works
        // We can't actually execute it without a real provider
    }
}
