//! Benchmark target adapters for LLM Latency Lens
//!
//! This module provides the canonical BenchTarget trait and registry
//! required for standardized benchmark execution across providers.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use super::benchmarks::{BenchmarkConfig, BenchmarkResult};
use super::config::Config;
use super::orchestrator::{Orchestrator, OrchestratorConfig};
use llm_latency_lens_metrics::{MetricsAggregator, MetricsCollector};
use llm_latency_lens_providers::{create_provider, MessageRole, StreamingRequest};

/// Canonical trait for benchmark targets
///
/// All benchmark targets must implement this trait to be compatible
/// with the standardized benchmark interface.
#[async_trait]
pub trait BenchTarget: Send + Sync {
    /// Get the unique identifier for this target
    /// Format: "{provider}:{model}" (e.g., "openai:gpt-4o")
    fn id(&self) -> String;

    /// Run the benchmark for this target
    ///
    /// # Arguments
    ///
    /// * `config` - Application configuration with credentials
    /// * `bench_config` - Benchmark-specific configuration
    /// * `shutdown_signal` - Signal for graceful shutdown
    ///
    /// # Returns
    ///
    /// A `BenchmarkResult` containing the benchmark metrics
    async fn run(
        &self,
        config: &Config,
        bench_config: &BenchmarkConfig,
        shutdown_signal: Arc<tokio::sync::Notify>,
    ) -> Result<BenchmarkResult>;
}

/// LLM Provider benchmark target
///
/// Represents a specific provider:model combination for benchmarking
pub struct LLMTarget {
    provider: String,
    model: String,
}

impl LLMTarget {
    /// Create a new LLM target
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// Create from a combined "provider:model" string
    pub fn from_target_string(target: &str) -> Option<Self> {
        let parts: Vec<&str> = target.splitn(2, ':').collect();
        if parts.len() == 2 {
            Some(Self::new(parts[0], parts[1]))
        } else {
            None
        }
    }

    /// Get the provider name
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Get the model name
    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl BenchTarget for LLMTarget {
    fn id(&self) -> String {
        format!("{}:{}", self.provider, self.model)
    }

    async fn run(
        &self,
        config: &Config,
        bench_config: &BenchmarkConfig,
        shutdown_signal: Arc<tokio::sync::Notify>,
    ) -> Result<BenchmarkResult> {
        // Get provider configuration
        let provider_config = config.get_provider(&self.provider)?;

        let api_key = provider_config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("API key not found for provider: {}", self.provider))?;

        // Create provider
        let provider = Arc::new(create_provider(&self.provider, api_key.clone())?);

        // Build request template
        let request_template = StreamingRequest::builder()
            .model(&self.model)
            .message(MessageRole::User, &bench_config.default_prompt)
            .max_tokens(bench_config.max_tokens)
            .temperature(bench_config.temperature)
            .timeout_secs(bench_config.timeout_secs)
            .build();

        // Create orchestrator
        let orchestrator_config = OrchestratorConfig {
            concurrency: bench_config.concurrency,
            total_requests: bench_config.requests,
            rate_limit: bench_config.rate_limit,
            show_progress: bench_config.show_progress,
            shutdown_timeout: std::time::Duration::from_secs(30),
        };

        let orchestrator = Orchestrator::new(orchestrator_config, Arc::clone(&shutdown_signal));
        let session_id = orchestrator.session_id();

        // Create metrics collector
        let collector = Arc::new(MetricsCollector::with_defaults(session_id)?);

        // Run warmup if configured
        if bench_config.warmup > 0 {
            let warmup_config = OrchestratorConfig {
                concurrency: bench_config.concurrency,
                total_requests: bench_config.warmup,
                rate_limit: bench_config.rate_limit,
                show_progress: false,
                shutdown_timeout: std::time::Duration::from_secs(30),
            };

            let warmup_orchestrator =
                Orchestrator::new(warmup_config, Arc::clone(&shutdown_signal));
            let warmup_collector =
                Arc::new(MetricsCollector::with_defaults(warmup_orchestrator.session_id())?);

            let _ = warmup_orchestrator
                .execute(Arc::clone(&provider), request_template.clone(), warmup_collector)
                .await;
        }

        // Execute benchmark
        let _summary = orchestrator
            .execute(provider, request_template, Arc::clone(&collector))
            .await?;

        // Aggregate metrics
        let aggregated = MetricsAggregator::aggregate(&collector)?;

        // Convert to canonical BenchmarkResult
        BenchmarkResult::from_aggregated_metrics(self.id(), &aggregated)
            .map_err(|e| anyhow::anyhow!("Failed to create BenchmarkResult: {}", e))
    }
}

/// Get all configured benchmark targets
///
/// This function returns a registry of all available benchmark targets
/// based on the application configuration.
///
/// # Arguments
///
/// * `config` - Application configuration
///
/// # Returns
///
/// A vector of boxed BenchTarget implementations
pub fn all_targets(config: &Config) -> Vec<Box<dyn BenchTarget>> {
    let mut targets: Vec<Box<dyn BenchTarget>> = Vec::new();

    // Add targets based on configured providers
    for (provider_name, provider_config) in config.providers.iter() {
        // Only add targets for providers with API keys configured
        if provider_config.api_key.is_some() {
            // Get default or recommended models for each provider
            let models = get_default_models(provider_name);

            for model in models {
                targets.push(Box::new(LLMTarget::new(provider_name.clone(), model)));
            }
        }
    }

    targets
}

/// Get default/recommended models for a provider
fn get_default_models(provider: &str) -> Vec<String> {
    match provider.to_lowercase().as_str() {
        "openai" => vec![
            "gpt-4o".to_string(),
            "gpt-4o-mini".to_string(),
            "gpt-4-turbo".to_string(),
        ],
        "anthropic" => vec![
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-5-haiku-20241022".to_string(),
            "claude-3-opus-20240229".to_string(),
        ],
        "google" => vec![
            "gemini-1.5-pro".to_string(),
            "gemini-1.5-flash".to_string(),
        ],
        _ => vec![],
    }
}

/// Create a single benchmark target from a target string
pub fn create_target(target_string: &str) -> Option<Box<dyn BenchTarget>> {
    LLMTarget::from_target_string(target_string).map(|t| Box::new(t) as Box<dyn BenchTarget>)
}

/// Create multiple benchmark targets from target strings
pub fn create_targets(target_strings: &[String]) -> Vec<Box<dyn BenchTarget>> {
    target_strings
        .iter()
        .filter_map(|s| create_target(s))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_target_creation() {
        let target = LLMTarget::new("openai", "gpt-4o");

        assert_eq!(target.provider(), "openai");
        assert_eq!(target.model(), "gpt-4o");
        assert_eq!(target.id(), "openai:gpt-4o");
    }

    #[test]
    fn test_llm_target_from_string() {
        let target = LLMTarget::from_target_string("anthropic:claude-3-5-sonnet-20241022").unwrap();

        assert_eq!(target.provider(), "anthropic");
        assert_eq!(target.model(), "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn test_llm_target_from_invalid_string() {
        assert!(LLMTarget::from_target_string("invalid").is_none());
    }

    #[test]
    fn test_get_default_models() {
        let openai_models = get_default_models("openai");
        assert!(openai_models.contains(&"gpt-4o".to_string()));

        let anthropic_models = get_default_models("anthropic");
        assert!(anthropic_models.contains(&"claude-3-5-sonnet-20241022".to_string()));

        let unknown_models = get_default_models("unknown");
        assert!(unknown_models.is_empty());
    }

    #[test]
    fn test_create_target() {
        let target = create_target("openai:gpt-4o").unwrap();
        assert_eq!(target.id(), "openai:gpt-4o");
    }

    #[test]
    fn test_create_targets() {
        let target_strings = vec![
            "openai:gpt-4o".to_string(),
            "anthropic:claude-3-5-sonnet-20241022".to_string(),
        ];
        let targets = create_targets(&target_strings);

        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].id(), "openai:gpt-4o");
        assert_eq!(targets[1].id(), "anthropic:claude-3-5-sonnet-20241022");
    }
}
