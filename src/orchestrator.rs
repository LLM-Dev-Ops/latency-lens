//! Request orchestrator for concurrent execution with rate limiting

use anyhow::{Context, Result};
use futures::stream::{FuturesUnordered, StreamExt};
use governor::{Quota, RateLimiter};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::Instant;
use tracing::{debug, info, warn};

use llm_latency_lens_core::{RequestId, SessionId, TimingEngine};
use llm_latency_lens_metrics::{MetricsCollector, RequestMetrics};
use llm_latency_lens_providers::{
    Provider, StreamingRequest,
};

/// Configuration for the orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Number of concurrent requests
    pub concurrency: u32,
    /// Total number of requests to execute
    pub total_requests: u32,
    /// Rate limit (requests per second, 0 = unlimited)
    pub rate_limit: u32,
    /// Show progress bars
    pub show_progress: bool,
    /// Graceful shutdown timeout
    pub shutdown_timeout: Duration,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            concurrency: 1,
            total_requests: 1,
            rate_limit: 0,
            show_progress: true,
            shutdown_timeout: Duration::from_secs(30),
        }
    }
}

/// Request orchestrator for managing concurrent LLM requests
pub struct Orchestrator {
    config: OrchestratorConfig,
    timing_engine: Arc<TimingEngine>,
    session_id: SessionId,
    shutdown_signal: Arc<tokio::sync::Notify>,
}

impl Orchestrator {
    /// Create a new orchestrator
    pub fn new(
        config: OrchestratorConfig,
        shutdown_signal: Arc<tokio::sync::Notify>,
    ) -> Self {
        Self {
            config,
            timing_engine: Arc::new(TimingEngine::new()),
            session_id: SessionId::new(),
            shutdown_signal,
        }
    }

    /// Get the session ID
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Execute multiple requests with the given provider
    pub async fn execute<P: Provider + 'static>(
        &self,
        provider: Arc<P>,
        request_template: StreamingRequest,
        collector: Arc<MetricsCollector>,
    ) -> Result<ExecutionSummary> {
        info!(
            "Starting orchestration: {} requests with concurrency {}",
            self.config.total_requests, self.config.concurrency
        );

        let start_time = Instant::now();

        // Create progress bars
        let multi_progress = if self.config.show_progress {
            Some(Arc::new(MultiProgress::new()))
        } else {
            None
        };

        let progress_bar = if let Some(ref mp) = multi_progress {
            let pb = mp.add(ProgressBar::new(self.config.total_requests as u64));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        // Create semaphore for concurrency control
        let semaphore = Arc::new(Semaphore::new(self.config.concurrency as usize));

        // Create rate limiter if needed
        let rate_limiter = if self.config.rate_limit > 0 {
            let quota = Quota::per_second(
                NonZeroU32::new(self.config.rate_limit)
                    .context("Invalid rate limit")?,
            );
            Some(Arc::new(RateLimiter::direct(quota)))
        } else {
            None
        };

        // Track execution statistics
        let mut summary = ExecutionSummary::default();
        summary.total_requests = self.config.total_requests;

        // Create tasks for all requests
        let mut tasks = FuturesUnordered::new();

        for i in 0..self.config.total_requests {
            let provider = Arc::clone(&provider);
            let timing_engine = Arc::clone(&self.timing_engine);
            let collector = Arc::clone(&collector);
            let semaphore = Arc::clone(&semaphore);
            let rate_limiter = rate_limiter.clone();
            let progress_bar = progress_bar.clone();
            let shutdown_signal = Arc::clone(&self.shutdown_signal);

            // Clone request template and assign new ID
            let mut request = request_template.clone();
            request.request_id = RequestId::new();
            request.session_id = self.session_id;

            let task = tokio::spawn(async move {
                // Check for shutdown signal
                tokio::select! {
                    _ = shutdown_signal.notified() => {
                        debug!("Request {} cancelled due to shutdown", i);
                        return Err(anyhow::anyhow!("Cancelled"));
                    }
                    result = async {
                        // Apply rate limiting
                        if let Some(limiter) = rate_limiter {
                            limiter.until_ready().await;
                        }

                        // Acquire semaphore for concurrency control
                        let _permit = semaphore.acquire().await?;

                        debug!("Starting request {}", i);

                        // Execute request
                        let result = execute_single_request(
                            provider.as_ref(),
                            request,
                            &timing_engine,
                        )
                        .await;

                        // Record metrics
                        if let Ok(ref metrics) = result {
                            if let Err(e) = collector.record(metrics.clone()) {
                                warn!("Failed to record metrics: {}", e);
                            }
                        }

                        // Update progress
                        if let Some(ref pb) = progress_bar {
                            pb.inc(1);
                        }

                        result
                    } => result,
                }
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        while let Some(result) = tasks.next().await {
            match result {
                Ok(Ok(_metrics)) => {
                    summary.successful_requests += 1;
                }
                Ok(Err(e)) => {
                    summary.failed_requests += 1;
                    warn!("Request failed: {}", e);
                }
                Err(e) => {
                    summary.failed_requests += 1;
                    warn!("Task panicked: {}", e);
                }
            }
        }

        // Finish progress bar
        if let Some(pb) = progress_bar {
            pb.finish_with_message("Complete");
        }

        summary.total_duration = start_time.elapsed();
        summary.requests_per_second =
            summary.successful_requests as f64 / summary.total_duration.as_secs_f64();

        info!(
            "Orchestration complete: {}/{} successful in {:.2}s ({:.2} req/s)",
            summary.successful_requests,
            summary.total_requests,
            summary.total_duration.as_secs_f64(),
            summary.requests_per_second
        );

        Ok(summary)
    }

    /// Execute a single request (useful for profiling)
    pub async fn execute_single<P: Provider>(
        &self,
        provider: &P,
        request: StreamingRequest,
    ) -> Result<RequestMetrics> {
        execute_single_request(
            provider,
            request,
            &self.timing_engine,
        )
        .await
    }
}

/// Execute a single request and return metrics
async fn execute_single_request<P: Provider>(
    provider: &P,
    request: StreamingRequest,
    timing_engine: &TimingEngine,
) -> Result<RequestMetrics> {
    let request_id = request.request_id;
    let session_id = request.session_id;
    let model = request.model.clone();

    let start_time = chrono::Utc::now();
    let start_instant = Instant::now();

    // Execute the request
    let result = provider.complete(request, timing_engine).await?;

    let total_latency = start_instant.elapsed();

    // Calculate TTFT
    let ttft = result.ttft().unwrap_or(Duration::ZERO);

    // Extract inter-token latencies
    let inter_token_latencies: Vec<Duration> = result
        .token_events
        .iter()
        .filter_map(|e| e.inter_token_latency)
        .collect();

    // Calculate throughput
    let tokens_per_second = if total_latency.as_secs_f64() > 0.0 {
        result.token_events.len() as f64 / total_latency.as_secs_f64()
    } else {
        0.0
    };

    // Get token counts from metadata
    let input_tokens = result.metadata.input_tokens.unwrap_or(0);
    let output_tokens = result.metadata.output_tokens.unwrap_or(result.token_events.len() as u64);
    let thinking_tokens = result.metadata.thinking_tokens;

    // Get cost
    let cost_usd = result.metadata.estimated_cost;

    Ok(RequestMetrics {
        request_id,
        session_id,
        provider: llm_latency_lens_core::Provider::OpenAI, // TODO: Get from provider
        model,
        timestamp: start_time,
        ttft,
        total_latency,
        inter_token_latencies,
        input_tokens,
        output_tokens,
        thinking_tokens,
        tokens_per_second,
        cost_usd,
        success: true,
        error: None,
    })
}

/// Summary of orchestration execution
#[derive(Debug, Clone, Default)]
pub struct ExecutionSummary {
    /// Total number of requests attempted
    pub total_requests: u32,
    /// Number of successful requests
    pub successful_requests: u32,
    /// Number of failed requests
    pub failed_requests: u32,
    /// Total duration of execution
    pub total_duration: Duration,
    /// Average requests per second
    pub requests_per_second: f64,
}

impl ExecutionSummary {
    /// Calculate success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_config_default() {
        let config = OrchestratorConfig::default();
        assert_eq!(config.concurrency, 1);
        assert_eq!(config.total_requests, 1);
        assert_eq!(config.rate_limit, 0);
        assert!(config.show_progress);
    }

    #[test]
    fn test_execution_summary_success_rate() {
        let mut summary = ExecutionSummary::default();
        summary.total_requests = 100;
        summary.successful_requests = 95;
        summary.failed_requests = 5;

        assert_eq!(summary.success_rate(), 95.0);
    }

    #[test]
    fn test_execution_summary_zero_requests() {
        let summary = ExecutionSummary::default();
        assert_eq!(summary.success_rate(), 0.0);
    }

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let config = OrchestratorConfig::default();
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let orchestrator = Orchestrator::new(config, shutdown);

        assert_eq!(orchestrator.config.concurrency, 1);
    }
}
