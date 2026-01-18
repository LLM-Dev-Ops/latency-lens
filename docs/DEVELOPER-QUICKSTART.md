# LLM-Latency-Lens: Developer Quickstart

**Target Audience**: Development team implementing the MVP
**Timeline**: Week 1-4 (MVP Phase)
**Goal**: Production-ready basic profiler with OpenAI & Anthropic support

---

## Week 1: Project Foundation

### Day 1-2: Project Setup

```bash
# Initialize Rust project
cargo new llm-latency-lens --bin
cd llm-latency-lens

# Set up Git
git init
git add .
git commit -m "Initial project structure"

# Create directory structure
mkdir -p src/{providers,timing,metrics,export,config}
```

### Cargo.toml Initial Dependencies

```toml
[package]
name = "llm-latency-lens"
version = "0.1.0"
edition = "2021"

[dependencies]
# Async runtime
tokio = { version = "1.35", features = ["full"] }
async-trait = "0.1"
futures = "0.3"

# HTTP client
reqwest = { version = "0.12", features = ["json", "stream"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# CLI
clap = { version = "4.5", features = ["derive"] }

# Utilities
anyhow = "1.0"
thiserror = "1.0"
chrono = { version = "0.4", features = ["serde"] }
guid-create = "0.4"

[dev-dependencies]
tokio-test = "0.4"
mockito = "1.4"
```

### Day 3-5: Core Architecture

**File**: `src/main.rs`
```rust
mod providers;
mod timing;
mod metrics;
mod export;
mod config;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "llm-latency-lens")]
#[command(about = "LLM performance profiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Profile {
        #[arg(long)]
        provider: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        api_key: Option<String>,
    },
    Benchmark {
        #[arg(long)]
        provider: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "10")]
        iterations: u32,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Profile { provider, model, prompt, api_key } => {
            // TODO: Implement in Week 2
        }
        Commands::Benchmark { provider, model, prompt, iterations } => {
            // TODO: Implement in Week 3
        }
    }

    Ok(())
}
```

**File**: `src/metrics.rs`
```rust
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestMetrics {
    pub request_id: String,
    pub provider: String,
    pub model: String,
    pub timestamp: DateTime<Utc>,

    // Timing
    pub ttfb_ms: f64,
    pub ttft_ms: f64,
    pub total_latency_ms: f64,
    pub token_timings: Vec<TokenTiming>,

    // Tokens
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub tokens_per_second: f64,

    // Cost
    pub input_cost: f64,
    pub output_cost: f64,
    pub total_cost: f64,

    // Status
    pub status: RequestStatus,
    pub error: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TokenTiming {
    pub token_index: u32,
    pub timestamp_ms: f64,
    pub delta_ms: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum RequestStatus {
    Success,
    RateLimited,
    NetworkError,
    AuthError,
    ServerError,
}
```

---

## Week 2: OpenAI Adapter

### File: `src/providers/mod.rs`
```rust
pub mod openai;
pub mod anthropic;

use async_trait::async_trait;
use crate::metrics::RequestMetrics;

#[async_trait]
pub trait ProviderAdapter: Send + Sync {
    async fn send_request(
        &self,
        prompt: &str,
    ) -> anyhow::Result<RequestMetrics>;

    fn get_pricing(&self) -> Pricing;
}

pub struct Pricing {
    pub input_cost_per_million: f64,
    pub output_cost_per_million: f64,
}
```

### File: `src/providers/openai.rs`
```rust
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use std::time::Instant;
use futures::StreamExt;
use crate::providers::{ProviderAdapter, Pricing};
use crate::metrics::{RequestMetrics, TokenTiming, RequestStatus};

pub struct OpenAIAdapter {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenAIAdapter {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
        }
    }
}

#[async_trait]
impl ProviderAdapter for OpenAIAdapter {
    async fn send_request(&self, prompt: &str) -> anyhow::Result<RequestMetrics> {
        let start = Instant::now();
        let request_id = guid_create::GUID::rand().to_string();

        // Build request payload
        let payload = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": true,
        });

        // Send request
        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await?;

        let ttfb_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Process streaming response
        let mut stream = response.bytes_stream();
        let mut token_timings = Vec::new();
        let mut ttft_ms = None;
        let mut output_tokens = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            // Parse SSE format (data: {...})
            for line in std::str::from_utf8(&chunk)?.lines() {
                if line.starts_with("data: ") && !line.contains("[DONE]") {
                    if ttft_ms.is_none() {
                        ttft_ms = Some(elapsed_ms);
                    }

                    let delta_ms = if let Some(prev) = token_timings.last() {
                        elapsed_ms - prev.timestamp_ms
                    } else {
                        0.0
                    };

                    token_timings.push(TokenTiming {
                        token_index: output_tokens,
                        timestamp_ms: elapsed_ms,
                        delta_ms,
                    });

                    output_tokens += 1;
                }
            }
        }

        let total_latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Calculate metrics
        let input_tokens = estimate_tokens(prompt);
        let tokens_per_second = if total_latency_ms > 0.0 {
            (output_tokens as f64) / (total_latency_ms / 1000.0)
        } else {
            0.0
        };

        // Calculate costs (GPT-4: $30/$60 per million)
        let pricing = self.get_pricing();
        let input_cost = (input_tokens as f64 / 1_000_000.0) * pricing.input_cost_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * pricing.output_cost_per_million;

        Ok(RequestMetrics {
            request_id,
            provider: "openai".to_string(),
            model: self.model.clone(),
            timestamp: chrono::Utc::now(),
            ttfb_ms,
            ttft_ms: ttft_ms.unwrap_or(0.0),
            total_latency_ms,
            token_timings,
            input_tokens,
            output_tokens,
            tokens_per_second,
            input_cost,
            output_cost,
            total_cost: input_cost + output_cost,
            status: RequestStatus::Success,
            error: None,
        })
    }

    fn get_pricing(&self) -> Pricing {
        // Default to GPT-4 pricing
        Pricing {
            input_cost_per_million: 30.0,
            output_cost_per_million: 60.0,
        }
    }
}

// Simple token estimation (4 chars per token approximation)
fn estimate_tokens(text: &str) -> u32 {
    (text.len() / 4) as u32
}
```

**Testing**:
```bash
export OPENAI_API_KEY="sk-..."
cargo run -- profile \
  --provider openai \
  --model gpt-4 \
  --prompt "Hello, world!"
```

---

## Week 3: Anthropic Adapter & Metrics

### File: `src/providers/anthropic.rs`
```rust
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use std::time::Instant;
use futures::StreamExt;
use crate::providers::{ProviderAdapter, Pricing};
use crate::metrics::{RequestMetrics, TokenTiming, RequestStatus};

pub struct AnthropicAdapter {
    client: Client,
    api_key: String,
    model: String,
}

impl AnthropicAdapter {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
        }
    }
}

#[async_trait]
impl ProviderAdapter for AnthropicAdapter {
    async fn send_request(&self, prompt: &str) -> anyhow::Result<RequestMetrics> {
        let start = Instant::now();
        let request_id = guid_create::GUID::rand().to_string();

        let payload = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "stream": true,
        });

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&payload)
            .send()
            .await?;

        // Similar streaming logic to OpenAI
        // (Implementation follows same pattern)

        // Claude 4.1 Sonnet pricing: $3/$15 per million
        let pricing = self.get_pricing();
        // ... rest of implementation

        Ok(RequestMetrics { /* ... */ })
    }

    fn get_pricing(&self) -> Pricing {
        Pricing {
            input_cost_per_million: 3.0,
            output_cost_per_million: 15.0,
        }
    }
}
```

### File: `src/export/json.rs`
```rust
use crate::metrics::RequestMetrics;
use std::path::Path;

pub fn export_json(metrics: &RequestMetrics, output: Option<&Path>) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(metrics)?;

    match output {
        Some(path) => std::fs::write(path, json)?,
        None => println!("{}", json),
    }

    Ok(())
}

pub fn export_json_lines(
    metrics_list: &[RequestMetrics],
    output: Option<&Path>,
) -> anyhow::Result<()> {
    let lines: Vec<String> = metrics_list
        .iter()
        .map(|m| serde_json::to_string(m))
        .collect::<Result<Vec<_>, _>>()?;

    let content = lines.join("\n");

    match output {
        Some(path) => std::fs::write(path, content)?,
        None => println!("{}", content),
    }

    Ok(())
}
```

---

## Week 4: Benchmark Mode & Testing

### File: `src/benchmark.rs`
```rust
use crate::providers::ProviderAdapter;
use crate::metrics::RequestMetrics;

pub struct BenchmarkRunner {
    adapter: Box<dyn ProviderAdapter>,
}

impl BenchmarkRunner {
    pub fn new(adapter: Box<dyn ProviderAdapter>) -> Self {
        Self { adapter }
    }

    pub async fn run_iterations(
        &self,
        prompt: &str,
        iterations: u32,
    ) -> anyhow::Result<Vec<RequestMetrics>> {
        let mut results = Vec::new();

        for i in 0..iterations {
            println!("Running iteration {}/{}", i + 1, iterations);
            let metrics = self.adapter.send_request(prompt).await?;
            results.push(metrics);
        }

        Ok(results)
    }

    pub fn calculate_summary(&self, results: &[RequestMetrics]) -> BenchmarkSummary {
        let mut ttft_values: Vec<f64> = results.iter().map(|r| r.ttft_ms).collect();
        ttft_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        BenchmarkSummary {
            total_requests: results.len() as u32,
            ttft_min: ttft_values.first().copied().unwrap_or(0.0),
            ttft_max: ttft_values.last().copied().unwrap_or(0.0),
            ttft_mean: ttft_values.iter().sum::<f64>() / ttft_values.len() as f64,
            ttft_median: ttft_values[ttft_values.len() / 2],
            ttft_p95: ttft_values[(ttft_values.len() as f64 * 0.95) as usize],
            // ... more stats
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct BenchmarkSummary {
    pub total_requests: u32,
    pub ttft_min: f64,
    pub ttft_max: f64,
    pub ttft_mean: f64,
    pub ttft_median: f64,
    pub ttft_p95: f64,
}
```

### Integration Tests
```rust
// tests/integration_test.rs
#[tokio::test]
async fn test_openai_profile() {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap();
    let adapter = OpenAIAdapter::new(api_key, "gpt-4".to_string());

    let metrics = adapter.send_request("Say hello").await.unwrap();

    assert_eq!(metrics.provider, "openai");
    assert!(metrics.ttft_ms > 0.0);
    assert!(metrics.output_tokens > 0);
    assert!(metrics.total_cost > 0.0);
}
```

---

## MVP Deliverables Checklist

### Core Functionality
- [ ] CLI argument parsing (clap)
- [ ] OpenAI provider adapter with streaming
- [ ] Anthropic provider adapter with streaming
- [ ] High-precision timing engine
- [ ] Token counting and cost calculation
- [ ] Single request profiling (`profile` command)
- [ ] Benchmark mode (`benchmark` command)
- [ ] JSON export

### Code Quality
- [ ] Unit tests for core modules
- [ ] Integration tests with real APIs
- [ ] Error handling with proper types
- [ ] Logging setup
- [ ] Documentation comments

### Documentation
- [ ] README with installation and usage
- [ ] API documentation (cargo doc)
- [ ] Example outputs

---

## Quick Commands Reference

```bash
# Development
cargo build
cargo test
cargo run -- --help

# Profile single request
cargo run -- profile \
  --provider openai \
  --model gpt-4 \
  --prompt "Explain Rust ownership"

# Benchmark mode
cargo run -- benchmark \
  --provider anthropic \
  --model claude-4.1-sonnet \
  --prompt "Write a haiku" \
  --iterations 20

# Run tests (requires API keys)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
cargo test
```

---

## Debugging Tips

1. **Enable logging**:
   ```bash
   RUST_LOG=debug cargo run -- profile ...
   ```

2. **Test streaming parsing**:
   - Create mock responses
   - Test with small prompts first
   - Validate SSE format handling

3. **Cost calculation verification**:
   - Compare against provider dashboards
   - Test with known token counts

---

## Next: Beta Phase (Week 5-10)

After MVP completion:
- Add Google Gemini adapter
- Implement constant-rate test pattern
- Add Prometheus export
- Integrate with Test-Bench
- Create configuration file system

**See**: `/workspaces/llm-latency-lens/plans/LLM-Latency-Lens-Plan.md` Section 7.2

---

**Version**: 1.0
**Date**: 2025-11-07
**Target**: Development Team
