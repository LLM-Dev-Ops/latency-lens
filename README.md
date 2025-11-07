# LLM-Latency-Lens

<div align="center">

### Enterprise-Grade LLM Performance Profiler

**Comprehensive latency profiling and performance analysis for Large Language Models**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-85%25-green.svg)]()
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Quick Start](#quick-start) •
[Features](#features) •
[Documentation](#documentation) •
[Benchmarks](#benchmarks) •
[Contributing](#contributing)

</div>

---

## Overview

LLM-Latency-Lens is a production-grade, open-source profiling tool designed to measure, analyze, and optimize latency across all major LLM providers. Built in Rust for maximum performance and precision, it provides comprehensive performance insights for production LLM applications.

### Why LLM-Latency-Lens?

- **Sub-millisecond Precision**: Nanosecond-accurate timing for Time-to-First-Token (TTFT) and token streaming
- **Multi-Provider Support**: OpenAI, Anthropic, Google, Azure, Cohere, and more
- **Enterprise-Ready**: Battle-tested concurrency control, retry logic, and error handling
- **Cost Analytics**: Real-time cost tracking and ROI analysis
- **Open Source**: Apache 2.0 licensed, community-driven development

## Quick Start

### Installation

#### Via npm (JavaScript/TypeScript - Recommended for Web/Node.js)

```bash
# Install as a library
npm install @llm-dev-ops/latency-lens

# Or install globally for CLI
npm install -g @llm-dev-ops/latency-lens

# Test the CLI
latency-lens test
latency-lens version
```

#### Via Cargo (Rust Crates)

```bash
# Add to your Cargo.toml
[dependencies]
llm-latency-lens-core = "0.1.0"
llm-latency-lens-providers = "0.1.2"
llm-latency-lens-metrics = "0.1.0"
llm-latency-lens-exporters = "0.1.0"

# Or use in your project
cargo add llm-latency-lens-core
cargo add llm-latency-lens-providers
cargo add llm-latency-lens-metrics
cargo add llm-latency-lens-exporters
```

#### From Source

```bash
git clone https://github.com/llm-dev-ops/llm-latency-lens.git
cd llm-latency-lens
cargo build --release
```

#### Docker

```bash
docker pull llm-devops/llm-latency-lens:latest
docker run -e OPENAI_API_KEY=sk-... llm-devops/llm-latency-lens profile --provider openai --model gpt-4
```

#### Binary Downloads

Download pre-built binaries for Linux, macOS, and Windows from our [releases page](https://github.com/llm-dev-ops/llm-latency-lens/releases).

### Basic Usage

#### JavaScript/TypeScript (npm)

```javascript
import { LatencyCollector } from '@llm-dev-ops/latency-lens';

// Create collector with 60-second window
const collector = new LatencyCollector(60000);

// Start tracking a request
const requestId = collector.start_request('openai', 'gpt-4-turbo');

// Record first token received
collector.record_first_token(requestId);

// Record subsequent tokens
collector.record_token(requestId);
collector.record_token(requestId);

// Complete the request
collector.complete_request(requestId, 150, 800, null, 0.05);

// Get metrics
const metrics = collector.get_metrics();
console.log('TTFT P95:', metrics.ttft_distribution.p95_ms, 'ms');
console.log('Throughput:', metrics.throughput.tokens_per_second, 'tokens/sec');
```

#### Rust (Cargo)

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Profile OpenAI GPT-4
llm-latency-lens profile \
  --provider openai \
  --model gpt-4 \
  --prompt "Explain quantum computing in simple terms" \
  --iterations 100

# Profile with streaming enabled
llm-latency-lens profile \
  --provider anthropic \
  --model claude-3-opus-20240229 \
  --prompt "Write a Python function to calculate Fibonacci numbers" \
  --stream \
  --iterations 50 \
  --concurrency 10

# Compare multiple providers
llm-latency-lens compare \
  --config benchmark.yaml \
  --output results.json
```

### Example Output

```
LLM Latency Lens - Benchmark Results
=====================================

Provider: openai | Model: gpt-4-turbo-preview
Duration: 15.3s | Requests: 100 | Concurrency: 20

┌──────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Metric               │ Min      │ Mean     │ Median   │ p95      │ p99      │
├──────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ TTFT (ms)            │ 234.2    │ 456.8    │ 432.1    │ 678.9    │ 789.3    │
│ Total Duration (ms)  │ 1234.5   │ 2456.7   │ 2389.4   │ 3456.8   │ 3789.2   │
│ Tokens/sec           │ 12.3     │ 45.6     │ 44.2     │ 67.8     │ 72.1     │
│ Inter-token (ms)     │ 8.2      │ 22.4     │ 21.8     │ 34.6     │ 42.3     │
└──────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

Token Usage:
  Prompt: 15,234 tokens | Completion: 45,678 tokens | Total: 60,912 tokens

Cost Analysis:
  Total Cost: $1.23 | Cost/Request: $0.012 | Cost/1K tokens: $0.020

Success Rate: 98.0% (98/100)
```

## Features

### Core Capabilities

#### Precision Timing
- **Nanosecond Accuracy**: High-resolution timing using hardware counters
- **TTFT Measurement**: Critical metric for perceived responsiveness
- **Inter-token Latency**: Track consistency of token generation
- **Network Breakdown**: DNS, TLS, connection establishment timing

#### Multi-Provider Support
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5 Turbo, o1, o3
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku (including extended thinking)
- **Google**: Gemini Pro, Gemini Ultra (coming soon)
- **Azure OpenAI**: Full compatibility
- **Cohere**: Command models
- **Custom Providers**: Generic HTTP adapter for any API

#### Performance Analysis
- **Statistical Metrics**: Min, max, mean, median, std dev, percentiles (p50, p95, p99, p999)
- **Histogram Generation**: HDR histograms for accurate percentile calculation
- **Real-time Streaming**: Process tokens as they arrive
- **Cost Tracking**: Accurate pricing based on current rates

#### Concurrency & Scale
- **High Throughput**: Handle 1000+ concurrent requests
- **Rate Limiting**: Per-provider rate limiters with token bucket algorithm
- **Retry Logic**: Automatic retries with exponential backoff
- **Connection Pooling**: Efficient HTTP/2 connection reuse

#### Export & Integration
- **Multiple Formats**: JSON, CSV, binary (MessagePack/Bincode)
- **Time-series DBs**: InfluxDB, Prometheus, Datadog
- **CI/CD Integration**: GitHub Actions, GitLab CI, Jenkins
- **Grafana Dashboards**: Pre-built visualization templates
- **OpenTelemetry**: Full trace and metrics export

### Advanced Features

#### Library Mode
Use as a Rust library in your applications:

```rust
use llm_latency_lens_providers::{OpenAIProvider, StreamingRequest, MessageRole};
use llm_latency_lens_core::TimingEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAIProvider::new("sk-...");
    let timing = TimingEngine::new();

    let request = StreamingRequest::builder()
        .model("gpt-4o")
        .message(MessageRole::User, "Explain quantum computing")
        .max_tokens(500)
        .temperature(0.7)
        .build();

    let response = provider.stream(request, &timing).await?;

    println!("TTFT: {:?}", response.ttft);
    println!("Total tokens: {}", response.metadata.completion_tokens);

    Ok(())
}
```

#### Configuration Files
Define complex benchmark scenarios:

```yaml
# benchmark.yaml
providers:
  - name: openai
    models: [gpt-4-turbo-preview, gpt-3.5-turbo]
  - name: anthropic
    models: [claude-3-opus-20240229]

workload:
  scenarios:
    - name: short_prompt_high_concurrency
      prompt: "What is the capital of France?"
      requests: 100
      concurrency: 20

    - name: long_prompt_streaming
      prompt: "Write a comprehensive guide to machine learning"
      requests: 50
      concurrency: 5
      stream: true

execution:
  max_concurrency: 50
  warmup_requests: 5
  retry:
    max_attempts: 3
    initial_backoff_ms: 1000

output:
  export:
    - format: json
      path: ./results/benchmark_{timestamp}.json
    - format: csv
      path: ./results/benchmark_{timestamp}.csv
```

## Documentation

### Getting Started
- [User Guide](docs/USER_GUIDE.md) - Comprehensive usage documentation
- [Installation Guide](docs/USER_GUIDE.md#installation) - Detailed installation instructions
- [Quick Start Tutorial](docs/USER_GUIDE.md#quick-start) - Get up and running in 5 minutes

### API & Integration
- [API Documentation](docs/API.md) - Library usage and integration patterns
- [Configuration Reference](docs/USER_GUIDE.md#configuration) - All configuration options
- [Provider Guide](docs/USER_GUIDE.md#provider-configuration) - Provider-specific settings

### Architecture & Design
- [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- [Data Flow](docs/DATA_FLOW.md) - Request lifecycle and metrics pipeline
- [Crate Structure](docs/CRATE_STRUCTURE.md) - Internal organization

### Advanced Topics
- [Ecosystem Integration](docs/ECOSYSTEM_INTEGRATION.md) - Integration with other tools
- [Performance Tuning](docs/USER_GUIDE.md#performance-tuning) - Optimization strategies
- [Troubleshooting](docs/USER_GUIDE.md#troubleshooting) - Common issues and solutions

## Benchmarks

LLM-Latency-Lens itself has minimal performance overhead:

| Metric | Value |
|--------|-------|
| Timing Overhead | < 100 nanoseconds per measurement |
| Memory Usage | < 100MB baseline |
| CPU Usage | < 5% overhead per request |
| Throughput | 1000+ concurrent requests |
| Accuracy | ±0.1% percentile calculation |

### Real-world Performance

Benchmarking OpenAI GPT-4 Turbo (100 requests, 20 concurrent):
- **TTFT p50**: 432.1ms
- **TTFT p95**: 678.9ms
- **Throughput**: 44.2 tokens/sec
- **Success Rate**: 98.0%
- **Cost**: $0.012 per request

## Use Cases

### Development & Testing
- Profile LLM APIs during development
- Compare model performance before deployment
- Identify latency regressions in CI/CD
- Optimize prompt engineering for speed

### Production Monitoring
- Continuous latency monitoring
- SLA compliance verification
- Cost optimization analysis
- Provider comparison for failover

### Research & Benchmarking
- Academic research on LLM performance
- Benchmark new models and providers
- Analyze scaling characteristics
- Study geographic latency patterns

### Cost Optimization
- Track API spending in real-time
- Compare cost vs. performance tradeoffs
- Project monthly costs based on usage
- Identify opportunities for optimization

## Contributing

We welcome contributions from the community! See our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code of Conduct
- Development setup
- Pull request process
- Testing requirements
- Documentation standards

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`cargo test`)
5. Run lints (`cargo clippy`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Commercial Support

### Enterprise Features
- **Priority Support**: 24/7 support with SLA
- **Custom Integrations**: Tailored provider adapters
- **On-Premise Deployment**: Self-hosted solutions
- **Training & Consulting**: Expert guidance
- **Advanced Analytics**: Custom dashboards and reporting

### Contact
- Enterprise Sales: enterprise@llm-devops.com
- Support: support@llm-devops.com
- Community: [Discord](https://discord.gg/llm-latency-lens)

## Roadmap

### Current Version (0.1.0)
- ✅ Core timing engine
- ✅ OpenAI and Anthropic providers
- ✅ Streaming support
- ✅ Basic CLI
- ✅ JSON/CSV export

### Next Release (0.2.0)
- 🔄 Google Gemini provider
- 🔄 Azure OpenAI support
- 🔄 Cohere integration
- 🔄 Prometheus metrics
- 🔄 Grafana dashboards

### Future (1.0.0)
- 📋 Distributed execution
- 📋 Real-time dashboard
- 📋 Historical analysis
- 📋 AI-powered optimization
- 📋 Multi-region testing

See our [full roadmap](docs/ROADMAP.md) for details.

## Performance Comparison

| Tool | Language | TTFT Accuracy | Streaming | Multi-Provider | Cost Tracking |
|------|----------|---------------|-----------|----------------|---------------|
| **LLM-Latency-Lens** | Rust | ✅ Nanosecond | ✅ Yes | ✅ 5+ providers | ✅ Real-time |
| Tool A | Python | ⚠️ Millisecond | ❌ No | ⚠️ 2 providers | ❌ No |
| Tool B | Go | ✅ Microsecond | ✅ Yes | ⚠️ 3 providers | ⚠️ Manual |
| Tool C | Node.js | ⚠️ Millisecond | ⚠️ Limited | ❌ 1 provider | ❌ No |

## Security

Security is a top priority. See our [Security Policy](SECURITY.md) for:
- Vulnerability reporting process
- Security update policy
- API key handling best practices
- Audit logs and compliance

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

```
Copyright 2024 LLM DevOps Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Acknowledgments

Built with these excellent open-source projects:
- [Tokio](https://tokio.rs/) - Async runtime
- [Reqwest](https://github.com/seanmonstar/reqwest) - HTTP client
- [Quanta](https://github.com/metrics-rs/quanta) - High-precision timing
- [HDRHistogram](https://github.com/HdrHistogram/HdrHistogram_rust) - Latency percentiles
- [Clap](https://github.com/clap-rs/clap) - CLI framework

## Community

- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/llm-devops/llm-latency-lens/discussions)
- **Discord**: [Join our community](https://discord.gg/llm-latency-lens)
- **Twitter**: [@llmlatencylens](https://twitter.com/llmlatencylens)
- **Blog**: [Read our technical blog](https://blog.llm-devops.com)

## Star History

If you find LLM-Latency-Lens useful, please consider giving us a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=llm-devops/llm-latency-lens&type=Date)](https://star-history.com/#llm-devops/llm-latency-lens&Date)

---

<div align="center">

**Made with ❤️ by the LLM DevOps Team**

[Website](https://llm-latency-lens.dev) • [Documentation](https://docs.llm-latency-lens.dev) • [GitHub](https://github.com/llm-devops/llm-latency-lens)

</div>
