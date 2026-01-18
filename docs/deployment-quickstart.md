# LLM-Latency-Lens: Deployment Quick Start Guide

## Choose Your Deployment Mode

### Decision Tree

```
Start Here: What's your use case?
│
├─ "I want to test locally or run quick benchmarks"
│  └─▶ Use: Standalone CLI
│     Time to deploy: 5 minutes
│     Complexity: Low
│     → Jump to: Section 1
│
├─ "I want to catch performance regressions in CI/CD"
│  └─▶ Use: CI/CD Integration
│     Time to deploy: 30 minutes
│     Complexity: Medium
│     → Jump to: Section 2
│
├─ "I want to profile LLM calls in my production service"
│  └─▶ Use: Embedded Library
│     Time to deploy: 1 hour
│     Complexity: Medium
│     → Jump to: Section 3
│
├─ "I need high-scale load testing across regions"
│  └─▶ Use: Distributed Execution
│     Time to deploy: 4 hours
│     Complexity: High
│     → Jump to: Section 4
│
└─ "I need real-time monitoring and alerting"
   └─▶ Use: Observatory Integration
      Time to deploy: 2 hours
      Complexity: Medium-High
      → Jump to: Section 5
```

---

## Section 1: Standalone CLI (5 Minutes)

### Installation

**Option A: Cargo (Rust users)**
```bash
cargo install llm-latency-lens
```

**Option B: Binary Download**
```bash
# Linux
curl -L https://github.com/your-org/llm-latency-lens/releases/latest/download/llm-lens-linux-x64 -o llm-lens
chmod +x llm-lens
sudo mv llm-lens /usr/local/bin/

# macOS
curl -L https://github.com/your-org/llm-latency-lens/releases/latest/download/llm-lens-darwin-arm64 -o llm-lens
chmod +x llm-lens
sudo mv llm-lens /usr/local/bin/
```

**Option C: Docker**
```bash
docker pull ghcr.io/your-org/llm-latency-lens:latest
alias llm-lens='docker run --rm -v $(pwd):/data ghcr.io/your-org/llm-latency-lens:latest'
```

### First Run

```bash
# 1. Set API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# 2. Generate config template
llm-lens init --format yaml > llm-lens.yaml

# 3. Edit config (minimal example below)
cat > llm-lens.yaml <<EOF
version: "1.0"

profiler:
  name: "my-first-benchmark"
  duration: 60s

providers:
  - name: openai
    api_key: "\${OPENAI_API_KEY}"
    models:
      - id: "gpt-4-turbo-preview"

workload:
  fixed:
    requests_per_second: 10

output:
  console:
    enabled: true
  files:
    - type: json
      path: "results.json"
EOF

# 4. Run benchmark
llm-lens run --config llm-lens.yaml
```

### Common Commands

```bash
# Quick benchmark without config
llm-lens run \
  --provider openai \
  --model gpt-4-turbo-preview \
  --duration 60s \
  --rps 10

# Compare models
llm-lens compare \
  --models gpt-4-turbo-preview,claude-3-opus-20240229 \
  --duration 60s

# Analyze results
llm-lens analyze results.json --output report.html

# Validate config
llm-lens validate --config llm-lens.yaml
```

---

## Section 2: CI/CD Integration (30 Minutes)

### GitHub Actions Setup

**Step 1: Create benchmark config** (`.github/benchmark-config.yaml`)

```yaml
version: "1.0"

profiler:
  name: "ci-benchmark"
  duration: 60s  # Keep it short for PRs

providers:
  - name: openai
    api_key: "${OPENAI_API_KEY}"
    models:
      - id: "gpt-4-turbo-preview"

workload:
  fixed:
    requests_per_second: 5

output:
  files:
    - type: json
      path: "results/benchmark.json"
```

**Step 2: Create workflow** (`.github/workflows/benchmark.yml`)

```yaml
name: LLM Benchmark

on:
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install llm-lens
        run: cargo install llm-latency-lens

      - name: Run benchmark
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          llm-lens run --config .github/benchmark-config.yaml

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results/
```

**Step 3: Add secrets**
```bash
# In GitHub repo settings → Secrets → Actions
# Add: OPENAI_API_KEY, ANTHROPIC_API_KEY
```

### GitLab CI Setup

**Create** `.gitlab-ci.yml`:

```yaml
benchmark:
  stage: test
  image: rust:latest
  script:
    - cargo install llm-latency-lens
    - llm-lens run --config benchmark.yaml
  artifacts:
    paths:
      - results/
  only:
    - merge_requests
```

---

## Section 3: Embedded Library (1 Hour)

### Step 1: Add Dependency

```toml
# Cargo.toml
[dependencies]
llm-latency-lens = "0.1.0"
tokio = { version = "1.35", features = ["full"] }
```

### Step 2: Basic Integration

```rust
// src/main.rs
use llm_latency_lens::{Profiler, ProfilerBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize profiler
    let profiler = ProfilerBuilder::new()
        .with_prometheus_export(9090)
        .build()?;

    // Profile your LLM calls
    let response = profiler.profile(async {
        your_llm_client.chat_completion("Hello").await
    }).await?;

    println!("Latency: {}ms", response.metrics.total_latency_ms);

    Ok(())
}
```

### Step 3: Enable Prometheus Metrics

```yaml
# Add to your docker-compose.yml or K8s deployment
ports:
  - "9090:9090"  # Prometheus metrics endpoint
```

### Step 4: View Metrics

```bash
# Check metrics endpoint
curl http://localhost:9090/metrics | grep llm_

# Expected output:
# llm_request_duration_seconds_bucket{...}
# llm_ttft_duration_seconds_bucket{...}
# llm_requests_total{...}
```

---

## Section 4: Distributed Execution (4 Hours)

### Prerequisites

```bash
# Required:
- Kubernetes cluster (or Docker Compose)
- Redis or NATS
- S3 or compatible object storage
```

### Quick Start with Docker Compose

**Step 1: Create `docker-compose.yml`**

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  coordinator:
    image: ghcr.io/your-org/llm-latency-lens:latest
    command: llm-lens distributed coordinator --redis-url redis://redis:6379
    ports:
      - "8080:8080"
    volumes:
      - ./config:/config
    depends_on:
      - redis

  worker:
    image: ghcr.io/your-org/llm-latency-lens:latest
    command: llm-lens distributed worker --coordinator http://coordinator:8080
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - coordinator
    deploy:
      replicas: 5
```

**Step 2: Start cluster**

```bash
# Set API keys
export OPENAI_API_KEY="your-key"

# Start services
docker-compose up -d

# Check status
docker-compose ps

# Submit benchmark
llm-lens distributed submit \
  --coordinator http://localhost:8080 \
  --config benchmark.yaml
```

### Kubernetes Deployment

```bash
# 1. Create namespace
kubectl create namespace benchmarking

# 2. Create secrets
kubectl create secret generic llm-api-keys \
  --from-literal=openai="${OPENAI_API_KEY}" \
  --namespace=benchmarking

# 3. Deploy
kubectl apply -f k8s/coordinator.yaml
kubectl apply -f k8s/worker.yaml

# 4. Check status
kubectl get pods -n benchmarking

# 5. Submit job
llm-lens distributed submit \
  --coordinator http://coordinator.benchmarking.svc:8080 \
  --config benchmark.yaml
```

---

## Section 5: Observatory Integration (2 Hours)

### Full Stack Deployment

**Step 1: Deploy monitoring stack**

```bash
# Using Docker Compose
docker-compose -f docker-compose-monitoring.yml up -d
```

**`docker-compose-monitoring.yml`:**

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9093:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC

volumes:
  prometheus-data:
  grafana-data:
```

**Step 2: Configure Prometheus** (`prometheus.yml`)

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llm-latency-lens'
    static_configs:
      - targets: ['host.docker.internal:9090']
```

**Step 3: Enable profiler exports**

```rust
let profiler = ProfilerBuilder::new()
    .with_prometheus_export(9090)
    .with_opentelemetry_export("http://localhost:4317")
    .build()?;
```

**Step 4: Access dashboards**

```bash
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9093
# Jaeger: http://localhost:16686
```

**Step 5: Import dashboard**

1. Open Grafana (http://localhost:3000)
2. Go to Dashboards → Import
3. Upload `grafana/dashboards/llm-performance.json`

---

## Common Issues & Solutions

### Issue: "API key not found"

```bash
# Solution: Ensure environment variables are set
echo $OPENAI_API_KEY  # Should output your key
export OPENAI_API_KEY="sk-..."

# Or use .env file
echo 'OPENAI_API_KEY=sk-...' > .env
source .env
```

### Issue: "Rate limit exceeded"

```yaml
# Solution: Reduce RPS in config
workload:
  fixed:
    requests_per_second: 5  # Lower value
```

### Issue: "Connection timeout"

```yaml
# Solution: Increase timeouts in config
resilience:
  timeouts:
    connect: 30s
    request: 180s
```

### Issue: "High memory usage"

```yaml
# Solution: Reduce buffer size
metrics:
  buffer_size: 1000  # Lower value
  flush_interval: 5s
```

### Issue: "Workers not connecting (distributed)"

```bash
# Solution: Check network connectivity
docker-compose logs coordinator
docker-compose logs worker

# Verify Redis connection
redis-cli -h localhost ping  # Should return PONG
```

---

## Configuration Templates

### Minimal Config (Development)

```yaml
version: "1.0"

profiler:
  name: "dev-test"
  duration: 30s

providers:
  - name: openai
    api_key: "${OPENAI_API_KEY}"
    models:
      - id: "gpt-3.5-turbo"

workload:
  fixed:
    requests_per_second: 1

output:
  console:
    enabled: true
```

### Production Config

```yaml
version: "1.0"

profiler:
  name: "production-benchmark"
  duration: 600s
  warmup: 30s

providers:
  - name: openai
    api_key: "${OPENAI_API_KEY}"
    models:
      - id: "gpt-4-turbo-preview"

workload:
  poisson:
    lambda: 50

metrics:
  percentiles: [50, 75, 90, 95, 99, 99.9]

output:
  files:
    - type: json
      path: "results/benchmark-{{timestamp}}.json"
  database:
    enabled: true
    type: postgres
    connection: "${DATABASE_URL}"
  prometheus:
    enabled: true
    port: 9090

resilience:
  retries:
    max_attempts: 3
  circuit_breaker:
    enabled: true
```

### Comprehensive Config (All Features)

```yaml
version: "1.0"

profiler:
  name: "comprehensive-benchmark"
  mode: "comprehensive"
  duration: 1800s
  warmup: 60s
  cooldown: 30s

providers:
  - name: openai
    api_key: "${OPENAI_API_KEY}"
    models:
      - id: "gpt-4-turbo-preview"
      - id: "gpt-3.5-turbo"

  - name: anthropic
    api_key: "${ANTHROPIC_API_KEY}"
    models:
      - id: "claude-3-opus-20240229"
      - id: "claude-3-sonnet-20240229"

workload:
  type: "mixed"
  poisson:
    lambda: 100
  prompts:
    - template: "Summarize: {{text}}"
      weight: 0.5
    - template: "Translate: {{text}}"
      weight: 0.5

metrics:
  collect: ["latency", "throughput", "cost", "errors", "token_usage"]
  percentiles: [50, 75, 90, 95, 99, 99.9]
  resolution: 1s

output:
  console:
    enabled: true
    format: "table"
  files:
    - type: json
      path: "results/benchmark-{{timestamp}}.json"
    - type: csv
      path: "results/metrics-{{timestamp}}.csv"
  database:
    enabled: true
    type: postgres
    connection: "${DATABASE_URL}"
  prometheus:
    enabled: true
    port: 9090
  remote:
    - type: s3
      bucket: "llm-benchmarks"
      prefix: "results/"

resilience:
  retries:
    max_attempts: 3
    backoff: exponential
  timeouts:
    connect: 10s
    request: 120s
  circuit_breaker:
    enabled: true
    failure_threshold: 5

logging:
  level: info
  format: json
  file:
    enabled: true
    path: "logs/llm-lens.log"
```

---

## Next Steps

### After Standalone CLI
- [ ] Integrate into CI/CD pipeline
- [ ] Set up result storage (S3/database)
- [ ] Create baseline benchmarks
- [ ] Schedule regular runs

### After CI/CD Integration
- [ ] Set up regression detection
- [ ] Configure quality gates
- [ ] Add PR comment automation
- [ ] Implement baseline updates

### After Embedded Library
- [ ] Deploy monitoring stack
- [ ] Create Grafana dashboards
- [ ] Set up alerts
- [ ] Implement distributed tracing

### After Distributed Execution
- [ ] Optimize worker scaling
- [ ] Set up multi-region deployment
- [ ] Implement cost controls
- [ ] Configure auto-scaling

### After Observatory Integration
- [ ] Fine-tune alert thresholds
- [ ] Create runbooks
- [ ] Set up on-call rotation
- [ ] Implement SLO tracking

---

## Support & Resources

### Documentation
- Full deployment guide: `deployment-strategy.md`
- Architecture diagrams: `deployment-diagrams.md`
- API reference: (coming soon)

### Examples
- Sample configurations: `examples/configs/`
- Integration examples: `examples/integrations/`
- Dashboard templates: `examples/dashboards/`

### Community
- GitHub Issues: https://github.com/your-org/llm-latency-lens/issues
- Discussions: https://github.com/your-org/llm-latency-lens/discussions
- Slack: #llm-latency-lens

### Getting Help
1. Check documentation
2. Search GitHub issues
3. Ask in Slack
4. Create new issue with:
   - Deployment mode
   - Configuration file
   - Error logs
   - Expected vs actual behavior
