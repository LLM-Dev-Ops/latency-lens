# LLM-Latency-Lens: Deployment Reference Tables

## Deployment Mode Comparison Matrix

| Feature | Standalone CLI | CI/CD Integration | Embedded Library | Distributed | Observatory |
|---------|---------------|-------------------|------------------|-------------|-------------|
| **Deployment Time** | 5 minutes | 30 minutes | 1 hour | 4 hours | 2 hours |
| **Complexity** | Low | Medium | Medium | High | Medium-High |
| **Infrastructure Required** | None | CI/CD platform | Application server | K8s/Docker | Monitoring stack |
| **Concurrent Requests** | 1-100 | 1-100 | 1-1000 | 1000-100k+ | Varies |
| **Geographic Distribution** | No | No | No | Yes | N/A |
| **Real-time Monitoring** | No | No | Limited | Yes | Yes |
| **Historical Analysis** | Manual | Via artifacts | Manual | Yes | Yes |
| **Auto-scaling** | No | No | No | Yes | N/A |
| **Cost (Monthly)** | $0-50 | $50-200 | $100-500 | $1000-10k+ | $200-1000 |
| **Best For** | Local dev, testing | PR validation | Production profiling | Load testing | Operations |

## Resource Requirements

### Standalone CLI

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| CLI Process | 0.5 cores | 256MB | 1GB | 1 Mbps |
| **Total** | **0.5 cores** | **256MB** | **1GB** | **1 Mbps** |

### CI/CD Integration

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| Runner | 1 core | 1GB | 5GB | 10 Mbps |
| Artifact Storage | - | - | 10GB/month | - |
| **Total** | **1 core** | **1GB** | **15GB** | **10 Mbps** |

### Embedded Library

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| Application + Library | +0.1-0.5 cores | +100-500MB | +1GB | +1 Mbps |
| Prometheus (optional) | 0.5 cores | 512MB | 50GB | 5 Mbps |
| **Total** | **0.6-1 cores** | **600MB-1GB** | **51GB** | **6 Mbps** |

### Distributed Execution (50 workers)

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| Coordinator | 4 cores | 8GB | 20GB | 100 Mbps |
| Redis/NATS | 2 cores | 4GB | 10GB | 50 Mbps |
| Workers (each) | 2 cores | 4GB | 5GB | 10 Mbps |
| Workers (total 50) | 100 cores | 200GB | 250GB | 500 Mbps |
| Storage (S3) | - | - | 1TB/month | - |
| **Total** | **106 cores** | **212GB** | **1280GB** | **650 Mbps** |

### Observatory Integration

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| Prometheus | 2 cores | 4GB | 500GB | 20 Mbps |
| Grafana | 1 core | 2GB | 10GB | 10 Mbps |
| Jaeger | 2 cores | 4GB | 100GB | 20 Mbps |
| AlertManager | 0.5 cores | 512MB | 5GB | 5 Mbps |
| **Total** | **5.5 cores** | **10.5GB** | **615GB** | **55 Mbps** |

## Cost Breakdown

### Development Environment

| Component | Service | Monthly Cost |
|-----------|---------|--------------|
| CLI usage | Local | $0 |
| LLM API calls (testing) | OpenAI/Anthropic | $20-50 |
| Storage | Local disk | $0 |
| **Total** | | **$20-50** |

### Small Team (CI/CD)

| Component | Service | Monthly Cost |
|-----------|---------|--------------|
| GitHub Actions | 3000 min/month | $0 (free tier) |
| LLM API calls | OpenAI/Anthropic | $100-200 |
| S3 Storage | AWS | $5-10 |
| **Total** | | **$105-210** |

### Medium Organization (Embedded + Observatory)

| Component | Service | Monthly Cost |
|-----------|---------|--------------|
| Application servers | AWS EC2 (3x c5.2xlarge) | $300 |
| Monitoring stack | AWS EC2 (2x t3.large) | $150 |
| RDS PostgreSQL | db.t3.medium | $70 |
| S3 Storage | AWS | $20 |
| LLM API calls | OpenAI/Anthropic | $500-1000 |
| Data transfer | AWS | $50 |
| **Total** | | **$1090-1590** |

### Enterprise (Distributed)

| Component | Service | Monthly Cost |
|-----------|---------|--------------|
| Kubernetes cluster | AWS EKS | $150 |
| Worker nodes | 50x c5.2xlarge spot | $2000 |
| Coordinator nodes | 5x c5.xlarge | $400 |
| ElastiCache Redis | cache.r5.xlarge | $250 |
| RDS PostgreSQL | db.r5.large | $300 |
| S3 Storage | AWS | $100 |
| LLM API calls | OpenAI/Anthropic | $5000-10000 |
| Data transfer | AWS | $500 |
| CloudWatch/Monitoring | AWS | $200 |
| **Total** | | **$8900-13900** |

## Port Reference

| Service | Default Port | Protocol | Purpose |
|---------|-------------|----------|---------|
| CLI (standalone) | N/A | N/A | No server |
| Prometheus metrics | 9090 | HTTP | Metrics export |
| Coordinator HTTP | 8080 | HTTP | API/UI |
| Worker HTTP | 8081 | HTTP | Health checks |
| Worker metrics | 9091 | HTTP | Worker metrics |
| Redis | 6379 | TCP | Coordination |
| NATS | 4222 | TCP | Coordination |
| PostgreSQL | 5432 | TCP | Result storage |
| Grafana | 3000 | HTTP | Dashboards |
| Prometheus server | 9093 | HTTP | Metrics storage |
| Jaeger UI | 16686 | HTTP | Trace UI |
| Jaeger OTLP | 4317 | gRPC | Trace ingestion |
| AlertManager | 9093 | HTTP | Alert routing |

## Configuration File Comparison

| Format | Extension | Best For | Pros | Cons |
|--------|-----------|----------|------|------|
| YAML | `.yaml`, `.yml` | General use | Human-readable, comments | Indentation-sensitive |
| TOML | `.toml` | Rust projects | Type-safe, clear syntax | Less common |
| JSON | `.json` | Programmatic | Machine-parseable, strict | No comments, verbose |
| Environment | `.env` | Secrets only | Simple, secure | Limited structure |

## Metrics Naming Convention

| Metric Type | Pattern | Example | Unit |
|-------------|---------|---------|------|
| Latency (histogram) | `llm_{component}_duration_seconds` | `llm_request_duration_seconds` | seconds |
| TTFT (histogram) | `llm_ttft_duration_seconds` | `llm_ttft_duration_seconds` | seconds |
| Counter | `llm_{resource}_total` | `llm_requests_total` | count |
| Gauge | `llm_{resource}_current` | `llm_concurrent_requests` | count |
| Cost | `llm_cost_usd_total` | `llm_cost_usd_total` | USD |
| Tokens | `llm_tokens_total` | `llm_tokens_total` | count |
| Errors | `llm_errors_total` | `llm_errors_total` | count |

### Standard Labels

| Label | Description | Example Values |
|-------|-------------|----------------|
| `provider` | LLM provider name | `openai`, `anthropic`, `google` |
| `model` | Model identifier | `gpt-4-turbo-preview`, `claude-3-opus-20240229` |
| `status` | Request status | `success`, `error`, `timeout` |
| `error_type` | Error category | `rate_limit`, `timeout`, `api_error` |
| `region` | Geographic region | `us-east-1`, `eu-west-1` |
| `worker_id` | Worker identifier | `worker-1`, `worker-2` |
| `type` | Token type | `input`, `output` |

## Alert Severity Levels

| Severity | Response Time | Escalation | Example Use Cases |
|----------|---------------|------------|-------------------|
| **Critical** | Immediate | PagerDuty | Service down, API key invalid, budget exceeded |
| **Warning** | < 30 minutes | Slack | High latency, elevated error rate, cost spike |
| **Info** | Best effort | Email | Baseline update, scheduled maintenance |

## Performance Baselines

### Expected Latencies (P95)

| Model | TTFT (P95) | Total Latency (P95) | Tokens/sec |
|-------|-----------|---------------------|------------|
| GPT-4 Turbo | 500-800ms | 3000-6000ms | 40-60 |
| GPT-3.5 Turbo | 300-500ms | 1500-3000ms | 60-80 |
| Claude 3 Opus | 400-700ms | 2500-5000ms | 50-70 |
| Claude 3 Sonnet | 300-500ms | 1500-3000ms | 70-90 |
| Claude 3 Haiku | 200-400ms | 1000-2000ms | 90-120 |
| Gemini Pro | 400-600ms | 2000-4000ms | 60-80 |

*Note: Actual values vary by prompt complexity, geographic region, and time of day.*

### Error Rate Expectations

| Provider | Expected Error Rate | Common Errors |
|----------|-------------------|---------------|
| OpenAI | < 0.5% | Rate limits (429), Timeouts (504) |
| Anthropic | < 0.3% | Overloaded (529), Timeouts |
| Google | < 1.0% | Quota exceeded, Service unavailable |

## Sampling Strategies

| Environment | Sampling Rate | Rationale |
|-------------|--------------|-----------|
| Development | 100% (1.0) | Full visibility for debugging |
| Staging | 100% (1.0) | Comprehensive testing |
| Production (low traffic) | 100% (1.0) | Can afford full sampling |
| Production (medium) | 10% (0.1) | Balance cost and visibility |
| Production (high traffic) | 1% (0.01) | Minimize overhead |
| Production (critical path) | 0.1% (0.001) | Minimal impact |

## Retention Policies

| Data Type | Hot Storage | Warm Storage | Cold Storage | Total |
|-----------|------------|--------------|--------------|-------|
| **Raw metrics** | 7 days | 30 days | 90 days | 127 days |
| **Aggregated metrics** | 30 days | 90 days | 1 year | ~1 year |
| **Traces** | 3 days | 7 days | 30 days | 40 days |
| **Logs** | 7 days | 30 days | 90 days | 127 days |
| **Results (JSON)** | 30 days | 90 days | 1 year | ~1 year |
| **Reports** | 90 days | 1 year | 3 years | ~3 years |

## Storage Costs (AWS)

| Storage Tier | Service | $/GB/month | Use Case |
|--------------|---------|-----------|----------|
| Hot | S3 Standard | $0.023 | < 7 days |
| Warm | S3 IA | $0.0125 | 7-90 days |
| Cold | S3 Glacier | $0.004 | > 90 days |
| Database | RDS PostgreSQL | ~$0.10 | Active queries |
| Time Series | TimescaleDB | ~$0.15 | Metrics storage |

## Instance Type Recommendations

### AWS EC2

| Use Case | Instance Type | vCPU | Memory | Cost/hour | Notes |
|----------|--------------|------|--------|-----------|-------|
| **CLI/Development** | t3.small | 2 | 2GB | $0.02 | Burstable |
| **Small coordinator** | t3.medium | 2 | 4GB | $0.04 | Burstable |
| **Production coordinator** | c5.2xlarge | 8 | 16GB | $0.34 | Compute optimized |
| **Worker (standard)** | c5.xlarge | 4 | 8GB | $0.17 | Compute optimized |
| **Worker (spot)** | c5.xlarge spot | 4 | 8GB | $0.05 | 70% savings |
| **Monitoring** | t3.large | 2 | 8GB | $0.08 | Memory for Prometheus |

### Kubernetes Resource Requests/Limits

| Component | Request CPU | Request Memory | Limit CPU | Limit Memory |
|-----------|------------|----------------|-----------|--------------|
| **Coordinator** | 500m | 512Mi | 2000m | 2Gi |
| **Worker** | 1000m | 1Gi | 4000m | 4Gi |
| **Prometheus** | 1000m | 2Gi | 2000m | 4Gi |
| **Grafana** | 250m | 512Mi | 1000m | 1Gi |
| **Redis** | 500m | 1Gi | 1000m | 2Gi |

## Scaling Thresholds

### Horizontal Pod Autoscaler (HPA)

| Metric | Scale Up Threshold | Scale Down Threshold | Cool-down |
|--------|-------------------|---------------------|-----------|
| CPU utilization | > 70% | < 30% | 5 minutes |
| Memory utilization | > 80% | < 40% | 5 minutes |
| Request queue depth | > 100 | < 10 | 3 minutes |
| Response time (P95) | > 5000ms | < 2000ms | 10 minutes |

### Cluster Autoscaler

| Trigger | Action | Min Nodes | Max Nodes |
|---------|--------|-----------|-----------|
| Pods pending > 2 min | Add nodes | 3 | 100 |
| Node utilization < 50% | Remove nodes | 3 | 100 |
| Scale up rate | +10 nodes/min | - | - |
| Scale down rate | -1 node/min | - | - |

## Network Bandwidth Requirements

| Scenario | Workers | RPS/Worker | Total RPS | Bandwidth |
|----------|---------|-----------|-----------|-----------|
| **Small** | 5 | 10 | 50 | 10 Mbps |
| **Medium** | 20 | 25 | 500 | 100 Mbps |
| **Large** | 50 | 50 | 2500 | 500 Mbps |
| **Extra Large** | 100 | 100 | 10000 | 2 Gbps |

*Assumes ~200KB average response size*

## API Rate Limits

| Provider | Model | Requests/min | Tokens/min | Daily Limit |
|----------|-------|-------------|------------|-------------|
| **OpenAI** | GPT-4 Turbo | 500 | 150K | 10K requests |
| **OpenAI** | GPT-3.5 Turbo | 3500 | 200K | 10K requests |
| **Anthropic** | Claude 3 Opus | 1000 | 200K | Unlimited |
| **Anthropic** | Claude 3 Sonnet | 4000 | 400K | Unlimited |
| **Google** | Gemini Pro | 60 | 32K | 1500 requests |

*Tier 2 pricing; higher tiers available*

## Cost Estimation Examples

### Example 1: Small Team Daily Benchmark

```
Configuration:
- Duration: 5 minutes
- RPS: 10
- Model: GPT-4 Turbo
- Total requests: 3000

Costs:
- Input tokens: 3000 req × 100 tokens × $0.01/1K = $3.00
- Output tokens: 3000 req × 200 tokens × $0.03/1K = $18.00
- Infrastructure: Negligible (local CLI)
- Total per run: $21.00
- Total per day (1 run): $21.00
- Total per month (22 workdays): $462.00
```

### Example 2: CI/CD Integration (Per PR)

```
Configuration:
- Duration: 1 minute
- RPS: 5
- Model: GPT-3.5 Turbo
- Total requests: 300

Costs:
- Input tokens: 300 req × 100 tokens × $0.0005/1K = $0.015
- Output tokens: 300 req × 200 tokens × $0.0015/1K = $0.09
- GitHub Actions: $0 (free tier)
- Total per PR: $0.11
- Total per month (100 PRs): $11.00
```

### Example 3: Production Profiling (Embedded)

```
Configuration:
- Sampling rate: 1% of production traffic
- Production RPS: 1000
- Sampled RPS: 10
- Running 24/7
- Model: Mixed (70% GPT-3.5, 30% GPT-4)

Costs (Monthly):
- Total sampled requests: 10 × 60 × 60 × 24 × 30 = 25.92M
- GPT-3.5 (70%): 18.14M × 300 tokens × $0.0020/1K = $10,884
- GPT-4 (30%): 7.78M × 300 tokens × $0.04/1K = $9,336
- Infrastructure (EC2 + monitoring): $500
- Total per month: $20,720
```

### Example 4: Large-Scale Load Test

```
Configuration:
- Duration: 1 hour
- Total RPS: 1000 (distributed across 50 workers)
- Model: Claude 3 Opus
- Total requests: 3.6M

Costs:
- Input tokens: 3.6M req × 100 tokens × $0.015/1K = $5,400
- Output tokens: 3.6M req × 200 tokens × $0.075/1K = $54,000
- Infrastructure (spot instances, 1 hour): $150
- Total per run: $59,550
- Total per month (weekly runs): $238,200
```

## Environment Variables Reference

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` | For OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` | For Anthropic |
| `GOOGLE_API_KEY` | Google AI API key | `AIza...` | For Google |
| `DATABASE_URL` | PostgreSQL connection | `postgres://user:pass@host/db` | For DB storage |
| `REDIS_URL` | Redis connection | `redis://host:6379` | For distributed |
| `AWS_ACCESS_KEY_ID` | AWS credentials | `AKIA...` | For S3 storage |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials | `...` | For S3 storage |
| `AWS_REGION` | AWS region | `us-east-1` | For S3 storage |
| `PROMETHEUS_URL` | Prometheus endpoint | `http://prometheus:9090` | For metrics |
| `JAEGER_ENDPOINT` | Jaeger endpoint | `http://jaeger:4317` | For tracing |
| `SLACK_WEBHOOK_URL` | Slack webhook | `https://hooks.slack.com/...` | For alerts |
| `PAGERDUTY_KEY` | PagerDuty key | `...` | For alerts |
| `RUST_LOG` | Logging level | `info`, `debug` | Optional |

## Quick Command Reference

### Standalone CLI

```bash
# Initialize
llm-lens init --format yaml > config.yaml

# Run
llm-lens run --config config.yaml

# Quick test
llm-lens run --provider openai --model gpt-4-turbo-preview --duration 60s

# Validate
llm-lens validate --config config.yaml

# Compare
llm-lens compare --models gpt-4,claude-3-opus --duration 60s

# Analyze
llm-lens analyze results.json --output report.html
```

### Distributed

```bash
# Start coordinator
llm-lens distributed coordinator --redis-url redis://localhost:6379

# Start worker
llm-lens distributed worker --coordinator http://localhost:8080

# Submit job
llm-lens distributed submit --coordinator http://localhost:8080 --config config.yaml

# Check status
llm-lens distributed status --coordinator http://localhost:8080 --job-id abc123

# Cancel job
llm-lens distributed cancel --coordinator http://localhost:8080 --job-id abc123
```

### Utility Commands

```bash
# Version
llm-lens --version

# Help
llm-lens --help
llm-lens run --help

# Export metrics
llm-lens export prometheus --input results.json

# Cost analysis
llm-lens cost --input results.json --pricing pricing.yaml

# Generate report
llm-lens report --input results.json --template template.html
```

## Troubleshooting Checklist

| Issue | Check | Solution |
|-------|-------|----------|
| API errors | API key set? | `echo $OPENAI_API_KEY` |
| Rate limits | RPS too high? | Reduce in config |
| Timeouts | Network issues? | Increase timeout values |
| High memory | Buffer too large? | Reduce buffer_size |
| Workers not connecting | Network connectivity? | Check coordinator URL |
| Metrics not appearing | Prometheus scraping? | Verify scrape config |
| High costs | Sampling enabled? | Enable sampling in prod |
| Slow performance | Resource limits? | Increase CPU/memory |

## Support Contacts

| Issue Type | Contact | Response Time |
|------------|---------|---------------|
| Bugs | GitHub Issues | 1-2 business days |
| Questions | Slack #llm-latency-lens | < 4 hours |
| Security | security@company.com | < 24 hours |
| Enterprise | enterprise@company.com | < 8 hours |

---

This reference guide provides quick lookup tables for common deployment scenarios, configurations, and operational parameters.
