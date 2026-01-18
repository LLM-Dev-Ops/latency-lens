# Quick Start Guide

Get LLM-Latency-Lens up and running in 5 minutes.

## Prerequisites

- Docker >= 20.10
- Docker Compose >= 2.0
- API keys from LLM providers

## 1. Clone Repository

```bash
git clone https://github.com/llm-devops/llm-latency-lens.git
cd llm-latency-lens
```

## 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env  # or vim, code, etc.
```

Minimum required configuration in `.env`:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## 3. Start Services

### Option A: Using Make (Recommended)

```bash
# Start everything
make up

# View logs
make logs

# Access services:
# - Grafana:     http://localhost:3000 (admin/admin)
# - Prometheus:  http://localhost:9091
# - Metrics:     http://localhost:9090/metrics
```

### Option B: Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option C: Using Deployment Script

```bash
# Deploy with checks
./scripts/deploy.sh deploy

# Check status
./scripts/deploy.sh status
```

## 4. Access Services

### Grafana Dashboard

1. Open http://localhost:3000
2. Login: `admin` / `admin` (change password on first login)
3. Navigate to: **Dashboards** → **LLM Monitoring** → **LLM Latency Lens - Overview**

### Prometheus

- UI: http://localhost:9091
- Query example: `llm_request_duration_seconds`

### Raw Metrics

- Endpoint: http://localhost:9090/metrics
- Format: Prometheus text format

## 5. Run Benchmarks

### Using Docker

```bash
# Basic benchmark
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  llm-latency-lens benchmark --provider openai --model gpt-4

# With custom configuration
docker run --rm \
  -v $(pwd)/config:/app/config:ro \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  llm-latency-lens benchmark --config /app/config/benchmark.toml
```

### Using Local Binary

```bash
# Build from source
cargo build --release

# Run benchmark
./target/release/llm-latency-lens benchmark \
  --provider openai \
  --model gpt-4 \
  --requests 100
```

## 6. View Results

Results appear in:
- Grafana dashboards (real-time)
- Console output (immediate)
- Prometheus metrics (queryable)
- Export files (CSV, JSON)

## Common Commands

```bash
# Development
make help           # Show all available commands
make test           # Run tests
make build          # Build binary
make ci             # Run CI checks locally

# Docker
make up             # Start services
make down           # Stop services
make logs           # View logs
make restart        # Restart services
make ps             # Show service status

# Monitoring
make grafana        # Open Grafana in browser
make prometheus     # Open Prometheus in browser
make metrics        # Show metrics

# Deployment
./scripts/deploy.sh deploy    # Deploy to production
./scripts/deploy.sh status    # Check status
./scripts/deploy.sh logs      # View logs
./scripts/deploy.sh backup    # Create backup
```

## Next Steps

### 1. Configure Alerts

Edit `monitoring/prometheus/alerts.yml`:

```yaml
- alert: HighLatency
  expr: llm_request_duration_seconds{quantile="0.95"} > 5
  for: 5m
```

### 2. Customize Dashboards

- Import additional dashboards in Grafana
- Create custom queries in Prometheus
- Edit `monitoring/grafana/dashboards/`

### 3. Set Up Notifications

Edit `monitoring/alertmanager/alertmanager.yml`:

```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK'
        channel: '#alerts'
```

### 4. Production Deployment

```bash
# Deploy with production settings
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or use the deployment script
./scripts/deploy.sh deploy
```

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs

# Check Docker
docker ps
docker stats
```

### Can't access Grafana

```bash
# Verify Grafana is running
docker-compose ps grafana

# Check port mapping
docker-compose port grafana 3000

# Reset admin password
docker-compose exec grafana grafana-cli admin reset-admin-password admin
```

### Metrics not appearing

```bash
# Check Prometheus targets
curl http://localhost:9091/targets

# Verify metrics endpoint
curl http://localhost:9090/metrics

# Restart Prometheus
docker-compose restart prometheus
```

### Out of disk space

```bash
# Clean up Docker
docker system prune -a

# Clean up old logs
find logs/ -mtime +7 -delete

# Reduce Prometheus retention
# Edit monitoring/prometheus/prometheus.yml:
# --storage.tsdb.retention.time=7d
```

## Documentation

- [Docker Guide](docs/DOCKER.md) - Docker usage and configuration
- [CI/CD Guide](docs/CI-CD.md) - CI/CD pipeline documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [Infrastructure Overview](INFRASTRUCTURE.md) - Complete infrastructure reference
- [Main README](README.md) - Project documentation

## Verification

Run the verification script to ensure all infrastructure is in place:

```bash
./scripts/verify-infrastructure.sh
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/llm-devops/llm-latency-lens/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llm-devops/llm-latency-lens/discussions)
- **Documentation**: [docs/](docs/)

## What's Included

This setup includes:

- ✅ **LLM-Latency-Lens**: Main application
- ✅ **Prometheus**: Metrics collection and storage
- ✅ **Grafana**: Visualization and dashboards
- ✅ **AlertManager**: Alert routing and notifications
- ✅ **Pre-configured dashboards**: LLM performance overview
- ✅ **Alert rules**: High latency, errors, service health
- ✅ **Production-ready configuration**: Security, resource limits, health checks

## Architecture

```
┌─────────────────────────────────────────────┐
│ LLM-Latency-Lens                            │
│ - Measures LLM performance                  │
│ - Exposes Prometheus metrics                │
└─────────────────┬───────────────────────────┘
                  │
                  │ scrapes
                  ↓
┌─────────────────────────────────────────────┐
│ Prometheus                                  │
│ - Collects metrics                          │
│ - Evaluates alerts                          │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
┌─────────────────┐  ┌──────────────────────┐
│ AlertManager    │  │ Grafana              │
│ - Routes alerts │  │ - Visualizes data    │
│ - Notifications │  │ - Dashboards         │
└─────────────────┘  └──────────────────────┘
```

## License

Apache-2.0 - See [LICENSE](LICENSE) for details.
