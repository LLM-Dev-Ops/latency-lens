# Infrastructure Overview

This document provides an overview of the Docker and CI/CD infrastructure for LLM-Latency-Lens.

## 📁 Infrastructure Files

### Docker Files

| File | Purpose | Size Target |
|------|---------|-------------|
| `Dockerfile` | Multi-stage production build | < 50MB |
| `.dockerignore` | Build context optimization | - |
| `docker-compose.yml` | Local development stack | - |
| `docker-compose.prod.yml` | Production overrides | - |

### CI/CD Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `.github/workflows/ci.yml` | Push, PR | Continuous Integration |
| `.github/workflows/security.yml` | Push, PR, Daily | Security Scanning |
| `.github/workflows/release.yml` | Tag push | Release Automation |
| `.github/workflows/docker-build.yml` | PR (Docker changes) | Docker Validation |

### Configuration Files

| File | Purpose |
|------|---------|
| `deny.toml` | Dependency policy enforcement |
| `cliff.toml` | Changelog generation config |
| `.github/dependabot.yml` | Automated dependency updates |
| `.env.example` | Environment variable template |

### Monitoring Configuration

```
monitoring/
├── prometheus/
│   ├── prometheus.yml       # Scrape configuration
│   └── alerts.yml           # Alert rules
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/     # Prometheus datasource
│   │   └── dashboards/      # Dashboard provisioning
│   └── dashboards/
│       └── llm-latency-overview.json  # Main dashboard
└── alertmanager/
    └── alertmanager.yml     # Alert routing
```

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/deploy.sh` | Production deployment automation |
| `scripts/release.sh` | Release version management |
| `Makefile` | Development convenience commands |

## 🐳 Docker Architecture

### Multi-Stage Build

```
┌─────────────────────────────────────────────┐
│ Stage 1: Chef (Dependency Caching)          │
│ - Install cargo-chef                        │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Stage 2: Planner                            │
│ - Generate dependency recipe                │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Stage 3: Builder (Dependencies)             │
│ - Build dependencies (cached)               │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Stage 4: App Builder                        │
│ - Build application                         │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│ Stage 5: Runtime (Distroless)               │
│ - Minimal image                             │
│ - Non-root user (UID 65532)                 │
│ - Size: < 50MB                              │
└─────────────────────────────────────────────┘
```

### Security Features

- ✅ Distroless base image (minimal attack surface)
- ✅ Non-root user execution
- ✅ Read-only root filesystem (production)
- ✅ Dropped capabilities
- ✅ No new privileges
- ✅ Multi-platform support (amd64, arm64)

## 🔄 CI/CD Pipeline

### Continuous Integration (ci.yml)

```
┌─────────────┐
│ Code Push   │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────┐
│ Format & Lint                               │
│ - cargo fmt --check                         │
│ - cargo clippy                              │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ Tests (Matrix)                              │
│ - OS: Linux, macOS, Windows                 │
│ - Rust: stable, beta                        │
│ - Unit, Integration, Doc tests              │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ Security                                    │
│ - cargo audit                               │
│ - cargo deny                                │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ Build (Multi-platform)                      │
│ - Linux (x86_64, ARM64, MUSL)              │
│ - macOS (x86_64, ARM64)                    │
│ - Windows (x86_64)                          │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ Docker Build & Push                         │
│ - Multi-arch (amd64, arm64)                 │
│ - Push to Docker Hub & GHCR                 │
└─────────────────────────────────────────────┘
```

### Security Pipeline (security.yml)

```
┌─────────────┐
│ Daily Scan  │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────┐
│ Cargo Audit                                 │
│ - Check known vulnerabilities               │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ SAST Scanning                               │
│ - Semgrep                                   │
│ - CodeQL                                    │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ Secret Scanning                             │
│ - Gitleaks                                  │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ Container Scanning                          │
│ - Trivy (filesystem & container)            │
│ - Snyk (optional)                           │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ SBOM Generation                             │
│ - cargo-sbom                                │
│ - Grype scanning                            │
└─────────────────────────────────────────────┘
```

### Release Pipeline (release.yml)

```
┌─────────────┐
│ Tag Push    │
│ (v1.0.0)    │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────┐
│ Create Release                              │
│ - Generate changelog (git-cliff)            │
│ - Create GitHub release                     │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ Build Binaries                              │
│ - All platforms                             │
│ - Create archives + checksums               │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ Docker Release                              │
│ - Build multi-arch images                   │
│ - Tag with version + latest                 │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ Publish Crates                              │
│ - Publish to crates.io                      │
│ - Update Homebrew (optional)                │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│ Notify                                      │
│ - Slack notification                        │
│ - Generate SBOM                             │
└─────────────────────────────────────────────┘
```

## 📊 Monitoring Stack

### Architecture

```
┌─────────────────────────────────────────────┐
│ LLM-Latency-Lens                            │
│ Port: 9090                                  │
│ Metrics: /metrics (Prometheus format)      │
└─────────────────┬───────────────────────────┘
                  │
                  │ scrape
                  │
┌─────────────────▼───────────────────────────┐
│ Prometheus                                  │
│ Port: 9091                                  │
│ - Metrics collection                        │
│ - Alert evaluation                          │
│ - 30-day retention                          │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        │ alerts            │ query
        │                   │
┌───────▼─────────┐  ┌──────▼──────────────────┐
│ AlertManager    │  │ Grafana                 │
│ Port: 9093      │  │ Port: 3000              │
│ - Alert routing │  │ - Dashboards            │
│ - Notifications │  │ - Visualization         │
└─────────────────┘  └─────────────────────────┘
```

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `llm_request_duration_seconds` | Histogram | Request latency |
| `llm_requests_total` | Counter | Total requests |
| `llm_request_errors_total` | Counter | Error count |
| `llm_tokens_generated_total` | Counter | Output tokens |
| `llm_tokens_prompt_total` | Counter | Input tokens |
| `llm_ttft_seconds` | Histogram | Time to first token |
| `llm_tpot_seconds` | Histogram | Time per output token |

### Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| HighLLMLatency | p95 > 5s for 5m | Warning |
| VeryHighLLMLatency | p95 > 10s for 2m | Critical |
| HighErrorRate | error_rate > 10% for 3m | Warning |
| ServiceDown | up == 0 for 1m | Critical |

## 🚀 Deployment Options

### Local Development

```bash
# Using Make
make up

# Using Docker Compose
docker-compose up -d

# Using script
./scripts/deploy.sh deploy
```

### Production (Single Server)

```bash
# Automated deployment
./scripts/deploy.sh deploy

# Manual deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Cloud Platforms

- **AWS**: EC2, ECS, EKS
- **Google Cloud**: GCE, Cloud Run, GKE
- **Azure**: ACI, AKS
- **DigitalOcean**: Droplets, App Platform, Kubernetes

### Container Orchestration

- **Kubernetes**: Deployment + Service
- **Docker Swarm**: Stack deployment
- **Nomad**: Job specification

## 📦 Release Process

### Semantic Versioning

```
MAJOR.MINOR.PATCH
  │     │     │
  │     │     └─ Bug fixes
  │     └─────── New features (backward compatible)
  └───────────── Breaking changes
```

### Release Steps

```bash
# Automated release
make release-patch  # 0.1.0 -> 0.1.1
make release-minor  # 0.1.0 -> 0.2.0
make release-major  # 0.1.0 -> 1.0.0

# Manual release
./scripts/release.sh patch

# Dry run
./scripts/release.sh dry
```

### Release Artifacts

- ✅ GitHub Release with changelog
- ✅ Cross-platform binaries (6 targets)
- ✅ Docker images (multi-arch)
- ✅ Crates.io publication
- ✅ Homebrew formula (optional)
- ✅ SBOM (Software Bill of Materials)

## 🔒 Security Features

### Build-time Security

- ✅ Multi-stage build (minimal attack surface)
- ✅ Dependency pinning (Cargo.lock)
- ✅ Security audits (cargo-audit)
- ✅ License compliance (cargo-deny)
- ✅ SBOM generation

### Runtime Security

- ✅ Non-root user execution
- ✅ Read-only root filesystem
- ✅ Dropped capabilities
- ✅ No new privileges flag
- ✅ Resource limits
- ✅ Network isolation

### CI/CD Security

- ✅ Daily security scans
- ✅ Dependency review (PRs)
- ✅ Secret scanning (Gitleaks)
- ✅ SAST (Semgrep, CodeQL)
- ✅ Container scanning (Trivy, Snyk)
- ✅ SLSA provenance

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [DOCKER.md](docs/DOCKER.md) | Docker usage guide |
| [CI-CD.md](docs/CI-CD.md) | CI/CD pipeline documentation |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment guide |
| [README.md](README.md) | Main project documentation |

## 🛠️ Quick Commands

```bash
# Development
make help           # Show all commands
make build          # Build debug binary
make test           # Run tests
make ci             # Run all CI checks locally

# Docker
make docker-build   # Build Docker image
make up             # Start services
make down           # Stop services
make logs           # View logs

# Deployment
./scripts/deploy.sh deploy    # Deploy to production
./scripts/deploy.sh status    # Check service status
./scripts/deploy.sh rollback  # Rollback deployment

# Release
./scripts/release.sh patch    # Create patch release
./scripts/release.sh minor    # Create minor release
./scripts/release.sh major    # Create major release
```

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/llm-devops/llm-latency-lens/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llm-devops/llm-latency-lens/discussions)
- **Documentation**: [docs/](docs/)

## 📄 License

Apache-2.0 - See [LICENSE](LICENSE) for details.
