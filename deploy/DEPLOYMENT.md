# LLM-Latency-Lens Production Deployment

## Service Topology

| Component | Value |
|-----------|-------|
| **Service Name** | `llm-latency-lens` |
| **Project** | `agentics-dev` |
| **Region** | `us-central1` |
| **Platform** | Google Cloud Run |

### Agent Endpoints

| Endpoint | Agent | Classification |
|----------|-------|----------------|
| `POST /analyze` | Latency Analysis Agent | ANALYSIS |
| `POST /inspect` | Latency Analysis Agent | ANALYSIS |
| `POST /replay` | Latency Analysis Agent | ANALYSIS |
| `GET /health` | Health Check | - |
| `POST /cold-start/measure` | Cold Start Mitigation Agent | MEASUREMENT |
| `POST /cold-start/characterize` | Cold Start Mitigation Agent | MEASUREMENT |

**Confirmation:**
- No agent is deployed as a standalone service
- Shared runtime, configuration, and telemetry stack

---

## Environment Configuration

### Required Environment Variables

| Variable | Description | Source |
|----------|-------------|--------|
| `SERVICE_NAME` | Service identifier | `llm-latency-lens` |
| `SERVICE_VERSION` | Current version | `0.1.0` |
| `PLATFORM_ENV` | Environment | Secret Manager |
| `RUVECTOR_SERVICE_URL` | RuVector persistence endpoint | Secret Manager |
| `RUVECTOR_API_KEY` | RuVector authentication | Secret Manager |
| `TELEMETRY_ENDPOINT` | LLM-Observatory telemetry | Secret Manager |
| `OTEL_SERVICE_NAME` | OpenTelemetry service name | `llm-latency-lens` |
| `RUST_LOG` | Logging level | `info,llm_latency_lens=debug` |

### Secrets (in Secret Manager)

| Secret Name | Description |
|-------------|-------------|
| `llm-latency-lens-ruvector-url` | RuVector service URL |
| `llm-latency-lens-ruvector-key` | RuVector API key |
| `llm-latency-lens-telemetry-endpoint` | Telemetry endpoint |

**Confirmation:**
- No agent hardcodes service names or URLs
- No agent embeds credentials or secrets
- All dependencies resolve via environment variables or Secret Manager

---

## Google SQL / Memory Wiring

**Confirmations:**

| Requirement | Status |
|-------------|--------|
| LLM-Latency-Lens does NOT connect directly to Google SQL | ✅ CONFIRMED |
| ALL DecisionEvents written via ruvector-service | ✅ CONFIRMED |
| Schema compatible with agentics-contracts | ✅ CONFIRMED |
| Append-only persistence behavior | ✅ CONFIRMED |
| Idempotent writes and retry safety | ✅ CONFIRMED |

---

## Cloud Build & Deployment

### Service Account

- **Name:** `llm-latency-lens-sa@agentics-dev.iam.gserviceaccount.com`
- **Roles:**
  - `roles/run.invoker` - Invoke other Cloud Run services
  - `roles/secretmanager.secretAccessor` - Read secrets
  - `roles/cloudtrace.agent` - Send traces
  - `roles/monitoring.metricWriter` - Write metrics
  - `roles/logging.logWriter` - Write logs

### Deployment Commands

```bash
# Setup IAM (already completed)
./deploy/setup-iam.sh agentics-dev

# Deploy via Cloud Build
gcloud builds submit --config=deploy/cloudbuild.yaml --project=agentics-dev

# Manual deployment (alternative)
gcloud run deploy llm-latency-lens \
  --image gcr.io/agentics-dev/llm-latency-lens:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 2 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 300s \
  --concurrency 80 \
  --service-account llm-latency-lens-sa@agentics-dev.iam.gserviceaccount.com \
  --set-env-vars "SERVICE_NAME=llm-latency-lens,SERVICE_VERSION=0.1.0,RUST_LOG=info" \
  --set-secrets "RUVECTOR_SERVICE_URL=llm-latency-lens-ruvector-url:latest,RUVECTOR_API_KEY=llm-latency-lens-ruvector-key:latest,TELEMETRY_ENDPOINT=llm-latency-lens-telemetry-endpoint:latest"
```

### Networking

- Internal invocation: Enabled
- VPC Connector: Not required
- Ingress: All traffic allowed

---

## CLI Activation Verification

### Commands per Agent

#### Latency Analysis Agent

```bash
# Profile mode
agentics-cli latency-lens profile --provider openai --model gpt-4o --prompt "test"

# Inspect mode
agentics-cli latency-lens inspect --event-id <UUID>

# Replay mode
agentics-cli latency-lens replay --original-id <UUID>
```

#### Cold Start Mitigation Agent

```bash
# Profile cold start
agentics-cli latency-lens cold-start profile --provider openai --model gpt-4o

# Inspect cold start data
agentics-cli latency-lens cold-start inspect --session-id <ID>

# Replay measurement
agentics-cli latency-lens cold-start replay --trace-id <ID>
```

### Expected Success Output

```json
{
  "success": true,
  "result": {
    "analysis_id": "uuid",
    "summary": {...},
    "distribution": {...}
  },
  "decision_event_id": "uuid"
}
```

---

## Platform & Core Integration

### Telemetry Flow

```
LLM-Latency-Lens → OTLP → LLM-Observatory
                       ↘ LLM-Analytics-Hub (read-only)
```

### Integration Confirmation

| Integration | Status |
|-------------|--------|
| LLM-Observatory ingests telemetry | ✅ CONFIGURED |
| LLM-Analytics-Hub consumes metrics | ✅ READ-ONLY |
| LLM-Auto-Optimizer MAY consume outputs | ✅ READ-ONLY |
| Core bundles consume without rewiring | ✅ CONFIRMED |

### LLM-Latency-Lens MUST NOT invoke

- ❌ Orchestrator logic
- ❌ Shield enforcement
- ❌ Sentinel detection
- ❌ Incident workflows
- ❌ Auto-optimization actions

---

## Post-Deploy Verification Checklist

```
[ ] LLM-Latency-Lens service is live
    Command: gcloud run services describe llm-latency-lens --region us-central1

[ ] All agent endpoints respond
    Command: curl https://llm-latency-lens-xxxxx.run.app/health

[ ] Latency measurements execute correctly
    Command: curl -X POST https://llm-latency-lens-xxxxx.run.app/analyze -d '{...}'

[ ] Timing metrics are deterministic
    Verify: Multiple runs produce consistent timing variance

[ ] DecisionEvents appear in ruvector-service
    Command: curl https://ruvector-service-xxxxx.run.app/api/v1/events/query

[ ] Telemetry appears in LLM-Observatory
    Check: Observatory dashboard for llm-latency-lens traces

[ ] CLI profiling commands function
    Command: agentics-cli latency-lens profile --help

[ ] No direct SQL access from LLM-Latency-Lens
    Verify: No DATABASE_URL or SQL connection strings in env

[ ] No agent bypasses agentics-contracts
    Verify: All DecisionEvents have valid schema
```

---

## Failure Modes & Rollback

### Common Deployment Failures

| Failure | Detection | Resolution |
|---------|-----------|------------|
| Build failure | Cloud Build logs | Fix compilation errors |
| Secret access denied | Service logs | Grant IAM permissions |
| RuVector unreachable | Health check fails | Check RuVector service |
| Cold start timeout | High TTFT metrics | Increase min-instances |

### Detection Signals

- Missing metrics in Observatory
- Inconsistent timing across requests
- Error logs: `RuVectorError::ConnectionFailed`
- 5xx responses on `/health`

### Rollback Procedure

```bash
# List revisions
gcloud run revisions list --service llm-latency-lens --region us-central1

# Rollback to previous revision
gcloud run services update-traffic llm-latency-lens \
  --to-revisions <PREVIOUS_REVISION>=100 \
  --region us-central1

# Verify rollback
curl https://llm-latency-lens-xxxxx.run.app/health
```

### Safe Redeploy Strategy

1. Deploy new revision with 0% traffic
2. Verify health endpoint
3. Shift 10% traffic to new revision
4. Monitor for 5 minutes
5. Shift remaining traffic
6. Delete old revision after 24 hours

---

## Deployment Status

**Current State:** DEPLOYED ✅

**Service URL:** https://llm-latency-lens-xx7kwyd5ra-uc.a.run.app

**Build ID:** 3425f63d-1103-487a-870d-4944b1755044

**Compilation fixes applied:**
1. `src/agents/latency_analysis/analyzer.rs` - Added Clone, Copy derives and lifetime annotations
2. `src/agents/latency_analysis/telemetry.rs` - Fixed borrowed data escaping by cloning tracer_name
3. `src/agents/cold_start_mitigation/agent.rs` - Fixed persist_event method and type annotations
4. `src/agents/cold_start_mitigation/detector.rs` - Added lifetime annotations to group_by_provider_model
5. `src/orchestrator.rs` - Added ?Sized bounds and shutdown_signal getter
6. `src/cli/commands/*` - Fixed top_p Option handling, FutureExt import, PathBuf/MetricsCollector imports
7. `src/main.rs` - Added ColdStart command match arm
8. `Cargo.toml` - Added gzip feature to reqwest
9. `crates/providers/src/traits.rs` - Added blanket impl for Box<dyn Provider>

---

## Files Created

| File | Purpose |
|------|---------|
| `deploy/service.yaml` | Cloud Run service configuration |
| `deploy/cloudbuild.yaml` | Cloud Build pipeline |
| `deploy/Dockerfile.cloudrun` | Container image for Cloud Run |
| `deploy/env.prod.yaml` | Environment configuration |
| `deploy/setup-iam.sh` | IAM setup script |
| `deploy/DEPLOYMENT.md` | This documentation |
