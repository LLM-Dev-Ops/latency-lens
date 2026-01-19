# Cold Start Mitigation Agent

## Agent Contract Summary

**Agent Name**: Cold Start Mitigation Agent
**Agent ID**: `cold-start-mitigation-agent`
**Version**: `0.1.0`
**Classification**: MEASUREMENT
**Decision Type**: `cold_start_measurement`

## Purpose Statement

Measure and characterize cold start behavior and startup latency of LLM services. This agent detects cold vs warm execution paths, measures initialization and startup delays, and quantifies cold start frequency and impact. All outputs are diagnostic only.

## Agent Classification

- **MEASUREMENT** - Captures raw timing data for cold start detection
- Does NOT interpret or make recommendations
- Produces diagnostic outputs only

## Input Schema (agentics-contracts)

```rust
ColdStartMeasurementInput {
    measurement_id: Uuid,
    session_id: SessionId,
    metrics: Vec<RequestMetrics>,
    config: ColdStartMeasurementConfig,
}

ColdStartMeasurementConfig {
    min_samples: u64,                           // Minimum samples required
    cold_start_threshold_multiplier: f64,       // Detection threshold
    baseline_warmup_count: u32,                 // Warmup requests for baseline
    include_first_request: bool,                // Include first as cold start
    confidence_level: f64,                      // Statistical confidence
    provider_filter: Option<Provider>,          // Filter by provider
    model_filter: Option<String>,               // Filter by model
    max_idle_time_ms: Option<u64>,              // Max idle time threshold
    min_idle_time_ms: u64,                      // Min idle time threshold
}
```

## Output Schema (agentics-contracts)

```rust
ColdStartMeasurementOutput {
    measurement_id: Uuid,
    session_id: SessionId,
    summary: ColdStartSummary,
    cold_start_events: Vec<ColdStartEvent>,
    baseline_latency: LatencyDistribution,
    cold_start_latency: Option<LatencyDistribution>,
    provider_analysis: Vec<ProviderColdStartAnalysis>,
    model_analysis: Vec<ModelColdStartAnalysis>,
    metadata: ColdStartMeasurementMetadata,
}
```

## Metric and Timing Definitions

| Metric | Description | Unit |
|--------|-------------|------|
| cold_start_count | Number of detected cold starts | count |
| warm_request_count | Number of warm requests | count |
| cold_start_frequency | Percentage of cold starts | percent |
| avg_cold_start_overhead | Mean latency overhead | nanoseconds |
| max_cold_start_overhead | Maximum latency overhead | nanoseconds |
| avg_cold_start_ratio | Mean cold/warm latency ratio | ratio |
| cold_start_impact_percent | Impact on total latency | percent |

## DecisionEvent Mapping

Every invocation emits exactly ONE `DecisionEvent` to ruvector-service:

```rust
DecisionEvent {
    event_id: Uuid,
    agent_id: "cold-start-mitigation-agent",
    agent_version: "0.1.0",
    decision_type: DecisionType::ColdStartMeasurement,
    classification: AgentClassification::Measurement,
    inputs_hash: SHA-256 hash of input,
    outputs: ColdStartMeasurementOutput (serialized),
    confidence: ConfidenceMetadata {
        score: 0.0-1.0,
        precision_unit: "nanoseconds",
        error_bounds: {...},
        sample_size: u64,
        confidence_interval_percent: 95.0,
    },
    constraints_applied: MeasurementConstraints {
        min_sample_size: Some(u64),
        warmup_excluded: bool,
        warmup_count: Some(u32),
        ...
    },
    execution_ref: measurement_id,
    timestamp: UTC,
}
```

## CLI Contract

### Profile Command
```bash
llm-latency-lens cold-start profile \
    --provider openai \
    --model gpt-4 \
    --prompt "Hello" \
    --requests 10 \
    --delay-ms 100 \
    --threshold 2.0 \
    --algorithm threshold \
    --output results.json \
    --persist
```

### Inspect Command
```bash
llm-latency-lens cold-start inspect \
    --input metrics.json \
    --threshold 2.0 \
    --algorithm zscore \
    --format table
```

### Replay Command
```bash
llm-latency-lens cold-start replay \
    --trace-id <uuid> \
    --format json
```

## Detection Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| `threshold` | Simple threshold-based detection | Default, fast |
| `zscore` | Z-score outlier detection | Statistical accuracy |
| `moving_average` | Moving average baseline comparison | Adaptive |
| `inter_arrival` | Inter-arrival time based detection | Idle time correlation |

## Downstream Consumers

These systems MAY consume this agent's output:
- LLM-Observatory (telemetry)
- LLM-Auto-Optimizer (advisory)
- LLM-Analytics-Hub (historical analysis)

## Explicit Non-Responsibilities

This agent MUST NEVER:
- ❌ Trigger warm-up logic
- ❌ Apply mitigations
- ❌ Modify deployment behavior
- ❌ Route traffic
- ❌ Execute SQL directly
- ❌ Orchestrate workflows
- ❌ Enforce policies

## Failure Modes

| Failure | Handling |
|---------|----------|
| Empty metrics | Return `ColdStartInputValidationError::EmptyMetrics` |
| Insufficient samples | Return `ColdStartInputValidationError::InsufficientSamples` |
| Invalid config | Return `ColdStartInputValidationError::InvalidConfig` |
| RuVector unavailable | Log warning, continue without persistence |
| Detection failure | Return `ColdStartAgentError::DetectionError` |

## Deployment Model

- **Platform**: Google Cloud Edge Functions
- **Service**: Unified LLM-Latency-Lens service
- **State**: Stateless at runtime
- **Persistence**: Via ruvector-service only
- **SQL Access**: NEVER direct - ruvector-service client only

## Telemetry

Emits LLM-Observatory compatible telemetry via tracing:

```rust
tracing::info!(
    target: "llm_observatory",
    measurement_id = %output.measurement_id,
    session_id = %output.session_id,
    cold_start_count = output.summary.cold_start_count,
    cold_start_frequency = output.summary.cold_start_frequency,
    "cold_start_measurement_complete"
);
```

## Verification Checklist

- [ ] Agent imports schemas from agentics-contracts only
- [ ] All inputs validated against contracts before processing
- [ ] All outputs conform to agentics-contracts schemas
- [ ] Exactly ONE DecisionEvent emitted per invocation
- [ ] DecisionEvent persisted to ruvector-service
- [ ] Telemetry visible in LLM-Observatory
- [ ] CLI commands work: profile, inspect, replay
- [ ] Agent does NOT modify system behavior
- [ ] Agent does NOT trigger warm-up logic
- [ ] Agent does NOT apply mitigations
- [ ] Agent does NOT execute SQL directly
- [ ] Results are deterministic and reproducible

## Smoke Test Commands

```bash
# Profile cold start behavior
llm-latency-lens cold-start profile \
    --provider openai \
    --model gpt-4 \
    --prompt "Say hello" \
    --requests 5 \
    --delay-ms 500

# Inspect from file
llm-latency-lens cold-start inspect \
    --input benchmarks/sample-metrics.json \
    --threshold 2.0 \
    --format json

# Replay from RuVector
llm-latency-lens cold-start replay \
    --trace-id <previous-measurement-id>
```

## Files

| File | Description |
|------|-------------|
| `src/agents/cold_start_mitigation/mod.rs` | Module definition |
| `src/agents/cold_start_mitigation/agent.rs` | Agent implementation |
| `src/agents/cold_start_mitigation/detector.rs` | Detection algorithms |
| `src/agents/cold_start_mitigation/schemas.rs` | Input/output schemas |
| `src/agents/ruvector.rs` | RuVector client |
| `src/agents/edge_function.rs` | Edge function handler |
| `src/cli/commands/cold_start.rs` | CLI commands |
