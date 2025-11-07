# llm-latency-lens-core

Core timing engine and types for LLM Latency Lens.

## Overview

This crate provides the foundational infrastructure for high-precision measurement of LLM API latency. It includes:

- **Timing Engine**: Monotonic clock-based timing with nanosecond precision using the `quanta` crate
- **Core Types**: Session IDs, Request IDs, timestamps, and timing events
- **Provider Trait**: Abstract interface for LLM providers
- **Error Handling**: Comprehensive error types for timing operations

## Features

- Zero-overhead timing measurements
- Thread-safe session and request tracking
- Precise event timestamps for:
  - Request start/end
  - First token received (TTFT)
  - Individual token arrivals
  - Stream completion
- Provider-agnostic design

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
llm-latency-lens-core = "0.1.0"
```

### Example

```rust
use llm_latency_lens_core::{TimingEngine, SessionId, RequestId};

// Create a timing engine
let engine = TimingEngine::new();
let session_id = SessionId::new();
let request_id = RequestId::new();

// Record timing events
engine.record_request_start(session_id, request_id);
// ... make LLM API call ...
engine.record_first_token(session_id, request_id);
// ... process tokens ...
engine.record_request_end(session_id, request_id);

// Get timing data
let metrics = engine.get_request_metrics(session_id, request_id)?;
println!("TTFT: {:?}", metrics.ttft);
```

## Core Types

### TimingEngine

The main timing component that provides monotonic clock access and event recording:

```rust
pub struct TimingEngine {
    clock: Clock,
}
```

### SessionId and RequestId

Type-safe identifiers for tracking sessions and individual requests:

```rust
pub struct SessionId(Uuid);
pub struct RequestId(Uuid);
```

### Provider Trait

Abstract interface that all LLM providers must implement:

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    async fn send_request(&self, request: StreamingRequest)
        -> Result<StreamingResponse>;
    fn name(&self) -> &str;
}
```

## Performance

The timing engine uses `quanta` for high-precision monotonic timing with minimal overhead:

- Sub-microsecond precision on modern hardware
- Lock-free timing operations
- Zero-allocation hot path

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Contributing

This crate is part of the LLM Latency Lens project. See the main repository for contribution guidelines.
