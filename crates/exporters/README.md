# llm-latency-lens-exporters

Multi-format export system for LLM Latency Lens metrics.

## Overview

This crate provides production-ready exporters for outputting metrics in various formats suitable for different use cases:

- **JSON**: Human-readable or compact JSON output
- **Console**: Beautiful colored table output with Unicode box drawing
- **CSV**: Comma-separated values for data analysis
- **Prometheus**: Exposition format for Prometheus monitoring

## Features

- Unified `Exporter` trait interface
- Pretty-printing and compact modes
- Colored terminal output
- File export utilities
- Comprehensive formatting options
- Full serialization support

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
llm-latency-lens-exporters = "0.1.0"
llm-latency-lens-metrics = "0.1.0"
```

### Quick Start

```rust
use llm_latency_lens_exporters::{Exporter, JsonExporter, ConsoleExporter};
use llm_latency_lens_metrics::MetricsCollector;

let collector = MetricsCollector::with_defaults(session_id)?;
// ... collect metrics ...
let aggregated = MetricsAggregator::aggregate(&collector)?;

// Export to JSON
let json_exporter = JsonExporter::new(true);  // pretty print
let json = json_exporter.export(&aggregated)?;
println!("{}", json);

// Export to console
let console_exporter = ConsoleExporter::new();
console_exporter.export(&aggregated)?;  // Prints formatted tables
```

## Exporters

### JsonExporter

Export metrics as JSON with optional pretty-printing:

```rust
use llm_latency_lens_exporters::JsonExporter;

// Pretty-printed JSON (human-readable)
let exporter = JsonExporter::new(true);
let json = exporter.export(&metrics)?;

// Compact JSON (machine-readable)
let exporter = JsonExporter::new(false);
let json = exporter.export(&metrics)?;

// Save to file
exporter.export_to_file(&metrics, "metrics.json".as_ref())?;
```

**Output Example:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_requests": 100,
  "successful_requests": 95,
  "success_rate": 95.0,
  "ttft_distribution": {
    "p50": "150ms",
    "p95": "280ms",
    "p99": "350ms"
  },
  "throughput": {
    "mean_tokens_per_second": 48.5
  },
  "total_cost_usd": 2.45
}
```

### ConsoleExporter

Beautiful terminal output with colored tables:

```rust
use llm_latency_lens_exporters::ConsoleExporter;

let exporter = ConsoleExporter::new();

// Print to stdout
exporter.export(&metrics)?;

// Capture as string
let output = exporter.export(&metrics)?;
```

**Output Example:**
```
╭─────────────────────────────────────────────────────────╮
│              LLM Latency Lens - Results                 │
├─────────────────────────────────────────────────────────┤
│ Session ID    │ 550e8400-e29b-41d4-a716-446655440000   │
│ Total Requests│ 100                                     │
│ Success Rate  │ 95.0%                                   │
│ Duration      │ 10m 5s                                  │
├─────────────────────────────────────────────────────────┤
│                   TTFT Statistics                       │
├─────────────────────────────────────────────────────────┤
│ P50           │ 150ms                                   │
│ P90           │ 250ms                                   │
│ P95           │ 280ms                                   │
│ P99           │ 350ms                                   │
├─────────────────────────────────────────────────────────┤
│               Throughput Statistics                     │
├─────────────────────────────────────────────────────────┤
│ Mean          │ 48.5 tokens/sec                         │
│ P95           │ 65.0 tokens/sec                         │
├─────────────────────────────────────────────────────────┤
│                    Cost Summary                         │
├─────────────────────────────────────────────────────────┤
│ Total Cost    │ $2.45                                   │
│ Avg/Request   │ $0.0245                                 │
╰─────────────────────────────────────────────────────────╯
```

Features:
- Color-coded success/failure indicators
- Unicode box drawing for tables
- Automatic terminal width detection
- Human-readable duration formatting
- Currency formatting for costs

### CsvExporter

Export to CSV for analysis in spreadsheet tools:

```rust
use llm_latency_lens_exporters::CsvExporter;

let exporter = CsvExporter::new();

// Export aggregated metrics
exporter.export_to_file(&metrics, "summary.csv".as_ref())?;

// Export individual requests (detailed)
let requests = collector.get_all_requests()?;
exporter.export_requests_to_file(&requests, "requests.csv".as_ref())?;
```

**Aggregated Metrics CSV:**
```csv
session_id,total_requests,successful_requests,failed_requests,success_rate,ttft_p50_ms,ttft_p95_ms,ttft_p99_ms,throughput_mean,total_cost_usd
550e8400-...,100,95,5,95.0,150,280,350,48.5,2.45
```

**Individual Requests CSV:**
```csv
request_id,timestamp,provider,model,ttft_ms,total_latency_ms,tokens_per_sec,input_tokens,output_tokens,cost_usd,success,error
a1b2c3d4-...,2024-01-15T10:30:00Z,OpenAI,gpt-4o,145,2150,47.5,100,200,0.025,true,
e5f6g7h8-...,2024-01-15T10:30:05Z,Anthropic,claude-3-5-sonnet,165,1890,52.3,150,250,0.032,true,
```

### PrometheusExporter

Export metrics in Prometheus exposition format:

```rust
use llm_latency_lens_exporters::PrometheusExporter;

let exporter = PrometheusExporter::new("llm_latency");

// Export to string
let prometheus = exporter.export(&metrics)?;

// Serve via HTTP (example with actix-web)
// GET /metrics
async fn metrics_handler(collector: web::Data<MetricsCollector>) -> HttpResponse {
    let aggregated = MetricsAggregator::aggregate(&collector)?;
    let exporter = PrometheusExporter::new("llm_latency");
    let output = exporter.export(&aggregated)?;
    HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4")
        .body(output)
}
```

**Output Example:**
```prometheus
# HELP llm_latency_requests_total Total number of LLM requests
# TYPE llm_latency_requests_total counter
llm_latency_requests_total 100

# HELP llm_latency_requests_success Successful LLM requests
# TYPE llm_latency_requests_success counter
llm_latency_requests_success 95

# HELP llm_latency_ttft_seconds Time to first token
# TYPE llm_latency_ttft_seconds summary
llm_latency_ttft_seconds{quantile="0.5"} 0.150
llm_latency_ttft_seconds{quantile="0.9"} 0.250
llm_latency_ttft_seconds{quantile="0.95"} 0.280
llm_latency_ttft_seconds{quantile="0.99"} 0.350
llm_latency_ttft_seconds_sum 15.5
llm_latency_ttft_seconds_count 100

# HELP llm_latency_throughput_tokens_per_second Token generation throughput
# TYPE llm_latency_throughput_tokens_per_second gauge
llm_latency_throughput_tokens_per_second 48.5

# HELP llm_latency_cost_usd_total Total cost in USD
# TYPE llm_latency_cost_usd_total counter
llm_latency_cost_usd_total 2.45
```

## Exporter Trait

All exporters implement the `Exporter` trait:

```rust
pub trait Exporter {
    /// Export aggregated metrics to a string
    fn export(&self, metrics: &AggregatedMetrics) -> Result<String>;

    /// Export individual request metrics to a string
    fn export_requests(&self, requests: &[RequestMetrics]) -> Result<String>;

    /// Write metrics to a file
    fn export_to_file(
        &self,
        metrics: &AggregatedMetrics,
        path: &Path,
    ) -> Result<()>;

    /// Write request metrics to a file
    fn export_requests_to_file(
        &self,
        requests: &[RequestMetrics],
        path: &Path,
    ) -> Result<()>;
}
```

## Custom Exporters

Implement the `Exporter` trait for custom formats:

```rust
use llm_latency_lens_exporters::{Exporter, ExportError, Result};
use llm_latency_lens_metrics::{AggregatedMetrics, RequestMetrics};

struct XmlExporter;

impl Exporter for XmlExporter {
    fn export(&self, metrics: &AggregatedMetrics) -> Result<String> {
        // Generate XML
        let xml = format!(
            r#"<?xml version="1.0"?>
            <metrics>
                <total_requests>{}</total_requests>
                <success_rate>{:.2}</success_rate>
            </metrics>"#,
            metrics.total_requests,
            metrics.success_rate()
        );
        Ok(xml)
    }

    fn export_requests(&self, requests: &[RequestMetrics]) -> Result<String> {
        // Generate XML for requests
        todo!()
    }
}
```

## File Export Examples

### Export to Multiple Formats

```rust
use std::path::Path;

// Export to all formats
let json_exporter = JsonExporter::new(true);
json_exporter.export_to_file(&metrics, Path::new("output.json"))?;

let csv_exporter = CsvExporter::new();
csv_exporter.export_to_file(&metrics, Path::new("output.csv"))?;

let prometheus_exporter = PrometheusExporter::new("llm");
prometheus_exporter.export_to_file(&metrics, Path::new("metrics.prom"))?;
```

### Export Individual Requests

```rust
// Get all requests from collector
let requests = collector.get_all_requests()?;

// Export detailed request data
let csv_exporter = CsvExporter::new();
csv_exporter.export_requests_to_file(&requests, Path::new("requests.csv"))?;

let json_exporter = JsonExporter::new(true);
json_exporter.export_requests_to_file(&requests, Path::new("requests.json"))?;
```

## Console Formatting Features

The console exporter provides rich formatting:

```rust
let exporter = ConsoleExporter::new();

// Colored output
// - Green for successful operations
// - Red for failures
// - Yellow for warnings
// - Blue for informational text

// Table formatting
// - Unicode box drawing characters
// - Automatic column width adjustment
// - Header/footer separators

// Human-readable units
// - Durations: ms, s, m, h
// - Costs: $X.XX format
// - Percentages: X.XX%
// - Throughput: tokens/sec
```

## Error Handling

All exporters return `Result<String, ExportError>`:

```rust
use llm_latency_lens_exporters::{Exporter, ExportError};

match exporter.export(&metrics) {
    Ok(output) => println!("{}", output),
    Err(ExportError::Serialization(e)) => eprintln!("JSON error: {}", e),
    Err(ExportError::Io(e)) => eprintln!("File error: {}", e),
    Err(ExportError::Format(e)) => eprintln!("Format error: {}", e),
    Err(ExportError::Csv(e)) => eprintln!("CSV error: {}", e),
}
```

## Performance

- **JSON export**: ~100µs for typical metrics
- **Console export**: ~500µs (includes formatting)
- **CSV export**: ~200µs per 1000 requests
- **Prometheus export**: ~150µs for typical metrics

All exporters are single-threaded and perform formatting synchronously.

## Testing

All exporters include comprehensive tests:

```bash
cargo test --package llm-latency-lens-exporters
```

Test coverage includes:
- Format validation
- File I/O operations
- Error handling
- Unicode handling
- Large dataset handling

## Dependencies

- `llm-latency-lens-core`: Core types
- `llm-latency-lens-metrics`: Metrics structures
- `serde`/`serde_json`: JSON serialization
- `tabled`: Table formatting
- `colored`: Terminal colors
- `prometheus`: Prometheus format
- `chrono`: Timestamp formatting

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Contributing

This crate is part of the LLM Latency Lens project. See the main repository for contribution guidelines.
