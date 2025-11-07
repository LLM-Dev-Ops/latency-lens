# @llm-devops/latency-lens

> High-precision LLM latency profiler for JavaScript/TypeScript applications

[![npm version](https://badge.fury.io/js/@llm-devops%2Flatency-lens.svg)](https://www.npmjs.com/package/@llm-devops/latency-lens)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Measure and analyze LLM API performance with microsecond precision. Built with WebAssembly for native-level performance in JavaScript/TypeScript applications.

## Features

- **High-precision timing** - Nanosecond-accurate measurements using WebAssembly
- **Comprehensive metrics** - TTFT, inter-token latency, total latency, throughput
- **Statistical analysis** - Percentiles (p50, p90, p95, p99, p99.9) using HDR Histogram
- **Multi-provider support** - OpenAI, Anthropic, Google, AWS Bedrock, Azure OpenAI
- **Zero dependencies** - Pure WebAssembly with no runtime dependencies
- **Type-safe** - Full TypeScript support with detailed type definitions

## Installation

```bash
npm install @llm-devops/latency-lens
```

## Quick Start

```typescript
import { LatencyCollector } from '@llm-devops/latency-lens';

// Create a collector
const collector = new LatencyCollector();

// Record metrics from your LLM requests
collector.record({
  provider: 'openai',
  model: 'gpt-4',
  ttft_ms: 150,                              // Time to first token
  total_latency_ms: 2000,                    // Total request time
  inter_token_latencies_ms: [10, 15, 12, 11], // Time between tokens
  input_tokens: 100,
  output_tokens: 50,
  tokens_per_second: 25.0,
  cost_usd: 0.05,
  success: true
});

// Get aggregated statistics
const stats = collector.aggregate();

console.log('TTFT p99:', stats.ttft_distribution.p99_ms, 'ms');
console.log('Success rate:', stats.success_rate, '%');
console.log('Avg throughput:', stats.throughput.mean_tokens_per_second, 'tokens/sec');
```

## Usage Examples

### Basic Usage

```typescript
import { LatencyCollector } from '@llm-devops/latency-lens';

// Initialize collector
const collector = new LatencyCollector();

// Record a successful request
collector.record({
  provider: 'anthropic',
  model: 'claude-3-opus',
  ttft_ms: 120,
  total_latency_ms: 1800,
  inter_token_latencies_ms: [8, 12, 10, 9, 11],
  input_tokens: 150,
  output_tokens: 75,
  tokens_per_second: 41.6,
  success: true
});

// Record a failed request
collector.record({
  provider: 'openai',
  model: 'gpt-4',
  ttft_ms: 0,
  total_latency_ms: 5000,
  inter_token_latencies_ms: [],
  input_tokens: 100,
  output_tokens: 0,
  tokens_per_second: 0,
  success: false,
  error: 'Request timeout'
});
```

### Analyzing Results

```typescript
// Get overall statistics
const stats = collector.aggregate();

console.log('Total requests:', stats.total_requests);
console.log('Success rate:', stats.success_rate.toFixed(2), '%');

// TTFT analysis
console.log('\nTime to First Token:');
console.log('  p50:', stats.ttft_distribution.p50_ms.toFixed(2), 'ms');
console.log('  p95:', stats.ttft_distribution.p95_ms.toFixed(2), 'ms');
console.log('  p99:', stats.ttft_distribution.p99_ms.toFixed(2), 'ms');

// Throughput analysis
console.log('\nThroughput:');
console.log('  Mean:', stats.throughput.mean_tokens_per_second.toFixed(2), 'tokens/sec');
console.log('  p50:', stats.throughput.p50_tokens_per_second.toFixed(2), 'tokens/sec');

// Cost analysis
if (stats.total_cost_usd) {
  console.log('\nCost:');
  console.log('  Total:', '$' + stats.total_cost_usd.toFixed(4));
  console.log('  Per request:', '$' + stats.avg_cost_per_request?.toFixed(4));
}
```

### Provider-Specific Analysis

```typescript
// Compare different providers
const openaiStats = collector.aggregate_by_provider('openai');
const anthropicStats = collector.aggregate_by_provider('anthropic');

console.log('OpenAI TTFT p99:', openaiStats.ttft_distribution.p99_ms, 'ms');
console.log('Anthropic TTFT p99:', anthropicStats.ttft_distribution.p99_ms, 'ms');

// Compare different models
const gpt4Stats = collector.aggregate_by_model('gpt-4');
const gpt35Stats = collector.aggregate_by_model('gpt-3.5-turbo');

console.log('GPT-4 avg throughput:', gpt4Stats.throughput.mean_tokens_per_second);
console.log('GPT-3.5 avg throughput:', gpt35Stats.throughput.mean_tokens_per_second);
```

### Integration with OpenAI

```typescript
import OpenAI from 'openai';
import { LatencyCollector } from '@llm-devops/latency-lens';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const collector = new LatencyCollector();

async function chatWithMetrics(prompt: string) {
  const startTime = Date.now();
  let firstTokenTime = 0;
  const interTokenLatencies: number[] = [];
  let lastTokenTime = 0;
  let tokenCount = 0;

  const stream = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: 'user', content: prompt }],
    stream: true,
  });

  let response = '';

  for await (const chunk of stream) {
    const now = Date.now();
    const content = chunk.choices[0]?.delta?.content || '';

    if (content && tokenCount === 0) {
      firstTokenTime = now - startTime;
    } else if (content && tokenCount > 0) {
      interTokenLatencies.push(now - lastTokenTime);
    }

    if (content) {
      tokenCount++;
      lastTokenTime = now;
      response += content;
    }
  }

  const totalTime = Date.now() - startTime;

  // Record metrics
  collector.record({
    provider: 'openai',
    model: 'gpt-4',
    ttft_ms: firstTokenTime,
    total_latency_ms: totalTime,
    inter_token_latencies_ms: interTokenLatencies,
    input_tokens: prompt.length / 4, // Rough estimate
    output_tokens: tokenCount,
    tokens_per_second: (tokenCount / totalTime) * 1000,
    success: true
  });

  return response;
}

// Use it
await chatWithMetrics('Explain quantum computing in simple terms');

// Get statistics
const stats = collector.aggregate();
console.log('TTFT p99:', stats.ttft_distribution.p99_ms, 'ms');
```

### Integration with Anthropic

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { LatencyCollector } from '@llm-devops/latency-lens';

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const collector = new LatencyCollector();

async function claudeWithMetrics(prompt: string) {
  const startTime = Date.now();
  let firstTokenTime = 0;
  const interTokenLatencies: number[] = [];
  let lastTokenTime = 0;
  let tokenCount = 0;

  const stream = await anthropic.messages.create({
    model: 'claude-3-opus-20240229',
    max_tokens: 1024,
    messages: [{ role: 'user', content: prompt }],
    stream: true,
  });

  let response = '';

  for await (const chunk of stream) {
    const now = Date.now();

    if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
      const content = chunk.delta.text;

      if (tokenCount === 0) {
        firstTokenTime = now - startTime;
      } else {
        interTokenLatencies.push(now - lastTokenTime);
      }

      tokenCount++;
      lastTokenTime = now;
      response += content;
    }
  }

  const totalTime = Date.now() - startTime;

  collector.record({
    provider: 'anthropic',
    model: 'claude-3-opus-20240229',
    ttft_ms: firstTokenTime,
    total_latency_ms: totalTime,
    inter_token_latencies_ms: interTokenLatencies,
    input_tokens: prompt.length / 4,
    output_tokens: tokenCount,
    tokens_per_second: (tokenCount / totalTime) * 1000,
    success: true
  });

  return response;
}
```

### Advanced Configuration

```typescript
// Custom configuration for longer timeouts
const collector = LatencyCollector.with_config(
  120,  // Track up to 120 seconds of latency
  4     // Use 4 significant digits for higher precision
);

// Get session ID for tracking
const sessionId = collector.session_id();
console.log('Session:', sessionId);

// Check collection status
console.log('Requests recorded:', collector.len());
console.log('Is empty:', collector.is_empty());

// Clear metrics when starting a new test
collector.clear();
```

### Browser Usage with Webpack/Vite

```typescript
import { LatencyCollector, init_wasm } from '@llm-devops/latency-lens';

// Initialize WASM (optional but recommended)
init_wasm();

// Use normally
const collector = new LatencyCollector();
// ... rest of your code
```

## API Reference

### `LatencyCollector`

Main class for collecting and analyzing metrics.

#### Constructor

```typescript
new LatencyCollector()
```

Creates a new collector with default configuration.

#### Static Methods

```typescript
LatencyCollector.with_config(max_value_seconds: number, significant_digits: number): LatencyCollector
```

Creates a collector with custom configuration:
- `max_value_seconds`: Maximum latency to track (default: 60)
- `significant_digits`: Precision for percentiles (1-5, default: 3)

#### Methods

- `session_id(): string` - Get the session UUID
- `record(metrics: RequestMetrics): void` - Record request metrics
- `len(): number` - Get number of recorded requests
- `is_empty(): boolean` - Check if collector is empty
- `clear(): void` - Clear all metrics
- `aggregate(): AggregatedMetrics` - Get overall statistics
- `aggregate_by_provider(provider: Provider): AggregatedMetrics` - Get provider-specific stats
- `aggregate_by_model(model: string): AggregatedMetrics` - Get model-specific stats

### Types

See [index.d.ts](./index.d.ts) for complete type definitions.

#### `RequestMetrics`

Metrics for a single request:

```typescript
interface RequestMetrics {
  provider: Provider;
  model: string;
  ttft_ms: number;
  total_latency_ms: number;
  inter_token_latencies_ms: number[];
  input_tokens: number;
  output_tokens: number;
  thinking_tokens?: number;
  tokens_per_second: number;
  cost_usd?: number;
  success: boolean;
  error?: string;
}
```

#### `AggregatedMetrics`

Statistical summary:

```typescript
interface AggregatedMetrics {
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  success_rate: number;
  ttft_distribution: LatencyDistribution;
  inter_token_distribution: LatencyDistribution;
  total_latency_distribution: LatencyDistribution;
  throughput: ThroughputStats;
  // ... more fields
}
```

## Performance

The WebAssembly implementation provides:

- **Recording overhead**: ~1-2μs per metric
- **Memory usage**: ~100KB per 10,000 samples
- **Aggregation time**: ~100μs for 10,000 samples
- **Percentile accuracy**: 0.1% (with 3 significant digits)

## Browser Support

Requires WebAssembly support:
- Chrome/Edge 57+
- Firefox 52+
- Safari 11+
- Node.js 14+

## License

Apache-2.0

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/llm-devops/llm-latency-lens/blob/main/CONTRIBUTING.md).

## Links

- [GitHub Repository](https://github.com/llm-devops/llm-latency-lens)
- [Issue Tracker](https://github.com/llm-devops/llm-latency-lens/issues)
- [CLI Tool](https://github.com/llm-devops/llm-latency-lens#readme)

## Support

For questions and support:
- GitHub Issues: https://github.com/llm-devops/llm-latency-lens/issues
- Documentation: https://github.com/llm-devops/llm-latency-lens/tree/main/docs
