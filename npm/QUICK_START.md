# Quick Start Guide

Get started with `@llm-devops/latency-lens` in 5 minutes.

## Installation

```bash
npm install @llm-devops/latency-lens
```

## Basic Usage

```typescript
import { LatencyCollector } from '@llm-devops/latency-lens';

// 1. Create a collector
const collector = new LatencyCollector();

// 2. Record metrics from your LLM requests
collector.record({
  provider: 'openai',
  model: 'gpt-4',
  ttft_ms: 150,                              // Time to first token
  total_latency_ms: 2000,                    // Total request time
  inter_token_latencies_ms: [10, 15, 12, 11], // Between each token
  input_tokens: 100,
  output_tokens: 50,
  tokens_per_second: 25.0,
  cost_usd: 0.05,
  success: true
});

// 3. Get statistics
const stats = collector.aggregate();

console.log('TTFT p99:', stats.ttft_distribution.p99_ms, 'ms');
console.log('Success rate:', stats.success_rate, '%');
```

## Real-World Example with OpenAI

```typescript
import OpenAI from 'openai';
import { LatencyCollector } from '@llm-devops/latency-lens';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const collector = new LatencyCollector();

async function chat(prompt: string) {
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

  for await (const chunk of stream) {
    const now = Date.now();
    const content = chunk.choices[0]?.delta?.content || '';

    if (content) {
      if (tokenCount === 0) {
        firstTokenTime = now - startTime;
      } else {
        interTokenLatencies.push(now - lastTokenTime);
      }
      tokenCount++;
      lastTokenTime = now;
    }
  }

  collector.record({
    provider: 'openai',
    model: 'gpt-4',
    ttft_ms: firstTokenTime,
    total_latency_ms: Date.now() - startTime,
    inter_token_latencies_ms: interTokenLatencies,
    input_tokens: prompt.length / 4,  // Rough estimate
    output_tokens: tokenCount,
    tokens_per_second: (tokenCount / (Date.now() - startTime)) * 1000,
    success: true
  });
}

// Use it
await chat('Explain quantum computing');

// Analyze
const stats = collector.aggregate();
console.log(stats.ttft_distribution);
```

## What You Get

```typescript
const stats = collector.aggregate();

// Overall metrics
stats.total_requests        // Total number of requests
stats.success_rate          // Percentage (0-100)
stats.total_cost_usd       // Total cost (if tracked)

// TTFT (Time to First Token) statistics
stats.ttft_distribution.p50_ms    // Median
stats.ttft_distribution.p95_ms    // 95th percentile
stats.ttft_distribution.p99_ms    // 99th percentile
stats.ttft_distribution.mean_ms   // Average

// Throughput
stats.throughput.mean_tokens_per_second
stats.throughput.p50_tokens_per_second

// Per-provider breakdown
stats.provider_breakdown    // [['openai', 50], ['anthropic', 30]]
stats.model_breakdown      // [['gpt-4', 40], ['gpt-3.5', 10]]
```

## Advanced: Compare Providers

```typescript
// Record metrics from different providers
collector.record({ provider: 'openai', ... });
collector.record({ provider: 'anthropic', ... });
collector.record({ provider: 'google', ... });

// Compare them
const openaiStats = collector.aggregate_by_provider('openai');
const anthropicStats = collector.aggregate_by_provider('anthropic');

console.log('OpenAI TTFT p99:', openaiStats.ttft_distribution.p99_ms);
console.log('Anthropic TTFT p99:', anthropicStats.ttft_distribution.p99_ms);
```

## Configuration

```typescript
// Track longer requests with higher precision
const collector = LatencyCollector.with_config(
  120,  // Track up to 120 seconds
  4     // Use 4 significant digits (higher precision)
);
```

## Common Use Cases

### 1. A/B Testing Models

```typescript
const collector = new LatencyCollector();

// Test GPT-4
for (let i = 0; i < 100; i++) {
  await testPrompt('gpt-4', testPrompts[i]);
}

// Test GPT-3.5
for (let i = 0; i < 100; i++) {
  await testPrompt('gpt-3.5-turbo', testPrompts[i]);
}

// Compare
const gpt4 = collector.aggregate_by_model('gpt-4');
const gpt35 = collector.aggregate_by_model('gpt-3.5-turbo');

console.log('GPT-4 p99 TTFT:', gpt4.ttft_distribution.p99_ms);
console.log('GPT-3.5 p99 TTFT:', gpt35.ttft_distribution.p99_ms);
console.log('Cost difference:', gpt4.total_cost_usd - gpt35.total_cost_usd);
```

### 2. Production Monitoring

```typescript
// Create a global collector
export const llmMetrics = new LatencyCollector();

// In your LLM service
async function generateText(prompt: string) {
  const result = await callLLM(prompt);

  llmMetrics.record({
    provider: 'openai',
    model: result.model,
    ttft_ms: result.ttft,
    total_latency_ms: result.totalTime,
    inter_token_latencies_ms: result.tokenLatencies,
    input_tokens: result.usage.input,
    output_tokens: result.usage.output,
    tokens_per_second: result.throughput,
    cost_usd: result.cost,
    success: true
  });

  return result.text;
}

// Periodic reporting
setInterval(() => {
  const stats = llmMetrics.aggregate();
  console.log('Last hour TTFT p95:', stats.ttft_distribution.p95_ms);

  // Send to your monitoring system
  sendToDatadog({
    metric: 'llm.ttft.p95',
    value: stats.ttft_distribution.p95_ms
  });

  llmMetrics.clear();  // Start fresh for next hour
}, 60 * 60 * 1000);  // Every hour
```

### 3. Performance Regression Testing

```typescript
// Load baseline metrics
const baseline = loadBaselineMetrics();

// Run current version
const collector = new LatencyCollector();
await runTestSuite(collector);

const current = collector.aggregate();

// Check for regressions
if (current.ttft_distribution.p99_ms > baseline.ttft_distribution.p99_ms * 1.1) {
  throw new Error('TTFT p99 regressed by more than 10%!');
}

if (current.success_rate < baseline.success_rate - 5) {
  throw new Error('Success rate dropped by more than 5%!');
}

console.log('All performance checks passed!');
```

## TypeScript Support

Full type safety included:

```typescript
import {
  LatencyCollector,
  RequestMetrics,
  AggregatedMetrics,
  Provider
} from '@llm-devops/latency-lens';

// All types are inferred
const collector = new LatencyCollector();
const stats: AggregatedMetrics = collector.aggregate();
const p99: number = stats.ttft_distribution.p99_ms;
```

## Next Steps

- Read the [full README](./README.md) for detailed examples
- Check out [examples/basic.ts](./examples/basic.ts)
- See integration examples for [OpenAI](./README.md#integration-with-openai) and [Anthropic](./README.md#integration-with-anthropic)

## Need Help?

- [GitHub Issues](https://github.com/llm-devops/llm-latency-lens/issues)
- [Documentation](https://github.com/llm-devops/llm-latency-lens/tree/main/docs)
