/**
 * Basic usage example for @llm-devops/latency-lens
 *
 * This example demonstrates how to collect and analyze LLM latency metrics
 * in a TypeScript application.
 */

import { LatencyCollector, RequestMetrics } from '@llm-devops/latency-lens';

// Create a collector instance
const collector = new LatencyCollector();

console.log('Session ID:', collector.session_id());
console.log();

// Simulate recording metrics from multiple LLM requests
const simulateRequests = () => {
  // OpenAI GPT-4 request
  collector.record({
    provider: 'openai',
    model: 'gpt-4',
    ttft_ms: 150,
    total_latency_ms: 2000,
    inter_token_latencies_ms: [10, 15, 12, 11, 13, 14, 12, 11, 10, 15],
    input_tokens: 100,
    output_tokens: 50,
    tokens_per_second: 25.0,
    cost_usd: 0.05,
    success: true
  });

  // Anthropic Claude request
  collector.record({
    provider: 'anthropic',
    model: 'claude-3-opus-20240229',
    ttft_ms: 120,
    total_latency_ms: 1800,
    inter_token_latencies_ms: [8, 12, 10, 9, 11, 10, 8, 9, 12, 11],
    input_tokens: 150,
    output_tokens: 60,
    tokens_per_second: 33.3,
    cost_usd: 0.045,
    success: true
  });

  // Google Gemini request
  collector.record({
    provider: 'google',
    model: 'gemini-pro',
    ttft_ms: 180,
    total_latency_ms: 2200,
    inter_token_latencies_ms: [15, 18, 16, 17, 19, 15, 16, 18, 17, 16],
    input_tokens: 120,
    output_tokens: 55,
    tokens_per_second: 25.0,
    cost_usd: 0.03,
    success: true
  });

  // Failed request example
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

  // More successful requests
  for (let i = 0; i < 10; i++) {
    collector.record({
      provider: 'openai',
      model: 'gpt-3.5-turbo',
      ttft_ms: 80 + Math.random() * 40,
      total_latency_ms: 1200 + Math.random() * 400,
      inter_token_latencies_ms: Array(20).fill(0).map(() => 8 + Math.random() * 8),
      input_tokens: 80 + Math.floor(Math.random() * 40),
      output_tokens: 40 + Math.floor(Math.random() * 20),
      tokens_per_second: 35 + Math.random() * 10,
      cost_usd: 0.01 + Math.random() * 0.01,
      success: true
    });
  }
};

// Record simulated requests
simulateRequests();

console.log(`Recorded ${collector.len()} requests\n`);

// Get overall statistics
const stats = collector.aggregate();

console.log('=== Overall Statistics ===\n');
console.log(`Total requests:      ${stats.total_requests}`);
console.log(`Successful:          ${stats.successful_requests}`);
console.log(`Failed:              ${stats.failed_requests}`);
console.log(`Success rate:        ${stats.success_rate.toFixed(2)}%`);
console.log();

console.log('=== Time to First Token (TTFT) ===\n');
console.log(`Min:      ${stats.ttft_distribution.min_ms.toFixed(2)} ms`);
console.log(`p50:      ${stats.ttft_distribution.p50_ms.toFixed(2)} ms`);
console.log(`p90:      ${stats.ttft_distribution.p90_ms.toFixed(2)} ms`);
console.log(`p95:      ${stats.ttft_distribution.p95_ms.toFixed(2)} ms`);
console.log(`p99:      ${stats.ttft_distribution.p99_ms.toFixed(2)} ms`);
console.log(`Max:      ${stats.ttft_distribution.max_ms.toFixed(2)} ms`);
console.log(`Mean:     ${stats.ttft_distribution.mean_ms.toFixed(2)} ms`);
console.log(`Std Dev:  ${stats.ttft_distribution.std_dev_ms.toFixed(2)} ms`);
console.log();

console.log('=== Total Request Latency ===\n');
console.log(`Min:      ${stats.total_latency_distribution.min_ms.toFixed(2)} ms`);
console.log(`p50:      ${stats.total_latency_distribution.p50_ms.toFixed(2)} ms`);
console.log(`p90:      ${stats.total_latency_distribution.p90_ms.toFixed(2)} ms`);
console.log(`p95:      ${stats.total_latency_distribution.p95_ms.toFixed(2)} ms`);
console.log(`p99:      ${stats.total_latency_distribution.p99_ms.toFixed(2)} ms`);
console.log(`Max:      ${stats.total_latency_distribution.max_ms.toFixed(2)} ms`);
console.log();

console.log('=== Throughput ===\n');
console.log(`Mean:     ${stats.throughput.mean_tokens_per_second.toFixed(2)} tokens/sec`);
console.log(`p50:      ${stats.throughput.p50_tokens_per_second.toFixed(2)} tokens/sec`);
console.log(`p95:      ${stats.throughput.p95_tokens_per_second.toFixed(2)} tokens/sec`);
console.log(`p99:      ${stats.throughput.p99_tokens_per_second.toFixed(2)} tokens/sec`);
console.log();

console.log('=== Token Counts ===\n');
console.log(`Total input tokens:  ${stats.total_input_tokens}`);
console.log(`Total output tokens: ${stats.total_output_tokens}`);
console.log();

console.log('=== Cost ===\n');
if (stats.total_cost_usd) {
  console.log(`Total cost:          $${stats.total_cost_usd.toFixed(4)}`);
  console.log(`Avg per request:     $${stats.avg_cost_per_request?.toFixed(4)}`);
} else {
  console.log('No cost data available');
}
console.log();

console.log('=== Provider Breakdown ===\n');
for (const [provider, count] of stats.provider_breakdown) {
  console.log(`${provider.padEnd(15)} ${count} requests`);
}
console.log();

console.log('=== Model Breakdown ===\n');
for (const [model, count] of stats.model_breakdown) {
  console.log(`${model.padEnd(30)} ${count} requests`);
}
console.log();

// Provider-specific analysis
console.log('=== Provider Comparison ===\n');
const openaiStats = collector.aggregate_by_provider('openai');
const anthropicStats = collector.aggregate_by_provider('anthropic');

console.log('OpenAI:');
console.log(`  TTFT p99:     ${openaiStats.ttft_distribution.p99_ms.toFixed(2)} ms`);
console.log(`  Throughput:   ${openaiStats.throughput.mean_tokens_per_second.toFixed(2)} tokens/sec`);
console.log(`  Success rate: ${openaiStats.success_rate.toFixed(2)}%`);
console.log();

console.log('Anthropic:');
console.log(`  TTFT p99:     ${anthropicStats.ttft_distribution.p99_ms.toFixed(2)} ms`);
console.log(`  Throughput:   ${anthropicStats.throughput.mean_tokens_per_second.toFixed(2)} tokens/sec`);
console.log(`  Success rate: ${anthropicStats.success_rate.toFixed(2)}%`);
console.log();

// Model comparison
console.log('=== Model Comparison ===\n');
const gpt4Stats = collector.aggregate_by_model('gpt-4');
const gpt35Stats = collector.aggregate_by_model('gpt-3.5-turbo');

console.log('GPT-4:');
console.log(`  Requests:     ${gpt4Stats.total_requests}`);
console.log(`  TTFT p50:     ${gpt4Stats.ttft_distribution.p50_ms.toFixed(2)} ms`);
console.log(`  Throughput:   ${gpt4Stats.throughput.mean_tokens_per_second.toFixed(2)} tokens/sec`);
console.log();

console.log('GPT-3.5-Turbo:');
console.log(`  Requests:     ${gpt35Stats.total_requests}`);
console.log(`  TTFT p50:     ${gpt35Stats.ttft_distribution.p50_ms.toFixed(2)} ms`);
console.log(`  Throughput:   ${gpt35Stats.throughput.mean_tokens_per_second.toFixed(2)} tokens/sec`);
