/**
 * TypeScript definitions for @llm-devops/latency-lens
 *
 * High-precision LLM latency profiler for JavaScript/TypeScript applications.
 */

/**
 * LLM provider types
 */
export type Provider = 'openai' | 'anthropic' | 'google' | 'aws-bedrock' | 'azure-openai' | 'generic';

/**
 * Request metrics to be recorded
 */
export interface RequestMetrics {
  /** Provider name (e.g., 'openai', 'anthropic') */
  provider: Provider;

  /** Model name/ID (e.g., 'gpt-4', 'claude-3-opus') */
  model: string;

  /** Time to first token in milliseconds */
  ttft_ms: number;

  /** Total request latency in milliseconds */
  total_latency_ms: number;

  /** Inter-token latencies in milliseconds */
  inter_token_latencies_ms: number[];

  /** Number of input tokens */
  input_tokens: number;

  /** Number of output tokens generated */
  output_tokens: number;

  /** Number of thinking tokens (optional, for extended thinking models) */
  thinking_tokens?: number;

  /** Token generation throughput (tokens per second) */
  tokens_per_second: number;

  /** Estimated cost in USD (optional) */
  cost_usd?: number;

  /** Whether the request completed successfully */
  success: boolean;

  /** Error message if request failed (optional) */
  error?: string;
}

/**
 * Latency distribution statistics with percentiles
 */
export interface LatencyDistribution {
  /** Minimum latency in milliseconds */
  min_ms: number;

  /** Maximum latency in milliseconds */
  max_ms: number;

  /** Mean (average) latency in milliseconds */
  mean_ms: number;

  /** Standard deviation in milliseconds */
  std_dev_ms: number;

  /** 50th percentile (median) in milliseconds */
  p50_ms: number;

  /** 90th percentile in milliseconds */
  p90_ms: number;

  /** 95th percentile in milliseconds */
  p95_ms: number;

  /** 99th percentile in milliseconds */
  p99_ms: number;

  /** 99.9th percentile in milliseconds */
  p99_9_ms: number;

  /** Number of samples in this distribution */
  sample_count: number;
}

/**
 * Token throughput statistics
 */
export interface ThroughputStats {
  /** Mean tokens per second */
  mean_tokens_per_second: number;

  /** Minimum tokens per second observed */
  min_tokens_per_second: number;

  /** Maximum tokens per second observed */
  max_tokens_per_second: number;

  /** Standard deviation of tokens per second */
  std_dev_tokens_per_second: number;

  /** 50th percentile tokens per second */
  p50_tokens_per_second: number;

  /** 95th percentile tokens per second */
  p95_tokens_per_second: number;

  /** 99th percentile tokens per second */
  p99_tokens_per_second: number;
}

/**
 * Aggregated metrics across multiple requests
 */
export interface AggregatedMetrics {
  /** Session identifier */
  session_id: string;

  /** Start time (ISO 8601 format) */
  start_time: string;

  /** End time (ISO 8601 format) */
  end_time: string;

  /** Total number of requests */
  total_requests: number;

  /** Number of successful requests */
  successful_requests: number;

  /** Number of failed requests */
  failed_requests: number;

  /** Success rate as percentage (0-100) */
  success_rate: number;

  /** TTFT (Time to First Token) distribution */
  ttft_distribution: LatencyDistribution;

  /** Inter-token latency distribution */
  inter_token_distribution: LatencyDistribution;

  /** Total request latency distribution */
  total_latency_distribution: LatencyDistribution;

  /** Token throughput statistics */
  throughput: ThroughputStats;

  /** Total input tokens processed */
  total_input_tokens: number;

  /** Total output tokens generated */
  total_output_tokens: number;

  /** Total thinking tokens (if applicable) */
  total_thinking_tokens?: number;

  /** Total cost in USD (if available) */
  total_cost_usd?: number;

  /** Average cost per request in USD (if available) */
  avg_cost_per_request?: number;

  /** Provider breakdown: [provider, count] pairs */
  provider_breakdown: Array<[string, number]>;

  /** Model breakdown: [model, count] pairs */
  model_breakdown: Array<[string, number]>;
}

/**
 * Error type for WASM operations
 */
export class WasmError extends Error {
  constructor(message: string);
  readonly message: string;
}

/**
 * High-precision metrics collector for LLM requests
 *
 * This is the main interface for collecting and aggregating latency metrics
 * in JavaScript/TypeScript applications.
 *
 * @example
 * ```typescript
 * import { LatencyCollector } from '@llm-devops/latency-lens';
 *
 * // Create a collector
 * const collector = new LatencyCollector();
 *
 * // Record metrics from your LLM requests
 * collector.record({
 *   provider: 'openai',
 *   model: 'gpt-4',
 *   ttft_ms: 150,
 *   total_latency_ms: 2000,
 *   inter_token_latencies_ms: [10, 15, 12, 11],
 *   input_tokens: 100,
 *   output_tokens: 50,
 *   tokens_per_second: 25.0,
 *   cost_usd: 0.05,
 *   success: true
 * });
 *
 * // Get aggregated statistics
 * const stats = collector.aggregate();
 * console.log('TTFT p99:', stats.ttft_distribution.p99_ms, 'ms');
 * console.log('Success rate:', stats.success_rate, '%');
 * ```
 */
export class LatencyCollector {
  /**
   * Create a new metrics collector with default configuration
   *
   * Default configuration:
   * - Maximum latency: 60 seconds
   * - Precision: 3 significant digits
   */
  constructor();

  /**
   * Create a new metrics collector with custom configuration
   *
   * @param max_value_seconds - Maximum latency value to track (default: 60)
   * @param significant_digits - Precision for percentile calculations (1-5, default: 3)
   *
   * @example
   * ```typescript
   * // Track up to 2 minutes of latency with high precision
   * const collector = LatencyCollector.with_config(120, 4);
   * ```
   */
  static with_config(max_value_seconds: number, significant_digits: number): LatencyCollector;

  /**
   * Get the session ID for this collector
   *
   * @returns UUID string representing the session
   */
  session_id(): string;

  /**
   * Record metrics from a completed request
   *
   * @param metrics - Request metrics to record
   * @throws {WasmError} If metrics format is invalid
   *
   * @example
   * ```typescript
   * collector.record({
   *   provider: 'anthropic',
   *   model: 'claude-3-opus',
   *   ttft_ms: 120,
   *   total_latency_ms: 1800,
   *   inter_token_latencies_ms: [8, 12, 10],
   *   input_tokens: 150,
   *   output_tokens: 75,
   *   tokens_per_second: 41.6,
   *   success: true
   * });
   * ```
   */
  record(metrics: RequestMetrics): void;

  /**
   * Get the number of metrics recorded
   *
   * @returns Number of requests recorded
   */
  len(): number;

  /**
   * Check if the collector is empty
   *
   * @returns true if no metrics have been recorded
   */
  is_empty(): boolean;

  /**
   * Clear all collected metrics
   *
   * Resets the collector to its initial state while keeping the same session ID.
   */
  clear(): void;

  /**
   * Aggregate all metrics and return summary statistics
   *
   * Calculates percentile distributions (p50, p90, p95, p99, p99.9) for:
   * - Time to First Token (TTFT)
   * - Inter-token latency
   * - Total request latency
   * - Token throughput
   *
   * @returns Aggregated metrics with statistical analysis
   *
   * @example
   * ```typescript
   * const stats = collector.aggregate();
   * console.log(`TTFT p50: ${stats.ttft_distribution.p50_ms}ms`);
   * console.log(`TTFT p99: ${stats.ttft_distribution.p99_ms}ms`);
   * console.log(`Success rate: ${stats.success_rate}%`);
   * console.log(`Avg throughput: ${stats.throughput.mean_tokens_per_second} tokens/sec`);
   * ```
   */
  aggregate(): AggregatedMetrics;

  /**
   * Aggregate metrics for a specific provider
   *
   * @param provider - Provider name to filter by
   * @returns Aggregated metrics for the specified provider
   *
   * @example
   * ```typescript
   * const openaiStats = collector.aggregate_by_provider('openai');
   * const anthropicStats = collector.aggregate_by_provider('anthropic');
   * ```
   */
  aggregate_by_provider(provider: Provider): AggregatedMetrics;

  /**
   * Aggregate metrics for a specific model
   *
   * @param model - Model name to filter by
   * @returns Aggregated metrics for the specified model
   *
   * @example
   * ```typescript
   * const gpt4Stats = collector.aggregate_by_model('gpt-4');
   * const gpt35Stats = collector.aggregate_by_model('gpt-3.5-turbo');
   * ```
   */
  aggregate_by_model(model: string): AggregatedMetrics;
}

/**
 * Initialize the WASM module
 *
 * Call this once when your application starts to set up better error messages.
 * Optional but recommended for improved debugging.
 *
 * @example
 * ```typescript
 * import { init_wasm } from '@llm-devops/latency-lens';
 *
 * init_wasm();
 * ```
 */
export function init_wasm(): void;

/**
 * Get the library version
 *
 * @returns Version string (e.g., "0.1.0")
 */
export function version(): string;
