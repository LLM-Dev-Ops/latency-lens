//! CLI argument parsing and command definitions

use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub mod commands;

/// LLM Latency Lens - Enterprise-grade CLI for LLM performance measurement
#[derive(Parser, Debug)]
#[command(
    name = "llm-latency-lens",
    version,
    author,
    about,
    long_about = None,
    arg_required_else_help = true,
    propagate_version = true,
)]
pub struct Cli {
    /// Subcommand to execute
    #[command(subcommand)]
    pub command: Commands,

    /// Output in JSON format
    #[arg(long, global = true, help = "Output results in JSON format")]
    pub json: bool,

    /// Quiet mode (suppress non-essential output)
    #[arg(short, long, global = true, help = "Suppress non-essential output")]
    pub quiet: bool,

    /// Verbose mode (can be repeated for more verbosity)
    #[arg(short, long, global = true, action = clap::ArgAction::Count, help = "Increase verbosity (-v, -vv, -vvv)")]
    pub verbose: u8,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Profile a single LLM request with detailed timing metrics
    #[command(visible_alias = "prof")]
    Profile(ProfileArgs),

    /// Run benchmark with multiple concurrent requests
    #[command(visible_alias = "bench")]
    Benchmark(BenchmarkArgs),

    /// Compare performance across multiple providers or models
    #[command(visible_alias = "comp")]
    Compare(CompareArgs),

    /// Validate API credentials and connectivity
    #[command(visible_alias = "val")]
    Validate(ValidateArgs),

    /// Export metrics to different formats
    #[command(visible_alias = "exp")]
    Export(ExportArgs),

    /// Run all benchmarks for configured targets (canonical interface)
    #[command(visible_alias = "r")]
    Run(RunArgs),

    /// Analyze latency data using the Latency Analysis Agent
    #[command(visible_alias = "a")]
    Analyze(AnalyzeArgs),

    /// Inspect previous analysis results and DecisionEvents
    #[command(visible_alias = "i")]
    Inspect(InspectArgs),

    /// Replay a previous analysis with different configuration
    #[command(visible_alias = "rep")]
    Replay(ReplayArgs),

    /// Cold start measurement and characterization (Cold Start Mitigation Agent)
    #[command(visible_alias = "cs")]
    ColdStart(ColdStartArgs),

    /// Start HTTP server for Cloud Run deployment
    #[command(visible_alias = "srv")]
    Serve(ServeArgs),
}

/// Arguments for the serve command (HTTP server mode for Cloud Run)
#[derive(Parser, Debug)]
pub struct ServeArgs {
    /// Port to listen on
    #[arg(short, long, default_value = "8080", env = "PORT")]
    pub port: u16,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0", env = "HOST")]
    pub host: String,

    /// Enable request logging
    #[arg(long, default_value = "true")]
    pub access_log: bool,

    /// Request timeout in seconds
    #[arg(long, default_value = "300", env = "REQUEST_TIMEOUT_SECS")]
    pub timeout: u64,

    /// RuVector service endpoint
    #[arg(long, env = "RUVECTOR_SERVICE_URL")]
    pub ruvector_url: Option<String>,

    /// RuVector API key
    #[arg(long, env = "RUVECTOR_API_KEY")]
    pub ruvector_key: Option<String>,

    /// Telemetry endpoint
    #[arg(long, env = "TELEMETRY_ENDPOINT")]
    pub telemetry_endpoint: Option<String>,
}

/// Arguments for the cold-start command (Cold Start Mitigation Agent)
#[derive(Parser, Debug)]
pub struct ColdStartArgs {
    /// Cold start subcommand
    #[command(subcommand)]
    pub subcommand: ColdStartSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum ColdStartSubcommand {
    /// Profile cold start behavior by sending multiple requests
    Profile(ColdStartProfileArgs),

    /// Inspect cold start data from file or previous analysis
    Inspect(ColdStartInspectArgs),

    /// Replay a previous cold start measurement
    Replay(ColdStartReplayArgs),
}

/// Arguments for the cold-start profile subcommand
#[derive(Parser, Debug)]
pub struct ColdStartProfileArgs {
    /// Provider to profile (openai, anthropic, google)
    #[arg(short, long, env = "LLM_PROVIDER")]
    pub provider: String,

    /// Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)
    #[arg(short, long, env = "LLM_MODEL")]
    pub model: String,

    /// Prompt or input text
    #[arg(short = 'P', long)]
    pub prompt: Option<String>,

    /// Path to file containing prompt
    #[arg(short = 'f', long, conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    /// API key
    #[arg(short = 'k', long, env = "LLM_API_KEY")]
    pub api_key: Option<String>,

    /// Number of requests to send for cold start detection
    #[arg(short, long, default_value = "10")]
    pub requests: u32,

    /// Delay between requests in milliseconds (to simulate idle time)
    #[arg(long, default_value = "100")]
    pub delay_ms: u64,

    /// Cold start threshold multiplier (cold if TTFT > baseline * threshold)
    #[arg(long, default_value = "2.0")]
    pub threshold: f64,

    /// Detection algorithm (threshold, zscore, moving_average, inter_arrival)
    #[arg(long, default_value = "threshold")]
    pub algorithm: String,

    /// Maximum tokens to generate per request
    #[arg(long, default_value = "100")]
    pub max_tokens: u32,

    /// Request timeout in seconds
    #[arg(long, default_value = "60")]
    pub timeout: u64,

    /// Configuration file path
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Output file for results (JSON format)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Persist results to ruvector-service
    #[arg(long)]
    pub persist: bool,

    /// RuVector service endpoint
    #[arg(long, env = "RUVECTOR_ENDPOINT")]
    pub ruvector_endpoint: Option<String>,

    /// Execution ID for agentics execution graph
    #[arg(long)]
    pub execution_id: Option<String>,

    /// Parent span ID from the Core orchestrator
    #[arg(long)]
    pub parent_span_id: Option<String>,
}

/// Arguments for the cold-start inspect subcommand
#[derive(Parser, Debug)]
pub struct ColdStartInspectArgs {
    /// Session ID to inspect
    #[arg(long)]
    pub session_id: Option<String>,

    /// Input file containing metrics (JSON format)
    #[arg(short, long)]
    pub input: Option<PathBuf>,

    /// Cold start threshold multiplier
    #[arg(long, default_value = "2.0")]
    pub threshold: f64,

    /// Detection algorithm
    #[arg(long, default_value = "threshold")]
    pub algorithm: String,

    /// Output format (table, json)
    #[arg(long, default_value = "table")]
    pub format: String,

    /// Output file
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Arguments for the cold-start replay subcommand
#[derive(Parser, Debug)]
pub struct ColdStartReplayArgs {
    /// Trace ID to replay
    #[arg(long, required = true)]
    pub trace_id: String,

    /// RuVector service endpoint
    #[arg(long, env = "RUVECTOR_ENDPOINT")]
    pub ruvector_endpoint: Option<String>,

    /// Output format (table, json)
    #[arg(long, default_value = "table")]
    pub format: String,
}

/// Arguments for the profile command
#[derive(Parser, Debug)]
pub struct ProfileArgs {
    /// Provider to use (openai, anthropic, google)
    #[arg(short, long, env = "LLM_PROVIDER")]
    pub provider: String,

    /// Model name (e.g., gpt-4o, claude-3-5-sonnet-20241022)
    #[arg(short, long, env = "LLM_MODEL")]
    pub model: String,

    /// Prompt or input text
    #[arg(short = 'P', long)]
    pub prompt: Option<String>,

    /// Path to file containing prompt
    #[arg(short = 'f', long, conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    /// API key (can also use environment variables)
    #[arg(short = 'k', long, env = "LLM_API_KEY")]
    pub api_key: Option<String>,

    /// API endpoint URL (optional, uses provider default)
    #[arg(short, long)]
    pub endpoint: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "1024")]
    pub max_tokens: u32,

    /// Temperature (0.0 to 2.0)
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Top-p sampling
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Request timeout in seconds
    #[arg(long, default_value = "120")]
    pub timeout: u64,

    /// Configuration file path
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Output file for metrics (JSON format)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Show streaming output
    #[arg(long)]
    pub stream: bool,
}

/// Arguments for the benchmark command
#[derive(Parser, Debug)]
pub struct BenchmarkArgs {
    /// Provider to use (openai, anthropic, google)
    #[arg(short, long, env = "LLM_PROVIDER")]
    pub provider: String,

    /// Model name
    #[arg(short, long, env = "LLM_MODEL")]
    pub model: String,

    /// Prompt or input text
    #[arg(short = 'P', long)]
    pub prompt: Option<String>,

    /// Path to file containing prompt
    #[arg(short = 'f', long, conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    /// API key
    #[arg(short = 'k', long, env = "LLM_API_KEY")]
    pub api_key: Option<String>,

    /// API endpoint URL
    #[arg(short, long)]
    pub endpoint: Option<String>,

    /// Number of requests to run
    #[arg(short, long, default_value = "10")]
    pub requests: u32,

    /// Number of concurrent requests
    #[arg(short, long, default_value = "1")]
    pub concurrency: u32,

    /// Rate limit (requests per second, 0 = unlimited)
    #[arg(long, default_value = "0")]
    pub rate_limit: u32,

    /// Maximum tokens to generate per request
    #[arg(long, default_value = "1024")]
    pub max_tokens: u32,

    /// Temperature
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Top-p sampling
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Request timeout in seconds
    #[arg(long, default_value = "120")]
    pub timeout: u64,

    /// Configuration file path
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Output file for results
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Warmup requests (not counted in results)
    #[arg(long, default_value = "0")]
    pub warmup: u32,

    /// Show live progress
    #[arg(long, default_value = "true")]
    pub progress: bool,
}

/// Arguments for the compare command
#[derive(Parser, Debug)]
pub struct CompareArgs {
    /// Configurations to compare (provider:model format)
    #[arg(required = true, value_name = "PROVIDER:MODEL")]
    pub targets: Vec<String>,

    /// Prompt or input text
    #[arg(short = 'P', long)]
    pub prompt: Option<String>,

    /// Path to file containing prompt
    #[arg(short = 'f', long, conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    /// Number of requests per target
    #[arg(short, long, default_value = "5")]
    pub requests: u32,

    /// Maximum tokens to generate per request
    #[arg(long, default_value = "1024")]
    pub max_tokens: u32,

    /// Temperature
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Top-p sampling
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Request timeout in seconds
    #[arg(long, default_value = "120")]
    pub timeout: u64,

    /// Configuration file path
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Output file for comparison results
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Metrics to compare (ttft, total, throughput, cost)
    #[arg(long, value_delimiter = ',', default_values = ["ttft", "total", "throughput"])]
    pub metrics: Vec<String>,
}

/// Arguments for the validate command
#[derive(Parser, Debug)]
pub struct ValidateArgs {
    /// Provider to validate (if not specified, validates all configured)
    #[arg(short, long)]
    pub provider: Option<String>,

    /// API key to validate
    #[arg(short = 'k', long, env = "LLM_API_KEY")]
    pub api_key: Option<String>,

    /// API endpoint URL
    #[arg(short, long)]
    pub endpoint: Option<String>,

    /// Configuration file path
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Test with a simple request
    #[arg(long)]
    pub test_request: bool,
}

/// Arguments for the export command
#[derive(Parser, Debug)]
pub struct ExportArgs {
    /// Input file containing metrics (JSON format)
    #[arg(short, long, required = true)]
    pub input: PathBuf,

    /// Output format (json, csv, prometheus, console)
    #[arg(short, long, default_value = "json")]
    pub format: String,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Pretty print JSON output
    #[arg(long, default_value = "true")]
    pub pretty: bool,
}

/// Arguments for the run command (canonical benchmark interface)
#[derive(Parser, Debug)]
pub struct RunArgs {
    /// Specific targets to benchmark (provider:model format)
    /// If not specified, benchmarks all configured targets
    #[arg(value_name = "PROVIDER:MODEL")]
    pub targets: Vec<String>,

    /// Prompt or input text for benchmarks
    #[arg(short = 'P', long)]
    pub prompt: Option<String>,

    /// Path to file containing prompt
    #[arg(short = 'f', long, conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    /// Number of requests per target
    #[arg(short, long, default_value = "10")]
    pub requests: u32,

    /// Number of concurrent requests
    #[arg(short, long, default_value = "1")]
    pub concurrency: u32,

    /// Rate limit (requests per second, 0 = unlimited)
    #[arg(long, default_value = "0")]
    pub rate_limit: u32,

    /// Maximum tokens to generate per request
    #[arg(long, default_value = "1024")]
    pub max_tokens: u32,

    /// Temperature
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Request timeout in seconds
    #[arg(long, default_value = "120")]
    pub timeout: u64,

    /// Configuration file path
    #[arg(short = 'C', long)]
    pub config: Option<PathBuf>,

    /// Warmup requests (not counted in results)
    #[arg(long, default_value = "0")]
    pub warmup: u32,

    /// Show live progress
    #[arg(long, default_value = "true")]
    pub progress: bool,

    /// Output directory for results (default: benchmarks/output)
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Generate summary only (don't run benchmarks)
    #[arg(long)]
    pub summary_only: bool,

    /// Clean up old results (keep N per target)
    #[arg(long)]
    pub cleanup: Option<usize>,
}

/// Arguments for the analyze command (Latency Analysis Agent)
#[derive(Parser, Debug)]
pub struct AnalyzeArgs {
    /// Input file containing RequestMetrics (JSON format)
    #[arg(short, long, required = true)]
    pub input: PathBuf,

    /// Analysis configuration file (JSON format)
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Output file for analysis results
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Minimum number of samples required
    #[arg(long)]
    pub min_samples: Option<u64>,

    /// Number of warmup requests to exclude
    #[arg(long)]
    pub warmup: Option<u32>,

    /// Disable outlier removal
    #[arg(long)]
    pub no_outlier_removal: bool,

    /// Outlier threshold (sigma)
    #[arg(long)]
    pub outlier_sigma: Option<f64>,

    /// Filter by provider (openai, anthropic, google, etc.)
    #[arg(long)]
    pub provider_filter: Option<String>,

    /// Filter by model name
    #[arg(long)]
    pub model_filter: Option<String>,

    /// RuVector service endpoint
    #[arg(long, env = "RUVECTOR_ENDPOINT")]
    pub ruvector_endpoint: Option<String>,

    /// RuVector API key
    #[arg(long, env = "RUVECTOR_API_KEY")]
    pub ruvector_api_key: Option<String>,

    /// Execution reference for correlation
    #[arg(long)]
    pub execution_ref: Option<String>,

    /// Execution ID for agentics execution graph
    #[arg(long)]
    pub execution_id: Option<String>,

    /// Parent span ID from the Core orchestrator
    #[arg(long)]
    pub parent_span_id: Option<String>,
}

/// Arguments for the inspect command
#[derive(Parser, Debug)]
pub struct InspectArgs {
    /// Event ID to inspect
    #[arg(long)]
    pub event_id: Option<String>,

    /// Analysis ID to inspect
    #[arg(long)]
    pub analysis_id: Option<String>,

    /// Start time for query (ISO 8601 format)
    #[arg(long)]
    pub from_time: Option<String>,

    /// End time for query (ISO 8601 format)
    #[arg(long)]
    pub to_time: Option<String>,

    /// Maximum results to return
    #[arg(long, default_value = "10")]
    pub limit: Option<u32>,

    /// RuVector service endpoint
    #[arg(long, env = "RUVECTOR_ENDPOINT")]
    pub ruvector_endpoint: Option<String>,

    /// RuVector API key
    #[arg(long, env = "RUVECTOR_API_KEY")]
    pub ruvector_api_key: Option<String>,
}

/// Arguments for the replay command
#[derive(Parser, Debug)]
pub struct ReplayArgs {
    /// Original analysis ID to replay
    #[arg(long, required = true)]
    pub original_id: String,

    /// New configuration file (JSON format)
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// RuVector service endpoint
    #[arg(long, env = "RUVECTOR_ENDPOINT")]
    pub ruvector_endpoint: Option<String>,

    /// RuVector API key
    #[arg(long, env = "RUVECTOR_API_KEY")]
    pub ruvector_api_key: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_structure() {
        // Verify CLI can be built
        Cli::command().debug_assert();
    }

    #[test]
    fn test_profile_args() {
        let args = Cli::parse_from(&[
            "llm-latency-lens",
            "profile",
            "--provider",
            "openai",
            "--model",
            "gpt-4o",
            "--prompt",
            "Hello",
        ]);

        if let Commands::Profile(profile) = args.command {
            assert_eq!(profile.provider, "openai");
            assert_eq!(profile.model, "gpt-4o");
            assert_eq!(profile.prompt, Some("Hello".to_string()));
        } else {
            panic!("Expected Profile command");
        }
    }

    #[test]
    fn test_benchmark_args() {
        let args = Cli::parse_from(&[
            "llm-latency-lens",
            "benchmark",
            "--provider",
            "anthropic",
            "--model",
            "claude-3-5-sonnet-20241022",
            "--prompt",
            "Test",
            "--requests",
            "100",
            "--concurrency",
            "10",
        ]);

        if let Commands::Benchmark(bench) = args.command {
            assert_eq!(bench.provider, "anthropic");
            assert_eq!(bench.model, "claude-3-5-sonnet-20241022");
            assert_eq!(bench.requests, 100);
            assert_eq!(bench.concurrency, 10);
        } else {
            panic!("Expected Benchmark command");
        }
    }

    #[test]
    fn test_compare_args() {
        let args = Cli::parse_from(&[
            "llm-latency-lens",
            "compare",
            "openai:gpt-4o",
            "anthropic:claude-3-5-sonnet-20241022",
            "--prompt",
            "Compare me",
        ]);

        if let Commands::Compare(compare) = args.command {
            assert_eq!(compare.targets.len(), 2);
            assert_eq!(compare.targets[0], "openai:gpt-4o");
            assert_eq!(compare.targets[1], "anthropic:claude-3-5-sonnet-20241022");
        } else {
            panic!("Expected Compare command");
        }
    }

    #[test]
    fn test_global_flags() {
        let args = Cli::parse_from(&[
            "llm-latency-lens",
            "--json",
            "--quiet",
            "validate",
        ]);

        assert!(args.json);
        assert!(args.quiet);
    }

    #[test]
    fn test_verbose_flag() {
        let args = Cli::parse_from(&[
            "llm-latency-lens",
            "-vvv",
            "validate",
        ]);

        assert_eq!(args.verbose, 3);
    }
}
