//! LLM Latency Lens - Enterprise-grade CLI for LLM performance measurement
//!
//! This is the main entry point for the CLI application.

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info};

mod adapters;
mod benchmarks;
mod cli;
mod config;
mod orchestrator;

use cli::{Cli, Commands};
use config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    init_logging(&cli)?;

    // Print banner if not in quiet or JSON mode
    if !cli.quiet && !cli.json {
        print_banner();
    }

    // Setup Ctrl+C handler
    let shutdown_signal = setup_shutdown_handler();

    // Run the appropriate command
    let result = match cli.command {
        Commands::Profile(args) => {
            let config = Config::load(&args.config)?;
            cli::commands::profile::run(args, config, cli.json, cli.quiet, shutdown_signal).await
        }
        Commands::Benchmark(args) => {
            let config = Config::load(&args.config)?;
            cli::commands::benchmark::run(args, config, cli.json, cli.quiet, shutdown_signal).await
        }
        Commands::Compare(args) => {
            let config = Config::load(&args.config)?;
            cli::commands::compare::run(args, config, cli.json, cli.quiet, shutdown_signal).await
        }
        Commands::Validate(args) => {
            let config = Config::load(&args.config)?;
            cli::commands::validate::run(args, config, cli.json, cli.quiet).await
        }
        Commands::Export(args) => {
            cli::commands::export::run(args, cli.json, cli.quiet).await
        }
        Commands::Run(args) => {
            let config = Config::load(&args.config)?;
            cli::commands::run::run(args, config, cli.json, cli.quiet).await
        }
    };

    // Handle errors gracefully
    if let Err(e) = result {
        if !cli.quiet {
            eprintln!("{} {}", "Error:".red().bold(), e);

            // Print error chain
            let mut source = e.source();
            while let Some(err) = source {
                eprintln!("  {} {}", "Caused by:".red(), err);
                source = err.source();
            }
        }
        std::process::exit(1);
    }

    Ok(())
}

/// Initialize logging based on verbosity level
fn init_logging(cli: &Cli) -> Result<()> {
    use tracing_subscriber::{fmt, EnvFilter, prelude::*};

    let env_filter = if cli.verbose > 0 {
        // Map verbose flags to log levels
        match cli.verbose {
            1 => EnvFilter::new("llm_latency_lens=debug"),
            2 => EnvFilter::new("llm_latency_lens=trace,llm_latency_lens_providers=debug"),
            _ => EnvFilter::new("trace"),
        }
    } else if cli.quiet {
        EnvFilter::new("error")
    } else {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("llm_latency_lens=info"))
    };

    let fmt_layer = if cli.json {
        fmt::layer()
            .json()
            .with_current_span(false)
            .with_span_list(false)
            .boxed()
    } else {
        fmt::layer()
            .with_target(false)
            .with_thread_ids(false)
            .with_thread_names(false)
            .compact()
            .boxed()
    };

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();

    Ok(())
}

/// Setup graceful shutdown handler for Ctrl+C
fn setup_shutdown_handler() -> Arc<tokio::sync::Notify> {
    let notify = Arc::new(tokio::sync::Notify::new());
    let notify_clone = Arc::clone(&notify);

    tokio::spawn(async move {
        if let Err(e) = signal::ctrl_c().await {
            error!("Failed to listen for Ctrl+C: {}", e);
            return;
        }

        info!("Received shutdown signal (Ctrl+C)");
        notify_clone.notify_waiters();
    });

    notify
}

/// Print a beautiful banner
fn print_banner() {
    let banner = format!(
        r#"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    {}  {}                           ║
║                                                               ║
║    {}                ║
║    {}                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"#,
        "LLM Latency Lens".bright_cyan().bold(),
        format!("v{}", env!("CARGO_PKG_VERSION")).bright_black(),
        "Enterprise-grade LLM performance profiler".bright_white(),
        format!("Measure • Benchmark • Optimize").bright_green()
    );

    println!("{}", banner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        use clap::CommandFactory;

        // Verify CLI can be built without errors
        let _cli = Cli::command();
    }

    #[test]
    fn test_version() {
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty());
    }
}
