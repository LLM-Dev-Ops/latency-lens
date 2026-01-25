//! Phase 2 Runtime
//!
//! Startup hardening and runtime management for Phase 2 agents.
//!
//! # Startup Behavior
//!
//! 1. Validate required environment variables
//! 2. Initialize and verify RuVector connection
//! 3. Fail fast if RuVector is unavailable
//! 4. Initialize caching layer
//! 5. Register signal emitters

use super::{
    agent_config::{ConfigError, Phase2Config, REQUIRED_ENV_VARS},
    cache::{CacheConfig, LineageCache},
    signals::SignalEmitter,
};
use crate::agents::ruvector::{RuVectorClient, RuVectorConfig, RuVectorError};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

/// Startup validation result
#[derive(Debug)]
pub struct StartupValidation {
    /// Whether all validations passed
    pub passed: bool,
    /// Missing environment variables
    pub missing_env_vars: Vec<String>,
    /// RuVector connection status
    pub ruvector_connected: bool,
    /// RuVector version if connected
    pub ruvector_version: Option<String>,
    /// Any validation errors
    pub errors: Vec<String>,
}

impl StartupValidation {
    /// Create a successful validation result
    pub fn success(ruvector_version: String) -> Self {
        Self {
            passed: true,
            missing_env_vars: Vec::new(),
            ruvector_connected: true,
            ruvector_version: Some(ruvector_version),
            errors: Vec::new(),
        }
    }

    /// Create a failed validation result
    pub fn failure(errors: Vec<String>) -> Self {
        Self {
            passed: false,
            missing_env_vars: Vec::new(),
            ruvector_connected: false,
            ruvector_version: None,
            errors,
        }
    }
}

/// Runtime error types
#[derive(Debug)]
pub enum RuntimeError {
    /// Configuration error
    Config(ConfigError),
    /// RuVector connection failed
    RuVectorUnavailable(String),
    /// Startup validation failed
    StartupFailed(StartupValidation),
    /// Runtime operation failed
    OperationFailed(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::Config(e) => write!(f, "Configuration error: {}", e),
            RuntimeError::RuVectorUnavailable(msg) => write!(f, "RuVector unavailable: {}", msg),
            RuntimeError::StartupFailed(validation) => {
                writeln!(f, "Startup validation failed:")?;
                if !validation.missing_env_vars.is_empty() {
                    writeln!(f, "  Missing environment variables:")?;
                    for var in &validation.missing_env_vars {
                        writeln!(f, "    - {}", var)?;
                    }
                }
                if !validation.ruvector_connected {
                    writeln!(f, "  RuVector connection: FAILED")?;
                }
                for err in &validation.errors {
                    writeln!(f, "  Error: {}", err)?;
                }
                Ok(())
            }
            RuntimeError::OperationFailed(msg) => write!(f, "Operation failed: {}", msg),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<ConfigError> for RuntimeError {
    fn from(e: ConfigError) -> Self {
        RuntimeError::Config(e)
    }
}

impl From<RuVectorError> for RuntimeError {
    fn from(e: RuVectorError) -> Self {
        RuntimeError::RuVectorUnavailable(e.to_string())
    }
}

/// Phase 2 Runtime
///
/// Manages startup, configuration, and runtime services for Phase 2 agents.
pub struct Phase2Runtime {
    /// Configuration
    config: Phase2Config,
    /// RuVector client
    ruvector: Arc<RuVectorClient>,
    /// Lineage cache
    cache: LineageCache,
    /// Signal emitter
    emitter: SignalEmitter,
    /// Startup validation result
    validation: StartupValidation,
}

impl Phase2Runtime {
    /// Initialize Phase 2 runtime with startup hardening
    ///
    /// This function:
    /// 1. Validates all required environment variables
    /// 2. Initializes RuVector client
    /// 3. Verifies RuVector connection
    /// 4. Fails fast if RuVector is unavailable
    /// 5. Initializes caching and signal emission
    pub async fn init() -> Result<Self, RuntimeError> {
        info!("Initializing Phase 2 runtime...");

        // Step 1: Validate environment variables
        let mut validation_errors = Vec::new();
        let missing: Vec<String> = REQUIRED_ENV_VARS
            .iter()
            .filter(|var| std::env::var(var).is_err())
            .map(|s| s.to_string())
            .collect();

        if !missing.is_empty() {
            for var in &missing {
                error!(var = %var, "Required environment variable is not set");
                validation_errors.push(format!("Missing required env var: {}", var));
            }

            return Err(RuntimeError::StartupFailed(StartupValidation {
                passed: false,
                missing_env_vars: missing,
                ruvector_connected: false,
                ruvector_version: None,
                errors: validation_errors,
            }));
        }

        // Step 2: Load configuration
        let config = Phase2Config::from_env()?;
        info!(
            agent_id = %config.agent.agent_id(),
            phase = %config.agent.phase,
            layer = %config.agent.layer,
            "Configuration loaded"
        );

        // Step 3: Initialize RuVector client
        let ruvector_config = RuVectorConfig {
            endpoint: config.ruvector_url.clone(),
            api_key: Some(config.ruvector_api_key.clone()),
            timeout_ms: 30000,
            max_retries: 3,
            compression: true,
            batch_size: 100,
        };

        let ruvector = RuVectorClient::new(ruvector_config)?;

        // Step 4: Verify RuVector connection (HARD FAILURE if unavailable)
        info!(endpoint = %config.ruvector_url, "Connecting to RuVector...");

        match ruvector.connect().await {
            Ok(()) => {
                info!("RuVector connection verified");
            }
            Err(e) => {
                error!(error = %e, "FATAL: RuVector is unavailable");
                return Err(RuntimeError::RuVectorUnavailable(format!(
                    "Hard startup failure: RuVector unavailable at {}: {}",
                    config.ruvector_url, e
                )));
            }
        }

        // Step 5: Get RuVector version for validation result
        let health = ruvector.health_check().await?;
        let ruvector_version = health.version.clone();

        info!(
            version = %ruvector_version,
            database_connected = health.database_connected,
            "RuVector health check passed"
        );

        // Step 6: Initialize caching layer
        let cache_config = CacheConfig::from_env();
        let cache = LineageCache::new(cache_config);
        info!(
            min_ttl = cache.config().min_ttl_secs,
            max_ttl = cache.config().max_ttl_secs,
            "Cache initialized"
        );

        // Step 7: Initialize signal emitter
        let emitter = SignalEmitter::new(config.agent.agent_id(), config.emit_signals)
            .with_ruvector(config.ruvector_url.clone());

        if config.emit_signals {
            info!("Signal emission enabled");
        } else {
            warn!("Signal emission disabled");
        }

        // Create validation result
        let validation = StartupValidation::success(ruvector_version);

        info!(
            agent_id = %config.agent.agent_id(),
            "Phase 2 runtime initialized successfully"
        );

        Ok(Self {
            config,
            ruvector: Arc::new(ruvector),
            cache,
            emitter,
            validation,
        })
    }

    /// Initialize with custom configuration (for testing)
    pub async fn init_with_config(
        config: Phase2Config,
        ruvector: RuVectorClient,
    ) -> Result<Self, RuntimeError> {
        // Verify RuVector connection
        ruvector.connect().await?;
        let health = ruvector.health_check().await?;

        let cache = LineageCache::new(CacheConfig::from_env());
        let emitter = SignalEmitter::new(config.agent.agent_id(), config.emit_signals);

        Ok(Self {
            config,
            ruvector: Arc::new(ruvector),
            cache,
            emitter,
            validation: StartupValidation::success(health.version),
        })
    }

    /// Get runtime configuration
    pub fn config(&self) -> &Phase2Config {
        &self.config
    }

    /// Get RuVector client
    pub fn ruvector(&self) -> Arc<RuVectorClient> {
        Arc::clone(&self.ruvector)
    }

    /// Get lineage cache
    pub fn cache(&self) -> &LineageCache {
        &self.cache
    }

    /// Get signal emitter
    pub fn emitter(&self) -> &SignalEmitter {
        &self.emitter
    }

    /// Get startup validation result
    pub fn validation(&self) -> &StartupValidation {
        &self.validation
    }

    /// Check if runtime is healthy
    pub async fn health_check(&self) -> Result<bool, RuntimeError> {
        // Check RuVector connection
        match self.ruvector.health_check().await {
            Ok(health) => {
                if !health.healthy {
                    warn!("RuVector reports unhealthy status");
                    return Ok(false);
                }
                Ok(true)
            }
            Err(e) => {
                error!(error = %e, "RuVector health check failed");
                Err(RuntimeError::RuVectorUnavailable(e.to_string()))
            }
        }
    }

    /// Get agent ID
    pub fn agent_id(&self) -> String {
        self.config.agent.agent_id()
    }

    /// Check if a latency exceeds the performance budget
    pub fn exceeds_latency_budget(&self, latency_ms: u64) -> bool {
        self.config.agent.budget.exceeds_latency(latency_ms)
    }

    /// Check if token count exceeds the performance budget
    pub fn exceeds_token_budget(&self, tokens: u32) -> bool {
        self.config.agent.budget.exceeds_tokens(tokens)
    }

    /// Check if call count exceeds the performance budget
    pub fn exceeds_call_budget(&self, calls: u32) -> bool {
        self.config.agent.budget.exceeds_calls(calls)
    }
}

/// Convenience function to validate environment on startup
pub fn validate_phase2_env() -> Result<(), Vec<String>> {
    Phase2Config::validate_env()
}

/// Print Phase 2 startup banner
pub fn print_startup_banner(config: &Phase2Config) {
    info!("╔════════════════════════════════════════════════════════════╗");
    info!("║            Phase 2 - Operational Intelligence              ║");
    info!("║                       Layer 1                              ║");
    info!("╠════════════════════════════════════════════════════════════╣");
    info!("║  Agent: {:<50} ║", config.agent.agent_id());
    info!("║  Domain: {:<49} ║", config.agent.domain);
    info!("║  Max Tokens: {:<45} ║", config.agent.budget.max_tokens);
    info!("║  Max Latency: {:<44} ║", format!("{}ms", config.agent.budget.max_latency_ms));
    info!("║  Max Calls/Run: {:<42} ║", config.agent.budget.max_calls_per_run);
    info!("╚════════════════════════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_startup_validation_success() {
        let validation = StartupValidation::success("1.0.0".to_string());
        assert!(validation.passed);
        assert!(validation.ruvector_connected);
        assert_eq!(validation.ruvector_version, Some("1.0.0".to_string()));
    }

    #[test]
    fn test_startup_validation_failure() {
        let validation = StartupValidation::failure(vec!["Test error".to_string()]);
        assert!(!validation.passed);
        assert!(!validation.ruvector_connected);
        assert_eq!(validation.errors.len(), 1);
    }
}
