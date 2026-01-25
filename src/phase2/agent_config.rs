//! Phase 2 Agent Configuration
//!
//! Defines configuration structures for Phase 2 agents including
//! performance budgets and required environment variables.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Required environment variables for Phase 2 agents
pub const REQUIRED_ENV_VARS: &[&str] = &[
    "RUVECTOR_SERVICE_URL",
    "RUVECTOR_API_KEY",
    "AGENT_NAME",
    "AGENT_DOMAIN",
    "AGENT_PHASE",
    "AGENT_LAYER",
];

/// Phase 2 layer identifier
pub const AGENT_PHASE: &str = "phase2";

/// Layer 1 identifier
pub const AGENT_LAYER: &str = "layer1";

/// Performance budget limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBudget {
    /// Maximum tokens per request (default: 1000)
    pub max_tokens: u32,
    /// Maximum latency in milliseconds (default: 2000)
    pub max_latency_ms: u64,
    /// Maximum API calls per run (default: 3)
    pub max_calls_per_run: u32,
}

impl Default for PerformanceBudget {
    fn default() -> Self {
        Self {
            max_tokens: 1000,
            max_latency_ms: 2000,
            max_calls_per_run: 3,
        }
    }
}

impl PerformanceBudget {
    /// Create from environment variables with defaults
    pub fn from_env() -> Self {
        Self {
            max_tokens: std::env::var("MAX_TOKENS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1000),
            max_latency_ms: std::env::var("MAX_LATENCY_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2000),
            max_calls_per_run: std::env::var("MAX_CALLS_PER_RUN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
        }
    }

    /// Check if a latency value exceeds the budget
    pub fn exceeds_latency(&self, latency_ms: u64) -> bool {
        latency_ms > self.max_latency_ms
    }

    /// Check if token count exceeds the budget
    pub fn exceeds_tokens(&self, tokens: u32) -> bool {
        tokens > self.max_tokens
    }

    /// Check if call count exceeds the budget
    pub fn exceeds_calls(&self, calls: u32) -> bool {
        calls > self.max_calls_per_run
    }
}

/// Agent configuration for Phase 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent name (from AGENT_NAME)
    pub name: String,
    /// Agent domain (from AGENT_DOMAIN)
    pub domain: String,
    /// Agent phase (must be "phase2")
    pub phase: String,
    /// Agent layer (must be "layer1")
    pub layer: String,
    /// Performance budget
    pub budget: PerformanceBudget,
}

impl AgentConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, ConfigError> {
        let name = std::env::var("AGENT_NAME")
            .map_err(|_| ConfigError::MissingEnvVar("AGENT_NAME".to_string()))?;

        let domain = std::env::var("AGENT_DOMAIN")
            .map_err(|_| ConfigError::MissingEnvVar("AGENT_DOMAIN".to_string()))?;

        let phase = std::env::var("AGENT_PHASE")
            .map_err(|_| ConfigError::MissingEnvVar("AGENT_PHASE".to_string()))?;

        let layer = std::env::var("AGENT_LAYER")
            .map_err(|_| ConfigError::MissingEnvVar("AGENT_LAYER".to_string()))?;

        // Validate phase and layer
        if phase != AGENT_PHASE {
            return Err(ConfigError::InvalidValue {
                var: "AGENT_PHASE".to_string(),
                expected: AGENT_PHASE.to_string(),
                actual: phase,
            });
        }

        if layer != AGENT_LAYER {
            return Err(ConfigError::InvalidValue {
                var: "AGENT_LAYER".to_string(),
                expected: AGENT_LAYER.to_string(),
                actual: layer,
            });
        }

        Ok(Self {
            name,
            domain,
            phase,
            layer,
            budget: PerformanceBudget::from_env(),
        })
    }

    /// Get full agent identifier
    pub fn agent_id(&self) -> String {
        format!("{}.{}.{}.{}", self.phase, self.layer, self.domain, self.name)
    }
}

/// Full Phase 2 configuration
#[derive(Debug, Clone)]
pub struct Phase2Config {
    /// Agent configuration
    pub agent: AgentConfig,
    /// RuVector service URL
    pub ruvector_url: String,
    /// RuVector API key
    pub ruvector_api_key: String,
    /// Enable telemetry emission
    pub emit_telemetry: bool,
    /// Enable signal emission
    pub emit_signals: bool,
}

impl Phase2Config {
    /// Load full configuration from environment
    pub fn from_env() -> Result<Self, ConfigError> {
        let ruvector_url = std::env::var("RUVECTOR_SERVICE_URL")
            .map_err(|_| ConfigError::MissingEnvVar("RUVECTOR_SERVICE_URL".to_string()))?;

        let ruvector_api_key = std::env::var("RUVECTOR_API_KEY")
            .map_err(|_| ConfigError::MissingEnvVar("RUVECTOR_API_KEY".to_string()))?;

        let emit_telemetry = std::env::var("EMIT_TELEMETRY")
            .map(|s| s.to_lowercase() != "false")
            .unwrap_or(true);

        let emit_signals = std::env::var("EMIT_SIGNALS")
            .map(|s| s.to_lowercase() != "false")
            .unwrap_or(true);

        Ok(Self {
            agent: AgentConfig::from_env()?,
            ruvector_url,
            ruvector_api_key,
            emit_telemetry,
            emit_signals,
        })
    }

    /// Validate all required environment variables are present
    pub fn validate_env() -> Result<(), Vec<String>> {
        let missing: Vec<String> = REQUIRED_ENV_VARS
            .iter()
            .filter(|var| std::env::var(var).is_err())
            .map(|s| s.to_string())
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }
}

/// Configuration error types
#[derive(Debug)]
pub enum ConfigError {
    /// Required environment variable is missing
    MissingEnvVar(String),
    /// Environment variable has invalid value
    InvalidValue {
        var: String,
        expected: String,
        actual: String,
    },
    /// Multiple configuration errors
    Multiple(Vec<ConfigError>),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::MissingEnvVar(var) => {
                write!(f, "Required environment variable '{}' is not set", var)
            }
            ConfigError::InvalidValue {
                var,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Environment variable '{}' has invalid value: expected '{}', got '{}'",
                    var, expected, actual
                )
            }
            ConfigError::Multiple(errors) => {
                writeln!(f, "Multiple configuration errors:")?;
                for err in errors {
                    writeln!(f, "  - {}", err)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_budget_defaults() {
        let budget = PerformanceBudget::default();
        assert_eq!(budget.max_tokens, 1000);
        assert_eq!(budget.max_latency_ms, 2000);
        assert_eq!(budget.max_calls_per_run, 3);
    }

    #[test]
    fn test_budget_exceeds_checks() {
        let budget = PerformanceBudget::default();

        assert!(!budget.exceeds_latency(1000));
        assert!(!budget.exceeds_latency(2000));
        assert!(budget.exceeds_latency(2001));

        assert!(!budget.exceeds_tokens(500));
        assert!(!budget.exceeds_tokens(1000));
        assert!(budget.exceeds_tokens(1001));

        assert!(!budget.exceeds_calls(2));
        assert!(!budget.exceeds_calls(3));
        assert!(budget.exceeds_calls(4));
    }
}
