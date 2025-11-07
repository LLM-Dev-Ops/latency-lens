//! Configuration management for LLM Latency Lens
//!
//! Supports loading configuration from:
//! - TOML/YAML files
//! - Environment variables
//! - CLI arguments (highest priority)

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

// CLI import removed - not used in library context

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Provider configurations
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,

    /// Default settings
    #[serde(default)]
    pub defaults: DefaultSettings,

    /// Rate limiting configuration
    #[serde(default)]
    pub rate_limiting: RateLimitConfig,

    /// Output preferences
    #[serde(default)]
    pub output: OutputConfig,
}

/// Provider-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// API key
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// API endpoint URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,

    /// Organization ID (for OpenAI)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub organization: Option<String>,

    /// API version (for Anthropic)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,

    /// Default model for this provider
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    /// Maximum retries
    #[serde(default = "default_retries")]
    pub max_retries: u32,

    /// Enable extended thinking (Claude)
    #[serde(default)]
    pub extended_thinking: bool,
}

/// Default settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultSettings {
    /// Default provider
    #[serde(default = "default_provider")]
    pub provider: String,

    /// Default model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Default max tokens
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Default temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Default top-p
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Default timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl Default for DefaultSettings {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: None,
            max_tokens: default_max_tokens(),
            temperature: None,
            top_p: None,
            timeout_secs: default_timeout(),
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    #[serde(default)]
    pub enabled: bool,

    /// Requests per second (0 = unlimited)
    #[serde(default)]
    pub requests_per_second: u32,

    /// Burst size
    #[serde(default = "default_burst")]
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_second: 0,
            burst_size: default_burst(),
        }
    }
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Default output format
    #[serde(default = "default_output_format")]
    pub format: String,

    /// Enable colored output
    #[serde(default = "default_true")]
    pub color: bool,

    /// Pretty print JSON
    #[serde(default = "default_true")]
    pub pretty_json: bool,

    /// Show progress bars
    #[serde(default = "default_true")]
    pub progress: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: default_output_format(),
            color: true,
            pretty_json: true,
            progress: true,
        }
    }
}

impl Config {
    /// Load configuration from file and CLI arguments
    pub fn load(config_path: &Option<PathBuf>) -> Result<Self> {
        let mut config = if let Some(path) = config_path {
            Self::from_file(path)?
        } else {
            // Try default locations
            Self::from_default_locations()?
        };

        // Override with environment variables
        config.apply_env_overrides()?;

        Ok(config)
    }

    /// Load configuration from a file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let config = if path.extension().and_then(|s| s.to_str()) == Some("yaml")
            || path.extension().and_then(|s| s.to_str()) == Some("yml")
        {
            serde_yaml::from_str(&content)
                .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?
        } else {
            toml::from_str(&content)
                .with_context(|| format!("Failed to parse TOML config: {}", path.display()))?
        };

        Ok(config)
    }

    /// Try to load configuration from default locations
    fn from_default_locations() -> Result<Self> {
        let candidates = vec![
            PathBuf::from("llm-latency-lens.toml"),
            PathBuf::from("llm-latency-lens.yaml"),
            PathBuf::from(".llm-latency-lens.toml"),
            PathBuf::from(".llm-latency-lens.yaml"),
        ];

        // Also check XDG_CONFIG_HOME or ~/.config
        if let Ok(home) = std::env::var("HOME") {
            let config_dir = std::env::var("XDG_CONFIG_HOME")
                .unwrap_or_else(|_| format!("{}/.config", home));

            candidates.iter().for_each(|name| {
                let path = PathBuf::from(&config_dir)
                    .join("llm-latency-lens")
                    .join(name);
                if path.exists() {
                    // Placeholder for loading
                }
            });
        }

        for path in candidates {
            if path.exists() {
                return Self::from_file(&path);
            }
        }

        // No config file found, use defaults
        Ok(Self::default())
    }

    /// Apply environment variable overrides
    fn apply_env_overrides(&mut self) -> Result<()> {
        // Check for provider-specific API keys
        for provider in ["openai", "anthropic", "google"] {
            let env_key = format!("{}_API_KEY", provider.to_uppercase());
            if let Ok(api_key) = std::env::var(&env_key) {
                self.providers
                    .entry(provider.to_string())
                    .or_insert_with(|| ProviderConfig {
                        api_key: Some(api_key.clone()),
                        endpoint: None,
                        organization: None,
                        api_version: None,
                        default_model: None,
                        timeout_secs: default_timeout(),
                        max_retries: default_retries(),
                        extended_thinking: false,
                    })
                    .api_key = Some(api_key.clone());
            }
        }

        // OpenAI organization
        if let Ok(org) = std::env::var("OPENAI_ORGANIZATION") {
            if let Some(openai) = self.providers.get_mut("openai") {
                openai.organization = Some(org);
            }
        }

        // Anthropic API version
        if let Ok(version) = std::env::var("ANTHROPIC_API_VERSION") {
            if let Some(anthropic) = self.providers.get_mut("anthropic") {
                anthropic.api_version = Some(version);
            }
        }

        Ok(())
    }

    /// Get provider configuration
    pub fn get_provider(&self, provider: &str) -> Result<&ProviderConfig> {
        self.providers
            .get(provider)
            .with_context(|| format!("Provider '{}' not configured", provider))
    }

    /// Get or create provider configuration
    pub fn get_or_create_provider(&mut self, provider: &str) -> &mut ProviderConfig {
        self.providers
            .entry(provider.to_string())
            .or_insert_with(|| ProviderConfig {
                api_key: None,
                endpoint: None,
                organization: None,
                api_version: None,
                default_model: None,
                timeout_secs: default_timeout(),
                max_retries: default_retries(),
                extended_thinking: false,
            })
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Check that at least one provider is configured
        if self.providers.is_empty() {
            anyhow::bail!("No providers configured. Please add at least one provider configuration.");
        }

        // Validate provider configurations
        for (name, provider) in &self.providers {
            if provider.api_key.is_none() {
                anyhow::bail!(
                    "Provider '{}' is missing API key. Set it via config file or environment variable.",
                    name
                );
            }

            if provider.timeout_secs == 0 {
                anyhow::bail!("Provider '{}' has invalid timeout (must be > 0)", name);
            }
        }

        // Validate defaults
        if self.defaults.max_tokens == 0 {
            anyhow::bail!("Default max_tokens must be greater than 0");
        }

        if let Some(temp) = self.defaults.temperature {
            if !(0.0..=2.0).contains(&temp) {
                anyhow::bail!("Default temperature must be between 0.0 and 2.0");
            }
        }

        if let Some(top_p) = self.defaults.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                anyhow::bail!("Default top_p must be between 0.0 and 1.0");
            }
        }

        Ok(())
    }

    /// Merge provider-specific overrides from CLI arguments
    pub fn merge_cli_overrides(
        &mut self,
        provider: &str,
        api_key: Option<String>,
        endpoint: Option<String>,
    ) {
        let provider_config = self.get_or_create_provider(provider);

        if let Some(key) = api_key {
            provider_config.api_key = Some(key);
        }

        if let Some(url) = endpoint {
            provider_config.endpoint = Some(url);
        }
    }

    /// Get timeout as Duration
    pub fn get_timeout(&self, provider: &str) -> Duration {
        self.providers
            .get(provider)
            .map(|p| Duration::from_secs(p.timeout_secs))
            .unwrap_or_else(|| Duration::from_secs(self.defaults.timeout_secs))
    }
}

// Default value functions
fn default_provider() -> String {
    "openai".to_string()
}

fn default_max_tokens() -> u32 {
    1024
}

fn default_timeout() -> u64 {
    120
}

fn default_retries() -> u32 {
    3
}

fn default_burst() -> u32 {
    10
}

fn default_output_format() -> String {
    "json".to_string()
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.defaults.provider, "openai");
        assert_eq!(config.defaults.max_tokens, 1024);
    }

    #[test]
    fn test_toml_parsing() {
        let toml_content = r#"
[defaults]
provider = "anthropic"
max_tokens = 2048

[providers.openai]
api_key = "sk-test"
timeout_secs = 60

[providers.anthropic]
api_key = "sk-ant-test"
api_version = "2023-06-01"
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(toml_content.as_bytes()).unwrap();

        let config = Config::from_file(file.path()).unwrap();
        assert_eq!(config.defaults.provider, "anthropic");
        assert_eq!(config.defaults.max_tokens, 2048);
        assert_eq!(config.providers.len(), 2);
        assert_eq!(
            config.providers["openai"].api_key,
            Some("sk-test".to_string())
        );
    }

    #[test]
    fn test_yaml_parsing() {
        let yaml_content = r#"
defaults:
  provider: google
  max_tokens: 512

providers:
  google:
    api_key: AIza-test
    timeout_secs: 30
"#;

        let mut file = NamedTempFile::with_suffix(".yaml").unwrap();
        file.write_all(yaml_content.as_bytes()).unwrap();

        let config = Config::from_file(file.path()).unwrap();
        assert_eq!(config.defaults.provider, "google");
        assert_eq!(config.defaults.max_tokens, 512);
    }

    #[test]
    fn test_validation() {
        let mut config = Config::default();

        // Should fail - no providers
        assert!(config.validate().is_err());

        // Add provider with API key
        config.providers.insert(
            "openai".to_string(),
            ProviderConfig {
                api_key: Some("test".to_string()),
                endpoint: None,
                organization: None,
                api_version: None,
                default_model: None,
                timeout_secs: 60,
                max_retries: 3,
                extended_thinking: false,
            },
        );

        // Should pass
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_get_provider() {
        let mut config = Config::default();
        config.providers.insert(
            "test".to_string(),
            ProviderConfig {
                api_key: Some("key".to_string()),
                endpoint: None,
                organization: None,
                api_version: None,
                default_model: None,
                timeout_secs: 30,
                max_retries: 2,
                extended_thinking: false,
            },
        );

        assert!(config.get_provider("test").is_ok());
        assert!(config.get_provider("nonexistent").is_err());
    }

    #[test]
    fn test_merge_cli_overrides() {
        let mut config = Config::default();

        config.merge_cli_overrides(
            "openai",
            Some("new-key".to_string()),
            Some("https://api.example.com".to_string()),
        );

        let provider = config.get_provider("openai").unwrap();
        assert_eq!(provider.api_key, Some("new-key".to_string()));
        assert_eq!(provider.endpoint, Some("https://api.example.com".to_string()));
    }
}
