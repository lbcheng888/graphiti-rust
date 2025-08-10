//! Configuration management system with environment variable support and hot reloading

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Configuration validation error
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// IO error when reading config file
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    /// YAML parsing error
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    /// TOML parsing error
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),
    /// Environment variable error
    #[error("Environment variable error: {0}")]
    EnvVar(#[from] std::env::VarError),
    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),
    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(String),
}

/// Configuration format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    /// JSON format
    Json,
    /// YAML format
    Yaml,
    /// TOML format
    Toml,
}

impl ConfigFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "json" => Some(Self::Json),
            "yaml" | "yml" => Some(Self::Yaml),
            "toml" => Some(Self::Toml),
            _ => None,
        }
    }
}

/// Configuration validation trait
pub trait ConfigValidation {
    /// Validate configuration
    fn validate(&self) -> Result<(), ConfigError>;
}

/// Environment variable resolver
#[derive(Debug, Clone)]
pub struct EnvResolver {
    /// Prefix for environment variables
    prefix: String,
    /// Default values
    defaults: HashMap<String, String>,
}

impl EnvResolver {
    /// Create a new environment resolver
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_uppercase(),
            defaults: HashMap::new(),
        }
    }

    /// Add a default value
    pub fn with_default(mut self, key: &str, value: &str) -> Self {
        self.defaults.insert(key.to_string(), value.to_string());
        self
    }

    /// Resolve environment variable
    pub fn resolve(&self, key: &str) -> Option<String> {
        let env_key = format!("{}_{}", self.prefix, key.to_uppercase());

        // Try environment variable first
        if let Ok(value) = std::env::var(&env_key) {
            debug!("Resolved {} from environment: {}", key, value);
            return Some(value);
        }

        // Fall back to default
        if let Some(default) = self.defaults.get(key) {
            debug!("Using default value for {}: {}", key, default);
            return Some(default.clone());
        }

        None
    }

    /// Resolve and parse environment variable
    pub fn resolve_parse<T>(&self, key: &str) -> Result<Option<T>, ConfigError>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        if let Some(value) = self.resolve(key) {
            match value.parse::<T>() {
                Ok(parsed) => Ok(Some(parsed)),
                Err(e) => Err(ConfigError::Validation(format!(
                    "Failed to parse {} for key {}: {}",
                    value, key, e
                ))),
            }
        } else {
            Ok(None)
        }
    }
}

/// Configuration loader
pub struct ConfigLoader {
    /// Environment resolver
    env_resolver: EnvResolver,
    /// Configuration search paths
    search_paths: Vec<PathBuf>,
}

impl ConfigLoader {
    /// Create a new configuration loader
    pub fn new(env_prefix: &str) -> Self {
        Self {
            env_resolver: EnvResolver::new(env_prefix),
            search_paths: vec![
                PathBuf::from("."),
                PathBuf::from("config"),
                PathBuf::from("/etc/graphiti"),
                PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".config/graphiti"),
            ],
        }
    }

    /// Add a search path
    pub fn add_search_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.search_paths.push(path.as_ref().to_path_buf());
        self
    }

    /// Add environment default
    pub fn with_env_default(mut self, key: &str, value: &str) -> Self {
        self.env_resolver = self.env_resolver.with_default(key, value);
        self
    }

    /// Load configuration from file
    pub fn load_from_file<T>(&self, filename: &str) -> Result<T, ConfigError>
    where
        T: DeserializeOwned + ConfigValidation,
    {
        // Try to find the config file
        let config_path = self.find_config_file(filename)?;
        info!("Loading configuration from: {}", config_path.display());

        // Read file content
        let content = std::fs::read_to_string(&config_path)?;

        // Parse based on format
        let format = ConfigFormat::from_extension(&config_path)
            .ok_or_else(|| ConfigError::Validation("Unknown config file format".to_string()))?;

        let mut config: T = match format {
            ConfigFormat::Json => serde_json::from_str(&content)?,
            ConfigFormat::Yaml => serde_yaml::from_str(&content)?,
            ConfigFormat::Toml => toml::from_str(&content)?,
        };

        // Apply environment variable overrides
        self.apply_env_overrides(&mut config)?;

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Load configuration with fallback to defaults
    pub fn load_with_defaults<T>(&self, filename: &str, defaults: T) -> T
    where
        T: DeserializeOwned + ConfigValidation + Clone,
    {
        match self.load_from_file(filename) {
            Ok(config) => {
                info!("Successfully loaded configuration from file");
                config
            }
            Err(e) => {
                warn!(
                    "Failed to load configuration from file: {}, using defaults",
                    e
                );
                defaults
            }
        }
    }

    /// Find configuration file in search paths
    fn find_config_file(&self, filename: &str) -> Result<PathBuf, ConfigError> {
        // Try exact filename first
        for search_path in &self.search_paths {
            let config_path = search_path.join(filename);
            if config_path.exists() {
                return Ok(config_path);
            }
        }

        // Try with different extensions
        let extensions = ["json", "yaml", "yml", "toml"];
        let base_name = Path::new(filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(filename);

        for search_path in &self.search_paths {
            for ext in &extensions {
                let config_path = search_path.join(format!("{}.{}", base_name, ext));
                if config_path.exists() {
                    return Ok(config_path);
                }
            }
        }

        Err(ConfigError::Validation(format!(
            "Configuration file '{}' not found in search paths: {:?}",
            filename, self.search_paths
        )))
    }

    /// Apply environment variable overrides (simplified implementation)
    fn apply_env_overrides<T>(&self, _config: &mut T) -> Result<(), ConfigError> {
        // In a real implementation, this would use reflection or a macro
        // to automatically apply environment variable overrides to config fields
        // For now, this is a placeholder
        Ok(())
    }
}

/// Hot-reloadable configuration manager
pub struct ConfigManager<T> {
    /// Current configuration
    config: Arc<RwLock<T>>,
    /// Configuration file path
    config_path: Option<PathBuf>,
    /// Configuration loader
    loader: ConfigLoader,
    /// File watcher (simplified)
    _watcher: Option<()>, // In real implementation, this would be a file watcher
}

impl<T> ConfigManager<T>
where
    T: DeserializeOwned + ConfigValidation + Clone + Send + Sync + 'static,
{
    /// Create a new configuration manager
    pub fn new(initial_config: T, loader: ConfigLoader) -> Self {
        Self {
            config: Arc::new(RwLock::new(initial_config)),
            config_path: None,
            loader,
            _watcher: None,
        }
    }

    /// Load configuration from file
    pub async fn load_from_file(&mut self, filename: &str) -> Result<(), ConfigError> {
        let config = self.loader.load_from_file(filename)?;
        let config_path = self.loader.find_config_file(filename)?;

        *self.config.write().await = config;
        self.config_path = Some(config_path);

        info!("Configuration loaded and updated");
        Ok(())
    }

    /// Get current configuration
    pub async fn get(&self) -> T {
        self.config.read().await.clone()
    }

    /// Update configuration
    pub async fn update<F>(&self, updater: F) -> Result<(), ConfigError>
    where
        F: FnOnce(&mut T),
    {
        let mut config = self.config.write().await;
        updater(&mut config);
        config.validate()?;
        info!("Configuration updated");
        Ok(())
    }

    /// Reload configuration from file
    pub async fn reload(&mut self) -> Result<(), ConfigError> {
        if let Some(config_path) = self.config_path.clone() {
            if let Some(filename) = config_path.file_name().and_then(|s| s.to_str()) {
                self.load_from_file(filename).await?;
                info!("Configuration reloaded from file");
            }
        }
        Ok(())
    }

    /// Start watching for configuration file changes (simplified)
    pub async fn start_watching(&mut self) -> Result<(), ConfigError> {
        // In a real implementation, this would set up a file watcher
        // that automatically reloads the configuration when the file changes
        info!("Configuration file watching started (placeholder)");
        Ok(())
    }
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database URL
    pub url: String,
    /// Maximum number of connections
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connection_timeout: u64,
    /// Query timeout in seconds
    pub query_timeout: u64,
    /// Enable connection pooling
    pub enable_pooling: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "sqlite://graphiti.db".to_string(),
            max_connections: 10,
            connection_timeout: 30,
            query_timeout: 60,
            enable_pooling: true,
        }
    }
}

impl ConfigValidation for DatabaseConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.url.is_empty() {
            return Err(ConfigError::MissingField("database.url".to_string()));
        }
        if self.max_connections == 0 {
            return Err(ConfigError::Validation(
                "max_connections must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// LLM service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Service provider (openai, anthropic, etc.)
    pub provider: String,
    /// API key
    pub api_key: String,
    /// Model name
    pub model: String,
    /// API endpoint URL
    pub endpoint: Option<String>,
    /// Request timeout in seconds
    pub timeout: u64,
    /// Maximum retries
    pub max_retries: u32,
    /// Rate limit (requests per minute)
    pub rate_limit: u32,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            api_key: String::new(),
            model: "gpt-4".to_string(),
            endpoint: None,
            timeout: 30,
            max_retries: 3,
            rate_limit: 60,
        }
    }
}

impl ConfigValidation for LLMConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.provider.is_empty() {
            return Err(ConfigError::MissingField("llm.provider".to_string()));
        }
        if self.api_key.is_empty() {
            return Err(ConfigError::MissingField("llm.api_key".to_string()));
        }
        if self.model.is_empty() {
            return Err(ConfigError::MissingField("llm.model".to_string()));
        }
        Ok(())
    }
}

/// Embedding service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Service provider
    pub provider: String,
    /// API key
    pub api_key: String,
    /// Model name
    pub model: String,
    /// Embedding dimensions
    pub dimensions: u32,
    /// Batch size for embedding requests
    pub batch_size: u32,
    /// Request timeout in seconds
    pub timeout: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            api_key: String::new(),
            model: "text-embedding-ada-002".to_string(),
            dimensions: 1536,
            batch_size: 100,
            timeout: 30,
        }
    }
}

impl ConfigValidation for EmbeddingConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.provider.is_empty() {
            return Err(ConfigError::MissingField("embedding.provider".to_string()));
        }
        if self.api_key.is_empty() {
            return Err(ConfigError::MissingField("embedding.api_key".to_string()));
        }
        if self.dimensions == 0 {
            return Err(ConfigError::Validation(
                "embedding dimensions must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server host
    pub host: String,
    /// Server port
    pub port: u16,
    /// Enable TLS
    pub tls: bool,
    /// TLS certificate path
    pub cert_path: Option<String>,
    /// TLS key path
    pub key_path: Option<String>,
    /// Request timeout in seconds
    pub request_timeout: u64,
    /// Maximum request body size in bytes
    pub max_body_size: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            tls: false,
            cert_path: None,
            key_path: None,
            request_timeout: 30,
            max_body_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

impl ConfigValidation for ServerConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.host.is_empty() {
            return Err(ConfigError::MissingField("server.host".to_string()));
        }
        if self.tls {
            if self.cert_path.is_none() {
                return Err(ConfigError::MissingField("server.cert_path".to_string()));
            }
            if self.key_path.is_none() {
                return Err(ConfigError::MissingField("server.key_path".to_string()));
            }
        }
        Ok(())
    }
}

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Database configuration
    pub database: DatabaseConfig,
    /// LLM configuration
    pub llm: LLMConfig,
    /// Embedding configuration
    pub embedding: EmbeddingConfig,
    /// Server configuration
    pub server: ServerConfig,
    /// Log level
    pub log_level: String,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics port
    pub metrics_port: u16,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            database: DatabaseConfig::default(),
            llm: LLMConfig::default(),
            embedding: EmbeddingConfig::default(),
            server: ServerConfig::default(),
            log_level: "info".to_string(),
            enable_metrics: true,
            metrics_port: 9090,
        }
    }
}

impl ConfigValidation for AppConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        self.database.validate()?;
        self.llm.validate()?;
        self.embedding.validate()?;
        self.server.validate()?;

        // Validate log level
        match self.log_level.to_lowercase().as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {}
            _ => return Err(ConfigError::Validation("Invalid log level".to_string())),
        }

        Ok(())
    }
}
