//! Configuration models for the MCP server

use serde::{Deserialize, Serialize};

/// Server settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
    /// Enable TLS with provided certificates
    #[serde(default)]
    pub tls: Option<TlsSettings>,
    /// Maximum number of in-flight requests allowed
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    /// Optional global requests-per-second rate limit
    #[serde(default = "default_requests_per_second")]
    pub requests_per_second: u32,
    /// Per-request timeout in seconds
    #[serde(default = "default_request_timeout_seconds")]
    pub request_timeout_seconds: u64,
    /// Max request body size in bytes
    #[serde(default = "default_request_body_limit_bytes")]
    pub request_body_limit_bytes: usize,
    /// Buffer capacity for pending requests before load shedding
    #[serde(default = "default_buffer_capacity")]
    pub buffer_capacity: usize,
    /// Require Authorization header for protected endpoints
    #[serde(default = "default_require_auth")]
    pub require_auth: bool,
    /// Allowed CORS origins (use ["*"] for any)
    #[serde(default)]
    pub allowed_origins: Option<Vec<String>>,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            tls: None,
            max_connections: default_max_connections(),
            requests_per_second: default_requests_per_second(),
            request_timeout_seconds: default_request_timeout_seconds(),
            request_body_limit_bytes: default_request_body_limit_bytes(),
            buffer_capacity: default_buffer_capacity(),
            require_auth: default_require_auth(),
            allowed_origins: None,
        }
    }
}

/// TLS settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsSettings {
    /// Path to PEM-encoded certificate chain
    pub cert_path: String,
    /// Path to PEM-encoded private key
    pub key_path: String,
}

/// Simple LLM configuration that matches the TOML structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleLLMConfig {
    pub provider: String,
    pub base_url: Option<String>,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_retries: Option<u32>,
}

/// CozoDB configuration wrapper for TOML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CozoConfig {
    /// Storage engine ("mem", "sqlite", "rocksdb")
    pub engine: String,
    /// Database path (for file-based engines)
    pub path: String,
    /// Additional options
    pub options: serde_json::Value,
}

impl Default for CozoConfig {
    fn default() -> Self {
        Self {
            engine: "mem".to_string(),
            path: "".to_string(),
            options: serde_json::json!({}),
        }
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server settings
    #[serde(default)]
    pub server: ServerSettings,
    /// CozoDB configuration
    #[serde(default)]
    pub cozo: CozoConfig,
    /// LLM configuration
    pub llm: SimpleLLMConfig,
    /// Embedder configuration
    pub embedder: graphiti_llm::EmbedderConfig,
    /// Graphiti configuration
    #[serde(default)]
    pub graphiti: graphiti_core::graphiti::GraphitiConfig,
}

// Defaults for new server settings
const fn default_max_connections() -> usize {
    100
}
const fn default_requests_per_second() -> u32 {
    50
}
const fn default_request_timeout_seconds() -> u64 {
    30
}
const fn default_request_body_limit_bytes() -> usize {
    1 * 1024 * 1024
}
const fn default_buffer_capacity() -> usize {
    1024
}
const fn default_require_auth() -> bool {
    true
}
