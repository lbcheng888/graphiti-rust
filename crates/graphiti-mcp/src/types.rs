/// MCP server command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Port to listen on (only for HTTP mode)
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// Host to bind to (only for HTTP mode)
    #[arg(short = 'H', long, default_value = "0.0.0.0")]
    pub host: String,

    /// Configuration file path (relative to project working directory)
    #[arg(short, long, default_value = "config.toml")]
    pub config: PathBuf,

    /// Log level
    #[arg(short, long, default_value = "info")]
    pub log_level: String,

    /// Run as stdio MCP server (default mode for Claude Desktop)
    #[arg(long, default_value = "false")]
    pub stdio: bool,

    /// Per-request timeout in seconds for stdio mode
    #[arg(long = "stdio-timeout", env = "GRAPHITI_STDIO_TIMEOUT_SECS")]
    pub stdio_timeout: Option<u64>,

    /// Project directory (for project-specific isolation, similar to Serena's --project)
    #[arg(long, env = "GRAPHITI_PROJECT")]
    pub project: Option<PathBuf>,

    /// Database path override (for project-specific databases)
    #[arg(long, env = "GRAPHITI_DB_PATH")]
    pub db_path: Option<PathBuf>,

    /// Data directory override (for project-specific data)
    #[arg(long, env = "GRAPHITI_DATA_DIR")]
    pub data_dir: Option<PathBuf>,
}

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

impl From<SimpleLLMConfig> for LLMConfig {
    fn from(simple: SimpleLLMConfig) -> Self {
        let mut config = LLMConfig::default();

        // Set the provider
        match simple.provider.to_lowercase().as_str() {
            "openai" => {
                config.provider = LLMProvider::OpenAI;
                config.openai.model = simple.model;
                if let Some(base_url) = simple.base_url {
                    config.openai.base_url = base_url;
                }
            }
            "ollama" => {
                config.provider = LLMProvider::Ollama;
                config.ollama.model = simple.model;
                if let Some(base_url) = simple.base_url {
                    config.ollama.base_url = base_url;
                }
            }
            "groq" => {
                config.provider = LLMProvider::Groq;
                config.groq.model = simple.model;
                if let Some(base_url) = simple.base_url {
                    config.groq.base_url = base_url;
                }
            }
            "huggingface" => {
                config.provider = LLMProvider::HuggingFace;
                config.huggingface.model = simple.model;
                if let Some(base_url) = simple.base_url {
                    config.huggingface.base_url = base_url;
                }
            }
            _ => {
                // Default to Ollama
                config.provider = LLMProvider::Ollama;
                config.ollama.model = simple.model;
            }
        }

        // Update from environment variables
        config.from_env();

        config
    }
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

impl From<CozoConfig> for CoreCozoConfig {
    fn from(config: CozoConfig) -> Self {
        Self {
            engine: config.engine,
            path: config.path,
            options: config.options,
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
    pub embedder: EmbedderConfig,
    /// Graphiti configuration
    #[serde(default)]
    pub graphiti: GraphitiConfig,
}

/// Application state
#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
    pub graphiti: Arc<dyn GraphitiService>,
    /// Underlying graph storage (CozoDB) for low-level graph ops in tools
    pub storage: StdArc<graphiti_cozo::CozoDriver>,
    #[allow(dead_code)]
    /// Learning detector for pattern recognition
    pub learning_detector: Arc<dyn LearningDetector>,
    #[allow(dead_code)]
    pub notification_manager: StdArc<NotificationManager>,
    pub notification_receiver:
        StdArc<tokio::sync::RwLock<Option<broadcast::Receiver<LearningNotification>>>>,
    pub project_scanner: StdArc<ProjectScanner>,
    pub rate_limiter: StdArc<Limiter>,
    pub auth_token: Option<String>,
    pub require_auth: bool,
    pub config: ServerConfig,
}

/// Trait for Graphiti service operations
#[async_trait::async_trait]
pub trait GraphitiService: Send + Sync {
    async fn add_memory(&self, req: AddMemoryRequest) -> GraphitiResult<AddMemoryResponse>;
    async fn search_memory(&self, req: SearchMemoryRequest)
        -> GraphitiResult<SearchMemoryResponse>;
    async fn get_memory(&self, id: Uuid) -> GraphitiResult<Option<MemoryNode>>;
    async fn get_related(&self, id: Uuid, depth: usize) -> GraphitiResult<Vec<RelatedMemory>>;
    /// Search extracted facts/relationships (minimal implementation for MCP parity)
    #[allow(dead_code)]
    async fn search_memory_facts(
        &self,
        query: String,
        limit: Option<usize>,
    ) -> GraphitiResult<Vec<SimpleExtractedRelationship>>;

    async fn search_memory_facts_json(
        &self,
        query: String,
        limit: Option<usize>,
    ) -> GraphitiResult<Vec<serde_json::Value>>;

    // Python-parity helpers
    async fn delete_episode(&self, id: Uuid) -> GraphitiResult<bool>;
    async fn get_episodes(&self, last_n: usize) -> GraphitiResult<Vec<EpisodeNode>>;
    async fn clear_graph(&self) -> GraphitiResult<()>;

    // In-memory entity edge helpers (minimal parity)
    async fn get_entity_edge_json(&self, id: Uuid) -> GraphitiResult<Option<serde_json::Value>>;
    async fn delete_entity_edge_by_uuid(&self, id: Uuid) -> GraphitiResult<bool>;

    // Code entity methods
    async fn add_code_entity(
        &self,
        req: AddCodeEntityRequest,
    ) -> GraphitiResult<AddCodeEntityResponse>;
    async fn record_activity(
        &self,
        req: RecordActivityRequest,
    ) -> GraphitiResult<RecordActivityResponse>;
    async fn search_code(&self, req: SearchCodeRequest) -> GraphitiResult<SearchCodeResponse>;

    // Batch operations
    async fn batch_add_code_entities(
        &self,
        req: BatchAddCodeEntitiesRequest,
    ) -> GraphitiResult<BatchAddCodeEntitiesResponse>;
    async fn batch_record_activities(
        &self,
        req: BatchRecordActivitiesRequest,
    ) -> GraphitiResult<BatchRecordActivitiesResponse>;

    // Intelligence features
    async fn get_context_suggestions(
        &self,
        req: ContextSuggestionRequest,
    ) -> GraphitiResult<ContextSuggestionResponse>;
}

/// Request to add a code entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddCodeEntityRequest {
    /// Entity type
    pub entity_type: String,
    /// Entity name
    pub name: String,
    /// Description
    pub description: String,
    /// File path (optional)
    pub file_path: Option<String>,
    /// Line range (optional)
    pub line_range: Option<(u32, u32)>,
    /// Programming language (optional)
    pub language: Option<String>,
    /// Framework (optional)
    pub framework: Option<String>,
    /// Complexity rating 1-10 (optional)
    pub complexity: Option<u8>,
    /// Importance rating 1-10 (optional)
    pub importance: Option<u8>,
}

/// Request to record development activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordActivityRequest {
    /// Activity type
    pub activity_type: String,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Developer name
    pub developer: String,
    /// Project name
    pub project: String,
    /// Related entity IDs (optional)
    pub related_entities: Option<Vec<String>>,
    /// Duration in minutes (optional)
    pub duration_minutes: Option<u32>,
    /// Difficulty rating 1-10 (optional)
    pub difficulty: Option<u8>,
    /// Quality rating 1-10 (optional)
    pub quality: Option<u8>,
}

/// Request to search code entities
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchCodeRequest {
    /// Search query
    pub query: String,
    /// Entity type filter (optional)
    pub entity_type: Option<String>,
    /// Language filter (optional)
    pub language: Option<String>,
    /// Framework filter (optional)
    pub framework: Option<String>,
    /// Maximum results
    pub limit: Option<u32>,
}

/// Request to batch add code entities
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchAddCodeEntitiesRequest {
    /// List of code entities to add
    pub entities: Vec<AddCodeEntityRequest>,
}

/// Request to batch record activities
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchRecordActivitiesRequest {
    /// List of activities to record
    pub activities: Vec<RecordActivityRequest>,
}

/// Request for intelligent context suggestions
#[derive(Debug, Serialize, Deserialize)]
pub struct ContextSuggestionRequest {
    /// Current development context
    context: String,
    /// Current working file (optional)
    current_file: Option<String>,
    /// Recent activities context (optional)
    recent_activities: Option<Vec<String>>,
    /// Maximum number of suggestions
    limit: Option<u32>,
}

/// Request to add a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddMemoryRequest {
    /// Content of the memory
    pub content: String,
    /// Optional name/title
    pub name: Option<String>,
    /// Source of the memory
    pub source: Option<String>,
    /// Type of memory
    pub memory_type: Option<String>,
    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
    /// Optional group/namespace id (not yet persisted)
    pub group_id: Option<String>,
    /// When the event occurred
    pub timestamp: Option<String>,
}

/// Response from adding a memory
#[derive(Debug, Serialize, Deserialize)]
pub struct AddMemoryResponse {
    /// ID of the created memory
    pub id: Uuid,
    /// Extracted entities
    pub entities: Vec<SimpleExtractedEntity>,
    /// Extracted relationships
    pub relationships: Vec<SimpleExtractedRelationship>,
}

/// Request to search memories
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchMemoryRequest {
    /// Search query
    pub query: String,
    /// Maximum results
    pub limit: Option<usize>,
    /// Entity type filter
    pub entity_types: Option<Vec<String>>,
    /// Time range filter (ISO 8601)
    pub start_time: Option<String>,
    pub end_time: Option<String>,
}

/// Response for add_code_entity
#[derive(Debug, Serialize, Deserialize)]
pub struct AddCodeEntityResponse {
    pub id: Uuid,
    pub message: String,
}

/// Response for record_activity
#[derive(Debug, Serialize, Deserialize)]
pub struct RecordActivityResponse {
    pub id: Uuid,
    pub message: String,
}

/// Response for search_code
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchCodeResponse {
    pub results: Vec<CodeEntity>,
    pub total: usize,
}

/// Response for batch add code entities
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchAddCodeEntitiesResponse {
    pub results: Vec<AddCodeEntityResponse>,
    pub successful_count: usize,
    pub failed_count: usize,
    pub errors: Vec<String>,
}

/// Response for batch record activities
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchRecordActivitiesResponse {
    pub results: Vec<RecordActivityResponse>,
    pub successful_count: usize,
    pub failed_count: usize,
    pub errors: Vec<String>,
}

/// A context suggestion
#[derive(Debug, Serialize, Deserialize)]
pub struct ContextSuggestion {
    /// Suggestion type
    suggestion_type: String,
    /// Suggestion title
    title: String,
    /// Detailed description
    description: String,
    /// Relevant code entities
    related_entities: Vec<String>,
    /// Confidence score (0.0-1.0)
    confidence: f32,
    /// Priority (1-10)
    priority: u8,
}

/// Response for context suggestions
#[derive(Debug, Serialize, Deserialize)]
pub struct ContextSuggestionResponse {
    pub suggestions: Vec<ContextSuggestion>,
    pub total: usize,
}

/// Response from searching memories
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchMemoryResponse {
    /// Search results
    pub results: Vec<SearchResult>,
    /// Total results found
    pub total: usize,
}

/// Search result
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Node ID
    pub id: Uuid,
    /// Node type
    pub node_type: String,
    /// Node name/title
    pub name: String,
    /// Content preview
    pub content_preview: Option<String>,
    /// Relevance score
    pub score: f32,
    /// Timestamp
    pub timestamp: String,
}

/// Memory node representation
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Node ID
    pub id: Uuid,
    /// Node type
    pub node_type: String,
    /// Node name
    pub name: String,
    /// Node content (if applicable)
    pub content: Option<String>,
    /// Creation time
    pub created_at: String,
    /// Event time
    pub event_time: String,
    /// Additional properties
    pub properties: serde_json::Value,
}

/// Related memory
#[derive(Debug, Serialize, Deserialize)]
pub struct RelatedMemory {
    /// The related node
    pub node: MemoryNode,
    /// Relationship type
    pub relationship: String,
    /// Distance from source
    pub distance: usize,
}

/// Extracted entity (simplified for API response)
#[derive(Debug, Serialize, Deserialize)]
pub struct SimpleExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub confidence: f32,
}

/// Extracted relationship (simplified for API response)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimpleExtractedRelationship {
    pub source: String,
    pub target: String,
    pub relationship: String,
    pub confidence: f32,
}
// Defaults for new server settings
const fn default_max_connections() -> usize { 100 }
const fn default_requests_per_second() -> u32 { 50 }
const fn default_request_timeout_seconds() -> u64 { 30 }
const fn default_request_body_limit_bytes() -> usize { 1 * 1024 * 1024 }
const fn default_buffer_capacity() -> usize { 1024 }
const fn default_require_auth() -> bool { true }

use clap::Parser;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use std::path::PathBuf;
use std::sync::Arc as StdArc;
use std::sync::Arc;
use tokio::sync::broadcast;
use governor::{RateLimiter, clock::DefaultClock, state::{InMemoryState, direct::NotKeyed}};
use graphiti_core::error::Result as GraphitiResult;
use graphiti_core::graph::EpisodeNode;
use graphiti_core::code_entities::CodeEntity;
use graphiti_core::graphiti::GraphitiConfig;
use graphiti_llm::{EmbedderConfig, LLMConfig, LLMProvider};
use graphiti_cozo::CozoConfig as CoreCozoConfig;
use crate::learning::{LearningDetector, NotificationManager, LearningNotification};
use crate::project_scanner::ProjectScanner;

pub type Limiter = RateLimiter<NotKeyed, InMemoryState, DefaultClock>;
