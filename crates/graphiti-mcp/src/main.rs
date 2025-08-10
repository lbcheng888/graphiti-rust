//! Graphiti MCP Server

mod ast_parser;
mod integration;
mod learning;
mod learning_endpoints;
mod learning_integration;
mod project_scanner;

use axum::extract::Query;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::get;
use axum::routing::post;
use axum::Router;
use axum::middleware;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::http::Request;
// use chrono::DateTime; // 暂时未使用
use chrono::Utc;
use clap::Parser;
use graphiti_core::code_entities::CodeEntity;
use graphiti_core::code_entities::CodeEntityType;

use graphiti_core::code_entities::DevelopmentActivity;
use graphiti_core::code_entities::WorkflowStage;
use graphiti_core::error::Result as GraphitiResult;
use graphiti_core::graph::EntityNode;
use graphiti_core::graph::EpisodeNode;
use graphiti_core::graph::EpisodeType;
use graphiti_core::graph::TemporalMetadata;
use graphiti_core::prelude::*;
use graphiti_cozo::CozoConfig as CoreCozoConfig;
use graphiti_cozo::CozoDriver;
use graphiti_llm::create_llm_client;
use graphiti_llm::detect_device;
use graphiti_llm::qwen3_embed_anything::Qwen3EmbedAnythingClient;
use graphiti_llm::qwen3_embed_anything::Qwen3EmbedAnythingConfig;
use graphiti_llm::EmbedderClient;
use graphiti_llm::EmbedderConfig;
use graphiti_llm::EmbeddingClient;
use graphiti_llm::EmbeddingProvider;
use graphiti_llm::ExtractedEntity;
use graphiti_llm::ExtractedRelationship;
// use graphiti_llm::LLMClient; // 暂时未使用
use graphiti_llm::LLMConfig;
use graphiti_llm::LLMProvider;
use graphiti_llm::MultiLLMClient;
use graphiti_llm::QwenCandleClient;
use graphiti_llm::QwenCandleConfig;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::DefaultMakeSpan;
use tower_http::trace::TraceLayer;
use tracing::error;
use tracing::info;
use tracing::warn;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use uuid::Uuid;
use metrics_exporter_prometheus::PrometheusBuilder;
use metrics_exporter_prometheus::PrometheusHandle;
use std::time::Duration;
use axum_server::tls_rustls::RustlsConfig;
use governor::{Quota, RateLimiter};
use nonzero_ext::nonzero;
use std::num::NonZeroU32;
use std::sync::Arc as StdArc;

// Learning system imports
use learning::ConsoleNotificationChannel;
use learning::LearningConfig;
use learning::LearningDetector;
use learning::LearningNotification;
use learning::MCPNotificationChannel;
use learning::NotificationManager;
use learning::SmartLearningDetector;
use learning_endpoints::dismiss_notification;
use learning_endpoints::get_active_notifications;
use learning_endpoints::learning_notifications_stream;
use learning_integration::LearningAwareGraphitiService;
use project_scanner::ProjectScanner;
use tokio::sync::broadcast;

/// MCP server command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port to listen on (only for HTTP mode)
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Host to bind to (only for HTTP mode)
    #[arg(short = 'H', long, default_value = "0.0.0.0")]
    host: String,

    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Run as stdio MCP server (default mode for Claude Desktop)
    #[arg(long, default_value = "false")]
    stdio: bool,

    /// Project directory (for project-specific isolation, similar to Serena's --project)
    #[arg(long, env = "GRAPHITI_PROJECT")]
    project: Option<PathBuf>,

    /// Database path override (for project-specific databases)
    #[arg(long, env = "GRAPHITI_DB_PATH")]
    db_path: Option<PathBuf>,

    /// Data directory override (for project-specific data)
    #[arg(long, env = "GRAPHITI_DATA_DIR")]
    data_dir: Option<PathBuf>,
}

/// Server settings
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ServerSettings {
    pub host: String,
    pub port: u16,
    /// Enable TLS with provided certificates
    #[serde(default)]
    pub tls: Option<TlsSettings>,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            tls: None,
        }
    }
}

/// TLS settings
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TlsSettings {
    /// Path to PEM-encoded certificate chain
    cert_path: String,
    /// Path to PEM-encoded private key
    key_path: String,
}

/// Simple LLM configuration that matches the TOML structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimpleLLMConfig {
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
struct ServerConfig {
    /// Server settings
    #[serde(default)]
    server: ServerSettings,
    /// CozoDB configuration
    #[serde(default)]
    cozo: CozoConfig,
    /// LLM configuration
    llm: SimpleLLMConfig,
    /// Embedder configuration
    embedder: EmbedderConfig,
    /// Graphiti configuration
    #[serde(default)]
    graphiti: GraphitiConfig,
}

/// Application state
#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
    graphiti: Arc<dyn GraphitiService>,
    #[allow(dead_code)]
    /// Learning detector for pattern recognition
    learning_detector: Arc<dyn LearningDetector>,
    #[allow(dead_code)]
    notification_manager: Arc<NotificationManager>,
    notification_receiver:
        Arc<tokio::sync::RwLock<Option<broadcast::Receiver<LearningNotification>>>>,
    project_scanner: Arc<ProjectScanner>,
    rate_limiter: StdArc<RateLimiter>,
    auth_token: Option<String>,
}

/// Trait for Graphiti service operations
#[async_trait::async_trait]
trait GraphitiService: Send + Sync {
    async fn add_memory(&self, req: AddMemoryRequest) -> GraphitiResult<AddMemoryResponse>;
    async fn search_memory(&self, req: SearchMemoryRequest)
        -> GraphitiResult<SearchMemoryResponse>;
    async fn get_memory(&self, id: Uuid) -> GraphitiResult<Option<MemoryNode>>;
    async fn get_related(&self, id: Uuid, depth: usize) -> GraphitiResult<Vec<RelatedMemory>>;
    /// Search extracted facts/relationships (minimal implementation for MCP parity)
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
struct AddCodeEntityRequest {
    /// Entity type
    entity_type: String,
    /// Entity name
    name: String,
    /// Description
    description: String,
    /// File path (optional)
    file_path: Option<String>,
    /// Line range (optional)
    line_range: Option<(u32, u32)>,
    /// Programming language (optional)
    language: Option<String>,
    /// Framework (optional)
    framework: Option<String>,
    /// Complexity rating 1-10 (optional)
    complexity: Option<u8>,
    /// Importance rating 1-10 (optional)
    importance: Option<u8>,
}

/// Request to record development activity
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecordActivityRequest {
    /// Activity type
    activity_type: String,
    /// Title
    title: String,
    /// Description
    description: String,
    /// Developer name
    developer: String,
    /// Project name
    project: String,
    /// Related entity IDs (optional)
    related_entities: Option<Vec<String>>,
    /// Duration in minutes (optional)
    duration_minutes: Option<u32>,
    /// Difficulty rating 1-10 (optional)
    difficulty: Option<u8>,
    /// Quality rating 1-10 (optional)
    quality: Option<u8>,
}

/// Request to search code entities
#[derive(Debug, Serialize, Deserialize)]
struct SearchCodeRequest {
    /// Search query
    query: String,
    /// Entity type filter (optional)
    entity_type: Option<String>,
    /// Language filter (optional)
    language: Option<String>,
    /// Framework filter (optional)
    framework: Option<String>,
    /// Maximum results
    limit: Option<u32>,
}

/// Request to batch add code entities
#[derive(Debug, Serialize, Deserialize)]
struct BatchAddCodeEntitiesRequest {
    /// List of code entities to add
    entities: Vec<AddCodeEntityRequest>,
}

/// Request to batch record activities
#[derive(Debug, Serialize, Deserialize)]
struct BatchRecordActivitiesRequest {
    /// List of activities to record
    activities: Vec<RecordActivityRequest>,
}

/// Request for intelligent context suggestions
#[derive(Debug, Serialize, Deserialize)]
struct ContextSuggestionRequest {
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
struct AddMemoryRequest {
    /// Content of the memory
    content: String,
    /// Optional name/title
    name: Option<String>,
    /// Source of the memory
    source: Option<String>,
    /// Type of memory
    memory_type: Option<String>,
    /// Additional metadata
    metadata: Option<serde_json::Value>,
    /// Optional group/namespace id (not yet persisted)
    group_id: Option<String>,
    /// When the event occurred
    timestamp: Option<String>,
}

/// Response from adding a memory
#[derive(Debug, Serialize, Deserialize)]
struct AddMemoryResponse {
    /// ID of the created memory
    id: Uuid,
    /// Extracted entities
    entities: Vec<SimpleExtractedEntity>,
    /// Extracted relationships
    relationships: Vec<SimpleExtractedRelationship>,
}

/// Request to search memories
#[derive(Debug, Serialize, Deserialize)]
struct SearchMemoryRequest {
    /// Search query
    query: String,
    /// Maximum results
    limit: Option<usize>,
    /// Entity type filter
    entity_types: Option<Vec<String>>,
    /// Time range filter (ISO 8601)
    start_time: Option<String>,
    end_time: Option<String>,
}

/// Response for add_code_entity
#[derive(Debug, Serialize, Deserialize)]
struct AddCodeEntityResponse {
    id: Uuid,
    message: String,
}

/// Response for record_activity
#[derive(Debug, Serialize, Deserialize)]
struct RecordActivityResponse {
    id: Uuid,
    message: String,
}

/// Response for search_code
#[derive(Debug, Serialize, Deserialize)]
struct SearchCodeResponse {
    results: Vec<CodeEntity>,
    total: usize,
}

/// Response for batch add code entities
#[derive(Debug, Serialize, Deserialize)]
struct BatchAddCodeEntitiesResponse {
    results: Vec<AddCodeEntityResponse>,
    successful_count: usize,
    failed_count: usize,
    errors: Vec<String>,
}

/// Response for batch record activities
#[derive(Debug, Serialize, Deserialize)]
struct BatchRecordActivitiesResponse {
    results: Vec<RecordActivityResponse>,
    successful_count: usize,
    failed_count: usize,
    errors: Vec<String>,
}

/// A context suggestion
#[derive(Debug, Serialize, Deserialize)]
struct ContextSuggestion {
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
struct ContextSuggestionResponse {
    suggestions: Vec<ContextSuggestion>,
    total: usize,
}

/// Response from searching memories
#[derive(Debug, Serialize, Deserialize)]
struct SearchMemoryResponse {
    /// Search results
    results: Vec<SearchResult>,
    /// Total results found
    total: usize,
}

/// Search result
#[derive(Debug, Serialize, Deserialize)]
struct SearchResult {
    /// Node ID
    id: Uuid,
    /// Node type
    node_type: String,
    /// Node name/title
    name: String,
    /// Content preview
    content_preview: Option<String>,
    /// Relevance score
    score: f32,
    /// Timestamp
    timestamp: String,
}

/// Memory node representation
#[derive(Debug, Serialize, Deserialize)]
struct MemoryNode {
    /// Node ID
    id: Uuid,
    /// Node type
    node_type: String,
    /// Node name
    name: String,
    /// Node content (if applicable)
    content: Option<String>,
    /// Creation time
    created_at: String,
    /// Event time
    event_time: String,
    /// Additional properties
    properties: serde_json::Value,
}

/// Related memory
#[derive(Debug, Serialize, Deserialize)]
struct RelatedMemory {
    /// The related node
    node: MemoryNode,
    /// Relationship type
    relationship: String,
    /// Distance from source
    distance: usize,
}

/// Extracted entity (simplified for API response)
#[derive(Debug, Serialize, Deserialize)]
struct SimpleExtractedEntity {
    name: String,
    entity_type: String,
    confidence: f32,
}

/// Extracted relationship (simplified for API response)
#[derive(Debug, Serialize, Deserialize, Clone)]
struct SimpleExtractedRelationship {
    source: String,
    target: String,
    relationship: String,
    confidence: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize tracing
    init_tracing(&args.log_level)?;

    // Load configuration with project isolation support
    let config = load_config(&args)?;

    // Initialize services
    info!("Initializing Graphiti services...");
    let (
        graphiti_service,
        learning_detector,
        notification_manager,
        notification_receiver,
        project_scanner,
    ) = initialize_services(config.clone()).await?;

    // Create application state
    let state = AppState {
        graphiti: graphiti_service.clone(),
        learning_detector,
        notification_manager,
        notification_receiver: Arc::new(tokio::sync::RwLock::new(Some(notification_receiver))),
        project_scanner: project_scanner.clone(),
        rate_limiter: StdArc::new(RateLimiter::direct(Quota::per_second(nonzero!(50u32)))) ,
        auth_token: std::env::var("GRAPHITI_AUTH_TOKEN").ok(),
    };

    // Perform initial project scan if in current directory
    if let Ok(current_dir) = std::env::current_dir() {
        if project_scanner.needs_scan(&current_dir).await {
            info!("🔍 Performing initial project scan...");
            tokio::spawn({
                let scanner = project_scanner.clone();
                let _service = graphiti_service.clone();
                async move {
                    match scanner.scan_project(&current_dir).await {
                        Ok(result) => {
                            info!(
                                "✅ Initial project scan completed: {} files, {} entities, {} memories",
                                result.files_scanned, result.entities_added, result.memories_added
                            );
                            scanner.mark_scanned(&current_dir).await;
                        }
                        Err(e) => {
                            warn!("⚠️ Initial project scan failed: {}", e);
                        }
                    }
                }
            });
        } else {
            info!("📁 Project already scanned, skipping initial scan");
        }
    }

    // Build router
        let app = Router::new()
        .route("/mcp", post(mcp_handler)) // Use standard MCP handler with all tools
        .route("/notifications", get(learning_notifications_stream)) // Learning notifications stream
        .route("/notifications/active", get(get_active_notifications)) // Get active notifications
        .route("/notifications/:id/dismiss", post(dismiss_notification)) // Dismiss notification
        .route("/health", get(health_check))
        .route("/ready", get(ready_check))
        // Protected write endpoints
        .route("/memory", post(add_memory))
            .route_layer(middleware::from_fn_with_state(state.clone(), auth_guard))
            .route("/memory/search", get(search_memory))
            .route("/memory/search", post(search_memory_json))
        .route("/memory/:id", get(get_memory))
        .route("/memory/:id/related", get(get_related))
        .route("/metrics", get(metrics))
        .layer(TraceLayer::new_for_http().make_span_with(DefaultMakeSpan::default()))
        .layer(build_cors_layer())
        .layer(RequestBodyLimitLayer::new(1 * 1024 * 1024))
        .layer(TimeoutLayer::new(Duration::from_secs(30)))
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_guard))
        .with_state(state.clone());

    // Check if running in stdio mode
    if args.stdio {
        info!("Starting MCP server in stdio mode");

        // Run stdio MCP server
        use tokio::io::stdin;
        use tokio::io::stdout;
        use tokio::io::AsyncBufReadExt;
        use tokio::io::AsyncWriteExt;
        use tokio::io::BufReader;

        let stdin = stdin();
        let mut stdout = stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break, // EOF
                Ok(_) => {
                    // Process MCP request using existing handler
                    if let Ok(request) = serde_json::from_str::<serde_json::Value>(&line) {
                        match mcp_handler(State(state.clone()), Json(request)).await {
                            Ok(Json(response)) => {
                                let response_json = serde_json::to_string(&response)?;
                                stdout.write_all(response_json.as_bytes()).await?;
                                stdout.write_all(b"\n").await?;
                                stdout.flush().await?;
                            }
                            Err(e) => {
                                let error_response = serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "error": {"code": -1, "message": format!("Error: {:?}", e)}
                                });
                                let response_json = serde_json::to_string(&error_response)?;
                                stdout.write_all(response_json.as_bytes()).await?;
                                stdout.write_all(b"\n").await?;
                                stdout.flush().await?;
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error reading from stdin: {}", e);
                    break;
                }
            }
        }
    } else {
        // Start HTTP/TLS server - use config values if not overridden by command line
        let host = if args.host == "0.0.0.0" {
            &config.server.host
        } else {
            &args.host
        };
        let port = if args.port == 8080 {
            config.server.port
        } else {
            args.port
        };
        let addr = SocketAddr::from((host.parse::<std::net::IpAddr>()?, port));

        // Initialize Prometheus exporter
        let recorder_handle = init_metrics()?;
        info!("/metrics enabled");

        // TLS optional
        if let Some(tls) = &config.server.tls {
            info!("Starting MCP server with TLS on {}", addr);
            let tls_config = RustlsConfig::from_pem_file(&tls.cert_path, &tls.key_path).await
                .map_err(|e| anyhow::anyhow!("Failed to load TLS certs: {}", e))?;
            axum_server::bind_rustls(addr, tls_config)
                .serve(app.into_make_service())
                .await?;
        } else {
            info!("Starting MCP server on {}", addr);
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            axum::serve(listener, app).await?;
        }
    }

    Ok(())
}

/// Initialize tracing
fn init_tracing(level: &str) -> anyhow::Result<()> {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();

    Ok(())
}

/// Load configuration from file with project isolation support
fn load_config(args: &Args) -> anyhow::Result<ServerConfig> {
    let config_path = resolve_config_path(args)?;

    let content = std::fs::read_to_string(&config_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read config file {}: {}",
            config_path.display(),
            e
        )
    })?;

    let mut config: ServerConfig = toml::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))?;

    // Apply project-specific overrides
    apply_project_overrides(&mut config, args)?;

    info!("Loaded configuration from: {}", config_path.display());
    info!("Using LLM provider: {}", config.llm.provider);
    info!("Using embedding provider: {:?}", config.embedder.provider);
    info!("Database path: {}", config.cozo.path);

    Ok(config)
}

/// Resolve configuration file path with project isolation
fn resolve_config_path(args: &Args) -> anyhow::Result<PathBuf> {
    // Determine the project directory (similar to Serena's --project)
    let project_dir = if let Some(project) = &args.project {
        project.clone()
    } else {
        // Use current working directory if no project specified
        std::env::current_dir()
            .map_err(|e| anyhow::anyhow!("Failed to get current directory: {}", e))?
    };

    let project_config = project_dir.join(".graphiti").join("config.toml");
    if project_config.exists() {
        info!(
            "Using project-specific config: {}",
            project_config.display()
        );
        return Ok(project_config);
    }

    // Create project-specific config directory if it doesn't exist
    let graphiti_dir = project_dir.join(".graphiti");
    if !graphiti_dir.exists() {
        std::fs::create_dir_all(&graphiti_dir)
            .map_err(|e| anyhow::anyhow!("Failed to create .graphiti directory: {}", e))?;
        info!(
            "Created project .graphiti directory: {}",
            graphiti_dir.display()
        );
    }

    // Copy default config if project config doesn't exist
    let template_path = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.join("project-config-template.toml")))
        .unwrap_or_else(|| PathBuf::from("project-config-template.toml"));

    let source_config = if template_path.exists() {
        template_path
    } else if args.config.exists() {
        args.config.clone()
    } else {
        // Create a minimal default config
        create_default_project_config(&project_config)?;
        info!(
            "Created default project config: {}",
            project_config.display()
        );
        return Ok(project_config);
    };

    std::fs::copy(&source_config, &project_config)
        .map_err(|e| anyhow::anyhow!("Failed to copy config to project directory: {}", e))?;
    info!(
        "Created project-specific config from: {}",
        source_config.display()
    );
    Ok(project_config)
}

/// Apply project-specific configuration overrides
fn apply_project_overrides(config: &mut ServerConfig, args: &Args) -> anyhow::Result<()> {
    // Override database path if specified
    if let Some(db_path) = &args.db_path {
        config.cozo.path = db_path.to_string_lossy().to_string();
        info!("Database path overridden to: {}", config.cozo.path);
    } else {
        // Determine the project directory (similar to Serena's --project)
        let project_dir = if let Some(project) = &args.project {
            project.clone()
        } else {
            // Use current working directory if no project specified
            std::env::current_dir()
                .map_err(|e| anyhow::anyhow!("Failed to get current directory: {}", e))?
        };

        // Use project-specific database path
        let project_db = project_dir
            .join(".graphiti")
            .join("data")
            .join("graphiti.db");

        // Ensure data directory exists
        if let Some(parent) = project_db.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("Failed to create data directory: {}", e))?;
        }

        config.cozo.path = project_db.to_string_lossy().to_string();
        info!("Using project-specific database: {}", config.cozo.path);
        info!("Project directory: {}", project_dir.display());
    }

    // Override data directory if specified
    if let Some(data_dir) = &args.data_dir {
        // This would be used for other data files like logs, cache, etc.
        std::fs::create_dir_all(data_dir)
            .map_err(|e| anyhow::anyhow!("Failed to create data directory: {}", e))?;
        info!("Data directory set to: {}", data_dir.display());
    }

    Ok(())
}

/// Create a minimal default project configuration
fn create_default_project_config(config_path: &PathBuf) -> anyhow::Result<()> {
    let default_config = r#"# Graphiti MCP Server Project Configuration
# Auto-generated default configuration

[server]
host = "127.0.0.1"
port = 8080
max_connections = 100

[cozo]
engine = "mem"
path = ""
options = {}

[llm]
provider = "openai"
model = "gpt-4"
api_key = ""
temperature = 0.7
max_tokens = 2048

[embedder]
provider = "local"  # use lightweight default to avoid heavy model downloads
model = "text-embedding-3-small"
device = "auto"
batch_size = 32

[graphiti]
max_episode_length = 1000
max_memories_per_search = 50
similarity_threshold = 0.7
learning_enabled = true
auto_scan_enabled = true
scan_interval_minutes = 30
"#;

    std::fs::write(config_path, default_config)
        .map_err(|e| anyhow::anyhow!("Failed to create default config: {}", e))?;

    Ok(())
}

/// Initialize all services with learning capabilities
async fn initialize_services(
    config: ServerConfig,
) -> anyhow::Result<(
    Arc<dyn GraphitiService>,
    Arc<dyn LearningDetector>,
    Arc<NotificationManager>,
    broadcast::Receiver<LearningNotification>,
    Arc<ProjectScanner>,
)> {
    info!(
        "Initializing services with LLM provider: {}",
        config.llm.provider
    );
    info!("Model: {}", config.llm.model);
    info!(
        "Using CozoDB with engine: {} at path: {}",
        config.cozo.engine, config.cozo.path
    );

    // Initialize CozoDB driver
    let cozo_config: CoreCozoConfig = config.cozo.into();
    let storage = CozoDriver::new(cozo_config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize CozoDB: {}", e))?;

    // Convert SimpleLLMConfig to LLMConfig
    let llm_config: LLMConfig = config.llm.into();

    // Create base Graphiti service
    let base_service = RealGraphitiService::new(
        Arc::new(storage),
        llm_config,
        config.embedder,
        config.graphiti,
    )
    .await?;

    // Initialize learning system
    info!("🧠 Initializing learning detection system...");

    // Load learning configuration (use default if not found)
    let learning_config = LearningConfig::default();

    // Create learning detector
    let detector = SmartLearningDetector::new(learning_config.detector);
    let detector_arc: Arc<dyn LearningDetector> = Arc::new(detector);

    // Create notification manager
    let notification_manager = Arc::new(NotificationManager::new(learning_config.notifications));

    // Create notification channels
    let (mcp_channel, notification_receiver) = MCPNotificationChannel::new();
    let console_channel = ConsoleNotificationChannel;

    // Add channels to notification manager
    notification_manager
        .add_channel(Box::new(mcp_channel))
        .await;
    notification_manager
        .add_channel(Box::new(console_channel))
        .await;

    // Wrap base service with learning awareness
    let learning_aware_service = Arc::new(LearningAwareGraphitiService::new(
        Arc::new(base_service),
        detector_arc.clone(),
        notification_manager.clone(),
    ));

    // Initialize project scanner
    info!("🔍 Initializing project scanner...");
    let project_scanner = Arc::new(ProjectScanner::new(
        learning_aware_service.clone() as Arc<dyn GraphitiService>
    ));

    info!("✅ Learning system initialized successfully");

    Ok((
        learning_aware_service,
        detector_arc,
        notification_manager,
        notification_receiver,
        project_scanner,
    ))
}

/// MCP protocol handler
pub async fn mcp_handler(
    State(state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    info!("MCP request: {}", request);

    // Basic MCP protocol implementation
    if let Some(method) = request.get("method").and_then(|m| m.as_str()) {
        match method {
            "initialize" => {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {
                                "listChanged": false
                            },
                            "resources": {
                                "subscribe": false,
                                "listChanged": false
                            },
                            "prompts": {
                                "listChanged": false
                            }
                        },
                        "serverInfo": {
                            "name": "Graphiti Knowledge Graph",
                            "version": "1.0.0"
                        }
                    }
                });
                Ok(Json(response))
            }
            "initialized" => {
                // This is a notification, no response needed
                info!("MCP client initialized");
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": null
                });
                Ok(Json(response))
            }
            "tools/list" => {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "tools": [
                            {
                                "name": "add_memory",
                                "description": "Add a new memory to the knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string", "description": "The content to remember"},
                                        "name": {"type": "string", "description": "Optional name/title"},
                                        "source": {"type": "string", "description": "The source of the memory"},
                                        "memory_type": {"type": "string", "description": "Optional memory type"},
                                        "group_id": {"type": "string", "description": "Optional group/namespace id"},
                                        "metadata": {"type": "object", "description": "Optional metadata object"},
                                        "timestamp": {"type": "string", "description": "Event time (ISO 8601)"}
                                    },
                                    "required": ["content"]
                                }
                            },
                            {
                                "name": "search_memory",
                                "description": "Search for memories in the knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search query"},
                                        "limit": {"type": "integer", "description": "Maximum results", "default": 10},
                                        "entity_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by entity types (optional)"},
                                        "start_time": {"type": "string", "description": "Start time (ISO 8601) (optional)"},
                                        "end_time": {"type": "string", "description": "End time (ISO 8601) (optional)"}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "search_memory_nodes",
                                "description": "Search for node summaries in the knowledge graph (alias)",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search query"},
                                        "max_nodes": {"type": "integer", "description": "Maximum results", "default": 10}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "search_memory_facts",
                                "description": "Search for facts (relationships) in the knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search query"},
                                        "limit": {"type": "integer", "description": "Maximum results", "default": 10}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "delete_episode",
                                "description": "Delete an episode by UUID (best-effort)",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "uuid": {"type": "string", "description": "Episode UUID"}
                                    },
                                    "required": ["uuid"]
                                }
                            },
                            {
                                "name": "get_episodes",
                                "description": "Get most recent episodes",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "last_n": {"type": "integer", "description": "Number of episodes", "default": 10}
                                    },
                                    "required": []
                                }
                            },
                            {
                                "name": "clear_graph",
                                "description": "Clear in-memory graph indexes (minimal parity)",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                            },
                            {
                                "name": "get_entity_edge",
                                "description": "Get a relationship by UUID (not implemented in minimal backend)",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "uuid": {"type": "string", "description": "Edge UUID"}
                                    },
                                    "required": ["uuid"]
                                }
                            },
                            {
                                "name": "delete_entity_edge",
                                "description": "Delete a relationship by UUID (not implemented in minimal backend)",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "uuid": {"type": "string", "description": "Edge UUID"}
                                    },
                                    "required": ["uuid"]
                                }
                            },
                            {
                                "name": "add_code_entity",
                                "description": "Add a code entity (class, function, module, etc.) to the knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "entity_type": {"type": "string", "description": "Type of code entity (Class, Function, Module, Api, etc.)"},
                                        "name": {"type": "string", "description": "Name of the code entity"},
                                        "description": {"type": "string", "description": "Description of the code entity"},
                                        "file_path": {"type": "string", "description": "File path (optional)"},
                                        "line_range": {"type": "array", "items": {"type": "number"}, "description": "Line range [start, end] (optional)"},
                                        "language": {"type": "string", "description": "Programming language (optional)"},
                                        "framework": {"type": "string", "description": "Framework or technology (optional)"},
                                        "complexity": {"type": "number", "description": "Complexity rating 1-10 (optional)"},
                                        "importance": {"type": "number", "description": "Importance rating 1-10 (optional)"}
                                    },
                                    "required": ["entity_type", "name", "description"]
                                }
                            },
                            {
                                "name": "search_code",
                                "description": "Search for code entities in the knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search query"},
                                        "entity_type": {"type": "string", "description": "Filter by entity type (optional)"},
                                        "language": {"type": "string", "description": "Filter by programming language (optional)"},
                                        "framework": {"type": "string", "description": "Filter by framework (optional)"},
                                        "limit": {"type": "integer", "description": "Maximum results", "default": 10}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "record_activity",
                                "description": "Record a development activity (feature development, bug fix, code review, etc.)",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "activity_type": {"type": "string", "description": "Type of activity (Implementation, BugFix, CodeReview, etc.)"},
                                        "title": {"type": "string", "description": "Activity title"},
                                        "description": {"type": "string", "description": "Activity description"},
                                        "developer": {"type": "string", "description": "Developer name"},
                                        "project": {"type": "string", "description": "Project name"},
                                        "related_entities": {"type": "array", "items": {"type": "string"}, "description": "Related entity IDs (optional)"},
                                        "duration_minutes": {"type": "number", "description": "Duration in minutes (optional)"},
                                        "difficulty": {"type": "number", "description": "Difficulty rating 1-10 (optional)"},
                                        "quality": {"type": "number", "description": "Quality rating 1-10 (optional)"}
                                    },
                                    "required": ["activity_type", "title", "description", "developer", "project"]
                                }
                            },
                            {
                                "name": "get_context_suggestions",
                                "description": "Get intelligent context suggestions based on current development work",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "context": {"type": "string", "description": "Current development context or question"},
                                        "current_file": {"type": "string", "description": "Current working file path (optional)"},
                                        "recent_activities": {"type": "array", "items": {"type": "string"}, "description": "Recent development activities (optional)"},
                                        "limit": {"type": "number", "description": "Maximum number of suggestions", "default": 5}
                                    },
                                    "required": ["context"]
                                }
                            },
                            {
                                "name": "batch_add_code_entities",
                                "description": "Batch add multiple code entities to the knowledge graph",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "entities": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "entity_type": {"type": "string", "description": "Type of code entity"},
                                                    "name": {"type": "string", "description": "Name of the code entity"},
                                                    "description": {"type": "string", "description": "Description of the code entity"},
                                                    "file_path": {"type": "string", "description": "File path (optional)"},
                                                    "line_range": {"type": "array", "items": {"type": "number"}, "description": "Line range [start, end] (optional)"},
                                                    "language": {"type": "string", "description": "Programming language (optional)"},
                                                    "framework": {"type": "string", "description": "Framework or technology (optional)"},
                                                    "complexity": {"type": "number", "description": "Complexity rating 1-10 (optional)"},
                                                    "importance": {"type": "number", "description": "Importance rating 1-10 (optional)"}
                                                },
                                                "required": ["entity_type", "name", "description"]
                                            },
                                            "description": "Array of code entities to add"
                                        }
                                    },
                                    "required": ["entities"]
                                }
                            },
                            {
                                "name": "batch_record_activities",
                                "description": "Batch record multiple development activities",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "activities": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "activity_type": {"type": "string", "description": "Type of activity"},
                                                    "title": {"type": "string", "description": "Activity title"},
                                                    "description": {"type": "string", "description": "Activity description"},
                                                    "developer": {"type": "string", "description": "Developer name"},
                                                    "project": {"type": "string", "description": "Project name"},
                                                    "related_entities": {"type": "array", "items": {"type": "string"}, "description": "Related entity IDs (optional)"},
                                                    "duration_minutes": {"type": "number", "description": "Duration in minutes (optional)"},
                                                    "difficulty": {"type": "number", "description": "Difficulty rating 1-10 (optional)"},
                                                    "quality": {"type": "number", "description": "Quality rating 1-10 (optional)"}
                                                },
                                                "required": ["activity_type", "title", "description", "developer", "project"]
                                            },
                                            "description": "Array of activities to record"
                                        }
                                    },
                                    "required": ["activities"]
                                }
                            },
                            {
                                "name": "scan_project",
                                "description": "Scan project structure and extract code entities automatically",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "project_path": {"type": "string", "description": "Path to the project directory (optional, defaults to current directory)"},
                                        "force_rescan": {"type": "boolean", "description": "Force rescan even if recently scanned (optional, defaults to false)"}
                                    },
                                    "required": []
                                }
                            },
                            {
                                "name": "get_related_memories",
                                "description": "Get memories related to a specific memory ID",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string", "description": "Memory ID"},
                                        "depth": {"type": "integer", "description": "Depth of relationship traversal", "default": 1}
                                    },
                                    "required": ["id"]
                                }
                            }
                        ]
                    }
                });
                Ok(Json(response))
            }
            "tools/call" => {
                if let Some(params) = request.get("params") {
                    if let Some(tool_name) = params.get("name").and_then(|n| n.as_str()) {
                        match tool_name {
                            "search_memory_facts" => {
                                // Minimal facts search using extracted relationships
                                if let Some(args) = params.get("arguments") {
                                    let query = args
                                        .get("query")
                                        .and_then(|q| q.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let limit = args.get("limit").and_then(|l| l.as_u64()).map(|v| v as usize);

                                    match state.graphiti.search_memory_facts_json(query, limit).await {
                                        Ok(facts_json) => {
                                            let total = facts_json.len();
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {
                                                    "content": [{
                                                        "type": "json",
                                                        "json": {"facts": facts_json, "total": total}
                                                    }]
                                                }
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to search facts: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            },
                            "add_memory" => {
                                if let Some(args) = params.get("arguments") {
                                    // Accept Python-compatible fields by mapping them into our schema
                                    let mut args_mut = args.clone();
                                    // Map episode_body -> content
                                    if args_mut.get("content").is_none() {
                                        if let Some(episode_body) = args_mut.get("episode_body").and_then(|v| v.as_str()) {
                                            args_mut["content"] = serde_json::json!(episode_body);
                                        }
                                    }
                                    // Map source_description -> metadata.source_description
                                    if let Some(src_desc) = args_mut.get("source_description").and_then(|v| v.as_str()) {
                                        // Build new metadata object without alias field
                                        let mut metadata = args_mut.get("metadata").and_then(|m| m.as_object()).cloned().unwrap_or_default();
                                        metadata.insert("source_description".to_string(), serde_json::json!(src_desc));
                                        args_mut["metadata"] = serde_json::Value::Object(metadata);
                                        // Remove alias to avoid schema parse failure
                                        if let Some(obj) = args_mut.as_object_mut() { obj.remove("source_description"); }
                                    }

                                    let req: AddMemoryRequest =
                                        match serde_json::from_value(args_mut) {
                                            Ok(req) => req,
                                            Err(e) => {
                                                error!("Failed to parse add_memory request: {}", e);
                                                return Err(StatusCode::BAD_REQUEST);
                                            }
                                        };

                                    match state.graphiti.add_memory(req).await {
                                        Ok(result) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {
                                                    "content": [{
                                                        "type": "json",
                                                        "json": result
                                                    }]
                                                }
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to add memory: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            }
                            ,
                            "search_memory_nodes" => {
                                // Alias to search_memory with Python-compatible argument names
                                if let Some(args) = params.get("arguments") {
                                    // Accept {query, max_nodes} or legacy {limit}
                                    let mut mapped = serde_json::json!({});
                                    if let Some(q) = args.get("query") { mapped["query"] = q.clone(); }
                                    if let Some(m) = args.get("max_nodes").or_else(|| args.get("limit")) {
                                        mapped["limit"] = m.clone();
                                    }

                                    let req: SearchMemoryRequest = match serde_json::from_value(mapped) {
                                        Ok(req) => req,
                                        Err(_) => return Err(StatusCode::BAD_REQUEST),
                                    };

                                    match state.graphiti.search_memory(req).await {
                                        Ok(result) => {
                                            // Adapt to Python-style NodeSearchResponse shape
                                            let nodes: Vec<serde_json::Value> = result
                                                .results
                                                .into_iter()
                                                .map(|r| serde_json::json!({
                                                    "uuid": r.id,
                                                    "name": r.name,
                                                    "summary": r.content_preview.unwrap_or_default(),
                                                    "labels": ["Episode"],
                                                    "group_id": "",
                                                    "created_at": r.timestamp,
                                                    "attributes": {}
                                                }))
                                                .collect();

                                            let json_payload = serde_json::json!({
                                                "message": if nodes.is_empty() { "No relevant nodes found" } else { "Nodes retrieved successfully" },
                                                "nodes": nodes
                                            });

                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {"content": [{"type": "json", "json": json_payload}]}
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to search memory (alias): {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            }
                            "search_memory" => {
                                if let Some(args) = params.get("arguments") {
                                    let req: SearchMemoryRequest =
                                        match serde_json::from_value(args.clone()) {
                                            Ok(req) => req,
                                            Err(_) => return Err(StatusCode::BAD_REQUEST),
                                        };

                                    match state.graphiti.search_memory(req).await {
                                        Ok(result) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {
                                                    "content": [{
                                                        "type": "json",
                                                        "json": result
                                                    }]
                                                }
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to search memory: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            }
                            "delete_episode" => {
                                if let Some(args) = params.get("arguments") {
                                    let id_str = args.get("uuid").and_then(|v| v.as_str()).ok_or(StatusCode::BAD_REQUEST)?;
                                    let id = uuid::Uuid::parse_str(id_str).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    match state.graphiti.delete_episode(id).await {
                                        Ok(_) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {"content": [{"type": "text", "text": "Episode deleted"}]}
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to delete episode: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "get_episodes" => {
                                let last_n = params.get("arguments").and_then(|a| a.get("last_n")).and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                                match state.graphiti.get_episodes(last_n).await {
                                    Ok(list) => {
                                        let json_payload = list; // EpisodeNode is Serialize
                                        let response = serde_json::json!({
                                            "jsonrpc": "2.0",
                                            "id": request.get("id"),
                                            "result": {"content": [{"type": "json", "json": json_payload}]}
                                        });
                                        Ok(Json(response))
                                    }
                                    Err(e) => {
                                        error!("Failed to get episodes: {}", e);
                                        Err(StatusCode::INTERNAL_SERVER_ERROR)
                                    }
                                }
                            }
                            "clear_graph" => {
                                match state.graphiti.clear_graph().await {
                                    Ok(_) => {
                                        let response = serde_json::json!({
                                            "jsonrpc": "2.0",
                                            "id": request.get("id"),
                                            "result": {"content": [{"type": "text", "text": "Graph cleared"}]}
                                        });
                                        Ok(Json(response))
                                    }
                                    Err(e) => {
                                        error!("Failed to clear graph: {}", e);
                                        Err(StatusCode::INTERNAL_SERVER_ERROR)
                                    }
                                }
                            }
                            "get_entity_edge" => {
                                if let Some(args) = params.get("arguments") {
                                    let id_str = args.get("uuid").and_then(|v| v.as_str()).ok_or(StatusCode::BAD_REQUEST)?;
                                    let id = uuid::Uuid::parse_str(id_str).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    match state.graphiti.get_entity_edge_json(id).await {
                                        Ok(Some(edge_json)) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {"content": [{"type": "json", "json": edge_json}]}
                                            });
                                            Ok(Json(response))
                                        }
                                        Ok(None) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "error": {"code": -32602, "message": "Entity edge not found"}
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to get entity edge: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "delete_entity_edge" => {
                                if let Some(args) = params.get("arguments") {
                                    let id_str = args.get("uuid").and_then(|v| v.as_str()).ok_or(StatusCode::BAD_REQUEST)?;
                                    let id = uuid::Uuid::parse_str(id_str).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    match state.graphiti.delete_entity_edge_by_uuid(id).await {
                                        Ok(true) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {"content": [{"type": "text", "text": "Entity edge deleted"}]}
                                            });
                                            Ok(Json(response))
                                        }
                                        Ok(false) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "error": {"code": -32602, "message": "Entity edge not found"}
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to delete entity edge: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "add_code_entity" => {
                                if let Some(args) = params.get("arguments") {
                                    let req: AddCodeEntityRequest =
                                        match serde_json::from_value(args.clone()) {
                                            Ok(req) => req,
                                            Err(e) => {
                                                error!(
                                                    "Failed to parse add_code_entity request: {}",
                                                    e
                                                );
                                                return Err(StatusCode::BAD_REQUEST);
                                            }
                                        };

                                    match state.graphiti.add_code_entity(req).await {
                                        Ok(result) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {
                                                    "content": [{
                                                        "type": "text",
                                                        "text": format!("Code entity added successfully with ID: {}", result.id)
                                                    }]
                                                }
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to add code entity: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            }
                            "record_activity" => {
                                if let Some(args) = params.get("arguments") {
                                    let req: RecordActivityRequest =
                                        match serde_json::from_value(args.clone()) {
                                            Ok(req) => req,
                                            Err(e) => {
                                                error!(
                                                    "Failed to parse record_activity request: {}",
                                                    e
                                                );
                                                return Err(StatusCode::BAD_REQUEST);
                                            }
                                        };

                                    match state.graphiti.record_activity(req).await {
                                        Ok(result) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {
                                                    "content": [{
                                                        "type": "text",
                                                        "text": format!("Activity recorded successfully with ID: {}", result.id)
                                                    }]
                                                }
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to record activity: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            }
                            "search_code" => {
                                if let Some(args) = params.get("arguments") {
                                    let req: SearchCodeRequest =
                                        match serde_json::from_value(args.clone()) {
                                            Ok(req) => req,
                                            Err(e) => {
                                                error!(
                                                    "Failed to parse search_code request: {}",
                                                    e
                                                );
                                                return Err(StatusCode::BAD_REQUEST);
                                            }
                                        };

                                    match state.graphiti.search_code(req).await {
                                        Ok(result) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {
                                                    "content": [{
                                                        "type": "json",
                                                        "json": result
                                                    }]
                                                }
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to search code: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            }
                            "batch_add_code_entities" => {
                                if let Some(args) = params.get("arguments") {
                                    let req: BatchAddCodeEntitiesRequest =
                                        match serde_json::from_value(args.clone()) {
                                            Ok(req) => req,
                                            Err(e) => {
                                                error!(
                                                    "Failed to parse batch_add_code_entities request: {}",
                                                    e
                                                );
                                                return Err(StatusCode::BAD_REQUEST);
                                            }
                                        };

                                    match state.graphiti.batch_add_code_entities(req).await {
                                        Ok(result) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {
                                                    "content": [{
                                                        "type": "text",
                                                        "text": format!("Batch added {} code entities successfully ({} succeeded, {} failed)",
                                                                result.results.len(), result.successful_count, result.failed_count)
                                                    }]
                                                }
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to batch add code entities: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            }
                            "batch_record_activities" => {
                                if let Some(args) = params.get("arguments") {
                                    let req: BatchRecordActivitiesRequest =
                                        match serde_json::from_value(args.clone()) {
                                            Ok(req) => req,
                                            Err(e) => {
                                                error!(
                                                    "Failed to parse batch_record_activities request: {}",
                                                    e
                                                );
                                                return Err(StatusCode::BAD_REQUEST);
                                            }
                                        };

                                    match state.graphiti.batch_record_activities(req).await {
                                        Ok(result) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {
                                                    "content": [{
                                                        "type": "text",
                                                        "text": format!("Batch recorded {} activities successfully ({} succeeded, {} failed)",
                                                                result.results.len(), result.successful_count, result.failed_count)
                                                    }]
                                                }
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to batch record activities: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            }
                            "get_context_suggestions" => {
                                if let Some(args) = params.get("arguments") {
                                    let req: ContextSuggestionRequest = match serde_json::from_value(
                                        args.clone(),
                                    ) {
                                        Ok(req) => req,
                                        Err(e) => {
                                            error!(
                                                "Failed to parse get_context_suggestions request: {}",
                                                e
                                            );
                                            return Err(StatusCode::BAD_REQUEST);
                                        }
                                    };

                                    match state.graphiti.get_context_suggestions(req).await {
                                        Ok(result) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {
                                                    "content": [{
                                                        "type": "text",
                                                        "text": format!("Generated {} context suggestions based on your development context", result.total)
                                                    }]
                                                }
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to get context suggestions: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            }
                            "scan_project" => {
                                let args = params
                                    .get("arguments")
                                    .cloned()
                                    .unwrap_or_else(|| serde_json::json!({}));
                                let project_path = args
                                    .get("project_path")
                                    .and_then(|p| p.as_str())
                                    .map(PathBuf::from)
                                    .unwrap_or_else(|| {
                                        std::env::current_dir()
                                            .unwrap_or_else(|_| PathBuf::from("."))
                                    });
                                let force_rescan = args
                                    .get("force_rescan")
                                    .and_then(|f| f.as_bool())
                                    .unwrap_or(false);

                                // Check if scan is needed
                                if !force_rescan
                                    && !state.project_scanner.needs_scan(&project_path).await
                                {
                                    let response = serde_json::json!({
                                        "jsonrpc": "2.0",
                                        "id": request.get("id"),
                                        "result": {
                                            "content": [{
                                                "type": "text",
                                                "text": format!("Project at {} already scanned recently. Use force_rescan=true to rescan.", project_path.display())
                                            }]
                                        }
                                    });
                                    return Ok(Json(response));
                                }

                                match state.project_scanner.scan_project(&project_path).await {
                                    Ok(result) => {
                                        state.project_scanner.mark_scanned(&project_path).await;
                                        let response = serde_json::json!({
                                            "jsonrpc": "2.0",
                                            "id": request.get("id"),
                                            "result": {
                                                "content": [{
                                                    "type": "text",
                                                    "text": format!("Project scan completed successfully!\n\n📊 **Scan Results:**\n- 📁 Files scanned: {}\n- 🧩 Code entities extracted: {}\n- 💭 Memories created: {}\n- 📂 Project: {}\n\nThe knowledge graph has been updated with your project structure and code entities.",
                                                        result.files_scanned, result.entities_added, result.memories_added,
                                                        result.project_info.as_ref().map(|p| p.name.as_str()).unwrap_or("Unknown"))
                                                }]
                                            }
                                        });
                                        Ok(Json(response))
                                    }
                                    Err(e) => {
                                        error!("Failed to scan project: {}", e);
                                        let response = serde_json::json!({
                                            "jsonrpc": "2.0",
                                            "id": request.get("id"),
                                            "result": {
                                                "content": [{
                                                    "type": "text",
                                                    "text": format!("❌ Project scan failed: {}", e)
                                                }]
                                            }
                                        });
                                        Ok(Json(response))
                                    }
                                }
                            }
                            "get_related_memories" => {
                                if let Some(args) = params.get("arguments") {
                                    let memory_id_str = args
                                        .get("id")
                                        .and_then(|id| id.as_str())
                                        .ok_or(StatusCode::BAD_REQUEST)?;
                                    let memory_id = uuid::Uuid::parse_str(memory_id_str)
                                        .map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let depth =
                                        args.get("depth").and_then(|d| d.as_i64()).unwrap_or(1)
                                            as usize;

                                    match state.graphiti.get_related(memory_id, depth).await {
                                        Ok(result) => {
                                            let response = serde_json::json!({
                                                "jsonrpc": "2.0",
                                                "id": request.get("id"),
                                                "result": {
                                                    "content": [{
                                                        "type": "json",
                                                        "json": {"results": result, "total": result.len()}
                                                    }]
                                                }
                                            });
                                            Ok(Json(response))
                                        }
                                        Err(e) => {
                                            error!("Failed to get related memories: {}", e);
                                            Err(StatusCode::INTERNAL_SERVER_ERROR)
                                        }
                                    }
                                } else {
                                    Err(StatusCode::BAD_REQUEST)
                                }
                            }
                            _ => Err(StatusCode::NOT_FOUND),
                        }
                    } else {
                        Err(StatusCode::BAD_REQUEST)
                    }
                } else {
                    Err(StatusCode::BAD_REQUEST)
                }
            }
            _ => {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
                });
                Ok(Json(response))
            }
        }
    } else {
        Err(StatusCode::BAD_REQUEST)
    }
}

/// Health check endpoint
async fn health_check() -> StatusCode {
    // Liveness: always OK if the server loop is alive
    StatusCode::OK
}

/// Readiness probe: ensure storage is reachable
async fn ready_check(State(state): State<AppState>) -> StatusCode {
    // Try a lightweight operation: read episodes list from in-memory index; storage ping could be added
    // If in-memory index is accessible, consider ready; for stricter checks, query storage
    let _ = state.project_scanner.clone();
    StatusCode::OK
}

fn build_cors_layer() -> CorsLayer {
    // Tighten as needed: allow any origin/methods for now with limited headers
    CorsLayer::new()
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_origin(Any)
        .allow_headers(Any)
}

async fn rate_limit_guard<B>(
    State(state): State<AppState>,
    req: Request<B>,
    next: middleware::Next<B>,
) -> Result<Response, StatusCode> {
    state
        .rate_limiter
        .check()
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;
    Ok(next.run(req).await)
}

async fn auth_guard<B>(
    State(state): State<AppState>,
    req: Request<B>,
    next: middleware::Next<B>,
) -> Result<Response, StatusCode> {
    if let Some(expected) = &state.auth_token {
        let headers = req.headers();
        if !is_authorized(headers, expected) {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }
    Ok(next.run(req).await)
}

fn is_authorized(headers: &HeaderMap, expected: &str) -> bool {
    if let Some(val) = headers.get(axum::http::header::AUTHORIZATION) {
        if let Ok(s) = val.to_str() {
            // Support: Bearer <token> or raw token
            let bearer = format!("Bearer {}", expected);
            return s == expected || s == bearer;
        }
    }
    false
}

/// Add memory endpoint
async fn add_memory(
    State(state): State<AppState>,
    Json(req): Json<AddMemoryRequest>,
) -> std::result::Result<Json<AddMemoryResponse>, StatusCode> {
    match state.graphiti.add_memory(req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => {
            error!("Failed to add memory: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Search memory endpoint
async fn search_memory(
    State(state): State<AppState>,
    Query(req): Query<SearchMemoryRequest>,
) -> std::result::Result<Json<SearchMemoryResponse>, StatusCode> {
    match state.graphiti.search_memory(req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => {
            error!("Failed to search memory: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Search memory endpoint (POST JSON body)
async fn search_memory_json(
    State(state): State<AppState>,
    Json(req): Json<SearchMemoryRequest>,
) -> std::result::Result<Json<SearchMemoryResponse>, StatusCode> {
    match state.graphiti.search_memory(req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => {
            error!("Failed to search memory (json): {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Get memory by ID endpoint
async fn get_memory(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<Uuid>,
) -> std::result::Result<Json<MemoryNode>, StatusCode> {
    match state.graphiti.get_memory(id).await {
        Ok(Some(memory)) => Ok(Json(memory)),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            error!("Failed to get memory: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Get related memories endpoint
async fn get_related(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<Uuid>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> std::result::Result<Json<Vec<RelatedMemory>>, StatusCode> {
    let depth = params
        .get("depth")
        .and_then(|d| d.parse().ok())
        .unwrap_or(1);

    match state.graphiti.get_related(id, depth).await {
        Ok(related) => Ok(Json(related)),
        Err(e) => {
            error!("Failed to get related memories: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Metrics endpoint
async fn metrics() -> String {
    METRICS_EXPORTER
        .get()
        .map(|h| h.render())
        .unwrap_or_else(|| "".to_string())
}

static METRICS_EXPORTER: once_cell::sync::OnceCell<PrometheusHandle> = once_cell::sync::OnceCell::new();

fn init_metrics() -> anyhow::Result<()> {
    if METRICS_EXPORTER.get().is_some() {
        return Ok(());
    }
    let builder = PrometheusBuilder::new();
    let handle = builder.install_recorder().map_err(|e| anyhow::anyhow!("{}", e))?;
    METRICS_EXPORTER.set(handle).ok();
    Ok(())
}

/// Real implementation using CozoDB with unified intelligence
struct RealGraphitiService {
    storage: Arc<CozoDriver>,
    llm_client: Option<Arc<MultiLLMClient>>,
    embedder: Arc<dyn EmbeddingClient>,
    ner_extractor: Arc<dyn graphiti_ner::EntityExtractor>,
    config: GraphitiConfig,
    /// Simple in-memory index for episodes to support search/get operations
    memory_index: Arc<RwLock<HashMap<Uuid, EpisodeNode>>>,
    /// In-memory mapping from episode id to extracted entity names
    memory_entities: Arc<RwLock<HashMap<Uuid, Vec<String>>>>,
    /// In-memory collection of extracted relationships records
    memory_relationships: Arc<RwLock<Vec<SimpleExtractedRelationship>>>,
    /// In-memory mapping from synthetic UUID to relationship record (minimal parity)
    memory_relationships_by_id: Arc<RwLock<HashMap<Uuid, SimpleExtractedRelationship>>>,
}

impl RealGraphitiService {
    async fn new(
        storage: Arc<CozoDriver>,
        llm_config: LLMConfig,
        embedder_config: EmbedderConfig,
        config: GraphitiConfig,
    ) -> anyhow::Result<Self> {
        // Create LLM client (optional based on config)
        let llm_client = if !config.skip_entity_extraction {
            match create_llm_client(&llm_config).await {
                Ok(client) => Some(Arc::new(client)),
                Err(e) => {
                    info!("LLM client creation failed: {}. Using rule-based NER.", e);
                    None
                }
            }
        } else {
            None
        };

        // Create embedder client based on provider
        let embedder: Arc<dyn EmbeddingClient> = match embedder_config.provider {
            EmbeddingProvider::QwenCandle => {
                info!("Using Qwen Candle embedder (pure Rust)");
                let qwen_config = QwenCandleConfig {
                    model_repo: embedder_config.model.clone(),
                    device: detect_device(),
                    dimension: embedder_config.dimension,
                    batch_size: embedder_config.batch_size,
                    max_length: 512,
                    normalize: true,
                    revision: "main".to_string(),
                    cache_dir: Some(std::path::PathBuf::from("./Qwen3-Embedding-0.6B")),
                    dtype: graphiti_llm::candle_core::DType::F32,
                };
                Arc::new(QwenCandleClient::new(qwen_config)?)
            }
            EmbeddingProvider::Qwen3EmbedAnything => {
                info!("🚀 Using Qwen3-Embedding with embed_anything (recommended)");
                let qwen3_config = Qwen3EmbedAnythingConfig {
                    model_id: embedder_config.model.clone(),
                    batch_size: embedder_config.batch_size,
                    max_length: 8192,           // Qwen3 supports up to 32K
                    device: "auto".to_string(), // Auto-detect best device
                };
                Arc::new(Qwen3EmbedAnythingClient::new(qwen3_config).await?)
            }
            _ => {
                info!("Using generic embedder client");
                Arc::new(EmbedderClient::new(embedder_config)?)
            }
        };

        // Create NER extractor with Candle support
        let ner_extractor: Arc<dyn graphiti_ner::EntityExtractor> = {
            // Create a wrapper that implements the required trait
            struct EmbedderWrapper(Arc<dyn graphiti_llm::EmbeddingClient>);

            #[async_trait::async_trait]
            impl graphiti_llm::EmbeddingClient for EmbedderWrapper {
                async fn embed_batch(
                    &self,
                    texts: &[String],
                ) -> graphiti_core::error::Result<Vec<Vec<f32>>> {
                    self.0.embed_batch(texts).await
                }

                async fn embed(&self, text: &str) -> graphiti_core::error::Result<Vec<f32>> {
                    self.0.embed(text).await
                }
            }

            // Create Candle-based NER extractor using the wrapper
            let candle_ner =
                graphiti_ner::CandleNerExtractor::new(Box::new(EmbedderWrapper(embedder.clone())))
                    .with_similarity_threshold(0.6);

            // Create hybrid configuration favoring Candle NER
            let config = graphiti_ner::HybridConfig {
                use_rust_bert: false, // 不使用 rust-bert (避免 PyTorch 依赖)
                use_candle_ner: true, // 使用 Candle NER
                use_rule_fallback: true,
                min_confidence: 0.5,
                merge_overlapping: true,
                max_overlap_ratio: 0.7,
            };

            match graphiti_ner::HybridExtractor::new_with_candle(config, Some(candle_ner)).await {
                Ok(hybrid) => {
                    tracing::info!("Using hybrid NER extractor (Candle + rules)");
                    Arc::new(hybrid)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to initialize Candle NER extractor: {}, falling back to rule-based",
                        e
                    );
                    Arc::new(graphiti_ner::RuleBasedExtractor::default())
                }
            }
        };

        Ok(Self {
            storage,
            llm_client,
            embedder,
            ner_extractor,
            config,
            memory_index: Arc::new(RwLock::new(HashMap::new())),
            memory_entities: Arc::new(RwLock::new(HashMap::new())),
            memory_relationships: Arc::new(RwLock::new(Vec::new())),
            memory_relationships_by_id: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn relationship_to_json(r: &SimpleExtractedRelationship) -> serde_json::Value {
        serde_json::json!({
            "source": r.source,
            "target": r.target,
            "relationship": r.relationship,
            "confidence": r.confidence
        })
    }

    /// Extract entities using rule-based NER
    async fn extract_with_ner(
        &self,
        text: &str,
    ) -> GraphitiResult<(
        Vec<ExtractedEntity>,
        Vec<ExtractedRelationship>,
        Option<String>,
    )> {
        let ner_entities = self.ner_extractor.extract(text).await?;

        // Convert NER entities to LLM-style entities
        let entities: Vec<ExtractedEntity> = ner_entities
            .into_iter()
            .map(|e| ExtractedEntity {
                name: e.text,
                entity_type: e.label.as_str().to_string(),
                confidence: e.score,
                attributes: {
                    let mut attrs = std::collections::HashMap::new();
                    attrs.insert("start".to_string(), serde_json::json!(e.start));
                    attrs.insert("end".to_string(), serde_json::json!(e.end));
                    attrs.insert(
                        "extraction_method".to_string(),
                        serde_json::json!("rule_based"),
                    );
                    attrs
                },
                span: Some((e.start, e.end)),
            })
            .collect();

        // Simple relationship inference based on proximity
        let mut relationships = Vec::new();

        // If we have multiple entities in the same sentence, infer relationships
        let sentences: Vec<&str> = text.split(&['.', '!', '?', '。', '！', '？'][..]).collect();
        for sentence in sentences {
            let sentence_entities: Vec<_> = entities
                .iter()
                .filter(|e| {
                    e.span
                        .map(|(start, _)| text[..start].contains(sentence))
                        .unwrap_or(false)
                })
                .collect();

            // Create relationships between entities in the same sentence
            for i in 0..sentence_entities.len() {
                for j in i + 1..sentence_entities.len() {
                    relationships.push(ExtractedRelationship {
                        source: sentence_entities[i].name.clone(),
                        target: sentence_entities[j].name.clone(),
                        relationship: "RELATED_TO".to_string(),
                        confidence: 0.6,
                        attributes: {
                            let mut attrs = std::collections::HashMap::new();
                            attrs.insert("inferred".to_string(), serde_json::json!(true));
                            attrs.insert("method".to_string(), serde_json::json!("proximity"));
                            attrs
                        },
                    });
                }
            }
        }

        // Simple summary generation
        let summary = if entities.is_empty() {
            None
        } else {
            Some(format!(
                "Text containing {} entities: {}",
                entities.len(),
                entities
                    .iter()
                    .take(3)
                    .map(|e| e.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        };

        Ok((entities, relationships, summary))
    }
}

#[async_trait::async_trait]
impl GraphitiService for RealGraphitiService {
    async fn add_memory(&self, req: AddMemoryRequest) -> GraphitiResult<AddMemoryResponse> {
        info!(
            "Adding memory: {:?}",
            req.name.as_deref().unwrap_or("unnamed")
        );

        // Create episode using the storage layer directly
        let episode_id = Uuid::new_v4();
        // Determine event time (valid_from); fall back to now if parsing fails or missing
        let now = chrono::Utc::now();
        let event_time = if let Some(ts) = &req.timestamp {
            match chrono::DateTime::parse_from_rfc3339(ts) {
                Ok(dt) => dt.with_timezone(&chrono::Utc),
                Err(_) => now,
            }
        } else {
            now
        };

        // Create episode node
        let episode = EpisodeNode {
            id: episode_id,
            name: req.name.unwrap_or_else(|| "Memory".to_string()),
            episode_type: EpisodeType::Message,
            content: req.content.clone(),
            source: req.source.unwrap_or_else(|| "user".to_string()),
            temporal: TemporalMetadata {
                created_at: now,
                valid_from: event_time,
                valid_to: None,
                expired_at: None,
            },
            embedding: None, // Will be generated if needed
        };

        // Store the episode
        self.storage.create_node(&episode).await?;

        info!("Successfully added memory with episode ID: {}", episode_id);

        // Update in-memory index (best-effort)
        {
            let mut idx = self.memory_index.write().await;
            idx.insert(episode_id, episode.clone());
        }

        // Perform lightweight entity and relationship extraction (best-effort)
        let (entities, relationships) = match self.extract_with_ner(&req.content).await {
            Ok((entities, relationships, _summary)) => (entities, relationships),
            Err(e) => {
                tracing::warn!("NER extraction failed (continuing without entities): {}", e);
                (Vec::new(), Vec::new())
            }
        };

        let simple_entities: Vec<SimpleExtractedEntity> = entities
            .into_iter()
            .map(|e| SimpleExtractedEntity {
                name: e.name,
                entity_type: e.entity_type,
                confidence: e.confidence,
            })
            .collect();

        let simple_relationships: Vec<SimpleExtractedRelationship> = relationships
            .into_iter()
            .map(|r| SimpleExtractedRelationship {
                source: r.source,
                target: r.target,
                relationship: r.relationship,
                confidence: r.confidence,
            })
            .collect();

        // Update entity/relationship in-memory indexes (best-effort)
        {
            let mut ents = self.memory_entities.write().await;
            ents.insert(
                episode_id,
                simple_entities
                    .iter()
                    .map(|e| e.name.clone())
                    .collect::<Vec<_>>(),
            );
        }
        {
            let mut rels = self.memory_relationships.write().await;
            rels.extend(simple_relationships.iter().cloned());
        }
        {
            let mut rel_map = self.memory_relationships_by_id.write().await;
            for r in &simple_relationships {
                let id = Uuid::new_v4();
                rel_map.insert(id, r.clone());
            }
        }

        Ok(AddMemoryResponse {
            id: episode_id,
            entities: simple_entities,
            relationships: simple_relationships,
        })
    }

    async fn search_memory(
        &self,
        req: SearchMemoryRequest,
    ) -> GraphitiResult<SearchMemoryResponse> {
        info!("Searching memory: {}", req.query);

        let limit = req.limit.unwrap_or(10) as usize;
        let query_lower = req.query.to_lowercase();

        let index_snapshot: Vec<EpisodeNode> = {
            let idx = self.memory_index.read().await;
            idx.values().cloned().collect()
        };

        // Naive substring search with simple scoring by match length
        let mut scored: Vec<(f32, &EpisodeNode)> = index_snapshot
            .iter()
            .filter_map(|ep| {
                let content_lower = ep.content.to_lowercase();
                if content_lower.contains(&query_lower) {
                    // score by proportion of query length over content length
                    let score = (req.query.len() as f32) / (ep.content.len().max(1) as f32);
                    Some((score, ep))
                } else if ep.name.to_lowercase().contains(&query_lower) {
                    Some((0.5, ep))
                } else {
                    None
                }
            })
            .collect();

        // sort by score desc, then created_at desc, then id asc
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.temporal.created_at.cmp(&a.1.temporal.created_at))
                .then_with(|| a.1.id.cmp(&b.1.id))
        });

        let total = scored.len();

        let results: Vec<SearchResult> = scored
            .into_iter()
            .take(limit)
            .map(|(score, ep)| SearchResult {
                id: ep.id,
                node_type: "Episode".to_string(),
                name: ep.name.clone(),
                content_preview: Some(ep.content.chars().take(200).collect()),
                score,
                timestamp: ep.temporal.valid_from.to_rfc3339(),
            })
            .collect();

        info!("Search completed, found {} results ({} returned)", total, results.len());

        Ok(SearchMemoryResponse { results, total })
    }

    async fn get_memory(&self, id: Uuid) -> GraphitiResult<Option<MemoryNode>> {
        info!("Getting memory: {}", id);

        // Try in-memory index first
        if let Some(ep) = self.memory_index.read().await.get(&id).cloned() {
            let node = MemoryNode {
                id: ep.id,
                node_type: "Episode".to_string(),
                name: ep.name,
                content: Some(ep.content),
                created_at: ep.temporal.created_at.to_rfc3339(),
                event_time: ep.temporal.valid_from.to_rfc3339(),
                properties: serde_json::json!({
                    "source": ep.source,
                }),
            };
            return Ok(Some(node));
        }

        // Fallback to storage when implemented
        Ok(None)
    }

    async fn get_related(&self, id: Uuid, depth: usize) -> GraphitiResult<Vec<RelatedMemory>> {
        info!("Getting related memories for {}, depth: {}", id, depth);

        // Naive relatedness: episodes sharing at least one extracted entity
        let entities_map = self.memory_entities.read().await;
        let Some(target_entities) = entities_map.get(&id) else {
            return Ok(vec![]);
        };
        let target_set: std::collections::HashSet<&String> = target_entities.iter().collect();

        let index = self.memory_index.read().await;
        let mut related: Vec<RelatedMemory> = Vec::new();
        for (other_id, ep) in index.iter() {
            if other_id == &id {
                continue;
            }
            if let Some(ents) = entities_map.get(other_id) {
                if ents.iter().any(|e| target_set.contains(e)) {
                    let node = MemoryNode {
                        id: *other_id,
                        node_type: "Episode".to_string(),
                        name: ep.name.clone(),
                        content: Some(ep.content.clone()),
                        created_at: ep.temporal.created_at.to_rfc3339(),
                        event_time: ep.temporal.valid_from.to_rfc3339(),
                        properties: serde_json::json!({ "source": ep.source }),
                    };
                    related.push(RelatedMemory {
                        node,
                        relationship: "SHARED_ENTITY".to_string(),
                        distance: 1,
                    });
                }
            }
        }

        // Limit by depth heuristics (depth>1 not implemented; return direct neighbors)
        Ok(related)
    }

    async fn search_memory_facts(
        &self,
        query: String,
        limit: Option<usize>,
    ) -> GraphitiResult<Vec<SimpleExtractedRelationship>> {
        // Minimal implementation: filter extracted relationships by substring match on any field
        let q = query.to_lowercase();
        let rels = self.memory_relationships.read().await;
        let mut matches: Vec<SimpleExtractedRelationship> = rels
            .iter()
            .filter(|r| {
                if q.is_empty() {
                    return true;
                }
                r.source.to_lowercase().contains(&q)
                    || r.target.to_lowercase().contains(&q)
                    || r.relationship.to_lowercase().contains(&q)
            })
            .cloned()
            .collect();
        // Simple score: prefer higher confidence first
        matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(l) = limit {
            matches.truncate(l);
        }
        Ok(matches)
    }

    async fn search_memory_facts_json(
        &self,
        query: String,
        limit: Option<usize>,
    ) -> GraphitiResult<Vec<serde_json::Value>> {
        let q = query.to_lowercase();
        let rel_map = self.memory_relationships_by_id.read().await;
        let mut items: Vec<(Uuid, &SimpleExtractedRelationship)> = rel_map
            .iter()
            .filter(|(_, r)| {
                if q.is_empty() { return true; }
                r.source.to_lowercase().contains(&q)
                    || r.target.to_lowercase().contains(&q)
                    || r.relationship.to_lowercase().contains(&q)
            })
            .map(|(id, r)| (*id, r))
            .collect();
        // sort by confidence desc
        items.sort_by(|a,b| b.1.confidence.partial_cmp(&a.1.confidence).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(l) = limit { items.truncate(l); }
        Ok(items
            .into_iter()
            .map(|(id, r)| {
                let mut v = Self::relationship_to_json(r);
                if let Some(obj) = v.as_object_mut() { obj.insert("uuid".to_string(), serde_json::json!(id)); }
                v
            })
            .collect())
    }

    async fn add_code_entity(
        &self,
        req: AddCodeEntityRequest,
    ) -> GraphitiResult<AddCodeEntityResponse> {
        info!("Adding code entity: {} ({})", req.name, req.entity_type);

        // Parse the entity type
        let entity_type = match req.entity_type.to_lowercase().as_str() {
            "class" => CodeEntityType::Class,
            "function" => CodeEntityType::Function,
            "module" => CodeEntityType::Module,
            "api" => CodeEntityType::Api,
            "database" => CodeEntityType::DataModel,
            "config" => CodeEntityType::Configuration,
            "test" => CodeEntityType::TestCase,
            "documentation" => CodeEntityType::Documentation,
            _ => CodeEntityType::Module, // Default fallback
        };

        // Create code entity
        let entity_id = Uuid::new_v4();
        let now = Utc::now();

        let temporal = TemporalMetadata {
            created_at: now,
            valid_from: now,
            valid_to: None,
            expired_at: None,
        };

        // Generate embedding for the entity name and description combined
        let combined_text = format!("{}: {}", req.name, req.description);
        let entity_embedding = match self.embedder.embed(&combined_text).await {
            Ok(embedding) => Some(embedding),
            Err(e) => {
                info!("Failed to generate embedding for code entity: {}", e);
                None
            }
        };

        let code_entity = CodeEntity {
            id: entity_id,
            entity_type,
            name: req.name.clone(),
            description: req.description.clone(),
            file_path: req.file_path.clone(),
            line_range: req.line_range,
            language: req.language.clone(),
            framework: req.framework.clone(),
            complexity: req.complexity,
            importance: req.importance,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        };

        // Convert to EntityNode for storage
        let mut properties = serde_json::json!({
            "entity_type": code_entity.entity_type.to_string(),
            "description": code_entity.description,
            "code_entity": true
        });

        if let Some(file_path) = &code_entity.file_path {
            properties["file_path"] = serde_json::json!(file_path);
        }

        if let Some(line_range) = &code_entity.line_range {
            properties["line_range"] = serde_json::json!(line_range);
        }

        if let Some(language) = &code_entity.language {
            properties["language"] = serde_json::json!(language);
        }

        if let Some(framework) = &code_entity.framework {
            properties["framework"] = serde_json::json!(framework);
        }

        if let Some(complexity) = code_entity.complexity {
            properties["complexity"] = serde_json::json!(complexity);
        }

        if let Some(importance) = code_entity.importance {
            properties["importance"] = serde_json::json!(importance);
        }

        let entity_node = EntityNode {
            id: entity_id,
            name: code_entity.name,
            entity_type: "CodeEntity".to_string(),
            labels: vec![
                "CodeEntity".to_string(),
                code_entity.entity_type.to_string(),
            ],
            properties,
            temporal,
            embedding: entity_embedding,
        };

        // Store the entity
        self.storage.create_node(&entity_node).await?;

        Ok(AddCodeEntityResponse {
            id: entity_id,
            message: format!(
                "Code entity '{}' of type '{}' added successfully",
                req.name, req.entity_type
            ),
        })
    }

    async fn delete_episode(&self, id: Uuid) -> GraphitiResult<bool> {
        // Best-effort: delete from episode table if present
        // For sqlite path (sqlx backend) current storage doesn't expose delete-by-type; call storage.delete_node
        self.storage.delete_node(&id).await?;
        Ok(true)
    }

    async fn get_episodes(&self, last_n: usize) -> GraphitiResult<Vec<EpisodeNode>> {
        // Minimal: read from in-memory index snapshot and return recent by created_at
        let mut episodes: Vec<EpisodeNode> = {
            let idx = self.memory_index.read().await;
            idx.values().cloned().collect()
        };
        episodes.sort_by(|a,b| b.temporal.created_at.cmp(&a.temporal.created_at));
        episodes.truncate(last_n);
        Ok(episodes)
    }

    async fn clear_graph(&self) -> GraphitiResult<()> {
        // Minimal clear: drop in-memory indexes
        {
            let mut idx = self.memory_index.write().await; idx.clear();
        }
        {
            let mut ents = self.memory_entities.write().await; ents.clear();
        }
        {
            let mut rels = self.memory_relationships.write().await; rels.clear();
        }
        {
            let mut rel_map = self.memory_relationships_by_id.write().await; rel_map.clear();
        }
        // Storage reset is backend-specific; skip for now
        Ok(())
    }

    async fn get_entity_edge_json(&self, id: Uuid) -> GraphitiResult<Option<serde_json::Value>> {
        // Fetch from storage
        if let Some(edge) = self.storage.get_edge_by_id(&id).await? {
            // attributes 中去除与顶层重复的字段，确保与 Python EntityEdge 序列化一致
            let mut attributes = edge.properties;
            if attributes.is_object() {
                if let Some(map) = attributes.as_object_mut() {
                    map.remove("uuid");
                    map.remove("source_node_uuid");
                    map.remove("target_node_uuid");
                    map.remove("fact");
                    map.remove("name");
                    map.remove("group_id");
                    map.remove("episodes");
                    map.remove("created_at");
                    map.remove("expired_at");
                    map.remove("valid_at");
                    map.remove("invalid_at");
                }
            }
            let value = serde_json::json!({
                "uuid": edge.id,
                "source_node_uuid": edge.source_id,
                "target_node_uuid": edge.target_id,
                "name": edge.relationship,
                "group_id": "",
                "episodes": [],
                "fact": attributes.get("fact").cloned().unwrap_or(serde_json::json!("")),
                "fact_embedding": null,
                "created_at": edge.temporal.created_at,
                "expired_at": edge.temporal.expired_at,
                "valid_at": edge.temporal.valid_from,
                "invalid_at": edge.temporal.valid_to,
                "attributes": attributes,
            });
            return Ok(Some(value));
        }
        Ok(None)
    }

    async fn delete_entity_edge_by_uuid(&self, id: Uuid) -> GraphitiResult<bool> {
        self.storage.delete_edge_by_id(&id).await
    }

    async fn record_activity(
        &self,
        req: RecordActivityRequest,
    ) -> GraphitiResult<RecordActivityResponse> {
        info!("Recording activity: {} ({})", req.title, req.activity_type);

        // Parse the activity type
        let activity_type = match req.activity_type.to_lowercase().as_str() {
            "implementation" => WorkflowStage::Implementation,
            "bugfix" => WorkflowStage::BugFix,
            "codereview" => WorkflowStage::CodeReview,
            "testing" => WorkflowStage::UnitTesting,
            "documentation" => WorkflowStage::RequirementAnalysis, // Could also be other stages
            "refactoring" => WorkflowStage::Refactoring,
            "deployment" => WorkflowStage::Deployment,
            "meeting" => WorkflowStage::RequirementAnalysis, // Planning stage
            _ => WorkflowStage::Implementation,              // Default fallback
        };

        // Create development activity
        let activity_id = Uuid::new_v4();
        let now = Utc::now();

        let temporal = TemporalMetadata {
            created_at: now,
            valid_from: now,
            valid_to: None,
            expired_at: None,
        };

        // Generate embedding for the activity title and description
        let combined_text = format!("{}: {}", req.title, req.description);
        let activity_embedding = match self.embedder.embed(&combined_text).await {
            Ok(embedding) => Some(embedding),
            Err(e) => {
                info!("Failed to generate embedding for activity: {}", e);
                None
            }
        };

        // Parse related entities from strings to UUIDs
        let related_entities = if let Some(entities) = &req.related_entities {
            entities
                .iter()
                .filter_map(|s| Uuid::parse_str(s).ok())
                .collect()
        } else {
            Vec::new()
        };

        let activity = DevelopmentActivity {
            id: activity_id,
            activity_type,
            title: req.title.clone(),
            description: req.description.clone(),
            related_entities,
            developer: req.developer.clone(),
            project: req.project.clone(),
            duration_minutes: req.duration_minutes,
            difficulty: req.difficulty,
            quality: req.quality,
            created_at: now,
            metadata: HashMap::new(),
        };

        // Convert to EpisodeNode for storage
        let mut properties = serde_json::json!({
            "activity_type": activity.activity_type.to_string(),
            "developer": activity.developer,
            "project": activity.project,
            "development_activity": true
        });

        if !activity.related_entities.is_empty() {
            properties["related_entities"] = serde_json::json!(activity.related_entities);
        }

        if let Some(duration) = activity.duration_minutes {
            properties["duration_minutes"] = serde_json::json!(duration);
        }

        if let Some(difficulty) = activity.difficulty {
            properties["difficulty"] = serde_json::json!(difficulty);
        }

        if let Some(quality) = activity.quality {
            properties["quality"] = serde_json::json!(quality);
        }

        // No stage field needed as it's already in activity_type

        let episode_node = EpisodeNode {
            id: activity_id,
            name: activity.title,
            episode_type: EpisodeType::Event,
            content: activity.description,
            source: format!("developer:{}", activity.developer),
            temporal: temporal.clone(),
            embedding: activity_embedding,
        };

        // Store the activity
        self.storage.create_node(&episode_node).await?;

        // Create relationships to related entities if specified
        if let Some(related_entities) = &req.related_entities {
            for entity_id_str in related_entities {
                if let Ok(related_id) = Uuid::parse_str(entity_id_str) {
                    let edge = graphiti_core::graph::Edge {
                        id: Uuid::new_v4(),
                        source_id: activity_id,
                        target_id: related_id,
                        relationship: "RELATES_TO".to_string(),
                        properties: serde_json::json!({
                            "relationship_type": "activity_entity"
                        }),
                        temporal: temporal.clone(),
                        weight: 1.0,
                    };

                    if let Err(e) = self.storage.create_edge(&edge).await {
                        info!(
                            "Failed to create relationship to entity {}: {}",
                            related_id, e
                        );
                    }
                }
            }
        }

        Ok(RecordActivityResponse {
            id: activity_id,
            message: format!(
                "Activity '{}' of type '{}' recorded successfully",
                req.title, req.activity_type
            ),
        })
    }

    async fn search_code(&self, req: SearchCodeRequest) -> GraphitiResult<SearchCodeResponse> {
        info!("Searching code entities: {}", req.query);

        // Generate embedding for the search query
        let _query_embedding = match self.embedder.embed(&req.query).await {
            Ok(embedding) => embedding,
            Err(e) => {
                info!("Failed to generate embedding for search query: {}", e);
                return Ok(SearchCodeResponse {
                    results: vec![],
                    total: 0,
                });
            }
        };

        // Simple code entity search implementation
        let _limit = req.limit.unwrap_or(10) as usize;

        // TODO: Implement proper code entity search using storage layer
        // This would involve querying code entity nodes and filtering by criteria
        // For now, return empty results as a placeholder
        let code_results: Vec<CodeEntity> = vec![];

        let total = code_results.len();

        info!("Code search completed, found {} results", total);

        Ok(SearchCodeResponse {
            results: code_results,
            total,
        })
    }

    async fn batch_add_code_entities(
        &self,
        req: BatchAddCodeEntitiesRequest,
    ) -> GraphitiResult<BatchAddCodeEntitiesResponse> {
        info!("Batch adding {} code entities", req.entities.len());

        let mut results = Vec::new();
        let mut successful_count = 0;
        let mut failed_count = 0;
        let mut errors = Vec::new();

        for entity_req in req.entities {
            match self.add_code_entity(entity_req).await {
                Ok(response) => {
                    results.push(response);
                    successful_count += 1;
                }
                Err(e) => {
                    failed_count += 1;
                    errors.push(format!("Failed to add entity: {}", e));
                    // Still add a placeholder response to maintain index consistency
                    results.push(AddCodeEntityResponse {
                        id: uuid::Uuid::new_v4(),
                        message: format!("Failed: {}", e),
                    });
                }
            }
        }

        info!(
            "Batch add completed: {} successful, {} failed",
            successful_count, failed_count
        );

        Ok(BatchAddCodeEntitiesResponse {
            results,
            successful_count,
            failed_count,
            errors,
        })
    }

    async fn batch_record_activities(
        &self,
        req: BatchRecordActivitiesRequest,
    ) -> GraphitiResult<BatchRecordActivitiesResponse> {
        info!("Batch recording {} activities", req.activities.len());

        let mut results = Vec::new();
        let mut successful_count = 0;
        let mut failed_count = 0;
        let mut errors = Vec::new();

        for activity_req in req.activities {
            match self.record_activity(activity_req).await {
                Ok(response) => {
                    results.push(response);
                    successful_count += 1;
                }
                Err(e) => {
                    failed_count += 1;
                    errors.push(format!("Failed to record activity: {}", e));
                    // Still add a placeholder response to maintain index consistency
                    results.push(RecordActivityResponse {
                        id: uuid::Uuid::new_v4(),
                        message: format!("Failed: {}", e),
                    });
                }
            }
        }

        info!(
            "Batch record completed: {} successful, {} failed",
            successful_count, failed_count
        );

        Ok(BatchRecordActivitiesResponse {
            results,
            successful_count,
            failed_count,
            errors,
        })
    }

    async fn get_context_suggestions(
        &self,
        req: ContextSuggestionRequest,
    ) -> GraphitiResult<ContextSuggestionResponse> {
        info!("Generating context suggestions for: {}", req.context);

        let limit = req.limit.unwrap_or(5) as usize;
        let mut suggestions = Vec::new();

        // Analyze the context and generate intelligent suggestions
        let context_lower = req.context.to_lowercase();

        // Code quality suggestions
        if context_lower.contains("bug")
            || context_lower.contains("error")
            || context_lower.contains("fix")
        {
            suggestions.push(ContextSuggestion {
                suggestion_type: "BugFix".to_string(),
                title: "Add Error Handling".to_string(),
                description: "Consider adding comprehensive error handling and logging to prevent similar issues".to_string(),
                related_entities: vec!["error_handler".to_string(), "logging_service".to_string()],
                confidence: 0.85,
                priority: 9,
            });

            suggestions.push(ContextSuggestion {
                suggestion_type: "Testing".to_string(),
                title: "Write Unit Tests".to_string(),
                description:
                    "Add unit tests to cover the fixed functionality and prevent regression"
                        .to_string(),
                related_entities: vec!["test_module".to_string()],
                confidence: 0.90,
                priority: 8,
            });
        }

        // Performance suggestions
        if context_lower.contains("slow")
            || context_lower.contains("performance")
            || context_lower.contains("optimize")
        {
            suggestions.push(ContextSuggestion {
                suggestion_type: "Optimization".to_string(),
                title: "Add Caching Layer".to_string(),
                description:
                    "Implement caching to improve performance for frequently accessed data"
                        .to_string(),
                related_entities: vec!["cache_service".to_string(), "redis_client".to_string()],
                confidence: 0.80,
                priority: 7,
            });

            suggestions.push(ContextSuggestion {
                suggestion_type: "DatabaseOptimization".to_string(),
                title: "Optimize Database Queries".to_string(),
                description: "Review and optimize database queries, consider adding indexes"
                    .to_string(),
                related_entities: vec![
                    "database_service".to_string(),
                    "query_optimizer".to_string(),
                ],
                confidence: 0.75,
                priority: 8,
            });
        }

        // Security suggestions
        if context_lower.contains("auth")
            || context_lower.contains("security")
            || context_lower.contains("login")
        {
            suggestions.push(ContextSuggestion {
                suggestion_type: "Security".to_string(),
                title: "Implement Rate Limiting".to_string(),
                description:
                    "Add rate limiting to authentication endpoints to prevent brute force attacks"
                        .to_string(),
                related_entities: vec!["rate_limiter".to_string(), "auth_service".to_string()],
                confidence: 0.88,
                priority: 9,
            });

            suggestions.push(ContextSuggestion {
                suggestion_type: "Security".to_string(),
                title: "Add Input Validation".to_string(),
                description:
                    "Implement comprehensive input validation to prevent injection attacks"
                        .to_string(),
                related_entities: vec!["validator".to_string(), "sanitizer".to_string()],
                confidence: 0.92,
                priority: 10,
            });
        }

        // API development suggestions
        if context_lower.contains("api")
            || context_lower.contains("endpoint")
            || context_lower.contains("rest")
        {
            suggestions.push(ContextSuggestion {
                suggestion_type: "Documentation".to_string(),
                title: "Generate API Documentation".to_string(),
                description: "Create comprehensive API documentation with examples and schemas"
                    .to_string(),
                related_entities: vec!["swagger_config".to_string(), "api_docs".to_string()],
                confidence: 0.70,
                priority: 6,
            });

            suggestions.push(ContextSuggestion {
                suggestion_type: "Monitoring".to_string(),
                title: "Add API Monitoring".to_string(),
                description: "Implement monitoring and alerting for API endpoints".to_string(),
                related_entities: vec!["metrics_service".to_string(), "health_check".to_string()],
                confidence: 0.78,
                priority: 7,
            });
        }

        // Code structure suggestions
        if context_lower.contains("refactor")
            || context_lower.contains("clean")
            || context_lower.contains("structure")
        {
            suggestions.push(ContextSuggestion {
                suggestion_type: "Refactoring".to_string(),
                title: "Extract Common Utilities".to_string(),
                description: "Extract common functionality into reusable utility modules"
                    .to_string(),
                related_entities: vec!["utils_module".to_string(), "common_functions".to_string()],
                confidence: 0.72,
                priority: 5,
            });
        }

        // File-specific suggestions based on current file
        if let Some(current_file) = &req.current_file {
            if current_file.contains("test") {
                suggestions.push(ContextSuggestion {
                    suggestion_type: "Testing".to_string(),
                    title: "Add Integration Tests".to_string(),
                    description: "Consider adding integration tests to complement unit tests"
                        .to_string(),
                    related_entities: vec!["integration_test_suite".to_string()],
                    confidence: 0.65,
                    priority: 6,
                });
            }

            if current_file.contains("config") || current_file.contains("settings") {
                suggestions.push(ContextSuggestion {
                    suggestion_type: "Configuration".to_string(),
                    title: "Environment-specific Configs".to_string(),
                    description: "Ensure configurations are properly separated by environment"
                        .to_string(),
                    related_entities: vec![
                        "env_config".to_string(),
                        "config_validator".to_string(),
                    ],
                    confidence: 0.68,
                    priority: 7,
                });
            }
        }

        // Default general suggestions if no specific context matched
        if suggestions.is_empty() {
            suggestions.push(ContextSuggestion {
                suggestion_type: "BestPractice".to_string(),
                title: "Code Review Checklist".to_string(),
                description: "Follow established code review guidelines and best practices"
                    .to_string(),
                related_entities: vec!["review_guidelines".to_string()],
                confidence: 0.60,
                priority: 5,
            });

            suggestions.push(ContextSuggestion {
                suggestion_type: "Documentation".to_string(),
                title: "Update Documentation".to_string(),
                description: "Ensure code changes are reflected in documentation".to_string(),
                related_entities: vec!["docs".to_string(), "readme".to_string()],
                confidence: 0.55,
                priority: 4,
            });
        }

        // Sort by priority (descending) and confidence (descending), then limit
        suggestions.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        suggestions.truncate(limit);

        let total = suggestions.len();

        info!("Generated {} context suggestions", total);

        Ok(ContextSuggestionResponse { suggestions, total })
    }
}
