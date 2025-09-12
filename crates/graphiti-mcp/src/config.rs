use std::path::PathBuf;

/// Initialize tracing
use tracing::{info, debug};
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use crate::types::{Args, ServerConfig};

pub(crate) fn init_tracing(level: &str) -> anyhow::Result<()> {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();

    Ok(())
}

/// Load configuration from file with project isolation support
pub(crate) fn load_config(args: &Args) -> anyhow::Result<ServerConfig> {
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

    // Note: keep engine semantics as configured; do not override path here to respect per-project DB

    info!("Loaded configuration from: {}", config_path.display());
    info!("Using LLM provider: {}", config.llm.provider);
    info!("Using embedding provider: {:?}", config.embedder.provider);
    info!("Database path: {}", config.cozo.path);
    debug!(
        "Server settings: max_connections={} rps={} req_timeout_s={} body_limit_bytes={}",
        config.server.max_connections,
        config.server.requests_per_second,
        config.server.request_timeout_seconds,
        config.server.request_body_limit_bytes
    );

    Ok(config)
}

/// Resolve configuration file path with project isolation
pub(crate) fn resolve_config_path(args: &Args) -> anyhow::Result<PathBuf> {
    // Support ~ expansion in provided path for convenience
    fn expand_home(p: &PathBuf) -> PathBuf {
        let s = p.to_string_lossy();
        if s.starts_with("~/") {
            if let Some(home) = dirs::home_dir() {
                return home.join(&s[2..]);
            }
        }
        p.clone()
    }

    let provided_path = expand_home(&args.config);
    let explicit_force = std::env::var("GRAPHITI_CONFIG_FORCE").ok().as_deref() == Some("1");
    let is_explicit = provided_path.file_name().map(|n| n != "config.toml").unwrap_or(true);

    // 1) Explicit --config path takes precedence
    if explicit_force || is_explicit {
        if provided_path.exists() {
            info!(
                "Using explicit --config: {}{}",
                provided_path.display(),
                if explicit_force { " (forced)" } else { "" }
            );
            return Ok(provided_path);
        } else if explicit_force {
            return Err(anyhow::anyhow!(
                "GRAPHITI_CONFIG_FORCE=1 but --config does not exist: {}",
                provided_path.display()
            ));
        }
        // Not forced and doesn't exist; fall back to default
    }

    // 2) Project isolation fallback in working directory
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

    // 3) 若项目配置不存在，则尝试创建并回落
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
pub(crate) fn apply_project_overrides(config: &mut ServerConfig, args: &Args) -> anyhow::Result<()> {
    use std::fs::OpenOptions;

    // Override database path if specified
    if let Some(db_path) = &args.db_path {
        config.cozo.path = db_path.to_string_lossy().to_string();
        info!("Database path overridden to: {}", config.cozo.path);

        // Ensure parent directory exists and DB file is present (sqlx/sqlite requires file on some systems)
        let db_path_buf = db_path.clone();
        if let Some(parent) = db_path_buf.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("Failed to create data directory: {}", e))?;
        }
        if !db_path_buf.exists() {
            // Create an empty file so sqlite can open it reliably
            let _ = OpenOptions::new()
                .write(true)
                .create(true)
                .open(&db_path_buf)
                .map_err(|e| anyhow::anyhow!("Failed to create DB file {}: {}", db_path_buf.display(), e))?;
        }
    } else {
        // Determine the project directory (similar to Serena's --project)
        let project_dir = if let Some(project) = &args.project {
            project.clone()
        } else {
            // Use current working directory if no project specified
            std::env::current_dir()
                .map_err(|e| anyhow::anyhow!("Failed to get current directory: {}", e))?
        };

        // Use project-specific database path under working directory
        let project_db = project_dir
            .join(".graphiti")
            .join("data")
            .join("graphiti.db");

        // Ensure data directory exists
        if let Some(parent) = project_db.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("Failed to create data directory: {}", e))?;
        }
        // Ensure DB file exists (avoid sqlite 'unable to open database file')
        if !project_db.exists() {
            let _ = OpenOptions::new()
                .write(true)
                .create(true)
                .open(&project_db)
                .map_err(|e| anyhow::anyhow!("Failed to create DB file {}: {}", project_db.display(), e))?;
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
pub(crate) fn create_default_project_config(config_path: &PathBuf) -> anyhow::Result<()> {
    let default_config = r#"# Graphiti MCP Server Project Configuration
# Auto-generated default configuration

[server]
host = "127.0.0.1"
port = 8080
max_connections = 100

[cozo]
engine = "sqlite"
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
