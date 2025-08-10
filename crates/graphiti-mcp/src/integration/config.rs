//! Configuration management for implicit integration features

use super::*;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs;

/// Configuration manager for integration settings
#[allow(dead_code)]
pub struct ConfigManager {
    config_path: PathBuf,
    config: Arc<RwLock<IntegrationConfig>>,
    watchers: Vec<tokio::sync::mpsc::Sender<ConfigUpdate>>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum ConfigUpdate {
    Reload,
    EnableFeature(String),
    DisableFeature(String),
    UpdateThreshold(f32),
    UpdateRateLimit(u32),
}

#[allow(dead_code)]
impl ConfigManager {
    /// Create a new config manager
    pub async fn new(config_path: impl AsRef<Path>) -> Result<Self, String> {
        let config_path = config_path.as_ref().to_path_buf();
        let config = Self::load_config(&config_path).await?;

        Ok(Self {
            config_path,
            config: Arc::new(RwLock::new(config)),
            watchers: Vec::new(),
        })
    }

    /// Load configuration from file
    async fn load_config(path: &Path) -> Result<IntegrationConfig, String> {
        if !path.exists() {
            // Create default config if not exists
            let default_config = IntegrationConfig::default();
            Self::save_config(path, &default_config).await?;
            return Ok(default_config);
        }

        let content = fs::read_to_string(path)
            .await
            .map_err(|e| format!("Failed to read config: {}", e))?;

        toml::from_str(&content).map_err(|e| format!("Failed to parse config: {}", e))
    }

    /// Save configuration to file
    async fn save_config(path: &Path, config: &IntegrationConfig) -> Result<(), String> {
        let content = toml::to_string_pretty(config)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("Failed to create config directory: {}", e))?;
        }

        fs::write(path, content)
            .await
            .map_err(|e| format!("Failed to write config: {}", e))?;

        Ok(())
    }

    /// Get current configuration
    pub async fn get_config(&self) -> IntegrationConfig {
        self.config.read().await.clone()
    }

    /// Update configuration
    pub async fn update_config<F>(&self, updater: F) -> Result<(), String>
    where
        F: FnOnce(&mut IntegrationConfig),
    {
        let mut config = self.config.write().await;
        updater(&mut config);

        // Save to file
        Self::save_config(&self.config_path, &config).await?;

        // Notify watchers
        for watcher in &self.watchers {
            let _ = watcher.send(ConfigUpdate::Reload).await;
        }

        Ok(())
    }

    /// Watch for configuration changes
    pub fn watch(&mut self) -> tokio::sync::mpsc::Receiver<ConfigUpdate> {
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        self.watchers.push(tx);
        rx
    }

    /// Enable a specific feature
    pub async fn enable_feature(&self, feature: &str) -> Result<(), String> {
        self.update_config(|config| match feature {
            "auto_capture" => config.auto_capture_entities = true,
            "auto_record" => config.auto_record_activities = true,
            "auto_suggest" => config.auto_suggestions = true,
            "all" => {
                config.enabled = true;
                config.auto_capture_entities = true;
                config.auto_record_activities = true;
                config.auto_suggestions = true;
            }
            _ => {}
        })
        .await?;

        Ok(())
    }

    /// Disable a specific feature
    pub async fn disable_feature(&self, feature: &str) -> Result<(), String> {
        self.update_config(|config| match feature {
            "auto_capture" => config.auto_capture_entities = false,
            "auto_record" => config.auto_record_activities = false,
            "auto_suggest" => config.auto_suggestions = false,
            "all" => config.enabled = false,
            _ => {}
        })
        .await?;

        Ok(())
    }

    /// Add exclusion pattern
    pub async fn add_exclusion(&self, pattern: String) -> Result<(), String> {
        self.update_config(|config| {
            if !config.exclusion_patterns.contains(&pattern) {
                config.exclusion_patterns.push(pattern);
            }
        })
        .await
    }

    /// Remove exclusion pattern
    pub async fn remove_exclusion(&self, pattern: &str) -> Result<(), String> {
        self.update_config(|config| {
            config.exclusion_patterns.retain(|p| p != pattern);
        })
        .await
    }

    /// Update confidence threshold
    pub async fn set_threshold(&self, threshold: f32) -> Result<(), String> {
        if threshold < 0.0 || threshold > 1.0 {
            return Err("Threshold must be between 0.0 and 1.0".to_string());
        }

        self.update_config(|config| {
            config.confidence_threshold = threshold;
        })
        .await
    }

    /// Update rate limit
    pub async fn set_rate_limit(&self, limit: u32) -> Result<(), String> {
        self.update_config(|config| {
            config.rate_limit = limit;
        })
        .await
    }
}

/// Default configuration file template
#[allow(dead_code)]
pub const DEFAULT_CONFIG_TEMPLATE: &str = r#"# Graphiti MCP Implicit Integration Configuration

# Enable automatic knowledge capture
enabled = true

# Automatically capture code entities (classes, functions, etc.)
auto_capture_entities = true

# Automatically record development activities
auto_record_activities = true

# Provide context-aware suggestions
auto_suggestions = true

# Minimum confidence threshold for automatic actions (0.0-1.0)
confidence_threshold = 0.7

# Cooldown period between similar captures (seconds)
capture_cooldown = 30

# Maximum captures per hour
rate_limit = 100

# File patterns to exclude from monitoring
exclusion_patterns = [
    "*.key",
    "*.pem",
    "*secret*",
    "*password*",
    ".env",
    "node_modules/*",
    "target/*",
    "dist/*",
    "build/*",
    ".git/*"
]

# File patterns to include for monitoring
include_patterns = [
    "*.rs",
    "*.py",
    "*.js",
    "*.ts",
    "*.jsx",
    "*.tsx",
    "*.java",
    "*.go",
    "*.cpp",
    "*.c",
    "*.h",
    "*.hpp",
    "*.cs",
    "*.rb",
    "*.php",
    "*.swift",
    "*.kt",
    "*.scala"
]

# Notification preferences
[notifications]
# Show notifications for automatic captures
show_captures = false

# Show context-aware suggestions
show_suggestions = true

# Notification level: "debug", "info", "none"
level = "info"
"#;

/// User preferences that can override configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Temporary disable all features
    pub paused: bool,

    /// Session-specific overrides
    pub session_overrides: SessionOverrides,

    /// Per-project settings
    pub project_settings: std::collections::HashMap<String, ProjectSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionOverrides {
    pub disable_for_files: Vec<String>,
    pub disable_for_duration: Option<Duration>,
    pub started_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSettings {
    pub enabled: bool,
    pub custom_patterns: Vec<String>,
    pub team_shared: bool,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            paused: false,
            session_overrides: SessionOverrides {
                disable_for_files: vec![],
                disable_for_duration: None,
                started_at: None,
            },
            project_settings: std::collections::HashMap::new(),
        }
    }
}
