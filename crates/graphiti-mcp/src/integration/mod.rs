//! Integration layer for implicit knowledge capture in Claude Code
//!
//! This module provides automatic knowledge management without explicit user calls

pub mod activity_detector;
pub mod config;
pub mod context_analyzer;
pub mod hooks;
pub mod knowledge_manager;

use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable automatic knowledge capture
    pub enabled: bool,

    /// Capture code entities automatically
    pub auto_capture_entities: bool,

    /// Record development activities
    pub auto_record_activities: bool,

    /// Provide context-aware suggestions
    pub auto_suggestions: bool,

    /// Minimum confidence threshold for automatic actions (0.0-1.0)
    pub confidence_threshold: f32,

    /// Cooldown period between similar captures (seconds)
    pub capture_cooldown: u64,

    /// Maximum captures per hour
    pub rate_limit: u32,

    /// Sensitive patterns to exclude
    pub exclusion_patterns: Vec<String>,

    /// File patterns to monitor
    pub include_patterns: Vec<String>,

    /// Notification preferences
    pub notifications: NotificationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Show notifications for captures
    pub show_captures: bool,

    /// Show suggestions
    pub show_suggestions: bool,

    /// Notification level (debug, info, none)
    pub level: String,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_capture_entities: true,
            auto_record_activities: true,
            auto_suggestions: true,
            confidence_threshold: 0.7,
            capture_cooldown: 30,
            rate_limit: 100,
            exclusion_patterns: vec![
                "*.key".to_string(),
                "*.pem".to_string(),
                "*secret*".to_string(),
                "*password*".to_string(),
                ".env".to_string(),
            ],
            include_patterns: vec![
                "*.rs".to_string(),
                "*.py".to_string(),
                "*.js".to_string(),
                "*.ts".to_string(),
                "*.java".to_string(),
                "*.go".to_string(),
                "*.cpp".to_string(),
                "*.c".to_string(),
                "*.h".to_string(),
            ],
            notifications: NotificationConfig {
                show_captures: false,
                show_suggestions: true,
                level: "info".to_string(),
            },
        }
    }
}

/// Context of current development activity
#[derive(Debug, Clone)]
pub struct DevelopmentContext {
    /// Current file being worked on
    pub current_file: Option<String>,

    /// Recent files accessed
    pub recent_files: Vec<String>,

    /// Recent search queries
    pub recent_searches: Vec<String>,

    /// Recent code modifications
    pub recent_edits: Vec<EditContext>,

    /// Detected activity type
    pub activity_type: Option<ActivityType>,

    /// Confidence in activity detection
    pub confidence: f32,

    /// Start time of current context
    #[allow(dead_code)]
    pub start_time: DateTime<Utc>,

    /// Related entities detected
    #[allow(dead_code)]
    pub related_entities: Vec<Uuid>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EditContext {
    #[allow(dead_code)]
    /// Path to the file being edited
    pub file_path: String,
    #[allow(dead_code)]
    /// Original content before edit
    pub old_content: String,
    #[allow(dead_code)]
    /// New content after edit
    pub new_content: String,
    #[allow(dead_code)]
    /// When the edit occurred
    pub timestamp: DateTime<Utc>,
    #[allow(dead_code)]
    /// Optional line range for the edit
    pub line_range: Option<(u32, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum ActivityType {
    #[allow(dead_code)]
    FeatureDevelopment,
    #[allow(dead_code)]
    BugFix,
    #[allow(dead_code)]
    Refactoring,
    Testing,
    #[allow(dead_code)]
    Documentation,
    #[allow(dead_code)]
    CodeReview,
    #[allow(dead_code)]
    Learning,
    #[allow(dead_code)]
    Debugging,
}

/// Integration manager that coordinates all implicit features
pub struct IntegrationManager {
    /// Integration configuration
    pub config: Arc<RwLock<IntegrationConfig>>,
    /// Current development context
    pub context: Arc<RwLock<DevelopmentContext>>,
    #[allow(dead_code)]
    /// Graphiti service instance
    pub graphiti_service: Arc<dyn super::GraphitiService + Send + Sync>,
    /// Rate limiter for API calls
    pub rate_limiter: Arc<RwLock<RateLimiter>>,
}

#[derive(Debug)]
pub struct RateLimiter {
    /// Number of captures in current hour
    captures_this_hour: u32,
    /// Start of current hour window
    hour_start: DateTime<Utc>,
    #[allow(dead_code)]
    /// Last capture times by operation type
    last_capture_times: std::collections::HashMap<String, DateTime<Utc>>,
}

#[allow(dead_code)]
impl RateLimiter {
    pub fn new() -> Self {
        Self {
            captures_this_hour: 0,
            hour_start: Utc::now(),
            last_capture_times: std::collections::HashMap::new(),
        }
    }

    pub fn check_rate_limit(&mut self, config: &IntegrationConfig) -> bool {
        let now = Utc::now();

        // Reset hourly counter if needed
        if now.signed_duration_since(self.hour_start).num_hours() >= 1 {
            self.captures_this_hour = 0;
            self.hour_start = now;
        }

        self.captures_this_hour < config.rate_limit
    }

    pub fn check_cooldown(&self, key: &str, config: &IntegrationConfig) -> bool {
        if let Some(last_time) = self.last_capture_times.get(key) {
            let elapsed = Utc::now().signed_duration_since(*last_time).num_seconds();
            elapsed as u64 >= config.capture_cooldown
        } else {
            true
        }
    }

    pub fn record_capture(&mut self, key: String) {
        self.captures_this_hour += 1;
        self.last_capture_times.insert(key, Utc::now());
    }
}

#[allow(dead_code)]
impl IntegrationManager {
    pub fn new(
        config: IntegrationConfig,
        graphiti_service: Arc<dyn super::GraphitiService + Send + Sync>,
    ) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            context: Arc::new(RwLock::new(DevelopmentContext {
                current_file: None,
                recent_files: Vec::new(),
                recent_searches: Vec::new(),
                recent_edits: Vec::new(),
                activity_type: None,
                confidence: 0.0,
                start_time: Utc::now(),
                related_entities: Vec::new(),
            })),
            graphiti_service,
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new())),
        }
    }

    /// Check if a file should be monitored
    pub async fn should_monitor_file(&self, file_path: &str) -> bool {
        let config = self.config.read().await;

        // Check exclusion patterns
        for pattern in &config.exclusion_patterns {
            if glob::Pattern::new(pattern)
                .ok()
                .map(|p| p.matches(file_path))
                .unwrap_or(false)
            {
                return false;
            }
        }

        // Check inclusion patterns
        for pattern in &config.include_patterns {
            if glob::Pattern::new(pattern)
                .ok()
                .map(|p| p.matches(file_path))
                .unwrap_or(false)
            {
                return true;
            }
        }

        false
    }

    /// Update development context based on tool usage
    pub async fn update_context(&self, tool: &str, params: &serde_json::Value) {
        let mut context = self.context.write().await;

        match tool {
            "Read" => {
                if let Some(file_path) = params.get("file_path").and_then(|p| p.as_str()) {
                    context.current_file = Some(file_path.to_string());
                    context.recent_files.push(file_path.to_string());
                    if context.recent_files.len() > 10 {
                        context.recent_files.remove(0);
                    }
                }
            }
            "Edit" | "MultiEdit" => {
                if let Some(file_path) = params.get("file_path").and_then(|p| p.as_str()) {
                    let edit = EditContext {
                        file_path: file_path.to_string(),
                        old_content: params
                            .get("old_string")
                            .and_then(|s| s.as_str())
                            .unwrap_or("")
                            .to_string(),
                        new_content: params
                            .get("new_string")
                            .and_then(|s| s.as_str())
                            .unwrap_or("")
                            .to_string(),
                        timestamp: Utc::now(),
                        line_range: None,
                    };
                    context.recent_edits.push(edit);
                    if context.recent_edits.len() > 20 {
                        context.recent_edits.remove(0);
                    }
                }
            }
            "Grep" | "Glob" => {
                if let Some(query) = params
                    .get("pattern")
                    .or_else(|| params.get("query"))
                    .and_then(|q| q.as_str())
                {
                    context.recent_searches.push(query.to_string());
                    if context.recent_searches.len() > 10 {
                        context.recent_searches.remove(0);
                    }
                }
            }
            _ => {}
        }
    }

    /// Get current development context
    pub async fn get_context(&self) -> DevelopmentContext {
        self.context.read().await.clone()
    }
}
