//! Learning detection and notification system
//!
//! This module provides functionality to detect when the knowledge graph
//! learns new content and provides real-time notifications to users.

pub mod config;
pub mod detector;
pub mod events;
pub mod notifications;

pub use config::LearningConfig;
pub use detector::LearningDetector;
pub use detector::SmartLearningDetector;
pub use events::LearningContext;
pub use events::LearningEvent;
pub use notifications::ConsoleNotificationChannel;
pub use notifications::LearningNotification;
pub use notifications::MCPNotificationChannel;

pub use notifications::NotificationManager;
pub use notifications::NotificationStats;

use async_trait::async_trait;
use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;

/// Result type for learning detection operations
pub type LearningResult<T> = Result<T, LearningError>;

/// Errors that can occur during learning detection
#[derive(Debug, thiserror::Error)]
pub enum LearningError {
    #[error("Detection failed: {0}")]
    #[allow(dead_code)]
    DetectionFailed(String),

    #[error("Notification error: {0}")]
    NotificationError(String),

    #[error("Configuration error: {0}")]
    #[allow(dead_code)]
    ConfigError(String),

    #[error("Analysis error: {0}")]
    #[allow(dead_code)]
    AnalysisError(String),
}

/// Statistics about learning activities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    /// Total learning events detected
    pub total_events: u64,

    /// Events by type
    pub events_by_type: HashMap<String, u64>,

    /// Learning rate (events per hour)
    pub learning_rate: f64,

    /// Most active learning categories
    pub top_categories: Vec<String>,

    /// Last learning event timestamp
    pub last_event_at: Option<DateTime<Utc>>,
}

impl Default for LearningStats {
    fn default() -> Self {
        Self {
            total_events: 0,
            events_by_type: HashMap::new(),
            learning_rate: 0.0,
            top_categories: Vec::new(),
            last_event_at: None,
        }
    }
}

/// Trait for components that can learn from new content
#[async_trait]
#[allow(dead_code)]
pub trait LearningAware: Send + Sync {
    /// Called when new learning is detected
    async fn on_learning_detected(&self, event: &LearningEvent) -> LearningResult<()>;

    /// Get learning statistics
    async fn get_learning_stats(&self) -> LearningResult<LearningStats>;

    /// Reset learning state
    async fn reset_learning_state(&self) -> LearningResult<()>;
}
