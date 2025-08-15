//! Notification management for learning events

use super::events::LearningEvent;
use super::LearningError;
use super::LearningResult;
use async_trait::async_trait;
use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::sync::RwLock;
use tracing::debug;
use tracing::info;
use tracing::warn;
use uuid::Uuid;

/// Notification levels for different types of learning events
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum NotificationLevel {
    /// Informational notifications (low priority)
    Info,
    /// Success notifications (medium priority)
    Success,
    /// Warning notifications (high priority)
    Warning,
    /// Critical notifications (highest priority)
    Critical,
}

impl NotificationLevel {
    /// Get priority score for sorting
    pub fn priority_score(&self) -> u8 {
        match self {
            NotificationLevel::Info => 1,
            NotificationLevel::Success => 2,
            NotificationLevel::Warning => 3,
            NotificationLevel::Critical => 4,
        }
    }

    /// Get emoji representation
    pub fn emoji(&self) -> &'static str {
        match self {
            NotificationLevel::Info => "INFO",
            NotificationLevel::Success => "SUCCESS",
            NotificationLevel::Warning => "WARNING",
            NotificationLevel::Critical => "CRITICAL",
        }
    }
}

/// A learning notification that can be sent to users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningNotification {
    /// Unique identifier
    pub id: Uuid,

    /// The learning event that triggered this notification
    pub event_id: Uuid,

    /// Notification level
    pub level: NotificationLevel,

    /// Title (short summary)
    pub title: String,

    /// Detailed message
    pub message: String,

    /// Action items or suggestions
    pub actions: Vec<NotificationAction>,

    /// When this notification was created
    pub created_at: DateTime<Utc>,

    /// When this notification should expire (optional)
    pub expires_at: Option<DateTime<Utc>>,

    /// Whether this notification has been seen
    pub seen: bool,

    /// Whether this notification has been dismissed
    pub dismissed: bool,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Actions that can be taken in response to a notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationAction {
    /// Action identifier
    pub id: String,

    /// Display label
    pub label: String,

    /// Action type
    pub action_type: ActionType,

    /// Parameters for the action
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Types of actions available for notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    /// View more details about the learning event
    ViewDetails,

    /// Search for related content
    SearchRelated,

    /// Add to bookmarks/favorites
    Bookmark,

    /// Dismiss this notification
    Dismiss,

    /// Provide feedback on the learning detection
    ProvideFeedback,

    /// Open relevant file or location
    OpenSource,

    /// Execute a custom command
    ExecuteCommand { command: String },
}

/// Configuration for notification behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct NotificationConfig {
    /// Enable notifications
    pub enabled: bool,

    /// Maximum number of notifications to keep in history
    pub max_history: usize,

    /// Minimum notification level to display
    pub min_level: NotificationLevel,

    /// Auto-dismiss notifications after this many seconds
    pub auto_dismiss_seconds: Option<u64>,

    /// Group similar notifications
    pub group_similar: bool,

    /// Maximum notifications per minute (rate limiting)
    pub max_per_minute: u32,

    /// Enable sound notifications
    pub enable_sound: bool,

    /// Enable desktop notifications (if available)
    pub enable_desktop_notifications: bool,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_history: 100,
            min_level: NotificationLevel::Info,
            auto_dismiss_seconds: Some(300), // 5 minutes
            group_similar: true,
            max_per_minute: 10,
            enable_sound: false,
            enable_desktop_notifications: false,
        }
    }
}

/// Statistics about notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationStats {
    /// Total notifications sent
    pub total_sent: u64,

    /// Notifications by level
    pub by_level: HashMap<String, u64>,

    /// Notifications by tag
    pub by_tag: HashMap<String, u64>,

    /// Average time to dismissal
    pub avg_dismiss_time_seconds: f64,

    /// Percentage of notifications that were seen
    pub seen_percentage: f64,

    /// Current active notifications
    pub active_count: u32,
}

/// Trait for notification delivery channels
#[async_trait]
pub trait NotificationChannel: Send + Sync {
    /// Send a notification through this channel
    async fn send_notification(&self, notification: &LearningNotification) -> LearningResult<()>;

    /// Check if this channel is available
    async fn is_available(&self) -> bool;

    /// Get channel identifier
    fn channel_id(&self) -> &str;
}

/// MCP-based notification channel for Claude Code integration
pub struct MCPNotificationChannel {
    /// Channel for broadcasting notifications
    sender: broadcast::Sender<LearningNotification>,
}

impl MCPNotificationChannel {
    pub fn new() -> (Self, broadcast::Receiver<LearningNotification>) {
        let (sender, receiver) = broadcast::channel(100);
        (Self { sender }, receiver)
    }
}

#[async_trait]
impl NotificationChannel for MCPNotificationChannel {
    async fn send_notification(&self, notification: &LearningNotification) -> LearningResult<()> {
        self.sender.send(notification.clone()).map_err(|e| {
            LearningError::NotificationError(format!("Failed to send MCP notification: {}", e))
        })?;

        debug!("Sent MCP notification: {}", notification.title);
        Ok(())
    }

    async fn is_available(&self) -> bool {
        self.sender.receiver_count() > 0
    }

    fn channel_id(&self) -> &str {
        "mcp"
    }
}

/// Console notification channel for development/testing
pub struct ConsoleNotificationChannel;

#[async_trait]
impl NotificationChannel for ConsoleNotificationChannel {
    async fn send_notification(&self, notification: &LearningNotification) -> LearningResult<()> {
        let level_str = format!("{} {:?}", notification.level.emoji(), notification.level);
        println!("\n[LEARNING] {} {}", level_str, notification.title);
        println!("   NOTE {}", notification.message);

        if !notification.actions.is_empty() {
            println!("   Actions:");
            for action in &notification.actions {
                println!("      - {}", action.label);
            }
        }

        println!("   ⏰ {}", notification.created_at.format("%H:%M:%S"));
        println!();

        Ok(())
    }

    async fn is_available(&self) -> bool {
        true
    }

    fn channel_id(&self) -> &str {
        "console"
    }
}

/// Main notification manager
pub struct NotificationManager {
    /// Configuration
    config: Arc<RwLock<NotificationConfig>>,

    /// Notification history
    history: Arc<RwLock<VecDeque<LearningNotification>>>,

    /// Active notifications (not dismissed)
    active: Arc<RwLock<HashMap<Uuid, LearningNotification>>>,

    /// Notification channels
    channels: Arc<RwLock<Vec<Box<dyn NotificationChannel>>>>,

    /// Statistics
    stats: Arc<RwLock<NotificationStats>>,

    /// Rate limiting state
    rate_limiter: Arc<RwLock<RateLimiter>>,
}

#[derive(Debug)]
struct RateLimiter {
    /// Timestamps of recent notifications
    recent_notifications: VecDeque<DateTime<Utc>>,
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            recent_notifications: VecDeque::new(),
        }
    }

    /// Check if we can send a notification without exceeding rate limits
    fn can_send(&mut self, max_per_minute: u32) -> bool {
        let now = Utc::now();
        let one_minute_ago = now - chrono::Duration::minutes(1);

        // Remove old entries
        while let Some(&front) = self.recent_notifications.front() {
            if front < one_minute_ago {
                self.recent_notifications.pop_front();
            } else {
                break;
            }
        }

        // Check if we're under the limit
        self.recent_notifications.len() < max_per_minute as usize
    }

    /// Record a notification send
    fn record_send(&mut self) {
        self.recent_notifications.push_back(Utc::now());
    }
}

#[allow(dead_code)]
impl NotificationManager {
    /// Create a new notification manager
    pub fn new(config: NotificationConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            history: Arc::new(RwLock::new(VecDeque::new())),
            active: Arc::new(RwLock::new(HashMap::new())),
            channels: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(NotificationStats {
                total_sent: 0,
                by_level: HashMap::new(),
                by_tag: HashMap::new(),
                avg_dismiss_time_seconds: 0.0,
                seen_percentage: 0.0,
                active_count: 0,
            })),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new())),
        }
    }

    /// Add a notification channel
    pub async fn add_channel(&self, channel: Box<dyn NotificationChannel>) {
        let mut channels = self.channels.write().await;
        info!("Added notification channel: {}", channel.channel_id());
        channels.push(channel);
    }

    /// Create and send a notification for a learning event
    pub async fn notify_learning_event(&self, event: &LearningEvent) -> LearningResult<()> {
        let config = self.config.read().await;

        if !config.enabled {
            return Ok(());
        }

        // Check rate limiting
        {
            let mut rate_limiter = self.rate_limiter.write().await;
            if !rate_limiter.can_send(config.max_per_minute) {
                warn!(
                    "Rate limit exceeded, skipping notification for event: {}",
                    event.id
                );
                return Ok(());
            }
            rate_limiter.record_send();
        }

        // Create notification from learning event
        let notification = self.create_notification_from_event(event, &config).await?;

        // Check if notification level meets minimum requirement
        if notification.level.priority_score() < config.min_level.priority_score() {
            debug!(
                "Notification level too low, skipping: {:?}",
                notification.level
            );
            return Ok(());
        }

        // Send through all available channels
        let channels = self.channels.read().await;
        let mut send_results = Vec::new();

        for channel in channels.iter() {
            if channel.is_available().await {
                let result = channel.send_notification(&notification).await;
                if let Err(e) = &result {
                    warn!(
                        "Failed to send notification via {}: {}",
                        channel.channel_id(),
                        e
                    );
                } else {
                    debug!("Sent notification via {}", channel.channel_id());
                }
                send_results.push(result);
            }
        }

        // Store notification
        self.store_notification(notification).await?;

        // Update statistics
        self.update_stats(&event.type_name()).await;

        Ok(())
    }

    /// Create a notification from a learning event
    async fn create_notification_from_event(
        &self,
        event: &LearningEvent,
        config: &NotificationConfig,
    ) -> LearningResult<LearningNotification> {
        let level = self.determine_notification_level(event);
        let actions = self.create_actions_for_event(event);
        let tags = self.extract_tags_from_event(event);

        let expires_at = config
            .auto_dismiss_seconds
            .map(|seconds| Utc::now() + chrono::Duration::seconds(seconds as i64));

        Ok(LearningNotification {
            id: Uuid::new_v4(),
            event_id: event.id,
            level,
            title: event.summary.clone(),
            message: event.description.clone(),
            actions,
            created_at: Utc::now(),
            expires_at,
            seen: false,
            dismissed: false,
            tags,
            metadata: HashMap::new(),
        })
    }

    /// Determine appropriate notification level for an event
    fn determine_notification_level(&self, event: &LearningEvent) -> NotificationLevel {
        match &event.event_type {
            super::events::LearningEventType::ConceptualBreakthrough { .. } => {
                if event.impact_score > 0.8 {
                    NotificationLevel::Success
                } else {
                    NotificationLevel::Info
                }
            }
            super::events::LearningEventType::AnomalyDetected { severity, .. } => match severity {
                super::events::AnomalySeverity::Critical => NotificationLevel::Critical,
                super::events::AnomalySeverity::High => NotificationLevel::Warning,
                super::events::AnomalySeverity::Medium => NotificationLevel::Warning,
                super::events::AnomalySeverity::Low => NotificationLevel::Info,
            },
            super::events::LearningEventType::NewEntityType { .. } => NotificationLevel::Info,
            super::events::LearningEventType::NewRelationshipPattern { .. } => {
                NotificationLevel::Info
            }
            super::events::LearningEventType::NewCodePattern { .. } => NotificationLevel::Success,
            super::events::LearningEventType::KnowledgeConnection { .. } => {
                NotificationLevel::Success
            }
            super::events::LearningEventType::DomainExpansion { .. } => NotificationLevel::Info,
        }
    }

    /// Create action buttons for an event
    fn create_actions_for_event(&self, event: &LearningEvent) -> Vec<NotificationAction> {
        let mut actions = Vec::new();

        // Always add view details action
        actions.push(NotificationAction {
            id: "view_details".to_string(),
            label: "查看详情".to_string(),
            action_type: ActionType::ViewDetails,
            parameters: [(
                "event_id".to_string(),
                serde_json::Value::String(event.id.to_string()),
            )]
            .into_iter()
            .collect(),
        });

        // Add search related action
        actions.push(NotificationAction {
            id: "search_related".to_string(),
            label: "搜索相关内容".to_string(),
            action_type: ActionType::SearchRelated,
            parameters: HashMap::new(),
        });

        // Add bookmark action for important events
        if event.impact_score > 0.7 {
            actions.push(NotificationAction {
                id: "bookmark".to_string(),
                label: "添加书签".to_string(),
                action_type: ActionType::Bookmark,
                parameters: HashMap::new(),
            });
        }

        // Add source action if available
        if let Some(source) = &event.context.source {
            actions.push(NotificationAction {
                id: "open_source".to_string(),
                label: "打开源文件".to_string(),
                action_type: ActionType::OpenSource,
                parameters: [(
                    "source".to_string(),
                    serde_json::Value::String(source.clone()),
                )]
                .into_iter()
                .collect(),
            });
        }

        // Always add dismiss action
        actions.push(NotificationAction {
            id: "dismiss".to_string(),
            label: "忽略".to_string(),
            action_type: ActionType::Dismiss,
            parameters: HashMap::new(),
        });

        actions
    }

    /// Extract tags from an event for categorization
    fn extract_tags_from_event(&self, event: &LearningEvent) -> Vec<String> {
        let mut tags = vec![event.type_name().to_string()];

        // Add context-based tags
        if let Some(language) = &event.context.language {
            tags.push(format!("lang:{}", language));
        }

        if let Some(project) = &event.context.project {
            tags.push(format!("project:{}", project));
        }

        // Add confidence-based tags
        if event.confidence > 0.8 {
            tags.push("high-confidence".to_string());
        } else if event.confidence < 0.5 {
            tags.push("low-confidence".to_string());
        }

        // Add impact-based tags
        if event.impact_score > 0.8 {
            tags.push("high-impact".to_string());
        }

        tags
    }

    /// Store notification in history and active list
    async fn store_notification(&self, notification: LearningNotification) -> LearningResult<()> {
        let config = self.config.read().await;

        // Add to active notifications
        {
            let mut active = self.active.write().await;
            active.insert(notification.id, notification.clone());
        }

        // Add to history
        {
            let mut history = self.history.write().await;
            history.push_back(notification);

            // Trim history if needed
            while history.len() > config.max_history {
                history.pop_front();
            }
        }

        Ok(())
    }

    /// Update notification statistics
    async fn update_stats(&self, event_type: &str) {
        let mut stats = self.stats.write().await;
        stats.total_sent += 1;

        *stats.by_tag.entry(event_type.to_string()).or_insert(0) += 1;

        let active = self.active.read().await;
        stats.active_count = active.len() as u32;
    }

    /// Mark a notification as seen
    pub async fn mark_seen(&self, notification_id: Uuid) -> LearningResult<()> {
        let mut active = self.active.write().await;
        if let Some(notification) = active.get_mut(&notification_id) {
            notification.seen = true;
            debug!("Marked notification as seen: {}", notification_id);
        }
        Ok(())
    }

    /// Dismiss a notification
    pub async fn dismiss(&self, notification_id: Uuid) -> LearningResult<()> {
        let mut active = self.active.write().await;
        if let Some(mut notification) = active.remove(&notification_id) {
            notification.dismissed = true;
            debug!("Dismissed notification: {}", notification_id);
        }
        Ok(())
    }

    /// Get all active notifications
    pub async fn get_active_notifications(&self) -> Vec<LearningNotification> {
        let active = self.active.read().await;
        let mut notifications: Vec<_> = active.values().cloned().collect();

        // Sort by priority and timestamp
        notifications.sort_by(|a, b| {
            let priority_cmp = b.level.priority_score().cmp(&a.level.priority_score());
            if priority_cmp == std::cmp::Ordering::Equal {
                b.created_at.cmp(&a.created_at)
            } else {
                priority_cmp
            }
        });

        notifications
    }

    /// Get notification statistics
    pub async fn get_stats(&self) -> NotificationStats {
        self.stats.read().await.clone()
    }

    /// Clean up expired notifications
    pub async fn cleanup_expired(&self) -> LearningResult<()> {
        let now = Utc::now();
        let mut active = self.active.write().await;

        let expired_ids: Vec<Uuid> = active
            .values()
            .filter(|n| n.expires_at.map_or(false, |exp| exp < now))
            .map(|n| n.id)
            .collect();

        for id in expired_ids {
            active.remove(&id);
            debug!("Removed expired notification: {}", id);
        }

        Ok(())
    }
}
