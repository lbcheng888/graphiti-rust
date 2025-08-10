//! Automatic activity detection and classification

use super::*;
use crate::RecordActivityRequest;
use chrono::Duration;
use serde_json::Value;

/// Detects and classifies development activities automatically
#[allow(dead_code)]
pub struct ActivityDetector {
    integration_manager: Arc<IntegrationManager>,
    current_session: tokio::sync::RwLock<Option<DevelopmentSession>>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DevelopmentSession {
    pub activity_type: ActivityType,
    pub start_time: DateTime<Utc>,
    pub last_action_time: DateTime<Utc>,
    pub file_set: std::collections::HashSet<String>,
    pub action_count: usize,
    pub confidence: f32,
}

#[allow(dead_code)]
impl ActivityDetector {
    pub fn new(integration_manager: Arc<IntegrationManager>) -> Self {
        Self {
            integration_manager,
            current_session: tokio::sync::RwLock::new(None),
        }
    }

    /// Process a tool action and update activity detection
    pub async fn process_action(
        &self,
        tool: &str,
        params: &Value,
    ) -> Option<RecordActivityRequest> {
        let config = self.integration_manager.config.read().await;
        if !config.auto_record_activities {
            return None;
        }

        let now = Utc::now();
        let mut session_guard = self.current_session.write().await;

        // Check if we should start a new session
        if let Some(session) = session_guard.as_ref() {
            if now.signed_duration_since(session.last_action_time) > Duration::minutes(10) {
                // Session timeout - complete the previous session
                let activity = self.complete_session(session);
                *session_guard = None;
                return Some(activity);
            }
        }

        // Update or create session
        match session_guard.as_mut() {
            Some(session) => {
                self.update_session(session, tool, params, now).await;

                // Check if session is complete
                if self.is_session_complete(session) {
                    let activity = self.complete_session(session);
                    *session_guard = None;
                    return Some(activity);
                }
            }
            None => {
                // Start new session
                if let Some(new_session) = self.start_new_session(tool, params, now).await {
                    *session_guard = Some(new_session);
                }
            }
        }

        None
    }

    /// Start a new development session
    async fn start_new_session(
        &self,
        tool: &str,
        params: &Value,
        now: DateTime<Utc>,
    ) -> Option<DevelopmentSession> {
        let activity_type = self.infer_activity_from_action(tool, params).await?;

        let mut session = DevelopmentSession {
            activity_type,
            start_time: now,
            last_action_time: now,
            file_set: std::collections::HashSet::new(),
            action_count: 1,
            confidence: 0.5,
        };

        // Add files from this action
        if let Some(file) = self.extract_file_from_params(params) {
            session.file_set.insert(file);
        }

        Some(session)
    }

    /// Update existing session with new action
    async fn update_session(
        &self,
        session: &mut DevelopmentSession,
        tool: &str,
        params: &Value,
        now: DateTime<Utc>,
    ) {
        session.last_action_time = now;
        session.action_count += 1;

        // Add any files referenced
        if let Some(file) = self.extract_file_from_params(params) {
            session.file_set.insert(file);
        }

        // Update activity type confidence based on consistent actions
        if let Some(inferred_type) = self.infer_activity_from_action(tool, params).await {
            if inferred_type == session.activity_type {
                session.confidence = (session.confidence + 0.1).min(1.0);
            } else {
                session.confidence = (session.confidence - 0.1).max(0.0);
            }
        }

        // Refine activity type based on accumulated evidence
        if session.confidence < 0.3 {
            if let Some(new_type) = self.refine_activity_type(session, tool, params).await {
                session.activity_type = new_type;
                session.confidence = 0.5;
            }
        }
    }

    /// Check if a session is complete
    fn is_session_complete(&self, session: &DevelopmentSession) -> bool {
        // Complete session if:
        // - High confidence and significant work done
        // - Detected completion patterns (e.g., running tests after implementation)

        session.action_count > 10 && session.confidence > 0.7
    }

    /// Complete a session and create activity record
    fn complete_session(&self, session: &DevelopmentSession) -> RecordActivityRequest {
        let duration = session
            .last_action_time
            .signed_duration_since(session.start_time)
            .num_minutes() as u32;

        let title = match session.activity_type {
            ActivityType::FeatureDevelopment => "Feature implementation session",
            ActivityType::BugFix => "Bug fixing session",
            ActivityType::Refactoring => "Code refactoring session",
            ActivityType::Testing => "Test development session",
            ActivityType::Documentation => "Documentation update session",
            ActivityType::CodeReview => "Code review session",
            ActivityType::Debugging => "Debugging session",
            ActivityType::Learning => "Learning and exploration session",
        };

        let description = format!(
            "Worked on {} files over {} minutes with {} actions. Confidence: {:.0}%",
            session.file_set.len(),
            duration,
            session.action_count,
            session.confidence * 100.0
        );

        RecordActivityRequest {
            activity_type: format!("{:?}", session.activity_type),
            title: title.to_string(),
            description,
            developer: "Claude Code User".to_string(),
            project: self.detect_project_name().to_string(),
            related_entities: None,
            duration_minutes: Some(duration),
            difficulty: self.estimate_difficulty(session),
            quality: None,
        }
    }

    /// Infer activity type from a single action
    async fn infer_activity_from_action(&self, tool: &str, params: &Value) -> Option<ActivityType> {
        match tool {
            "Bash" => {
                if let Some(cmd) = params.get("command").and_then(|c| c.as_str()) {
                    let cmd_lower = cmd.to_lowercase();

                    if cmd_lower.contains("test")
                        || cmd_lower.contains("pytest")
                        || cmd_lower.contains("jest")
                        || cmd_lower.contains("cargo test")
                    {
                        return Some(ActivityType::Testing);
                    }

                    if cmd_lower.contains("git") && cmd_lower.contains("diff") {
                        return Some(ActivityType::CodeReview);
                    }

                    if cmd_lower.contains("debug")
                        || cmd_lower.contains("gdb")
                        || cmd_lower.contains("lldb")
                    {
                        return Some(ActivityType::Debugging);
                    }
                }
            }
            "Edit" | "Write" => {
                if let Some(file) = params.get("file_path").and_then(|f| f.as_str()) {
                    if file.ends_with(".md") || file.contains("README") || file.contains("docs/") {
                        return Some(ActivityType::Documentation);
                    }

                    if file.contains("test")
                        || file.ends_with("_test.py")
                        || file.ends_with(".test.js")
                    {
                        return Some(ActivityType::Testing);
                    }
                }

                // Check content for clues
                if let Some(content) = params
                    .get("new_string")
                    .or_else(|| params.get("content"))
                    .and_then(|c| c.as_str())
                {
                    let content_lower = content.to_lowercase();

                    if content_lower.contains("fix")
                        || content_lower.contains("bug")
                        || content_lower.contains("issue")
                    {
                        return Some(ActivityType::BugFix);
                    }

                    if content_lower.contains("refactor") || content_lower.contains("cleanup") {
                        return Some(ActivityType::Refactoring);
                    }
                }
            }
            "Grep" => {
                if let Some(pattern) = params.get("pattern").and_then(|p| p.as_str()) {
                    let pattern_lower = pattern.to_lowercase();

                    if pattern_lower.contains("error")
                        || pattern_lower.contains("exception")
                        || pattern_lower.contains("bug")
                    {
                        return Some(ActivityType::Debugging);
                    }

                    if pattern_lower.contains("todo") || pattern_lower.contains("fixme") {
                        return Some(ActivityType::BugFix);
                    }
                }
            }
            _ => {}
        }

        // Default to feature development for file modifications
        if matches!(tool, "Edit" | "Write" | "MultiEdit") {
            Some(ActivityType::FeatureDevelopment)
        } else {
            None
        }
    }

    /// Refine activity type based on accumulated session data
    async fn refine_activity_type(
        &self,
        session: &DevelopmentSession,
        _tool: &str,
        _params: &Value,
    ) -> Option<ActivityType> {
        // Analyze file patterns
        let test_files = session
            .file_set
            .iter()
            .filter(|f| f.contains("test") || f.ends_with("_test.py") || f.ends_with(".test.js"))
            .count();

        let doc_files = session
            .file_set
            .iter()
            .filter(|f| f.ends_with(".md") || f.contains("README") || f.contains("docs/"))
            .count();

        let total_files = session.file_set.len();

        if total_files > 0 {
            if test_files as f32 / total_files as f32 > 0.6 {
                return Some(ActivityType::Testing);
            }

            if doc_files as f32 / total_files as f32 > 0.6 {
                return Some(ActivityType::Documentation);
            }
        }

        None
    }

    /// Extract file path from tool parameters
    fn extract_file_from_params(&self, params: &Value) -> Option<String> {
        params
            .get("file_path")
            .or_else(|| params.get("path"))
            .and_then(|p| p.as_str())
            .map(|s| s.to_string())
    }

    /// Estimate difficulty based on session data
    fn estimate_difficulty(&self, session: &DevelopmentSession) -> Option<u8> {
        let base_difficulty = match session.activity_type {
            ActivityType::BugFix => 5,
            ActivityType::FeatureDevelopment => 6,
            ActivityType::Refactoring => 7,
            ActivityType::Debugging => 8,
            _ => 4,
        };

        // Adjust based on duration and file count
        let duration_factor = (session
            .last_action_time
            .signed_duration_since(session.start_time)
            .num_hours() as u8)
            .min(3);

        let file_factor = (session.file_set.len() / 5) as u8;

        Some((base_difficulty + duration_factor + file_factor).min(10))
    }

    fn detect_project_name(&self) -> String {
        // This would be enhanced to actually detect from package.json, Cargo.toml, etc.
        "Current Project".to_string()
    }
}
