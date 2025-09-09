//! Tool hooks for intercepting Claude Code operations

use super::*;
use crate::AddCodeEntityRequest;
use crate::AddMemoryRequest;
use crate::RecordActivityRequest;
use async_trait::async_trait;
use serde_json::Value;
use tracing::debug;
use tracing::info;

/// Trait for tool hooks
#[async_trait]
#[allow(dead_code)]
pub trait ToolHook: Send + Sync {
    /// Called before a tool is executed
    async fn pre_execute(&self, tool: &str, params: &Value) -> HookResult;

    /// Called after a tool is executed
    async fn post_execute(&self, tool: &str, params: &Value, result: &Value) -> HookResult;
}

#[derive(Debug)]
pub struct HookResult {
    /// Whether to continue with the original operation
    #[allow(dead_code)]
    pub continue_execution: bool,

    /// Additional actions to perform
    #[allow(dead_code)]
    pub actions: Vec<ImplicitAction>,
}

#[derive(Debug, Clone)]
pub enum ImplicitAction {
    /// Add a memory automatically
    #[allow(dead_code)]
    AddMemory(AddMemoryRequest),

    /// Add a code entity
    #[allow(dead_code)]
    AddCodeEntity(AddCodeEntityRequest),

    /// Record an activity
    #[allow(dead_code)]
    RecordActivity(RecordActivityRequest),

    /// Provide a suggestion
    #[allow(dead_code)]
    Suggest(String),
}

/// Main hook implementation that integrates with IntegrationManager
pub struct GraphitiHook {
    #[allow(dead_code)]
    integration_manager: Arc<IntegrationManager>,
}

#[allow(dead_code)]
impl GraphitiHook {
    pub fn new(integration_manager: Arc<IntegrationManager>) -> Self {
        Self {
            integration_manager,
        }
    }
}

#[async_trait]
impl ToolHook for GraphitiHook {
    async fn pre_execute(&self, tool: &str, params: &Value) -> HookResult {
        let config = self.integration_manager.config.read().await;
        if !config.enabled {
            return HookResult {
                continue_execution: true,
                actions: vec![],
            };
        }

        // Update context
        self.integration_manager.update_context(tool, params).await;

        let actions = vec![];

        // Tool-specific pre-processing
        match tool {
            "Edit" | "MultiEdit" => {
                // Before editing, search for similar code patterns
                if config.auto_suggestions {
                    if let Some(file_path) = params.get("file_path").and_then(|p| p.as_str()) {
                        if self
                            .integration_manager
                            .should_monitor_file(file_path)
                            .await
                        {
                            // Search for similar edits in history
                            debug!("Searching for similar code patterns before edit");
                            // This would trigger a search and potentially provide suggestions
                        }
                    }
                }
            }
            "Bash" => {
                // Detect if running tests, builds, etc.
                if let Some(command) = params.get("command").and_then(|c| c.as_str()) {
                    if command.contains("test")
                        || command.contains("pytest")
                        || command.contains("jest")
                    {
                        debug!("Detected test execution");
                        // Update activity type
                        let mut context = self.integration_manager.context.write().await;
                        context.activity_type = Some(ActivityType::Testing);
                        context.confidence = 0.8;
                    }
                }
            }
            _ => {}
        }

        HookResult {
            continue_execution: true,
            actions,
        }
    }

    async fn post_execute(&self, tool: &str, params: &Value, result: &Value) -> HookResult {
        let config = self.integration_manager.config.read().await;
        if !config.enabled {
            return HookResult {
                continue_execution: true,
                actions: vec![],
            };
        }

        let mut actions = vec![];

        // Check rate limits
        let mut rate_limiter = self.integration_manager.rate_limiter.write().await;
        if !rate_limiter.check_rate_limit(&config) {
            debug!("Rate limit exceeded, skipping automatic capture");
            return HookResult {
                continue_execution: true,
                actions,
            };
        }

        // Tool-specific post-processing
        match tool {
            "Edit" | "MultiEdit" => {
                if config.auto_capture_entities {
                    actions.extend(self.handle_edit_completion(params, result).await);
                }
            }
            "Write" => {
                if config.auto_capture_entities {
                    actions.extend(self.handle_file_creation(params, result).await);
                }
            }
            "Bash" => {
                if config.auto_record_activities {
                    actions.extend(self.handle_command_execution(params, result).await);
                }
            }
            "Grep" | "Glob" => {
                if let Some(pattern) = params.get("pattern").and_then(|p| p.as_str()) {
                    debug!("Recording search pattern: {}", pattern);
                }
            }
            _ => {}
        }

        HookResult {
            continue_execution: true,
            actions,
        }
    }
}

impl GraphitiHook {
    /// Handle completion of file edits
    #[allow(dead_code)]
    async fn handle_edit_completion(&self, params: &Value, _result: &Value) -> Vec<ImplicitAction> {
        let mut actions = vec![];

        if let Some(file_path) = params.get("file_path").and_then(|p| p.as_str()) {
            if !self
                .integration_manager
                .should_monitor_file(file_path)
                .await
            {
                return actions;
            }

            // Analyze the edit to detect what was changed
            if let (Some(old_str), Some(new_str)) = (
                params.get("old_string").and_then(|s| s.as_str()),
                params.get("new_string").and_then(|s| s.as_str()),
            ) {
                // Detect function/class additions
                if self.looks_like_function(new_str) && !self.looks_like_function(old_str) {
                    info!("Detected new function in {}", file_path);

                    if let Some(entity) = self.extract_function_entity(new_str, file_path).await {
                        actions.push(ImplicitAction::AddCodeEntity(entity));
                    }
                }

                // Detect bug fixes
                if self.looks_like_bug_fix(old_str, new_str).await {
                    info!("Detected potential bug fix in {}", file_path);

                    let activity = RecordActivityRequest {
                        activity_type: "BugFix".to_string(),
                        title: format!("Fixed issue in {}", file_path),
                        description: self.summarize_change(old_str, new_str),
                        developer: "Claude Code User".to_string(),
                        project: self.detect_project_name().await,
                        related_entities: None,
                        duration_minutes: None,
                        difficulty: None,
                        quality: None,
                    };

                    actions.push(ImplicitAction::RecordActivity(activity));
                }
            }
        }

        actions
    }

    /// Handle new file creation
    #[allow(dead_code)]
    async fn handle_file_creation(&self, params: &Value, _result: &Value) -> Vec<ImplicitAction> {
        let mut actions = vec![];

        if let (Some(file_path), Some(content)) = (
            params.get("file_path").and_then(|p| p.as_str()),
            params.get("content").and_then(|c| c.as_str()),
        ) {
            if !self
                .integration_manager
                .should_monitor_file(file_path)
                .await
            {
                return actions;
            }

            info!("New file created: {}", file_path);

            // Detect what type of file was created
            if file_path.ends_with(".test.js")
                || file_path.ends_with("_test.py")
                || file_path.contains("/test/")
            {
                let activity = RecordActivityRequest {
                    activity_type: "Testing".to_string(),
                    title: format!("Created test file: {}", file_path),
                    description: "New test file added to the project".to_string(),
                    developer: "Claude Code User".to_string(),
                    project: self.detect_project_name().await,
                    related_entities: None,
                    duration_minutes: None,
                    difficulty: None,
                    quality: None,
                };

                actions.push(ImplicitAction::RecordActivity(activity));
            }

            // Extract entities from the new file
            if let Some(entities) = self.extract_entities_from_file(content, file_path).await {
                for entity in entities {
                    actions.push(ImplicitAction::AddCodeEntity(entity));
                }
            }
        }

        actions
    }

    /// Handle command execution results
    #[allow(dead_code)]
    async fn handle_command_execution(
        &self,
        params: &Value,
        result: &Value,
    ) -> Vec<ImplicitAction> {
        let mut actions = vec![];

        if let Some(command) = params.get("command").and_then(|c| c.as_str()) {
            // Detect successful deployments
            if command.contains("deploy") || command.contains("push") {
                info!("Detected deployment command");

                let activity = RecordActivityRequest {
                    activity_type: "Deployment".to_string(),
                    title: "Code deployment".to_string(),
                    description: format!("Executed: {}", command),
                    developer: "Claude Code User".to_string(),
                    project: self.detect_project_name().await,
                    related_entities: None,
                    duration_minutes: None,
                    difficulty: None,
                    quality: None,
                };

                actions.push(ImplicitAction::RecordActivity(activity));
            }

            // Detect test execution
            if command.contains("test") && self.test_passed(result) {
                let memory = AddMemoryRequest {
                    content: format!("All tests passed for command: {}", command),
                    name: Some("Test Success".to_string()),
                    source: Some("Automated Testing".to_string()),
                    memory_type: Some("test_result".to_string()),
                    metadata: None,
                    group_id: None,
                    timestamp: None,
                };

                actions.push(ImplicitAction::AddMemory(memory));
            }
        }

        actions
    }

    // Helper methods

    #[allow(dead_code)]
    fn looks_like_function(&self, code: &str) -> bool {
        // Simple heuristic for function detection
        code.contains("fn ")
            || code.contains("def ")
            || code.contains("function ")
            || code.contains("func ")
            || code.contains("public ")
            || code.contains("private ")
    }

    #[allow(dead_code)]
    async fn looks_like_bug_fix(&self, old_code: &str, new_code: &str) -> bool {
        // Simple heuristics for bug fix detection
        let old_lower = old_code.to_lowercase();
        let new_lower = new_code.to_lowercase();

        // Common bug fix patterns
        (old_lower.contains("null") && !new_lower.contains("null"))
            || (old_lower.contains("undefined") && !new_lower.contains("undefined"))
            || (new_lower.contains("fix") || new_lower.contains("fixed"))
            || (new_lower.contains("handle") && !old_lower.contains("handle"))
            || (new_lower.contains("check") && !old_lower.contains("check"))
    }

    #[allow(dead_code)]
    fn summarize_change(&self, old_code: &str, new_code: &str) -> String {
        format!(
            "Changed {} lines",
            old_code.lines().count().abs_diff(new_code.lines().count())
        )
    }

    #[allow(dead_code)]
    async fn extract_function_entity(
        &self,
        code: &str,
        file_path: &str,
    ) -> Option<AddCodeEntityRequest> {
        // Simple function extraction (would be enhanced with proper parsing)
        let lines: Vec<&str> = code.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            if line.contains("fn ") || line.contains("def ") || line.contains("function ") {
                let name = self.extract_function_name(line)?;

                return Some(AddCodeEntityRequest {
                    entity_type: "Function".to_string(),
                    name,
                    description: "Automatically detected function".to_string(),
                    file_path: Some(file_path.to_string()),
                    line_range: Some((i as u32 + 1, i as u32 + 10)), // Approximate
                    language: self.detect_language(file_path),
                    framework: None,
                    complexity: None,
                    importance: None,
                });
            }
        }

        None
    }

    #[allow(dead_code)]
    fn extract_function_name(&self, line: &str) -> Option<String> {
        // Simple regex-like extraction
        let patterns = vec!["fn ", "def ", "function ", "func "];

        for pattern in patterns {
            if let Some(start) = line.find(pattern) {
                let name_start = start + pattern.len();
                let rest = &line[name_start..];
                let name_end = rest
                    .find(|c: char| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(rest.len());
                let name = &rest[..name_end];
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }

        None
    }

    #[allow(dead_code)]
    async fn extract_entities_from_file(
        &self,
        _content: &str,
        _file_path: &str,
    ) -> Option<Vec<AddCodeEntityRequest>> {
        // This would use proper AST parsing in a real implementation
        None
    }

    #[allow(dead_code)]
    fn detect_language(&self, file_path: &str) -> Option<String> {
        let ext = file_path.split('.').last()?;

        match ext {
            "rs" => Some("Rust".to_string()),
            "py" => Some("Python".to_string()),
            "js" => Some("JavaScript".to_string()),
            "ts" => Some("TypeScript".to_string()),
            "java" => Some("Java".to_string()),
            "go" => Some("Go".to_string()),
            "cpp" | "cc" => Some("C++".to_string()),
            "c" => Some("C".to_string()),
            _ => None,
        }
    }

    #[allow(dead_code)]
    async fn detect_project_name(&self) -> String {
        // Would read from package.json, Cargo.toml, etc.
        "Current Project".to_string()
    }

    #[allow(dead_code)]
    fn test_passed(&self, result: &Value) -> bool {
        // Check if test command was successful
        if let Some(output) = result.as_str() {
            !output.contains("FAILED")
                && !output.contains("error")
                && (output.contains("passed") || output.contains("PASSED") || output.contains("ok"))
        } else {
            false
        }
    }
}
