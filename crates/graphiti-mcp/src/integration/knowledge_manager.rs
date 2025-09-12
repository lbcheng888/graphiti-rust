//! Knowledge management integration for capturing development insights

use super::hooks::ImplicitAction;
use super::*;
use crate::types::{AddCodeEntityRequest, AddMemoryRequest};
use crate::RecordActivityRequest;
use crate::SearchCodeRequest;
use std::collections::VecDeque;

/// Manages implicit knowledge capture with deduplication and quality control
#[allow(dead_code)]
pub struct KnowledgeManager {
    integration_manager: Arc<IntegrationManager>,
    recent_captures: tokio::sync::RwLock<RecentCaptures>,
}

#[derive(Debug)]
struct RecentCaptures {
    memories: VecDeque<CapturedMemory>,
    entities: VecDeque<CapturedEntity>,
    activities: VecDeque<CapturedActivity>,
}

#[derive(Debug, Clone)]
struct CapturedMemory {
    #[allow(dead_code)]
    content: String,
    hash: u64,
    #[allow(dead_code)]
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct CapturedEntity {
    #[allow(dead_code)]
    entity_type: String,
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    file_path: String,
    hash: u64,
    #[allow(dead_code)]
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct CapturedActivity {
    #[allow(dead_code)]
    activity_type: String,
    #[allow(dead_code)]
    title: String,
    hash: u64,
    #[allow(dead_code)]
    timestamp: DateTime<Utc>,
}

#[allow(dead_code)]
impl KnowledgeManager {
    pub fn new(integration_manager: Arc<IntegrationManager>) -> Self {
        Self {
            integration_manager,
            recent_captures: tokio::sync::RwLock::new(RecentCaptures {
                memories: VecDeque::with_capacity(100),
                entities: VecDeque::with_capacity(100),
                activities: VecDeque::with_capacity(50),
            }),
        }
    }

    /// Process an implicit action with deduplication and quality checks
    pub async fn process_implicit_action(&self, action: ImplicitAction) -> Result<bool, String> {
        let config = self.integration_manager.config.read().await;

        // Check if action passes quality threshold
        if !self.meets_quality_threshold(&action, &config).await {
            return Ok(false);
        }

        // Check for duplicates
        if self.is_duplicate(&action).await {
            return Ok(false);
        }

        // Check rate limits
        let mut rate_limiter = self.integration_manager.rate_limiter.write().await;
        let key = self.get_action_key(&action);

        if !rate_limiter.check_cooldown(&key, &config) {
            return Ok(false);
        }

        // Process the action
        let result = match &action {
            ImplicitAction::AddMemory(req) => self.process_memory_capture(req.clone()).await,
            ImplicitAction::AddCodeEntity(req) => self.process_entity_capture(req.clone()).await,
            ImplicitAction::RecordActivity(req) => self.process_activity_capture(req.clone()).await,
            ImplicitAction::Suggest(suggestion) => {
                self.process_suggestion(suggestion.clone()).await
            }
        };

        if result.is_ok() {
            rate_limiter.record_capture(key);
            self.record_capture(&action).await;
        }

        result.map(|_| true)
    }

    /// Check if action meets quality threshold
    async fn meets_quality_threshold(
        &self,
        action: &ImplicitAction,
        _config: &IntegrationConfig,
    ) -> bool {
        match action {
            ImplicitAction::AddMemory(req) => {
                // Check content length and quality
                req.content.len() > 20 && !self.is_trivial_content(&req.content)
            }
            ImplicitAction::AddCodeEntity(req) => {
                // Check entity has meaningful description
                req.description.len() > 10 && req.name.len() > 2
            }
            ImplicitAction::RecordActivity(req) => {
                // Check activity has substance
                req.description.len() > 20 && req.title.len() > 5
            }
            ImplicitAction::Suggest(_) => true,
        }
    }

    /// Check if action is a duplicate of recent captures
    async fn is_duplicate(&self, action: &ImplicitAction) -> bool {
        let captures = self.recent_captures.read().await;
        let hash = self.calculate_hash(action);

        match action {
            ImplicitAction::AddMemory(_) => captures.memories.iter().any(|m| m.hash == hash),
            ImplicitAction::AddCodeEntity(_) => captures.entities.iter().any(|e| e.hash == hash),
            ImplicitAction::RecordActivity(_) => captures.activities.iter().any(|a| a.hash == hash),
            ImplicitAction::Suggest(_) => false,
        }
    }

    /// Process memory capture with enhancement
    async fn process_memory_capture(&self, mut req: AddMemoryRequest) -> Result<(), String> {
        // Enhance memory with context
        let context = self.integration_manager.get_context().await;

        // Add contextual metadata
        if req.source.is_none() {
            req.source = Some(self.infer_source(&context));
        }

        if req.memory_type.is_none() {
            req.memory_type = Some(self.infer_memory_type(&req.content));
        }

        // Add the memory
        self.integration_manager
            .graphiti_service
            .add_memory(req)
            .await
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Process entity capture with enrichment
    async fn process_entity_capture(&self, mut req: AddCodeEntityRequest) -> Result<(), String> {
        // Enhance entity with additional context
        let context = self.integration_manager.get_context().await;

        // Infer complexity if not provided
        if req.complexity.is_none() {
            req.complexity = Some(self.estimate_complexity(&req));
        }

        // Infer importance based on usage patterns
        if req.importance.is_none() {
            req.importance = Some(self.estimate_importance(&req, &context).await);
        }

        // Check if entity already exists
        let search_req = SearchCodeRequest {
            query: req.name.clone(),
            entity_type: Some(req.entity_type.clone()),
            language: req.language.clone(),
            framework: req.framework.clone(),
            limit: Some(1),
        };

        // Only add if not already exists
        if let Ok(results) = self
            .integration_manager
            .graphiti_service
            .search_code(search_req)
            .await
        {
            if results.results.is_empty() {
                self.integration_manager
                    .graphiti_service
                    .add_code_entity(req)
                    .await
                    .map_err(|e| e.to_string())?;
            }
        }

        Ok(())
    }

    /// Process activity capture with validation
    async fn process_activity_capture(&self, mut req: RecordActivityRequest) -> Result<(), String> {
        // Enhance activity with metrics
        let context = self.integration_manager.get_context().await;

        // Add related entities from context
        if req.related_entities.is_none() {
            let entity_ids: Vec<String> = context
                .related_entities
                .iter()
                .map(|id| id.to_string())
                .collect();

            if !entity_ids.is_empty() {
                req.related_entities = Some(entity_ids);
            }
        }

        // Record the activity
        self.integration_manager
            .graphiti_service
            .record_activity(req)
            .await
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Process suggestion display
    async fn process_suggestion(&self, suggestion: String) -> Result<(), String> {
        let config = self.integration_manager.config.read().await;

        if config.notifications.show_suggestions {
            // In real implementation, this would show in VS Code UI
            tracing::info!("💡 Suggestion: {}", suggestion);
        }

        Ok(())
    }

    /// Record successful capture for deduplication
    async fn record_capture(&self, action: &ImplicitAction) {
        let mut captures = self.recent_captures.write().await;
        let now = Utc::now();

        match action {
            ImplicitAction::AddMemory(req) => {
                let capture = CapturedMemory {
                    content: req.content.clone(),
                    hash: self.calculate_hash(action),
                    timestamp: now,
                };

                captures.memories.push_back(capture);
                if captures.memories.len() > 100 {
                    captures.memories.pop_front();
                }
            }
            ImplicitAction::AddCodeEntity(req) => {
                let capture = CapturedEntity {
                    entity_type: req.entity_type.clone(),
                    name: req.name.clone(),
                    file_path: req.file_path.clone().unwrap_or_default(),
                    hash: self.calculate_hash(action),
                    timestamp: now,
                };

                captures.entities.push_back(capture);
                if captures.entities.len() > 100 {
                    captures.entities.pop_front();
                }
            }
            ImplicitAction::RecordActivity(req) => {
                let capture = CapturedActivity {
                    activity_type: req.activity_type.clone(),
                    title: req.title.clone(),
                    hash: self.calculate_hash(action),
                    timestamp: now,
                };

                captures.activities.push_back(capture);
                if captures.activities.len() > 50 {
                    captures.activities.pop_front();
                }
            }
            _ => {}
        }
    }

    // Helper methods

    fn get_action_key(&self, action: &ImplicitAction) -> String {
        match action {
            ImplicitAction::AddMemory(req) => {
                format!("memory:{}", &req.content[..20.min(req.content.len())])
            }
            ImplicitAction::AddCodeEntity(req) => {
                format!("entity:{}:{}", req.entity_type, req.name)
            }
            ImplicitAction::RecordActivity(req) => {
                format!("activity:{}:{}", req.activity_type, req.title)
            }
            ImplicitAction::Suggest(s) => format!("suggest:{}", &s[..20.min(s.len())]),
        }
    }

    fn calculate_hash(&self, action: &ImplicitAction) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();

        match action {
            ImplicitAction::AddMemory(req) => {
                req.content.hash(&mut hasher);
                req.source.hash(&mut hasher);
            }
            ImplicitAction::AddCodeEntity(req) => {
                req.entity_type.hash(&mut hasher);
                req.name.hash(&mut hasher);
                req.file_path.hash(&mut hasher);
            }
            ImplicitAction::RecordActivity(req) => {
                req.activity_type.hash(&mut hasher);
                req.title.hash(&mut hasher);
            }
            ImplicitAction::Suggest(s) => {
                s.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    fn is_trivial_content(&self, content: &str) -> bool {
        let trivial_patterns = ["test", "todo", "fixme", "debug", "temp", "tmp"];
        let content_lower = content.to_lowercase();

        content_lower.len() < 30 && trivial_patterns.iter().any(|p| content_lower.contains(p))
    }

    fn infer_source(&self, context: &DevelopmentContext) -> String {
        if let Some(activity) = &context.activity_type {
            match activity {
                ActivityType::BugFix => "Bug Fix",
                ActivityType::FeatureDevelopment => "Feature Development",
                ActivityType::Refactoring => "Code Refactoring",
                ActivityType::Testing => "Test Development",
                ActivityType::Documentation => "Documentation",
                _ => "Development",
            }
        } else {
            "Implicit Capture"
        }
        .to_string()
    }

    fn infer_memory_type(&self, content: &str) -> String {
        let content_lower = content.to_lowercase();

        if content_lower.contains("bug")
            || content_lower.contains("fix")
            || content_lower.contains("error")
        {
            "bug_fix"
        } else if content_lower.contains("decision")
            || content_lower.contains("chose")
            || content_lower.contains("because")
        {
            "decision"
        } else if content_lower.contains("learned") || content_lower.contains("discovered") {
            "learning"
        } else if content_lower.contains("todo") || content_lower.contains("plan") {
            "planning"
        } else {
            "general"
        }
        .to_string()
    }

    fn estimate_complexity(&self, req: &AddCodeEntityRequest) -> u8 {
        let mut complexity = 5;

        // Adjust based on entity type
        match req.entity_type.as_str() {
            "Class" => complexity += 2,
            "Module" => complexity += 1,
            "Api" => complexity += 1,
            _ => {}
        }

        // Adjust based on line count if available
        if let Some((start, end)) = req.line_range {
            let lines = end - start;
            if lines > 100 {
                complexity += 2;
            } else if lines > 50 {
                complexity += 1;
            }
        }

        complexity.min(10)
    }

    async fn estimate_importance(
        &self,
        req: &AddCodeEntityRequest,
        context: &DevelopmentContext,
    ) -> u8 {
        let mut importance = 5;

        // Higher importance for APIs and main classes
        match req.entity_type.as_str() {
            "Api" => importance += 3,
            "Class" if req.name.contains("Service") || req.name.contains("Controller") => {
                importance += 2
            }
            "Module" => importance += 1,
            _ => {}
        }

        // Higher importance if recently accessed multiple times
        let access_count = context
            .recent_files
            .iter()
            .filter(|f| {
                req.file_path
                    .as_ref()
                    .map(|fp| f.contains(fp))
                    .unwrap_or(false)
            })
            .count();

        if access_count > 3 {
            importance += 1;
        }

        importance.min(10)
    }
}
