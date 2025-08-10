//! Integration of learning detection with MCP server

use super::learning::LearningContext;
use super::learning::LearningDetector;
use super::learning::NotificationManager;
use super::AddMemoryRequest;
use super::AddMemoryResponse;
use super::GraphitiService;
use graphiti_core::error::Result as GraphitiResult;
use std::sync::Arc;
use tracing::debug;
use tracing::info;
use tracing::warn;

use uuid::Uuid;

/// Enhanced GraphitiService that integrates learning detection
pub struct LearningAwareGraphitiService {
    /// The underlying graphiti service
    inner: Arc<dyn GraphitiService>,

    /// Learning detector
    detector: Arc<dyn LearningDetector>,

    /// Notification manager
    notification_manager: Arc<NotificationManager>,
}

impl LearningAwareGraphitiService {
    /// Create a new learning-aware service
    pub fn new(
        inner: Arc<dyn GraphitiService>,
        detector: Arc<dyn LearningDetector>,
        notification_manager: Arc<NotificationManager>,
    ) -> Self {
        Self {
            inner,
            detector,
            notification_manager,
        }
    }

    /// Create learning context from request and environment
    fn create_learning_context(
        &self,
        request: &AddMemoryRequest,
        session_id: Option<Uuid>,
    ) -> LearningContext {
        LearningContext {
            activity: Some("add_memory".to_string()),
            source: request.source.clone(),
            language: self.infer_language_from_content(&request.content),
            project: self.infer_project_from_source(&request.source),
            session_id,
            related_events: Vec::new(),
            context_confidence: 0.8,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Infer programming language from content
    fn infer_language_from_content(&self, content: &str) -> Option<String> {
        let content_lower = content.to_lowercase();

        // Simple heuristics for language detection
        if content_lower.contains("fn ") && content_lower.contains("rust") {
            Some("rust".to_string())
        } else if content_lower.contains("def ") && content_lower.contains("python") {
            Some("python".to_string())
        } else if content_lower.contains("function") && content_lower.contains("javascript") {
            Some("javascript".to_string())
        } else if content_lower.contains("class") && content_lower.contains("java") {
            Some("java".to_string())
        } else if content_lower.contains("async") && content_lower.contains("await") {
            Some("typescript".to_string())
        } else {
            None
        }
    }

    /// Infer project name from source
    fn infer_project_from_source(&self, source: &Option<String>) -> Option<String> {
        source.as_ref().and_then(|s| {
            if s.contains("/") {
                // Extract project name from path
                s.split('/')
                    .find(|part| !part.is_empty() && *part != "src" && *part != "lib")
                    .map(|s| s.to_string())
            } else {
                Some(s.clone())
            }
        })
    }
}

#[async_trait::async_trait]
impl GraphitiService for LearningAwareGraphitiService {
    async fn add_memory(&self, req: AddMemoryRequest) -> GraphitiResult<AddMemoryResponse> {
        info!("🧠 Learning-aware memory addition starting");

        // Call the inner service first
        let response = self.inner.add_memory(req.clone()).await?;

        // Create learning context
        let context = self.create_learning_context(&req, None);

        // Perform learning detection
        match self
            .detector
            .detect_learning(&req, &response, &context)
            .await
        {
            Ok(events) => {
                if !events.is_empty() {
                    info!("🎯 Detected {} learning events", events.len());

                    // Send notifications for each learning event
                    for event in &events {
                        if let Err(e) = self.notification_manager.notify_learning_event(event).await
                        {
                            warn!("Failed to send learning notification: {}", e);
                        } else {
                            debug!("📢 Sent notification for learning event: {}", event.summary);
                        }
                    }
                } else {
                    debug!("No learning events detected");
                }
            }
            Err(e) => {
                warn!("Learning detection failed: {}", e);
                // Don't fail the memory addition if learning detection fails
            }
        }

        info!("✅ Learning-aware memory addition completed");
        Ok(response)
    }

    // Delegate all other methods to the inner service
    async fn search_memory(
        &self,
        req: super::SearchMemoryRequest,
    ) -> GraphitiResult<super::SearchMemoryResponse> {
        self.inner.search_memory(req).await
    }

    async fn get_memory(&self, id: Uuid) -> GraphitiResult<Option<super::MemoryNode>> {
        self.inner.get_memory(id).await
    }

    async fn get_related(
        &self,
        id: Uuid,
        depth: usize,
    ) -> GraphitiResult<Vec<super::RelatedMemory>> {
        self.inner.get_related(id, depth).await
    }

    async fn search_memory_facts(
        &self,
        query: String,
        limit: Option<usize>,
    ) -> GraphitiResult<Vec<super::SimpleExtractedRelationship>> {
        self.inner.search_memory_facts(query, limit).await
    }

    async fn search_memory_facts_json(
        &self,
        query: String,
        limit: Option<usize>,
    ) -> GraphitiResult<Vec<serde_json::Value>> {
        self.inner.search_memory_facts_json(query, limit).await
    }

    async fn delete_episode(&self, id: Uuid) -> GraphitiResult<bool> {
        self.inner.delete_episode(id).await
    }

    async fn get_episodes(&self, last_n: usize) -> GraphitiResult<Vec<super::EpisodeNode>> {
        self.inner.get_episodes(last_n).await
    }

    async fn clear_graph(&self) -> GraphitiResult<()> {
        self.inner.clear_graph().await
    }

    async fn get_entity_edge_json(
        &self,
        id: uuid::Uuid,
    ) -> GraphitiResult<Option<serde_json::Value>> {
        self.inner.get_entity_edge_json(id).await
    }

    async fn delete_entity_edge_by_uuid(&self, id: uuid::Uuid) -> GraphitiResult<bool> {
        self.inner.delete_entity_edge_by_uuid(id).await
    }

    async fn add_code_entity(
        &self,
        req: super::AddCodeEntityRequest,
    ) -> GraphitiResult<super::AddCodeEntityResponse> {
        self.inner.add_code_entity(req).await
    }

    async fn record_activity(
        &self,
        req: super::RecordActivityRequest,
    ) -> GraphitiResult<super::RecordActivityResponse> {
        self.inner.record_activity(req).await
    }

    async fn search_code(
        &self,
        req: super::SearchCodeRequest,
    ) -> GraphitiResult<super::SearchCodeResponse> {
        self.inner.search_code(req).await
    }

    async fn batch_add_code_entities(
        &self,
        req: super::BatchAddCodeEntitiesRequest,
    ) -> GraphitiResult<super::BatchAddCodeEntitiesResponse> {
        self.inner.batch_add_code_entities(req).await
    }

    async fn batch_record_activities(
        &self,
        req: super::BatchRecordActivitiesRequest,
    ) -> GraphitiResult<super::BatchRecordActivitiesResponse> {
        self.inner.batch_record_activities(req).await
    }

    async fn get_context_suggestions(
        &self,
        req: super::ContextSuggestionRequest,
    ) -> GraphitiResult<super::ContextSuggestionResponse> {
        self.inner.get_context_suggestions(req).await
    }
}
