//! Learning detection algorithms and implementations

use super::LearningResult;

use super::events::LearningContext;
use super::events::LearningEvent;
use super::events::LearningEventType;
use crate::types::{
    AddMemoryRequest, AddMemoryResponse, SimpleExtractedEntity, SimpleExtractedRelationship,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;
use tracing::info;

/// Trait for learning detection algorithms
#[async_trait]
#[allow(dead_code)]
pub trait LearningDetector: Send + Sync {
    /// Analyze a memory addition request and detect learning events
    async fn detect_learning(
        &self,
        request: &AddMemoryRequest,
        response: &AddMemoryResponse,
        context: &LearningContext,
    ) -> LearningResult<Vec<LearningEvent>>;

    /// Update internal state based on new information
    async fn update_knowledge_state(
        &self,
        entities: &[SimpleExtractedEntity],
        relationships: &[SimpleExtractedRelationship],
    ) -> LearningResult<()>;

    /// Get detector statistics
    async fn get_detector_stats(&self) -> LearningResult<DetectorStats>;
}

/// Statistics about detector performance
#[derive(Debug, Clone)]
pub struct DetectorStats {
    pub total_analyses: u64,
    pub events_detected: u64,
    #[allow(dead_code)]
    pub false_positives: u64,
    pub confidence_scores: Vec<f32>,
    pub processing_times_ms: Vec<u64>,
}

impl Default for DetectorStats {
    fn default() -> Self {
        Self {
            total_analyses: 0,
            events_detected: 0,
            false_positives: 0,
            confidence_scores: Vec::new(),
            processing_times_ms: Vec::new(),
        }
    }
}

/// Comprehensive learning detector with multiple detection algorithms
pub struct SmartLearningDetector {
    /// Known entity types and their frequencies
    entity_knowledge: Arc<RwLock<HashMap<String, EntityKnowledge>>>,

    /// Known relationship patterns
    relationship_patterns: Arc<RwLock<HashMap<String, RelationshipKnowledge>>>,

    /// Code patterns detected
    #[allow(dead_code)]
    code_patterns: Arc<RwLock<HashMap<String, CodePatternKnowledge>>>,

    /// Domain knowledge tracking
    #[allow(dead_code)]
    domain_knowledge: Arc<RwLock<HashMap<String, DomainKnowledge>>>,

    /// Detector statistics
    stats: Arc<RwLock<DetectorStats>>,

    /// Configuration
    config: DetectorConfig,
}

#[derive(Debug, Clone)]
struct EntityKnowledge {
    frequency: u32,
    #[allow(dead_code)]
    first_seen: chrono::DateTime<chrono::Utc>,
    last_seen: chrono::DateTime<chrono::Utc>,
    contexts: HashSet<String>,
    confidence_scores: Vec<f32>,
}

#[derive(Debug, Clone)]
struct RelationshipKnowledge {
    frequency: u32,
    entity_types: HashSet<(String, String)>,
    #[allow(dead_code)]
    first_seen: chrono::DateTime<chrono::Utc>,
    patterns: HashSet<String>,
}

#[derive(Debug, Clone)]
struct CodePatternKnowledge {
    #[allow(dead_code)]
    frequency: u32,
    #[allow(dead_code)]
    languages: HashSet<String>,
    #[allow(dead_code)]
    complexity_scores: Vec<f32>,
    #[allow(dead_code)]
    first_seen: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
struct DomainKnowledge {
    #[allow(dead_code)]
    topics: HashSet<String>,
    #[allow(dead_code)]
    expertise_indicators: Vec<String>,
    #[allow(dead_code)]
    confidence: f32,
    #[allow(dead_code)]
    last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DetectorConfig {
    /// Minimum confidence threshold for new entity detection
    pub entity_novelty_threshold: f32,

    /// Minimum confidence for relationship pattern detection
    pub relationship_novelty_threshold: f32,

    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,

    /// Enable conceptual breakthrough detection
    pub enable_breakthrough_detection: bool,

    /// Maximum events per analysis to prevent spam
    pub max_events_per_analysis: usize,

    /// Minimum time between similar events (seconds)
    pub deduplication_window_seconds: u64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            entity_novelty_threshold: 0.7,
            relationship_novelty_threshold: 0.6,
            enable_anomaly_detection: true,
            enable_breakthrough_detection: true,
            max_events_per_analysis: 5,
            deduplication_window_seconds: 300, // 5 minutes
        }
    }
}

impl SmartLearningDetector {
    /// Create a new smart learning detector
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            entity_knowledge: Arc::new(RwLock::new(HashMap::new())),
            relationship_patterns: Arc::new(RwLock::new(HashMap::new())),
            code_patterns: Arc::new(RwLock::new(HashMap::new())),
            domain_knowledge: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(DetectorStats::default())),
            config,
        }
    }

    /// Detect new entity types
    async fn detect_new_entities(
        &self,
        entities: &[SimpleExtractedEntity],
        context: &LearningContext,
    ) -> LearningResult<Vec<LearningEvent>> {
        let mut events = Vec::new();
        let mut knowledge = self.entity_knowledge.write().await;

        for entity in entities {
            let entity_type = &entity.entity_type;
            let is_new = !knowledge.contains_key(entity_type);
            let now = chrono::Utc::now();

            if is_new && entity.confidence >= self.config.entity_novelty_threshold {
                // New entity type discovered!
                info!("Discovered new entity type: {}", entity_type);

                let event = LearningEvent::new(
                    LearningEventType::NewEntityType {
                        entity_type: entity_type.clone(),
                        confidence: entity.confidence,
                    },
                    context.clone(),
                    format!("发现新实体类型: {}", entity_type),
                    format!(
                        "系统识别出新的实体类型 '{}', 置信度: {:.2}",
                        entity_type, entity.confidence
                    ),
                );

                events.push(event);

                // Update knowledge base
                knowledge.insert(
                    entity_type.clone(),
                    EntityKnowledge {
                        frequency: 1,
                        first_seen: now,
                        last_seen: now,
                        contexts: [context.activity.clone().unwrap_or_default()]
                            .into_iter()
                            .collect(),
                        confidence_scores: vec![entity.confidence],
                    },
                );
            } else if let Some(existing_knowledge) = knowledge.get_mut(entity_type) {
                // Update existing knowledge
                existing_knowledge.frequency += 1;
                existing_knowledge.last_seen = now;
                existing_knowledge.confidence_scores.push(entity.confidence);
                if let Some(activity) = &context.activity {
                    existing_knowledge.contexts.insert(activity.clone());
                }
            }
        }

        Ok(events)
    }

    /// Detect new relationship patterns
    async fn detect_new_relationships(
        &self,
        relationships: &[SimpleExtractedRelationship],
        context: &LearningContext,
    ) -> LearningResult<Vec<LearningEvent>> {
        let mut events = Vec::new();
        let mut knowledge = self.relationship_patterns.write().await;

        for relationship in relationships {
            let pattern_key = format!(
                "{}->{}:{}",
                relationship.source, relationship.target, relationship.relationship
            );
            let pattern_type = relationship.relationship.clone();

            let is_new_pattern = !knowledge.contains_key(&pattern_type);
            let now = chrono::Utc::now();

            if is_new_pattern
                && relationship.confidence >= self.config.relationship_novelty_threshold
            {
                info!("Discovered new relationship pattern: {}", pattern_type);

                let event = LearningEvent::new(
                    LearningEventType::NewRelationshipPattern {
                        pattern: pattern_type.clone(),
                        entities: vec![relationship.source.clone(), relationship.target.clone()],
                        confidence: relationship.confidence,
                    },
                    context.clone(),
                    format!("发现新关系模式: {}", pattern_type),
                    format!(
                        "识别出新的关系类型 '{}', 连接 {} 和 {}",
                        pattern_type, relationship.source, relationship.target
                    ),
                );

                events.push(event);

                // Update knowledge base
                knowledge.insert(
                    pattern_type.clone(),
                    RelationshipKnowledge {
                        frequency: 1,
                        entity_types: [(relationship.source.clone(), relationship.target.clone())]
                            .into_iter()
                            .collect(),
                        first_seen: now,
                        patterns: [pattern_key].into_iter().collect(),
                    },
                );
            } else if let Some(existing_knowledge) = knowledge.get_mut(&pattern_type) {
                existing_knowledge.frequency += 1;
                existing_knowledge
                    .entity_types
                    .insert((relationship.source.clone(), relationship.target.clone()));
                existing_knowledge.patterns.insert(pattern_key);
            }
        }

        Ok(events)
    }

    /// Detect code patterns and architectural insights
    async fn detect_code_patterns(
        &self,
        content: &str,
        context: &LearningContext,
    ) -> LearningResult<Vec<LearningEvent>> {
        let mut events = Vec::new();

        // Basic pattern detection (can be enhanced with more sophisticated analysis)
        let patterns = self.analyze_code_patterns(content, context).await?;

        for (pattern_type, confidence) in patterns {
            if confidence >= 0.7 {
                let event = LearningEvent::new(
                    LearningEventType::NewCodePattern {
                        pattern_type: pattern_type.clone(),
                        language: context.language.clone().unwrap_or("unknown".to_string()),
                        description: format!("检测到代码模式: {}", pattern_type),
                        confidence,
                    },
                    context.clone(),
                    format!("识别代码模式: {}", pattern_type),
                    format!(
                        "在代码中发现 {} 模式, 置信度: {:.2}",
                        pattern_type, confidence
                    ),
                );

                events.push(event);
            }
        }

        Ok(events)
    }

    /// Analyze code for common patterns
    async fn analyze_code_patterns(
        &self,
        content: &str,
        context: &LearningContext,
    ) -> LearningResult<Vec<(String, f32)>> {
        let mut patterns = Vec::new();
        let content_lower = content.to_lowercase();

        // Design pattern detection
        if content_lower.contains("observer") && content_lower.contains("notify") {
            patterns.push(("Observer Pattern".to_string(), 0.8));
        }

        if content_lower.contains("singleton") && content_lower.contains("instance") {
            patterns.push(("Singleton Pattern".to_string(), 0.75));
        }

        if content_lower.contains("factory") && content_lower.contains("create") {
            patterns.push(("Factory Pattern".to_string(), 0.7));
        }

        // Architectural patterns
        if content_lower.contains("controller") && content_lower.contains("service") {
            patterns.push(("MVC Architecture".to_string(), 0.6));
        }

        if content_lower.contains("api") && content_lower.contains("endpoint") {
            patterns.push(("REST API Pattern".to_string(), 0.65));
        }

        // Language-specific patterns
        if let Some(language) = &context.language {
            match language.to_lowercase().as_str() {
                "rust" => {
                    if content_lower.contains("trait") && content_lower.contains("impl") {
                        patterns.push(("Rust Trait Pattern".to_string(), 0.8));
                    }
                    if content_lower.contains("async") && content_lower.contains("await") {
                        patterns.push(("Async/Await Pattern".to_string(), 0.85));
                    }
                }
                "python" => {
                    if content_lower.contains("__init__") && content_lower.contains("self") {
                        patterns.push(("Python Class Pattern".to_string(), 0.7));
                    }
                }
                "javascript" | "typescript" => {
                    if content_lower.contains("promise") && content_lower.contains("then") {
                        patterns.push(("Promise Pattern".to_string(), 0.75));
                    }
                }
                _ => {}
            }
        }

        Ok(patterns)
    }

    /// Detect conceptual breakthroughs and connections
    async fn detect_breakthroughs(
        &self,
        request: &AddMemoryRequest,
        entities: &[SimpleExtractedEntity],
        context: &LearningContext,
    ) -> LearningResult<Vec<LearningEvent>> {
        let mut events = Vec::new();

        if !self.config.enable_breakthrough_detection {
            return Ok(events);
        }

        // Check for breakthrough indicators in content
        let content_lower = request.content.to_lowercase();
        let breakthrough_keywords = [
            "breakthrough",
            "discovery",
            "insight",
            "realization",
            "understanding",
            "eureka",
            "aha",
            "suddenly",
            "finally understood",
            "now I see",
            "突破",
            "发现",
            "洞察",
            "理解",
            "顿悟",
            "原来如此",
            "明白了",
        ];

        let has_breakthrough_language = breakthrough_keywords
            .iter()
            .any(|keyword| content_lower.contains(keyword));

        if has_breakthrough_language && !entities.is_empty() {
            // Extract the main concept from entities
            let main_concept = entities
                .first()
                .map(|e| e.name.clone())
                .unwrap_or_else(|| "Unknown Concept".to_string());

            let related_concepts: Vec<String> =
                entities.iter().skip(1).map(|e| e.name.clone()).collect();

            let confidence = entities
                .iter()
                .map(|e| e.confidence)
                .fold(0.0, |acc, c| acc + c)
                / entities.len() as f32;

            let event = LearningEvent::new(
                LearningEventType::ConceptualBreakthrough {
                    concept: main_concept.clone(),
                    related_concepts: related_concepts.clone(),
                    insight: self.extract_insight(&request.content),
                    confidence,
                },
                context.clone(),
                format!("💡 概念突破: {}", main_concept),
                format!("在 {} 方面获得新的理解和洞察", main_concept),
            );

            events.push(event);
        }

        Ok(events)
    }

    /// Extract insight text from content
    fn extract_insight(&self, content: &str) -> String {
        // Simple extraction - could be enhanced with NLP
        let sentences: Vec<&str> = content.split('.').collect();
        for sentence in &sentences {
            let sentence_lower = sentence.to_lowercase();
            if sentence_lower.contains("understand")
                || sentence_lower.contains("realize")
                || sentence_lower.contains("insight")
                || sentence_lower.contains("理解")
                || sentence_lower.contains("明白")
            {
                return sentence.trim().to_string();
            }
        }

        // Fallback to first sentence
        sentences
            .first()
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "New insight gained".to_string())
    }
}

#[async_trait]
impl LearningDetector for SmartLearningDetector {
    async fn detect_learning(
        &self,
        request: &AddMemoryRequest,
        response: &AddMemoryResponse,
        context: &LearningContext,
    ) -> LearningResult<Vec<LearningEvent>> {
        let start_time = std::time::Instant::now();
        let mut all_events = Vec::new();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_analyses += 1;
        }

        // Detect new entities
        let entity_events = self
            .detect_new_entities(&response.entities, context)
            .await?;
        all_events.extend(entity_events);

        // Detect new relationships
        let relationship_events = self
            .detect_new_relationships(&response.relationships, context)
            .await?;
        all_events.extend(relationship_events);

        // Detect code patterns
        let code_pattern_events = self.detect_code_patterns(&request.content, context).await?;
        all_events.extend(code_pattern_events);

        // Detect conceptual breakthroughs
        let breakthrough_events = self
            .detect_breakthroughs(request, &response.entities, context)
            .await?;
        all_events.extend(breakthrough_events);

        // Limit number of events to prevent spam
        all_events.truncate(self.config.max_events_per_analysis);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.events_detected += all_events.len() as u64;
            stats
                .processing_times_ms
                .push(start_time.elapsed().as_millis() as u64);

            for event in &all_events {
                stats.confidence_scores.push(event.confidence);
            }
        }

        debug!(
            "Learning detection completed: {} events detected",
            all_events.len()
        );
        Ok(all_events)
    }

    async fn update_knowledge_state(
        &self,
        entities: &[SimpleExtractedEntity],
        relationships: &[SimpleExtractedRelationship],
    ) -> LearningResult<()> {
        // This is called during detect_learning, so state is already updated there
        debug!(
            "Knowledge state updated with {} entities and {} relationships",
            entities.len(),
            relationships.len()
        );
        Ok(())
    }

    async fn get_detector_stats(&self) -> LearningResult<DetectorStats> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AddMemoryRequest;
    use uuid::Uuid;

    fn create_test_request() -> AddMemoryRequest {
        AddMemoryRequest {
            content: "I discovered a new design pattern in Rust code".to_string(),
            source: Some("test".to_string()),
            name: Some("test memory".to_string()),
            group_id: None,
            memory_type: None,
            metadata: None,
            timestamp: None,
        }
    }

    fn create_test_response() -> AddMemoryResponse {
        AddMemoryResponse {
            id: Uuid::new_v4(),
            entities: vec![SimpleExtractedEntity {
                name: "Design Pattern".to_string(),
                entity_type: "Concept".to_string(),
                confidence: 0.9,
            }],
            relationships: vec![],
        }
    }

    #[tokio::test]
    async fn test_new_entity_detection() {
        let detector = SmartLearningDetector::new(DetectorConfig::default());
        let request = create_test_request();
        let response = create_test_response();
        let context = LearningContext::default();

        let events = detector
            .detect_learning(&request, &response, &context)
            .await
            .unwrap();

        assert!(!events.is_empty());
        assert!(events
            .iter()
            .any(|e| matches!(e.event_type, LearningEventType::NewEntityType { .. })));
    }
}
