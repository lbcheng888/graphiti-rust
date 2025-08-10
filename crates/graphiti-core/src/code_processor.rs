//! Code-specific knowledge graph processing
//!
//! This module provides specialized processing for code-related entities,
//! relationships, and development workflows.

use crate::code_entities::CodeEntity;
use crate::code_entities::CodeEntityType;
use crate::code_entities::CodeRelation;
use crate::code_entities::CodeRelationType;
use crate::code_entities::DevelopmentActivity;
use crate::code_entities::KnowledgePattern;
use crate::code_entities::WorkflowStage;
use crate::episode_processor::EmbeddingClient;
use crate::episode_processor::LLMClient;
use crate::error::Result;
use crate::storage::GraphStorage;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;
use tracing::instrument;
use tracing::warn;

/// Code-specific episode processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeProcessingResult {
    /// Extracted code entities
    pub entities: Vec<CodeEntity>,
    /// Extracted code relationships
    pub relationships: Vec<CodeRelation>,
    /// Development activities identified
    pub activities: Vec<DevelopmentActivity>,
    /// Knowledge patterns discovered
    pub patterns: Vec<KnowledgePattern>,
    /// Processing metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Code knowledge graph processor
pub struct CodeProcessor<S, L, E>
where
    S: GraphStorage,
    L: LLMClient,
    E: EmbeddingClient,
{
    _storage: Arc<S>,
    llm_client: Arc<L>,
    _embedding_client: Arc<E>,
}

impl<S, L, E> CodeProcessor<S, L, E>
where
    S: GraphStorage,
    L: LLMClient,
    E: EmbeddingClient,
{
    /// Create a new code processor
    pub fn new(storage: Arc<S>, llm_client: Arc<L>, embedding_client: Arc<E>) -> Self {
        Self {
            _storage: storage,
            llm_client,
            _embedding_client: embedding_client,
        }
    }

    /// Process a code-related episode
    #[instrument(skip(self, content))]
    pub async fn process_code_episode(
        &self,
        content: &str,
        context: CodeContext,
    ) -> Result<CodeProcessingResult> {
        info!("Processing code episode with context: {:?}", context);

        let mut result = CodeProcessingResult {
            entities: Vec::new(),
            relationships: Vec::new(),
            activities: Vec::new(),
            patterns: Vec::new(),
            metadata: HashMap::new(),
        };

        // Step 1: Extract code entities
        result.entities = self.extract_code_entities(content, &context).await?;
        info!("Extracted {} code entities", result.entities.len());

        // Step 2: Extract code relationships
        result.relationships = self
            .extract_code_relationships(content, &result.entities)
            .await?;
        info!(
            "Extracted {} code relationships",
            result.relationships.len()
        );

        // Step 3: Identify development activities
        result.activities = self
            .identify_development_activities(content, &context)
            .await?;
        info!(
            "Identified {} development activities",
            result.activities.len()
        );

        // Step 4: Discover knowledge patterns
        result.patterns = self
            .discover_knowledge_patterns(&result.entities, &result.relationships)
            .await?;
        info!("Discovered {} knowledge patterns", result.patterns.len());

        // Step 5: Store in graph database
        self.store_code_knowledge(&result).await?;

        Ok(result)
    }

    /// Extract code entities from content
    async fn extract_code_entities(
        &self,
        content: &str,
        context: &CodeContext,
    ) -> Result<Vec<CodeEntity>> {
        info!("Extracting code entities from content");

        let mut entities = Vec::new();

        // Use LLM to extract code entities
        let prompt = self.create_code_entity_extraction_prompt(content, context);
        match self.llm_client.complete(&prompt).await {
            Ok(response) => {
                entities.extend(self.parse_code_entities_response(&response, context)?);
            }
            Err(e) => {
                warn!("LLM extraction failed, using fallback: {}", e);
                entities.extend(self.fallback_code_entity_extraction(content, context));
            }
        }

        Ok(entities)
    }

    /// Extract code relationships
    async fn extract_code_relationships(
        &self,
        content: &str,
        entities: &[CodeEntity],
    ) -> Result<Vec<CodeRelation>> {
        info!("Extracting code relationships");

        let mut relationships = Vec::new();

        if entities.len() < 2 {
            return Ok(relationships);
        }

        // Use LLM to identify relationships between entities
        let prompt = self.create_relationship_extraction_prompt(content, entities);
        match self.llm_client.complete(&prompt).await {
            Ok(response) => {
                relationships.extend(self.parse_relationships_response(&response, entities)?);
            }
            Err(e) => {
                warn!("LLM relationship extraction failed, using fallback: {}", e);
                relationships.extend(self.fallback_relationship_extraction(entities));
            }
        }

        Ok(relationships)
    }

    /// Identify development activities
    async fn identify_development_activities(
        &self,
        content: &str,
        context: &CodeContext,
    ) -> Result<Vec<DevelopmentActivity>> {
        info!("Identifying development activities");

        let mut activities = Vec::new();

        // Analyze content for development workflow indicators
        let activity_type = self.classify_workflow_stage(content);

        if let Some(stage) = activity_type {
            let activity = DevelopmentActivity::new(
                stage,
                context
                    .title
                    .clone()
                    .unwrap_or_else(|| "Code Activity".to_string()),
                content.to_string(),
                context
                    .developer
                    .clone()
                    .unwrap_or_else(|| "Unknown".to_string()),
                context
                    .project
                    .clone()
                    .unwrap_or_else(|| "Unknown".to_string()),
            );
            activities.push(activity);
        }

        Ok(activities)
    }

    /// Discover knowledge patterns
    async fn discover_knowledge_patterns(
        &self,
        entities: &[CodeEntity],
        relationships: &[CodeRelation],
    ) -> Result<Vec<KnowledgePattern>> {
        info!("Discovering knowledge patterns");

        let mut patterns = Vec::new();

        // Analyze entity and relationship patterns
        if let Some(pattern) = self.analyze_architectural_patterns(entities, relationships) {
            patterns.push(pattern);
        }

        if let Some(pattern) = self.analyze_design_patterns(entities, relationships) {
            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Store code knowledge in graph database
    async fn store_code_knowledge(&self, result: &CodeProcessingResult) -> Result<()> {
        info!("Storing code knowledge in graph database");

        // Store entities as nodes
        for entity in &result.entities {
            // Convert CodeEntity to a graph node (simplified)
            // In a real implementation, we'd have proper conversion
            info!("Storing code entity: {}", entity.name);
        }

        // Store relationships as edges
        for relationship in &result.relationships {
            // Convert CodeRelation to a graph edge (simplified)
            info!(
                "Storing code relationship: {:?}",
                relationship.relation_type
            );
        }

        Ok(())
    }

    /// Create LLM prompt for code entity extraction
    fn create_code_entity_extraction_prompt(&self, content: &str, context: &CodeContext) -> String {
        format!(
            "Extract code entities from the following content. Focus on classes, functions, modules, APIs, and other code constructs.\n\nContext: {:?}\n\nContent:\n{}\n\nPlease identify and list code entities with their types.",
            context, content
        )
    }

    /// Parse LLM response for code entities
    fn parse_code_entities_response(
        &self,
        response: &str,
        context: &CodeContext,
    ) -> Result<Vec<CodeEntity>> {
        let mut entities = Vec::new();

        // Try to parse JSON response first
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(response) {
            if let Some(entities_array) = parsed.get("entities").and_then(|e| e.as_array()) {
                for entity_json in entities_array {
                    if let Ok(entity) = self.parse_entity_from_json(entity_json, context) {
                        entities.push(entity);
                    }
                }
                return Ok(entities);
            }
        }

        // Fallback to text-based parsing
        entities.extend(self.parse_entities_from_text(response, context)?);
        Ok(entities)
    }

    /// Parse entity from JSON object
    fn parse_entity_from_json(
        &self,
        entity_json: &serde_json::Value,
        context: &CodeContext,
    ) -> Result<CodeEntity> {
        let entity_type_str = entity_json
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("Function");

        let entity_type = match entity_type_str.to_lowercase().as_str() {
            "class" | "struct" => CodeEntityType::Class,
            "function" | "method" => CodeEntityType::Function,
            "module" | "package" => CodeEntityType::Module,
            "api" | "endpoint" => CodeEntityType::Api,
            "datamodel" | "model" => CodeEntityType::DataModel,
            "config" | "configuration" => CodeEntityType::Configuration,
            "test" | "testcase" => CodeEntityType::TestCase,
            "doc" | "documentation" => CodeEntityType::Documentation,
            _ => CodeEntityType::Function,
        };

        let name = entity_json
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("UnknownEntity")
            .to_string();

        let description = entity_json
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();

        Ok(CodeEntity::new(entity_type, name, description)
            .with_technology(context.language.clone(), context.framework.clone()))
    }

    /// Parse entities from plain text response
    fn parse_entities_from_text(
        &self,
        response: &str,
        context: &CodeContext,
    ) -> Result<Vec<CodeEntity>> {
        let mut entities = Vec::new();
        let lines: Vec<&str> = response.lines().collect();

        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Look for patterns like "Class: ClassName - Description"
            if let Some(entity) = self.extract_entity_from_line(line, context) {
                entities.push(entity);
            }
        }

        Ok(entities)
    }

    /// Extract entity from a single line of text
    fn extract_entity_from_line(&self, line: &str, context: &CodeContext) -> Option<CodeEntity> {
        // Pattern: "Type: Name - Description"
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        if parts.len() != 2 {
            return None;
        }

        let entity_type_str = parts[0].trim();
        let rest = parts[1].trim();

        let entity_type = match entity_type_str.to_lowercase().as_str() {
            "class" | "struct" => CodeEntityType::Class,
            "function" | "method" => CodeEntityType::Function,
            "module" | "package" => CodeEntityType::Module,
            "api" | "endpoint" => CodeEntityType::Api,
            "datamodel" | "model" => CodeEntityType::DataModel,
            "config" | "configuration" => CodeEntityType::Configuration,
            "test" | "testcase" => CodeEntityType::TestCase,
            "doc" | "documentation" => CodeEntityType::Documentation,
            _ => return None,
        };

        let (name, description) = if let Some(dash_pos) = rest.find(" - ") {
            let name = rest[..dash_pos].trim().to_string();
            let desc = rest[dash_pos + 3..].trim().to_string();
            (name, desc)
        } else {
            (rest.to_string(), String::new())
        };

        if name.is_empty() {
            return None;
        }

        Some(
            CodeEntity::new(entity_type, name, description)
                .with_technology(context.language.clone(), context.framework.clone()),
        )
    }

    /// Fallback code entity extraction using simple rules
    fn fallback_code_entity_extraction(
        &self,
        content: &str,
        _context: &CodeContext,
    ) -> Vec<CodeEntity> {
        let mut entities = Vec::new();
        let content_lower = content.to_lowercase();

        // Simple keyword-based extraction
        if content_lower.contains("class") {
            entities.push(CodeEntity::new(
                CodeEntityType::Class,
                "DetectedClass".to_string(),
                "Class detected by keyword analysis".to_string(),
            ));
        }

        if content_lower.contains("function") || content_lower.contains("def ") {
            entities.push(CodeEntity::new(
                CodeEntityType::Function,
                "DetectedFunction".to_string(),
                "Function detected by keyword analysis".to_string(),
            ));
        }

        if content_lower.contains("api") || content_lower.contains("endpoint") {
            entities.push(CodeEntity::new(
                CodeEntityType::Api,
                "DetectedAPI".to_string(),
                "API detected by keyword analysis".to_string(),
            ));
        }

        entities
    }

    /// Create relationship extraction prompt
    fn create_relationship_extraction_prompt(
        &self,
        content: &str,
        entities: &[CodeEntity],
    ) -> String {
        let entity_names: Vec<&String> = entities.iter().map(|e| &e.name).collect();
        format!(
            "Analyze the relationships between these code entities: {:?}\n\nContent:\n{}\n\nIdentify relationships like dependencies, inheritance, calls, etc.",
            entity_names, content
        )
    }

    /// Parse relationship extraction response
    fn parse_relationships_response(
        &self,
        response: &str,
        entities: &[CodeEntity],
    ) -> Result<Vec<CodeRelation>> {
        let mut relationships = Vec::new();

        // Simplified relationship extraction
        if entities.len() >= 2 && response.contains("depends") {
            relationships.push(CodeRelation::new(
                CodeRelationType::DependsOn,
                entities[0].id,
                entities[1].id,
                0.8,
                "Dependency relationship detected".to_string(),
            ));
        }

        Ok(relationships)
    }

    /// Fallback relationship extraction
    fn fallback_relationship_extraction(&self, entities: &[CodeEntity]) -> Vec<CodeRelation> {
        let mut relationships = Vec::new();

        // Create simple relationships between entities
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                relationships.push(CodeRelation::new(
                    CodeRelationType::SimilarTo,
                    entities[i].id,
                    entities[j].id,
                    0.5,
                    "Default similarity relationship".to_string(),
                ));
            }
        }

        relationships
    }

    /// Classify workflow stage from content
    fn classify_workflow_stage(&self, content: &str) -> Option<WorkflowStage> {
        let content_lower = content.to_lowercase();

        if content_lower.contains("test") || content_lower.contains("testing") {
            Some(WorkflowStage::UnitTesting)
        } else if content_lower.contains("review") || content_lower.contains("code review") {
            Some(WorkflowStage::CodeReview)
        } else if content_lower.contains("deploy") || content_lower.contains("deployment") {
            Some(WorkflowStage::Deployment)
        } else if content_lower.contains("bug") || content_lower.contains("fix") {
            Some(WorkflowStage::BugFix)
        } else if content_lower.contains("refactor") {
            Some(WorkflowStage::Refactoring)
        } else if content_lower.contains("implement") || content_lower.contains("coding") {
            Some(WorkflowStage::Implementation)
        } else {
            None
        }
    }

    /// Analyze architectural patterns
    fn analyze_architectural_patterns(
        &self,
        entities: &[CodeEntity],
        _relationships: &[CodeRelation],
    ) -> Option<KnowledgePattern> {
        // Simplified pattern detection
        let has_api = entities
            .iter()
            .any(|e| matches!(e.entity_type, CodeEntityType::Api));
        let has_data_model = entities
            .iter()
            .any(|e| matches!(e.entity_type, CodeEntityType::DataModel));

        if has_api && has_data_model {
            Some(
                KnowledgePattern::new(
                    "API-Data Pattern".to_string(),
                    "Pattern involving API endpoints and data models".to_string(),
                )
                .add_scenario("Web API development".to_string()),
            )
        } else {
            None
        }
    }

    /// Analyze design patterns
    fn analyze_design_patterns(
        &self,
        entities: &[CodeEntity],
        _relationships: &[CodeRelation],
    ) -> Option<KnowledgePattern> {
        // Look for common design patterns
        let class_count = entities
            .iter()
            .filter(|e| matches!(e.entity_type, CodeEntityType::Class))
            .count();

        if class_count >= 3 {
            Some(
                KnowledgePattern::new(
                    "Multi-Class Design".to_string(),
                    "Design pattern involving multiple classes".to_string(),
                )
                .add_scenario("Object-oriented design".to_string()),
            )
        } else {
            None
        }
    }
}

/// Context for code processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeContext {
    /// Programming language
    pub language: Option<String>,
    /// Framework or technology stack
    pub framework: Option<String>,
    /// Project name
    pub project: Option<String>,
    /// Developer name
    pub developer: Option<String>,
    /// Activity title
    pub title: Option<String>,
    /// File path (if applicable)
    pub file_path: Option<String>,
}

impl Default for CodeContext {
    fn default() -> Self {
        Self {
            language: None,
            framework: None,
            project: None,
            developer: None,
            title: None,
            file_path: None,
        }
    }
}
