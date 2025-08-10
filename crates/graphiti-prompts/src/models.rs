//! Data models for prompt templates

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Extracted entity from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Name of the extracted entity
    pub name: String,
    /// ID of the classified entity type
    pub entity_type_id: i32,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Option<f32>,
}

/// List of extracted entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntities {
    /// List of extracted entities
    pub extracted_entities: Vec<ExtractedEntity>,
}

/// Entity classification triple
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityClassificationTriple {
    /// UUID of the entity
    pub uuid: String,
    /// Name of the entity
    pub name: String,
    /// Type of the entity
    pub entity_type: Option<String>,
}

/// Entity classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityClassification {
    /// List of entity classifications
    pub entity_classifications: Vec<EntityClassificationTriple>,
}

/// Extracted relationship/edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEdge {
    /// Source entity name
    pub source_entity: String,
    /// Target entity name
    pub target_entity: String,
    /// Relationship type
    pub relationship: String,
    /// Relationship description
    pub description: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Option<f32>,
    /// When the relationship was valid
    pub valid_at: Option<DateTime<Utc>>,
}

/// List of extracted edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEdges {
    /// List of extracted edges
    pub extracted_edges: Vec<ExtractedEdge>,
}

/// Node deduplication result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDeduplication {
    /// Groups of duplicate nodes
    pub duplicate_groups: Vec<Vec<String>>,
    /// Explanation of deduplication logic
    pub explanation: String,
}

/// Edge deduplication result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDeduplication {
    /// Groups of duplicate edges
    pub duplicate_groups: Vec<Vec<String>>,
    /// Explanation of deduplication logic
    pub explanation: String,
}

/// Node summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSummary {
    /// Updated summary text
    pub summary: String,
    /// Key attributes extracted
    pub attributes: serde_json::Value,
}

/// Edge invalidation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeInvalidation {
    /// List of edge UUIDs to invalidate
    pub invalidated_edges: Vec<String>,
    /// Explanation of invalidation logic
    pub explanation: String,
}

/// Entity type definition for prompts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityType {
    /// Type ID
    pub id: i32,
    /// Type name
    pub name: String,
    /// Type description
    pub description: String,
    /// Example entities of this type
    pub examples: Vec<String>,
}

/// Context for entity extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionContext {
    /// Current episode content
    pub episode_content: String,
    /// Previous episodes for context
    pub previous_episodes: Vec<String>,
    /// Available entity types
    pub entity_types: Vec<EntityType>,
    /// Excluded entity types
    pub excluded_types: Vec<String>,
}

/// Context for edge extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeExtractionContext {
    /// Current episode content
    pub episode_content: String,
    /// Extracted entities
    pub entities: Vec<ExtractedEntity>,
    /// Previous episodes for context
    pub previous_episodes: Vec<String>,
    /// Available relationship types
    pub relationship_types: Vec<String>,
}

/// Context for deduplication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationContext {
    /// Items to deduplicate
    pub items: Vec<serde_json::Value>,
    /// Additional context
    pub context: String,
}

/// Context for summarization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizationContext {
    /// Current node data
    pub node_data: serde_json::Value,
    /// New information to incorporate
    pub new_information: String,
    /// Previous summary
    pub previous_summary: Option<String>,
}

/// Context for edge invalidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidationContext {
    /// New edges to check against
    pub new_edges: Vec<serde_json::Value>,
    /// Existing edges that might be invalidated
    pub existing_edges: Vec<serde_json::Value>,
    /// Episode context
    pub episode_context: String,
}
