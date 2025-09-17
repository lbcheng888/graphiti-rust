//! Response models for the MCP server

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Response from adding a memory
#[derive(Debug, Serialize, Deserialize)]
pub struct AddMemoryResponse {
    /// ID of the created memory
    pub id: Uuid,
    /// Extracted entities
    pub entities: Vec<SimpleExtractedEntity>,
    /// Extracted relationships
    pub relationships: Vec<SimpleExtractedRelationship>,
}

/// Response for add_code_entity
#[derive(Debug, Serialize, Deserialize)]
pub struct AddCodeEntityResponse {
    pub id: Uuid,
    pub message: String,
}

/// Response for record_activity
#[derive(Debug, Serialize, Deserialize)]
pub struct RecordActivityResponse {
    pub id: Uuid,
    pub message: String,
}

/// Response for search_code
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchCodeResponse {
    pub results: Vec<CodeEntity>,
    pub total: usize,
}

/// Response for batch add code entities
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchAddCodeEntitiesResponse {
    pub results: Vec<AddCodeEntityResponse>,
    pub successful_count: usize,
    pub failed_count: usize,
    pub errors: Vec<String>,
}

/// Response for batch record activities
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchRecordActivitiesResponse {
    pub results: Vec<RecordActivityResponse>,
    pub successful_count: usize,
    pub failed_count: usize,
    pub errors: Vec<String>,
}

/// A context suggestion
#[derive(Debug, Serialize, Deserialize)]
pub struct ContextSuggestion {
    /// Suggestion type
    pub suggestion_type: String,
    /// Suggestion title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Relevant code entities
    pub related_entities: Vec<String>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Priority (1-10)
    pub priority: u8,
}

/// Response for context suggestions
#[derive(Debug, Serialize, Deserialize)]
pub struct ContextSuggestionResponse {
    pub suggestions: Vec<ContextSuggestion>,
    pub total: usize,
}

/// Response from searching memories
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchMemoryResponse {
    /// Search results
    pub results: Vec<SearchResult>,
    /// Total results found
    pub total: usize,
}

/// Search result
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    /// Node ID
    pub id: Uuid,
    /// Node type
    pub node_type: String,
    /// Node name/title
    pub name: String,
    /// Content preview
    pub content_preview: Option<String>,
    /// Relevance score
    pub score: f32,
    /// Timestamp
    pub timestamp: String,
}

/// Memory node representation
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Node ID
    pub id: Uuid,
    /// Node type
    pub node_type: String,
    /// Node name
    pub name: String,
    /// Node content (if applicable)
    pub content: Option<String>,
    /// Creation time
    pub created_at: String,
    /// Event time
    pub event_time: String,
    /// Additional properties
    pub properties: serde_json::Value,
}

/// Related memory
#[derive(Debug, Serialize, Deserialize)]
pub struct RelatedMemory {
    /// The related node
    pub node: MemoryNode,
    /// Relationship type
    pub relationship: String,
    /// Distance from source
    pub distance: usize,
}

/// Extracted entity (simplified for API response)
#[derive(Debug, Serialize, Deserialize)]
pub struct SimpleExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub confidence: f32,
}

/// Extracted relationship (simplified for API response)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SimpleExtractedRelationship {
    pub source: String,
    pub target: String,
    pub relationship: String,
    pub confidence: f32,
}

/// Code entity representation
#[derive(Debug, Serialize, Deserialize)]
pub struct CodeEntity {
    pub id: Uuid,
    pub entity_type: String,
    pub name: String,
    pub description: String,
    pub file_path: Option<String>,
    pub line_range: Option<(u32, u32)>,
    pub language: Option<String>,
    pub framework: Option<String>,
    pub complexity: Option<u8>,
    pub importance: Option<u8>,
    pub created_at: String,
    pub updated_at: String,
    pub metadata: std::collections::HashMap<String, String>,
}
