//! Request models for the MCP server

use serde::{Deserialize, Serialize};

/// Request to add a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddMemoryRequest {
    /// Content of the memory
    pub content: String,
    /// Optional name/title
    pub name: Option<String>,
    /// Source of the memory
    pub source: Option<String>,
    /// Type of memory
    pub memory_type: Option<String>,
    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
    /// Optional group/namespace id (not yet persisted)
    pub group_id: Option<String>,
    /// When the event occurred
    pub timestamp: Option<String>,
}

/// Request to search memories
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchMemoryRequest {
    /// Search query
    pub query: String,
    /// Maximum results
    pub limit: Option<usize>,
    /// Entity type filter
    pub entity_types: Option<Vec<String>>,
    /// Time range filter (ISO 8601)
    pub start_time: Option<String>,
    pub end_time: Option<String>,
}

/// Request to add a code entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddCodeEntityRequest {
    /// Entity type
    pub entity_type: String,
    /// Entity name
    pub name: String,
    /// Description
    pub description: String,
    /// File path (optional)
    pub file_path: Option<String>,
    /// Line range (optional)
    pub line_range: Option<(u32, u32)>,
    /// Programming language (optional)
    pub language: Option<String>,
    /// Framework (optional)
    pub framework: Option<String>,
    /// Complexity rating 1-10 (optional)
    pub complexity: Option<u8>,
    /// Importance rating 1-10 (optional)
    pub importance: Option<u8>,
}

/// Request to record development activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordActivityRequest {
    /// Activity type
    pub activity_type: String,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Developer name
    pub developer: String,
    /// Project name
    pub project: String,
    /// Related entity IDs (optional)
    pub related_entities: Option<Vec<String>>,
    /// Duration in minutes (optional)
    pub duration_minutes: Option<u32>,
    /// Difficulty rating 1-10 (optional)
    pub difficulty: Option<u8>,
    /// Quality rating 1-10 (optional)
    pub quality: Option<u8>,
}

/// Request to search code entities
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchCodeRequest {
    /// Search query
    pub query: String,
    /// Entity type filter (optional)
    pub entity_type: Option<String>,
    /// Language filter (optional)
    pub language: Option<String>,
    /// Framework filter (optional)
    pub framework: Option<String>,
    /// Maximum results
    pub limit: Option<u32>,
}

/// Request to batch add code entities
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchAddCodeEntitiesRequest {
    /// List of code entities to add
    pub entities: Vec<AddCodeEntityRequest>,
}

/// Request to batch record activities
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchRecordActivitiesRequest {
    /// List of activities to record
    pub activities: Vec<RecordActivityRequest>,
}

/// Request for intelligent context suggestions
#[derive(Debug, Serialize, Deserialize)]
pub struct ContextSuggestionRequest {
    /// Current development context
    pub context: String,
    /// Current working file (optional)
    pub current_file: Option<String>,
    /// Recent activities context (optional)
    pub recent_activities: Option<Vec<String>>,
    /// Maximum number of suggestions
    pub limit: Option<u32>,
}
