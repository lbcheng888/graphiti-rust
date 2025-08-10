//! Graph data structures for nodes and edges

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Temporal metadata for bi-temporal tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMetadata {
    /// When the data was recorded in the system
    pub created_at: DateTime<Utc>,
    /// When the event actually occurred
    pub valid_from: DateTime<Utc>,
    /// When the data was superseded (if applicable)
    pub valid_to: Option<DateTime<Utc>>,
    /// When the data expired (if applicable)
    pub expired_at: Option<DateTime<Utc>>,
}

/// Base trait for all node types
pub trait Node: Send + Sync + std::fmt::Debug {
    /// Get the node's unique identifier
    fn id(&self) -> &Uuid;

    /// Get the node's labels
    fn labels(&self) -> Vec<String>;

    /// Get the node's properties as JSON
    fn properties(&self) -> serde_json::Value;

    /// Get the node's temporal metadata
    fn temporal(&self) -> &TemporalMetadata;
}

/// Entity node representing a person, place, thing, or concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    /// Unique identifier
    pub id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Entity type (e.g., "Person", "Organization", "Location")
    pub entity_type: String,
    /// Graph labels
    pub labels: Vec<String>,
    /// Additional properties
    pub properties: serde_json::Value,
    /// Temporal metadata
    pub temporal: TemporalMetadata,
    /// Optional embedding vector
    pub embedding: Option<Vec<f32>>,
}

/// Episode types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeType {
    /// A message or conversation
    Message,
    /// An event that occurred
    Event,
    /// A document or article
    Document,
    /// A general note or observation
    Note,
}

/// Episode node representing a unit of knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeNode {
    /// Unique identifier
    pub id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Type of episode
    pub episode_type: EpisodeType,
    /// Content of the episode
    pub content: String,
    /// Source of the episode
    pub source: String,
    /// Temporal metadata
    pub temporal: TemporalMetadata,
    /// Optional embedding vector
    pub embedding: Option<Vec<f32>>,
}

/// Community node representing a group of related entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityNode {
    /// Unique identifier
    pub id: Uuid,
    /// Community name
    pub name: String,
    /// Community summary/description
    pub summary: String,
    /// Member entity IDs
    pub members: Vec<Uuid>,
    /// Temporal metadata
    pub temporal: TemporalMetadata,
    /// Community detection metadata
    pub metadata: serde_json::Value,
}

/// Edge representing a relationship between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique identifier
    pub id: Uuid,
    /// Source node ID
    pub source_id: Uuid,
    /// Target node ID
    pub target_id: Uuid,
    /// Relationship type
    pub relationship: String,
    /// Additional properties
    pub properties: serde_json::Value,
    /// Temporal metadata
    pub temporal: TemporalMetadata,
    /// Edge weight (for ranking)
    pub weight: f32,
}

impl Node for EntityNode {
    fn id(&self) -> &Uuid {
        &self.id
    }

    fn labels(&self) -> Vec<String> {
        self.labels.clone()
    }

    fn properties(&self) -> serde_json::Value {
        self.properties.clone()
    }

    fn temporal(&self) -> &TemporalMetadata {
        &self.temporal
    }
}

impl Node for EpisodeNode {
    fn id(&self) -> &Uuid {
        &self.id
    }

    fn labels(&self) -> Vec<String> {
        vec!["Episode".to_string()]
    }

    fn properties(&self) -> serde_json::Value {
        serde_json::json!({
            "content": self.content,
            "source": self.source,
            "episode_type": self.episode_type,
        })
    }

    fn temporal(&self) -> &TemporalMetadata {
        &self.temporal
    }
}

impl Node for CommunityNode {
    fn id(&self) -> &Uuid {
        &self.id
    }

    fn labels(&self) -> Vec<String> {
        vec!["Community".to_string()]
    }

    fn properties(&self) -> serde_json::Value {
        serde_json::json!({
            "summary": self.summary,
            "members": self.members,
            "metadata": self.metadata,
        })
    }

    fn temporal(&self) -> &TemporalMetadata {
        &self.temporal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_temporal_metadata() -> TemporalMetadata {
        TemporalMetadata {
            created_at: Utc::now(),
            valid_from: Utc::now(),
            valid_to: None,
            expired_at: None,
        }
    }

    #[test]
    fn test_entity_node_creation() {
        let node = EntityNode {
            id: Uuid::new_v4(),
            name: "Alice".to_string(),
            entity_type: "Person".to_string(),
            labels: vec!["Person".to_string(), "User".to_string()],
            properties: serde_json::json!({"age": 30, "city": "New York"}),
            temporal: create_temporal_metadata(),
            embedding: Some(vec![0.1, 0.2, 0.3]),
        };

        assert_eq!(node.name, "Alice");
        assert_eq!(node.entity_type, "Person");
        assert_eq!(node.labels.len(), 2);
        assert!(node.embedding.is_some());
    }

    #[test]
    fn test_episode_node_creation() {
        let node = EpisodeNode {
            id: Uuid::new_v4(),
            name: "Meeting Notes".to_string(),
            episode_type: EpisodeType::Event,
            content: "Alice met Bob at the conference.".to_string(),
            source: "meeting_system".to_string(),
            temporal: create_temporal_metadata(),
            embedding: None,
        };

        assert_eq!(node.name, "Meeting Notes");
        assert_eq!(node.content, "Alice met Bob at the conference.");
        assert!(node.embedding.is_none());
    }

    #[test]
    fn test_community_node_creation() {
        let member_ids = vec![Uuid::new_v4(), Uuid::new_v4()];
        let node = CommunityNode {
            id: Uuid::new_v4(),
            name: "Tech Community".to_string(),
            summary: "A community of technology professionals".to_string(),
            members: member_ids.clone(),
            temporal: create_temporal_metadata(),
            metadata: serde_json::json!({"algorithm": "louvain", "modularity": 0.85}),
        };

        assert_eq!(node.name, "Tech Community");
        assert_eq!(node.members.len(), 2);
        assert_eq!(node.members, member_ids);
    }

    #[test]
    fn test_edge_creation() {
        let edge = Edge {
            id: Uuid::new_v4(),
            source_id: Uuid::new_v4(),
            target_id: Uuid::new_v4(),
            relationship: "knows".to_string(),
            properties: serde_json::json!({"since": "2023", "context": "work"}),
            temporal: create_temporal_metadata(),
            weight: 0.9,
        };

        assert_eq!(edge.relationship, "knows");
        assert_eq!(edge.weight, 0.9);
    }

    #[test]
    fn test_node_trait_implementation() {
        let entity = EntityNode {
            id: Uuid::new_v4(),
            name: "Test Entity".to_string(),
            entity_type: "TestType".to_string(),
            labels: vec!["Test".to_string()],
            properties: serde_json::json!({"key": "value"}),
            temporal: create_temporal_metadata(),
            embedding: None,
        };

        // Test Node trait methods
        let node: &dyn Node = &entity;
        assert_eq!(node.labels(), vec!["Test".to_string()]);
        assert_eq!(node.properties(), serde_json::json!({"key": "value"}));
    }

    #[test]
    fn test_serialization() {
        let node = EntityNode {
            id: Uuid::new_v4(),
            name: "Serializable".to_string(),
            entity_type: "Test".to_string(),
            labels: vec!["Test".to_string()],
            properties: serde_json::json!({}),
            temporal: create_temporal_metadata(),
            embedding: Some(vec![1.0, 2.0]),
        };

        // Test serialization
        let serialized = serde_json::to_string(&node).unwrap();
        let deserialized: EntityNode = serde_json::from_str(&serialized).unwrap();

        assert_eq!(node.id, deserialized.id);
        assert_eq!(node.name, deserialized.name);
        assert_eq!(node.embedding, deserialized.embedding);
    }

    #[test]
    fn test_episode_types() {
        let message_json = serde_json::to_string(&EpisodeType::Message).unwrap();
        assert_eq!(message_json, "\"message\"");

        let event_type: EpisodeType = serde_json::from_str("\"event\"").unwrap();
        assert!(matches!(event_type, EpisodeType::Event));
    }
}
