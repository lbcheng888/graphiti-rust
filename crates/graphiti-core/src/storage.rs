//! Storage traits and types for graph database abstraction

use async_trait::async_trait;
use uuid::Uuid;

use crate::graph::{Edge, Node};

/// Direction of graph traversal
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Direction {
    /// Outgoing edges
    Outgoing,
    /// Incoming edges
    Incoming,
    /// Both directions
    Both,
}

/// Trait for graph storage implementations
#[async_trait]
pub trait GraphStorage: Send + Sync {
    /// Error type for storage operations
    type Error: std::error::Error + Send + Sync + 'static;

    /// Create a new node
    async fn create_node(&self, node: &dyn Node) -> Result<(), Self::Error>;

    /// Get a node by ID
    async fn get_node(&self, id: &Uuid) -> Result<Option<Box<dyn Node>>, Self::Error>;

    /// Update an existing node
    async fn update_node(&self, node: &dyn Node) -> Result<(), Self::Error>;

    /// Delete a node
    async fn delete_node(&self, id: &Uuid) -> Result<(), Self::Error>;

    /// Create a new edge
    async fn create_edge(&self, edge: &Edge) -> Result<(), Self::Error>;

    /// Get an edge by ID
    async fn get_edge_by_id(&self, id: &Uuid) -> Result<Option<Edge>, Self::Error>;

    /// Delete an edge by ID. Returns true if an edge was deleted
    async fn delete_edge_by_id(&self, id: &Uuid) -> Result<bool, Self::Error>;

    /// Get edges for a node
    async fn get_edges(
        &self,
        node_id: &Uuid,
        direction: Direction,
    ) -> Result<Vec<Edge>, Self::Error>;

    /// Get all nodes in the graph
    async fn get_all_nodes(&self) -> Result<Vec<Box<dyn Node>>, Self::Error>;

    /// Get all edges in the graph
    async fn get_all_edges(&self) -> Result<Vec<Edge>, Self::Error>;

    /// Get nodes valid at a specific time
    async fn get_nodes_at_time(
        &self,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<Box<dyn crate::graph::Node>>, Self::Error>;

    /// Get edges valid at a specific time
    async fn get_edges_at_time(
        &self,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<Edge>, Self::Error>;

    /// Get node history (all versions)
    async fn get_node_history(
        &self,
        node_id: &Uuid,
    ) -> Result<Vec<Box<dyn crate::graph::Node>>, Self::Error>;

    /// Get edge history (all versions)
    async fn get_edge_history(&self, edge_id: &Uuid) -> Result<Vec<Edge>, Self::Error>;
}
