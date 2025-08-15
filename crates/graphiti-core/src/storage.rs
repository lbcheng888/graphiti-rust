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

    /// Create multiple edges in a batch. Default implementation falls back to per-edge writes.
    async fn create_edges_batch(&self, edges: &[Edge]) -> Result<(), Self::Error> {
        for edge in edges {
            self.create_edge(edge).await?;
        }
        Ok(())
    }

    /// Prune oldest edges to keep at most `max_edges`. Returns number of deleted edges.
    /// Default implementation fetches all edges and deletes extra ones by age.
    async fn prune_to_limit(&self, max_edges: usize) -> Result<usize, Self::Error> {
        let mut edges = self.get_all_edges().await?;
        if edges.len() <= max_edges {
            return Ok(0);
        }
        edges.sort_by_key(|e| e.temporal.created_at);
        let to_remove = edges.len().saturating_sub(max_edges);
        for e in edges.into_iter().take(to_remove) {
            let _ = self.delete_edge_by_id(&e.id).await?;
        }
        Ok(to_remove)
    }

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
