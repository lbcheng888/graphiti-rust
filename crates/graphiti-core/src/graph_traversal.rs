//! Graph traversal algorithms and utilities

use crate::error::Error;
use crate::error::Result;
use crate::graph::Edge;
use crate::storage::Direction;
use crate::storage::GraphStorage;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use tracing::info;
use tracing::instrument;
use uuid::Uuid;

/// Graph traversal configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalConfig {
    /// Maximum depth to traverse
    pub max_depth: usize,
    /// Maximum number of nodes to visit
    pub max_nodes: usize,
    /// Relationship types to follow (empty = all)
    pub relationship_types: Vec<String>,
    /// Direction to traverse
    pub direction: Direction,
    /// Whether to include the starting node in results
    pub include_start: bool,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_nodes: 100,
            relationship_types: Vec::new(),
            direction: Direction::Both,
            include_start: true,
        }
    }
}

/// A node in the traversal result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalNode {
    /// The node ID
    pub id: Uuid,
    /// Distance from the starting node
    pub depth: usize,
    /// The edge that led to this node (None for starting node)
    pub incoming_edge: Option<Edge>,
    /// Path from start to this node (node IDs)
    pub path: Vec<Uuid>,
}

/// Graph traversal result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResult {
    /// Starting node ID
    pub start_node: Uuid,
    /// Visited nodes
    pub nodes: Vec<TraversalNode>,
    /// Total nodes visited
    pub total_visited: usize,
    /// Maximum depth reached
    pub max_depth_reached: usize,
}

/// Graph traversal algorithms
pub struct GraphTraversal<S>
where
    S: GraphStorage,
{
    storage: std::sync::Arc<S>,
}

impl<S> GraphTraversal<S>
where
    S: GraphStorage<Error = Error>,
{
    /// Create a new graph traversal instance
    pub fn new(storage: std::sync::Arc<S>) -> Self {
        Self { storage }
    }

    /// Perform breadth-first search from a starting node
    #[instrument(skip(self))]
    pub async fn bfs(&self, start_node: Uuid, config: &TraversalConfig) -> Result<TraversalResult> {
        info!("Starting BFS from node: {}", start_node);

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result_nodes = Vec::new();
        let mut max_depth_reached = 0;

        // Initialize with starting node
        if config.include_start {
            result_nodes.push(TraversalNode {
                id: start_node,
                depth: 0,
                incoming_edge: None,
                path: vec![start_node],
            });
        }

        queue.push_back((start_node, 0, vec![start_node]));
        visited.insert(start_node);

        while let Some((current_node, depth, path)) = queue.pop_front() {
            if depth >= config.max_depth || result_nodes.len() >= config.max_nodes {
                break;
            }

            max_depth_reached = max_depth_reached.max(depth);

            // Get edges from current node
            let edges = self
                .storage
                .get_edges(&current_node, config.direction)
                .await?;

            for edge in edges {
                // Filter by relationship type if specified
                if !config.relationship_types.is_empty()
                    && !config.relationship_types.contains(&edge.relationship)
                {
                    continue;
                }

                // Determine the next node based on direction
                let next_node = match config.direction {
                    Direction::Outgoing => edge.target_id,
                    Direction::Incoming => edge.source_id,
                    Direction::Both => {
                        if edge.source_id == current_node {
                            edge.target_id
                        } else {
                            edge.source_id
                        }
                    }
                };

                // Skip if already visited
                if visited.contains(&next_node) {
                    continue;
                }

                visited.insert(next_node);

                let mut new_path = path.clone();
                new_path.push(next_node);

                // Add to result
                result_nodes.push(TraversalNode {
                    id: next_node,
                    depth: depth + 1,
                    incoming_edge: Some(edge),
                    path: new_path.clone(),
                });

                // Add to queue for further exploration
                if depth + 1 < config.max_depth {
                    queue.push_back((next_node, depth + 1, new_path));
                }
            }
        }

        Ok(TraversalResult {
            start_node,
            nodes: result_nodes,
            total_visited: visited.len(),
            max_depth_reached,
        })
    }

    /// Perform depth-first search from a starting node
    #[instrument(skip(self))]
    pub async fn dfs(&self, start_node: Uuid, config: &TraversalConfig) -> Result<TraversalResult> {
        info!("Starting DFS from node: {}", start_node);

        let mut visited = HashSet::new();
        let mut result_nodes = Vec::new();
        let mut max_depth_reached = 0;
        let mut stack = Vec::new();

        // Initialize stack with starting node
        stack.push((start_node, 0, vec![start_node], None));

        while let Some((current_node, depth, path, incoming_edge)) = stack.pop() {
            // Check limits
            if depth >= config.max_depth || result_nodes.len() >= config.max_nodes {
                continue;
            }

            // Skip if already visited
            if visited.contains(&current_node) {
                continue;
            }
            visited.insert(current_node);

            max_depth_reached = max_depth_reached.max(depth);

            // Add to result (skip start node if not included)
            if depth > 0 || config.include_start {
                result_nodes.push(TraversalNode {
                    id: current_node,
                    depth,
                    incoming_edge,
                    path: path.clone(),
                });
            }

            // Get edges and add to stack (in reverse order for DFS)
            let edges = self
                .storage
                .get_edges(&current_node, config.direction)
                .await?;

            for edge in edges.into_iter().rev() {
                // Filter by relationship type if specified
                if !config.relationship_types.is_empty()
                    && !config.relationship_types.contains(&edge.relationship)
                {
                    continue;
                }

                // Determine the next node based on direction
                let next_node = match config.direction {
                    Direction::Outgoing => edge.target_id,
                    Direction::Incoming => edge.source_id,
                    Direction::Both => {
                        if edge.source_id == current_node {
                            edge.target_id
                        } else {
                            edge.source_id
                        }
                    }
                };

                // Skip if already visited
                if visited.contains(&next_node) {
                    continue;
                }

                let mut new_path = path.clone();
                new_path.push(next_node);

                // Add to stack for exploration
                stack.push((next_node, depth + 1, new_path, Some(edge)));
            }
        }

        Ok(TraversalResult {
            start_node,
            nodes: result_nodes,
            total_visited: visited.len(),
            max_depth_reached,
        })
    }

    /// Find shortest path between two nodes
    #[instrument(skip(self))]
    pub async fn shortest_path(
        &self,
        start_node: Uuid,
        target_node: Uuid,
        config: &TraversalConfig,
    ) -> Result<Option<Vec<Uuid>>> {
        info!(
            "Finding shortest path from {} to {}",
            start_node, target_node
        );

        if start_node == target_node {
            return Ok(Some(vec![start_node]));
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<Uuid, Uuid> = HashMap::new();

        queue.push_back((start_node, 0));
        visited.insert(start_node);

        while let Some((current_node, depth)) = queue.pop_front() {
            if depth >= config.max_depth {
                break;
            }

            // Get edges from current node
            let edges = self
                .storage
                .get_edges(&current_node, config.direction)
                .await?;

            for edge in edges {
                // Filter by relationship type if specified
                if !config.relationship_types.is_empty()
                    && !config.relationship_types.contains(&edge.relationship)
                {
                    continue;
                }

                // Determine the next node based on direction
                let next_node = match config.direction {
                    Direction::Outgoing => edge.target_id,
                    Direction::Incoming => edge.source_id,
                    Direction::Both => {
                        if edge.source_id == current_node {
                            edge.target_id
                        } else {
                            edge.source_id
                        }
                    }
                };

                // Skip if already visited
                if visited.contains(&next_node) {
                    continue;
                }

                visited.insert(next_node);
                parent.insert(next_node, current_node);

                // Check if we found the target
                if next_node == target_node {
                    // Reconstruct path
                    let mut path = Vec::new();
                    let mut current = target_node;
                    path.push(current);

                    while let Some(&prev) = parent.get(&current) {
                        path.push(prev);
                        current = prev;
                        if current == start_node {
                            break;
                        }
                    }

                    path.reverse();
                    return Ok(Some(path));
                }

                // Add to queue for further exploration
                queue.push_back((next_node, depth + 1));
            }
        }

        Ok(None) // No path found
    }
}
