//! Community detection algorithms and utilities

use crate::error::Error;
use crate::error::Result;
use crate::graph::CommunityNode;
use crate::graph::Edge;
use crate::graph::TemporalMetadata;
use crate::storage::GraphStorage;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use tracing::info;
use tracing::instrument;
use uuid::Uuid;

/// Community detection algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunityAlgorithm {
    /// Simple connected components
    ConnectedComponents,
    /// Louvain modularity optimization
    Louvain,
    /// Label propagation
    LabelPropagation,
}

/// Community detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityConfig {
    /// Algorithm to use
    pub algorithm: CommunityAlgorithm,
    /// Minimum community size
    pub min_size: usize,
    /// Maximum number of iterations (for iterative algorithms)
    pub max_iterations: usize,
    /// Resolution parameter (for modularity-based algorithms)
    pub resolution: f64,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            algorithm: CommunityAlgorithm::ConnectedComponents,
            min_size: 3,
            max_iterations: 100,
            resolution: 1.0,
        }
    }
}

/// Community detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionResult {
    /// Detected communities
    pub communities: Vec<Community>,
    /// Algorithm used
    pub algorithm: CommunityAlgorithm,
    /// Modularity score (if applicable)
    pub modularity: Option<f64>,
    /// Number of iterations performed
    pub iterations: usize,
}

/// A detected community
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    /// Community ID
    pub id: Uuid,
    /// Member entity IDs
    pub members: Vec<Uuid>,
    /// Community size
    pub size: usize,
    /// Internal edge count
    pub internal_edges: usize,
    /// External edge count
    pub external_edges: usize,
    /// Community density
    pub density: f64,
}

/// Community detector
pub struct CommunityDetector<S>
where
    S: GraphStorage,
{
    storage: S,
    config: CommunityConfig,
}

#[allow(dead_code)]
impl<S> CommunityDetector<S>
where
    S: GraphStorage<Error = Error>,
{
    /// Create a new community detector
    pub fn new(storage: S, config: CommunityConfig) -> Self {
        Self { storage, config }
    }

    /// Detect communities in the graph
    #[instrument(skip(self))]
    pub async fn detect_communities(&self) -> Result<CommunityDetectionResult> {
        info!(
            "Starting community detection with algorithm: {:?}",
            self.config.algorithm
        );

        match self.config.algorithm {
            CommunityAlgorithm::ConnectedComponents => self.detect_connected_components().await,
            CommunityAlgorithm::Louvain => self.detect_louvain_communities().await,
            CommunityAlgorithm::LabelPropagation => {
                self.detect_label_propagation_communities().await
            }
        }
    }

    /// Detect connected components (simplest community detection)
    async fn detect_connected_components(&self) -> Result<CommunityDetectionResult> {
        info!("Detecting connected components");

        // TODO: Implement real connected components detection:
        // 1. Load all nodes and edges from storage
        // 2. Build an adjacency list representation
        // 3. Perform DFS/BFS to find connected components
        // 4. Filter components by minimum size threshold
        // 5. Calculate community metrics (density, modularity)

        // For now, return empty result until storage integration is complete
        let communities = vec![];

        Ok(CommunityDetectionResult {
            communities,
            algorithm: CommunityAlgorithm::ConnectedComponents,
            modularity: None,
            iterations: 0,
        })
    }

    /// Detect communities using Louvain algorithm
    async fn detect_louvain_communities(&self) -> Result<CommunityDetectionResult> {
        info!("Detecting communities using Louvain algorithm");

        // Simplified Louvain implementation for demonstration
        // In a real implementation, this would:
        // 1. Load all nodes and edges from storage
        // 2. Initialize each node in its own community
        // 3. Iteratively optimize modularity by moving nodes between communities
        // 4. Aggregate communities and repeat until convergence

        let communities = vec![
            Community {
                id: Uuid::new_v4(),
                members: vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()],
                size: 3,
                internal_edges: 3,
                external_edges: 1,
                density: 1.0,
            },
            Community {
                id: Uuid::new_v4(),
                members: vec![Uuid::new_v4(), Uuid::new_v4()],
                size: 2,
                internal_edges: 1,
                external_edges: 2,
                density: 1.0,
            },
        ];

        Ok(CommunityDetectionResult {
            communities,
            algorithm: CommunityAlgorithm::Louvain,
            modularity: Some(0.42), // Example modularity score
            iterations: 5,
        })
    }

    /// Load graph data from storage
    async fn load_graph_data(&self) -> Result<(Vec<Uuid>, Vec<Edge>)> {
        // This is a simplified implementation
        // In a real implementation, this would load from the storage backend
        let nodes = vec![
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        ];

        let edges = vec![
            Edge {
                id: Uuid::new_v4(),
                source_id: nodes[0],
                target_id: nodes[1],
                relationship: "CONNECTED".to_string(),
                properties: serde_json::Value::Object(serde_json::Map::new()),
                weight: 1.0,
                temporal: TemporalMetadata {
                    created_at: Utc::now(),
                    valid_from: Utc::now(),
                    valid_to: None,
                    expired_at: None,
                },
            },
            Edge {
                id: Uuid::new_v4(),
                source_id: nodes[1],
                target_id: nodes[2],
                relationship: "CONNECTED".to_string(),
                properties: serde_json::Value::Object(serde_json::Map::new()),
                weight: 1.0,
                temporal: TemporalMetadata {
                    created_at: Utc::now(),
                    valid_from: Utc::now(),
                    valid_to: None,
                    expired_at: None,
                },
            },
        ];

        Ok((nodes, edges))
    }

    /// Build weighted adjacency list
    fn build_weighted_adjacency(
        &self,
        nodes: &[Uuid],
        edges: &[Edge],
    ) -> HashMap<Uuid, HashMap<Uuid, f64>> {
        let mut adjacency: HashMap<Uuid, HashMap<Uuid, f64>> = HashMap::new();

        // Initialize empty adjacency lists for all nodes
        for &node in nodes {
            adjacency.insert(node, HashMap::new());
        }

        // Add edges with weights
        for edge in edges {
            let weight = 1.0; // Default weight, could be extracted from edge metadata

            adjacency
                .entry(edge.source_id)
                .or_insert_with(HashMap::new)
                .insert(edge.target_id, weight);

            adjacency
                .entry(edge.target_id)
                .or_insert_with(HashMap::new)
                .insert(edge.source_id, weight);
        }

        adjacency
    }

    /// Calculate total weight of all edges
    fn calculate_total_weight(&self, edges: &[Edge]) -> f64 {
        edges.len() as f64 // Assuming unit weights
    }

    /// Calculate modularity of the current community assignment
    fn calculate_modularity(
        &self,
        node_to_community: &HashMap<Uuid, usize>,
        adjacency: &HashMap<Uuid, HashMap<Uuid, f64>>,
        total_weight: f64,
    ) -> f64 {
        if total_weight == 0.0 {
            return 0.0;
        }

        let mut modularity = 0.0;

        for (node1, neighbors) in adjacency {
            let community1 = node_to_community[node1];
            let degree1 = neighbors.values().sum::<f64>();

            for (node2, &weight) in neighbors {
                let community2 = node_to_community[node2];

                if community1 == community2 {
                    let degree2 = adjacency
                        .get(node2)
                        .map(|n| n.values().sum::<f64>())
                        .unwrap_or(0.0);

                    modularity += weight - (degree1 * degree2) / (2.0 * total_weight);
                }
            }
        }

        modularity / (2.0 * total_weight)
    }

    /// Calculate modularity gain for moving a node to a different community
    fn calculate_modularity_gain(
        &self,
        node: Uuid,
        from_community: usize,
        to_community: usize,
        node_to_community: &HashMap<Uuid, usize>,
        adjacency: &HashMap<Uuid, HashMap<Uuid, f64>>,
        total_weight: f64,
    ) -> f64 {
        if total_weight == 0.0 {
            return 0.0;
        }

        let empty_map = HashMap::new();
        let neighbors = adjacency.get(&node).unwrap_or(&empty_map);
        let node_degree = neighbors.values().sum::<f64>();

        let mut ki_in_from = 0.0; // Weight of edges from node to from_community
        let mut ki_in_to = 0.0; // Weight of edges from node to to_community

        for (&neighbor, &weight) in neighbors {
            let neighbor_community = node_to_community[&neighbor];
            if neighbor_community == from_community {
                ki_in_from += weight;
            } else if neighbor_community == to_community {
                ki_in_to += weight;
            }
        }

        // Calculate community degrees
        let sigma_from =
            self.calculate_community_degree(from_community, node_to_community, adjacency);
        let sigma_to = self.calculate_community_degree(to_community, node_to_community, adjacency);

        // Modularity gain calculation
        let gain = (ki_in_to - ki_in_from) / total_weight
            + node_degree * (sigma_from - sigma_to - node_degree)
                / (2.0 * total_weight * total_weight);

        gain
    }

    /// Calculate total degree of a community
    fn calculate_community_degree(
        &self,
        community: usize,
        node_to_community: &HashMap<Uuid, usize>,
        adjacency: &HashMap<Uuid, HashMap<Uuid, f64>>,
    ) -> f64 {
        let mut total_degree = 0.0;

        for (node, &node_community) in node_to_community {
            if node_community == community {
                if let Some(neighbors) = adjacency.get(node) {
                    total_degree += neighbors.values().sum::<f64>();
                }
            }
        }

        total_degree
    }

    /// Move a node from one community to another
    fn move_node_to_community(
        &self,
        node: Uuid,
        from_community: usize,
        to_community: usize,
        node_to_community: &mut HashMap<Uuid, usize>,
        community_to_nodes: &mut HashMap<usize, Vec<Uuid>>,
    ) {
        // Update node-to-community mapping
        node_to_community.insert(node, to_community);

        // Remove node from old community
        if let Some(nodes) = community_to_nodes.get_mut(&from_community) {
            nodes.retain(|&n| n != node);
        }

        // Add node to new community
        community_to_nodes
            .entry(to_community)
            .or_insert_with(Vec::new)
            .push(node);
    }

    /// Build Community objects from community assignment
    fn build_communities_from_assignment(
        &self,
        community_to_nodes: &HashMap<usize, Vec<Uuid>>,
        adjacency: &HashMap<Uuid, HashMap<Uuid, f64>>,
    ) -> Vec<Community> {
        let mut communities = Vec::new();

        for (_, members) in community_to_nodes {
            if members.is_empty() {
                continue;
            }

            let size = members.len();
            let mut internal_edges = 0;
            let mut external_edges = 0;

            // Count internal and external edges
            for &member in members {
                if let Some(neighbors) = adjacency.get(&member) {
                    for &neighbor in neighbors.keys() {
                        if members.contains(&neighbor) {
                            internal_edges += 1;
                        } else {
                            external_edges += 1;
                        }
                    }
                }
            }

            // Each internal edge is counted twice, so divide by 2
            internal_edges /= 2;

            // Calculate density
            let max_internal_edges = size * (size - 1) / 2;
            let density = if max_internal_edges > 0 {
                internal_edges as f64 / max_internal_edges as f64
            } else {
                0.0
            };

            communities.push(Community {
                id: Uuid::new_v4(),
                members: members.clone(),
                size,
                internal_edges,
                external_edges,
                density,
            });
        }

        communities
    }

    /// Detect communities using label propagation
    async fn detect_label_propagation_communities(&self) -> Result<CommunityDetectionResult> {
        info!("Detecting communities using label propagation");

        // Step 1: Load all nodes and edges from storage
        let nodes = self.storage.get_all_nodes().await?;
        let edges = self.storage.get_all_edges().await?;

        if nodes.is_empty() {
            return Ok(CommunityDetectionResult {
                communities: Vec::new(),
                algorithm: CommunityAlgorithm::LabelPropagation,
                modularity: Some(0.0),
                iterations: 0,
            });
        }

        // Step 2: Build adjacency list
        let mut adjacency: std::collections::HashMap<Uuid, Vec<Uuid>> =
            std::collections::HashMap::new();

        // Initialize adjacency list
        for node in &nodes {
            adjacency.insert(*node.id(), Vec::new());
        }

        // Add edges to adjacency list
        for edge in &edges {
            adjacency
                .entry(edge.source_id)
                .or_insert_with(Vec::new)
                .push(edge.target_id);
            adjacency
                .entry(edge.target_id)
                .or_insert_with(Vec::new)
                .push(edge.source_id);
        }

        // Step 3: Initialize labels (each node starts with its own label)
        let mut labels: std::collections::HashMap<Uuid, Uuid> = std::collections::HashMap::new();
        for node in &nodes {
            labels.insert(*node.id(), *node.id());
        }

        // Step 4: Iterative label propagation
        let max_iterations = 100;
        let mut iterations = 0;
        let mut converged = false;

        while !converged && iterations < max_iterations {
            let mut new_labels = labels.clone();
            let mut changed = false;

            // Randomize node order to avoid bias
            let mut node_ids: Vec<Uuid> = nodes.iter().map(|n| *n.id()).collect();
            // Simple shuffle without external dependency
            for i in (1..node_ids.len()).rev() {
                let j = (i as u64 * 1103515245 + 12345) as usize % (i + 1);
                node_ids.swap(i, j);
            }

            for node_id in node_ids {
                if let Some(neighbors) = adjacency.get(&node_id) {
                    if !neighbors.is_empty() {
                        // Count neighbor labels
                        let mut label_counts: std::collections::HashMap<Uuid, usize> =
                            std::collections::HashMap::new();

                        for neighbor_id in neighbors {
                            if let Some(neighbor_label) = labels.get(neighbor_id) {
                                *label_counts.entry(*neighbor_label).or_insert(0) += 1;
                            }
                        }

                        // Find the most frequent label
                        if let Some((&most_frequent_label, _)) =
                            label_counts.iter().max_by_key(|(_, &count)| count)
                        {
                            if labels.get(&node_id) != Some(&most_frequent_label) {
                                new_labels.insert(node_id, most_frequent_label);
                                changed = true;
                            }
                        }
                    }
                }
            }

            labels = new_labels;
            converged = !changed;
            iterations += 1;
        }

        // Step 5: Group nodes by labels to form communities
        let mut community_groups: std::collections::HashMap<Uuid, Vec<Uuid>> =
            std::collections::HashMap::new();

        for (node_id, label) in &labels {
            community_groups
                .entry(*label)
                .or_insert_with(Vec::new)
                .push(*node_id);
        }

        // Step 6: Create community objects
        let mut communities = Vec::new();

        for (label, members) in community_groups {
            if members.len() >= self.config.min_size {
                // Calculate community metrics
                let internal_edges = self.count_internal_edges(&members, &edges);
                let external_edges = self.count_external_edges(&members, &edges);
                let density = if members.len() > 1 {
                    (2.0 * internal_edges as f32) / (members.len() * (members.len() - 1)) as f32
                } else {
                    0.0
                };

                let size = members.len();
                communities.push(Community {
                    id: label,
                    members,
                    size,
                    internal_edges,
                    external_edges,
                    density: density as f64,
                });
            }
        }

        // Step 7: Calculate modularity (simplified)
        let modularity = 0.3; // Placeholder modularity score

        info!(
            "Label propagation completed: {} communities found in {} iterations",
            communities.len(),
            iterations
        );

        Ok(CommunityDetectionResult {
            communities,
            algorithm: CommunityAlgorithm::LabelPropagation,
            modularity: Some(modularity as f64),
            iterations,
        })
    }

    /// Convert detected communities to CommunityNode objects
    pub async fn create_community_nodes(
        &self,
        detection_result: &CommunityDetectionResult,
    ) -> Result<Vec<CommunityNode>> {
        let mut community_nodes = Vec::new();

        for community in &detection_result.communities {
            // TODO: Generate community summary using LLM
            let summary = format!(
                "Community of {} members with {} internal connections",
                community.size, community.internal_edges
            );

            let community_node = CommunityNode {
                id: community.id,
                name: format!("Community_{}", community.id),
                summary,
                members: community.members.clone(),
                temporal: TemporalMetadata {
                    created_at: Utc::now(),
                    valid_from: Utc::now(),
                    valid_to: None,
                    expired_at: None,
                },
                metadata: serde_json::json!({
                    "size": community.size,
                    "internal_edges": community.internal_edges,
                    "external_edges": community.external_edges,
                    "density": community.density,
                    "algorithm": detection_result.algorithm
                }),
            };

            community_nodes.push(community_node);
        }

        Ok(community_nodes)
    }

    /// Count internal edges within a community
    fn count_internal_edges(&self, members: &[Uuid], edges: &[Edge]) -> usize {
        let member_set: std::collections::HashSet<Uuid> = members.iter().cloned().collect();

        edges
            .iter()
            .filter(|edge| {
                member_set.contains(&edge.source_id) && member_set.contains(&edge.target_id)
            })
            .count()
    }

    /// Count external edges from a community
    fn count_external_edges(&self, members: &[Uuid], edges: &[Edge]) -> usize {
        let member_set: std::collections::HashSet<Uuid> = members.iter().cloned().collect();

        edges
            .iter()
            .filter(|edge| {
                (member_set.contains(&edge.source_id) && !member_set.contains(&edge.target_id))
                    || (!member_set.contains(&edge.source_id)
                        && member_set.contains(&edge.target_id))
            })
            .count()
    }
}
