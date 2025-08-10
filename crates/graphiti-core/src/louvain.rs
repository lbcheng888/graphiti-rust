//! Louvain community detection algorithm implementation

use crate::error::Error;
use crate::error::Result;
use crate::graph::Edge;

use crate::storage::GraphStorage;

#[cfg(test)]
mod tests;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::HashSet;
use uuid::Uuid;

/// Louvain algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LouvainConfig {
    /// Resolution parameter for modularity optimization
    pub resolution: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Minimum modularity improvement threshold
    pub min_improvement: f64,
    /// Whether to use randomization in node ordering
    pub randomize: bool,
    /// Random seed for reproducible results
    pub seed: Option<u64>,
}

impl Default for LouvainConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_iterations: 100,
            min_improvement: 1e-6,
            randomize: true,
            seed: None,
        }
    }
}

/// Community assignment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LouvainResult {
    /// Node to community mapping
    pub communities: HashMap<Uuid, usize>,
    /// Final modularity score
    pub modularity: f64,
    /// Number of communities found
    pub num_communities: usize,
    /// Number of iterations performed
    pub iterations: usize,
    /// Community hierarchy (for hierarchical Louvain)
    pub hierarchy: Vec<HashMap<Uuid, usize>>,
}

/// Weighted edge for Louvain algorithm
#[derive(Debug, Clone)]
struct WeightedEdge {
    source: Uuid,
    target: Uuid,
    weight: f64,
}

/// Node degree information
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct NodeDegree {
    node_id: Uuid,
    degree: f64,
    community: usize,
}

/// Community information
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Community {
    id: usize,
    total_weight: f64,
    internal_weight: f64,
    nodes: HashSet<Uuid>,
}

/// Louvain community detection algorithm
pub struct LouvainDetector<S> {
    storage: S,
    config: LouvainConfig,
}

#[allow(dead_code)]
impl<S> LouvainDetector<S>
where
    S: GraphStorage<Error = Error>,
{
    /// Create a new Louvain detector
    pub fn new(storage: S, config: LouvainConfig) -> Self {
        Self { storage, config }
    }

    /// Run Louvain community detection
    pub async fn detect_communities(&self) -> Result<LouvainResult> {
        // Load graph data
        let (nodes, edges) = self.load_graph_data().await?;

        if nodes.is_empty() {
            return Ok(LouvainResult {
                communities: HashMap::new(),
                modularity: 0.0,
                num_communities: 0,
                iterations: 0,
                hierarchy: vec![],
            });
        }

        // Convert to weighted edges and handle undirected graph
        let weighted_edges = self.convert_to_weighted_edges(&edges);

        // Calculate total edge weight (m)
        let total_edge_weight = self.calculate_total_edge_weight(&weighted_edges);

        if total_edge_weight == 0.0 {
            // No edges - each node is its own community
            let final_assignment: HashMap<Uuid, usize> = nodes
                .iter()
                .enumerate()
                .map(|(i, &node_id)| (node_id, i))
                .collect();

            return Ok(LouvainResult {
                communities: final_assignment,
                modularity: 0.0,
                num_communities: nodes.len(),
                iterations: 0,
                hierarchy: vec![],
            });
        }

        // Calculate node degrees (sum of edge weights for each node)
        let node_degrees = self.calculate_node_degrees_optimized(&nodes, &weighted_edges);

        // Initialize: each node in its own community
        let mut community_assignment: HashMap<Uuid, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node_id)| (node_id, i))
            .collect();

        let mut current_modularity = self.calculate_modularity_optimized(
            &community_assignment,
            &weighted_edges,
            &node_degrees,
            total_edge_weight,
        );

        let mut hierarchy = vec![community_assignment.clone()];
        let mut iteration = 0;
        let mut improvement = f64::INFINITY;

        // Main Louvain loop
        while iteration < self.config.max_iterations && improvement > self.config.min_improvement {
            let mut nodes_moved = 0;
            let previous_modularity = current_modularity;

            // Randomize node order if configured
            let mut node_order: Vec<Uuid> = nodes.clone();
            if self.config.randomize {
                // Better pseudo-random shuffle
                if let Some(seed) = self.config.seed {
                    node_order.sort_by_key(|&id| {
                        let mut hash = seed;
                        for _byte in id.as_bytes() {
                            hash = hash.wrapping_mul(1103515245).wrapping_add(12345);
                        }
                        hash
                    });
                } else {
                    node_order.sort_by_key(|&id| {
                        let mut hash = 0u64;
                        for byte in id.as_bytes() {
                            hash = hash.wrapping_mul(31).wrapping_add(*byte as u64);
                        }
                        hash
                    });
                }
            }

            // Phase 1: Local optimization - try to move each node to a better community
            for &node_id in &node_order {
                let current_community = community_assignment[&node_id];
                let (best_community, best_gain) = self.find_best_community_optimized(
                    node_id,
                    current_community,
                    &community_assignment,
                    &weighted_edges,
                    &node_degrees,
                    total_edge_weight,
                );

                // Only move if there's a positive gain
                if best_gain > 0.0 && best_community != current_community {
                    community_assignment.insert(node_id, best_community);
                    nodes_moved += 1;
                }
            }

            // Calculate new modularity
            current_modularity = self.calculate_modularity_optimized(
                &community_assignment,
                &weighted_edges,
                &node_degrees,
                total_edge_weight,
            );

            improvement = current_modularity - previous_modularity;

            // If no nodes moved or improvement is too small, stop
            if nodes_moved == 0 || improvement <= self.config.min_improvement {
                break;
            }

            hierarchy.push(community_assignment.clone());
            iteration += 1;
        }

        // Post-processing: merge small communities to reduce over-splitting
        let merged_assignment = self.merge_small_communities(
            &community_assignment,
            &weighted_edges,
            &node_degrees,
            total_edge_weight,
        );

        // Recalculate modularity after merging
        let final_modularity = self.calculate_modularity_optimized(
            &merged_assignment,
            &weighted_edges,
            &node_degrees,
            total_edge_weight,
        );

        // Renumber communities to be consecutive
        let final_assignment = self.renumber_communities(&merged_assignment);
        let num_communities = final_assignment.values().max().map(|&x| x + 1).unwrap_or(0);

        Ok(LouvainResult {
            communities: final_assignment,
            modularity: final_modularity,
            num_communities,
            iterations: iteration,
            hierarchy,
        })
    }

    /// Load graph data from storage
    async fn load_graph_data(&self) -> Result<(Vec<Uuid>, Vec<Edge>)> {
        // Get all nodes
        let nodes = self.storage.get_all_nodes().await?;
        let node_ids: Vec<Uuid> = nodes.into_iter().map(|node| *node.id()).collect();

        // Get all edges
        let edges = self.storage.get_all_edges().await?;

        Ok((node_ids, edges))
    }

    /// Convert edges to weighted edges
    fn convert_to_weighted_edges(&self, edges: &[Edge]) -> Vec<WeightedEdge> {
        edges
            .iter()
            .map(|edge| WeightedEdge {
                source: edge.source_id,
                target: edge.target_id,
                weight: edge.weight as f64,
            })
            .collect()
    }

    /// Calculate node degrees
    fn calculate_node_degrees(
        &self,
        nodes: &[Uuid],
        edges: &[WeightedEdge],
    ) -> HashMap<Uuid, NodeDegree> {
        let mut degrees = HashMap::new();

        // Initialize degrees
        for (i, &node_id) in nodes.iter().enumerate() {
            degrees.insert(
                node_id,
                NodeDegree {
                    node_id,
                    degree: 0.0,
                    community: i,
                },
            );
        }

        // Calculate degrees from edges
        for edge in edges {
            if let Some(source_degree) = degrees.get_mut(&edge.source) {
                source_degree.degree += edge.weight;
            }
            if let Some(target_degree) = degrees.get_mut(&edge.target) {
                target_degree.degree += edge.weight;
            }
        }

        degrees
    }

    /// Initialize communities (each node in its own community)
    fn initialize_communities(&self, nodes: &[Uuid]) -> Vec<Community> {
        nodes
            .iter()
            .enumerate()
            .map(|(i, &node_id)| Community {
                id: i,
                total_weight: 0.0,
                internal_weight: 0.0,
                nodes: {
                    let mut set = HashSet::new();
                    set.insert(node_id);
                    set
                },
            })
            .collect()
    }

    /// Calculate total weight of all edges
    fn calculate_total_weight(&self, edges: &[WeightedEdge]) -> f64 {
        edges.iter().map(|edge| edge.weight).sum()
    }

    /// Calculate total edge weight (for undirected graph, each edge contributes once)
    fn calculate_total_edge_weight(&self, edges: &[WeightedEdge]) -> f64 {
        edges.iter().map(|edge| edge.weight).sum()
    }

    /// Optimized node degree calculation
    fn calculate_node_degrees_optimized(
        &self,
        nodes: &[Uuid],
        edges: &[WeightedEdge],
    ) -> HashMap<Uuid, f64> {
        let mut degrees = HashMap::new();

        // Initialize all nodes with degree 0
        for &node_id in nodes {
            degrees.insert(node_id, 0.0);
        }

        // Calculate degrees from edges (for undirected graph, each edge contributes to both nodes)
        for edge in edges {
            *degrees.entry(edge.source).or_insert(0.0) += edge.weight;
            *degrees.entry(edge.target).or_insert(0.0) += edge.weight;
        }

        degrees
    }

    /// Calculate modularity score
    fn calculate_modularity(
        &self,
        assignment: &HashMap<Uuid, usize>,
        edges: &[WeightedEdge],
        node_degrees: &HashMap<Uuid, NodeDegree>,
        total_weight: f64,
    ) -> f64 {
        if total_weight == 0.0 {
            return 0.0;
        }

        let mut modularity = 0.0;
        let two_m = 2.0 * total_weight;

        // Calculate modularity for each edge
        for edge in edges {
            let source_community = assignment.get(&edge.source).copied().unwrap_or(0);
            let target_community = assignment.get(&edge.target).copied().unwrap_or(0);

            if source_community == target_community {
                let source_degree = node_degrees
                    .get(&edge.source)
                    .map(|d| d.degree)
                    .unwrap_or(0.0);
                let target_degree = node_degrees
                    .get(&edge.target)
                    .map(|d| d.degree)
                    .unwrap_or(0.0);

                modularity += edge.weight - (source_degree * target_degree) / two_m;
            }
        }

        modularity / total_weight * self.config.resolution
    }

    /// Optimized modularity calculation using correct formula
    fn calculate_modularity_optimized(
        &self,
        assignment: &HashMap<Uuid, usize>,
        edges: &[WeightedEdge],
        node_degrees: &HashMap<Uuid, f64>,
        total_edge_weight: f64,
    ) -> f64 {
        if total_edge_weight == 0.0 {
            return 0.0;
        }

        let two_m = 2.0 * total_edge_weight;
        let mut modularity = 0.0;

        // Create adjacency map for efficient lookup
        let mut adjacency: HashMap<(Uuid, Uuid), f64> = HashMap::new();
        for edge in edges {
            adjacency.insert((edge.source, edge.target), edge.weight);
            adjacency.insert((edge.target, edge.source), edge.weight); // For undirected graph
        }

        // Calculate modularity using the standard formula:
        // Q = (1/2m) * Σ[A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)
        // We need to consider all node pairs, not just edges
        let nodes: Vec<Uuid> = node_degrees.keys().copied().collect();

        for i in 0..nodes.len() {
            for j in i..nodes.len() {
                // Only consider upper triangle to avoid double counting
                let node_i = nodes[i];
                let node_j = nodes[j];

                let community_i = assignment.get(&node_i).copied().unwrap_or(0);
                let community_j = assignment.get(&node_j).copied().unwrap_or(0);

                // Only calculate for nodes in the same community
                if community_i == community_j {
                    // A_ij term (actual edge weight)
                    let edge_weight = if i == j {
                        // Self-loop case
                        adjacency.get(&(node_i, node_j)).copied().unwrap_or(0.0)
                    } else {
                        // Regular edge case
                        adjacency.get(&(node_i, node_j)).copied().unwrap_or(0.0)
                    };

                    // k_i * k_j / (2m) term (expected edge weight)
                    let degree_i = node_degrees.get(&node_i).copied().unwrap_or(0.0);
                    let degree_j = node_degrees.get(&node_j).copied().unwrap_or(0.0);
                    let expected_weight = (degree_i * degree_j) / two_m;

                    // Add contribution to modularity
                    if i == j {
                        // Self-loop contributes once
                        modularity += edge_weight - expected_weight;
                    } else {
                        // Regular edge contributes twice (since we only consider upper triangle)
                        modularity += 2.0 * (edge_weight - expected_weight);
                    }
                }
            }
        }

        // Apply resolution parameter and normalize by 2m
        (modularity / two_m) * self.config.resolution
    }

    /// Find the best community for a node
    fn find_best_community(
        &self,
        node_id: Uuid,
        current_community: usize,
        assignment: &HashMap<Uuid, usize>,
        edges: &[WeightedEdge],
        node_degrees: &HashMap<Uuid, NodeDegree>,
        total_weight: f64,
    ) -> usize {
        let mut best_community = current_community;
        let mut best_gain = 0.0;

        // Get neighboring communities
        let mut neighbor_communities = HashSet::new();
        neighbor_communities.insert(current_community);

        for edge in edges {
            if edge.source == node_id {
                if let Some(&community) = assignment.get(&edge.target) {
                    neighbor_communities.insert(community);
                }
            } else if edge.target == node_id {
                if let Some(&community) = assignment.get(&edge.source) {
                    neighbor_communities.insert(community);
                }
            }
        }

        // Calculate gain for each neighboring community
        for &community in &neighbor_communities {
            let gain = self.calculate_modularity_gain(
                node_id,
                current_community,
                community,
                assignment,
                edges,
                node_degrees,
                total_weight,
            );

            if gain > best_gain {
                best_gain = gain;
                best_community = community;
            }
        }

        best_community
    }

    /// Optimized method to find the best community for a node
    fn find_best_community_optimized(
        &self,
        node_id: Uuid,
        current_community: usize,
        assignment: &HashMap<Uuid, usize>,
        edges: &[WeightedEdge],
        node_degrees: &HashMap<Uuid, f64>,
        total_edge_weight: f64,
    ) -> (usize, f64) {
        let mut best_community = current_community;
        let mut best_gain = 0.0;

        // Get neighboring communities and their connection weights
        let mut neighbor_weights: HashMap<usize, f64> = HashMap::new();
        neighbor_weights.insert(current_community, 0.0);

        for edge in edges {
            if edge.source == node_id {
                if let Some(&community) = assignment.get(&edge.target) {
                    *neighbor_weights.entry(community).or_insert(0.0) += edge.weight;
                }
            } else if edge.target == node_id {
                if let Some(&community) = assignment.get(&edge.source) {
                    *neighbor_weights.entry(community).or_insert(0.0) += edge.weight;
                }
            }
        }

        let node_degree = node_degrees.get(&node_id).copied().unwrap_or(0.0);
        let two_m = 2.0 * total_edge_weight;

        // Calculate current community's total degree (excluding the node we're considering moving)
        let current_community_degree: f64 = assignment
            .iter()
            .filter(|(_, &comm)| comm == current_community)
            .filter(|(&node, _)| node != node_id) // Exclude the node we're moving
            .map(|(&node, _)| node_degrees.get(&node).copied().unwrap_or(0.0))
            .sum();

        // Calculate gain for each neighboring community
        for (&community, &weight_to_community) in &neighbor_weights {
            if community == current_community {
                continue; // Skip current community for now
            }

            // Calculate target community's total degree
            let target_community_degree: f64 = assignment
                .iter()
                .filter(|(_, &comm)| comm == community)
                .map(|(&node, _)| node_degrees.get(&node).copied().unwrap_or(0.0))
                .sum();

            // Calculate modularity gain using the correct formula
            // ΔQ = [k_i_in - k_i * Σ_tot / (2m)] / m
            // where k_i_in is the sum of weights from node i to nodes in the target community
            // k_i is the degree of node i
            // Σ_tot is the sum of degrees of nodes in the target community

            let gain_from_joining =
                weight_to_community - (node_degree * target_community_degree) / two_m;
            let loss_from_leaving = neighbor_weights
                .get(&current_community)
                .copied()
                .unwrap_or(0.0)
                - (node_degree * current_community_degree) / two_m;

            let total_gain = gain_from_joining - loss_from_leaving;

            // Apply a small bias towards staying in current community to reduce over-splitting
            let adjusted_gain = if community != current_community {
                total_gain - 0.001 // Small penalty for moving
            } else {
                total_gain
            };

            if adjusted_gain > best_gain {
                best_gain = adjusted_gain;
                best_community = community;
            }
        }

        (best_community, best_gain)
    }

    /// Calculate modularity gain from moving a node to a different community
    fn calculate_modularity_gain(
        &self,
        node_id: Uuid,
        from_community: usize,
        to_community: usize,
        assignment: &HashMap<Uuid, usize>,
        edges: &[WeightedEdge],
        node_degrees: &HashMap<Uuid, NodeDegree>,
        total_weight: f64,
    ) -> f64 {
        if from_community == to_community {
            return 0.0;
        }

        let node_degree = node_degrees.get(&node_id).map(|d| d.degree).unwrap_or(0.0);
        let two_m = 2.0 * total_weight;

        // Calculate connections to communities
        let mut to_community_weight = 0.0;
        let mut from_community_weight = 0.0;

        for edge in edges {
            if edge.source == node_id {
                if let Some(&community) = assignment.get(&edge.target) {
                    if community == to_community {
                        to_community_weight += edge.weight;
                    } else if community == from_community {
                        from_community_weight += edge.weight;
                    }
                }
            } else if edge.target == node_id {
                if let Some(&community) = assignment.get(&edge.source) {
                    if community == to_community {
                        to_community_weight += edge.weight;
                    } else if community == from_community {
                        from_community_weight += edge.weight;
                    }
                }
            }
        }

        // Calculate community degrees
        let to_community_degree =
            self.calculate_community_degree(to_community, assignment, node_degrees);
        let from_community_degree =
            self.calculate_community_degree(from_community, assignment, node_degrees);

        // Calculate gain
        let gain = (to_community_weight - from_community_weight) / total_weight
            - self.config.resolution * node_degree * (to_community_degree - from_community_degree)
                / two_m;

        gain
    }

    /// Calculate total degree of a community
    fn calculate_community_degree(
        &self,
        community: usize,
        assignment: &HashMap<Uuid, usize>,
        node_degrees: &HashMap<Uuid, NodeDegree>,
    ) -> f64 {
        assignment
            .iter()
            .filter(|(_, &comm)| comm == community)
            .map(|(&node_id, _)| node_degrees.get(&node_id).map(|d| d.degree).unwrap_or(0.0))
            .sum()
    }

    /// Create super-graph for hierarchical Louvain
    fn create_super_graph(
        &self,
        assignment: &HashMap<Uuid, usize>,
        edges: &[WeightedEdge],
    ) -> (Vec<Uuid>, Vec<WeightedEdge>, HashMap<usize, Uuid>) {
        // Create mapping from community to super-node
        let mut community_to_super_node = HashMap::new();
        let mut super_nodes = Vec::new();

        for &community in assignment.values() {
            if !community_to_super_node.contains_key(&community) {
                let super_node_id = Uuid::new_v4();
                community_to_super_node.insert(community, super_node_id);
                super_nodes.push(super_node_id);
            }
        }

        // Create super-edges
        let mut super_edge_weights: HashMap<(Uuid, Uuid), f64> = HashMap::new();

        for edge in edges {
            let source_community = assignment.get(&edge.source).copied().unwrap_or(0);
            let target_community = assignment.get(&edge.target).copied().unwrap_or(0);

            if source_community != target_community {
                let source_super = community_to_super_node[&source_community];
                let target_super = community_to_super_node[&target_community];

                let key = if source_super < target_super {
                    (source_super, target_super)
                } else {
                    (target_super, source_super)
                };

                *super_edge_weights.entry(key).or_insert(0.0) += edge.weight;
            }
        }

        let super_edges: Vec<WeightedEdge> = super_edge_weights
            .into_iter()
            .map(|((source, target), weight)| WeightedEdge {
                source,
                target,
                weight,
            })
            .collect();

        (super_nodes, super_edges, community_to_super_node)
    }

    /// Detect communities on super-graph (simplified recursive call)
    async fn detect_communities_on_super_graph(
        &self,
        _super_nodes: &[Uuid],
        _super_edges: &[WeightedEdge],
        _super_assignment: &HashMap<usize, Uuid>,
    ) -> Result<LouvainResult> {
        // Simplified implementation - in practice, this would recursively apply Louvain
        Ok(LouvainResult {
            communities: HashMap::new(),
            modularity: 0.0,
            num_communities: 0,
            iterations: 0,
            hierarchy: vec![],
        })
    }

    /// Merge small communities to reduce over-splitting
    fn merge_small_communities(
        &self,
        assignment: &HashMap<Uuid, usize>,
        edges: &[WeightedEdge],
        node_degrees: &HashMap<Uuid, f64>,
        total_edge_weight: f64,
    ) -> HashMap<Uuid, usize> {
        let mut merged_assignment = assignment.clone();

        // Count nodes in each community
        let mut community_sizes: HashMap<usize, usize> = HashMap::new();
        for &community in assignment.values() {
            *community_sizes.entry(community).or_insert(0) += 1;
        }

        // Find communities with only 1 node (singletons)
        let singleton_communities: Vec<usize> = community_sizes
            .iter()
            .filter(|(_, &size)| size == 1)
            .map(|(&community, _)| community)
            .collect();

        // For each singleton, try to merge it with the best neighboring community
        for singleton_community in singleton_communities {
            // Find the node in this singleton community
            let singleton_node = assignment
                .iter()
                .find(|(_, &comm)| comm == singleton_community)
                .map(|(&node, _)| node);

            if let Some(node_id) = singleton_node {
                // Find the best community to merge with
                let (best_community, best_gain) = self.find_best_community_optimized(
                    node_id,
                    singleton_community,
                    &merged_assignment,
                    edges,
                    node_degrees,
                    total_edge_weight,
                );

                // Merge if there's any positive gain or if it's better than staying alone
                if best_community != singleton_community && best_gain >= -0.01 {
                    merged_assignment.insert(node_id, best_community);
                }
            }
        }

        merged_assignment
    }

    /// Renumber communities to be consecutive starting from 0
    fn renumber_communities(&self, assignment: &HashMap<Uuid, usize>) -> HashMap<Uuid, usize> {
        let mut unique_communities: Vec<usize> = assignment.values().copied().collect();
        unique_communities.sort_unstable();
        unique_communities.dedup();

        let community_mapping: HashMap<usize, usize> = unique_communities
            .into_iter()
            .enumerate()
            .map(|(new_id, old_id)| (old_id, new_id))
            .collect();

        assignment
            .iter()
            .map(|(&node_id, &old_community)| {
                let new_community = community_mapping.get(&old_community).copied().unwrap_or(0);
                (node_id, new_community)
            })
            .collect()
    }
}
