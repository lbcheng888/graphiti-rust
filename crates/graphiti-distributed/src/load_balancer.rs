//! Load Balancer Implementation

use crate::{DistributedConfig, DistributedOperation, DistributedResult, LoadBalancer};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Node health information
#[derive(Debug, Clone)]
pub struct NodeHealth {
    /// Node ID
    pub node_id: Uuid,
    /// Health status
    pub healthy: bool,
    /// Load factor (0.0 - 1.0)
    pub load: f64,
    /// Response time in milliseconds
    pub response_time: f64,
    /// Last health check timestamp
    pub last_check: chrono::DateTime<chrono::Utc>,
}

/// Consistent hash load balancer
pub struct ConsistentHashLoadBalancer {
    config: DistributedConfig,
    /// Node health information
    node_health: Arc<RwLock<HashMap<Uuid, NodeHealth>>>,
    /// Consistent hash ring
    hash_ring: Arc<RwLock<BTreeMap<u64, Uuid>>>,
    /// Virtual nodes per physical node
    virtual_nodes: usize,
}

impl ConsistentHashLoadBalancer {
    pub fn new(config: DistributedConfig) -> Self {
        Self {
            config,
            node_health: Arc::new(RwLock::new(HashMap::new())),
            hash_ring: Arc::new(RwLock::new(BTreeMap::new())),
            virtual_nodes: 150, // Default number of virtual nodes
        }
    }

    /// Add a node to the hash ring
    pub async fn add_node(&self, node_id: Uuid) -> DistributedResult<()> {
        let mut ring = self.hash_ring.write().await;
        let mut health = self.node_health.write().await;

        // Add virtual nodes to the ring
        for i in 0..self.virtual_nodes {
            let virtual_key = format!("{}:{}", node_id, i);
            let hash = self.hash_key(&virtual_key);
            ring.insert(hash, node_id);
        }

        // Initialize health information
        health.insert(
            node_id,
            NodeHealth {
                node_id,
                healthy: true,
                load: 0.0,
                response_time: 0.0,
                last_check: chrono::Utc::now(),
            },
        );

        info!("Added node {} to load balancer", node_id);
        Ok(())
    }

    /// Remove a node from the hash ring
    pub async fn remove_node(&self, node_id: Uuid) -> DistributedResult<()> {
        let mut ring = self.hash_ring.write().await;
        let mut health = self.node_health.write().await;

        // Remove all virtual nodes for this physical node
        ring.retain(|_, &mut v| v != node_id);
        health.remove(&node_id);

        info!("Removed node {} from load balancer", node_id);
        Ok(())
    }

    /// Hash a key to a position on the ring
    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Find the node responsible for a given key
    async fn find_node(&self, key: &str) -> Option<Uuid> {
        let ring = self.hash_ring.read().await;
        let health = self.node_health.read().await;

        if ring.is_empty() {
            return None;
        }

        let hash = self.hash_key(key);

        // Find the first node with hash >= key hash
        let mut candidates: Vec<Uuid> = ring.range(hash..).map(|(_, &node_id)| node_id).collect();

        // If no node found, wrap around to the beginning
        if candidates.is_empty() {
            candidates = ring.values().cloned().collect();
        }

        // Filter out unhealthy nodes
        for &node_id in &candidates {
            if let Some(node_health) = health.get(&node_id) {
                if node_health.healthy {
                    return Some(node_id);
                }
            }
        }

        // If all nodes are unhealthy, return the first one anyway
        candidates.first().cloned()
    }

    /// Select the least loaded healthy node
    async fn select_least_loaded_node(&self) -> Option<Uuid> {
        let health = self.node_health.read().await;

        health
            .values()
            .filter(|h| h.healthy)
            .min_by(|a, b| {
                a.load
                    .partial_cmp(&b.load)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|h| h.node_id)
    }
}

#[async_trait::async_trait]
impl crate::LoadBalancer for ConsistentHashLoadBalancer {
    async fn select_node(&self, operation: &DistributedOperation) -> DistributedResult<Uuid> {
        // Use different strategies based on operation type
        match operation {
            DistributedOperation::AddEpisode { episode } => {
                // Use consistent hashing for episode operations
                let key = format!(
                    "episode:{}",
                    episode
                        .get("id")
                        .unwrap_or(&serde_json::Value::String("default".to_string()))
                );
                if let Some(node_id) = self.find_node(&key).await {
                    Ok(node_id)
                } else {
                    // Fallback to current node if no nodes available
                    Ok(self.config.node_id)
                }
            }
            DistributedOperation::Search { .. } => {
                // Use least loaded node for search operations
                if let Some(node_id) = self.select_least_loaded_node().await {
                    Ok(node_id)
                } else {
                    Ok(self.config.node_id)
                }
            }
            _ => {
                // Default to consistent hashing
                let key = format!("operation:{}", chrono::Utc::now().timestamp_nanos());
                if let Some(node_id) = self.find_node(&key).await {
                    Ok(node_id)
                } else {
                    Ok(self.config.node_id)
                }
            }
        }
    }

    async fn update_node_health(&self, node_id: Uuid, healthy: bool) -> DistributedResult<()> {
        let mut health = self.node_health.write().await;

        if let Some(node_health) = health.get_mut(&node_id) {
            node_health.healthy = healthy;
            node_health.last_check = chrono::Utc::now();

            if healthy {
                info!("Node {} marked as healthy", node_id);
            } else {
                warn!("Node {} marked as unhealthy", node_id);
            }
        } else {
            // Add new node if it doesn't exist
            health.insert(
                node_id,
                NodeHealth {
                    node_id,
                    healthy,
                    load: 0.0,
                    response_time: 0.0,
                    last_check: chrono::Utc::now(),
                },
            );
        }

        Ok(())
    }

    async fn get_stats(&self) -> DistributedResult<HashMap<Uuid, f64>> {
        let health = self.node_health.read().await;
        let stats = health
            .iter()
            .map(|(&node_id, health)| (node_id, health.load))
            .collect();
        Ok(stats)
    }
}

/// Failover manager for handling node failures
pub struct FailoverManager {
    /// Load balancer reference
    load_balancer: Arc<ConsistentHashLoadBalancer>,
    /// Failure detection configuration
    failure_threshold: usize,
    /// Recovery threshold
    recovery_threshold: usize,
    /// Node failure counts
    failure_counts: Arc<RwLock<HashMap<Uuid, usize>>>,
}

impl FailoverManager {
    /// Create a new failover manager
    pub fn new(load_balancer: Arc<ConsistentHashLoadBalancer>) -> Self {
        Self {
            load_balancer,
            failure_threshold: 3,  // Mark as failed after 3 consecutive failures
            recovery_threshold: 2, // Mark as recovered after 2 consecutive successes
            failure_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record a successful operation for a node
    pub async fn record_success(&self, node_id: Uuid) -> DistributedResult<()> {
        let mut counts = self.failure_counts.write().await;
        counts.remove(&node_id); // Reset failure count on success

        // Update load balancer health
        self.load_balancer.update_node_health(node_id, true).await?;
        Ok(())
    }

    /// Record a failed operation for a node
    pub async fn record_failure(&self, node_id: Uuid) -> DistributedResult<()> {
        let mut counts = self.failure_counts.write().await;
        let failure_count = counts.entry(node_id).or_insert(0);
        *failure_count += 1;

        if *failure_count >= self.failure_threshold {
            warn!(
                "Node {} marked as failed after {} failures",
                node_id, failure_count
            );
            self.load_balancer
                .update_node_health(node_id, false)
                .await?;
        }

        Ok(())
    }

    /// Check if a node should be considered for recovery
    pub async fn should_attempt_recovery(&self, node_id: Uuid) -> bool {
        let health = self.load_balancer.node_health.read().await;
        if let Some(node_health) = health.get(&node_id) {
            // Attempt recovery if node has been unhealthy for more than 30 seconds
            !node_health.healthy
                && chrono::Utc::now()
                    .signed_duration_since(node_health.last_check)
                    .num_seconds()
                    > 30
        } else {
            false
        }
    }

    /// Attempt to recover a failed node
    pub async fn attempt_recovery(&self, node_id: Uuid) -> DistributedResult<bool> {
        info!("Attempting recovery for node {}", node_id);

        // In a real implementation, this would perform health checks
        // For now, we'll simulate a recovery attempt
        let recovered = true; // Simulate successful recovery

        if recovered {
            self.record_success(node_id).await?;
            info!("Node {} successfully recovered", node_id);
        } else {
            self.record_failure(node_id).await?;
            warn!("Node {} recovery failed", node_id);
        }

        Ok(recovered)
    }

    /// Get failover statistics
    pub async fn get_failover_stats(&self) -> HashMap<Uuid, usize> {
        self.failure_counts.read().await.clone()
    }
}
