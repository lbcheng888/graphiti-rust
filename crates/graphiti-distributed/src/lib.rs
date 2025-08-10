//! Graphiti Distributed Architecture
//!
//! This crate provides distributed computing capabilities for Graphiti using libp2p,
//! enabling large-scale deployment with automatic discovery, load balancing,
//! and fault tolerance.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod consensus;
pub mod discovery;
pub mod distributed_graphiti;
pub mod graph_sync;
pub mod health;
pub mod libp2p_compat;
pub mod load_balancer;
pub mod network;
pub mod network_manager;
pub mod node;
pub mod replication;
pub mod sharding;
pub mod simple_network;
pub mod transport;

// use graphiti_core::error::{Error, Result}; // Unused for now
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Re-export core types
pub use graphiti_core;

/// Distributed system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Node ID
    pub node_id: Uuid,
    /// Listen address for P2P networking
    pub listen_address: String,
    /// Bind address (for backward compatibility)
    pub bind_address: String,
    /// Port number
    pub port: u16,
    /// Bootstrap nodes for initial connection
    pub bootstrap_nodes: Vec<String>,
    /// Bootstrap peers (alternative naming)
    pub bootstrap_peers: Vec<String>,
    /// Cluster name
    pub cluster_name: String,
    /// Enable automatic discovery
    pub enable_discovery: bool,
    /// Enable gossip protocol
    pub enable_gossip: bool,
    /// Replication factor
    pub replication_factor: usize,
    /// Sharding strategy
    pub sharding_strategy: ShardingStrategy,
    /// Consensus algorithm
    pub consensus_algorithm: ConsensusAlgorithm,
    /// Maximum number of peers
    pub max_peers: usize,
    /// Heartbeat interval in seconds
    pub heartbeat_interval_secs: u64,
    /// Sync interval in seconds
    pub sync_interval_secs: u64,
    /// Data directory
    pub data_dir: String,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            node_id: Uuid::new_v4(),
            listen_address: "/ip4/0.0.0.0/tcp/0".to_string(),
            bind_address: "0.0.0.0".to_string(),
            port: 8001,
            bootstrap_nodes: Vec::new(),
            bootstrap_peers: Vec::new(),
            cluster_name: "graphiti-cluster".to_string(),
            enable_discovery: true,
            enable_gossip: true,
            replication_factor: 3,
            sharding_strategy: ShardingStrategy::ConsistentHashing,
            consensus_algorithm: ConsensusAlgorithm::Raft,
            max_peers: 50,
            heartbeat_interval_secs: 30,
            sync_interval_secs: 300,
            data_dir: "/tmp/graphiti".to_string(),
        }
    }
}

/// Sharding strategies for data distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Consistent hashing for even distribution
    ConsistentHashing,
    /// Range-based sharding
    RangeBased,
    /// Hash-based sharding
    HashBased,
    /// Custom sharding logic
    Custom(String),
}

/// Consensus algorithms for distributed coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    /// Raft consensus algorithm
    Raft,
    /// PBFT (Practical Byzantine Fault Tolerance)
    PBFT,
    /// Gossip-based eventual consistency
    Gossip,
}

/// Node role in the distributed system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeRole {
    /// Leader node (coordinates operations)
    Leader,
    /// Follower node (executes operations)
    Follower,
    /// Observer node (read-only)
    Observer,
    /// Bootstrap node (helps with discovery)
    Bootstrap,
    /// Regular peer node
    Peer,
    /// Worker node (processes tasks)
    Worker,
}

/// Node status in the cluster
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is healthy and active
    Active,
    /// Node is starting up
    Starting,
    /// Node is shutting down
    Stopping,
    /// Node is temporarily unavailable
    Unavailable,
    /// Node has failed
    Failed,
}

/// Cluster member information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMember {
    /// Node ID
    pub node_id: Uuid,
    /// Node address
    pub address: String,
    /// Node role
    pub role: NodeRole,
    /// Node status
    pub status: NodeStatus,
    /// Node capabilities
    pub capabilities: Vec<String>,
    /// Last heartbeat timestamp
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    /// Last seen timestamp
    pub last_seen: chrono::DateTime<chrono::Utc>,
    /// Peer ID for P2P networking
    pub peer_id: String,
    /// Node metadata
    pub metadata: HashMap<String, String>,
}

impl Default for ClusterMember {
    fn default() -> Self {
        Self {
            node_id: Uuid::new_v4(),
            address: String::new(),
            role: NodeRole::Peer,
            status: NodeStatus::Starting,
            capabilities: Vec::new(),
            last_heartbeat: chrono::Utc::now(),
            last_seen: chrono::Utc::now(),
            peer_id: String::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Distributed operation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedRequest {
    /// Request ID
    pub id: Uuid,
    /// Operation type
    pub operation: DistributedOperation,
    /// Target shard (if applicable)
    pub shard_id: Option<String>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// Distributed operation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedResponse {
    /// Request ID
    pub request_id: Uuid,
    /// Response data
    pub data: serde_json::Value,
    /// Success status
    pub success: bool,
    /// Error message (if any)
    pub error: Option<String>,
    /// Response metadata
    pub metadata: HashMap<String, String>,
}

/// Types of distributed operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedOperation {
    /// Add episode to the graph
    AddEpisode {
        /// Episode data
        episode: serde_json::Value,
    },
    /// Search the graph
    Search {
        /// Search query
        query: String,
        /// Search parameters
        params: HashMap<String, serde_json::Value>,
    },
    /// Get node by ID
    GetNode {
        /// Node ID
        node_id: Uuid,
    },
    /// Update node
    UpdateNode {
        /// Node data
        node: serde_json::Value,
    },
    /// Delete node
    DeleteNode {
        /// Node ID
        node_id: Uuid,
    },
    /// Detect communities
    DetectCommunities {
        /// Detection parameters
        params: HashMap<String, serde_json::Value>,
    },
    /// Health check
    HealthCheck,
    /// Cluster status
    ClusterStatus,
}

/// Distributed system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedMetrics {
    /// Total number of nodes in cluster
    pub total_nodes: usize,
    /// Active nodes
    pub active_nodes: usize,
    /// Failed nodes
    pub failed_nodes: usize,
    /// Total operations processed
    pub total_operations: u64,
    /// Operations per second
    pub operations_per_second: f64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Network bandwidth usage
    pub network_bandwidth_mbps: f64,
    /// Data distribution across shards
    pub shard_distribution: HashMap<String, usize>,
}

/// Error types specific to distributed operations
#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    /// Network communication error
    #[error("Network error: {0}")]
    Network(String),

    /// Network error (alternative name for compatibility)
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Internal system error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Consensus error
    #[error("Consensus error: {0}")]
    Consensus(String),

    /// Sharding error
    #[error("Sharding error: {0}")]
    Sharding(String),

    /// Replication error
    #[error("Replication error: {0}")]
    Replication(String),

    /// Node unavailable
    #[error("Node unavailable: {0}")]
    NodeUnavailable(Uuid),

    /// Cluster not ready
    #[error("Cluster not ready")]
    ClusterNotReady,

    /// Operation timeout
    #[error("Operation timeout")]
    Timeout,

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Core Graphiti error
    #[error("Core error: {0}")]
    Core(#[from] graphiti_core::error::Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// libp2p noise error
    #[error("Noise error: {0}")]
    Noise(String),

    /// Transport error
    #[error("Transport error: {0}")]
    Transport(String),
}

/// Result type for distributed operations
pub type DistributedResult<T> = std::result::Result<T, DistributedError>;

impl From<libp2p::noise::Error> for DistributedError {
    fn from(err: libp2p::noise::Error) -> Self {
        DistributedError::Noise(err.to_string())
    }
}

/// Trait for distributed storage backends
#[async_trait::async_trait]
pub trait DistributedStorage: Send + Sync {
    /// Store data in the distributed system
    async fn store(&self, key: String, value: serde_json::Value) -> DistributedResult<()>;

    /// Retrieve data from the distributed system
    async fn retrieve(&self, key: &str) -> DistributedResult<Option<serde_json::Value>>;

    /// Delete data from the distributed system
    async fn delete(&self, key: &str) -> DistributedResult<()>;

    /// List all keys with optional prefix
    async fn list_keys(&self, prefix: Option<&str>) -> DistributedResult<Vec<String>>;

    /// Get storage statistics
    async fn get_stats(&self) -> DistributedResult<HashMap<String, serde_json::Value>>;
}

/// Trait for distributed consensus
#[async_trait::async_trait]
pub trait DistributedConsensus: Send + Sync {
    /// Propose a new operation to the cluster
    async fn propose(
        &self,
        operation: DistributedOperation,
    ) -> DistributedResult<DistributedResponse>;

    /// Get current cluster leader
    async fn get_leader(&self) -> DistributedResult<Option<Uuid>>;

    /// Check if this node is the leader
    async fn is_leader(&self) -> DistributedResult<bool>;

    /// Get cluster membership
    async fn get_members(&self) -> DistributedResult<Vec<ClusterMember>>;

    /// Add a new member to the cluster
    async fn add_member(&self, member: ClusterMember) -> DistributedResult<()>;

    /// Remove a member from the cluster
    async fn remove_member(&self, node_id: Uuid) -> DistributedResult<()>;
}

/// Trait for load balancing
#[async_trait::async_trait]
pub trait LoadBalancer: Send + Sync {
    /// Select the best node for a given operation
    async fn select_node(&self, operation: &DistributedOperation) -> DistributedResult<Uuid>;

    /// Update node health status
    async fn update_node_health(&self, node_id: Uuid, healthy: bool) -> DistributedResult<()>;

    /// Get load balancing statistics
    async fn get_stats(&self) -> DistributedResult<HashMap<Uuid, f64>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedConfig::default();
        assert_eq!(config.replication_factor, 3);
        assert_eq!(config.max_peers, 50);
        assert!(config.enable_discovery);
    }

    #[test]
    fn test_cluster_member_serialization() {
        let member = ClusterMember {
            node_id: Uuid::new_v4(),
            address: "127.0.0.1:8080".to_string(),
            role: NodeRole::Leader,
            status: NodeStatus::Active,
            capabilities: vec!["storage".to_string(), "compute".to_string()],
            last_heartbeat: chrono::Utc::now(),
            last_seen: chrono::Utc::now(),
            peer_id: "peer-127.0.0.1-8080".to_string(),
            metadata: HashMap::new(),
        };

        let serialized = serde_json::to_string(&member).unwrap();
        let deserialized: ClusterMember = serde_json::from_str(&serialized).unwrap();

        assert_eq!(member.node_id, deserialized.node_id);
        assert_eq!(member.role, deserialized.role);
        assert_eq!(member.status, deserialized.status);
    }

    #[test]
    fn test_distributed_operation_serialization() {
        let operation = DistributedOperation::AddEpisode {
            episode: serde_json::json!({"content": "test episode"}),
        };

        let serialized = serde_json::to_string(&operation).unwrap();
        let deserialized: DistributedOperation = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            DistributedOperation::AddEpisode { episode } => {
                assert_eq!(episode["content"], "test episode");
            }
            _ => panic!("Wrong operation type"),
        }
    }
}
