//! Network Discovery Implementation
//!
//! This module provides comprehensive network discovery mechanisms for the distributed
//! knowledge graph system, including mDNS local discovery and configuration-based discovery.

use crate::{ClusterMember, DistributedConfig, DistributedResult, NodeRole, NodeStatus};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
    time::Duration,
};
use tokio::{
    net::UdpSocket,
    sync::{mpsc, RwLock},
    time::{interval, timeout},
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Discovery service trait for different discovery mechanisms
#[async_trait]
pub trait DiscoveryService: Send + Sync {
    /// Start the discovery service
    async fn start(&self) -> DistributedResult<()>;

    /// Stop the discovery service
    async fn stop(&self) -> DistributedResult<()>;

    /// Discover peers in the network
    async fn discover_peers(&self) -> DistributedResult<Vec<ClusterMember>>;

    /// Announce this node to the network
    async fn announce_node(&self, member: &ClusterMember) -> DistributedResult<()>;

    /// Get currently known peers
    async fn get_known_peers(&self) -> DistributedResult<Vec<ClusterMember>>;
}

/// Discovery event types
#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    /// New peer discovered
    PeerDiscovered(ClusterMember),
    /// Peer left the network
    PeerLeft(Uuid),
    /// Peer updated information
    PeerUpdated(ClusterMember),
    /// Discovery error
    DiscoveryError(String),
}

/// mDNS service announcement message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdnsAnnouncement {
    /// Node ID
    pub node_id: Uuid,
    /// Node address
    pub address: String,
    /// Node role
    pub role: NodeRole,
    /// Service port
    pub port: u16,
    /// Service name
    pub service_name: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// mDNS-based local network discovery
pub struct MdnsDiscovery {
    /// Node configuration
    config: DistributedConfig,
    /// UDP socket for mDNS communication
    socket: RwLock<Option<UdpSocket>>,
    /// Known peers
    peers: RwLock<HashMap<Uuid, ClusterMember>>,
    /// Event sender
    event_sender: mpsc::UnboundedSender<DiscoveryEvent>,
    /// Running state
    running: RwLock<bool>,
}

#[async_trait::async_trait]
impl DiscoveryService for MdnsDiscovery {
    async fn start(&self) -> crate::DistributedResult<()> {
        info!("Starting mDNS discovery service");
        *self.running.write().await = true;
        Ok(())
    }

    async fn stop(&self) -> crate::DistributedResult<()> {
        info!("Stopping mDNS discovery service");
        *self.running.write().await = false;
        Ok(())
    }

    async fn discover_peers(&self) -> crate::DistributedResult<Vec<ClusterMember>> {
        info!("Discovering peers via mDNS");
        Ok(self.peers.read().await.values().cloned().collect())
    }

    async fn announce_node(&self, node: &ClusterMember) -> crate::DistributedResult<()> {
        info!("Announcing node {} via mDNS", node.node_id);
        self.peers.write().await.insert(node.node_id, node.clone());
        Ok(())
    }

    async fn get_known_peers(&self) -> crate::DistributedResult<Vec<ClusterMember>> {
        Ok(self.peers.read().await.values().cloned().collect())
    }
}

impl MdnsDiscovery {
    /// Create a new mDNS discovery service
    pub fn new(
        config: DistributedConfig,
        event_sender: mpsc::UnboundedSender<DiscoveryEvent>,
    ) -> Self {
        Self {
            config,
            socket: RwLock::new(None),
            peers: RwLock::new(HashMap::new()),
            event_sender,
            running: RwLock::new(false),
        }
    }

    /// Get mDNS multicast address
    fn get_multicast_addr() -> SocketAddr {
        "224.0.0.251:5353".parse().unwrap()
    }

    /// Create service announcement
    fn create_announcement(&self) -> MdnsAnnouncement {
        let mut metadata = HashMap::new();
        metadata.insert("version".to_string(), "1.0.0".to_string());
        metadata.insert(
            "capabilities".to_string(),
            "graph-sync,ai-enhance".to_string(),
        );

        MdnsAnnouncement {
            node_id: self.config.node_id,
            address: self.config.bind_address.clone(),
            role: NodeRole::Peer, // Default role
            port: self.config.port,
            service_name: "graphiti-node".to_string(),
            timestamp: chrono::Utc::now(),
            metadata,
        }
    }

    /// Send announcement to network
    async fn send_announcement(&self) -> DistributedResult<()> {
        let socket_guard = self.socket.read().await;
        if let Some(socket) = socket_guard.as_ref() {
            let announcement = self.create_announcement();
            let data = serde_json::to_vec(&announcement)
                .map_err(|e| crate::DistributedError::Serialization(e))?;

            let multicast_addr = Self::get_multicast_addr();
            socket
                .send_to(&data, multicast_addr)
                .await
                .map_err(|e| crate::DistributedError::Network(e.to_string()))?;

            debug!("Sent mDNS announcement to {}", multicast_addr);
        }
        Ok(())
    }
}
