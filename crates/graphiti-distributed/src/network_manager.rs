//! Network Manager Implementation
//!
//! This module provides a unified network management layer that combines
//! discovery services and transport protocols for the distributed knowledge graph.

use crate::{
    discovery::{DiscoveryEvent, DiscoveryService, MdnsDiscovery},
    graph_sync::GraphSyncMessage,
    transport::{TcpTransport, TransportMessage, TransportProtocol, TransportService},
    ClusterMember, DistributedConfig, DistributedError, DistributedResult, NodeRole, NodeStatus,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::Duration};
use tokio::{
    sync::{mpsc, RwLock},
    time::interval,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Network manager that coordinates discovery and transport
pub struct NetworkManager {
    /// Configuration
    config: DistributedConfig,
    /// Discovery service
    discovery: Option<Box<dyn DiscoveryService>>,
    /// Transport service
    transport: Option<Box<dyn TransportService>>,
    /// Known peers
    peers: Arc<RwLock<HashMap<Uuid, ClusterMember>>>,
    /// Message handlers
    message_handlers: Arc<RwLock<HashMap<String, Box<dyn MessageHandler>>>>,
    /// Event channels
    discovery_event_sender: mpsc::UnboundedSender<DiscoveryEvent>,
    discovery_event_receiver: Option<mpsc::UnboundedReceiver<DiscoveryEvent>>,
    transport_message_sender: mpsc::UnboundedSender<TransportMessage>,
    transport_message_receiver: Option<mpsc::UnboundedReceiver<TransportMessage>>,
    /// Running state
    running: Arc<RwLock<bool>>,
}

/// Message handler trait for different message types
#[async_trait]
pub trait MessageHandler: Send + Sync {
    /// Handle incoming message
    async fn handle_message(&self, message: TransportMessage) -> DistributedResult<()>;
}

/// Network event types
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// Peer discovered
    PeerDiscovered(ClusterMember),
    /// Peer disconnected
    PeerDisconnected(Uuid),
    /// Message received
    MessageReceived(TransportMessage),
    /// Network error
    NetworkError(String),
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Connected peers count
    pub connected_peers: usize,
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
    /// Discovery events
    pub discovery_events: u64,
    /// Transport errors
    pub transport_errors: u64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

impl NetworkManager {
    /// Create a new network manager
    pub async fn new(config: DistributedConfig) -> DistributedResult<Self> {
        info!("Creating network manager with config: {:?}", config);

        // Create event channels
        let (discovery_event_sender, discovery_event_receiver) = mpsc::unbounded_channel();
        let (transport_message_sender, transport_message_receiver) = mpsc::unbounded_channel();

        Ok(Self {
            config,
            discovery: None,
            transport: None,
            peers: Arc::new(RwLock::new(HashMap::new())),
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
            discovery_event_sender,
            discovery_event_receiver: Some(discovery_event_receiver),
            transport_message_sender,
            transport_message_receiver: Some(transport_message_receiver),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Initialize discovery service
    pub async fn init_discovery(&mut self) -> DistributedResult<()> {
        info!("Initializing mDNS discovery service");

        let discovery =
            MdnsDiscovery::new(self.config.clone(), self.discovery_event_sender.clone());

        self.discovery = Some(Box::new(discovery));
        Ok(())
    }

    /// Initialize transport service
    pub async fn init_transport(&mut self) -> DistributedResult<()> {
        info!("Initializing TCP transport service");

        let transport =
            TcpTransport::new(self.config.clone(), self.transport_message_sender.clone());

        self.transport = Some(Box::new(transport));
        Ok(())
    }

    /// Start the network manager
    pub async fn start(&mut self) -> DistributedResult<()> {
        info!("Starting network manager");

        // Start discovery service
        if let Some(discovery) = &mut self.discovery {
            discovery.start().await?;
        }

        // Start transport service
        if let Some(transport) = &mut self.transport {
            transport.start().await?;
        }

        *self.running.write().await = true;

        // Start event processing
        self.start_event_processing().await?;

        info!("Network manager started successfully");
        Ok(())
    }

    /// Stop the network manager
    pub async fn stop(&mut self) -> DistributedResult<()> {
        info!("Stopping network manager");

        *self.running.write().await = false;

        // Stop services
        if let Some(discovery) = &mut self.discovery {
            discovery.stop().await?;
        }

        if let Some(transport) = &mut self.transport {
            transport.stop().await?;
        }

        info!("Network manager stopped");
        Ok(())
    }

    /// Start event processing loops
    async fn start_event_processing(&mut self) -> DistributedResult<()> {
        // Take receivers to move into spawned tasks
        let discovery_receiver = self.discovery_event_receiver.take().ok_or_else(|| {
            DistributedError::Network("Discovery receiver already taken".to_string())
        })?;

        let transport_receiver = self.transport_message_receiver.take().ok_or_else(|| {
            DistributedError::Network("Transport receiver already taken".to_string())
        })?;

        // Start discovery event processing
        let peers = self.peers.clone();
        let running = self.running.clone();
        tokio::spawn(async move {
            Self::process_discovery_events(discovery_receiver, peers, running).await;
        });

        // Start transport message processing
        let message_handlers = self.message_handlers.clone();
        let running = self.running.clone();
        tokio::spawn(async move {
            Self::process_transport_messages(transport_receiver, message_handlers, running).await;
        });

        // Start periodic tasks
        let peers = self.peers.clone();
        let running = self.running.clone();
        tokio::spawn(async move {
            Self::periodic_maintenance(peers, running).await;
        });

        Ok(())
    }

    /// Process discovery events
    async fn process_discovery_events(
        mut receiver: mpsc::UnboundedReceiver<DiscoveryEvent>,
        peers: Arc<RwLock<HashMap<Uuid, ClusterMember>>>,
        running: Arc<RwLock<bool>>,
    ) {
        while *running.read().await {
            match receiver.recv().await {
                Some(event) => match event {
                    DiscoveryEvent::PeerDiscovered(member) => {
                        info!("Peer discovered: {} at {}", member.node_id, member.address);
                        let mut peers = peers.write().await;
                        peers.insert(member.node_id, member);
                    }
                    DiscoveryEvent::PeerLeft(node_id) => {
                        info!("Peer left: {}", node_id);
                        let mut peers = peers.write().await;
                        peers.remove(&node_id);
                    }
                    DiscoveryEvent::PeerUpdated(member) => {
                        debug!("Peer updated: {}", member.node_id);
                        let mut peers = peers.write().await;
                        peers.insert(member.node_id, member);
                    }
                    DiscoveryEvent::DiscoveryError(error) => {
                        warn!("Discovery error: {}", error);
                    }
                },
                None => {
                    debug!("Discovery event channel closed");
                    break;
                }
            }
        }
    }

    /// Process transport messages
    async fn process_transport_messages(
        mut receiver: mpsc::UnboundedReceiver<TransportMessage>,
        message_handlers: Arc<RwLock<HashMap<String, Box<dyn MessageHandler>>>>,
        running: Arc<RwLock<bool>>,
    ) {
        while *running.read().await {
            match receiver.recv().await {
                Some(message) => {
                    debug!("Received transport message: {}", message.message_type);

                    let handlers = message_handlers.read().await;
                    if let Some(handler) = handlers.get(&message.message_type) {
                        if let Err(e) = handler.handle_message(message).await {
                            error!("Message handler error: {}", e);
                        }
                    } else {
                        warn!("No handler for message type: {}", message.message_type);
                    }
                }
                None => {
                    debug!("Transport message channel closed");
                    break;
                }
            }
        }
    }

    /// Periodic maintenance tasks
    async fn periodic_maintenance(
        peers: Arc<RwLock<HashMap<Uuid, ClusterMember>>>,
        running: Arc<RwLock<bool>>,
    ) {
        let mut maintenance_interval = interval(Duration::from_secs(60));

        while *running.read().await {
            maintenance_interval.tick().await;

            // Clean up stale peers
            let mut peers = peers.write().await;
            let now = chrono::Utc::now();
            let stale_threshold = chrono::Duration::minutes(5);

            peers.retain(|node_id, member| {
                let is_stale = now.signed_duration_since(member.last_heartbeat) > stale_threshold;
                if is_stale {
                    info!("Removing stale peer: {}", node_id);
                }
                !is_stale
            });

            debug!("Maintenance completed. Active peers: {}", peers.len());
        }
    }

    /// Register a message handler
    pub async fn register_handler<H>(
        &self,
        message_type: String,
        handler: H,
    ) -> DistributedResult<()>
    where
        H: MessageHandler + 'static,
    {
        let mut handlers = self.message_handlers.write().await;
        handlers.insert(message_type.clone(), Box::new(handler));
        info!("Registered handler for message type: {}", message_type);
        Ok(())
    }

    /// Send a message to a specific peer
    pub async fn send_to_peer(
        &self,
        peer_id: Uuid,
        message_type: String,
        payload: Vec<u8>,
    ) -> DistributedResult<()> {
        let peers = self.peers.read().await;
        let peer = peers
            .get(&peer_id)
            .ok_or_else(|| DistributedError::Network(format!("Peer not found: {}", peer_id)))?;

        let addr: SocketAddr = peer
            .address
            .parse()
            .map_err(|e| DistributedError::Network(format!("Invalid peer address: {}", e)))?;

        let transport_message = TransportMessage {
            id: Uuid::new_v4(),
            source: self.config.node_id,
            target: Some(peer_id),
            payload,
            message_type,
            timestamp: chrono::Utc::now(),
        };

        if let Some(transport) = &self.transport {
            transport.send_to(addr, transport_message).await?;
        }

        Ok(())
    }

    /// Broadcast a message to all peers
    pub async fn broadcast(&self, message_type: String, payload: Vec<u8>) -> DistributedResult<()> {
        let transport_message = TransportMessage {
            id: Uuid::new_v4(),
            source: self.config.node_id,
            target: None,
            payload,
            message_type,
            timestamp: chrono::Utc::now(),
        };

        if let Some(transport) = &self.transport {
            transport.broadcast(transport_message).await?;
        }

        Ok(())
    }

    /// Get connected peers
    pub async fn get_peers(&self) -> Vec<ClusterMember> {
        let peers = self.peers.read().await;
        peers.values().cloned().collect()
    }

    /// Get network statistics
    pub async fn get_stats(&self) -> NetworkStats {
        let peers = self.peers.read().await;
        let transport_stats = if let Some(transport) = &self.transport {
            transport.get_stats()
        } else {
            crate::transport::TransportStats {
                total_connections: 0,
                active_connections: 0,
                messages_sent: 0,
                messages_received: 0,
                bytes_sent: 0,
                bytes_received: 0,
                connection_errors: 0,
            }
        };

        NetworkStats {
            connected_peers: peers.len(),
            messages_sent: transport_stats.messages_sent,
            messages_received: transport_stats.messages_received,
            discovery_events: 0, // Would track in real implementation
            transport_errors: transport_stats.connection_errors,
            uptime_seconds: 0, // Would track in real implementation
        }
    }
}

/// Graph sync message handler
pub struct GraphSyncMessageHandler {
    /// Node ID
    node_id: Uuid,
}

impl GraphSyncMessageHandler {
    /// Create new graph sync message handler
    pub fn new(node_id: Uuid) -> Self {
        Self { node_id }
    }
}

#[async_trait]
impl MessageHandler for GraphSyncMessageHandler {
    async fn handle_message(&self, message: TransportMessage) -> DistributedResult<()> {
        info!("Handling graph sync message from {}", message.source);

        // Deserialize graph sync message
        let graph_message: GraphSyncMessage = serde_json::from_slice(&message.payload)
            .map_err(|e| DistributedError::Serialization(e))?;

        // Process the graph sync message
        match graph_message {
            GraphSyncMessage::EntitySync { entities, .. } => {
                info!("Received entity sync with {} entities", entities.len());
                // Process entities...
            }
            GraphSyncMessage::ConflictResolution { conflicts, .. } => {
                info!(
                    "Received conflict resolution with {} conflicts",
                    conflicts.len()
                );
                // Process conflicts...
            }
            GraphSyncMessage::EdgeSync { edges, .. } => {
                info!("Received edge sync with {} edges", edges.len());
                // Handle edge sync request...
            }
            GraphSyncMessage::DeduplicationRequest { .. } => {
                info!("Received deduplication request");
                // Handle deduplication request...
            }
            GraphSyncMessage::DeduplicationResponse { .. } => {
                info!("Received deduplication response");
                // Handle deduplication response...
            }
            GraphSyncMessage::GraphStateHash { .. } => {
                info!("Received graph state hash");
                // Handle graph state hash...
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_network_manager_creation() {
        let config = DistributedConfig::default();
        let manager = NetworkManager::new(config).await.unwrap();

        let stats = manager.get_stats().await;
        assert_eq!(stats.connected_peers, 0);
    }

    #[tokio::test]
    async fn test_message_handler_registration() {
        let config = DistributedConfig::default();
        let manager = NetworkManager::new(config).await.unwrap();

        let handler = GraphSyncMessageHandler::new(Uuid::new_v4());
        manager
            .register_handler("graph-sync".to_string(), handler)
            .await
            .unwrap();

        // Handler should be registered
        let handlers = manager.message_handlers.read().await;
        assert!(handlers.contains_key("graph-sync"));
    }
}
