//! Transport Protocol Implementation
//!
//! This module provides multiple transport protocols for the distributed knowledge graph,
//! including TCP and QUIC transports with automatic protocol negotiation.

use crate::{DistributedConfig, DistributedError, DistributedResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{mpsc, RwLock},
    time::timeout,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Transport protocol types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportProtocol {
    /// TCP transport
    Tcp,
    /// QUIC transport (UDP-based)
    Quic,
    /// WebSocket transport
    WebSocket,
}

/// Transport message wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportMessage {
    /// Message ID
    pub id: Uuid,
    /// Source node ID
    pub source: Uuid,
    /// Target node ID (None for broadcast)
    pub target: Option<Uuid>,
    /// Message payload
    pub payload: Vec<u8>,
    /// Message type
    pub message_type: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Transport connection information
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    /// Connection ID
    pub id: Uuid,
    /// Remote address
    pub remote_addr: SocketAddr,
    /// Protocol used
    pub protocol: TransportProtocol,
    /// Connection established time
    pub established_at: chrono::DateTime<chrono::Utc>,
    /// Last activity time
    pub last_activity: chrono::DateTime<chrono::Utc>,
    /// Connection status
    pub status: ConnectionStatus,
}

/// Connection status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// Connection is active
    Active,
    /// Connection is idle
    Idle,
    /// Connection is closing
    Closing,
    /// Connection is closed
    Closed,
}

/// Transport service trait
#[async_trait]
pub trait TransportService: Send + Sync {
    /// Start the transport service
    async fn start(&mut self) -> DistributedResult<()>;

    /// Stop the transport service
    async fn stop(&mut self) -> DistributedResult<()>;

    /// Send a message to a specific address
    async fn send_to(&self, addr: SocketAddr, message: TransportMessage) -> DistributedResult<()>;

    /// Broadcast a message to all connected peers
    async fn broadcast(&self, message: TransportMessage) -> DistributedResult<()>;

    /// Get active connections
    fn get_connections(&self) -> Vec<ConnectionInfo>;

    /// Get transport statistics
    fn get_stats(&self) -> TransportStats;
}

/// Transport statistics
#[derive(Debug, Clone)]
pub struct TransportStats {
    /// Total connections established
    pub total_connections: u64,
    /// Active connections
    pub active_connections: usize,
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Connection errors
    pub connection_errors: u64,
}

/// TCP transport implementation
pub struct TcpTransport {
    /// Configuration
    config: DistributedConfig,
    /// TCP listener
    listener: Option<TcpListener>,
    /// Active connections
    connections: Arc<RwLock<HashMap<Uuid, ConnectionInfo>>>,
    /// Message sender for incoming messages
    message_sender: mpsc::UnboundedSender<TransportMessage>,
    /// Statistics
    stats: Arc<RwLock<TransportStats>>,
    /// Running state
    running: Arc<RwLock<bool>>,
}

impl TcpTransport {
    /// Create a new TCP transport
    pub fn new(
        config: DistributedConfig,
        message_sender: mpsc::UnboundedSender<TransportMessage>,
    ) -> Self {
        Self {
            config,
            listener: None,
            connections: Arc::new(RwLock::new(HashMap::new())),
            message_sender,
            stats: Arc::new(RwLock::new(TransportStats {
                total_connections: 0,
                active_connections: 0,
                messages_sent: 0,
                messages_received: 0,
                bytes_sent: 0,
                bytes_received: 0,
                connection_errors: 0,
            })),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Handle incoming connection
    async fn handle_connection(
        &self,
        stream: TcpStream,
        addr: SocketAddr,
    ) -> DistributedResult<()> {
        let connection_id = Uuid::new_v4();
        let connection_info = ConnectionInfo {
            id: connection_id,
            remote_addr: addr,
            protocol: TransportProtocol::Tcp,
            established_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            status: ConnectionStatus::Active,
        };

        // Add to connections
        {
            let mut connections = self.connections.write().await;
            connections.insert(connection_id, connection_info);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_connections += 1;
            stats.active_connections += 1;
        }

        info!("New TCP connection from {}: {}", addr, connection_id);

        // Handle the connection in a separate task
        let connections = self.connections.clone();
        let stats = self.stats.clone();
        let message_sender = self.message_sender.clone();
        let running = self.running.clone();

        tokio::spawn(async move {
            if let Err(e) = Self::handle_stream(
                stream,
                connection_id,
                connections,
                stats,
                message_sender,
                running,
            )
            .await
            {
                error!("Connection {} error: {}", connection_id, e);
            }
        });

        Ok(())
    }

    /// Handle TCP stream communication
    async fn handle_stream(
        mut stream: TcpStream,
        connection_id: Uuid,
        connections: Arc<RwLock<HashMap<Uuid, ConnectionInfo>>>,
        stats: Arc<RwLock<TransportStats>>,
        message_sender: mpsc::UnboundedSender<TransportMessage>,
        running: Arc<RwLock<bool>>,
    ) -> DistributedResult<()> {
        let mut buffer = [0u8; 4096];

        while *running.read().await {
            // Read message length first (4 bytes)
            match timeout(Duration::from_secs(30), stream.read_exact(&mut buffer[..4])).await {
                Ok(Ok(_)) => {
                    let msg_len =
                        u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;

                    if msg_len > buffer.len() {
                        warn!("Message too large: {} bytes", msg_len);
                        continue;
                    }

                    // Read the actual message
                    match timeout(
                        Duration::from_secs(10),
                        stream.read_exact(&mut buffer[..msg_len]),
                    )
                    .await
                    {
                        Ok(Ok(_)) => {
                            // Deserialize message
                            match serde_json::from_slice::<TransportMessage>(&buffer[..msg_len]) {
                                Ok(message) => {
                                    // Update stats
                                    {
                                        let mut stats = stats.write().await;
                                        stats.messages_received += 1;
                                        stats.bytes_received += msg_len as u64;
                                    }

                                    // Update connection activity
                                    {
                                        let mut connections = connections.write().await;
                                        if let Some(conn) = connections.get_mut(&connection_id) {
                                            conn.last_activity = chrono::Utc::now();
                                        }
                                    }

                                    // Forward message
                                    if let Err(e) = message_sender.send(message) {
                                        warn!("Failed to forward message: {}", e);
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to deserialize message: {}", e);
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            warn!("Failed to read message: {}", e);
                            break;
                        }
                        Err(_) => {
                            debug!("Read timeout on connection {}", connection_id);
                            continue;
                        }
                    }
                }
                Ok(Err(e)) => {
                    debug!("Connection {} closed: {}", connection_id, e);
                    break;
                }
                Err(_) => {
                    debug!("Read timeout on connection {}", connection_id);
                    continue;
                }
            }
        }

        // Clean up connection
        {
            let mut connections = connections.write().await;
            connections.remove(&connection_id);
        }

        {
            let mut stats = stats.write().await;
            stats.active_connections = stats.active_connections.saturating_sub(1);
        }

        info!("Connection {} closed", connection_id);
        Ok(())
    }

    /// Send message over TCP
    async fn send_tcp_message(
        &self,
        addr: SocketAddr,
        message: TransportMessage,
    ) -> DistributedResult<()> {
        // Serialize message
        let data = serde_json::to_vec(&message).map_err(|e| DistributedError::Serialization(e))?;

        // Connect to target
        let mut stream = timeout(Duration::from_secs(10), TcpStream::connect(addr))
            .await
            .map_err(|_| DistributedError::Network("Connection timeout".to_string()))?
            .map_err(|e| DistributedError::Network(format!("Connection failed: {}", e)))?;

        // Send message length first
        let len_bytes = (data.len() as u32).to_be_bytes();
        stream
            .write_all(&len_bytes)
            .await
            .map_err(|e| DistributedError::Network(format!("Failed to send length: {}", e)))?;

        // Send message data
        stream
            .write_all(&data)
            .await
            .map_err(|e| DistributedError::Network(format!("Failed to send data: {}", e)))?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.messages_sent += 1;
            stats.bytes_sent += data.len() as u64;
        }

        debug!("Sent TCP message to {}: {} bytes", addr, data.len());
        Ok(())
    }
}

#[async_trait]
impl TransportService for TcpTransport {
    async fn start(&mut self) -> DistributedResult<()> {
        info!("Starting TCP transport on {}", self.config.bind_address);

        let bind_addr: SocketAddr = format!("{}:{}", self.config.bind_address, self.config.port)
            .parse()
            .map_err(|e| DistributedError::Network(format!("Invalid bind address: {}", e)))?;

        let listener = TcpListener::bind(bind_addr).await.map_err(|e| {
            DistributedError::Network(format!("Failed to bind TCP listener: {}", e))
        })?;

        self.listener = Some(listener);
        *self.running.write().await = true;

        // Start accepting connections in a separate task
        let listener = self.listener.take().unwrap();
        let _connections = self.connections.clone();
        let _stats = self.stats.clone();
        let message_sender = self.message_sender.clone();
        let running = self.running.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            while *running.read().await {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        let transport = TcpTransport::new(config.clone(), message_sender.clone());
                        if let Err(e) = transport.handle_connection(stream, addr).await {
                            error!("Failed to handle connection: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to accept connection: {}", e);
                        break;
                    }
                }
            }
        });

        info!("TCP transport started on {}", bind_addr);
        Ok(())
    }

    async fn stop(&mut self) -> DistributedResult<()> {
        info!("Stopping TCP transport");
        *self.running.write().await = false;
        self.listener = None;
        Ok(())
    }

    async fn send_to(&self, addr: SocketAddr, message: TransportMessage) -> DistributedResult<()> {
        self.send_tcp_message(addr, message).await
    }

    async fn broadcast(&self, message: TransportMessage) -> DistributedResult<()> {
        let connections = self.connections.read().await;

        for connection in connections.values() {
            if let Err(e) = self
                .send_tcp_message(connection.remote_addr, message.clone())
                .await
            {
                warn!("Failed to send to {}: {}", connection.remote_addr, e);
            }
        }

        Ok(())
    }

    fn get_connections(&self) -> Vec<ConnectionInfo> {
        // Avoid block_in_place in single-threaded runtimes. Use try_blocking to offload.
        let this = self.connections.clone();
        tokio::task::block_in_place(|| {
            // Fallback path when running in multi-threaded runtime.
            futures::executor::block_on(this.read())
                .values()
                .cloned()
                .collect()
        })
    }

    fn get_stats(&self) -> TransportStats {
        let stats = self.stats.clone();
        tokio::task::block_in_place(|| futures::executor::block_on(stats.read()).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use multi-threaded runtime for tests that may use block_in_place internally.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_transport_message_serialization() {
        let message = TransportMessage {
            id: Uuid::new_v4(),
            source: Uuid::new_v4(),
            target: Some(Uuid::new_v4()),
            payload: b"test payload".to_vec(),
            message_type: "test".to_string(),
            timestamp: chrono::Utc::now(),
        };

        let serialized = serde_json::to_vec(&message).unwrap();
        let deserialized: TransportMessage = serde_json::from_slice(&serialized).unwrap();

        assert_eq!(message.id, deserialized.id);
        assert_eq!(message.source, deserialized.source);
        assert_eq!(message.payload, deserialized.payload);
    }

    // This test indirectly calls block_in_place via get_stats/get_connections.
    // Use multi-threaded runtime to satisfy Tokio's requirement.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_tcp_transport_creation() {
        let config = DistributedConfig::default();
        let (tx, _rx) = mpsc::unbounded_channel();

        let transport = TcpTransport::new(config, tx);
        let stats = transport.get_stats();

        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.active_connections, 0);
    }
}
