//! Distributed Network Example
//!
//! This example demonstrates how to use the network discovery and transport protocols
//! in the distributed knowledge graph system.

use graphiti_distributed::{
    graph_sync::GraphSyncMessage,
    network_manager::{GraphSyncMessageHandler, NetworkManager},
    ClusterMember, DistributedConfig, DistributedResult, NodeRole, NodeStatus,
};
use std::{collections::HashMap, time::Duration};
use tokio::time::sleep;
use tracing::{info, warn};
use uuid::Uuid;

#[tokio::main]
async fn main() -> DistributedResult<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("🚀 Starting Distributed Network Example");

    // Create multiple nodes to demonstrate networking
    let node1_config = create_node_config("127.0.0.1", 8001);
    let node2_config = create_node_config("127.0.0.1", 8002);
    let node3_config = create_node_config("127.0.0.1", 8003);

    // Start nodes
    let mut node1 = create_and_start_node(node1_config, "Node-1").await?;
    let mut node2 = create_and_start_node(node2_config, "Node-2").await?;
    let mut node3 = create_and_start_node(node3_config, "Node-3").await?;

    // Wait for nodes to discover each other
    info!("⏳ Waiting for peer discovery...");
    sleep(Duration::from_secs(3)).await;

    // Check discovered peers
    check_peer_discovery(&node1, "Node-1").await?;
    check_peer_discovery(&node2, "Node-2").await?;
    check_peer_discovery(&node3, "Node-3").await?;

    // Demonstrate message broadcasting
    info!("📡 Testing message broadcasting...");
    test_message_broadcasting(&node1).await?;

    // Demonstrate peer-to-peer communication
    info!("🔗 Testing peer-to-peer communication...");
    test_p2p_communication(&node1, &node2).await?;

    // Demonstrate graph synchronization
    info!("🔄 Testing graph synchronization...");
    test_graph_synchronization(&node1, &node2, &node3).await?;

    // Show network statistics
    info!("📊 Network Statistics:");
    show_network_stats(&node1, "Node-1").await?;
    show_network_stats(&node2, "Node-2").await?;
    show_network_stats(&node3, "Node-3").await?;

    // Simulate network partition and recovery
    info!("🔌 Testing network partition and recovery...");
    test_network_partition_recovery(&mut node2).await?;

    // Wait a bit more to see the effects
    sleep(Duration::from_secs(2)).await;

    // Final statistics
    info!("📈 Final Network Statistics:");
    show_network_stats(&node1, "Node-1").await?;
    show_network_stats(&node3, "Node-3").await?;

    // Cleanup
    info!("🧹 Shutting down nodes...");
    node1.stop().await?;
    node2.stop().await?;
    node3.stop().await?;

    info!("✅ Distributed Network Example completed successfully!");
    Ok(())
}

/// Create node configuration
fn create_node_config(address: &str, port: u16) -> DistributedConfig {
    DistributedConfig {
        node_id: Uuid::new_v4(),
        bind_address: address.to_string(),
        port,
        cluster_name: "graphiti-test-cluster".to_string(),
        heartbeat_interval_secs: 30,
        sync_interval_secs: 60,
        max_peers: 10,
        bootstrap_peers: vec![],
        enable_discovery: true,
        enable_gossip: true,
        data_dir: format!("/tmp/graphiti-node-{}", port),
    }
}

/// Create and start a network node
async fn create_and_start_node(
    config: DistributedConfig,
    name: &str,
) -> DistributedResult<NetworkManager> {
    info!("🔧 Creating {} with config: {:?}", name, config);

    let mut manager = NetworkManager::new(config.clone()).await?;

    // Initialize discovery and transport
    manager.init_discovery().await?;
    manager.init_transport().await?;

    // Register graph sync message handler
    let handler = GraphSyncMessageHandler::new(config.node_id);
    manager
        .register_handler("graph-sync".to_string(), handler)
        .await?;

    // Start the network manager
    manager.start().await?;

    info!("✅ {} started successfully", name);
    Ok(manager)
}

/// Check peer discovery results
async fn check_peer_discovery(manager: &NetworkManager, name: &str) -> DistributedResult<()> {
    let peers = manager.get_peers().await;
    info!("🔍 {} discovered {} peers:", name, peers.len());

    for peer in &peers {
        info!(
            "  - Peer {}: {} ({})",
            peer.node_id, peer.address, peer.role
        );
    }

    if peers.is_empty() {
        warn!("⚠️  {} has not discovered any peers yet", name);
    }

    Ok(())
}

/// Test message broadcasting
async fn test_message_broadcasting(manager: &NetworkManager) -> DistributedResult<()> {
    info!("📢 Broadcasting test message...");

    let test_message = "Hello from broadcast!".as_bytes().to_vec();
    manager
        .broadcast("test-broadcast".to_string(), test_message)
        .await?;

    info!("✅ Broadcast message sent");
    Ok(())
}

/// Test peer-to-peer communication
async fn test_p2p_communication(
    sender: &NetworkManager,
    receiver: &NetworkManager,
) -> DistributedResult<()> {
    let receiver_peers = receiver.get_peers().await;
    if receiver_peers.is_empty() {
        warn!("⚠️  No peers available for P2P test");
        return Ok(());
    }

    let target_peer = &receiver_peers[0];
    info!("📤 Sending P2P message to peer: {}", target_peer.node_id);

    let test_message = "Hello from P2P!".as_bytes().to_vec();
    sender
        .send_to_peer(target_peer.node_id, "test-p2p".to_string(), test_message)
        .await?;

    info!("✅ P2P message sent");
    Ok(())
}

/// Test graph synchronization
async fn test_graph_synchronization(
    node1: &NetworkManager,
    node2: &NetworkManager,
    node3: &NetworkManager,
) -> DistributedResult<()> {
    info!("🔄 Testing graph synchronization between nodes...");

    // Create a sample graph sync message
    let entities = vec![
        create_sample_entity("entity-1", "Person", "Alice"),
        create_sample_entity("entity-2", "Person", "Bob"),
        create_sample_entity("entity-3", "Company", "TechCorp"),
    ];

    let sync_message = GraphSyncMessage::EntitySync {
        entities,
        source_node: node1.config.node_id,
        timestamp: chrono::Utc::now(),
        sync_id: Uuid::new_v4(),
    };

    // Serialize the sync message
    let payload = serde_json::to_vec(&sync_message)
        .map_err(|e| graphiti_distributed::DistributedError::Serialization(e.to_string()))?;

    // Broadcast to all peers
    node1.broadcast("graph-sync".to_string(), payload).await?;

    info!("✅ Graph sync message broadcasted");

    // Wait for propagation
    sleep(Duration::from_millis(500)).await;

    info!("🔄 Graph synchronization test completed");
    Ok(())
}

/// Create a sample entity for testing
fn create_sample_entity(
    id: &str,
    entity_type: &str,
    name: &str,
) -> graphiti_distributed::graph_sync::EntityCandidate {
    use graphiti_distributed::graph_sync::EntityCandidate;
    use std::collections::HashMap;

    let mut properties = HashMap::new();
    properties.insert(
        "name".to_string(),
        serde_json::Value::String(name.to_string()),
    );
    properties.insert(
        "type".to_string(),
        serde_json::Value::String(entity_type.to_string()),
    );

    EntityCandidate {
        id: id.to_string(),
        name: name.to_string(),
        entity_type: entity_type.to_string(),
        properties,
        confidence: 0.95,
        source_node: Uuid::new_v4(),
        created_at: chrono::Utc::now(),
        metadata: HashMap::new(),
    }
}

/// Show network statistics
async fn show_network_stats(manager: &NetworkManager, name: &str) -> DistributedResult<()> {
    let stats = manager.get_stats().await;

    info!("📊 {} Statistics:", name);
    info!("  - Connected Peers: {}", stats.connected_peers);
    info!("  - Messages Sent: {}", stats.messages_sent);
    info!("  - Messages Received: {}", stats.messages_received);
    info!("  - Discovery Events: {}", stats.discovery_events);
    info!("  - Transport Errors: {}", stats.transport_errors);

    Ok(())
}

/// Test network partition and recovery
async fn test_network_partition_recovery(manager: &mut NetworkManager) -> DistributedResult<()> {
    info!("🔌 Simulating network partition...");

    // Stop the node to simulate partition
    manager.stop().await?;

    info!("⏸️  Node partitioned from network");
    sleep(Duration::from_secs(2)).await;

    // Restart to simulate recovery
    info!("🔄 Simulating network recovery...");
    manager.start().await?;

    info!("✅ Node recovered and rejoined network");
    Ok(())
}

/// Additional helper functions for demonstration

/// Create a cluster member for testing
fn create_test_member(id: Uuid, address: &str, role: NodeRole) -> ClusterMember {
    ClusterMember {
        node_id: id,
        address: address.to_string(),
        role,
        status: NodeStatus::Active,
        last_heartbeat: chrono::Utc::now(),
        capabilities: vec!["graph-sync".to_string(), "ai-enhance".to_string()],
        metadata: HashMap::new(),
    }
}

/// Demonstrate advanced networking features
async fn demonstrate_advanced_features() -> DistributedResult<()> {
    info!("🚀 Demonstrating advanced networking features...");

    // This would include:
    // 1. Load balancing across multiple nodes
    // 2. Automatic failover and recovery
    // 3. Network topology optimization
    // 4. Bandwidth management
    // 5. Security and encryption
    // 6. Monitoring and metrics

    info!("✨ Advanced features demonstration completed");
    Ok(())
}

/// Performance benchmarking
async fn run_performance_benchmark() -> DistributedResult<()> {
    info!("⚡ Running network performance benchmark...");

    // This would include:
    // 1. Message throughput testing
    // 2. Latency measurements
    // 3. Connection establishment time
    // 4. Memory usage analysis
    // 5. CPU utilization monitoring

    info!("📈 Performance benchmark completed");
    Ok(())
}
