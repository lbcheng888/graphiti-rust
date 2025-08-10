//! Distributed Knowledge Graph Example
//!
//! This example demonstrates the distributed knowledge graph capabilities,
//! including P2P networking, entity deduplication, and graph synchronization.

use graphiti_core::{
    graph::{EntityNode, TemporalMetadata},
    storage::MemoryStorage,
};
use graphiti_distributed::{
    distributed_graphiti::{ClusterStatus, DistributedGraphiti},
    graph_sync::GraphSyncMessage,
    ConsensusAlgorithm, DistributedConfig, ShardingStrategy,
};
use std::collections::HashMap;
use tokio::time::{sleep, Duration};
use tracing::{info, Level};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("Starting Distributed Knowledge Graph Example");

    // Create multiple distributed nodes
    let node1 = create_distributed_node("Node-1", "/ip4/127.0.0.1/tcp/8001").await?;
    let node2 = create_distributed_node("Node-2", "/ip4/127.0.0.1/tcp/8002").await?;
    let node3 = create_distributed_node("Node-3", "/ip4/127.0.0.1/tcp/8003").await?;

    // Start all nodes
    info!("Starting distributed nodes...");
    node1.start().await?;
    node2.start().await?;
    node3.start().await?;

    // Wait for nodes to discover each other
    sleep(Duration::from_secs(5)).await;

    // Demonstrate distributed entity addition and deduplication
    demonstrate_entity_sync(&node1, &node2, &node3).await?;

    // Demonstrate distributed search
    demonstrate_distributed_search(&node1, &node2).await?;

    // Demonstrate pattern learning across nodes
    demonstrate_pattern_learning(&node1, &node2).await?;

    // Show cluster status
    demonstrate_cluster_status(&node1).await?;

    // Demonstrate conflict resolution
    demonstrate_conflict_resolution(&node1, &node2).await?;

    info!("Distributed Knowledge Graph Example completed successfully!");
    Ok(())
}

/// Create a distributed node with the given configuration
async fn create_distributed_node(
    name: &str,
    listen_addr: &str,
) -> Result<DistributedGraphiti<MemoryStorage>, Box<dyn std::error::Error>> {
    let config = DistributedConfig {
        node_id: Uuid::new_v4(),
        listen_address: listen_addr.to_string(),
        bootstrap_nodes: vec![
            "/ip4/127.0.0.1/tcp/8001".to_string(),
            "/ip4/127.0.0.1/tcp/8002".to_string(),
            "/ip4/127.0.0.1/tcp/8003".to_string(),
        ],
        enable_discovery: true,
        replication_factor: 2,
        sharding_strategy: ShardingStrategy::ConsistentHashing,
        consensus_algorithm: ConsensusAlgorithm::Raft,
        max_peers: 10,
        heartbeat_interval_secs: 30,
    };

    let storage = MemoryStorage::new();
    let node = DistributedGraphiti::new(config, storage).await?;

    info!("Created distributed node: {}", name);
    Ok(node)
}

/// Demonstrate entity synchronization and deduplication
async fn demonstrate_entity_sync(
    node1: &DistributedGraphiti<MemoryStorage>,
    node2: &DistributedGraphiti<MemoryStorage>,
    node3: &DistributedGraphiti<MemoryStorage>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Entity Synchronization and Deduplication ===");

    // Create similar entities on different nodes (potential duplicates)
    let entity1 = EntityNode {
        id: Uuid::new_v4(),
        name: "John Smith".to_string(),
        entity_type: "Person".to_string(),
        properties: serde_json::json!({
            "age": 30,
            "occupation": "Software Engineer",
            "location": "San Francisco"
        }),
        temporal: TemporalMetadata {
            created_at: chrono::Utc::now(),
            valid_from: chrono::Utc::now(),
            valid_to: None,
            expired_at: None,
        },
    };

    let entity2 = EntityNode {
        id: Uuid::new_v4(),
        name: "John Smith".to_string(), // Same name - potential duplicate
        entity_type: "Person".to_string(),
        properties: serde_json::json!({
            "age": 30,
            "occupation": "Developer", // Slightly different
            "location": "San Francisco"
        }),
        temporal: TemporalMetadata {
            created_at: chrono::Utc::now(),
            valid_from: chrono::Utc::now(),
            valid_to: None,
            expired_at: None,
        },
    };

    let entity3 = EntityNode {
        id: Uuid::new_v4(),
        name: "Alice Johnson".to_string(),
        entity_type: "Person".to_string(),
        properties: serde_json::json!({
            "age": 28,
            "occupation": "Data Scientist",
            "location": "New York"
        }),
        temporal: TemporalMetadata {
            created_at: chrono::Utc::now(),
            valid_from: chrono::Utc::now(),
            valid_to: None,
            expired_at: None,
        },
    };

    // Add entities to different nodes
    info!("Adding entity to Node 1: {}", entity1.name);
    let result1 = node1.add_entities(vec![entity1]).await?;
    info!("Node 1 result: {:?}", result1);

    info!("Adding entity to Node 2: {}", entity2.name);
    let result2 = node2.add_entities(vec![entity2]).await?;
    info!("Node 2 result: {:?}", result2);

    info!("Adding entity to Node 3: {}", entity3.name);
    let result3 = node3.add_entities(vec![entity3]).await?;
    info!("Node 3 result: {:?}", result3);

    // Wait for synchronization and deduplication
    sleep(Duration::from_secs(10)).await;
    info!("Entity synchronization and deduplication completed");

    Ok(())
}

/// Demonstrate distributed search capabilities
async fn demonstrate_distributed_search(
    node1: &DistributedGraphiti<MemoryStorage>,
    node2: &DistributedGraphiti<MemoryStorage>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Distributed Search ===");

    // Search for entities across the distributed network
    let search_queries = vec![
        "John Smith",
        "Software Engineer",
        "San Francisco",
        "Data Scientist",
    ];

    for query in search_queries {
        info!("Searching for: '{}'", query);

        let results1 = node1.search_distributed(query).await?;
        info!("Node 1 found {} results", results1.len());

        let results2 = node2.search_distributed(query).await?;
        info!("Node 2 found {} results", results2.len());

        // Compare results to show consistency
        if results1.len() == results2.len() {
            info!("✓ Search results consistent across nodes");
        } else {
            info!("⚠ Search results differ between nodes");
        }
    }

    Ok(())
}

/// Demonstrate pattern learning across distributed nodes
async fn demonstrate_pattern_learning(
    node1: &DistributedGraphiti<MemoryStorage>,
    node2: &DistributedGraphiti<MemoryStorage>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Distributed Pattern Learning ===");

    // Create a pattern context
    let context = graphiti_core::knowledge_patterns::PatternContext {
        entities: vec!["Person".to_string(), "Organization".to_string()],
        relationships: vec!["WORKS_FOR".to_string(), "LOCATED_IN".to_string()],
        stage: "Development".to_string(),
        project_type: "Knowledge Graph".to_string(),
        team_size: Some(5),
        technologies: vec!["Rust".to_string(), "libp2p".to_string()],
    };

    info!("Getting pattern recommendations from Node 1");
    let recommendations1 = node1.get_pattern_recommendations(context.clone()).await?;
    info!(
        "Node 1 found {} pattern recommendations",
        recommendations1.len()
    );

    info!("Getting pattern recommendations from Node 2");
    let recommendations2 = node2.get_pattern_recommendations(context).await?;
    info!(
        "Node 2 found {} pattern recommendations",
        recommendations2.len()
    );

    // Display top recommendations
    for (i, rec) in recommendations1.iter().take(3).enumerate() {
        info!(
            "Recommendation {}: {} (relevance: {:.2})",
            i + 1,
            rec.pattern.name,
            rec.relevance
        );
        info!("  Reasoning: {}", rec.reasoning);
    }

    Ok(())
}

/// Demonstrate cluster status monitoring
async fn demonstrate_cluster_status(
    node: &DistributedGraphiti<MemoryStorage>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Cluster Status Monitoring ===");

    let status = node.get_cluster_status().await?;

    info!("Cluster Status:");
    info!("  Total nodes: {}", status.total_nodes);
    info!("  Active nodes: {}", status.active_nodes);
    info!("  Entities synced: {}", status.total_entities_synced);
    info!("  Edges synced: {}", status.total_edges_synced);
    info!("  Duplicates resolved: {}", status.duplicates_resolved);
    info!("  Conflicts resolved: {}", status.conflicts_resolved);
    info!("  Avg sync latency: {:.2}ms", status.avg_sync_latency_ms);

    if let Some(last_sync) = status.last_full_sync {
        info!("  Last full sync: {:?}", last_sync);
    }

    Ok(())
}

/// Demonstrate conflict resolution
async fn demonstrate_conflict_resolution(
    node1: &DistributedGraphiti<MemoryStorage>,
    node2: &DistributedGraphiti<MemoryStorage>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Conflict Resolution ===");

    // Create conflicting entities with the same ID but different properties
    let entity_id = Uuid::new_v4();

    let entity_v1 = EntityNode {
        id: entity_id,
        name: "Conflicted Entity".to_string(),
        entity_type: "Test".to_string(),
        properties: serde_json::json!({
            "version": 1,
            "data": "original_value"
        }),
        temporal: TemporalMetadata {
            created_at: chrono::Utc::now(),
            valid_from: chrono::Utc::now(),
            valid_to: None,
            expired_at: None,
        },
    };

    let entity_v2 = EntityNode {
        id: entity_id, // Same ID
        name: "Conflicted Entity".to_string(),
        entity_type: "Test".to_string(),
        properties: serde_json::json!({
            "version": 2,
            "data": "updated_value" // Different value
        }),
        temporal: TemporalMetadata {
            created_at: chrono::Utc::now(),
            valid_from: chrono::Utc::now(),
            valid_to: None,
            expired_at: None,
        },
    };

    info!("Creating conflicting entities on different nodes");

    // Add conflicting entities to different nodes
    let _result1 = node1.add_entities(vec![entity_v1]).await?;
    sleep(Duration::from_millis(100)).await; // Small delay
    let _result2 = node2.add_entities(vec![entity_v2]).await?;

    // Wait for conflict resolution
    sleep(Duration::from_secs(5)).await;
    info!("Conflict resolution completed");

    Ok(())
}
