//! Benchmarks for Graphiti

use chrono::Utc;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use graphiti_core::graph::{EntityNode, TemporalMetadata};
use uuid::Uuid;

fn bench_entity_creation(c: &mut Criterion) {
    c.bench_function("entity_node_creation", |b| {
        b.iter(|| {
            let node = EntityNode {
                id: Uuid::new_v4(),
                name: black_box("Test Entity".to_string()),
                entity_type: black_box("Person".to_string()),
                labels: vec!["Person".to_string(), "Entity".to_string()],
                properties: serde_json::json!({"age": 30, "city": "New York"}),
                temporal: TemporalMetadata {
                    created_at: Utc::now(),
                    valid_from: Utc::now(),
                    valid_to: None,
                    expired_at: None,
                },
                embedding: Some(vec![0.1; 1536]),
            };
            node
        })
    });
}

fn bench_json_serialization(c: &mut Criterion) {
    let node = EntityNode {
        id: Uuid::new_v4(),
        name: "Test Entity".to_string(),
        entity_type: "Person".to_string(),
        labels: vec!["Person".to_string(), "Entity".to_string()],
        properties: serde_json::json!({"age": 30, "city": "New York"}),
        temporal: TemporalMetadata {
            created_at: Utc::now(),
            valid_from: Utc::now(),
            valid_to: None,
            expired_at: None,
        },
        embedding: Some(vec![0.1; 1536]),
    };

    c.bench_function("entity_json_serialization", |b| {
        b.iter(|| {
            let json = serde_json::to_string(&node).unwrap();
            black_box(json)
        })
    });
}

criterion_group!(benches, bench_entity_creation, bench_json_serialization);
criterion_main!(benches);
