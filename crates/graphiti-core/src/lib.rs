//! Graphiti Core - Core types and traits for the Graphiti knowledge graph system
//!
//! This crate provides the fundamental building blocks for the Graphiti system:
//! - Graph data structures (nodes, edges, temporal metadata)
//! - Storage traits for graph database abstraction
//! - Error types and result definitions
//! - Common utilities and helpers

#![warn(missing_docs)]
#![allow(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod advanced_search;
pub mod batch_processing;
/// Code entity definitions and structures
pub mod code_entities;
pub mod code_processor;
pub mod community;
pub mod config;
pub mod episode_processor;
pub mod error;
pub mod error_handling;
pub mod graph;
pub mod graph_traversal;
pub mod graphiti;
pub mod louvain;
pub mod metrics;
pub mod performance_optimizer;
pub mod scheduler;
pub mod storage;

// AI Enhancement modules
pub mod ai_enhancement;
pub mod knowledge_patterns;

#[cfg(test)]
pub mod tests;

pub use error::Error;
pub use error::Result;

/// Re-export commonly used types
pub mod prelude {
    pub use crate::error::Error;
    pub use crate::error::Result;
    pub use crate::graph::CommunityNode;
    pub use crate::graph::Edge;
    pub use crate::graph::EntityNode;
    pub use crate::graph::EpisodeNode;
    pub use crate::graph::EpisodeType;
    pub use crate::graph::Node;
    pub use crate::graph::TemporalMetadata;
    pub use crate::graphiti::Graphiti;
    pub use crate::graphiti::GraphitiConfig;
    pub use crate::graphiti::SearchFilters;
    pub use crate::graphiti::SearchResult;
    pub use crate::storage::Direction;
    pub use crate::storage::GraphStorage;
}
