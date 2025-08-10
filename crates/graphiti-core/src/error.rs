//! Error types for Graphiti

use thiserror::Error;
use uuid::Uuid;

/// Main error type for Graphiti operations
#[derive(Error, Debug)]
pub enum Error {
    /// Storage-related errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// Database connection errors
    #[error("Database connection error: {0}")]
    DatabaseConnection(String),

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Validation errors
    #[error("Validation error: {0}")]
    Validation(String),

    /// Schema validation errors
    #[error("Schema validation failed: {0}")]
    SchemaValidation(String),

    /// Processing errors
    #[error("Processing error: {0}")]
    Processing(String),

    /// Not found errors
    #[error("Not found: {0}")]
    NotFound(String),

    /// Node not found
    #[error("Node not found: {id}")]
    NodeNotFound {
        /// The ID of the missing node
        id: Uuid,
    },

    /// Edge not found
    #[error("Edge not found: {id}")]
    EdgeNotFound {
        /// The ID of the missing edge
        id: Uuid,
    },

    /// LLM provider errors
    #[error("LLM error: {0}")]
    LLMProvider(String),

    /// Embedding provider errors
    #[error("Embedding error: {0}")]
    EmbeddingProvider(String),

    /// Search errors
    #[error("Search error: {0}")]
    Search(String),

    /// Rate limit errors
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    /// Timeout errors
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic errors
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type alias for Graphiti operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a storage error
    pub fn storage(msg: impl Into<String>) -> Self {
        Self::Storage(msg.into())
    }

    /// Create a validation error
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    /// Create a not found error
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    /// Check if this is a not found error
    pub fn is_not_found(&self) -> bool {
        matches!(
            self,
            Self::NotFound(_) | Self::NodeNotFound { .. } | Self::EdgeNotFound { .. }
        )
    }

    /// Check if this is a temporary error that can be retried
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimit(_) | Self::Timeout(_) | Self::DatabaseConnection(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::storage("Failed to connect");
        assert_eq!(err.to_string(), "Storage error: Failed to connect");

        let err = Error::NodeNotFound { id: Uuid::nil() };
        assert_eq!(err.to_string(), format!("Node not found: {}", Uuid::nil()));
    }

    #[test]
    fn test_error_helpers() {
        let err = Error::not_found("Entity missing");
        assert!(err.is_not_found());
        assert!(!err.is_retryable());

        let err = Error::RateLimit("Too many requests".to_string());
        assert!(!err.is_not_found());
        assert!(err.is_retryable());
    }

    #[test]
    fn test_error_from_traits() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Serialization(_)));
    }
}
