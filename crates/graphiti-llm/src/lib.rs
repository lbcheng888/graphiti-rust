//! LLM client implementations for Graphiti

#![warn(missing_docs)]

use async_trait::async_trait;
use graphiti_core::error::Result;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;

mod openai;
pub use openai::OpenAIClient;
pub use openai::OpenAIConfig;

mod ollama;
pub use ollama::OllamaClient;
pub use ollama::OllamaConfig;
pub use ollama::OllamaOptions;

mod config;
pub use config::GroqConfig;
pub use config::HuggingFaceConfig;
pub use config::LLMConfig;
pub use config::LLMProvider;

mod factory;
pub use factory::create_llm_client;
pub use factory::GroqClient;
pub use factory::HuggingFaceClient;
pub use factory::MultiLLMClient;

mod fallback;
pub use fallback::create_fallback_clients;
pub use fallback::FallbackConfig;
pub use fallback::FallbackEmbeddingClient;
pub use fallback::FallbackLLMClient;

mod embedder;
pub use embedder::EmbedderClient;
pub use embedder::EmbedderConfig;
pub use embedder::EmbeddingClient;
pub use embedder::EmbeddingProvider;

// Qwen-related modules removed from public API; using embed_anything + Candle
pub mod gemma_candle;
pub use gemma_candle::{GemmaCandleClient, GemmaCandleConfig};

// Re-export candle types for external use
pub use candle_core;

pub mod embed_anything_client;
pub use embed_anything_client::EmbedAnythingClient;
pub use embed_anything_client::EmbedAnythingConfig;

mod service_factory;
pub use service_factory::EmbeddingServiceConfig;
pub use service_factory::LLMServiceConfig;
pub use service_factory::ServiceConfig;
pub use service_factory::ServiceFactory;
pub use service_factory::ServiceHealthStatus;

/// Message role in a conversation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message (instructions)
    System,
    /// User message
    User,
    /// Assistant message
    Assistant,
}

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender
    pub role: MessageRole,
    /// Content of the message
    pub content: String,
}

impl Message {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

/// Extracted entity from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity name
    pub name: String,
    /// Entity type (e.g., "Person", "Organization", "Location")
    pub entity_type: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Additional attributes
    pub attributes: HashMap<String, serde_json::Value>,
    /// Text span where entity was found
    pub span: Option<(usize, usize)>,
}

/// Extracted relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    /// Source entity name
    pub source: String,
    /// Target entity name
    pub target: String,
    /// Relationship type
    pub relationship: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Additional attributes
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Extraction results from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// Extracted entities
    pub entities: Vec<ExtractedEntity>,
    /// Extracted relationships
    pub relationships: Vec<ExtractedRelationship>,
    /// Summary of the text
    pub summary: Option<String>,
    /// Key facts extracted
    pub facts: Vec<String>,
}

/// LLM completion parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionParams {
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature for sampling (0.0 to 2.0)
    pub temperature: Option<f32>,
    /// Top-p sampling
    pub top_p: Option<f32>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Whether to stream the response
    pub stream: bool,
}

impl Default for CompletionParams {
    fn default() -> Self {
        Self {
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: Some(1.0),
            stop: None,
            stream: false,
        }
    }
}

/// Trait for LLM clients
#[async_trait]
pub trait LLMClient: Send + Sync {
    /// Complete a conversation
    async fn complete(&self, messages: &[Message], params: &CompletionParams) -> Result<String>;

    /// Complete with structured output (returns JSON value)
    async fn complete_structured(
        &self,
        messages: &[Message],
        params: &CompletionParams,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value>;

    /// Extract entities and relationships from text
    async fn extract(&self, text: &str, context: Option<&str>) -> Result<ExtractionResult>;

    /// Summarize text
    async fn summarize(&self, text: &str, max_length: Option<usize>) -> Result<String>;

    /// Check if two entities refer to the same thing
    async fn deduplicate(
        &self,
        entity1: &ExtractedEntity,
        entity2: &ExtractedEntity,
        context: Option<&str>,
    ) -> Result<bool>;

    /// Generate a question to clarify ambiguity
    async fn clarify(&self, ambiguity: &str, context: &str) -> Result<String>;
}

/// Prompt templates for common tasks
pub mod prompts {
    /// Entity extraction prompt
    pub const ENTITY_EXTRACTION: &str = r#"
Extract all entities and relationships from the following text.

For entities, identify:
- Name
- Type (Person, Organization, Location, Event, Concept, etc.)
- Key attributes

For relationships, identify:
- Source entity
- Target entity  
- Relationship type
- Direction

Also provide:
- A brief summary
- Key facts as bullet points

Text: {text}

Context: {context}
"#;

    /// Deduplication prompt
    pub const DEDUPLICATION: &str = r#"
Determine if these two entities refer to the same real-world entity.

Entity 1:
Name: {name1}
Type: {type1}
Attributes: {attrs1}

Entity 2:
Name: {name2}
Type: {type2}
Attributes: {attrs2}

Context: {context}

Respond with only "true" or "false".
"#;

    /// Summarization prompt
    pub const SUMMARIZATION: &str = r#"
Summarize the following text in {max_length} words or less.
Focus on the key information and main points.

Text: {text}
"#;

    /// Clarification prompt
    pub const CLARIFICATION: &str = r#"
Given the following ambiguous situation, generate a clarifying question that would help resolve the ambiguity.

Ambiguity: {ambiguity}
Context: {context}

Generate a single, clear question:
"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let system = Message::system("You are a helpful assistant");
        assert!(matches!(system.role, MessageRole::System));
        assert_eq!(system.content, "You are a helpful assistant");

        let user = Message::user("Hello");
        assert!(matches!(user.role, MessageRole::User));
        assert_eq!(user.content, "Hello");
    }

    #[test]
    fn test_extraction_result_serialization() {
        let result = ExtractionResult {
            entities: vec![ExtractedEntity {
                name: "Alice".to_string(),
                entity_type: "Person".to_string(),
                confidence: 0.95,
                attributes: HashMap::new(),
                span: Some((0, 5)),
            }],
            relationships: vec![],
            summary: Some("A story about Alice".to_string()),
            facts: vec!["Alice is a person".to_string()],
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ExtractionResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.entities.len(), deserialized.entities.len());
        assert_eq!(result.entities[0].name, deserialized.entities[0].name);
    }
}
