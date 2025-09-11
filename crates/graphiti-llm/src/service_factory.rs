//! Service factory for creating LLM and embedding clients

use graphiti_core::error::Result;
use std::sync::Arc;

use crate::EmbedderClient;
use crate::{GemmaCandleClient, GemmaCandleConfig};
use crate::EmbedderConfig;
use crate::EmbeddingClient;
use crate::LLMClient;
use crate::OllamaClient;
use crate::OllamaConfig;
use crate::OpenAIClient;
use crate::OpenAIConfig;
// Qwen-specific paths removed; use Generic embedder only

/// Configuration for the service factory
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// LLM configuration
    pub llm: LLMServiceConfig,
    /// Embedding configuration
    pub embedding: EmbeddingServiceConfig,
}

/// LLM service configuration
#[derive(Debug, Clone)]
pub enum LLMServiceConfig {
    /// OpenAI configuration
    OpenAI(OpenAIConfig),
    /// Ollama configuration
    Ollama(OllamaConfig),
}

/// Embedding service configuration
#[derive(Debug, Clone)]
pub enum EmbeddingServiceConfig {
    /// Generic embedder configuration (embed_anything + Candle)
    Generic(EmbedderConfig),
    /// Native Candle Gemma (approximate via token embeddings)
    GemmaCandle(GemmaCandleConfig),
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            llm: LLMServiceConfig::Ollama(OllamaConfig::default()),
            // 默认与服务器一致，使用原生 Candle 近似（无需权重，仅 tokenizer.json）
            embedding: EmbeddingServiceConfig::GemmaCandle(GemmaCandleConfig::default()),
        }
    }
}

/// Service factory for creating LLM and embedding clients
pub struct ServiceFactory;

impl ServiceFactory {
    /// Create both LLM and embedding clients from configuration
    pub async fn create_services(
        config: &ServiceConfig,
    ) -> Result<(Arc<dyn LLMClient>, Arc<dyn EmbeddingClient>)> {
        let llm_client = Self::create_llm_client(&config.llm).await?;
        let embedding_client = Self::create_embedding_client(&config.embedding).await?;
        Ok((llm_client, embedding_client))
    }

    /// Create LLM client from configuration
    pub async fn create_llm_client(config: &LLMServiceConfig) -> Result<Arc<dyn LLMClient>> {
        match config {
            LLMServiceConfig::OpenAI(openai_config) => {
                let client = OpenAIClient::new(openai_config.clone())?;
                Ok(Arc::new(client))
            }
            LLMServiceConfig::Ollama(ollama_config) => {
                let client = OllamaClient::new(ollama_config.clone())?;
                Ok(Arc::new(client))
            }
        }
    }

    /// Create embedding client from configuration
    pub async fn create_embedding_client(
        config: &EmbeddingServiceConfig,
    ) -> Result<Arc<dyn EmbeddingClient>> {
        match config {
            EmbeddingServiceConfig::Generic(embedder_config) => {
                let client = EmbedderClient::new(embedder_config.clone())?;
                Ok(Arc::new(client))
            }
            EmbeddingServiceConfig::GemmaCandle(cfg) => {
                let client = GemmaCandleClient::new(cfg.clone())?;
                Ok(Arc::new(client))
            }
        }
    }

    /// Create services optimized for local development
    pub async fn create_local_services() -> Result<(Arc<dyn LLMClient>, Arc<dyn EmbeddingClient>)> {
        let config = ServiceConfig {
            llm: LLMServiceConfig::Ollama(OllamaConfig {
                model: "llama3.2:3b".to_string(), // Smaller model for local dev
                ..Default::default()
            }),
            embedding: EmbeddingServiceConfig::Generic(EmbedderConfig::default()),
        };

        Self::create_services(&config).await
    }

    /// Create services with OpenAI LLM and Qwen local embeddings
    // Hybrid creator removed; use Generic embedder consistently

    /// Create services with custom Qwen configuration
    // Qwen-specific service creators removed

    /// Check if services are healthy and ready
    pub async fn health_check(
        llm_client: &Arc<dyn LLMClient>,
        embedding_client: &Arc<dyn EmbeddingClient>,
    ) -> Result<ServiceHealthStatus> {
        // Test LLM with a simple completion
        let llm_healthy = match llm_client
            .complete(
                &[crate::Message::user("Hello")],
                &crate::CompletionParams {
                    max_tokens: Some(10),
                    temperature: Some(0.1),
                    ..Default::default()
                },
            )
            .await
        {
            Ok(_) => true,
            Err(_) => false,
        };

        // Test embedding with a simple text
        let embedding_healthy = match embedding_client.embed("test").await {
            Ok(_) => true,
            Err(_) => false,
        };

        Ok(ServiceHealthStatus {
            llm_healthy,
            embedding_healthy,
            overall_healthy: llm_healthy && embedding_healthy,
        })
    }
}

/// Health status of services
#[derive(Debug, Clone)]
pub struct ServiceHealthStatus {
    /// Whether LLM service is healthy
    pub llm_healthy: bool,
    /// Whether embedding service is healthy
    pub embedding_healthy: bool,
    /// Whether all services are healthy
    pub overall_healthy: bool,
}

impl ServiceHealthStatus {
    /// Get a human-readable status message
    pub fn status_message(&self) -> String {
        match (self.llm_healthy, self.embedding_healthy) {
            (true, true) => "All services are healthy".to_string(),
            (true, false) => "LLM service is healthy, but embedding service is not".to_string(),
            (false, true) => "Embedding service is healthy, but LLM service is not".to_string(),
            (false, false) => "Both LLM and embedding services are unhealthy".to_string(),
        }
    }
}
