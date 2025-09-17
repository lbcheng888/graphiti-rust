//! Fallback mechanism for LLM and embedding providers

use crate::{
    embedder::EmbeddingClient, factory::create_llm_client, CompletionParams, EmbedderClient,
    EmbedderConfig, ExtractedEntity, ExtractionResult, LLMClient, LLMConfig, LLMProvider, Message,
    MultiLLMClient,
};
use async_trait::async_trait;
use graphiti_core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Fallback configuration for LLM providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackConfig {
    /// Primary and fallback LLM configurations in order of preference
    pub llm_providers: Vec<LLMConfig>,
    /// Primary and fallback embedding configurations in order of preference  
    pub embedding_providers: Vec<EmbedderConfig>,
    /// Maximum number of retry attempts per provider
    pub max_retries_per_provider: u32,
    /// Whether to fail fast or try all providers
    pub fail_fast: bool,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            llm_providers: vec![
                // Default free providers in order of preference
                LLMConfig {
                    provider: LLMProvider::Ollama,
                    ..Default::default()
                },
                LLMConfig {
                    provider: LLMProvider::Groq,
                    ..Default::default()
                },
                LLMConfig {
                    provider: LLMProvider::HuggingFace,
                    ..Default::default()
                },
            ],
            embedding_providers: vec![EmbedderConfig::default()],
            max_retries_per_provider: 2,
            fail_fast: false,
        }
    }
}

/// Fallback LLM client that tries multiple providers
pub struct FallbackLLMClient {
    clients: Vec<MultiLLMClient>,
    config: FallbackConfig,
}

impl FallbackLLMClient {
    /// Create a new fallback LLM client
    pub async fn new(mut config: FallbackConfig) -> Result<Self> {
        if config.llm_providers.is_empty() {
            return Err(Error::Configuration(
                "At least one LLM provider must be configured".to_string(),
            ));
        }

        // Update configurations from environment
        for llm_config in config.llm_providers.iter_mut() {
            llm_config.from_env();
        }

        let mut clients = Vec::new();
        let mut successful_providers = Vec::new();

        for llm_config in &config.llm_providers {
            match create_llm_client(llm_config).await {
                Ok(client) => {
                    info!(
                        "Successfully initialized {} LLM provider",
                        llm_config.provider_name()
                    );
                    clients.push(client);
                    successful_providers.push(llm_config.provider_name());
                }
                Err(e) => {
                    warn!(
                        "Failed to initialize {} LLM provider: {}",
                        llm_config.provider_name(),
                        e
                    );
                    if config.fail_fast {
                        return Err(Error::Configuration(format!(
                            "Failed to initialize {} LLM provider in fail-fast mode: {}",
                            llm_config.provider_name(),
                            e
                        )));
                    }
                }
            }
        }

        if clients.is_empty() {
            return Err(Error::Configuration(
                "No LLM providers could be initialized".to_string(),
            ));
        }

        info!(
            "Initialized fallback LLM client with providers: {}",
            successful_providers.join(", ")
        );

        Ok(Self { clients, config })
    }

    /// Try complete operation with fallback
    async fn try_complete_with_fallback(
        &self,
        messages: &[Message],
        params: &CompletionParams,
    ) -> Result<String> {
        let mut last_error = Error::LLMProvider("No providers available".to_string());

        for (index, client) in self.clients.iter().enumerate() {
            for attempt in 0..self.config.max_retries_per_provider {
                match client.complete(messages, params).await {
                    Ok(result) => {
                        if index > 0 || attempt > 0 {
                            info!(
                                "Successfully used fallback provider: {} (attempt {})",
                                client.provider_name(),
                                attempt + 1
                            );
                        }
                        return Ok(result);
                    }
                    Err(e) => {
                        warn!(
                            "Provider {} failed (attempt {}): {}",
                            client.provider_name(),
                            attempt + 1,
                            e
                        );
                        last_error = e;

                        // Small delay before retry
                        if attempt < self.config.max_retries_per_provider - 1 {
                            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                        }
                    }
                }
            }
        }

        Err(Error::LLMProvider(format!(
            "All LLM providers failed. Last error: {}",
            last_error
        )))
    }

    /// Try structured completion with fallback
    async fn try_complete_structured_with_fallback(
        &self,
        messages: &[Message],
        params: &CompletionParams,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let mut last_error = Error::LLMProvider("No providers available".to_string());

        for (index, client) in self.clients.iter().enumerate() {
            for attempt in 0..self.config.max_retries_per_provider {
                match client.complete_structured(messages, params, schema).await {
                    Ok(result) => {
                        if index > 0 || attempt > 0 {
                            info!(
                                "Successfully used fallback provider: {} (attempt {})",
                                client.provider_name(),
                                attempt + 1
                            );
                        }
                        return Ok(result);
                    }
                    Err(e) => {
                        warn!(
                            "Provider {} failed (attempt {}): {}",
                            client.provider_name(),
                            attempt + 1,
                            e
                        );
                        last_error = e;

                        // Small delay before retry
                        if attempt < self.config.max_retries_per_provider - 1 {
                            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                        }
                    }
                }
            }
        }

        Err(Error::LLMProvider(format!(
            "All LLM providers failed. Last error: {}",
            last_error
        )))
    }
}

#[async_trait]
impl LLMClient for FallbackLLMClient {
    async fn complete(&self, messages: &[Message], params: &CompletionParams) -> Result<String> {
        self.try_complete_with_fallback(messages, params).await
    }

    async fn complete_structured(
        &self,
        messages: &[Message],
        params: &CompletionParams,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.try_complete_structured_with_fallback(messages, params, schema)
            .await
    }

    async fn extract(&self, text: &str, context: Option<&str>) -> Result<ExtractionResult> {
        // Try each client for extraction
        let mut last_error = Error::LLMProvider("No providers available".to_string());

        for (index, client) in self.clients.iter().enumerate() {
            for attempt in 0..self.config.max_retries_per_provider {
                match client.extract(text, context).await {
                    Ok(result) => {
                        if index > 0 || attempt > 0 {
                            info!("Successfully used fallback provider for extraction: {} (attempt {})", 
                                  client.provider_name(), attempt + 1);
                        }
                        return Ok(result);
                    }
                    Err(e) => {
                        warn!(
                            "Provider {} failed extraction (attempt {}): {}",
                            client.provider_name(),
                            attempt + 1,
                            e
                        );
                        last_error = e;

                        if attempt < self.config.max_retries_per_provider - 1 {
                            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                        }
                    }
                }
            }
        }

        Err(Error::LLMProvider(format!(
            "All LLM providers failed extraction. Last error: {}",
            last_error
        )))
    }

    async fn summarize(&self, text: &str, max_length: Option<usize>) -> Result<String> {
        // Try each client for summarization
        let mut last_error = Error::LLMProvider("No providers available".to_string());

        for (index, client) in self.clients.iter().enumerate() {
            for attempt in 0..self.config.max_retries_per_provider {
                match client.summarize(text, max_length).await {
                    Ok(result) => {
                        if index > 0 || attempt > 0 {
                            info!("Successfully used fallback provider for summarization: {} (attempt {})", 
                                  client.provider_name(), attempt + 1);
                        }
                        return Ok(result);
                    }
                    Err(e) => {
                        warn!(
                            "Provider {} failed summarization (attempt {}): {}",
                            client.provider_name(),
                            attempt + 1,
                            e
                        );
                        last_error = e;

                        if attempt < self.config.max_retries_per_provider - 1 {
                            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                        }
                    }
                }
            }
        }

        Err(Error::LLMProvider(format!(
            "All LLM providers failed summarization. Last error: {}",
            last_error
        )))
    }

    async fn deduplicate(
        &self,
        entity1: &ExtractedEntity,
        entity2: &ExtractedEntity,
        context: Option<&str>,
    ) -> Result<bool> {
        // Try each client for deduplication
        let mut last_error = Error::LLMProvider("No providers available".to_string());

        for (index, client) in self.clients.iter().enumerate() {
            for attempt in 0..self.config.max_retries_per_provider {
                match client.deduplicate(entity1, entity2, context).await {
                    Ok(result) => {
                        if index > 0 || attempt > 0 {
                            info!("Successfully used fallback provider for deduplication: {} (attempt {})", 
                                  client.provider_name(), attempt + 1);
                        }
                        return Ok(result);
                    }
                    Err(e) => {
                        warn!(
                            "Provider {} failed deduplication (attempt {}): {}",
                            client.provider_name(),
                            attempt + 1,
                            e
                        );
                        last_error = e;

                        if attempt < self.config.max_retries_per_provider - 1 {
                            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                        }
                    }
                }
            }
        }

        Err(Error::LLMProvider(format!(
            "All LLM providers failed deduplication. Last error: {}",
            last_error
        )))
    }

    async fn clarify(&self, ambiguity: &str, context: &str) -> Result<String> {
        // Try each client for clarification
        let mut last_error = Error::LLMProvider("No providers available".to_string());

        for (index, client) in self.clients.iter().enumerate() {
            for attempt in 0..self.config.max_retries_per_provider {
                match client.clarify(ambiguity, context).await {
                    Ok(result) => {
                        if index > 0 || attempt > 0 {
                            info!("Successfully used fallback provider for clarification: {} (attempt {})", 
                                  client.provider_name(), attempt + 1);
                        }
                        return Ok(result);
                    }
                    Err(e) => {
                        warn!(
                            "Provider {} failed clarification (attempt {}): {}",
                            client.provider_name(),
                            attempt + 1,
                            e
                        );
                        last_error = e;

                        if attempt < self.config.max_retries_per_provider - 1 {
                            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                        }
                    }
                }
            }
        }

        Err(Error::LLMProvider(format!(
            "All LLM providers failed clarification. Last error: {}",
            last_error
        )))
    }
}

/// Fallback embedding client that tries multiple providers
pub struct FallbackEmbeddingClient {
    clients: Vec<EmbedderClient>,
    config: FallbackConfig,
}

impl FallbackEmbeddingClient {
    /// Create a new fallback embedding client
    pub async fn new(config: FallbackConfig) -> Result<Self> {
        if config.embedding_providers.is_empty() {
            return Err(Error::Configuration(
                "At least one embedding provider must be configured".to_string(),
            ));
        }

        // No API keys needed for embed_anything

        let mut clients = Vec::new();
        let mut successful_providers = Vec::new();

        for embedding_config in &config.embedding_providers {
            match EmbedderClient::new(embedding_config.clone()) {
                Ok(client) => {
                    info!(
                        "Successfully initialized {:?} embedding provider",
                        embedding_config.provider
                    );
                    clients.push(client);
                    successful_providers.push(format!("{:?}", embedding_config.provider));
                }
                Err(e) => {
                    warn!(
                        "Failed to initialize {:?} embedding provider: {}",
                        embedding_config.provider, e
                    );
                    if config.fail_fast {
                        return Err(Error::Configuration(format!(
                            "Failed to initialize {:?} embedding provider in fail-fast mode: {}",
                            embedding_config.provider, e
                        )));
                    }
                }
            }
        }

        if clients.is_empty() {
            return Err(Error::Configuration(
                "No embedding providers could be initialized".to_string(),
            ));
        }

        info!(
            "Initialized fallback embedding client with providers: {}",
            successful_providers.join(", ")
        );

        Ok(Self { clients, config })
    }
}

#[async_trait]
impl EmbeddingClient for FallbackEmbeddingClient {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut last_error = Error::EmbeddingProvider("No providers available".to_string());

        for (index, client) in self.clients.iter().enumerate() {
            for attempt in 0..self.config.max_retries_per_provider {
                match client.embed_batch(texts).await {
                    Ok(result) => {
                        if index > 0 || attempt > 0 {
                            info!(
                                "Successfully used fallback embedding provider (attempt {})",
                                attempt + 1
                            );
                        }
                        return Ok(result);
                    }
                    Err(e) => {
                        warn!(
                            "Embedding provider failed batch embedding (attempt {}): {}",
                            attempt + 1,
                            e
                        );
                        last_error = e;

                        // Small delay before retry
                        if attempt < self.config.max_retries_per_provider - 1 {
                            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                        }
                    }
                }
            }
        }

        Err(Error::EmbeddingProvider(format!(
            "All embedding providers failed batch embedding. Last error: {}",
            last_error
        )))
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut last_error = Error::EmbeddingProvider("No providers available".to_string());

        for (index, client) in self.clients.iter().enumerate() {
            for attempt in 0..self.config.max_retries_per_provider {
                match client.embed(text).await {
                    Ok(result) => {
                        if index > 0 || attempt > 0 {
                            info!(
                                "Successfully used fallback embedding provider (attempt {})",
                                attempt + 1
                            );
                        }
                        return Ok(result);
                    }
                    Err(e) => {
                        warn!(
                            "Embedding provider failed single embedding (attempt {}): {}",
                            attempt + 1,
                            e
                        );
                        last_error = e;

                        // Small delay before retry
                        if attempt < self.config.max_retries_per_provider - 1 {
                            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                        }
                    }
                }
            }
        }

        Err(Error::EmbeddingProvider(format!(
            "All embedding providers failed single embedding. Last error: {}",
            last_error
        )))
    }
}

/// Create fallback clients from configuration
pub async fn create_fallback_clients(
    config: FallbackConfig,
) -> Result<(FallbackLLMClient, FallbackEmbeddingClient)> {
    let llm_client = FallbackLLMClient::new(config.clone()).await?;
    let embedding_client = FallbackEmbeddingClient::new(config).await?;

    Ok((llm_client, embedding_client))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_config_default() {
        let config = FallbackConfig::default();
        assert!(!config.llm_providers.is_empty());
        assert!(!config.embedding_providers.is_empty());
        assert_eq!(config.max_retries_per_provider, 2);
        assert!(!config.fail_fast);
    }

    #[test]
    fn test_fallback_config_serialization() {
        let config = FallbackConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: FallbackConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.llm_providers.len(), deserialized.llm_providers.len());
        assert_eq!(
            config.max_retries_per_provider,
            deserialized.max_retries_per_provider
        );
    }
}
