//! Embedding client implementations

use async_trait::async_trait;
use graphiti_core::error::Error;
use graphiti_core::error::Result;
use moka::future::Cache;
use reqwest::Client;
use serde::Deserialize;
use serde::Serialize;
use std::time::Duration;
use tracing::debug;
use tracing::instrument;

/// Serde helpers for Duration
mod duration_serde {
    use serde::Deserialize;
    use serde::Deserializer;
    use serde::Serialize;
    use serde::Serializer;
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

/// Embedding provider configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbedderConfig {
    /// Provider type
    pub provider: EmbeddingProvider,
    /// API key
    pub api_key: String,
    /// Model name
    pub model: String,
    /// Embedding dimension
    pub dimension: usize,
    /// Batch size for embedding requests
    pub batch_size: usize,
    /// Request timeout
    #[serde(with = "duration_serde")]
    pub timeout: Duration,
    /// Preferred device (cpu/cuda/metal/auto), optional
    #[serde(default)]
    pub device: Option<String>,
    /// Max sequence length, optional
    #[serde(default)]
    pub max_length: Option<usize>,
    /// Optional local cache directory for HF models
    #[serde(default)]
    pub cache_dir: Option<String>,
}

/// Embedding provider
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
    /// OpenAI embeddings
    OpenAI,
    /// Voyage AI embeddings
    Voyage,
    /// Cohere embeddings
    Cohere,
    /// Local SentenceTransformers model
    Local,
    /// Ollama embeddings
    Ollama,
    /// Hugging Face Inference API
    HuggingFace,
    /// Qwen model using Candle (pure Rust)
    QwenCandle,
    /// Qwen3-Embedding using embed_anything crate
    #[serde(alias = "qwen3_embed_anything")]
    Qwen3EmbedAnything,
    /// Generic embed_anything backend (HF model via Candle)
    #[serde(alias = "embed_anything")]
    EmbedAnything,
    /// Native Candle Gemma (approximate, tokenizer-only)
    #[serde(alias = "gemma_candle")]
    GemmaCandleApprox,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            // 默认使用原生 Candle 近似（无需权重，仅需 tokenizer.json）
            provider: EmbeddingProvider::GemmaCandleApprox,
            api_key: String::new(),
            model: "google/embeddinggemma-300m".to_string(),
            dimension: 768,
            batch_size: 32,
            timeout: Duration::from_secs(60),
            device: Some("auto".to_string()),
            max_length: Some(8192),
            cache_dir: None,
        }
    }
}

/// Trait for embedding clients
#[async_trait]
#[allow(dead_code)]
pub trait EmbeddingClient: Send + Sync {
    /// Generate embeddings for a batch of texts
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Generate embedding for a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| Error::EmbeddingProvider("No embedding returned".to_string()))
    }
}

/// Generic embedder client that supports multiple providers
pub struct EmbedderClient {
    config: EmbedderConfig,
    client: Client,
    cache: Cache<String, Vec<f32>>,
}

impl EmbedderClient {
    /// Create a new embedder client
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        // 仅支持 EmbedAnything，其它方案将在上层屏蔽

        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| Error::Configuration(format!("Failed to create HTTP client: {}", e)))?;

        // Create cache with 4 hour TTL and 10000 max entries
        let cache = Cache::builder()
            .time_to_live(Duration::from_secs(14400))
            .max_capacity(10000)
            .build();

        Ok(Self {
            config,
            client,
            cache,
        })
    }

    /// Get the base URL for the provider
    fn base_url(&self) -> &str {
        ""
    }
}

#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbedding>,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbedding {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct VoyageEmbeddingRequest {
    model: String,
    input: Vec<String>,
    input_type: String,
}

#[derive(Debug, Deserialize)]
struct VoyageEmbeddingResponse {
    data: Vec<VoyageEmbedding>,
}

#[derive(Debug, Deserialize)]
struct VoyageEmbedding {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct CohereEmbeddingRequest {
    model: String,
    texts: Vec<String>,
    input_type: String,
}

#[derive(Debug, Deserialize)]
struct CohereEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl EmbeddingClient for EmbedderClient {
    #[instrument(skip(self, texts))]
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Check cache for all texts
        let mut results = vec![None; texts.len()];
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            if let Some(embedding) = self.cache.get(text).await {
                debug!("Cache hit for embedding");
                results[i] = Some(embedding);
            } else {
                uncached_indices.push(i);
                uncached_texts.push(text.clone());
            }
        }

        // If all are cached, return early
        if uncached_texts.is_empty() {
            return Ok(results.into_iter().map(|r| r.unwrap()).collect());
        }

        // Process uncached texts in batches
        let mut uncached_embeddings = Vec::new();

        for chunk in uncached_texts.chunks(self.config.batch_size) {
            let embeddings = match self.config.provider {
                EmbeddingProvider::OpenAI => self.embed_openai(chunk).await?,
                EmbeddingProvider::Voyage => self.embed_voyage(chunk).await?,
                EmbeddingProvider::Cohere => self.embed_cohere(chunk).await?,
                EmbeddingProvider::Local => self.embed_local(chunk).await?,
                EmbeddingProvider::Ollama => self.embed_ollama(chunk).await?,
                EmbeddingProvider::HuggingFace => self.embed_huggingface(chunk).await?,
                EmbeddingProvider::QwenCandle => {
                    return Err(Error::Configuration(
                        "QwenCandle should be handled by QwenCandleClient, not EmbedderClient"
                            .to_string(),
                    ));
                }
                EmbeddingProvider::Qwen3EmbedAnything => {
                    return Err(Error::Configuration(
                        "Qwen3EmbedAnything should be handled by Qwen3EmbedAnythingClient, not EmbedderClient"
                            .to_string(),
                    ));
                }
                EmbeddingProvider::EmbedAnything => {
                    return Err(Error::Configuration(
                        "EmbedAnything should be handled by EmbedAnythingClient, not EmbedderClient"
                            .to_string(),
                    ));
                }
                EmbeddingProvider::GemmaCandleApprox => {
                    return Err(Error::Configuration(
                        "GemmaCandleApprox should be handled by GemmaCandleClient, not EmbedderClient"
                            .to_string(),
                    ));
                }
            };
            uncached_embeddings.extend(embeddings);
        }

        // Fill in results and update cache
        for (idx, (original_idx, text)) in uncached_indices
            .iter()
            .zip(uncached_texts.iter())
            .enumerate()
        {
            let embedding = uncached_embeddings[idx].clone();
            self.cache.insert(text.clone(), embedding.clone()).await;
            results[*original_idx] = Some(embedding);
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
}

impl EmbedderClient {
    async fn embed_openai(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url());

        let request = OpenAIEmbeddingRequest {
            model: self.config.model.clone(),
            input: texts.to_vec(),
            dimensions: if self.config.dimension != 1536 {
                Some(self.config.dimension)
            } else {
                None
            },
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::EmbeddingProvider(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::EmbeddingProvider(format!(
                "API error: {}",
                error_text
            )));
        }

        let data: OpenAIEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| Error::EmbeddingProvider(format!("Failed to parse response: {}", e)))?;

        Ok(data.data.into_iter().map(|e| e.embedding).collect())
    }

    async fn embed_voyage(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.base_url());

        let request = VoyageEmbeddingRequest {
            model: self.config.model.clone(),
            input: texts.to_vec(),
            input_type: "document".to_string(),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::EmbeddingProvider(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::EmbeddingProvider(format!(
                "API error: {}",
                error_text
            )));
        }

        let data: VoyageEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| Error::EmbeddingProvider(format!("Failed to parse response: {}", e)))?;

        Ok(data.data.into_iter().map(|e| e.embedding).collect())
    }

    async fn embed_cohere(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embed", self.base_url());

        let request = CohereEmbeddingRequest {
            model: self.config.model.clone(),
            texts: texts.to_vec(),
            input_type: "search_document".to_string(),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::EmbeddingProvider(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::EmbeddingProvider(format!(
                "API error: {}",
                error_text
            )));
        }

        let data: CohereEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| Error::EmbeddingProvider(format!("Failed to parse response: {}", e)))?;

        Ok(data.embeddings)
    }

    async fn embed_local(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Local embedding using a hypothetical local server
        // This would typically run a SentenceTransformers model
        let url = format!("{}/embed", self.base_url());

        let request = serde_json::json!({
            "model": self.config.model,
            "texts": texts
        });

        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::EmbeddingProvider(format!(
                "Local embedding server request failed: {}. Make sure the embedding server is running on {}",
                e, self.base_url()
            )))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::EmbeddingProvider(format!(
                "Local embedding server error: {}. Check if the server is running and the model is available.",
                error_text
            )));
        }

        let data: serde_json::Value = response.json().await.map_err(|e| {
            Error::EmbeddingProvider(format!("Failed to parse local embedding response: {}", e))
        })?;

        // Handle different response formats
        if let Some(embeddings) = data.get("embeddings").and_then(|e| e.as_array()) {
            let mut result = Vec::new();
            for embedding in embeddings {
                if let Some(vec) = embedding.as_array() {
                    let floats: std::result::Result<Vec<f32>, Error> = vec
                        .iter()
                        .map(|v| {
                            v.as_f64().map(|f| f as f32).ok_or_else(|| {
                                Error::EmbeddingProvider("Invalid embedding value".to_string())
                            })
                        })
                        .collect();
                    result.push(floats?);
                }
            }
            Ok(result)
        } else {
            Err(Error::EmbeddingProvider(
                "Invalid local embedding response format".to_string(),
            ))
        }
    }

    async fn embed_ollama(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Use Ollama's embedding API
        let url = format!("{}/api/embeddings", self.base_url());

        let mut results = Vec::new();

        // Ollama typically processes one text at a time for embeddings
        for text in texts {
            let request = serde_json::json!({
                "model": self.config.model,
                "prompt": text
            });

            let response = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
                .map_err(|e| {
                    Error::EmbeddingProvider(format!(
                        "Ollama embedding request failed: {}. Make sure Ollama is running on {}",
                        e,
                        self.base_url()
                    ))
                })?;

            if !response.status().is_success() {
                let error_text = response.text().await.unwrap_or_default();
                return Err(Error::EmbeddingProvider(format!(
                    "Ollama embedding error: {}. Check if Ollama is running and the model '{}' supports embeddings.",
                    error_text, self.config.model
                )));
            }

            let data: serde_json::Value = response.json().await.map_err(|e| {
                Error::EmbeddingProvider(format!(
                    "Failed to parse Ollama embedding response: {}",
                    e
                ))
            })?;

            if let Some(embedding) = data.get("embedding").and_then(|e| e.as_array()) {
                let floats: std::result::Result<Vec<f32>, Error> = embedding
                    .iter()
                    .map(|v| {
                        v.as_f64().map(|f| f as f32).ok_or_else(|| {
                            Error::EmbeddingProvider("Invalid embedding value".to_string())
                        })
                    })
                    .collect();
                results.push(floats?);
            } else {
                return Err(Error::EmbeddingProvider(
                    "Invalid Ollama embedding response format".to_string(),
                ));
            }
        }

        Ok(results)
    }

    async fn embed_huggingface(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Use Hugging Face Inference API for embeddings
        let url = format!("{}/{}", self.base_url(), self.config.model);

        let mut results = Vec::new();

        // Process each text individually for HF Inference API
        for text in texts {
            let request = serde_json::json!({
                "inputs": text,
                "options": {
                    "wait_for_model": true,
                    "use_cache": false
                }
            });

            let mut req_builder = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&request);

            // Add authorization header if API key is provided
            if !self.config.api_key.is_empty() {
                req_builder =
                    req_builder.header("Authorization", format!("Bearer {}", self.config.api_key));
            }

            let response = req_builder
                .send()
                .await
                .map_err(|e| Error::EmbeddingProvider(format!(
                    "Hugging Face Inference API request failed: {}. Make sure the model '{}' supports embeddings.",
                    e, self.config.model
                )))?;

            if !response.status().is_success() {
                let error_text = response.text().await.unwrap_or_default();
                return Err(Error::EmbeddingProvider(format!(
                    "Hugging Face Inference API error: {}. Model: {}",
                    error_text, self.config.model
                )));
            }

            // HF Inference API returns embeddings directly as an array of floats
            let embedding: Vec<f32> = response.json().await.map_err(|e| {
                Error::EmbeddingProvider(format!("Failed to parse HF embedding response: {}", e))
            })?;

            results.push(embedding);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_config_default() {
        let config = EmbedderConfig::default();
        assert!(matches!(config.provider, EmbeddingProvider::Local));
        assert_eq!(config.model, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(config.dimension, 384);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_embedder_creation_fails_without_api_key() {
        // For cloud providers, API key is required
        let mut config = EmbedderConfig::default();
        config.provider = EmbeddingProvider::OpenAI;
        config.model = "text-embedding-3-small".to_string();
        let result = EmbedderClient::new(config);
        assert!(result.is_err());
    }
}
