//! Qwen3-Embedding implementation using embed_anything crate
//!
//! This implementation uses the embed_anything crate which provides
//! proper support for Qwen3-Embedding models through Candle backend.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::EmbeddingClient;
use graphiti_core::error::Error;
use graphiti_core::error::Result;

#[cfg(feature = "embed-anything")]
use embed_anything::embed_query;
#[cfg(feature = "embed-anything")]
use embed_anything::embeddings::embed::{EmbedData, Embedder, EmbeddingResult};

/// Configuration for Qwen3 embedding using embed_anything
#[derive(Debug, Clone)]
pub struct Qwen3EmbedAnythingConfig {
    /// Model ID from Hugging Face
    pub model_id: String,
    /// Batch size for processing
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Device type (CPU, CUDA, Metal)
    pub device: String,
}

impl Default for Qwen3EmbedAnythingConfig {
    fn default() -> Self {
        Self {
            model_id: "Qwen/Qwen3-Embedding-0.6B".to_string(),
            batch_size: 32,
            max_length: 8192,
            device: "auto".to_string(),
        }
    }
}

/// Qwen3 embedding client using embed_anything
pub struct Qwen3EmbedAnythingClient {
    #[cfg(feature = "embed-anything")]
    embedder: Embedder,
    config: Qwen3EmbedAnythingConfig,
    #[cfg(not(feature = "embed-anything"))]
    _phantom: std::marker::PhantomData<()>,
}

impl Qwen3EmbedAnythingClient {
    /// Create a new Qwen3 embedding client using embed_anything
    pub async fn new(config: Qwen3EmbedAnythingConfig) -> Result<Self> {
        #[cfg(feature = "embed-anything")]
        {
            info!("🚀 初始化 Qwen3-Embedding 使用 embed_anything");
            info!("📋 模型: {}", config.model_id);
            info!("🔧 设备: {}", config.device);

            // Create embedder with the model
            let embedder = Embedder::from_pretrained_hf(
                &config.model_id,
                "main", // revision
                None,   // cache_dir
                None,   // dtype
                None,   // additional dtype parameter
            )
            .map_err(|e| Error::Configuration(format!("Failed to create embedder: {}", e)))?;

            info!("✅ Qwen3-Embedding 客户端创建成功");

            Ok(Self { embedder, config })
        }

        #[cfg(not(feature = "embed-anything"))]
        {
            warn!("⚠️ embed-anything 特性未启用，使用占位符实现");
            Ok(Self {
                config,
                _phantom: std::marker::PhantomData,
            })
        }
    }

    /// Generate embeddings for a single text
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        #[cfg(feature = "embed-anything")]
        {
            debug!("🔄 生成嵌入向量，文本长度: {}", text.len());

            // Use embed_query function with proper parameters
            let texts: Vec<&str> = vec![text];
            let embedding = embed_query(&texts, &self.embedder, None)
                .await
                .map_err(|e| {
                    Error::EmbeddingProvider(format!("Embedding generation failed: {}", e))
                })?;

            if embedding.is_empty() {
                return Err(Error::EmbeddingProvider(
                    "No embeddings generated".to_string(),
                ));
            }

            // Extract the embedding vector from EmbedData
            match &embedding[0].embedding {
                EmbeddingResult::DenseVector(vec) => Ok(vec.clone()),
                _ => Err(Error::EmbeddingProvider(
                    "Unexpected embedding result type".to_string(),
                )),
            }
        }

        #[cfg(not(feature = "embed-anything"))]
        {
            warn!("使用占位符嵌入实现");
            // Return a placeholder embedding with correct dimension (1024 for Qwen3-0.6B)
            Ok(vec![0.0; 1024])
        }
    }

    /// Generate embeddings for multiple texts
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "embed-anything")]
        {
            debug!("🔄 批量生成嵌入向量，文本数量: {}", texts.len());

            // Convert Vec<String> to Vec<&str>
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let embeddings: Vec<EmbedData> = embed_query(&text_refs, &self.embedder, None)
                .await
                .map_err(|e| {
                    Error::EmbeddingProvider(format!("Batch embedding generation failed: {}", e))
                })?;

            // Extract embedding vectors from EmbedData
            let mut result: Vec<Vec<f32>> = Vec::new();
            for embed_data in embeddings {
                match embed_data.embedding {
                    EmbeddingResult::DenseVector(vec) => {
                        result.push(vec);
                    }
                    _ => {
                        return Err(Error::EmbeddingProvider(
                            "Unexpected embedding result type in batch".to_string(),
                        ));
                    }
                }
            }

            Ok(result)
        }

        #[cfg(not(feature = "embed-anything"))]
        {
            warn!("使用占位符批量嵌入实现");
            // Return placeholder embeddings
            Ok(texts.iter().map(|_| vec![0.0; 1024]).collect())
        }
    }

    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        1024 // Qwen3-Embedding-0.6B has 1024 dimensions
    }

    /// Get model configuration
    pub fn config(&self) -> &Qwen3EmbedAnythingConfig {
        &self.config
    }
}

/// Embedding service trait implementation
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32;
    fn embedding_dim(&self) -> usize;
}

#[async_trait]
impl EmbeddingService for Qwen3EmbedAnythingClient {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_text(text).await
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch(texts).await
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        self.cosine_similarity(a, b)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim()
    }
}

/// Implement EmbeddingClient trait for Qwen3EmbedAnythingClient
#[async_trait]
impl EmbeddingClient for Qwen3EmbedAnythingClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_text(text).await
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch(texts).await
    }
}

/// Create a new Qwen3 embedding service using embed_anything
pub async fn create_qwen3_embed_anything_service(
    config: Option<Qwen3EmbedAnythingConfig>,
) -> Result<Arc<dyn EmbeddingService>> {
    let config = config.unwrap_or_default();
    let client = Qwen3EmbedAnythingClient::new(config).await?;
    Ok(Arc::new(client))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_qwen3_embed_anything_config() {
        let config = Qwen3EmbedAnythingConfig::default();
        assert_eq!(config.model_id, "Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.max_length, 8192);
    }

    #[tokio::test]
    async fn test_cosine_similarity() {
        let config = Qwen3EmbedAnythingConfig::default();
        let client = Qwen3EmbedAnythingClient::new(config).await.unwrap();

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = client.cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        let similarity = client.cosine_similarity(&a, &c);
        assert!((similarity - 0.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_embedding_dimension() {
        let config = Qwen3EmbedAnythingConfig::default();
        let client = Qwen3EmbedAnythingClient::new(config).await.unwrap();
        assert_eq!(client.embedding_dim(), 1024);
    }
}
