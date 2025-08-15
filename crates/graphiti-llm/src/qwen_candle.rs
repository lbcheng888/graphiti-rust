//! Qwen embedding implementation using Candle
//!
//! This is a simplified implementation that demonstrates the concept.
//! For production use, you would want to implement proper model loading and inference.

use async_trait::async_trait;
use candle_core::DType;
use candle_core::Device;

use graphiti_core::error::Error;
use graphiti_core::error::Result;
use moka::future::Cache;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::debug;
use tracing::info;
use tracing::instrument;

#[cfg(feature = "embed-anything")]
use embed_anything::embed_query;
#[cfg(feature = "embed-anything")]
use embed_anything::embeddings::embed::{EmbedData, Embedder, EmbeddingResult};

use crate::EmbeddingClient;

/// Detect the best available device for computation
pub fn detect_device() -> Device {
    if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    }
}

/// Configuration for Candle-based Qwen embedding model
#[derive(Debug, Clone)]
pub struct QwenCandleConfig {
    /// Model repository (e.g., "Qwen/Qwen3-Embedding-0.6B")
    pub model_repo: String,
    /// Model revision/branch
    pub revision: String,
    /// Local cache directory
    pub cache_dir: Option<PathBuf>,
    /// Device to use (CPU, CUDA, Metal)
    pub device: Device,
    /// Embedding dimension
    pub dimension: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Enforce offline-only behavior: if true, requires local cache files and never attempts network
    pub offline: bool,
    /// Data type for computations
    pub dtype: DType,
}

impl Default for QwenCandleConfig {
    fn default() -> Self {
        Self {
            model_repo: "Qwen/Qwen3-Embedding-0.6B".to_string(),
            revision: "main".to_string(),
            cache_dir: Some(PathBuf::from("./Qwen3-Embedding-0.6B")),
            device: Device::Cpu,
            dimension: 1536,
            batch_size: 16,
            max_length: 8192,
            normalize: true,
            offline: true,
            dtype: DType::F32,
        }
    }
}

// 使用 EmbedAnything 的 Embedder 作为真实模型（基于 Candle 后端）
#[cfg(feature = "embed-anything")]
pub type QwenInnerModel = Embedder;
#[cfg(not(feature = "embed-anything"))]
pub struct QwenInnerModelPlaceholder;
#[cfg(not(feature = "embed-anything"))]
pub type QwenInnerModel = QwenInnerModelPlaceholder;

/// Candle-based Qwen embedding client
pub struct QwenCandleClient {
    config: QwenCandleConfig,
    model: Arc<RwLock<Option<QwenInnerModel>>>,
    cache: Cache<String, Vec<f32>>,
}

impl QwenCandleClient {
    /// Create a new Qwen Candle client
    pub fn new(config: QwenCandleConfig) -> Result<Self> {
        // Create cache with 2 hour TTL and 5000 max entries
        let cache = Cache::builder()
            .time_to_live(Duration::from_secs(7200))
            .max_capacity(5000)
            .build();

        info!(
            "Initialized Qwen Candle client with model: {} on device: {:?}",
            config.model_repo, config.device
        );

        Ok(Self {
            config,
            model: Arc::new(RwLock::new(None)),
            cache,
        })
    }

    /// Load the model (EmbedAnything-based, Candle backend)
    pub async fn load_model(&self) -> Result<()> {
        info!("Loading Qwen model: {}", self.config.model_repo);

        // 优先使用本地缓存目录（配置的 cache_dir 或环境变量 QWEN_CANDLE_CACHE_DIR）
        let env_dir = std::env::var("QWEN_CANDLE_CACHE_DIR").ok().map(PathBuf::from);
        let local_dir = env_dir.or_else(|| self.config.cache_dir.clone());
        if let Some(dir) = &local_dir {
            info!("Using local Qwen cache directory: {}", dir.as_path().display());
        }

        #[cfg(feature = "embed-anything")]
        let cache_dir_string = local_dir.as_ref().map(|p| p.to_string_lossy().to_string());
        #[cfg(feature = "embed-anything")]
        let cache_dir_ref = cache_dir_string.as_deref();

        // 若 offline 开启，且目录缺失关键文件，直接报错，绝不联网
        if self.config.offline {
            if let Some(dir) = &local_dir {
                let cfg = dir.join("config.json");
                let tok = dir.join("tokenizer.json");
                let mdl = dir.join("model.safetensors");
                if !(cfg.is_file() && tok.is_file() && mdl.is_file()) {
                    return Err(Error::Configuration(format!(
                        "Offline mode: missing required files in {} (need config.json, tokenizer.json, model.safetensors)",
                        dir.display()
                    )));
                }
            } else {
                return Err(Error::Configuration(
                    "Offline mode: cache_dir not set and QWEN_CANDLE_CACHE_DIR not provided".to_string(),
                ));
            }
        }

        #[cfg(feature = "embed-anything")]
        let embedder = Embedder::from_pretrained_hf(
            &self.config.model_repo,
            &self.config.revision,
            cache_dir_ref,
            None,
            None,
        )
        .map_err(|e| Error::Configuration(format!("Failed to create embedder: {}", e)))?;

        #[cfg(feature = "embed-anything")]
        {
            *self.model.write().await = Some(embedder);
            info!("Qwen model loaded successfully");
            Ok(())
        }

        #[cfg(not(feature = "embed-anything"))]
        {
            Err(Error::Configuration(
                "embed-anything feature is required for QwenCandleClient".to_string(),
            ))
        }
    }

    /// Check if model is loaded
    pub async fn is_loaded(&self) -> bool {
        self.model.read().await.is_some()
    }

    /// Generate embeddings for texts using EmbedAnything
    async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if !self.is_loaded().await {
            self.load_model().await?;
        }

        let model_guard = self.model.read().await;
        let embedder = model_guard
            .as_ref()
            .ok_or_else(|| Error::EmbeddingProvider("Model not loaded".to_string()))?;

        // Convert to &str slice
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        #[cfg(feature = "embed-anything")]
        let embeddings: Vec<EmbedData> = embed_query(&text_refs, embedder, None)
            .await
            .map_err(|e| Error::EmbeddingProvider(format!("Embedding generation failed: {}", e)))?;

        #[cfg(feature = "embed-anything")]
        {
            let mut result: Vec<Vec<f32>> = Vec::with_capacity(embeddings.len());
            for embed_data in embeddings {
                match embed_data.embedding {
                    EmbeddingResult::DenseVector(mut vec) => {
                        if self.config.normalize {
                            let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if norm > 0.0 {
                                for v in &mut vec {
                                    *v /= norm;
                                }
                            }
                        }
                        result.push(vec);
                    }
                    _ => {
                        return Err(Error::EmbeddingProvider(
                            "Unexpected embedding result type".to_string(),
                        ));
                    }
                }
            }
            debug!("Generated {} embeddings", result.len());
            Ok(result)
        }

        #[cfg(not(feature = "embed-anything"))]
        {
            Err(Error::Configuration(
                "embed-anything feature is required for embedding generation".to_string(),
            ))
        }
    }
}

#[async_trait]
impl EmbeddingClient for QwenCandleClient {
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
                debug!("Cache hit for Qwen Candle embedding");
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
            let embeddings = self.generate_embeddings(chunk).await?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_candle_config_default() {
        let config = QwenCandleConfig::default();
        assert_eq!(config.model_repo, "Qwen/Qwen3-0.6B-Base");
        assert_eq!(config.dimension, 768);
        assert_eq!(config.batch_size, 16);
        assert!(config.normalize);
    }

    #[test]
    fn test_device_detection() {
        let device = detect_device();
        // Should not panic and return a valid device
        assert!(matches!(
            device,
            Device::Cpu | Device::Cuda(_) | Device::Metal(_)
        ));
    }

    #[tokio::test]
    async fn test_qwen_candle_client_creation() {
        let config = QwenCandleConfig::default();
        let result = QwenCandleClient::new(config);
        assert!(result.is_ok());
    }
}
