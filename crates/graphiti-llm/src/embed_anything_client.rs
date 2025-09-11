//! Generic embedding client using embed_anything (Candle backend)
//!
//! Supports loading any Hugging Face embedding model, e.g. `google/embeddinggemma-300m`.

use async_trait::async_trait;

use graphiti_core::error::{Error, Result};

#[cfg(feature = "embed-anything")]
use embed_anything::embed_query;
#[cfg(feature = "embed-anything")]
use embed_anything::embeddings::embed::{EmbedData, Embedder, EmbeddingResult};
use tokio::sync::RwLock;

/// Configuration for generic embed_anything embedding
#[derive(Debug, Clone)]
pub struct EmbedAnythingConfig {
    /// Hugging Face model ID, e.g. "google/embeddinggemma-300m"
    pub model_id: String,
    /// Batch size for processing
    pub batch_size: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Device type (CPU, CUDA, Metal)
    pub device: String,
    /// Optional local cache directory for HF models
    pub cache_dir: Option<String>,
    /// Optional target embedding dimension; if set, vectors will be truncated/padded
    pub target_dim: Option<usize>,
}

impl Default for EmbedAnythingConfig {
    fn default() -> Self {
        Self {
            model_id: "google/embeddinggemma-300m".to_string(),
            batch_size: 32,
            max_length: 8192,
            device: "auto".to_string(),
            cache_dir: None,
            target_dim: None,
        }
    }
}

/// Generic embedding client powered by embed_anything
pub struct EmbedAnythingClient {
    #[cfg(feature = "embed-anything")]
    embedder: RwLock<Option<Embedder>>, // Lazy load to avoid blocking server startup
    config: EmbedAnythingConfig,
    #[cfg(not(feature = "embed-anything"))]
    _phantom: std::marker::PhantomData<()>,
}

impl EmbedAnythingClient {
    /// Create a new generic embed_anything client
    pub async fn new(config: EmbedAnythingConfig) -> Result<Self> {
        #[cfg(feature = "embed-anything")]
        {
            tracing::info!(
                "初始化 embed_anything 嵌入器，模型: {}，设备: {}",
                config.model_id,
                config.device
            );

            // Do not load model here to keep startup fast; defer to first use
            Ok(Self {
                embedder: RwLock::new(None),
                config,
            })
        }

        #[cfg(not(feature = "embed-anything"))]
        {
            tracing::warn!("embed-anything 特性未启用，使用占位符实现");
            Ok(Self {
                config,
                _phantom: std::marker::PhantomData,
            })
        }
    }

    async fn ensure_loaded(&self) -> Result<()> {
        #[cfg(feature = "embed-anything")]
        {
            if self.embedder.read().await.is_some() {
                return Ok(());
            }
            let mut w = self.embedder.write().await;
            if w.is_some() {
                return Ok(());
            }
            // Resolve cache directory preference order: config.cache_dir -> EMBEDDING_MODEL_DIR -> HF_HUB_CACHE -> HF_HOME
            let cache_dir_env = std::env::var("EMBEDDING_MODEL_DIR").ok();
            let hf_cache = std::env::var("HF_HUB_CACHE").ok();
            let hf_home = std::env::var("HF_HOME").ok();
            let cache_dir_choice = self
                .config
                .cache_dir
                .clone()
                .or(cache_dir_env)
                .or(hf_cache)
                .or(hf_home);
            let cache_dir_path = cache_dir_choice.as_deref();
            tracing::info!("Loading embedder from cache_dir={:?}", cache_dir_path);
            let embedder = Embedder::from_pretrained_hf(
                &self.config.model_id,
                "main",
                cache_dir_path,
                None,
                None,
            )
            .map_err(|e| Error::Configuration(format!("Failed to create embedder: {}", e)))?;
            *w = Some(embedder);
            return Ok(());
        }
        #[cfg(not(feature = "embed-anything"))]
        {
            Ok(())
        }
    }

    /// Generate embeddings for a single text
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        #[cfg(feature = "embed-anything")]
        {
            self.ensure_loaded().await?;
            let guard = self.embedder.read().await;
            let embedder = guard.as_ref().ok_or_else(|| Error::EmbeddingProvider("Embedder not loaded".to_string()))?;
            let texts: Vec<&str> = vec![text];
            let embedding = embed_query(&texts, embedder, None)
                .await
                .map_err(|e| Error::EmbeddingProvider(format!("Embedding generation failed: {}", e)))?;

            if embedding.is_empty() {
                return Err(Error::EmbeddingProvider("No embeddings generated".to_string()));
            }

            match &embedding[0].embedding {
                EmbeddingResult::DenseVector(vec) => {
                    let mut out = vec.clone();
                    if let Some(td) = self.config.target_dim {
                        adjust_dim_in_place(&mut out, td);
                    }
                    Ok(out)
                },
                _ => Err(Error::EmbeddingProvider(
                    "Unexpected embedding result type".to_string(),
                )),
            }
        }

        #[cfg(not(feature = "embed-anything"))]
        {
            // Return a placeholder embedding with a common dimension (e.g., 1024)
            Ok(vec![0.0; 1024])
        }
    }

    /// Generate embeddings for multiple texts
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "embed-anything")]
        {
            self.ensure_loaded().await?;
            let guard = self.embedder.read().await;
            let embedder = guard.as_ref().ok_or_else(|| Error::EmbeddingProvider("Embedder not loaded".to_string()))?;
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let embeddings: Vec<EmbedData> = embed_query(&refs, embedder, None)
                .await
                .map_err(|e| Error::EmbeddingProvider(format!("Batch embedding generation failed: {}", e)))?;

            let mut result: Vec<Vec<f32>> = Vec::new();
            for embed_data in embeddings {
                match embed_data.embedding {
                    EmbeddingResult::DenseVector(mut vec) => {
                        if let Some(td) = self.config.target_dim {
                            adjust_dim_in_place(&mut vec, td);
                        }
                        result.push(vec)
                    },
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
            Ok(texts.iter().map(|_| vec![0.0; 1024]).collect())
        }
    }
}

fn adjust_dim_in_place(v: &mut Vec<f32>, target: usize) {
    if v.len() > target {
        v.truncate(target);
    } else if v.len() < target {
        v.resize(target, 0.0);
    }
}

#[async_trait]
impl crate::EmbeddingClient for EmbedAnythingClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_text(text).await
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch(texts).await
    }
}
