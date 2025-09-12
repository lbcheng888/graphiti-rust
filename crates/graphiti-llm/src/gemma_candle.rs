//! Native lightweight embedding client (deterministic, CPU) without embed_anything.
//! NOTE: This is an approximation: it uses tokenizer + deterministic hashing-based
//! projections to 768-dim and mean pooling, then L2-normalization. It does not run
//! the full transformer. Suitable for offline, fast, and consistent embeddings.

use async_trait::async_trait;
use graphiti_core::error::{Error, Result};
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Configuration for native Gemma (Candle) embedding
#[derive(Debug, Clone)]
pub struct GemmaCandleConfig {
    /// Root directory containing tokenizer and weights (e.g., EMBEDDING_MODEL_DIR)
    pub model_dir: Option<PathBuf>,
    /// Device: cpu/metal/cuda (defaults to cpu if unsupported)
    pub device: String,
    /// Target output dimension (EmbeddingGemma-300m is 768)
    pub target_dim: usize,
    /// L2 normalize final vectors
    pub normalize: bool,
}

impl Default for GemmaCandleConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            device: "auto".to_string(),
            target_dim: 768,
            normalize: true,
        }
    }
}

/// Native Candle client that approximates embeddings using token embedding average.
/// Note: This does not run the full Transformer encoder; it uses the token embedding matrix
/// with mean pooling as a lightweight approximation.
pub struct GemmaCandleClient {
    tokenizer: Tokenizer,
    cfg: GemmaCandleConfig,
}

impl GemmaCandleClient {
    /// Create a new GemmaCandleClient
    /// Requires tokenizer.json in `model_dir` or `EMBEDDING_MODEL_DIR`.
    pub fn new(mut cfg: GemmaCandleConfig) -> Result<Self> {
        // Resolve model directory (for tokenizer only)
        let model_dir = if let Some(dir) = cfg.model_dir.take() {
            dir
        } else if let Ok(env) = std::env::var("EMBEDDING_MODEL_DIR") {
            PathBuf::from(env)
        } else {
            return Err(Error::Configuration(
                "GemmaCandle: EMBEDDING_MODEL_DIR not set and model_dir not provided".into(),
            ));
        };

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(Error::Configuration(format!(
                "GemmaCandle: tokenizer.json not found in {}",
                model_dir.display()
            )));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Configuration(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self { tokenizer, cfg })
    }

    #[allow(dead_code)]
    fn adjust_dim(mut v: Vec<f32>, target: usize) -> Vec<f32> {
        if v.len() > target {
            v.truncate(target);
            v
        } else if v.len() < target {
            v.resize(target, 0.0);
            v
        } else {
            v
        }
    }

    fn l2_normalize(v: &mut [f32]) {
        let mut sum = 0.0f32;
        for x in v.iter() {
            sum += x * x;
        }
        let n = sum.sqrt();
        if n > 0.0 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
    }

    fn hash32(mut x: u64) -> u32 {
        // SplitMix64 then truncate to 32-bit
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) & 0xFFFF_FFFF) as u32
    }

    fn token_projection(token_id: u32, dim: usize, _out_dim: usize) -> f32 {
        // Deterministically map (token_id, dim) -> [-1, 1]
        let seed = ((token_id as u64) << 32) ^ (dim as u64) ^ 0xA5A5_5A5A_1234_5678;
        let rnd = Self::hash32(seed);
        // Map to [0,1)
        let u = (rnd as f32) / (u32::MAX as f32);
        // Center to [-1,1]
        2.0 * u - 1.0
    }
}

#[async_trait]
impl crate::EmbeddingClient for GemmaCandleClient {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(texts.len());

        for text in texts {
            let enc = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| Error::EmbeddingProvider(format!("Tokenize failed: {}", e)))?;
            let ids = enc.get_ids();
            let mut v = vec![0.0f32; self.cfg.target_dim];
            if !ids.is_empty() {
                for (i, val) in v.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for &tid in ids {
                        acc += Self::token_projection(tid, i, self.cfg.target_dim);
                    }
                    *val = acc / (ids.len() as f32);
                }
            }
            // Normalize if requested
            if self.cfg.normalize {
                Self::l2_normalize(&mut v);
            }
            out.push(v);
        }

        Ok(out)
    }
}
