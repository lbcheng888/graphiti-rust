//! Local Qwen model integration for embeddings

use async_trait::async_trait;
use graphiti_core::error::Error;
use graphiti_core::error::Result;
use moka::future::Cache;
use reqwest::Client;
use serde::Deserialize;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Duration;
use tracing::debug;
use tracing::info;
use tracing::instrument;

use crate::EmbeddingClient;

/// Configuration for local Qwen embedding model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenLocalConfig {
    /// Path to the Qwen model directory
    pub model_path: PathBuf,
    /// Model name (e.g., "Qwen/Qwen3-0.6B-Base")
    pub model_name: String,
    /// Embedding dimension
    pub dimension: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Device to use ("cpu", "cuda", "mps")
    pub device: String,
    /// Maximum sequence length
    pub max_length: usize,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Local server URL (if using a separate embedding server)
    pub server_url: Option<String>,
    /// Request timeout
    pub timeout: Duration,
}

impl Default for QwenLocalConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("./models/Qwen3-0.6B-Base"),
            model_name: "Qwen/Qwen3-0.6B-Base".to_string(),
            dimension: 768,            // Typical dimension for Qwen models
            batch_size: 16,            // Conservative batch size for local inference
            device: "cpu".to_string(), // Default to CPU for compatibility
            max_length: 512,           // Standard sequence length
            normalize: true,           // Normalize embeddings by default
            server_url: Some("http://localhost:8001".to_string()), // Default local server
            timeout: Duration::from_secs(120), // Longer timeout for local inference
        }
    }
}

/// Local Qwen embedding client
pub struct QwenLocalClient {
    config: QwenLocalConfig,
    client: Client,
    cache: Cache<String, Vec<f32>>,
}

impl QwenLocalClient {
    /// Create a new Qwen local client
    pub fn new(config: QwenLocalConfig) -> Result<Self> {
        // Validate model path if not using server
        if config.server_url.is_none() && !config.model_path.exists() {
            return Err(Error::Configuration(format!(
                "Qwen model path does not exist: {}",
                config.model_path.display()
            )));
        }

        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| Error::Configuration(format!("Failed to create HTTP client: {}", e)))?;

        // Create cache with 2 hour TTL and 5000 max entries
        let cache = Cache::builder()
            .time_to_live(Duration::from_secs(7200))
            .max_capacity(5000)
            .build();

        info!(
            "Initialized Qwen local client with model: {} on device: {}",
            config.model_name, config.device
        );

        Ok(Self {
            config,
            client,
            cache,
        })
    }

    /// Check if the local embedding server is available
    pub async fn health_check(&self) -> Result<bool> {
        if let Some(server_url) = &self.config.server_url {
            let url = format!("{}/health", server_url);

            match self.client.get(&url).send().await {
                Ok(response) => Ok(response.status().is_success()),
                Err(_) => Ok(false),
            }
        } else {
            // If not using server, assume direct model access is available
            Ok(self.config.model_path.exists())
        }
    }

    /// Start the local embedding server (if needed)
    pub async fn start_server(&self) -> Result<()> {
        if let Some(server_url) = &self.config.server_url {
            // Check if server is already running
            if self.health_check().await? {
                info!("Qwen embedding server is already running at {}", server_url);
                return Ok(());
            }

            info!("Starting Qwen embedding server...");

            // This would typically start a Python process with the embedding server
            // For now, we'll just return an error with instructions
            return Err(Error::Configuration(format!(
                "Qwen embedding server is not running at {}. Please start the server manually using the provided Python script.",
                server_url
            )));
        }

        Ok(())
    }

    /// Embed texts using the local server
    async fn embed_via_server(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let server_url = self
            .config
            .server_url
            .as_ref()
            .ok_or_else(|| Error::Configuration("No server URL configured".to_string()))?;

        let url = format!("{}/embed", server_url);

        let request = QwenEmbeddingRequest {
            texts: texts.to_vec(),
            model: self.config.model_name.clone(),
            normalize: self.config.normalize,
            max_length: self.config.max_length,
        };

        debug!(
            "Sending embedding request to Qwen server: {} texts",
            texts.len()
        );

        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::EmbeddingProvider(format!(
                "Qwen server request failed: {}. Make sure the Qwen embedding server is running at {}",
                e, server_url
            )))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::EmbeddingProvider(format!(
                "Qwen server error: {}. Check server logs for details.",
                error_text
            )));
        }

        let embedding_response: QwenEmbeddingResponse = response.json().await.map_err(|e| {
            Error::EmbeddingProvider(format!("Failed to parse Qwen server response: {}", e))
        })?;

        if embedding_response.embeddings.len() != texts.len() {
            return Err(Error::EmbeddingProvider(format!(
                "Mismatch in embedding count: expected {}, got {}",
                texts.len(),
                embedding_response.embeddings.len()
            )));
        }

        debug!(
            "Successfully received {} embeddings from Qwen server",
            embedding_response.embeddings.len()
        );
        Ok(embedding_response.embeddings)
    }
}

/// Request format for Qwen embedding server
#[derive(Debug, Serialize)]
struct QwenEmbeddingRequest {
    texts: Vec<String>,
    model: String,
    normalize: bool,
    max_length: usize,
}

/// Response format from Qwen embedding server
#[derive(Debug, Deserialize)]
struct QwenEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    dimension: usize,
    #[serde(default)]
    #[allow(dead_code)]
    processing_time: f64,
}

#[async_trait]
impl EmbeddingClient for QwenLocalClient {
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
                debug!("Cache hit for Qwen embedding");
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
            let embeddings = self.embed_via_server(chunk).await?;
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

/// Create a Python embedding server script for Qwen
pub fn create_qwen_server_script() -> String {
    r#"#!/usr/bin/env python3
"""
Qwen Local Embedding Server

This script provides a simple HTTP server for generating embeddings using Qwen models.
It's designed to work with the Rust Graphiti implementation.

Requirements:
- transformers
- torch
- fastapi
- uvicorn
- numpy

Install with: pip install transformers torch fastapi uvicorn numpy

Usage:
    python qwen_embedding_server.py --model Qwen/Qwen3-0.6B-Base --port 8001
"""

import argparse
import logging
import time
from typing import List, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: str
    normalize: bool = True
    max_length: int = 512

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    processing_time: float

class QwenEmbeddingServer:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading Qwen model: {model_name} on {device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(device)
            self.model.eval()

            # Get embedding dimension
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt", max_length=10, truncation=True)
                test_input = {k: v.to(device) for k, v in test_input.items()}
                test_output = self.model(**test_input)
                self.dimension = test_output.last_hidden_state.shape[-1]

            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def embed_texts(self, texts: List[str], normalize: bool = True, max_length: int = 512) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        start_time = time.time()

        try:
            # Tokenize all texts
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Use mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)

                # Normalize if requested
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to list
                embeddings = embeddings.cpu().numpy().tolist()

            processing_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {processing_time:.3f}s")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

# Global model instance
embedding_server = None

# FastAPI app
app = FastAPI(title="Qwen Embedding Server", version="1.0.0")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": embedding_server.model_name if embedding_server else "not_loaded",
        "device": embedding_server.device if embedding_server else "unknown"
    }

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """Generate embeddings for texts"""
    if embedding_server is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    if len(request.texts) > 100:  # Reasonable limit
        raise HTTPException(status_code=400, detail="Too many texts (max 100)")

    try:
        start_time = time.time()
        embeddings = embedding_server.embed_texts(
            request.texts,
            normalize=request.normalize,
            max_length=request.max_length
        )
        processing_time = time.time() - start_time

        return EmbeddingResponse(
            embeddings=embeddings,
            model=embedding_server.model_name,
            dimension=embedding_server.dimension,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error in embed endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    parser = argparse.ArgumentParser(description="Qwen Embedding Server")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base", help="Model name or path")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")

    args = parser.parse_args()

    # Initialize the embedding server
    global embedding_server
    embedding_server = QwenEmbeddingServer(args.model, args.device)

    # Start the server
    logger.info(f"Starting Qwen embedding server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_config_default() {
        let config = QwenLocalConfig::default();
        assert_eq!(config.model_name, "Qwen/Qwen3-0.6B-Base");
        assert_eq!(config.dimension, 768);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.device, "cpu");
        assert!(config.normalize);
    }

    #[test]
    fn test_qwen_client_creation() {
        let mut config = QwenLocalConfig::default();
        config.server_url = Some("http://localhost:8001".to_string());

        let result = QwenLocalClient::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_server_script_generation() {
        let script = create_qwen_server_script();
        assert!(script.contains("QwenEmbeddingServer"));
        assert!(script.contains("FastAPI"));
        assert!(script.contains("/embed"));
    }
}
