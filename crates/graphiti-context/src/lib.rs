pub mod chunking;
pub mod embedding;
pub mod rules;
pub mod scanning;
pub mod storage;
pub mod types;
pub mod util;
pub mod vectordb;
pub mod vectordb_qdrant;

use crate::chunking::Chunker;
use crate::embedding::{EmbeddingModel, HashingEmbedding};
use crate::rules::RuleSet;
use crate::scanning::scan_code_files;
use crate::storage::{LocalIndexStorage, SearchResult};
use crate::types::{IndexProgress, ProgressPhase};
use crate::vectordb::{VectorDatabase, VectorInsert};
use anyhow::Result;
use md5::{Digest, Md5};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub supported_extensions: Vec<String>,
    pub ignore_patterns: Vec<String>,
    pub custom_extensions: Vec<String>,
    pub custom_ignore_patterns: Vec<String>,
    pub hybrid_mode: bool,
    // Vector DB
    pub use_qdrant: bool,
    pub qdrant_url: Option<String>,
    // Embedding
    pub use_candle_gemma: bool,
    pub embedding_model_dir: Option<String>,
    pub fallback_dim: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            supported_extensions: rules::default_supported_extensions(),
            ignore_patterns: rules::default_ignore_patterns(),
            custom_extensions: rules::env_custom_extensions(),
            custom_ignore_patterns: rules::env_custom_ignore_patterns(),
            hybrid_mode: util::env_flag("HYBRID_MODE").unwrap_or(true),
            use_qdrant: std::env::var("QDRANT_URL").is_ok(),
            qdrant_url: std::env::var("QDRANT_URL").ok(),
            use_candle_gemma: std::env::var("USE_CANDLE_GEMMA")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            embedding_model_dir: std::env::var("EMBEDDING_MODEL_DIR").ok(),
            fallback_dim: 1024,
        }
    }
}

pub struct Context {
    cfg: ContextConfig,
    rules: RuleSet,
    chunker: Chunker,
    storage: LocalIndexStorage,
    embedder: Box<dyn EmbeddingModel>,
    vectordb: Option<Box<dyn VectorDatabase>>,
}

impl Context {
    pub fn new(cfg: Option<ContextConfig>) -> Result<Self> {
        let cfg = cfg.unwrap_or_default();
        let rules = RuleSet::from_config(&cfg)?;
        let chunker = Chunker::default();
        let storage = LocalIndexStorage::new()?;
        // Embedding initialization (prefer Qwen3 via embed_anything -> Gemma -> Hashing)
        #[allow(unused_mut)]
        let mut embedder: Box<dyn EmbeddingModel> =
            Box::new(HashingEmbedding::new(cfg.fallback_dim));

        // Prefer Qwen3-Embedding (1024-dim) via embed_anything if requested
        if std::env::var("USE_QWEN3")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
        {
            #[cfg(feature = "embed-anything")]
            {
                if let Ok(e) = embedding::qwen3_embed_anything::Qwen3EmbedAnything::load() {
                    embedder = Box::new(e);
                }
            }
        }
        #[cfg(feature = "candle-gemma")]
        if cfg.use_candle_gemma {
            // Bridge common token env vars for hf-hub.
            if std::env::var("HF_TOKEN").is_err() {
                if let Ok(tok) = std::env::var("HUGGINGFACE_TOKEN") {
                    std::env::set_var("HF_TOKEN", tok);
                }
            }
            if std::env::var("HF_HUB_TOKEN").is_err() {
                if let Ok(tok) =
                    std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGINGFACE_TOKEN"))
                {
                    std::env::set_var("HF_HUB_TOKEN", tok);
                }
            }
            let model_dir = cfg
                .embedding_model_dir
                .as_ref()
                .map(|s| std::path::PathBuf::from(s));
            if let Ok(e) = embedding::gemma_candle::CandleGemmaEmbedding::load(model_dir) {
                embedder = Box::new(e);
            }
        }

        // Vector DB: Qdrant if configured
        let vectordb: Option<Box<dyn VectorDatabase>> = if cfg.use_qdrant {
            let url = cfg
                .qdrant_url
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:6334".to_string());
            let db = crate::vectordb_qdrant::QdrantVectorDb::new(&url)?;
            Some(Box::new(db))
        } else {
            None
        };

        Ok(Self {
            cfg,
            rules,
            chunker,
            storage,
            embedder,
            vectordb,
        })
    }

    pub fn collection_name(&self, codebase_path: &Path) -> String {
        let normalized = codebase_path
            .canonicalize()
            .unwrap_or_else(|_| codebase_path.to_path_buf());
        let mut hasher = Md5::new();
        hasher.update(normalized.to_string_lossy().as_bytes());
        let digest = hasher.finalize();
        let hash = format!("{:x}", digest);
        let prefix = if self.cfg.hybrid_mode {
            "hybrid_code_chunks"
        } else {
            "code_chunks"
        };
        format!("{}_{}", prefix, &hash[..8])
    }

    pub fn has_index(&self, codebase_path: &Path) -> Result<bool> {
        let name = self.collection_name(codebase_path);
        self.storage.has_collection(&name)
    }

    pub fn clear_index(&self, codebase_path: &Path) -> Result<()> {
        let name = self.collection_name(codebase_path);
        self.storage.drop_collection(&name)
    }

    pub fn index_codebase<F>(
        &self,
        codebase_path: &Path,
        mut progress: Option<F>,
        force_reindex: bool,
    ) -> Result<(usize, usize)>
    where
        F: FnMut(IndexProgress),
    {
        let name = self.collection_name(codebase_path);
        if force_reindex {
            let _ = self.storage.drop_collection(&name);
        }
        self.storage.ensure_collection(&name)?;

        if let Some(cb) = progress.as_mut() {
            cb(IndexProgress {
                phase: ProgressPhase::Preparing,
                current: 0,
                total: 100,
                percentage: 0,
            });
        }

        let files = scan_code_files(codebase_path, &self.rules)?;
        if let Some(cb) = progress.as_mut() {
            cb(IndexProgress {
                phase: ProgressPhase::Scanning,
                current: 5,
                total: 100,
                percentage: 5,
            });
        }

        let total_files = files.len();
        let mut processed_files = 0usize;
        let mut total_chunks = 0usize;
        for (i, file) in files.iter().enumerate() {
            let content = std::fs::read_to_string(file)?;
            let rel_path = util::relative_path(codebase_path, file);
            let chunks = self.chunker.chunk(&content, file, &rel_path);
            total_chunks += chunks.len();

            if let Some(db) = &self.vectordb {
                // embed & insert to qdrant
                let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
                let vectors = self.embedder.embed_batch(&texts)?;
                if vectors.is_empty() {
                    continue;
                }
                db.ensure_collection(&name, vectors[0].len())?;
                let mut inserts = Vec::with_capacity(chunks.len());
                for (i, ch) in chunks.iter().enumerate() {
                    inserts.push(VectorInsert {
                        id: ch.id.clone(),
                        vector: vectors[i].clone(),
                        content: ch.content.clone(),
                        relative_path: ch.relative_path.clone(),
                        start_line: ch.start_line,
                        end_line: ch.end_line,
                        language: ch.language.clone(),
                        file_extension: ch.file_extension.clone(),
                    });
                }
                db.insert(&name, &inserts)?;
            } else {
                // local JSONL fallback
                self.storage.insert_chunks(&name, &chunks)?;
            }
            processed_files += 1;
            if let Some(cb) = progress.as_mut() {
                // Reserve 10% for prep, 90% for indexing
                let pct = 10 + ((i + 1) as f64 / total_files.max(1) as f64 * 90.0) as u64;
                cb(IndexProgress {
                    phase: ProgressPhase::Indexing,
                    current: i + 1,
                    total: total_files,
                    percentage: pct.min(100),
                });
            }
        }

        if let Some(cb) = progress.as_mut() {
            cb(IndexProgress {
                phase: ProgressPhase::Completed,
                current: processed_files,
                total: total_files,
                percentage: 100,
            });
        }
        Ok((processed_files, total_chunks))
    }

    pub fn search(
        &self,
        codebase_path: &Path,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        let name = self.collection_name(codebase_path);
        if let Some(db) = &self.vectordb {
            let v = self.embedder.embed(query)?;
            let hits = db.search(&name, &v, top_k)?;
            let out = hits
                .into_iter()
                .map(|h| SearchResult {
                    content: h.content,
                    relative_path: h.relative_path,
                    start_line: h.start_line,
                    end_line: h.end_line,
                    language: h.language,
                    score: h.score,
                })
                .collect();
            Ok(out)
        } else {
            self.storage.search(&name, query, top_k)
        }
    }

    pub fn collection_dir(&self, codebase_path: &Path) -> Result<PathBuf> {
        let name = self.collection_name(codebase_path);
        self.storage.collection_dir(&name)
    }

    pub fn get_embedder_name(&self) -> &'static str {
        self.embedder.name()
    }
    pub fn get_embedder_dim(&self) -> usize {
        self.embedder.dimension()
    }
    pub fn is_qdrant_enabled(&self) -> bool {
        self.vectordb.is_some()
    }

    // Incremental: index only specified files
    pub fn index_paths(&self, codebase_path: &Path, files: &[PathBuf]) -> Result<(usize, usize)> {
        let name = self.collection_name(codebase_path);
        let mut total_chunks = 0usize;
        for file in files {
            if !file.is_file() {
                continue;
            }
            let content = std::fs::read_to_string(file)?;
            let rel_path = util::relative_path(codebase_path, file);
            let chunks = self.chunker.chunk(&content, file, &rel_path);
            total_chunks += chunks.len();
            if let Some(db) = &self.vectordb {
                let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
                let vectors = self.embedder.embed_batch(&texts)?;
                if vectors.is_empty() {
                    continue;
                }
                db.ensure_collection(&name, vectors[0].len())?;
                let mut inserts = Vec::with_capacity(chunks.len());
                for (i, ch) in chunks.iter().enumerate() {
                    inserts.push(VectorInsert {
                        id: ch.id.clone(),
                        vector: vectors[i].clone(),
                        content: ch.content.clone(),
                        relative_path: ch.relative_path.clone(),
                        start_line: ch.start_line,
                        end_line: ch.end_line,
                        language: ch.language.clone(),
                        file_extension: ch.file_extension.clone(),
                    });
                }
                db.insert(&name, &inserts)?;
            } else {
                self.storage.insert_chunks(&name, &chunks)?;
            }
        }
        Ok((files.len(), total_chunks))
    }

    // Incremental: remove points for a file
    pub fn remove_file_points(&self, codebase_path: &Path, relative_path: &str) -> Result<()> {
        let name = self.collection_name(codebase_path);
        if let Some(db) = &self.vectordb {
            db.delete_by_relative_paths(&name, &[relative_path.to_string()])
        } else {
            self.storage.remove_file(&name, relative_path)
        }
    }
}
