use anyhow::Result;

pub trait EmbeddingModel: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let v = self.embed_batch(&[text.to_string()])?;
        Ok(v.into_iter().next().unwrap_or_default())
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn name(&self) -> &'static str;
}

/// Lightweight deterministic fallback embedding
pub struct HashingEmbedding { dim: usize }
impl HashingEmbedding { pub fn new(dim: usize) -> Self { Self { dim } } }
impl EmbeddingModel for HashingEmbedding {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            let mut v = vec![0f32; self.dim];
            for (i, b) in t.as_bytes().iter().enumerate() { v[i % self.dim] += (*b as f32) / 255.0; }
            let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
            if norm > 0.0 { for x in &mut v { *x /= norm; } }
            out.push(v);
        }
        Ok(out)
    }
    fn dimension(&self) -> usize { self.dim }
    fn name(&self) -> &'static str { "hashing-fallback" }
}

#[cfg(feature = "embed-anything")]
pub mod qwen3_embed_anything {
    use super::EmbeddingModel;
    use anyhow::{anyhow, Result};
    use embed_anything::embed_query;
    use embed_anything::embeddings::embed::{EmbedData, Embedder, EmbeddingResult};
    use std::sync::Arc;

    pub struct Qwen3EmbedAnything { embedder: Arc<Embedder>, dim: usize }
    impl Qwen3EmbedAnything { pub fn load() -> Result<Self> {
        let cache_dir = std::env::var("EMBEDDING_MODEL_DIR").ok();
        let embedder = Embedder::from_pretrained_hf(
            "Qwen/Qwen3-Embedding-0.6B", "main", cache_dir.as_deref(), None, None
        ).map_err(|e| anyhow!("create embedder: {e}"))?;
        Ok(Self { embedder: Arc::new(embedder), dim: 1024 })
    }}
    impl EmbeddingModel for Qwen3EmbedAnything {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() { return Ok(vec![]); }
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let out: Vec<EmbedData> = rt.block_on(async { embed_query(&refs, &self.embedder, None).await })
                .map_err(|e| anyhow!("embed: {e}"))?;
            let mut result = Vec::with_capacity(out.len());
            for item in out.into_iter() {
                match item.embedding {
                    EmbeddingResult::DenseVector(mut v) => {
                        if v.len() == self.dim { result.push(v); }
                        else if v.len() > self.dim { v.truncate(self.dim); result.push(v); }
                        else { v.resize(self.dim, 0.0); result.push(v); }
                    }
                    _ => return Err(anyhow!("unexpected embedding result type")),
                }
            }
            Ok(result)
        }
        fn dimension(&self) -> usize { self.dim }
        fn name(&self) -> &'static str { "qwen3-embedding-0.6b-embed-anything" }
    }
}

#[cfg(feature = "candle-gemma")]
pub mod gemma_candle {
    use super::EmbeddingModel;
    use anyhow::{anyhow, Result};
    use candle_core::{DType, Device, Tensor, IndexOp};
    use hf_hub::api::sync::Api;
    use serde_json::Value as JsonValue;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tokenizers::Tokenizer;

    pub struct CandleGemmaEmbedding { dim: usize, device: Device, emb_weight: Tensor, tokenizer: Option<Tokenizer>, vocab_size: usize }

    impl CandleGemmaEmbedding {
        pub fn load(model_dir: Option<PathBuf>) -> Result<Self> {
            let device = Device::Cpu;
            let (model_dir, _using_hub) = resolve_model_dir(model_dir)?;
            let (tokenizer_opt, hidden_size, vocab_size) = load_tokenizer_and_config(&model_dir).unwrap_or_else(|e| {
                eprintln!("[gemma] tokenizer/config load failed: {e}");
                let cfg_path = model_dir.join("config.json");
                let cfg_text = fs::read_to_string(&cfg_path).unwrap_or_else(|_| "{}".into());
                let cfg_json: JsonValue = serde_json::from_str(&cfg_text).unwrap_or(JsonValue::Null);
                let hidden = pick_usize(&cfg_json, &["hidden_size","n_embed","model_dim","d_model"]).unwrap_or(Some(768)).unwrap_or(768);
                let vocab = pick_usize(&cfg_json, &["vocab_size"]).unwrap_or(Some(262144)).unwrap_or(262144);
                (None::<Tokenizer>, hidden, vocab)
            });

            let vb = mmap_var_builder(&model_dir)?;
            let mut emb_weight: Option<Tensor> = None;
            let mut candidates: Vec<String> = vec![
                "model.embed_tokens.weight","embed_tokens.weight","tok_embeddings.weight","embeddings.word_embeddings.weight",
                "model.vocab_emb.weight","model.text_model.embeddings.word_embeddings.weight","transformer.wte.weight","shared.weight",
                "model.encoder.embed_tokens.weight","model.decoder.embed_tokens.weight","model.embedder.weight","model.tok_embeddings.weight",
            ].into_iter().map(|s| s.to_string()).collect();
            let idx_path = model_dir.join("model.safetensors.index.json");
            if idx_path.exists() {
                if let Ok(text) = std::fs::read_to_string(&idx_path) {
                    if let Ok(j) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(map) = j.get("weight_map").and_then(|v| v.as_object()) {
                            let mut present: Vec<String> = vec![];
                            for k in map.keys() { if candidates.iter().any(|c| k == c) { present.push(k.clone()); } }
                            for k in map.keys() {
                                let ks = k.to_lowercase();
                                if (ks.contains("embed") || ks.contains("token")) && ks.ends_with(".weight") {
                                    if !present.iter().any(|p| p == k) { present.push(k.clone()); }
                                }
                            }
                            if !present.is_empty() {
                                let mut merged = present.clone();
                                for c in candidates { if !merged.iter().any(|p| p == &c) { merged.push(c); } }
                                candidates = merged;
                            }
                        }
                    }
                }
            }
            for name in &candidates { if let Ok(w) = vb_get_tensor(&vb, name.as_str(), (vocab_size, hidden_size), &device) { emb_weight = Some(w); break; } }
            if emb_weight.is_none() {
                let path = model_dir.join("model.safetensors");
                if path.exists() {
                    if let Ok(t) = super::load_safetensors_matrix(&path, &candidates, vocab_size, hidden_size) {
                        emb_weight = Some(Tensor::from_vec(t, (vocab_size, hidden_size), &device)?);
                    }
                }
            }
            let emb_weight = emb_weight.ok_or_else(|| anyhow!("embedding weight not found in model.safetensors"))?;
            Ok(Self { dim: hidden_size, device, emb_weight, tokenizer: tokenizer_opt, vocab_size })
        }
    }

    impl EmbeddingModel for CandleGemmaEmbedding {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                let ids: Vec<u32> = if let Some(tok) = &self.tokenizer { let e = tok.encode(t.as_str(), true).map_err(|e| anyhow!("tokenize: {e}"))?; e.get_ids().iter().map(|&i| i as u32).collect() } else {
                    let mut v = Vec::new();
                    for w in t.split_whitespace() { let mut h: u64 = 1469598103934665603; for b in w.as_bytes() { h ^= *b as u64; h = h.wrapping_mul(1099511628211); } v.push((h as usize % self.vocab_size) as u32); }
                    v
                };
                if ids.is_empty() { out.push(vec![0f32; self.dim]); continue; }
                let ids_t = Tensor::new(&ids[..], &self.device)?;
                let ids_2d = ids_t.unsqueeze(1)?;
                let embs = self.emb_weight.gather(&ids_2d, 0)?.squeeze(1)?;
                let mean = embs.mean(0)?;
                let mut v: Vec<f32> = mean.to_vec1()?;
                let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt(); if norm > 0.0 { for x in &mut v { *x /= norm; } }
                out.push(v);
            }
            Ok(out)
        }
        fn dimension(&self) -> usize { self.dim }
        fn name(&self) -> &'static str { "embeddinggemma-300m-candle" }
    }

    fn resolve_model_dir(model_dir: Option<PathBuf>) -> Result<(PathBuf, bool)> {
        if let Some(dir) = model_dir { return Ok((dir, false)); }
        if let Ok(p) = std::env::var("EMBEDDING_MODEL_DIR") { return Ok((PathBuf::from(p), false)); }
        let api = Api::new()?; let model = api.model("google/embeddinggemma-300m".to_string());
        let _ = model.get("model.safetensors.index.json"); let _ = model.get("model.safetensors"); let _ = model.get("config.json"); let tok = model.get("tokenizer.json")?;
        let dir = tok.parent().ok_or_else(|| anyhow!("cannot get model dir from tokenizer path"))?.to_path_buf();
        Ok((dir, true))
    }
    fn load_tokenizer_and_config(model_dir: &Path) -> Result<(Option<Tokenizer>, usize, usize)> {
        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| anyhow!("load tokenizer: {e}"))?;
        let cfg_path = model_dir.join("config.json"); let cfg_text = fs::read_to_string(&cfg_path)?; let cfg_json: JsonValue = serde_json::from_str(&cfg_text)?;
        let hidden_size = pick_usize(&cfg_json, &["hidden_size","n_embed","model_dim","d_model"])?.ok_or_else(|| anyhow!("hidden size not found in config"))?;
        let vocab_size = pick_usize(&cfg_json, &["vocab_size"])?.ok_or_else(|| anyhow!("vocab size not found in config"))?;
        Ok((Some(tokenizer), hidden_size, vocab_size))
    }
    fn pick_usize(v: &JsonValue, keys: &[&str]) -> Result<Option<usize>> { for k in keys { if let Some(n) = v.get(*k).and_then(|x| x.as_u64()) { return Ok(Some(n as usize)); } } Ok(None) }
    struct VB(candle_nn::VarBuilder<'static>);
    fn mmap_var_builder(model_dir: &Path) -> Result<VB> { let device = Device::Cpu; let idx = model_dir.join("model.safetensors.index.json");
        if idx.exists() { let idx_text = fs::read_to_string(&idx)?; let j: JsonValue = serde_json::from_str(&idx_text)?; let mut shards = std::collections::BTreeSet::new(); if let Some(wm) = j.get("weight_map").and_then(|x| x.as_object()) { for (_, file) in wm.iter() { if let Some(f) = file.as_str() { shards.insert(f.to_string()); } } } let paths: Vec<PathBuf> = shards.into_iter().map(|f| model_dir.join(f)).collect(); let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, DType::F32, &device)? }; Ok(VB(vb)) }
        else { let path = model_dir.join("model.safetensors"); let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)? }; Ok(VB(vb)) } }
    fn vb_get_tensor(vb: &VB, name: &str, shape: (usize, usize), _device: &Device) -> Result<Tensor> { let (vocab, dim) = shape; if let Ok(t) = vb.0.get((vocab, dim), name) { return Ok(t); } if let Ok(t) = vb.0.get((dim, vocab), name) { return Ok(t.transpose(0,1)?); } Err(anyhow!("tensor not found or shape mismatch for key: {}", name)) }
}

#[cfg(feature = "candle-gemma")]
pub fn load_safetensors_matrix(path: &std::path::Path, candidates: &[String], vocab: usize, dim: usize) -> Result<Vec<f32>> {
    use safetensors::SafeTensors;
    use std::fs::File; use std::io::Read;
    let mut data = vec![]; File::open(path)?.read_to_end(&mut data)?; let st = SafeTensors::deserialize(&data).map_err(|e| anyhow::anyhow!("safetensors deserialize: {e}"))?;
    for name in candidates { if let Ok(t) = st.tensor(&name) { let shape = t.shape(); if shape == [vocab, dim] || shape == [dim, vocab] { let mut buf = vec![0f32; (shape[0]*shape[1]) as usize]; t.data().read_exact(bytemuck::cast_slice_mut(&mut buf)).map_err(|e| anyhow::anyhow!("read tensor data: {e}"))?; if shape == [dim, vocab] { let mut tr = vec![0f32; buf.len()]; for i in 0..dim { for j in 0..vocab { tr[j*dim + i] = buf[i*vocab + j]; } } return Ok(tr); } return Ok(buf); } } }
    Err(anyhow::anyhow!("no matching tensor in safetensors"))
}

