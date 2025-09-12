use crate::types::CodeChunk;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMeta { pub name: String }

#[derive(Debug, Clone)]
pub struct Collection { pub dir: PathBuf, pub chunks_path: PathBuf, pub meta_path: PathBuf }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult { pub content: String, pub relative_path: String, pub start_line: u32, pub end_line: u32, pub language: String, pub score: f32 }

#[derive(Debug, Clone)]
pub struct LocalIndexStorage { base_dir: PathBuf }

impl LocalIndexStorage {
    pub fn new() -> Result<Self> {
        // Priority:
        // 1) Respect explicit env override CONTEXT_INDEX_DIR
        // 2) Use project working directory: ./.graphiti/data/context-indexes
        // 3) Fall back to user home: ~/.context/indexes
        let base = if let Ok(dir) = std::env::var("CONTEXT_INDEX_DIR") {
            PathBuf::from(dir)
        } else {
            // Try project directory from env or current dir
            let project_dir = std::env::var_os("GRAPHITI_PROJECT")
                .map(PathBuf::from)
                .unwrap_or(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            let proj_base = project_dir.join(".graphiti").join("data").join("context-indexes");
            if std::fs::create_dir_all(&proj_base).is_ok() {
                proj_base
            } else {
                // Fallback to home
                let mut d = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("no home dir"))?;
                d.push(".context");
                d.push("indexes");
                d
            }
        };
        std::fs::create_dir_all(&base)?;
        Ok(Self { base_dir: base })
    }
    pub fn collection_dir(&self, name: &str) -> Result<PathBuf> { Ok(self.base_dir.join(name)) }
    pub fn ensure_collection(&self, name: &str) -> Result<Collection> {
        let dir = self.base_dir.join(name); std::fs::create_dir_all(&dir)?;
        let chunks_path = dir.join("chunks.jsonl"); let meta_path = dir.join("meta.json");
        if !meta_path.exists() { let meta = CollectionMeta { name: name.to_string() }; let f = File::create(&meta_path)?; serde_json::to_writer_pretty(f, &meta)?; }
        Ok(Collection { dir, chunks_path, meta_path })
    }
    pub fn has_collection(&self, name: &str) -> Result<bool> { Ok(self.base_dir.join(name).is_dir()) }
    pub fn drop_collection(&self, name: &str) -> Result<()> { let dir = self.base_dir.join(name); if dir.exists() { std::fs::remove_dir_all(dir)?; } Ok(()) }
    pub fn insert_chunks(&self, name: &str, chunks: &[CodeChunk]) -> Result<()> {
        let c = self.ensure_collection(name)?; let file = OpenOptions::new().create(true).append(true).open(&c.chunks_path)?; let mut writer = BufWriter::new(file);
        for ch in chunks { let json = serde_json::to_string(ch)?; writer.write_all(json.as_bytes())?; writer.write_all(b"\n")?; } writer.flush()?; Ok(())
    }
    pub fn search(&self, name: &str, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let dir = self.base_dir.join(name); let path = dir.join("chunks.jsonl"); if !path.exists() { return Ok(vec![]); }
        let file = File::open(&path)?; let reader = BufReader::new(file);
        let query_tokens: Vec<String> = query.split_whitespace().map(|s| s.to_ascii_lowercase()).filter(|s| !s.is_empty()).collect();
        let mut results: Vec<SearchResult> = Vec::new();
        for line in reader.lines() { let line = line?; if line.trim().is_empty() { continue; } let chunk: CodeChunk = serde_json::from_str(&line)?; let lc = chunk.content.to_ascii_lowercase(); let mut score = 0f32; for t in &query_tokens { if lc.contains(t) { score += 1.0; } }
            if score > 0.0 { results.push(SearchResult { content: chunk.content.clone(), relative_path: chunk.relative_path.clone(), start_line: chunk.start_line, end_line: chunk.end_line, language: chunk.language.clone(), score }); }
        }
        results.sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)); results.truncate(top_k); Ok(results)
    }
    pub fn remove_file(&self, name: &str, relative_path: &str) -> Result<()> {
        let dir = self.base_dir.join(name); let path = dir.join("chunks.jsonl"); if !path.exists() { return Ok(()); }
        let tmp = dir.join("chunks.jsonl.tmp"); let in_f = File::open(&path)?; let mut reader = BufReader::new(in_f); let out_f = File::create(&tmp)?; let mut writer = BufWriter::new(out_f); let mut buf = String::new();
        while reader.read_line(&mut buf)? > 0 { let line = buf.trim_end_matches('\n'); if !line.is_empty() { if let Ok(ch) = serde_json::from_str::<CodeChunk>(line) { if ch.relative_path != relative_path { writer.write_all(line.as_bytes())?; writer.write_all(b"\n")?; } } } buf.clear(); }
        writer.flush()?; std::fs::rename(tmp, path)?; Ok(())
    }
}
