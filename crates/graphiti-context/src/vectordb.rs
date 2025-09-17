use anyhow::Result;

#[derive(Debug, Clone)]
pub struct VectorInsert {
    pub id: String,
    pub vector: Vec<f32>,
    pub content: String,
    pub relative_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub language: String,
    pub file_extension: String,
}

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub score: f32,
    pub content: String,
    pub relative_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub language: String,
}

pub trait VectorDatabase: Send + Sync {
    fn ensure_collection(&self, name: &str, dimension: usize) -> Result<()>;
    fn has_collection(&self, name: &str) -> Result<bool>;
    fn drop_collection(&self, name: &str) -> Result<()>;
    fn insert(&self, name: &str, points: &[VectorInsert]) -> Result<()>;
    fn search(&self, name: &str, vector: &[f32], top_k: usize) -> Result<Vec<VectorHit>>;
    fn delete_by_relative_paths(&self, name: &str, rel_paths: &[String]) -> Result<()>;
}
