use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    pub id: String,
    pub content: String,
    pub relative_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub file_extension: String,
    pub language: String,
}

#[derive(Debug, Clone)]
pub enum ProgressPhase {
    Preparing,
    Scanning,
    Indexing,
    Completed,
}

#[derive(Debug, Clone)]
pub struct IndexProgress {
    pub phase: ProgressPhase,
    pub current: usize,
    pub total: usize,
    pub percentage: u64,
}
