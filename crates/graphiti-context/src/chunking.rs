use crate::types::CodeChunk;
use md5::{Digest, Md5};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Chunker {
    pub max_lines: usize,
    pub overlap: usize,
}

impl Default for Chunker {
    fn default() -> Self {
        Self { max_lines: 200, overlap: 30 }
    }
}

impl Chunker {
    pub fn chunk(&self, content: &str, abs_path: &Path, rel_path: &str) -> Vec<CodeChunk> {
        let lines: Vec<&str> = content.split_inclusive('\n').collect();
        if lines.is_empty() { return vec![]; }
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < lines.len() {
            let end = (start + self.max_lines).min(lines.len());
            let piece = &lines[start..end].join("");
            let start_line = start as u32 + 1;
            let end_line = end as u32;
            let id = chunk_id(rel_path, start_line, end_line, piece);
            let language = language_from_extension(abs_path.extension().and_then(|s| s.to_str()).unwrap_or(""));
            let file_extension = abs_path.extension().and_then(|s| s.to_str()).map(|s| format!(".{}", s)).unwrap_or_default();
            chunks.push(CodeChunk { id, content: piece.clone(), relative_path: rel_path.to_string(), start_line, end_line, file_extension, language: language.to_string() });
            if end == lines.len() { break; }
            if self.overlap >= end { break; }
            start = end - self.overlap;
        }
        chunks
    }
}

fn chunk_id(relative_path: &str, start_line: u32, end_line: u32, content: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(relative_path.as_bytes());
    hasher.update(b":");
    hasher.update(start_line.to_le_bytes());
    hasher.update(b":");
    hasher.update(end_line.to_le_bytes());
    hasher.update(b":");
    hasher.update(content.as_bytes());
    let digest = hasher.finalize();
    format!("chunk_{:x}", digest)[..22].to_string()
}

fn language_from_extension(ext: &str) -> &'static str {
    match ext {
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "py" => "python",
        "java" => "java",
        "cpp" => "cpp",
        "c" | "h" => "c",
        "hpp" => "cpp",
        "cs" => "csharp",
        "go" => "go",
        "rs" => "rust",
        "php" => "php",
        "rb" => "ruby",
        "swift" => "swift",
        "kt" => "kotlin",
        "scala" => "scala",
        "m" | "mm" => "objective-c",
        "ipynb" => "jupyter",
        _ => "text",
    }
}

