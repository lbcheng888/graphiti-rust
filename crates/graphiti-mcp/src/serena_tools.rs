//! Lightweight bridge to expose a subset of Serena tools via Graphiti MCP.
//! This integrates essential file/search/edit operations without pulling full Serena runtime.

use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::fs;
use std::path::PathBuf;
use glob::Pattern;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListDirRequest { pub path: PathBuf }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListDirEntry { pub name: String, pub is_dir: bool }

pub async fn list_dir(req: ListDirRequest) -> Result<Vec<ListDirEntry>> {
    let mut out = Vec::new();
    let rd = fs::read_dir(&req.path)?;
    for e in rd.flatten() {
        let ftype = e.file_type().ok();
        out.push(ListDirEntry { name: e.file_name().to_string_lossy().to_string(), is_dir: ftype.map(|t| t.is_dir()).unwrap_or(false) });
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest { pub root: PathBuf, pub pattern: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit { pub path: PathBuf, pub line: usize, pub preview: String }

pub async fn search(req: SearchRequest) -> Result<Vec<SearchHit>> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(&req.root).into_iter().flatten() {
        let p = entry.path();
        if !p.is_file() { continue; }
        if let Ok(txt) = fs::read_to_string(p) {
            for (i, line) in txt.lines().enumerate() {
                if line.contains(&req.pattern) {
                    out.push(SearchHit { path: p.to_path_buf(), line: i + 1, preview: line.to_string() });
                }
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaceLinesRequest { pub path: PathBuf, pub start: usize, pub end: usize, pub content: String }

pub async fn replace_lines(req: ReplaceLinesRequest) -> Result<()> {
    let txt = fs::read_to_string(&req.path)?;
    let mut lines: Vec<String> = txt.lines().map(|s| s.to_string()).collect();
    let start = req.start.saturating_sub(1);
    let end = req.end.min(lines.len());
    lines.splice(start..end, req.content.lines().map(|s| s.to_string()));
    let out = if lines.is_empty() { String::new() } else { lines.join("\n") + "\n" };
    fs::write(&req.path, out)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteLinesRequest { pub path: PathBuf, pub start: usize, pub end: usize }

pub async fn delete_lines(req: DeleteLinesRequest) -> Result<()> {
    let txt = fs::read_to_string(&req.path)?;
    let mut lines: Vec<String> = txt.lines().map(|s| s.to_string()).collect();
    let start = req.start.saturating_sub(1);
    let end = req.end.min(lines.len());
    if start < end { lines.drain(start..end); }
    let out = if lines.is_empty() { String::new() } else { lines.join("\n") + "\n" };
    fs::write(&req.path, out)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertAtLineRequest { pub path: PathBuf, pub line: usize, pub content: String }

pub async fn insert_at_line(req: InsertAtLineRequest) -> Result<()> {
    let txt = fs::read_to_string(&req.path)?;
    let mut lines: Vec<String> = txt.lines().map(|s| s.to_string()).collect();
    let idx = req.line.saturating_sub(1).min(lines.len());
    let new_lines: Vec<String> = req.content.lines().map(|s| s.to_string()).collect();
    lines.splice(idx..idx, new_lines);
    let out = if lines.is_empty() { String::new() } else { lines.join("\n") + "\n" };
    fs::write(&req.path, out)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadFileRequest { pub path: PathBuf }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadFileResponse { pub content: String }

pub async fn read_file(req: ReadFileRequest) -> Result<ReadFileResponse> {
    let content = fs::read_to_string(&req.path).unwrap_or_default();
    Ok(ReadFileResponse { content })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteFileRequest { pub path: PathBuf, pub content: String }

pub async fn write_file(req: WriteFileRequest) -> Result<()> {
    // Ensure parent exists
    if let Some(parent) = req.path.parent() { let _ = fs::create_dir_all(parent); }
    fs::write(&req.path, req.content.as_bytes())?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindFileRequest { pub root: PathBuf, pub glob: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindFileHit { pub path: PathBuf }

pub async fn find_file(req: FindFileRequest) -> Result<Vec<FindFileHit>> {
    let pat = Pattern::new(&req.glob).unwrap_or_else(|_| Pattern::new("*").unwrap());
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(&req.root).into_iter().flatten() {
        let p = entry.path();
        if !p.is_file() { continue; }
        let rel = p.strip_prefix(&req.root).unwrap_or(p);
        if pat.matches_path(rel) {
            out.push(FindFileHit { path: p.to_path_buf() });
        }
    }
    Ok(out)
}
