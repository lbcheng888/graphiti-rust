use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo { pub name: String, pub kind: String, pub line: usize }

fn read_lines(path: &Path) -> Result<Vec<String>> { Ok(fs::read_to_string(path)?.lines().map(|s| s.to_string()).collect()) }

fn detect_lang(path: &Path) -> &'static str {
    match path.extension().and_then(|s| s.to_str()).unwrap_or("") {
        "rs" => "rust", "py" => "python", "ts"|"tsx"|"js"|"jsx" => "ts", _ => "text"
    }
}

pub fn list_symbols(path: &Path) -> Result<Vec<SymbolInfo>> {
    let lang = detect_lang(path);
    let lines = read_lines(path)?;
    let mut out = Vec::new();
    match lang {
        "rust" => {
            let re = Regex::new(r"^\s*(pub\s+)?(fn|struct|enum|mod|trait)\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap();
            for (i,l) in lines.iter().enumerate() { if let Some(c) = re.captures(l) { out.push(SymbolInfo{ name: c.get(3).unwrap().as_str().to_string(), kind: c.get(2).unwrap().as_str().to_string(), line: i+1 }); } }
        }
        "python" => {
            let re_def = Regex::new(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(").unwrap();
            let re_cls = Regex::new(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(:|\()").unwrap();
            for (i,l) in lines.iter().enumerate() {
                if let Some(c)=re_def.captures(l){ out.push(SymbolInfo{ name:c.get(1).unwrap().as_str().to_string(), kind:"def".into(), line:i+1 }); }
                if let Some(c)=re_cls.captures(l){ out.push(SymbolInfo{ name:c.get(1).unwrap().as_str().to_string(), kind:"class".into(), line:i+1 }); }
            }
        }
        "ts" => {
            let re_fn = Regex::new(r"^\s*(export\s+)?(async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(").unwrap();
            let re_cls = Regex::new(r"^\s*(export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b").unwrap();
            let re_const = Regex::new(r"^\s*(export\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\(").unwrap();
            for (i,l) in lines.iter().enumerate() {
                if let Some(c)=re_fn.captures(l){ out.push(SymbolInfo{ name:c.get(3).unwrap().as_str().to_string(), kind:"function".into(), line:i+1 }); }
                if let Some(c)=re_cls.captures(l){ out.push(SymbolInfo{ name:c.get(2).unwrap().as_str().to_string(), kind:"class".into(), line:i+1 }); }
                if let Some(c)=re_const.captures(l){ out.push(SymbolInfo{ name:c.get(2).unwrap().as_str().to_string(), kind:"const-fn".into(), line:i+1 }); }
            }
        }
        _ => {}
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindSymbolRequest { pub root: PathBuf, pub name: String }

pub fn find_symbol(req: &FindSymbolRequest) -> Result<Vec<(PathBuf, SymbolInfo)>> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(&req.root).into_iter().flatten() {
        let p = entry.path(); if !p.is_file() { continue; }
        let syms = list_symbols(p);
        if let Ok(v) = syms { for s in v { if s.name == req.name { out.push((p.to_path_buf(), s)); } } }
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindRefsRequest { pub root: PathBuf, pub name: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefHit { pub path: PathBuf, pub line: usize, pub preview: String }

pub fn find_references(req: &FindRefsRequest) -> Result<Vec<RefHit>> {
    let pat = Regex::new(&format!(r"\b{}\b", regex::escape(&req.name)))?;
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(&req.root).into_iter().flatten() {
        let p = entry.path(); if !p.is_file() { continue; }
        let Ok(txt) = fs::read_to_string(p) else { continue };
        for (i,l) in txt.lines().enumerate() {
            if pat.is_match(l) { out.push(RefHit{ path: p.to_path_buf(), line: i+1, preview: l.to_string() }); }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaceSymbolBodyRequest { pub path: PathBuf, pub name: String, pub new_body: String }

pub fn replace_symbol_body(req: &ReplaceSymbolBodyRequest) -> Result<()> {
    let lang = detect_lang(&req.path);
    let txt = fs::read_to_string(&req.path)?;
    let chars: Vec<char> = txt.chars().collect();
    let s = txt.as_str();
    // find def line index range
    let def_idx = match lang {
        "rust" | "ts" => {
            let re = Regex::new(&format!(r"(?m)^\s*(pub\s+)?(fn|struct|enum|class|const)\s+{}\b", regex::escape(&req.name)))?;
            re.find(s).map(|m| m.start())
        }
        "python" => {
            let re = Regex::new(&format!(r"(?m)^\s*(def|class)\s+{}\b", regex::escape(&req.name)))?;
            re.find(s).map(|m| m.start())
        }
        _ => None,
    }.ok_or_else(|| anyhow!("symbol not found"))?;

    if lang == "python" {
        // replace indented block
        let def_line_start = s[..def_idx].rfind('\n').map(|i| i+1).unwrap_or(0);
        let def_line_end = s[def_line_start..].find('\n').map(|i| def_line_start+i).unwrap_or(s.len());
        let def_line = &s[def_line_start..def_line_end];
        let indent = def_line.chars().take_while(|c| *c==' ' || *c=='\t').collect::<String>();
        // block starts at next line with strictly greater indent
        let mut pos = def_line_end+1;
        let mut start_block = pos; let mut end_block = pos;
        let mut in_block = false;
        while pos < s.len() {
            let line_start = pos;
            let line_end = s[pos..].find('\n').map(|i| pos+i).unwrap_or(s.len());
            let line = &s[line_start..line_end];
            let cur_indent = line.chars().take_while(|c| *c==' '||*c=='\t').collect::<String>();
            if !in_block {
                if line.trim().is_empty() { pos=line_end+1; continue; }
                if cur_indent.len()>indent.len() { in_block=true; start_block=line_start; end_block=line_end; } else { pos=line_end+1; continue; }
            } else {
                if line.trim().is_empty() { end_block=line_end; pos=line_end+1; continue; }
                if cur_indent.len()<=indent.len() { break; } else { end_block=line_end; }
            }
            pos=line_end+1;
        }
        let mut out = String::new();
        out.push_str(&s[..start_block]);
        for l in req.new_body.lines() { out.push_str(&indent); out.push_str("    "); out.push_str(l); out.push('\n'); }
        out.push_str(&s[end_block..]);
        fs::write(&req.path, out)?; return Ok(())
    }

    // brace-based languages: replace inside braces after def
    let brace_pos = s[def_idx..].find('{').map(|i| def_idx+i).ok_or_else(|| anyhow!("opening brace not found"))?;
    // scan for matching brace
    let mut depth = 0usize; let mut i = brace_pos; let n = chars.len();
    let mut end = None;
    while i < n {
        let c = chars[i];
        if c=='{' { depth+=1; }
        else if c=='}' { depth-=1; if depth==0 { end=Some(i); break; } }
        i+=1;
    }
    let end_brace = end.ok_or_else(|| anyhow!("closing brace not found"))?;
    // Build new content with same indentation level
    let line_start = s[..brace_pos].rfind('\n').map(|i| i+1).unwrap_or(0);
    let indent = s[line_start..brace_pos].chars().rev().take_while(|c| *c==' '||*c=='\t').collect::<String>();
    let mut out = String::new();
    out.push_str(&s[..brace_pos+1]); out.push('\n');
    for l in req.new_body.lines() { out.push_str(&indent); out.push_str("    "); out.push_str(l); out.push('\n'); }
    out.push_str(&s[end_brace..]);
    fs::write(&req.path, out)?;
    Ok(())
}
