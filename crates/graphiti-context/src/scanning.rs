use anyhow::Result;
use ignore::{DirEntry, WalkBuilder};
use std::path::{Path, PathBuf};

use crate::rules::RuleSet;

pub fn scan_code_files(codebase_path: &Path, rules: &RuleSet) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    let mut walker = WalkBuilder::new(codebase_path);
    walker.hidden(false).git_ignore(true).git_global(true).git_exclude(true);
    let walker = walker.build();
    for res in walker {
        if let Ok(entry) = res { let _ = handle_entry(codebase_path, rules, entry, &mut |p| paths.push(p)); }
    }
    Ok(paths)
}

fn handle_entry<F>(root: &Path, rules: &RuleSet, entry: DirEntry, push: &mut F) -> Result<()>
where F: FnMut(PathBuf) {
    let path = entry.path();
    if path == root { return Ok(()); }
    let file_type = entry.file_type();
    let is_dir = file_type.map(|t| t.is_dir()).unwrap_or(false);
    if rules.is_ignored(root, path, is_dir) { return Ok(()); }
    if is_dir { return Ok(()); }
    if !rules.is_supported_ext(path) { return Ok(()); }
    push(path.to_path_buf()); Ok(())
}

