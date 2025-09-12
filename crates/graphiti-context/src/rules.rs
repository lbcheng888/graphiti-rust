use anyhow::Result;
use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::gitignore::GitignoreBuilder;
use std::path::{Path, PathBuf};


#[derive(Debug, Clone)]
pub struct RuleSet {
    pub supported_extensions: Vec<String>,
    pub ignore_patterns: Vec<String>,
    globset: GlobSet,
}

impl RuleSet {
    pub fn from_config(cfg: &crate::ContextConfig) -> Result<Self> {
        let mut extensions = default_supported_extensions();
        extensions.extend(cfg.supported_extensions.iter().cloned());
        extensions.extend(cfg.custom_extensions.iter().cloned());
        extensions.sort();
        extensions.dedup();

        let mut patterns = default_ignore_patterns();
        patterns.extend(cfg.ignore_patterns.iter().cloned());
        patterns.extend(cfg.custom_ignore_patterns.iter().cloned());

        let globset = build_globset(&patterns)?;
        Ok(Self { supported_extensions: extensions, ignore_patterns: patterns, globset })
    }

    pub fn is_supported_ext(&self, path: &Path) -> bool {
        let ext = path.extension().and_then(|s| s.to_str()).map(|s| format!(".{}", s)).unwrap_or_default();
        self.supported_extensions.iter().any(|e| e == &ext)
    }

    pub fn is_ignored(&self, root: &Path, path: &Path, is_dir: bool) -> bool {
        if self.globset.is_match(path) { return true; }
        let mut builder = GitignoreBuilder::new(root);
        for f in find_ignore_files(root) { let _ = builder.add(f); }
        if let Ok(gi) = builder.build() { gi.matched_path_or_any_parents(path, is_dir).is_ignore() } else { false }
    }
}

pub fn default_supported_extensions() -> Vec<String> {
    vec![
        ".ts", ".tsx", ".js", ".jsx", ".py", ".java", ".cpp", ".c", ".h", ".hpp",
        ".cs", ".go", ".rs", ".php", ".rb", ".swift", ".kt", ".scala", ".m", ".mm",
        ".md", ".markdown", ".ipynb",
    ].into_iter().map(String::from).collect()
}

pub fn default_ignore_patterns() -> Vec<String> {
    vec![
        "node_modules/**", "dist/**", "build/**", "out/**", "target/**", "coverage/**", ".nyc_output/**",
        ".vscode/**", ".idea/**", "*.swp", "*.swo",
        ".git/**", ".svn/**", ".hg/**",
        ".cache/**", "__pycache__/**", ".pytest_cache/**",
        "logs/**", "tmp/**", "temp/**", "*.log",
        ".env", ".env.*", "*.local",
        "*.min.js", "*.min.css", "*.min.map", "*.bundle.js", "*.bundle.css", "*.chunk.js", "*.vendor.js",
        "*.polyfills.js", "*.runtime.js", "*.map",
        "node_modules", ".git", ".svn", ".hg", "build", "dist", "out", "target", ".vscode", ".idea",
        "__pycache__", ".pytest_cache", "coverage", ".nyc_output", "logs", "tmp", "temp",
    ].into_iter().map(String::from).collect()
}

pub fn env_custom_extensions() -> Vec<String> {
    let v = std::env::var("CUSTOM_EXTENSIONS").ok().unwrap_or_default();
    if v.trim().is_empty() { return vec![]; }
    v.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).map(|s| if s.starts_with('.') { s.to_string() } else { format!(".{s}") }).collect()
}

pub fn env_custom_ignore_patterns() -> Vec<String> {
    let v = std::env::var("CUSTOM_IGNORE_PATTERNS").ok().unwrap_or_default();
    if v.trim().is_empty() { return vec![]; }
    v.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
}

fn build_globset(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for p in patterns { if let Ok(g) = Glob::new(p) { builder.add(g); } }
    Ok(builder.build()?)
}

fn find_ignore_files(codebase_path: &Path) -> Vec<PathBuf> {
    let mut files = vec![];
    if let Ok(dir) = std::fs::read_dir(codebase_path) {
        for e in dir.flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue; };
            if e.file_type().map(|t| t.is_file()).unwrap_or(false) && name.starts_with('.') && name.ends_with("ignore") { files.push(e.path()); }
        }
    }
    if let Some(home) = dirs::home_dir() { let p = home.join(".context").join(".contextignore"); if p.exists() { files.push(p); } }
    files
}
