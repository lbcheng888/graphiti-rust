use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SerenaProjectConfig {
    pub active_project: Option<String>,
    pub modes: Vec<String>,
    pub onboarding_performed: bool,
}

fn config_path(project_root: &Path) -> PathBuf {
    project_root.join(".graphiti").join("serena-config.json")
}

pub fn load_or_default(project_root: &Path) -> SerenaProjectConfig {
    let p = config_path(project_root);
    match std::fs::read_to_string(&p) {
        Ok(txt) => serde_json::from_str(&txt).unwrap_or_default(),
        Err(_) => SerenaProjectConfig::default(),
    }
}

pub fn save(project_root: &Path, cfg: &SerenaProjectConfig) -> Result<()> {
    if let Some(parent) = config_path(project_root).parent() { std::fs::create_dir_all(parent)?; }
    let txt = serde_json::to_string_pretty(cfg)?;
    std::fs::write(config_path(project_root), txt)?;
    Ok(())
}

pub fn detect_project_root(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit { return Ok(p.to_path_buf()); }
    let cwd = std::env::current_dir()?;
    Ok(cwd)
}

