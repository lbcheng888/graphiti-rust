use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::warn;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SerenaProjectConfig {
    pub active_project: Option<String>,
    pub modes: Vec<String>,
    pub onboarding_performed: bool,
}

fn config_path(project_root: &Path) -> PathBuf {
    project_root.join(".graphiti").join("serena-config.json")
}

pub fn load_or_default(project_root: &Path) -> Result<SerenaProjectConfig> {
    let p = config_path(project_root);
    match std::fs::read_to_string(&p) {
        Ok(txt) => match serde_json::from_str(&txt) {
            Ok(cfg) => Ok(cfg),
            Err(err) => {
                warn!(path = %p.display(), error = %err, "Serena 配置解析失败，返回默认值");
                Ok(SerenaProjectConfig::default())
            }
        },
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Ok(SerenaProjectConfig::default())
        }
        Err(err) => Err(err).with_context(|| format!("读取配置文件失败: {}", p.display())),
    }
}

pub fn save(project_root: &Path, cfg: &SerenaProjectConfig) -> Result<()> {
    if let Some(parent) = config_path(project_root).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let txt = serde_json::to_string_pretty(cfg)?;
    std::fs::write(config_path(project_root), txt)?;
    Ok(())
}

pub fn detect_project_root(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        let canonical = std::fs::canonicalize(p)
            .with_context(|| format!("无法定位项目路径: {}", p.display()))?;
        return Ok(canonical);
    }
    let cwd = std::env::current_dir().context("无法获取当前工作目录")?;
    std::fs::canonicalize(&cwd).context("无法解析当前工作目录")
}
