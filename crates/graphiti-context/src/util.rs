use std::path::Path;

pub fn env_flag(name: &str) -> Option<bool> {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.to_ascii_lowercase();
            if v == "true" || v == "1" || v == "yes" {
                Some(true)
            } else if v == "false" || v == "0" || v == "no" {
                Some(false)
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

pub fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}
