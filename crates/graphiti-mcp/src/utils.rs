// Utility helpers extracted from the monolithic main module.

use axum::http::HeaderValue;
use tower_http::cors::{Any, CorsLayer};

pub fn build_cors_layer(server: &crate::types::ServerSettings) -> CorsLayer {
    let mut layer = CorsLayer::new()
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        // In production, restrict headers to common ones
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
            axum::http::header::ACCEPT,
        ]);

    if let Some(origins) = &server.allowed_origins {
        if origins.iter().any(|o| o == "*") {
            layer = layer.allow_origin(Any);
        } else {
            let values: Vec<HeaderValue> = origins
                .iter()
                .filter_map(|o| HeaderValue::from_str(o).ok())
                .collect();
            if !values.is_empty() {
                layer = layer.allow_origin(values);
            } else {
                layer = layer.allow_origin(Any);
            }
        }
    } else {
        // Default to any in absence of explicit configuration
        layer = layer.allow_origin(Any);
    }

    layer
}

pub fn collection_name(codebase_path: &std::path::Path) -> String {
    let normalized = codebase_path
        .canonicalize()
        .unwrap_or_else(|_| codebase_path.to_path_buf());
    let mut hasher = md5::Md5::new();
    use md5::Digest as _;
    hasher.update(normalized.to_string_lossy().as_bytes());
    let digest = hasher.finalize();
    let hash = format!("{:x}", digest);
    format!("code_chunks_{}", &hash[..8])
}

pub fn truncate(s: &str) -> String {
    let s = s.trim();
    if s.len() > 500 {
        format!("{}...", &s[..500])
    } else {
        s.to_string()
    }
}

pub fn language_from_ext(ext: &str) -> &str {
    match ext {
        "rs" => "Rust",
        "py" => "Python",
        "js" => "JavaScript",
        "ts" | "tsx" => "TypeScript",
        "jsx" => "JavaScript",
        "java" => "Java",
        "go" => "Go",
        "kt" | "kts" => "Kotlin",
        "swift" => "Swift",
        "nim" | "nims" => "Nim",
        "cpp" | "cc" | "c" | "h" | "hpp" => "C/C++",
        _ => "Unknown",
    }
}

pub fn chunk_file(text: &str, rel_path: &str) -> Vec<(String, usize, usize, String)> {
    // Simple line-based chunking: window=120 lines, stride=100
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let win = 120usize;
    let stride = 100usize;
    let mut start = 0usize;
    while start < lines.len() {
        let end = (start + win).min(lines.len());
        let chunk = lines[start..end].join("\n");
        if !chunk.trim().is_empty() {
            let id = format!("{}:{}-{}", rel_path, start + 1, end);
            out.push((id, start + 1, end, chunk));
        }
        if end == lines.len() {
            break;
        }
        start = start.saturating_add(stride);
    }
    out
}
