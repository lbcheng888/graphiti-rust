use anyhow::{anyhow, Result};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, Command};
use tokio::sync::{oneshot, Mutex, RwLock};
use tokio::time::{timeout, Duration};

#[derive(Clone)]
pub struct RaClient {
    #[allow(dead_code)]
    root: PathBuf,
    stdin: Arc<Mutex<ChildStdin>>,
    pending: Arc<Mutex<HashMap<i64, oneshot::Sender<serde_json::Value>>>>,
    id: Arc<AtomicI64>,
}

static REGISTRY: RwLock<Option<HashMap<PathBuf, Arc<RaClient>>>> = RwLock::const_new(None);

pub async fn list_roots() -> Vec<PathBuf> {
    let reg = REGISTRY.read().await;
    if let Some(map) = reg.as_ref() {
        map.keys().cloned().collect()
    } else {
        vec![]
    }
}

pub async fn shutdown(root: &Path) -> Result<()> {
    let mut reg = REGISTRY.write().await;
    if let Some(map) = reg.as_mut() {
        if let Some(client) = map.remove(root) {
            // LSP graceful shutdown
            let _ = client.request("shutdown", serde_json::Value::Null).await;
            let _ = client.notify("exit", serde_json::Value::Null).await;
        }
    }
    Ok(())
}

pub async fn status(root: Option<&Path>) -> serde_json::Value {
    match root {
        Some(r) => {
            // health check via lightweight workspace/symbol
            let ok = workspace_symbol(r, "").await.is_ok();
            serde_json::json!({"root": r.to_string_lossy(), "ok": ok})
        }
        None => {
            let roots = list_roots().await;
            let roots: Vec<String> = roots
                .into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            serde_json::json!({"roots": roots})
        }
    }
}

fn resolve_ra_binary() -> String {
    // Prefer explicit env variables
    if let Ok(p) = std::env::var("RUST_ANALYZER_BIN") {
        return p;
    }
    if let Ok(p) = std::env::var("RA_BINARY") {
        return p;
    }
    // Fallbacks: PATH search by name
    "rust-analyzer".to_string()
}

pub async fn get_or_start(root: &Path) -> Result<Arc<RaClient>> {
    {
        let reg = REGISTRY.read().await;
        if let Some(map) = reg.as_ref() {
            if let Some(c) = map.get(root) {
                return Ok(c.clone());
            }
        }
    }
    // Try spawn rust-analyzer, allow override via env RUST_ANALYZER_BIN/RA_BINARY
    let ra_bin = resolve_ra_binary();
    let mut child = Command::new(&ra_bin).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).spawn()
        .map_err(|e| anyhow!("rust-analyzer not available in PATH ({}). Please install rust-analyzer or rely on regex-based symbol tools.", e))?;
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;

    let client = Arc::new(RaClient {
        root: root.to_path_buf(),
        stdin: Arc::new(Mutex::new(stdin)),
        pending: Arc::new(Mutex::new(HashMap::new())),
        id: Arc::new(AtomicI64::new(1)),
    });

    // spawn reader
    let pending = client.pending.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        loop {
            // read headers
            let mut content_length: Option<usize> = None;
            loop {
                let mut line = String::new();
                let n = reader.read_line(&mut line).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let t = line.trim_end();
                if t.is_empty() {
                    break;
                }
                let lower = t.to_ascii_lowercase();
                if let Some(rest) = lower.strip_prefix("content-length:") {
                    content_length = rest.trim().parse::<usize>().ok();
                }
            }
            let n = match content_length {
                Some(v) => v,
                None => return,
            };
            let mut buf = vec![0u8; n];
            if reader.read_exact(&mut buf).await.is_err() {
                return;
            }
            let val: serde_json::Value = match serde_json::from_slice(&buf) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(id) = val.get("id").and_then(|v| v.as_i64()) {
                let mut p = pending.lock().await;
                if let Some(tx) = p.remove(&id) {
                    let _ = tx.send(val);
                }
            }
        }
    });

    // initialize
    let root_uri = url::Url::from_file_path(root)
        .map_err(|_| anyhow!("invalid root path"))?
        .to_string();
    let params = json!({
        "processId": null,
        "rootUri": root_uri,
        "capabilities": {},
        "clientInfo": {"name":"graphiti-mcp","version":"0.1"}
    });
    client.request("initialize", params).await?;
    client
        .notify("initialized", json!({"capabilities":{}}))
        .await?;

    {
        let mut reg = REGISTRY.write().await;
        let map = reg.get_or_insert_with(HashMap::new);
        map.insert(root.to_path_buf(), client.clone());
    }
    Ok(client)
}

impl RaClient {
    async fn send_raw(&self, val: &serde_json::Value) -> Result<()> {
        let bytes = serde_json::to_vec(val)?;
        let mut w = self.stdin.lock().await;
        w.write_all(format!("Content-Length: {}\r\n\r\n", bytes.len()).as_bytes())
            .await?;
        w.write_all(&bytes).await?;
        w.flush().await?;
        Ok(())
    }
    pub async fn notify(&self, method: &str, params: serde_json::Value) -> Result<()> {
        let req = json!({"jsonrpc":"2.0","method":method,"params":params});
        self.send_raw(&req).await
    }
    pub async fn request(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let id = self.id.fetch_add(1, Ordering::SeqCst);
        let req = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        let (tx, rx) = oneshot::channel();
        {
            self.pending.lock().await.insert(id, tx);
        }
        self.send_raw(&req).await?;
        let secs = std::env::var("GRAPHITI_RA_REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(10);
        let resp = timeout(Duration::from_secs(secs), rx)
            .await
            .map_err(|_| anyhow!("lsp request timed out ({}s)", secs))?
            .map_err(|_| anyhow!("lsp resp dropped"))?;
        if let Some(err) = resp.get("error") {
            return Err(anyhow!("lsp error: {}", err));
        }
        Ok(resp["result"].clone())
    }
}

fn uri_for(path: &Path) -> Result<String> {
    Ok(url::Url::from_file_path(path)
        .map_err(|_| anyhow!("bad path"))?
        .to_string())
}

pub async fn did_open(root: &Path, file: &Path) -> Result<Arc<RaClient>> {
    let client = get_or_start(root).await?;
    let text = std::fs::read_to_string(file).unwrap_or_default();
    let uri = uri_for(file)?;
    let lang = "rust"; // rust-analyzer
    client
        .notify(
            "textDocument/didOpen",
            json!({
                "textDocument": {"uri": uri, "languageId": lang, "version": 1, "text": text}
            }),
        )
        .await?;
    Ok(client)
}

pub async fn document_symbols(root: &Path, file: &Path) -> Result<serde_json::Value> {
    let client = match did_open(root, file).await {
        Ok(c) => c,
        Err(e) => {
            return Err(anyhow!("rust-analyzer unavailable: {}", e));
        }
    };
    let uri = uri_for(file)?;
    client
        .request(
            "textDocument/documentSymbol",
            json!({"textDocument":{"uri":uri}}),
        )
        .await
}

pub async fn workspace_symbol(root: &Path, query: &str) -> Result<serde_json::Value> {
    let client = get_or_start(root)
        .await
        .map_err(|e| anyhow!("rust-analyzer unavailable: {}", e))?;
    client
        .request("workspace/symbol", json!({"query": query}))
        .await
}

pub async fn references(
    root: &Path,
    file: &Path,
    line: u32,
    character: u32,
) -> Result<serde_json::Value> {
    let client = did_open(root, file)
        .await
        .map_err(|e| anyhow!("rust-analyzer unavailable: {}", e))?;
    let uri = uri_for(file)?;
    client
        .request(
            "textDocument/references",
            json!({
                "textDocument":{"uri":uri},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": false}
            }),
        )
        .await
}

fn pos_in_range(line: u32, character: u32, range: &serde_json::Value) -> bool {
    let s = &range["start"];
    let e = &range["end"];
    let sl = s["line"].as_u64().unwrap_or(0) as u32;
    let sc = s["character"].as_u64().unwrap_or(0) as u32;
    let el = e["line"].as_u64().unwrap_or(0) as u32;
    let ec = e["character"].as_u64().unwrap_or(0) as u32;
    if line < sl || line > el {
        return false;
    }
    if line == sl && character < sc {
        return false;
    }
    if line == el && character > ec {
        return false;
    }
    true
}

pub async fn symbol_at_position(
    root: &Path,
    file: &Path,
    line: u32,
    character: u32,
) -> Result<Option<(String, u64, serde_json::Value)>> {
    let doc = document_symbols(root, file).await?;
    fn walk<'a>(
        arr: &'a serde_json::Value,
        line: u32,
        character: u32,
        out: &mut Option<(&'a serde_json::Value, &'a serde_json::Value, u64)>,
    ) {
        if let Some(items) = arr.as_array() {
            for it in items {
                let range = &it["range"];
                if pos_in_range(line, character, range) {
                    let kind = it["kind"].as_u64().unwrap_or(0);
                    *out = Some((&it["name"], range, kind));
                    if let Some(children) = it.get("children") {
                        walk(children, line, character, out);
                    }
                }
            }
        }
    }
    let mut sel = None;
    walk(&doc, line, character, &mut sel);
    if let Some((name_v, range, kind)) = sel {
        Ok(Some((
            name_v.as_str().unwrap_or("").to_string(),
            kind,
            range.clone(),
        )))
    } else {
        Ok(None)
    }
}

pub async fn definition(
    root: &Path,
    file: &Path,
    line: u32,
    character: u32,
) -> Result<serde_json::Value> {
    let client = did_open(root, file)
        .await
        .map_err(|e| anyhow!("rust-analyzer unavailable: {}", e))?;
    let uri = uri_for(file)?;
    client
        .request(
            "textDocument/definition",
            json!({
                "textDocument":{"uri":uri},
                "position": {"line": line, "character": character}
            }),
        )
        .await
}

pub async fn find_target_symbol_name(
    root: &Path,
    def_uri: &str,
    def_range: &serde_json::Value,
) -> Option<String> {
    if let Ok(url) = url::Url::parse(def_uri) {
        if let Ok(path) = url.to_file_path() {
            if let Ok(doc) = document_symbols(root, &path).await {
                let dl = def_range["start"]["line"].as_u64().unwrap_or(0) as u32;
                let dc = def_range["start"]["character"].as_u64().unwrap_or(0) as u32;
                fn covers(pos_l: u32, pos_c: u32, obj: &serde_json::Value) -> bool {
                    if let Some(sr) = obj.get("selectionRange") {
                        return super::serena_ra::pos_in_range(pos_l, pos_c, sr);
                    }
                    super::serena_ra::pos_in_range(pos_l, pos_c, &obj["range"])
                }
                fn walk<'a>(
                    arr: &'a serde_json::Value,
                    pos_l: u32,
                    pos_c: u32,
                ) -> Option<&'a serde_json::Value> {
                    if let Some(a) = arr.as_array() {
                        // prefer最深层覆盖 selectionRange/range 的符号
                        for it in a {
                            let has = covers(pos_l, pos_c, it);
                            if let Some(ch) = it.get("children") {
                                if let Some(n) = walk(ch, pos_l, pos_c) {
                                    return Some(n);
                                }
                            }
                            if has {
                                return Some(&it["name"]);
                            }
                        }
                    }
                    None
                }
                if let Some(n) = walk(&doc, dl, dc) {
                    return n.as_str().map(|s| s.to_string());
                }
            }
        }
    }
    None
}
