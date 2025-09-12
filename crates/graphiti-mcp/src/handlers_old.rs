use axum::http::StatusCode;
use axum::response::Json;

pub mod serena;
use axum::extract::{State, Query, Path};
use crate::types::*;
use tracing::{error, debug};
use uuid::Uuid;

/// MCP 协议处理器（从 main.rs 抽出）
pub async fn mcp_handler(
    State(_state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    debug!("MCP request: {}", request);
    if let Some(method) = request.get("method").and_then(|m| m.as_str()) {
        match method {
            "tools/list" => Ok(tools_list_response(request.get("id").unwrap_or(&serde_json::Value::Null))),
            _ => {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {"code": -32601, "message": "Method not found"}
                });
                Ok(Json(response))
            }
        }
    } else {
        let response = serde_json::json!({
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": -32600, "message": "Invalid request"}
        });
        Ok(Json(response))
    }
}

pub fn tools_list_response(id: &serde_json::Value) -> Json<serde_json::Value> {
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "tools": [
                {"name":"serena.list_dir","description":"List directory entries","inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
                {"name":"serena.find_file","description":"Find files by glob under root","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"glob":{"type":"string"}},"required":["root","glob"]}},
                {"name":"serena.read_file","description":"Read file content","inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
                {"name":"serena.write_file","description":"Write file content","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}},
                {"name":"serena.search","description":"Search text in files under root","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"pattern":{"type":"string"}},"required":["root","pattern"]}},
                {"name":"serena.replace_lines","description":"Replace lines in a file [start,end] with content","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"start":{"type":"integer"},"end":{"type":"integer"},"content":{"type":"string"}},"required":["path","start","end","content"]}},
                {"name":"serena.delete_lines","description":"Delete lines in a file [start,end]","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"start":{"type":"integer"},"end":{"type":"integer"}},"required":["path","start","end"]}},
                {"name":"serena.insert_at_line","description":"Insert content at line","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"line":{"type":"integer"},"content":{"type":"string"}},"required":["path","line","content"]}},
                {"name":"serena.get_current_config","description":"Get project config resolved for current root","inputSchema":{"type":"object","properties":{"project":{"type":"string"}},"required":[]}},
                {"name":"serena_kg.replace_lines_and_record","description":"Replace lines in a file and record activity into Graphiti","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"start":{"type":"integer"},"end":{"type":"integer"},"content":{"type":"string"},"description":{"type":"string"}},"required":["path","start","end","content"]}},
                {"name":"serena.ra.document_symbols","description":"Precise document symbols via rust-analyzer","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"}},"required":["root","path"]}},
                {"name":"serena.ra.workspace_symbol","description":"Workspace symbol search via rust-analyzer","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"query":{"type":"string"}},"required":["root","query"]}},
                {"name":"serena.ra.references","description":"Find references via rust-analyzer","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"line":{"type":"integer"},"character":{"type":"integer"}},"required":["root","path","line","character"]}},
                {"name":"serena_kg.ra_document_symbols_to_entities","description":"Import RA document symbols into Graphiti code entities","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"limit":{"type":"integer"}},"required":["root","path"]}},
                {"name":"serena_kg.ra_references_to_edges","description":"Create REFERENCES edges from RA references (symbol/file level)","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"line":{"type":"integer"},"character":{"type":"integer"},"relation":{"type":"string"}},"required":["root","path","line","character"]}},
                {"name":"serena_kg.ra_file_symbols_connect","description":"Import file symbols and connect CONTAINS edges","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"limit":{"type":"integer"}},"required":["root","path"]}},
                {"name":"serena_kg.ra_build_workspace","description":"Build symbol graph for workspace (throttled)","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"maxFiles":{"type":"integer"},"symbolLimit":{"type":"integer"}},"required":["root"]}},
                {"name":"serena_kg.ra_symbol_definition_edge","description":"Create edge from source symbol to its definition symbol","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"line":{"type":"integer"},"character":{"type":"integer"},"relation":{"type":"string"}},"required":["root","path","line","character"]}},
                {"name":"serena_kg.ra_build_references_for_file","description":"Build REFERENCES edges for symbols in a file","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"maxSymbols":{"type":"integer"},"relation":{"type":"string"}},"required":["root","path"]}},
                {"name":"serena_kg.graph_summary","description":"Summarize graph nodes/edges","inputSchema":{"type":"object","properties":{},"required":[]}},
                {"name":"serena_kg.recent_edges","description":"List recent edges by relationship","inputSchema":{"type":"object","properties":{"relationship":{"type":"string"},"limit":{"type":"integer"}},"required":["relationship"]}},
                {"name":"serena_kg.prune_edges","description":"Prune oldest edges to keep at most limit","inputSchema":{"type":"object","properties":{"limit":{"type":"integer"}},"required":["limit"]}},
                {"name":"serena_kg.ra_build_references_workspace","description":"Build REFERENCES edges across workspace (throttled)","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"maxFiles":{"type":"integer"},"symbolLimit":{"type":"integer"},"relation":{"type":"string"}},"required":["root"]}},
                {"name":"serena.ra.definition","description":"Get definition location via rust-analyzer","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"line":{"type":"integer"},"character":{"type":"integer"}},"required":["root","path","line","character"]}},
                {"name":"serena.ra.status","description":"Check rust-analyzer status","inputSchema":{"type":"object","properties":{"root":{"type":"string"}},"required":[]}},
                {"name":"serena.ra.restart","description":"Restart rust-analyzer for a root","inputSchema":{"type":"object","properties":{"root":{"type":"string"}},"required":["root"]}}
            ]
        }
    });
    Json(response)
}

/// 健康检查和业务 HTTP 处理（从 main.rs 抽出精简版）
pub async fn health_check() -> StatusCode { StatusCode::OK }

pub async fn metrics() -> String { "".to_string() }

pub async fn add_memory(
    State(state): State<AppState>,
    Json(req): Json<AddMemoryRequest>,
) -> std::result::Result<Json<AddMemoryResponse>, StatusCode> {
    match state.graphiti.add_memory(req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => { error!("Failed to add memory: {}", e); Err(StatusCode::INTERNAL_SERVER_ERROR) }
    }
}

pub async fn search_memory(
    State(state): State<AppState>,
    Query(req): Query<SearchMemoryRequest>,
) -> std::result::Result<Json<SearchMemoryResponse>, StatusCode> {
    match state.graphiti.search_memory(req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => { error!("Failed to search memory: {}", e); Err(StatusCode::INTERNAL_SERVER_ERROR) }
    }
}

pub async fn search_memory_json(
    State(state): State<AppState>,
    Json(req): Json<SearchMemoryRequest>,
) -> std::result::Result<Json<SearchMemoryResponse>, StatusCode> {
    match state.graphiti.search_memory(req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => { error!("Failed to search memory (json): {}", e); Err(StatusCode::INTERNAL_SERVER_ERROR) }
    }
}

pub async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> std::result::Result<Json<MemoryNode>, StatusCode> {
    match state.graphiti.get_memory(id).await {
        Ok(Some(memory)) => Ok(Json(memory)),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(e) => { error!("Failed to get memory: {}", e); Err(StatusCode::INTERNAL_SERVER_ERROR) }
    }
}

pub async fn get_related(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> std::result::Result<Json<Vec<RelatedMemory>>, StatusCode> {
    let depth = params.get("depth").and_then(|d| d.parse().ok()).unwrap_or(1);
    match state.graphiti.get_related(id, depth).await {
        Ok(related) => Ok(Json(related)),
        Err(e) => { error!("Failed to get related memories: {}", e); Err(StatusCode::INTERNAL_SERVER_ERROR) }
    }
}
