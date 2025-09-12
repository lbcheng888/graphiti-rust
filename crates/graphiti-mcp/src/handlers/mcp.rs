//! MCP protocol handlers

use axum::http::StatusCode;
use axum::response::Json;
use axum::extract::State;
use crate::types::*;
use tracing::{debug, info};
use serde_json::Value as JsonValue;

/// MCP protocol handler
pub async fn mcp_handler(
    State(state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    debug!("MCP request: {}", request);

    // Basic MCP protocol implementation
    if let Some(method) = request.get("method").and_then(|m| m.as_str()) {
        match method {
            "initialize" => {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {
                                "listChanged": false
                            },
                            "resources": {
                                "subscribe": false,
                                "listChanged": false
                            },
                            "prompts": {
                                "listChanged": false
                            }
                        },
                        "serverInfo": {
                            "name": "Graphiti Knowledge Graph",
                            "version": "1.0.0"
                        }
                    }
                });
                Ok(Json(response))
            }
            "initialized" => {
                // This is a notification, no response needed
                info!("MCP client initialized");
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": null
                });
                Ok(Json(response))
            }
             "tools/list" => {
                 let response = serde_json::json!({
                     "jsonrpc": "2.0",
                     "id": request.get("id"),
                     "result": {
                         "tools": [
                            // Serena-bridge tools
                            {"name":"serena_list_dir","description":"List directory entries","inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
                            {"name":"serena_find_file","description":"Find files by glob under root","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"glob":{"type":"string"}},"required":["root","glob"]}},
                            {"name":"serena_read_file","description":"Read file content","inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
                            {"name":"serena_write_file","description":"Write file content","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}},
                            {"name":"serena_search","description":"Search text in files under root","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"pattern":{"type":"string"}},"required":["root","pattern"]}},
                            {"name":"serena_replace_lines","description":"Replace lines in a file [start,end] with content","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"start":{"type":"integer"},"end":{"type":"integer"},"content":{"type":"string"}},"required":["path","start","end","content"]}},
                            {"name":"serena_delete_lines","description":"Delete lines in a file [start,end]","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"start":{"type":"integer"},"end":{"type":"integer"}},"required":["path","start","end"]}},
                            {"name":"serena_insert_at_line","description":"Insert content at line","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"line":{"type":"integer"},"content":{"type":"string"}},"required":["path","line","content"]}},
                            {
                                "name": "ping",
                                "description": "Lightweight health check; returns pong and basic status",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "echo": {"type": "string", "description": "Optional text to echo back"}
                                    },
                                    "required": []
                                }
                            },
                             {
                                 "name": "add_memory",
                                 "description": "Add a new memory to the knowledge graph",
                                 "inputSchema": {
                                     "type": "object",
                                     "properties": {
                                         "content": {"type": "string", "description": "The content to remember"},
                                         "name": {"type": "string", "description": "Optional name/title"},
                                         "source": {"type": "string", "description": "The source of the memory"},
                                         "memory_type": {"type": "string", "description": "Optional memory type"},
                                         "group_id": {"type": "string", "description": "Optional group/namespace id"},
                                         "metadata": {"type": "object", "description": "Optional metadata object"},
                                         "timestamp": {"type": "string", "description": "Event time (ISO 8601)"}
                                     },
                                     "required": ["content"]
                                 }
                            },
                            {"name":"serena_get_current_config","description":"Get project config resolved for current root","inputSchema":{"type":"object","properties":{"project":{"type":"string"}},"required":[]}},
                            {"name":"serena_symbols_overview","description":"Heuristic document symbols (regex)","inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
                            {"name":"serena_find_symbol","description":"Find symbol by name (heuristic)","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"name":{"type":"string"}},"required":["root","name"]}},
                            {"name":"serena_find_referencing_symbols","description":"Find referencing symbols (heuristic)","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"name":{"type":"string"}},"required":["root","name"]}},
                            {"name":"serena_replace_symbol_body","description":"Replace a symbol body (brace/indent heuristics)","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"name":{"type":"string"},"new_body":{"type":"string"}},"required":["path","name","new_body"]}},
                            {"name":"serena_ra_document_symbols","description":"Precise document symbols via rust-analyzer","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"}},"required":["root","path"]}},
                            {"name":"serena_ra_workspace_symbol","description":"Workspace symbol search via rust-analyzer","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"query":{"type":"string"}},"required":["root","query"]}},
                            {"name":"serena_ra_references","description":"Find references via rust-analyzer","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"line":{"type":"integer"},"character":{"type":"integer"}},"required":["root","path","line","character"]}},
                            {"name":"serena_ra_definition","description":"Get definition location via rust-analyzer","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"line":{"type":"integer"},"character":{"type":"integer"}},"required":["root","path","line","character"]}},
                            {"name":"serena_ra_status","description":"Check rust-analyzer status","inputSchema":{"type":"object","properties":{"root":{"type":"string"}},"required":[]}},
                            {"name":"serena_ra_restart","description":"Restart rust-analyzer for a root","inputSchema":{"type":"object","properties":{"root":{"type":"string"}},"required":["root"]}},
                            {"name":"serena_kg_ra_document_symbols_to_entities","description":"Import RA document symbols into Graphiti code entities","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"limit":{"type":"integer"}},"required":["root","path"]}},
                            {"name":"serena_kg_ra_file_symbols_connect","description":"Import file symbols and connect CONTAINS edges","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"limit":{"type":"integer"},"dryRun":{"type":"boolean"}},"required":["root","path"]}},
                            {"name":"serena_kg_ra_workspace_file_symbols_connect","description":"Connect CONTAINS edges for workspace files","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"maxFiles":{"type":"integer"},"symbolLimit":{"type":"integer"},"dryRun":{"type":"boolean"}},"required":["root"]}},
                            {"name":"serena_kg_ra_import_and_connect_file","description":"Import symbols and connect CONTAINS edges for a file","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"limit":{"type":"integer"},"dryRun":{"type":"boolean"}},"required":["root","path"]}},
                            {"name":"serena_kg_ra_build_symbols_and_connect_workspace","description":"Import symbols and connect CONTAINS edges across workspace","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"maxFiles":{"type":"integer"},"symbolLimit":{"type":"integer"},"dryRun":{"type":"boolean"}},"required":["root"]}},
                            {"name":"serena_kg_replace_lines_and_record","description":"Replace lines in a file and record activity","inputSchema":{"type":"object","properties":{"path":{"type":"string"},"start":{"type":"integer"},"end":{"type":"integer"},"content":{"type":"string"},"description":{"type":"string"}},"required":["path","start","end","content"]}},
                            {"name":"serena_kg_graph_summary","description":"Summarize graph nodes/edges (limited)","inputSchema":{"type":"object","properties":{},"required":[]}},
                            {"name":"serena_kg_recent_edges","description":"List recent edges by relationship (limited)","inputSchema":{"type":"object","properties":{"relationship":{"type":"string"},"limit":{"type":"integer"}},"required":["relationship"]}},
                            {"name":"serena_kg_prune_edges","description":"Prune oldest edges to keep at most limit (limited)","inputSchema":{"type":"object","properties":{"limit":{"type":"integer"}},"required":["limit"]}},
                            {"name":"serena_kg_onboarding_done","description":"Mark onboarding completed for project","inputSchema":{"type":"object","properties":{"project":{"type":"string"}},"required":[]}},
                            {"name":"serena_kg_ra_build_workspace","description":"Build symbol graph for workspace (import symbols only)","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"maxFiles":{"type":"integer"},"symbolLimit":{"type":"integer"}},"required":["root"]}},
                            {"name":"serena_kg_ra_build_references_workspace","description":"Build REFERENCES edges across workspace","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"maxFiles":{"type":"integer"},"symbolLimit":{"type":"integer"},"relation":{"type":"string"},"dryRun":{"type":"boolean"}},"required":["root"]}},
                            {"name":"serena_kg_ra_references_to_edges","description":"Create REFERENCES edges from RA references (limited)","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"line":{"type":"integer"},"character":{"type":"integer"},"relation":{"type":"string"},"dryRun":{"type":"boolean"}},"required":["root","path","line","character"]}},
                            {"name":"serena_kg_ra_symbol_definition_edge","description":"Create edge from source symbol to its definition symbol (limited)","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"line":{"type":"integer"},"character":{"type":"integer"},"relation":{"type":"string"}},"required":["root","path","line","character"]}},
                            {"name":"serena_kg_ra_build_references_for_file","description":"Build REFERENCES for file (limited)","inputSchema":{"type":"object","properties":{"root":{"type":"string"},"path":{"type":"string"},"maxSymbols":{"type":"integer"},"relation":{"type":"string"},"dryRun":{"type":"boolean"}},"required":["root","path"]}}
                         ]
                     }
                 });
                 Ok(Json(response))
             }
            "tools/call" => {
                // Minimal, but complete, tools implementation matching tools/list above
                let params = request.get("params").and_then(|p| p.as_object()).ok_or(StatusCode::BAD_REQUEST)?;
                let name = params.get("name").and_then(|n| n.as_str()).ok_or(StatusCode::BAD_REQUEST)?;
                let args: JsonValue = params.get("arguments").cloned().unwrap_or(serde_json::json!({}));

                match name {
                    // Health check
                    "ping" => {
                        #[derive(serde::Deserialize)]
                        struct A { echo: Option<String> }
                        let a: A = serde_json::from_value(args).unwrap_or(A{ echo: None });
                        let response = serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [
                                    {"type":"text","text": format!("pong{}", a.echo.as_deref().map(|s| format!(" ({})", s)).unwrap_or_default())}
                                ]
                            }
                        });
                        Ok(Json(response))
                    }

                    // Knowledge graph memory
                    "add_memory" => {
                        let req: AddMemoryRequest = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        match state.graphiti.add_memory(req).await {
                            Ok(resp) => {
                                let response = serde_json::json!({
                                    "jsonrpc":"2.0",
                                    "id": request.get("id"),
                                    "result": {"content":[{"type":"json","json": serde_json::to_value(resp).unwrap_or(serde_json::json!({}))}]}
                                });
                                Ok(Json(response))
                            }
                            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR)
                        }
                    }

                    // Serena file ops
                    "serena.list_dir" | "serena_list_dir" => {
                        let a: crate::serena_tools::ListDirRequest = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let res = crate::serena_tools::list_dir(a).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                        let response = serde_json::json!({
                            "jsonrpc":"2.0","id":request.get("id"),
                            "result": {"content":[{"type":"json","json": res}]}
                        });
                        Ok(Json(response))
                    }
                    "serena.find_file" | "serena_find_file" => {
                        let a: crate::serena_tools::FindFileRequest = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let res = crate::serena_tools::find_file(a).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                        let response = serde_json::json!({
                            "jsonrpc":"2.0","id":request.get("id"),
                            "result": {"content":[{"type":"json","json": res}]}
                        });
                        Ok(Json(response))
                    }
                    "serena.read_file" | "serena_read_file" => {
                        let a: crate::serena_tools::ReadFileRequest = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let res = crate::serena_tools::read_file(a).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                        let response = serde_json::json!({
                            "jsonrpc":"2.0","id":request.get("id"),
                            "result": {"content":[{"type":"text","text": res.content}]}
                        });
                        Ok(Json(response))
                    }
                    "serena.write_file" | "serena_write_file" => {
                        let a: crate::serena_tools::WriteFileRequest = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        crate::serena_tools::write_file(a).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                        let response = serde_json::json!({
                            "jsonrpc":"2.0","id":request.get("id"),
                            "result": {"content":[{"type":"text","text": "ok"}]}
                        });
                        Ok(Json(response))
                    }
                    "serena.search" | "serena_search" => {
                        let a: crate::serena_tools::SearchRequest = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let res = crate::serena_tools::search(a).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                        let response = serde_json::json!({
                            "jsonrpc":"2.0","id":request.get("id"),
                            "result": {"content":[{"type":"json","json": res}]}
                        });
                        Ok(Json(response))
                    }
                    "serena.replace_lines" | "serena_replace_lines" => {
                        let a: crate::serena_tools::ReplaceLinesRequest = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        crate::serena_tools::replace_lines(a).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                        let response = serde_json::json!({
                            "jsonrpc":"2.0","id":request.get("id"),
                            "result": {"content":[{"type":"text","text": "ok"}]}
                        });
                        Ok(Json(response))
                    }
                    "serena.delete_lines" | "serena_delete_lines" => {
                        let a: crate::serena_tools::DeleteLinesRequest = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        crate::serena_tools::delete_lines(a).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                        let response = serde_json::json!({
                            "jsonrpc":"2.0","id":request.get("id"),
                            "result": {"content":[{"type":"text","text": "ok"}]}
                        });
                        Ok(Json(response))
                    }
                    "serena.insert_at_line" | "serena_insert_at_line" => {
                        let a: crate::serena_tools::InsertAtLineRequest = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        crate::serena_tools::insert_at_line(a).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                        let response = serde_json::json!({
                            "jsonrpc":"2.0","id":request.get("id"),
                            "result": {"content":[{"type":"text","text": "ok"}]}
                        });
                        Ok(Json(response))
                    }

                    // Serena config / symbols (heuristic)
                    "serena.get_current_config" | "serena_get_current_config" => {
                        #[derive(serde::Deserialize)] struct A{ project: Option<String> }
                        let a: A = serde_json::from_value(args).unwrap_or(A{ project: None });
                        let v = crate::handlers::serena::get_current_config(crate::handlers::serena::GetConfigArgs{ project: a.project }).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena.symbols_overview" | "serena_symbols_overview" => {
                        let a: crate::handlers::serena::SymbolsOverviewArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let v = crate::handlers::serena::get_symbols_overview(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena.find_symbol" | "serena_find_symbol" => {
                        let a: crate::handlers::serena::FindSymbolArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let v = crate::handlers::serena::find_symbol(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena.find_referencing_symbols" | "serena_find_referencing_symbols" => {
                        let a: crate::handlers::serena::FindRefsArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let v = crate::handlers::serena::find_referencing_symbols(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena.replace_symbol_body" | "serena_replace_symbol_body" => {
                        let a: crate::handlers::serena::ReplaceSymbolBodyArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let _ = crate::handlers::serena::replace_symbol_body(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": "ok"}]}});
                        Ok(Json(response))
                    }

                    // rust-analyzer bridging
                    "serena.ra.document_symbols" | "serena_ra_document_symbols" => {
                        let a: crate::handlers::serena::RaDocumentSymbolsArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let v = crate::handlers::serena::ra_document_symbols(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena.ra.workspace_symbol" | "serena_ra_workspace_symbol" => {
                        let a: crate::handlers::serena::RaWorkspaceSymbolArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let v = crate::handlers::serena::ra_workspace_symbol(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena.ra.references" | "serena_ra_references" => {
                        let a: crate::handlers::serena::RaReferencesArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let v = crate::handlers::serena::ra_references(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena.ra.definition" | "serena_ra_definition" => {
                        let a: crate::handlers::serena::RaDefinitionArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let v = crate::handlers::serena::ra_definition(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena.ra.status" | "serena_ra_status" => {
                        let a: crate::handlers::serena::RaRootArgs = serde_json::from_value(args).unwrap_or(crate::handlers::serena::RaRootArgs{ root: String::new() });
                        let v = crate::handlers::serena::ra_status(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena.ra.restart" | "serena_ra_restart" => {
                        let a: crate::handlers::serena::RaRootArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let v = crate::handlers::serena::ra_restart(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }

                    // Graph integrations (limited)
                    "serena_kg.ra_document_symbols_to_entities" | "serena_kg_ra_document_symbols_to_entities" => {
                        let a: crate::handlers::serena::RaDocSymbolsToEntitiesArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::ra_document_symbols_to_entities(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.ra_file_symbols_connect" | "serena_kg_ra_file_symbols_connect" => {
                        let a: crate::handlers::serena::RaFileSymbolsConnectArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::ra_file_symbols_connect(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.ra_workspace_file_symbols_connect" | "serena_kg_ra_workspace_file_symbols_connect" => {
                        let a: crate::handlers::serena::RaWorkspaceFileSymbolsConnectArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::ra_workspace_file_symbols_connect(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.ra_import_and_connect_file" | "serena_kg_ra_import_and_connect_file" => {
                        let a: crate::handlers::serena::RaImportAndConnectFileArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::ra_import_and_connect_file(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.ra_build_symbols_and_connect_workspace" | "serena_kg_ra_build_symbols_and_connect_workspace" => {
                        let a: crate::handlers::serena::RaBuildSymbolsAndConnectWorkspaceArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::ra_build_symbols_and_connect_workspace(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.replace_lines_and_record" | "serena_kg_replace_lines_and_record" => {
                        let a: crate::handlers::serena::ReplaceLinesAndRecordArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let _ = crate::handlers::serena::replace_lines_and_record(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": "ok"}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.graph_summary" | "serena_kg_graph_summary" => {
                        let v = crate::handlers::serena::graph_summary(&state).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.recent_edges" | "serena_kg_recent_edges" => {
                        let a: crate::handlers::serena::RecentEdgesArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let v = crate::handlers::serena::recent_edges(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": v}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.prune_edges" | "serena_kg_prune_edges" => {
                        let a: crate::handlers::serena::PruneEdgesArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::prune_edges(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.onboarding_done" | "serena_kg_onboarding_done" => {
                        let a: crate::handlers::serena::OnboardingDoneArgs = serde_json::from_value(args).unwrap_or(crate::handlers::serena::OnboardingDoneArgs{ project: None });
                        let _ = crate::handlers::serena::onboarding_done(a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": "ok"}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.ra_build_workspace" | "serena_kg_ra_build_workspace" => {
                        let a: crate::handlers::serena::RaBuildWorkspaceArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::ra_build_workspace(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.ra_build_references_workspace" | "serena_kg_ra_build_references_workspace" => {
                        let a: crate::handlers::serena::RaBuildReferencesWorkspaceArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::ra_build_references_workspace(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.ra_references_to_edges" | "serena_kg_ra_references_to_edges" => {
                        let a: crate::handlers::serena::RaReferencesToEdgesArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::ra_references_to_edges(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.ra_symbol_definition_edge" | "serena_kg_ra_symbol_definition_edge" => {
                        let a: crate::handlers::serena::RaSymbolDefinitionEdgeArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::ra_symbol_definition_edge(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }
                    "serena_kg.ra_build_references_for_file" | "serena_kg_ra_build_references_for_file" => {
                        let a: crate::handlers::serena::RaBuildReferencesForFileArgs = serde_json::from_value(args).map_err(|_| StatusCode::BAD_REQUEST)?;
                        let msg = crate::handlers::serena::ra_build_references_for_file(&state, a).await?;
                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": msg}]}});
                        Ok(Json(response))
                    }

                    _ => {
                        let response = serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {"code": -32601, "message": format!("Unknown tool: {}", name)}
                        });
                        Ok(Json(response))
                    }
                }
            }
            _ => {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
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
