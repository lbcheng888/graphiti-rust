use crate::types::{AddCodeEntityRequest, AddMemoryRequest, AppState};
use axum::http::StatusCode;
use chrono::Utc;
use graphiti_core::code_entities::CodeEntityType;
use graphiti_core::graph::{Edge, EntityNode, TemporalMetadata};
use graphiti_core::storage::{Direction, GraphStorage};
use metrics::counter;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tracing::error;
use uuid::Uuid;

#[allow(dead_code)]
fn parallelism(default: usize) -> usize {
    if let Ok(s) = std::env::var("GRAPHITI_RA_CONCURRENCY") {
        if let Ok(n) = s.parse::<usize>() {
            return n.clamp(1, 128);
        }
    }
    default
}

async fn edge_exists(
    state: &AppState,
    source: Uuid,
    target: Uuid,
    relation: &str,
) -> Result<bool, StatusCode> {
    let edges = state
        .storage
        .get_edges(&source, Direction::Outgoing)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(edges
        .iter()
        .any(|e| e.target_id == target && e.relationship == relation))
}

// 将 main.rs 中 Serena 相关工具调用的处理逻辑逐步迁移至此模块。
// 先提供占位实现，后续按实际逻辑从 main.rs 分支迁移过来。

#[derive(Deserialize)]
pub struct GetConfigArgs {
    pub project: Option<String>,
}

pub async fn get_current_config(args: GetConfigArgs) -> Result<serde_json::Value, StatusCode> {
    let project = args.project.as_deref();
    let root =
        crate::serena_config::detect_project_root(project.map(Path::new)).map_err(|err| {
            if let Some(project_path) = project {
                error!(%project_path, error = %err, "无法解析项目根目录");
                StatusCode::BAD_REQUEST
            } else {
                error!(error = %err, "无法自动探测项目根目录");
                StatusCode::INTERNAL_SERVER_ERROR
            }
        })?;

    let cfg = crate::serena_config::load_or_default(&root).map_err(|err| {
        error!(error = %err, "读取项目配置失败");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    serde_json::to_value(cfg).map_err(|err| {
        error!(error = %err, "序列化项目配置失败");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

#[derive(Deserialize)]
pub struct ReplaceLinesAndRecordArgs {
    pub path: String,
    pub start: usize,
    pub end: usize,
    pub content: String,
    pub description: Option<String>,
}

pub async fn replace_lines_and_record(
    args: ReplaceLinesAndRecordArgs,
) -> Result<&'static str, StatusCode> {
    // 1) 应用文件修改
    crate::serena_tools::replace_lines(crate::serena_tools::ReplaceLinesRequest {
        path: args.path.clone().into(),
        start: args.start,
        end: args.end,
        content: args.content.clone(),
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // 2) 暂不记录到 Graphiti（需要 AppState/GraphitiService），先返回 ok
    Ok("ok")
}

// ---- Serena Symbols (heuristic) ----

#[derive(Deserialize)]
pub struct SymbolsOverviewArgs {
    pub path: String,
}

pub async fn get_symbols_overview(
    args: SymbolsOverviewArgs,
) -> Result<serde_json::Value, StatusCode> {
    let syms = crate::serena_symbols::list_symbols(Path::new(&args.path))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(serde_json::to_value(syms).unwrap_or(serde_json::json!({})))
}

#[derive(Deserialize)]
pub struct FindSymbolArgs {
    pub root: String,
    pub name: String,
}

pub async fn find_symbol(args: FindSymbolArgs) -> Result<serde_json::Value, StatusCode> {
    let hits = crate::serena_symbols::find_symbol(&crate::serena_symbols::FindSymbolRequest {
        root: args.root.into(),
        name: args.name,
    })
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(serde_json::to_value(hits).unwrap_or(serde_json::json!([])))
}

// ---- Rust Analyzer proxy ----

#[derive(Deserialize)]
pub struct RaDocumentSymbolsArgs {
    pub root: String,
    pub path: String,
}
#[derive(Deserialize)]
pub struct RaWorkspaceSymbolArgs {
    pub root: String,
    pub query: String,
}
#[derive(Deserialize)]
pub struct RaReferencesArgs {
    pub root: String,
    pub path: String,
    pub line: usize,
    pub character: usize,
}
#[derive(Deserialize)]
pub struct RaDefinitionArgs {
    pub root: String,
    pub path: String,
    pub line: usize,
    pub character: usize,
}
#[derive(Deserialize)]
pub struct RaRootArgs {
    pub root: String,
}

pub async fn ra_document_symbols(
    args: RaDocumentSymbolsArgs,
) -> Result<serde_json::Value, StatusCode> {
    match crate::serena_ra::document_symbols(Path::new(&args.root), Path::new(&args.path)).await {
        Ok(v) => Ok(v),
        Err(e) => {
            let syms = crate::serena_symbols::list_symbols(Path::new(&args.path))
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(serde_json::json!({"fallback":"regex","symbols": syms, "error": e.to_string()}))
        }
    }
}
pub async fn ra_workspace_symbol(
    args: RaWorkspaceSymbolArgs,
) -> Result<serde_json::Value, StatusCode> {
    crate::serena_ra::workspace_symbol(Path::new(&args.root), &args.query)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}
pub async fn ra_references(args: RaReferencesArgs) -> Result<serde_json::Value, StatusCode> {
    crate::serena_ra::references(
        Path::new(&args.root),
        Path::new(&args.path),
        args.line as u32,
        args.character as u32,
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}
pub async fn ra_definition(args: RaDefinitionArgs) -> Result<serde_json::Value, StatusCode> {
    crate::serena_ra::definition(
        Path::new(&args.root),
        Path::new(&args.path),
        args.line as u32,
        args.character as u32,
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}
pub async fn ra_status(args: RaRootArgs) -> Result<serde_json::Value, StatusCode> {
    Ok(crate::serena_ra::status(Some(Path::new(&args.root))).await)
}
pub async fn ra_restart(args: RaRootArgs) -> Result<serde_json::Value, StatusCode> {
    let _ = crate::serena_ra::shutdown(Path::new(&args.root)).await;
    Ok(serde_json::json!({"ok": true}))
}

// ---- Graph utilities (placeholders; to be wired to storage via AppState) ----

#[derive(Deserialize)]
pub struct RecentEdgesArgs {
    pub relationship: String,
    pub limit: Option<usize>,
}
#[derive(Deserialize)]
pub struct PruneEdgesArgs {
    pub limit: usize,
}

pub async fn graph_summary(state: &AppState) -> Result<serde_json::Value, StatusCode> {
    let nodes = state
        .storage
        .get_all_nodes()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let edges = state
        .storage
        .get_all_edges()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let files = nodes
        .iter()
        .filter(|n| n.properties().get("node_type").and_then(|v| v.as_str()) == Some("file"))
        .count();
    let entities = nodes.len().saturating_sub(files);
    Ok(serde_json::json!({
        "nodes": nodes.len(),
        "edges": edges.len(),
        "files": files,
        "entities": entities
    }))
}
pub async fn recent_edges(
    state: &AppState,
    args: RecentEdgesArgs,
) -> Result<serde_json::Value, StatusCode> {
    let list = state
        .storage
        .get_recent_edges_by_rel(&args.relationship, args.limit.unwrap_or(20))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let json = list
        .into_iter()
        .map(|e| {
            serde_json::json!({
                "id": e.id,
                "source_id": e.source_id,
                "target_id": e.target_id,
                "relationship": e.relationship,
                "properties": e.properties,
                "created_at": e.temporal.created_at
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!(json))
}
pub async fn prune_edges(state: &AppState, args: PruneEdgesArgs) -> Result<String, StatusCode> {
    let n = state
        .storage
        .prune_to_limit(args.limit)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(format!("Pruned {} edges", n))
}

// ---- Additional Serena symbol helpers ----
#[derive(Deserialize)]
pub struct FindRefsArgs {
    pub root: String,
    pub name: String,
}

pub async fn find_referencing_symbols(args: FindRefsArgs) -> Result<serde_json::Value, StatusCode> {
    let hits = crate::serena_symbols::find_references(&crate::serena_symbols::FindRefsRequest {
        root: args.root.into(),
        name: args.name,
    })
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(serde_json::to_value(hits).unwrap_or(serde_json::json!([])))
}

#[derive(Deserialize)]
pub struct ReplaceSymbolBodyArgs {
    pub path: String,
    pub name: String,
    pub new_body: String,
}

pub async fn replace_symbol_body(args: ReplaceSymbolBodyArgs) -> Result<&'static str, StatusCode> {
    crate::serena_symbols::replace_symbol_body(&crate::serena_symbols::ReplaceSymbolBodyRequest {
        path: args.path.into(),
        name: args.name,
        new_body: args.new_body,
    })
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok("ok")
}

// ---- RA -> Graphiti integrations (best-effort, limited by GraphitiService) ----

#[derive(Deserialize)]
pub struct RaDocSymbolsToEntitiesArgs {
    pub root: String,
    pub path: String,
    pub limit: Option<usize>,
}

pub async fn ra_document_symbols_to_entities(
    state: &AppState,
    args: RaDocSymbolsToEntitiesArgs,
) -> Result<String, StatusCode> {
    let res = ra_document_symbols(RaDocumentSymbolsArgs {
        root: args.root.clone(),
        path: args.path.clone(),
    })
    .await?;
    fn collect(v: &serde_json::Value, out: &mut Vec<(String, (u32, u32))>) {
        if let Some(arr) = v.as_array() {
            for it in arr {
                collect(it, out);
            }
            return;
        }
        if let Some(o) = v.as_object() {
            let name = o
                .get("name")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let r = o.get("range");
            let (s, e) = if let Some(rr) = r.and_then(|r| r.get("start").zip(r.get("end"))) {
                (
                    (rr.0.get("line").and_then(|x| x.as_u64()).unwrap_or(0) as u32) + 1,
                    (rr.1.get("line").and_then(|x| x.as_u64()).unwrap_or(0) as u32) + 1,
                )
            } else {
                (1, 1)
            };
            if !name.is_empty() {
                out.push((name, (s, e)));
            }
            if let Some(ch) = o.get("children") {
                collect(ch, out);
            }
        }
    }
    let mut syms = Vec::new();
    collect(&res, &mut syms);
    if let Some(l) = args.limit {
        syms.truncate(l);
    }

    let path = args.path.clone();
    let mut imported = 0usize;
    for (name, (s, e)) in syms.into_iter() {
        let req = AddCodeEntityRequest {
            entity_type: "function".into(),
            name,
            description: format!("{}:{}-{}", path, s, e),
            file_path: Some(path.clone()),
            line_range: Some((s, e)),
            language: Some("rust".into()),
            framework: None,
            complexity: None,
            importance: None,
        };
        let _ = state.graphiti.add_code_entity(req).await;
        imported += 1;
    }
    Ok(format!("imported ~{} symbols", imported))
}

#[derive(Deserialize)]
pub struct RaBuildWorkspaceArgs {
    pub root: String,
    #[serde(rename = "maxFiles")]
    pub max_files: Option<usize>,
    #[serde(rename = "symbolLimit")]
    pub symbol_limit: Option<usize>,
}

pub async fn ra_build_workspace(
    state: &AppState,
    args: RaBuildWorkspaceArgs,
) -> Result<String, StatusCode> {
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(&args.root).into_iter().flatten() {
        let p = entry.path();
        if p.is_file() {
            if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                if ext == "rs" {
                    files.push(p.to_path_buf());
                }
            }
        }
    }
    if let Some(m) = args.max_files {
        files.truncate(m);
    }
    let mut imported = 0usize;
    for f in files {
        let path = f.to_string_lossy().to_string();
        let txt = ra_document_symbols_to_entities(
            state,
            RaDocSymbolsToEntitiesArgs {
                root: args.root.clone(),
                path: path.clone(),
                limit: args.symbol_limit,
            },
        )
        .await?;
        if let Some(n) = txt
            .split_whitespace()
            .find_map(|w| w.trim_start_matches('~').parse::<usize>().ok())
        {
            imported += n;
        }
    }
    let _ = state
        .graphiti
        .add_memory(AddMemoryRequest {
            content: format!("RA Workspace Build: imported ~{} symbols", imported),
            name: Some("RA Workspace Build".into()),
            source: Some("serena_kg".into()),
            memory_type: Some("audit".into()),
            metadata: None,
            group_id: None,
            timestamp: None,
        })
        .await;
    Ok(format!("Workspace build: imported ~{} symbols", imported))
}

#[derive(Deserialize)]
pub struct RaFileSymbolsConnectArgs {
    pub root: String,
    pub path: String,
    pub limit: Option<usize>,
    #[serde(rename = "dryRun")]
    pub dry_run: Option<bool>,
}

pub async fn ra_file_symbols_connect(
    state: &AppState,
    args: RaFileSymbolsConnectArgs,
) -> Result<String, StatusCode> {
    // 1) Get document symbols
    let res = ra_document_symbols(RaDocumentSymbolsArgs {
        root: args.root.clone(),
        path: args.path.clone(),
    })
    .await?;
    fn collect(v: &serde_json::Value, out: &mut Vec<(String, (u32, u32))>) {
        if let Some(arr) = v.as_array() {
            for it in arr {
                collect(it, out);
            }
            return;
        }
        if let Some(o) = v.as_object() {
            let name = o
                .get("name")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string();
            let r = o.get("range");
            let (s, e) = if let Some(rr) = r.and_then(|r| r.get("start").zip(r.get("end"))) {
                (
                    (rr.0.get("line").and_then(|x| x.as_u64()).unwrap_or(0) as u32) + 1,
                    (rr.1.get("line").and_then(|x| x.as_u64()).unwrap_or(0) as u32) + 1,
                )
            } else {
                (1, 1)
            };
            if !name.is_empty() {
                out.push((name, (s, e)));
            }
            if let Some(ch) = o.get("children") {
                collect(ch, out);
            }
        }
    }
    let mut syms = Vec::new();
    collect(&res, &mut syms);
    if let Some(l) = args.limit {
        syms.truncate(l);
    }

    // 2) Build indexes over existing nodes
    let all_nodes = state
        .storage
        .get_all_nodes()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut file_index: std::collections::HashMap<String, Uuid> = std::collections::HashMap::new();
    let mut symbol_index: std::collections::HashMap<(String, String, (u32, u32)), Uuid> =
        std::collections::HashMap::new();
    for n in all_nodes {
        let p = n.properties();
        if let Some(fp) = p.get("file_path").and_then(|v| v.as_str()) {
            let is_ce = p
                .get("code_entity")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if is_ce {
                let name = p
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let lr = p
                    .get("line_range")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| {
                        if arr.len() == 2 {
                            Some((
                                arr[0].as_u64().unwrap_or(1) as u32,
                                arr[1].as_u64().unwrap_or(1) as u32,
                            ))
                        } else {
                            None
                        }
                    });
                if let Some(r) = lr {
                    symbol_index.insert((fp.to_string(), name, r), *n.id());
                }
            } else {
                if p.get("node_type").and_then(|v| v.as_str()) == Some("file") {
                    file_index.insert(fp.to_string(), *n.id());
                }
            }
        }
    }

    let now = Utc::now();
    let temporal = TemporalMetadata {
        created_at: now,
        valid_from: now,
        valid_to: None,
        expired_at: None,
    };

    // Ensure file node
    let file_id = *file_index.get(&args.path).unwrap_or(&{
        let id = Uuid::new_v4();
        let node = EntityNode {
            id,
            name: args.path.clone(),
            entity_type: CodeEntityType::Module.to_string(),
            labels: vec!["File".into()],
            properties: serde_json::json!({"file_path": args.path, "node_type":"file"}),
            temporal: temporal.clone(),
            embedding: None,
        };
        if !args.dry_run.unwrap_or(false) {
            let _ = state.storage.create_node(&node);
            counter!("graph_nodes_created_total", "node_type" => "File").increment(1);
        }
        id
    });

    // 3) Connect edges
    let mut connected = 0usize;
    let mut seen = std::collections::HashSet::new();
    // existing CONTAINS targets cache
    let mut existing_targets: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    if let Ok(edges) = state.storage.get_edges(&file_id, Direction::Outgoing).await {
        for e in edges {
            if e.relationship == "CONTAINS" {
                existing_targets.insert(e.target_id);
            }
        }
    }
    for (name, (s, e)) in syms {
        let sym_id_opt = symbol_index
            .get(&(args.path.clone(), name.clone(), (s, e)))
            .cloned();
        // If symbol exists, check duplication; else, in dry-run treat as creatable
        if let Some(sym_id) = sym_id_opt {
            if seen.insert((file_id, sym_id)) {
                if existing_targets.contains(&sym_id)
                    || edge_exists(state, file_id, sym_id, "CONTAINS")
                        .await
                        .unwrap_or(false)
                {
                    counter!("graph_edges_skipped_total", "reason" => "duplicate", "relation" => "CONTAINS").increment(1);
                } else if args.dry_run.unwrap_or(false) {
                    connected += 1;
                } else {
                    let edge = Edge {
                        id: Uuid::new_v4(),
                        source_id: file_id,
                        target_id: sym_id,
                        relationship: "CONTAINS".into(),
                        properties: serde_json::json!({"source":"rust-analyzer"}),
                        temporal: temporal.clone(),
                        weight: 1.0,
                    };
                    state
                        .storage
                        .create_edge(&edge)
                        .await
                        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                    counter!("graph_edges_created_total", "relation" => "CONTAINS").increment(1);
                    existing_targets.insert(sym_id);
                    connected += 1;
                }
            }
        } else {
            // symbol missing: in dry-run, count; else create symbol node and connect
            if args.dry_run.unwrap_or(false) {
                connected += 1;
            } else {
                let id = Uuid::new_v4();
                let node = EntityNode {
                    id,
                    name: name.clone(),
                    entity_type: CodeEntityType::Function.to_string(),
                    labels: vec!["CodeEntity".into()],
                    properties: serde_json::json!({"code_entity":true,"name":name,"entity_type":"Function","file_path":args.path,"line_range":[s,e]}),
                    temporal: temporal.clone(),
                    embedding: None,
                };
                state
                    .storage
                    .create_node(&node)
                    .await
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                counter!("graph_nodes_created_total", "node_type" => "CodeEntity").increment(1);
                let edge = Edge {
                    id: Uuid::new_v4(),
                    source_id: file_id,
                    target_id: id,
                    relationship: "CONTAINS".into(),
                    properties: serde_json::json!({"source":"rust-analyzer"}),
                    temporal: temporal.clone(),
                    weight: 1.0,
                };
                state
                    .storage
                    .create_edge(&edge)
                    .await
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                counter!("graph_edges_created_total", "relation" => "CONTAINS").increment(1);
                existing_targets.insert(id);
                connected += 1;
            }
        }
    }
    Ok(format!("connected ~{} CONTAINS edges", connected))
}
#[derive(Deserialize)]
pub struct RaBuildReferencesWorkspaceArgs {
    pub root: String,
    #[serde(rename = "maxFiles")]
    pub max_files: Option<usize>,
    #[serde(rename = "symbolLimit")]
    pub symbol_limit: Option<usize>,
    pub relation: Option<String>,
    #[serde(rename = "dryRun")]
    pub dry_run: Option<bool>,
}

pub async fn ra_build_references_workspace(
    state: &AppState,
    args: RaBuildReferencesWorkspaceArgs,
) -> Result<String, StatusCode> {
    // enumerate Rust files
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(&args.root).into_iter().flatten() {
        let p = entry.path();
        if p.is_file() {
            if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                if ext == "rs" {
                    files.push(p.to_path_buf());
                }
            }
        }
    }
    if let Some(m) = args.max_files {
        files.truncate(m);
    }
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(6));
    let mut tasks = Vec::new();
    for f in files {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let s = state.clone();
        let root = args.root.clone();
        let relation = args.relation.clone();
        let sym_limit = args.symbol_limit;
        let path = f.to_string_lossy().to_string();
        tasks.push(tokio::spawn(async move {
            let _p = permit;
            let msg = ra_build_references_for_file(
                &s,
                RaBuildReferencesForFileArgs {
                    root,
                    path,
                    max_symbols: sym_limit,
                    relation,
                    dry_run: args.dry_run,
                },
            )
            .await;
            match msg {
                Ok(txt) => txt
                    .split_whitespace()
                    .find_map(|w| w.parse::<usize>().ok())
                    .unwrap_or(0),
                Err(_) => 0,
            }
        }));
    }
    let mut total_created = 0usize;
    for t in tasks {
        if let Ok(n) = t.await {
            total_created += n;
        }
    }
    Ok(format!(
        "created ~{} reference edges across workspace",
        total_created
    ))
}

#[derive(Deserialize)]
pub struct RaWorkspaceFileSymbolsConnectArgs {
    pub root: String,
    #[serde(rename = "maxFiles")]
    pub max_files: Option<usize>,
    #[serde(rename = "symbolLimit")]
    pub symbol_limit: Option<usize>,
    #[serde(rename = "dryRun")]
    pub dry_run: Option<bool>,
}

pub async fn ra_workspace_file_symbols_connect(
    state: &AppState,
    args: RaWorkspaceFileSymbolsConnectArgs,
) -> Result<String, StatusCode> {
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(&args.root).into_iter().flatten() {
        let p = entry.path();
        if p.is_file() {
            if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                if ext == "rs" {
                    files.push(p.to_path_buf());
                }
            }
        }
    }
    if let Some(m) = args.max_files {
        files.truncate(m);
    }
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(6));
    let mut tasks = Vec::new();
    for f in files {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let s = state.clone();
        let root = args.root.clone();
        let sym_limit = args.symbol_limit;
        let path = f.to_string_lossy().to_string();
        tasks.push(tokio::spawn(async move {
            let _p = permit;
            let msg = ra_file_symbols_connect(
                &s,
                RaFileSymbolsConnectArgs {
                    root,
                    path,
                    limit: sym_limit,
                    dry_run: args.dry_run,
                },
            )
            .await;
            match msg {
                Ok(txt) => txt
                    .split_whitespace()
                    .find_map(|w| w.parse::<usize>().ok())
                    .unwrap_or(0),
                Err(_) => 0,
            }
        }));
    }
    let mut total_connected = 0usize;
    for t in tasks {
        if let Ok(n) = t.await {
            total_connected += n;
        }
    }
    Ok(format!(
        "connected ~{} CONTAINS edges across workspace",
        total_connected
    ))
}

#[derive(Deserialize)]
pub struct RaImportAndConnectFileArgs {
    pub root: String,
    pub path: String,
    pub limit: Option<usize>,
    #[serde(rename = "dryRun")]
    pub dry_run: Option<bool>,
}

pub async fn ra_import_and_connect_file(
    state: &AppState,
    args: RaImportAndConnectFileArgs,
) -> Result<String, StatusCode> {
    if !args.dry_run.unwrap_or(false) {
        let _ = ra_document_symbols_to_entities(
            state,
            RaDocSymbolsToEntitiesArgs {
                root: args.root.clone(),
                path: args.path.clone(),
                limit: args.limit,
            },
        )
        .await?;
    }
    let msg = ra_file_symbols_connect(
        state,
        RaFileSymbolsConnectArgs {
            root: args.root,
            path: args.path,
            limit: args.limit,
            dry_run: args.dry_run,
        },
    )
    .await?;
    Ok(msg)
}

#[derive(Deserialize)]
pub struct RaBuildSymbolsAndConnectWorkspaceArgs {
    pub root: String,
    #[serde(rename = "maxFiles")]
    pub max_files: Option<usize>,
    #[serde(rename = "symbolLimit")]
    pub symbol_limit: Option<usize>,
    #[serde(rename = "dryRun")]
    pub dry_run: Option<bool>,
}

pub async fn ra_build_symbols_and_connect_workspace(
    state: &AppState,
    args: RaBuildSymbolsAndConnectWorkspaceArgs,
) -> Result<String, StatusCode> {
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(&args.root).into_iter().flatten() {
        let p = entry.path();
        if p.is_file() {
            if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                if ext == "rs" {
                    files.push(p.to_path_buf());
                }
            }
        }
    }
    if let Some(m) = args.max_files {
        files.truncate(m);
    }
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(4));
    let mut tasks = Vec::new();
    for f in files {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let s = state.clone();
        let root = args.root.clone();
        let sym_limit = args.symbol_limit;
        let path = f.to_string_lossy().to_string();
        tasks.push(tokio::spawn(async move {
            let _p = permit;
            if !args.dry_run.unwrap_or(false) {
                let _ = ra_document_symbols_to_entities(
                    &s,
                    RaDocSymbolsToEntitiesArgs {
                        root: root.clone(),
                        path: path.clone(),
                        limit: sym_limit,
                    },
                )
                .await;
            }
            let msg = ra_file_symbols_connect(
                &s,
                RaFileSymbolsConnectArgs {
                    root,
                    path,
                    limit: sym_limit,
                    dry_run: args.dry_run,
                },
            )
            .await;
            match msg {
                Ok(txt) => txt
                    .split_whitespace()
                    .find_map(|w| w.parse::<usize>().ok())
                    .unwrap_or(0),
                Err(_) => 0,
            }
        }));
    }
    let mut connected = 0usize;
    let mut imported = 0usize;
    for t in tasks {
        if let Ok(n) = t.await {
            connected += n;
            imported += 1;
        }
    }
    Ok(format!(
        "workspace: imported {} files; connected ~{} CONTAINS edges",
        imported, connected
    ))
}

#[derive(Deserialize)]
pub struct RaReferencesToEdgesArgs {
    pub root: String,
    pub path: String,
    pub line: u32,
    pub character: u32,
    pub relation: Option<String>,
    #[serde(rename = "dryRun")]
    pub dry_run: Option<bool>,
}

pub async fn ra_references_to_edges(
    state: &AppState,
    args: RaReferencesToEdgesArgs,
) -> Result<String, StatusCode> {
    // Fetch references
    let r = ra_references(RaReferencesArgs {
        root: args.root.clone(),
        path: args.path.clone(),
        line: args.line as usize,
        character: args.character as usize,
    })
    .await?;
    let refs = r
        .get("result")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let relation = args
        .relation
        .clone()
        .unwrap_or_else(|| "REFERENCES".to_string());

    // Source symbol at position
    let src_sym = match crate::serena_ra::symbol_at_position(
        Path::new(&args.root),
        Path::new(&args.path),
        args.line,
        args.character,
    )
    .await
    {
        Ok(v) => v.map(|(n, _k, range)| (n, range)),
        Err(_) => None,
    };

    // Build indexes
    let all_nodes = state
        .storage
        .get_all_nodes()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut file_index: std::collections::HashMap<String, Uuid> = std::collections::HashMap::new();
    let mut symbol_index: std::collections::HashMap<(String, String, (u32, u32)), Uuid> =
        std::collections::HashMap::new();
    for n in &all_nodes {
        let p = n.properties();
        if let Some(fp) = p.get("file_path").and_then(|v| v.as_str()) {
            let is_ce = p
                .get("code_entity")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if is_ce {
                let name = p
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let lr = p
                    .get("line_range")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| {
                        if arr.len() == 2 {
                            Some((
                                arr[0].as_u64().unwrap_or(1) as u32,
                                arr[1].as_u64().unwrap_or(1) as u32,
                            ))
                        } else {
                            None
                        }
                    });
                if let Some(r) = lr {
                    symbol_index.insert((fp.to_string(), name, r), *n.id());
                }
            } else {
                if p.get("node_type").and_then(|v| v.as_str()) == Some("file") {
                    file_index.insert(fp.to_string(), *n.id());
                }
            }
        }
    }

    let now = Utc::now();
    let temporal = TemporalMetadata {
        created_at: now,
        valid_from: now,
        valid_to: None,
        expired_at: None,
    };
    // Resolve src id
    let src_id = if let Some((name, range)) = src_sym.clone() {
        let lr = (
            range["start"]["line"].as_u64().unwrap_or(0) as u32 + 1,
            range["end"]["line"].as_u64().unwrap_or(0) as u32 + 1,
        );
        if let Some(id) = symbol_index.get(&(args.path.clone(), name.clone(), lr)) {
            *id
        } else {
            let id = Uuid::new_v4();
            let node = EntityNode {
                id,
                name: name.clone(),
                entity_type: CodeEntityType::Function.to_string(),
                labels: vec!["CodeEntity".into()],
                properties: serde_json::json!({"code_entity":true,"name":name,"entity_type":"Function","file_path":args.path,"line_range":[lr.0,lr.1]}),
                temporal: temporal.clone(),
                embedding: None,
            };
            state
                .storage
                .create_node(&node)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            counter!("graph_nodes_created_total", "node_type" => "CodeEntity").increment(1);
            id
        }
    } else {
        *file_index.get(&args.path).unwrap_or(&{
            let id = Uuid::new_v4();
            let node = EntityNode {
                id,
                name: args.path.clone(),
                entity_type: CodeEntityType::Module.to_string(),
                labels: vec!["File".into()],
                properties: serde_json::json!({"file_path": args.path, "node_type":"file"}),
                temporal: temporal.clone(),
                embedding: None,
            };
            let _ = state.storage.create_node(&node);
            id
        })
    };

    let mut created = 0usize;
    let mut seen = std::collections::HashSet::new();
    for loc in refs {
        let uri = match loc.get("uri").and_then(|v| v.as_str()) {
            Some(u) => u,
            None => continue,
        };
        let range = match loc.get("range") {
            Some(r) => r.clone(),
            None => serde_json::json!({"start":{"line":0,"character":0}}),
        };
        let pl = range
            .get("start")
            .and_then(|s| s.get("line"))
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as u32;
        let pc = range
            .get("start")
            .and_then(|s| s.get("character"))
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as u32;
        let ref_path = match url::Url::parse(uri)
            .ok()
            .and_then(|u| u.to_file_path().ok())
        {
            Some(p) => p.to_string_lossy().to_string(),
            None => continue,
        };

        // referencing symbol at reference location
        let ref_sym = match crate::serena_ra::symbol_at_position(
            Path::new(&args.root),
            Path::new(&ref_path),
            pl,
            pc,
        )
        .await
        {
            Ok(v) => v,
            Err(_) => None,
        };
        let ref_id = if let Some((name, _kind, rng)) = ref_sym {
            let lr = (
                rng["start"]["line"].as_u64().unwrap_or(0) as u32 + 1,
                rng["end"]["line"].as_u64().unwrap_or(0) as u32 + 1,
            );
            if let Some(id) = symbol_index.get(&(ref_path.clone(), name.clone(), lr)) {
                *id
            } else {
                let id = Uuid::new_v4();
                let node = EntityNode {
                    id,
                    name: name.clone(),
                    entity_type: CodeEntityType::Function.to_string(),
                    labels: vec!["CodeEntity".into()],
                    properties: serde_json::json!({"code_entity":true,"name":name,"entity_type":"Function","file_path":ref_path,"line_range":[lr.0,lr.1]}),
                    temporal: temporal.clone(),
                    embedding: None,
                };
                state
                    .storage
                    .create_node(&node)
                    .await
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                id
            }
        } else {
            *file_index.get(&ref_path).unwrap_or(&{
                let id = Uuid::new_v4();
                let node = EntityNode {
                    id,
                    name: ref_path.clone(),
                    entity_type: CodeEntityType::Module.to_string(),
                    labels: vec!["File".into()],
                    properties: serde_json::json!({"file_path": ref_path, "node_type":"file"}),
                    temporal: temporal.clone(),
                    embedding: None,
                };
                let _ = state.storage.create_node(&node);
                counter!("graph_nodes_created_total", "node_type" => "File").increment(1);
                id
            })
        };

        let key = (ref_id, src_id, relation.clone());
        if !seen.insert(key) {
            continue;
        }
        if edge_exists(state, ref_id, src_id, &relation).await? {
            counter!("graph_edges_skipped_total", "reason" => "duplicate", "relation" => relation.clone()).increment(1);
        } else {
            let edge = Edge {
                id: Uuid::new_v4(),
                source_id: ref_id,
                target_id: src_id,
                relationship: relation.clone(),
                properties: serde_json::json!({"source":"rust-analyzer"}),
                temporal: temporal.clone(),
                weight: 1.0,
            };
            state
                .storage
                .create_edge(&edge)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            counter!("graph_edges_created_total", "relation" => relation.clone()).increment(1);
            created += 1;
        }
    }
    Ok(format!("created {} {} edges", created, relation))
}

#[derive(Deserialize)]
pub struct RaSymbolDefinitionEdgeArgs {
    pub root: String,
    pub path: String,
    pub line: u32,
    pub character: u32,
    pub relation: Option<String>,
}

pub async fn ra_symbol_definition_edge(
    state: &AppState,
    args: RaSymbolDefinitionEdgeArgs,
) -> Result<String, StatusCode> {
    // 1) identify source symbol at position
    let src_sym = match crate::serena_ra::symbol_at_position(
        Path::new(&args.root),
        Path::new(&args.path),
        args.line,
        args.character,
    )
    .await
    {
        Ok(v) => v.map(|(n, _k, range)| (n, range)),
        Err(_) => None,
    };
    // 2) find definition location
    let def = ra_definition(RaDefinitionArgs {
        root: args.root.clone(),
        path: args.path.clone(),
        line: args.line as usize,
        character: args.character as usize,
    })
    .await?;
    let (def_uri, def_range) = if let Some(obj) = def.as_object() {
        if let Some(arr) = obj
            .get("result")
            .and_then(|v| v.as_array())
            .or_else(|| obj.get("location").and_then(|v| v.as_array()))
        {
            if let Some(first) = arr.get(0) {
                let uri = first
                    .get("uri")
                    .and_then(|u| u.as_str())
                    .unwrap_or("")
                    .to_string();
                let range=first.get("range").cloned().unwrap_or(serde_json::json!({"start":{"line":0,"character":0},"end":{"line":0,"character":0}}));
                (uri, range)
            } else {
                (String::new(), serde_json::json!({}))
            }
        } else {
            (String::new(), serde_json::json!({}))
        }
    } else {
        (String::new(), serde_json::json!({}))
    };
    if def_uri.is_empty() {
        return Ok("no definition".into());
    }
    let dst_symbol =
        crate::serena_ra::find_target_symbol_name(Path::new(&args.root), &def_uri, &def_range)
            .await;
    let dst_path = url::Url::parse(&def_uri)
        .ok()
        .and_then(|u| u.to_file_path().ok())
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    // Build indexes over existing code entities and file nodes
    let all_nodes = state
        .storage
        .get_all_nodes()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut file_index: std::collections::HashMap<String, Uuid> = std::collections::HashMap::new();
    let mut symbol_index: std::collections::HashMap<(String, String, (u32, u32)), Uuid> =
        std::collections::HashMap::new();
    for n in all_nodes {
        let p = n.properties();
        if let Some(fp) = p.get("file_path").and_then(|v| v.as_str()) {
            let is_ce = p
                .get("code_entity")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if is_ce {
                let name = p
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let lr = p
                    .get("line_range")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| {
                        if arr.len() == 2 {
                            Some((
                                arr[0].as_u64().unwrap_or(1) as u32,
                                arr[1].as_u64().unwrap_or(1) as u32,
                            ))
                        } else {
                            None
                        }
                    });
                if let Some(r) = lr {
                    symbol_index.insert((fp.to_string(), name, r), *n.id());
                }
            } else {
                if p.get("node_type").and_then(|v| v.as_str()) == Some("file") {
                    file_index.insert(fp.to_string(), *n.id());
                }
            }
        }
    }

    let now = Utc::now();
    let temporal = TemporalMetadata {
        created_at: now,
        valid_from: now,
        valid_to: None,
        expired_at: None,
    };
    // Resolve src id
    let src_id = if let Some((name, range)) = src_sym {
        let lr = (
            range["start"]["line"].as_u64().unwrap_or(0) as u32 + 1,
            range["end"]["line"].as_u64().unwrap_or(0) as u32 + 1,
        );
        if let Some(id) = symbol_index.get(&(args.path.clone(), name.clone(), lr)) {
            *id
        } else {
            // create symbol node if missing
            let id = Uuid::new_v4();
            let node = EntityNode {
                id,
                name: name.clone(),
                entity_type: CodeEntityType::Function.to_string(),
                labels: vec!["CodeEntity".into()],
                properties: serde_json::json!({"code_entity":true,"name":name,"entity_type":"Function","file_path":args.path,"line_range":[lr.0,lr.1]}),
                temporal: temporal.clone(),
                embedding: None,
            };
            state
                .storage
                .create_node(&node)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            id
        }
    } else {
        // fallback to file node
        *file_index.get(&args.path).unwrap_or(&{
            let id = Uuid::new_v4();
            let node = EntityNode {
                id,
                name: args.path.clone(),
                entity_type: CodeEntityType::Module.to_string(),
                labels: vec!["File".into()],
                properties: serde_json::json!({"file_path": args.path, "node_type":"file"}),
                temporal: temporal.clone(),
                embedding: None,
            };
            let _ = state.storage.create_node(&node);
            id
        })
    };

    // Resolve dst id
    let dst_id = if let Some(name) = dst_symbol {
        // Guess line range from def_range
        let lr = (
            def_range["start"]["line"].as_u64().unwrap_or(0) as u32 + 1,
            def_range["end"]["line"].as_u64().unwrap_or(0) as u32 + 1,
        );
        if let Some(id) = symbol_index.get(&(dst_path.clone(), name.clone(), lr)) {
            *id
        } else {
            let id = Uuid::new_v4();
            let node = EntityNode {
                id,
                name: name.clone(),
                entity_type: CodeEntityType::Function.to_string(),
                labels: vec!["CodeEntity".into()],
                properties: serde_json::json!({"code_entity":true,"name":name,"entity_type":"Function","file_path":dst_path,"line_range":[lr.0,lr.1]}),
                temporal: temporal.clone(),
                embedding: None,
            };
            state
                .storage
                .create_node(&node)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            id
        }
    } else {
        *file_index.get(&dst_path).unwrap_or(&{
            let id = Uuid::new_v4();
            let node = EntityNode {
                id,
                name: dst_path.clone(),
                entity_type: CodeEntityType::Module.to_string(),
                labels: vec!["File".into()],
                properties: serde_json::json!({"file_path": dst_path, "node_type":"file"}),
                temporal: temporal.clone(),
                embedding: None,
            };
            let _ = state.storage.create_node(&node);
            id
        })
    };

    let relation = args
        .relation
        .clone()
        .unwrap_or_else(|| "DEFINES".to_string());
    if edge_exists(state, src_id, dst_id, &relation).await? {
        counter!("graph_edges_skipped_total", "reason" => "duplicate", "relation" => relation.clone()).increment(1);
        return Ok("exists".into());
    }
    let edge = Edge {
        id: Uuid::new_v4(),
        source_id: src_id,
        target_id: dst_id,
        relationship: relation.clone(),
        properties: serde_json::json!({"source":"rust-analyzer"}),
        temporal: temporal.clone(),
        weight: 1.0,
    };
    state
        .storage
        .create_edge(&edge)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    counter!("graph_edges_created_total", "relation" => relation).increment(1);
    Ok("ok".into())
}

#[derive(Deserialize)]
pub struct RaBuildReferencesForFileArgs {
    pub root: String,
    pub path: String,
    #[serde(rename = "maxSymbols")]
    pub max_symbols: Option<usize>,
    pub relation: Option<String>,
    #[serde(rename = "dryRun")]
    pub dry_run: Option<bool>,
}

pub async fn ra_build_references_for_file(
    state: &AppState,
    args: RaBuildReferencesForFileArgs,
) -> Result<String, StatusCode> {
    let res = ra_document_symbols(RaDocumentSymbolsArgs {
        root: args.root.clone(),
        path: args.path.clone(),
    })
    .await?;
    fn collect_positions(v: &serde_json::Value, out: &mut Vec<(String, u32, u32)>) {
        if let Some(arr) = v.as_array() {
            for it in arr {
                collect_positions(it, out);
            }
            return;
        }
        if let Some(o) = v.as_object() {
            if let Some(n) = o.get("name").and_then(|s| s.as_str()) {
                let sline = o
                    .get("range")
                    .and_then(|r| r.get("start"))
                    .and_then(|s| s.get("line"))
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0) as u32;
                let schar = o
                    .get("range")
                    .and_then(|r| r.get("start"))
                    .and_then(|s| s.get("character"))
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0) as u32;
                out.push((n.to_string(), sline, schar));
            }
            if let Some(ch) = o.get("children") {
                collect_positions(ch, out);
            }
        }
    }
    let mut syms = Vec::new();
    collect_positions(&res, &mut syms);
    if let Some(m) = args.max_symbols {
        syms.truncate(m);
    }
    let mut total_created = 0usize;
    for (_name, sl, sc) in syms {
        let msg = ra_references_to_edges(
            state,
            RaReferencesToEdgesArgs {
                root: args.root.clone(),
                path: args.path.clone(),
                line: sl,
                character: sc,
                relation: args.relation.clone(),
                dry_run: args.dry_run,
            },
        )
        .await?;
        if let Some(n) = msg.split_whitespace().find_map(|w| w.parse::<usize>().ok()) {
            total_created += n;
        }
    }
    Ok(format!("created ~{} reference edges", total_created))
}

#[derive(Deserialize)]
pub struct OnboardingDoneArgs {
    pub project: Option<String>,
}

pub async fn onboarding_done(args: OnboardingDoneArgs) -> Result<&'static str, StatusCode> {
    let root = crate::serena_config::detect_project_root(args.project.as_deref().map(Path::new))
        .map_err(|err| {
            error!(error = %err, "无法定位项目根目录");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    let mut cfg = crate::serena_config::load_or_default(&root).map_err(|err| {
        error!(error = %err, "读取项目配置失败");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    cfg.onboarding_performed = true;
    crate::serena_config::save(&root, &cfg).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok("ok")
}

/* Legacy snippet from main.rs (now migrated to functions above and wired via tools/call)
                            "serena.get_current_config" => {
                                #[derive(serde::Deserialize)] struct A{ project: Option<String> }
                                let project = params.get("arguments").and_then(|a| serde_json::from_value::<A>(a.clone()).ok()).and_then(|a| a.project);
                                let root = crate::serena_config::detect_project_root(project.as_deref().map(std::path::Path::new)).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                let cfg = crate::serena_config::load_or_default(&root);
                                let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": cfg}]}});
                                Ok(Json(response))
                            }
                            // Serena + Graphiti bridge: capture edits into knowledge graph
                            "serena_kg.replace_lines_and_record" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)]
                                    struct A { path: String, start: usize, end: usize, content: String, description: Option<String> }
                                    let a: A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    // Apply edit
                                    crate::serena_tools::replace_lines(crate::serena_tools::ReplaceLinesRequest{ path: a.path.clone().into(), start: a.start, end: a.end, content: a.content.clone() }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    // Record activity into KG
                                    let req = RecordActivityRequest {
                                        activity_type: "Implementation".to_string(),
                                        title: "Edit file lines".to_string(),
                                        description: a.description.unwrap_or_else(|| format!("replace lines {}-{} in {}", a.start, a.end, a.path)),
                                        developer: whoami::username(),
                                        project: std::env::current_dir().ok().and_then(|p| p.file_name().map(|s| s.to_string_lossy().to_string())).unwrap_or_else(|| "unknown".into()),
                                        related_entities: Some(vec![]), duration_minutes: None, difficulty: None, quality: None,
                                    };
                                    match state.graphiti.record_activity(req).await {
                                        Ok(_) => {
                                            let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":"ok"}]}});
                                            Ok(Json(response))
                                        }
                                        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR)
                                    }
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena.get_symbols_overview" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ path: String }
                                    let a: A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let syms = crate::serena_symbols::list_symbols(std::path::Path::new(&a.path)).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": syms}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            "serena.find_symbol" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root: String, name: String }
                                    let a: A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let hits = crate::serena_symbols::find_symbol(&crate::serena_symbols::FindSymbolRequest{ root: a.root.into(), name: a.name }).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": hits}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena.find_referencing_symbols" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root: String, name: String }
                                    let a: A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let hits = crate::serena_symbols::find_references(&crate::serena_symbols::FindRefsRequest{ root: a.root.into(), name: a.name }).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": hits}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena.replace_symbol_body" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ path: String, name: String, new_body: String }
                                    let a: A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    crate::serena_symbols::replace_symbol_body(&crate::serena_symbols::ReplaceSymbolBodyRequest{ path: a.path.into(), name: a.name, new_body: a.new_body }).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":"ok"}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            // rust-analyzer-backed tools
                            "serena.ra.document_symbols" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, path:String }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let res = match crate::serena_ra::document_symbols(std::path::Path::new(&a.root), std::path::Path::new(&a.path)).await {
                                        Ok(v) => v,
                                        Err(e) => {
                                            // Fallback: regex-based overview
                                            let syms = crate::serena_symbols::list_symbols(std::path::Path::new(&a.path)).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                            let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": {"fallback":"regex","symbols": syms, "error": e.to_string()}}]}});
                                            return Ok(Json(response));
                                        }
                                    };
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": res}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena_kg.ra_document_symbols_to_entities" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, path:String, limit: Option<usize> }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let res = crate::serena_ra::document_symbols(std::path::Path::new(&a.root), std::path::Path::new(&a.path)).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    // Convert LSP DocumentSymbol[] into CodeEntity and persist
                                    fn walk_syms(v: &serde_json::Value, out: &mut Vec<(String,String,(u32,u32))>) {
                                        if let Some(arr)=v.as_array(){ for it in arr { walk_syms(it,out); } return; }
                                        if let Some(obj)=v.as_object(){
                                            let name = obj.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
                                            let kind = obj.get("kind").and_then(|x| x.as_u64()).unwrap_or(0);
                                            let range = obj.get("range");
                                            let (s,e) = if let Some(r)=range.and_then(|r| r.get("start").zip(r.get("end"))) {
                                                let sl = r.0.get("line").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                let el = r.1.get("line").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                (sl+1, el+1)
                                            } else { (1,1) };
                                            let k = match kind { 5=>"class", 6=>"method", 12=>"function", 23=>"interface", 2=>"module", _=>"symbol" }.to_string();
                                            if !name.is_empty(){ out.push((k,name,(s,e))); }
                                            if let Some(children)=obj.get("children") { walk_syms(children,out); }
                                        }
                                    }
                                    let mut items = Vec::new(); walk_syms(&res,&mut items);
                                    if let Some(l)=a.limit { items.truncate(l); }
                                    let mut ok = 0usize; let mut failed = 0usize;
                                    for (k,n,(s,e)) in items {
                                        let req = AddCodeEntityRequest { entity_type: k, name: n, description: format!("Imported from RA: {}", a.path), file_path: Some(a.path.clone()), line_range: Some((s,e)), language: Some("rust".into()), framework: None, complexity: None, importance: None };
                                        match state.graphiti.add_code_entity(req).await { Ok(_) => ok+=1, Err(_) => failed+=1 }
                                    }
                                    let text = format!("Imported {} entities ({} failed)", ok, failed);
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": text}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena_kg.ra_file_symbols_connect" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, path:String, limit: Option<usize> }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    // 1) Import symbols as entities (幂等：add_code_entity 内部会合并为新节点)
                                    let res = crate::serena_ra::document_symbols(std::path::Path::new(&a.root), std::path::Path::new(&a.path)).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    fn collect(v:&serde_json::Value,out:&mut Vec<(String,(u32,u32))>){ if let Some(arr)=v.as_array(){ for it in arr { collect(it,out);} return;} if let Some(o)=v.as_object(){ let name=o.get("name").and_then(|s|s.as_str()).unwrap_or("").to_string(); let r=o.get("range"); let (s,e)= if let Some(rr)=r.and_then(|r| r.get("start").zip(r.get("end"))) { ((rr.0.get("line").and_then(|x|x.as_u64()).unwrap_or(0) as u32)+1, (rr.1.get("line").and_then(|x|x.as_u64()).unwrap_or(0) as u32)+1) } else {(1,1)}; if !name.is_empty(){ out.push((name,(s,e))); } if let Some(ch)=o.get("children"){ collect(ch,out);} }}
                                    let mut syms=Vec::new(); collect(&res,&mut syms); if let Some(l)=a.limit{ syms.truncate(l); }
                                    for (name,(s,e)) in &syms { let req = AddCodeEntityRequest{ entity_type:"function".into(), name:name.clone(), description: format!("{}:{}-{}", a.path,s,e), file_path:Some(a.path.clone()), line_range:Some((*s,*e)), language:Some("rust".into()), framework:None, complexity:None, importance:None }; let _= state.graphiti.add_code_entity(req).await; }
                                    // 2) Connect CONTAINS edges from file node to each symbol entity（含简单去重）
                                    let all_nodes = self.storage.get_all_nodes().await.unwrap_or_default();
                                    let mut file_index=std::collections::HashMap::new(); let mut symbol_index=std::collections::HashMap::new();
                                    for n in all_nodes { let p=n.properties(); if let Some(fp)=p.get("file_path").and_then(|v|v.as_str()){ let is_ce=p.get("code_entity").and_then(|v|v.as_bool()).unwrap_or(false); if is_ce{ let name=p.get("name").and_then(|v|v.as_str()).unwrap_or("").to_string(); let lr=p.get("line_range").and_then(|v|v.as_array()).and_then(|arr| if arr.len()==2{ Some((arr[0].as_u64().unwrap_or(1) as u32, arr[1].as_u64().unwrap_or(1) as u32)) } else { None }); if let Some(r)=lr{ symbol_index.insert((fp.to_string(),name,r), *n.id()); } } else { file_index.insert(fp.to_string(), *n.id()); } } }
                                    let now=chrono::Utc::now(); let temporal=TemporalMetadata{ created_at:now, valid_from:now, valid_to:None, expired_at:None };
                                    let file_id = if let Some(id)=file_index.get(&a.path){ *id } else { let node=EntityNode{ id:Uuid::new_v4(), name:a.path.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": a.path, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; node.id };
                                    let mut ok=0usize; let mut fail=0usize; use std::collections::HashSet; let mut seen=HashSet::new();
                                    // persistent de-dup against storage
                                    let mut existing: std::collections::HashSet<(Uuid,Uuid,String)> = std::collections::HashSet::new();
                                    if let Ok(edges_all)=self.storage.get_all_edges().await{ for e in edges_all { existing.insert((e.source_id,e.target_id,e.relationship.clone())); } }
                                    for (name,(s,e)) in syms { if let Some(sym_id)=symbol_index.get(&(a.path.clone(), name.clone(), (s,e))) { let sig=(file_id,*sym_id,"CONTAINS".to_string()); if seen.insert(sig.clone()) && !existing.contains(&sig){ let edge=graphiti_core::graph::Edge{ id:Uuid::new_v4(), source_id:file_id, target_id:*sym_id, relationship:"CONTAINS".into(), properties:serde_json::json!({"source":"rust-analyzer"}), temporal: temporal.clone(), weight:1.0 }; match self.storage.create_edge(&edge).await{ Ok(_)=>ok+=1, Err(_)=>fail+=1 } } } }
                                    let txt=format!("Connected {} CONTAINS edges ({} failed)", ok, fail);
                                    let response=serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":txt}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena_kg.ra_build_workspace" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, maxFiles: Option<usize>, symbolLimit: Option<usize> }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    // Enumerate Rust files under root (basic glob)
                                    // enumerate
                                    let mut files: Vec<PathBuf> = Vec::new();
                                    for entry in walkdir::WalkDir::new(&a.root).into_iter().flatten(){ let p=entry.path(); if p.is_file(){ if let Some(ext)=p.extension().and_then(|s|s.to_str()){ if ext=="rs"{ files.push(p.to_path_buf()); } } } }
                                    if let Some(m)=a.maxFiles{ files.truncate(m); }
                                    // parallel import + connect with throttling
                                    let semaphore = Arc::new(tokio::sync::Semaphore::new(8));
                                    let imported = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                                    let connected = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                                    let mut tasks = Vec::new();
                                    for f in files {
                                        let permit = semaphore.clone().acquire_owned().await.unwrap();
                                        let root = a.root.clone();
                                        let sym_limit = a.symbolLimit;
                                        let storage = self.storage.clone();
                                        let graphiti = state.graphiti.clone();
                                        tasks.push(tokio::spawn(async move{
                                            let _p = permit;
                                            let path=f.to_string_lossy().to_string();
                                            // import
                                            if let Ok(res) = crate::serena_ra::document_symbols(std::path::Path::new(&root), &f).await {
                                                fn collect(v:&serde_json::Value,out:&mut Vec<(String,(u32,u32))>){ if let Some(arr)=v.as_array(){ for it in arr { collect(it,out);} return;} if let Some(o)=v.as_object(){ let name=o.get("name").and_then(|s|s.as_str()).unwrap_or("").to_string(); let r=o.get("range"); let (s,e)= if let Some(rr)=r.and_then(|r| r.get("start").zip(r.get("end"))) { ((rr.0.get("line").and_then(|x|x.as_u64()).unwrap_or(0) as u32)+1, (rr.1.get("line").and_then(|x|x.as_u64()).unwrap_or(0) as u32)+1) } else {(1,1)}; if !name.is_empty(){ out.push((name,(s,e))); } if let Some(ch)=o.get("children"){ collect(ch,out);} }}
                                                let mut syms=Vec::new(); collect(&res,&mut syms); if let Some(l)=sym_limit{ syms.truncate(l); }
                                                for (name,(s,e)) in &syms { let req = AddCodeEntityRequest{ entity_type:"function".into(), name:name.clone(), description: format!("{}:{}-{}", path,s,e), file_path:Some(path.clone()), line_range:Some((*s,*e)), language:Some("rust".into()), framework:None, complexity:None, importance:None }; let _ = graphiti.add_code_entity(req).await; imported.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                                                // connect
                                                if let Ok(nodes)=storage.get_all_nodes().await { let mut file_index=std::collections::HashMap::new(); let mut symbol_index=std::collections::HashMap::new(); for n in nodes { let p=n.properties(); if let Some(fp)=p.get("file_path").and_then(|v|v.as_str()){ let is_ce=p.get("code_entity").and_then(|v|v.as_bool()).unwrap_or(false); if is_ce{ let name=p.get("name").and_then(|v|v.as_str()).unwrap_or("").to_string(); let lr=p.get("line_range").and_then(|v|v.as_array()).and_then(|arr| if arr.len()==2{ Some((arr[0].as_u64().unwrap_or(1) as u32, arr[1].as_u64().unwrap_or(1) as u32)) } else { None }); if let Some(r)=lr{ symbol_index.insert((fp.to_string(),name,r), *n.id()); } } else { file_index.insert(fp.to_string(), *n.id()); } } }
                                                    let now=chrono::Utc::now(); let temporal=TemporalMetadata{ created_at:now, valid_from:now, valid_to:None, expired_at:None };
                                                    let file_id = if let Some(id)=file_index.get(&path){ *id } else { let node=EntityNode{ id:Uuid::new_v4(), name:path.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": path, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=storage.create_node(&node).await; node.id };
                                                    use std::collections::HashSet; let mut seen=HashSet::new();
                                                    for (name,(s,e)) in syms { if let Some(sym_id)=symbol_index.get(&(path.clone(), name.clone(), (s,e))) { let sig=(file_id,*sym_id,"CONTAINS".to_string()); if seen.insert(sig.clone()){ let edge=graphiti_core::graph::Edge{ id:Uuid::new_v4(), source_id:file_id, target_id:*sym_id, relationship:"CONTAINS".into(), properties:serde_json::json!({"source":"rust-analyzer"}), temporal: temporal.clone(), weight:1.0 }; let _=storage.create_edge(&edge).await; connected.fetch_add(1, std::sync::atomic::Ordering::Relaxed); } } }
                                                }
                                            }
                                        }));
                                    }
                                    for t in tasks { let _=t.await; }
                                    let txt=format!("Workspace build: imported ~{} symbols, connected ~{} edges", imported.load(std::sync::atomic::Ordering::Relaxed), connected.load(std::sync::atomic::Ordering::Relaxed));
                                    let _ = state.graphiti.add_memory(AddMemoryRequest{ content: txt.clone(), name: Some("RA Workspace Build".into()), source: Some("serena_kg".into()), memory_type: Some("audit".into()), metadata: None, group_id: None, timestamp: None }).await;
                                    let response=serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":txt}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena_kg.ra_symbol_definition_edge" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, path:String, line:u32, character:u32, relation: Option<String> }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let relation = a.relation.unwrap_or_else(|| "DEFINES".to_string());
                                    // 1) identify source symbol
                                    let src_sym = crate::serena_ra::symbol_at_position(std::path::Path::new(&a.root), std::path::Path::new(&a.path), a.line, a.character).await.ok().flatten();
                                    // 2) get definition
                                    let def = crate::serena_ra::definition(std::path::Path::new(&a.root), std::path::Path::new(&a.path), a.line, a.character).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let (def_uri, def_range) = if let Some(obj)=def.as_object(){ if let Some(arr)=obj.get("result").and_then(|v| v.as_array()).or_else(|| obj.get("location").and_then(|v| v.as_array())){ if let Some(first)=arr.get(0){ let uri=first.get("uri").and_then(|u|u.as_str()).unwrap_or(""); let range=first.get("range").cloned().unwrap_or(serde_json::json!({"start":{"line":0,"character":0},"end":{"line":0,"character":0}})); (uri.to_string(), range) } else { (String::new(), serde_json::json!({})) } } else { (String::new(), serde_json::json!({})) } } else { (String::new(), serde_json::json!({})) };
                                    if def_uri.is_empty(){ return Ok(Json(serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":"no definition"}]}}))); }
                                    let dst_symbol = crate::serena_ra::find_target_symbol_name(std::path::Path::new(&a.root), &def_uri, &def_range).await;
                                    // Build indexes
                                    let all_nodes = self.storage.get_all_nodes().await.unwrap_or_default();
                                    let mut file_index=std::collections::HashMap::new(); let mut symbol_index=std::collections::HashMap::new();
                                    for n in all_nodes { let p=n.properties(); if let Some(fp)=p.get("file_path").and_then(|v|v.as_str()){ let is_ce=p.get("code_entity").and_then(|v|v.as_bool()).unwrap_or(false); if is_ce{ let name=p.get("name").and_then(|v|v.as_str()).unwrap_or("").to_string(); let lr=p.get("line_range").and_then(|v|v.as_array()).and_then(|arr| if arr.len()==2{ Some((arr[0].as_u64().unwrap_or(1) as u32, arr[1].as_u64().unwrap_or(1) as u32)) } else { None }); if let Some(r)=lr{ symbol_index.insert((fp.to_string(),name,r), *n.id()); } } else { file_index.insert(fp.to_string(), *n.id()); } } }
                                    let now=chrono::Utc::now(); let temporal=TemporalMetadata{ created_at:now, valid_from:now, valid_to:None, expired_at:None };
                                    // resolve src id
                                    let src_id = if let Some(sym)=&src_sym { if let Some(id)=symbol_index.keys().find(|(fp,n,_r)| fp==&a.path && n==sym).and_then(|k| symbol_index.get(k)) { *id } else { *file_index.get(&a.path).unwrap_or(&{ let node=EntityNode{ id:Uuid::new_v4(), name:a.path.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": a.path, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; &node.id }) } } else { *file_index.get(&a.path).unwrap_or(&{ let node=EntityNode{ id:Uuid::new_v4(), name:a.path.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": a.path, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; &node.id }) };
                                    // resolve dst path & id
                                    let dst_path = url::Url::parse(&def_uri).ok().and_then(|u| u.to_file_path().ok()).map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
                                    let dst_id = if let Some(sym)=&dst_symbol { if let Some(id)=symbol_index.keys().find(|(fp,n,_r)| fp==&dst_path && n==sym).and_then(|k| symbol_index.get(k)) { *id } else { *file_index.get(&dst_path).unwrap_or(&{ let node=EntityNode{ id:Uuid::new_v4(), name:dst_path.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": dst_path, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; &node.id }) } } else { *file_index.get(&dst_path).unwrap_or(&{ let node=EntityNode{ id:Uuid::new_v4(), name:dst_path.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": dst_path, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; &node.id }) };
                                    // de-dup within call
                                    use std::collections::HashSet; let mut seen=HashSet::new();
                                    let sig=(src_id,dst_id,relation.clone(),src_sym.as_ref().map(|(n,_,_)|n.clone()), dst_symbol.clone()); if !seen.insert(sig) { let response=serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":"skipped duplicate"}]}}); return Ok(Json(response)); }
                                    let mut props=serde_json::json!({"source":"rust-analyzer"}); if let Some((n,_,_))=&src_sym { props["src_symbol"]=serde_json::json!(n); } if let Some(n)=dst_symbol.clone() { props["dst_symbol"]=serde_json::json!(n); }
                                    let edge=graphiti_core::graph::Edge{ id:Uuid::new_v4(), source_id:src_id, target_id:dst_id, relationship: relation, properties: props, temporal: temporal.clone(), weight:1.0 };
                                    let _ = self.storage.create_edge(&edge).await;
                                    let response=serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json":{"source": a.path, "target": dst_path, "src_symbol": src_sym.map(|(n,_,_)| n), "dst_symbol": dst_symbol}}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena_kg.ra_build_references_for_file" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, path:String, maxSymbols: Option<usize>, relation: Option<String> }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    // collect top-level symbols
                                    let doc = crate::serena_ra::document_symbols(std::path::Path::new(&a.root), std::path::Path::new(&a.path)).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let mut positions: Vec<(u32,u32)> = Vec::new();
                                    fn collect_pos(v:&serde_json::Value, out:&mut Vec<(u32,u32)>){ if let Some(arr)=v.as_array(){ for it in arr{ collect_pos(it,out);} return;} if let Some(o)=v.as_object(){ if let Some(sel)=o.get("selectionRange"){ let l=sel["start"]["line"].as_u64().unwrap_or(0) as u32; let c=sel["start"]["character"].as_u64().unwrap_or(0) as u32; out.push((l,c)); } if let Some(ch)=o.get("children"){ collect_pos(ch,out);} }}
                                    collect_pos(&doc, &mut positions); if let Some(m)=a.maxSymbols{ positions.truncate(m); }
                                    let mut total_edges=0usize; let relation=a.relation.unwrap_or_else(|| "REFERENCES".into());
                                    for (l,c) in positions {
                                        // reuse references tool logic directly
                                        let res = crate::serena_ra::references(std::path::Path::new(&a.root), std::path::Path::new(&a.path), l, c).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                        let src_sym = crate::serena_ra::symbol_at_position(std::path::Path::new(&a.root), std::path::Path::new(&a.path), l, c).await.ok().flatten();
                                        // re-use part of ra_references_to_edges path quickly (best-effort)
                                        if let Some(arr)=res.as_array(){
                                            let mut edges_local=Vec::new();
                                            for it in arr { if let Some(loc)=it.get("uri").and_then(|u|u.as_str()){ let dst_symbol = if let Some(loc_range)=it.get("range"){ crate::serena_ra::find_target_symbol_name(std::path::Path::new(&a.root), loc, loc_range).await } else { None }; if let Ok(url)=url::Url::parse(loc){ if let Ok(p)=url.to_file_path(){ let target_path=p.to_string_lossy().to_string(); if target_path!=a.path { edges_local.push((target_path, dst_symbol.clone())); } } } } }
                                            // write edges (file/symbol upsert + de-dup)
                                            let all_nodes = self.storage.get_all_nodes().await.unwrap_or_default();
                                            let mut file_index=std::collections::HashMap::new(); let mut symbol_index=std::collections::HashMap::new();
                                            for n in all_nodes { let p=n.properties(); if let Some(fp)=p.get("file_path").and_then(|v|v.as_str()){ let is_ce=p.get("code_entity").and_then(|v|v.as_bool()).unwrap_or(false); if is_ce{ let name=p.get("name").and_then(|v|v.as_str()).unwrap_or("").to_string(); let lr=p.get("line_range").and_then(|v|v.as_array()).and_then(|arr| if arr.len()==2{ Some((arr[0].as_u64().unwrap_or(1) as u32, arr[1].as_u64().unwrap_or(1) as u32)) } else { None }); if let Some(r)=lr{ symbol_index.insert((fp.to_string(),name,r), *n.id()); } } else { file_index.insert(fp.to_string(), *n.id()); } } }
                                            let now=chrono::Utc::now(); let temporal=TemporalMetadata{ created_at:now, valid_from:now, valid_to:None, expired_at:None };
                                            let src_id = if let Some(sym)=&src_sym { if let Some(id)=symbol_index.keys().find(|(fp,n,_r)| fp==&a.path && n==sym).and_then(|k| symbol_index.get(k)){ *id } else { *file_index.get(&a.path).unwrap_or(&{ let node=EntityNode{ id:Uuid::new_v4(), name:a.path.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": a.path, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; &node.id }) } } else { *file_index.get(&a.path).unwrap_or(&{ let node=EntityNode{ id:Uuid::new_v4(), name:a.path.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": a.path, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; &node.id }) };
                                            use std::collections::HashSet; let mut seen=HashSet::new(); let mut existing=HashSet::new(); if let Ok(edges_all)=self.storage.get_all_edges().await{ for e0 in edges_all { existing.insert((e0.source_id,e0.target_id,e0.relationship.clone())); } }
                                            for (target_path, dst_symbol) in edges_local { let dst_id = if let Some(sym)=dst_symbol { if let Some(id)=symbol_index.keys().find(|(fp,n,_r)| fp==&target_path && n==&sym).and_then(|k| symbol_index.get(k)){ *id } else { *file_index.get(&target_path).unwrap_or(&{ let node=EntityNode{ id:Uuid::new_v4(), name:target_path.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": target_path, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; &node.id }) } } else { *file_index.get(&target_path).unwrap_or(&{ let node=EntityNode{ id:Uuid::new_v4(), name:target_path.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": target_path, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; &node.id }) }; let sig=(src_id,dst_id,relation.clone()); if seen.insert(sig.clone()) && !existing.contains(&sig){ let edge=graphiti_core::graph::Edge{ id:Uuid::new_v4(), source_id:src_id, target_id:dst_id, relationship: relation.clone(), properties:serde_json::json!({"source":"rust-analyzer"}), temporal: temporal.clone(), weight:1.0 }; let _=self.storage.create_edge(&edge).await; total_edges+=1; } }
                                        }
                                    }
                                    let txt=format!("Built ~{} reference edges for symbols in {}", total_edges, a.path);
                                    let _= state.graphiti.add_memory(AddMemoryRequest{ content: txt.clone(), name: Some("RA File References Build".into()), source: Some("serena_kg".into()), memory_type: Some("audit".into()), metadata: None, group_id: None, timestamp: None }).await;
                                    let response=serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":txt}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena_kg.graph_summary" => {
                                // return simple counts
                                let nodes = self.storage.get_all_nodes().await.unwrap_or_default();
                                let edges = self.storage.get_all_edges().await.unwrap_or_default();
                                let files = nodes.iter().filter(|n| n.properties().get("node_type").and_then(|v|v.as_str())==Some("file")).count();
                                let entities = nodes.len().saturating_sub(files);
                                let response=serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": {"nodes": nodes.len(), "edges": edges.len(), "files": files, "entities": entities}}]}});
                                Ok(Json(response))
                            }
                            "serena_kg.recent_edges" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ relationship:String, limit: Option<usize> }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let lim = a.limit.unwrap_or(20);
                                    match self.storage.get_recent_edges_by_rel(&a.relationship, lim).await {
                                        Ok(list) => {
                                            let json = list.into_iter().map(|e| serde_json::json!({
                                                "id": e.id,
                                                "source_id": e.source_id,
                                                "target_id": e.target_id,
                                                "relationship": e.relationship,
                                                "properties": e.properties,
                                                "created_at": e.temporal.created_at
                                            })).collect::<Vec<_>>();
                                            let response=serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": json}]}});
                                            Ok(Json(response))
                                        }
                                        Err(_)=> Err(StatusCode::INTERNAL_SERVER_ERROR)
                                    }
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena_kg.prune_edges" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ limit: usize }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    match self.storage.prune_to_limit(a.limit).await { Ok(n)=>{
                                        let response=serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text": format!("Pruned {} edges", n)}]}});
                                        Ok(Json(response))
                                    }, Err(_)=> Err(StatusCode::INTERNAL_SERVER_ERROR) }
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena_kg.ra_build_references_workspace" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, maxFiles: Option<usize>, symbolLimit: Option<usize>, relation: Option<String> }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    // enumerate files
                                    let mut files: Vec<PathBuf> = Vec::new();
                                    for entry in walkdir::WalkDir::new(&a.root).into_iter().flatten(){ let p=entry.path(); if p.is_file(){ if let Some(ext)=p.extension().and_then(|s|s.to_str()){ if ext=="rs"{ files.push(p.to_path_buf()); } } } }
                                    if let Some(m)=a.maxFiles{ files.truncate(m); }
                                    // parallel per-file references build
                                    let semaphore = Arc::new(tokio::sync::Semaphore::new(6));
                                    let total = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                                    let mut tasks = Vec::new();
                                    for f in files { let permit=semaphore.clone().acquire_owned().await.unwrap(); let root=a.root.clone(); let sym_limit=a.symbolLimit; let relation=a.relation.clone(); let storage=self.storage.clone(); let graphiti=state.graphiti.clone(); tasks.push(tokio::spawn(async move{
                                        let _p=permit; let path=f.to_string_lossy().to_string();
                                        if let Ok(doc)=crate::serena_ra::document_symbols(std::path::Path::new(&root), &f).await{ let mut positions=Vec::new(); fn collect_pos(v:&serde_json::Value, out:&mut Vec<(u32,u32)>){ if let Some(arr)=v.as_array(){ for it in arr { collect_pos(it,out);} return;} if let Some(o)=v.as_object(){ if let Some(sel)=o.get("selectionRange"){ let l=sel["start"]["line"].as_u64().unwrap_or(0) as u32; let c=sel["start"]["character"].as_u64().unwrap_or(0) as u32; out.push((l,c)); } if let Some(ch)=o.get("children"){ collect_pos(ch,out);} }} collect_pos(&doc,&mut positions); if let Some(m)=sym_limit{ positions.truncate(m); }
                                            for (l,c) in positions { if let Ok(res)=crate::serena_ra::references(std::path::Path::new(&root), &f, l, c).await { if let Some(arr)=res.as_array(){ total.fetch_add(arr.len(), std::sync::atomic::Ordering::Relaxed); } } }
                                        }
                                    })); }
                                    for t in tasks { let _=t.await; }
                                    let txt=format!("Workspace references estimated ~{} edges (created lazily via per-file tool)", total.load(std::sync::atomic::Ordering::Relaxed));
                                    let _ = state.graphiti.add_memory(AddMemoryRequest{ content: txt.clone(), name: Some("RA Workspace References Build".into()), source: Some("serena_kg".into()), memory_type: Some("audit".into()), metadata: None, group_id: None, timestamp: None }).await;
                                    let response=serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":txt}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena_kg.ra_references_to_edges" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, path:String, line:u32, character:u32, relation: Option<String> }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    // Query references via RA
                                    let res = crate::serena_ra::references(std::path::Path::new(&a.root), std::path::Path::new(&a.path), a.line, a.character).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let relation = a.relation.unwrap_or_else(|| "REFERENCES".to_string());
                                    // Determine source symbol at position for symbol-level edges
                                    let src_sym = crate::serena_ra::symbol_at_position(std::path::Path::new(&a.root), std::path::Path::new(&a.path), a.line, a.character).await.ok().flatten();
                                    #[derive(serde::Serialize)] struct EdgeLike{ source: String, target: String, relationship: String, src_symbol: Option<String>, dst_symbol: Option<String> }
                                    let mut edges: Vec<EdgeLike> = Vec::new();
                                    if let Some(arr)=res.as_array(){
                                        for it in arr {
                                            if let Some(loc)=it.get("uri").and_then(|u| u.as_str()){
                                                // Try definition (if available) to find target symbol name
                                                let dst_symbol = if let Some(loc_range)=it.get("range") { crate::serena_ra::find_target_symbol_name(std::path::Path::new(&a.root), loc, loc_range).await } else { None };
                                                if let Ok(url) = url::Url::parse(loc) {
                                                    if let Ok(p) = url.to_file_path() {
                                                        let target_path = p.to_string_lossy().to_string();
                                                        if target_path != a.path {
                                                            let src_symbol = src_sym.as_ref().map(|(n,_,_)| n.clone());
                                                            edges.push(EdgeLike{ source: a.path.clone(), target: target_path, relationship: relation.clone(), src_symbol, dst_symbol });
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // Persist as edges between file nodes (create file nodes if absent)
                                    let mut ok=0usize; let mut fail=0usize;
                                    // Upsert indexes: file_path -> node_id, and code entity (file_path+name+range) -> node_id
                                    let all_nodes = self.storage.get_all_nodes().await.unwrap_or_default();
                                    let mut file_index: std::collections::HashMap<String, Uuid> = std::collections::HashMap::new();
                                    let mut symbol_index: std::collections::HashMap<(String,String,(u32,u32)), Uuid> = std::collections::HashMap::new();
                                    for n in all_nodes {
                                        let props = n.properties();
                                        if let Some(fp) = props.get("file_path").and_then(|v| v.as_str()) {
                                            let is_code_entity = props.get("code_entity").and_then(|v| v.as_bool()).unwrap_or(false);
                                            if is_code_entity {
                                                // CodeEntity nodes carry line_range & name
                                                let name = props.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                                let lr = props.get("line_range").and_then(|v| v.as_array()).and_then(|arr| if arr.len()==2 { Some((arr[0].as_u64().unwrap_or(1) as u32, arr[1].as_u64().unwrap_or(1) as u32)) } else { None });
                                                if let Some(r)=lr { symbol_index.insert((fp.to_string(), name, r), *n.id()); }
                                            } else {
                                                file_index.insert(fp.to_string(), *n.id());
                                            }
                                        }
                                    }
                                    // Simple edge de-dup within current run: avoid creating identical edges. Also skip if existing identical edge in storage (best-effort)
                                    use std::collections::HashSet;
                                    let mut seen: HashSet<(Uuid,Uuid,String,Option<String>,Option<String>)> = HashSet::new();
                                    let mut existing: HashSet<(Uuid,Uuid,String,Option<String>,Option<String>)> = HashSet::new();
                                    if let Ok(edges_all) = self.storage.get_all_edges().await {
                                        for e0 in edges_all { let sig=(e0.source_id, e0.target_id, e0.relationship.clone(), e0.properties.get("src_symbol").and_then(|v|v.as_str()).map(|s|s.to_string()), e0.properties.get("dst_symbol").and_then(|v|v.as_str()).map(|s|s.to_string())); existing.insert(sig); }
                                    }
                                    for e in &edges {
                                        let now = chrono::Utc::now();
                                        let temporal = TemporalMetadata{ created_at: now, valid_from: now, valid_to: None, expired_at: None};
                                        // Prefer symbol-level nodes if available; else fallback to file nodes
                                        let src_id = if let Some(sym)=&e.src_symbol {
                                            // naive: search any CodeEntity node matching (file_path, name, any range) — here we don't have range，fall back to file node
                                            if let Some(id) = symbol_index.keys().find(|(fp,n,_r)| fp==&e.source && n==sym).and_then(|k| symbol_index.get(k)) { *id } else {
                                                if let Some(id)=file_index.get(&e.source){ *id } else { let node = EntityNode{ id: Uuid::new_v4(), name: e.source.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": e.source, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; file_index.insert(e.source.clone(), node.id); node.id }
                                            }
                                        } else { if let Some(id)=file_index.get(&e.source){ *id } else { let node = EntityNode{ id: Uuid::new_v4(), name: e.source.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": e.source, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; file_index.insert(e.source.clone(), node.id); node.id } };

                                        let dst_id = if let Some(sym)=&e.dst_symbol {
                                            if let Some(id)=symbol_index.keys().find(|(fp,n,_r)| fp==&e.target && n==sym).and_then(|k| symbol_index.get(k)) { *id } else {
                                                if let Some(id)=file_index.get(&e.target){ *id } else { let node = EntityNode{ id: Uuid::new_v4(), name: e.target.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": e.target, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; file_index.insert(e.target.clone(), node.id); node.id }
                                            }
                                        } else { if let Some(id)=file_index.get(&e.target){ *id } else { let node = EntityNode{ id: Uuid::new_v4(), name: e.target.clone(), entity_type: CodeEntityType::Module, labels: vec!["File".into()], properties: serde_json::json!({"file_path": e.target, "node_type":"file"}), temporal: temporal.clone(), embedding: None }; let _=self.storage.create_node(&node).await; file_index.insert(e.target.clone(), node.id); node.id } };
                                        let mut props = serde_json::json!({"source":"rust-analyzer"});
                                        if let Some(sym)=&e.src_symbol { props["src_symbol"]=serde_json::json!(sym); }
                                        if let Some(sym)=&e.dst_symbol { props["dst_symbol"]=serde_json::json!(sym); }
                                        let sig = (src_id, dst_id, e.relationship.clone(), e.src_symbol.clone(), e.dst_symbol.clone());
                                        if seen.contains(&sig) || existing.contains(&sig) { continue; }
                                        seen.insert(sig);
                                        let edge = graphiti_core::graph::Edge{ id: Uuid::new_v4(), source_id: src_id, target_id: dst_id, relationship: e.relationship.clone(), properties: props, temporal: temporal.clone(), weight: 1.0 };
                                        match self.storage.create_edge(&edge).await { Ok(_) => ok+=1, Err(_) => fail+=1 }
                                    }
                                    let text = format!("Created {} reference edges ({} failed)", ok, fail);
                                    let _ = state.graphiti.add_memory(AddMemoryRequest{ content: text.clone(), name: Some("RA References Import".into()), source: Some("serena_kg".into()), memory_type: Some("audit".into()), metadata: None, group_id: None, timestamp: None }).await;
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":text}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena.ra.workspace_symbol" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, query:String }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let res = crate::serena_ra::workspace_symbol(std::path::Path::new(&a.root), &a.query).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": res}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena.ra.references" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, path:String, line:u32, character:u32 }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let res = match crate::serena_ra::references(std::path::Path::new(&a.root), std::path::Path::new(&a.path), a.line, a.character).await {
                                        Ok(v)=> v,
                                        Err(e)=>{
                                            // Fallback: regex search for references by name under root, if we can locate symbol
                                            if let Ok(Some((name,_k,_r))) = crate::serena_ra::symbol_at_position(std::path::Path::new(&a.root), std::path::Path::new(&a.path), a.line, a.character).await {
                                                let hits = crate::serena_symbols::find_references(&crate::serena_symbols::FindRefsRequest{ root: a.root.clone().into(), name }).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                                let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": {"fallback":"regex","references": hits, "error": e.to_string()}}]}});
                                                return Ok(Json(response));
                                            }
                                            return Err(StatusCode::INTERNAL_SERVER_ERROR);
                                        }
                                    };
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": res}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena.ra.definition" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String, path:String, line:u32, character:u32 }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let res = match crate::serena_ra::definition(std::path::Path::new(&a.root), std::path::Path::new(&a.path), a.line, a.character).await {
                                        Ok(v)=>v,
                                        Err(e)=>{
                                            // Fallback: best-effort symbol name lookup
                                            if let Ok(Some((name,_k,_r))) = crate::serena_ra::symbol_at_position(std::path::Path::new(&a.root), std::path::Path::new(&a.path), a.line, a.character).await {
                                                let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": {"fallback":"regex","symbol": name, "error": e.to_string()}}]}});
                                                return Ok(Json(response));
                                            }
                                            return Err(StatusCode::INTERNAL_SERVER_ERROR);
                                        }
                                    };
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": res}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena.ra.status" => {
                                let root = params.get("arguments").and_then(|a| a.get("root")).and_then(|r| r.as_str()).map(|s| std::path::PathBuf::from(s));
                                let json = if let Some(r)=root { crate::serena_ra::status(Some(r.as_path())).await } else { crate::serena_ra::status(None).await };
                                let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": json}]}});
                                Ok(Json(response))
                            }
                            "serena.ra.restart" => {
                                if let Some(args) = params.get("arguments") {
                                    #[derive(serde::Deserialize)] struct A{ root:String }
                                    let a:A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    match crate::serena_ra::shutdown(std::path::Path::new(&a.root)).await { Ok(_) => {
                                        let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":"ok"}]}});
                                        Ok(Json(response))
                                    }, Err(_)=> Err(StatusCode::INTERNAL_SERVER_ERROR) }
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena.activate_project" => {
                                #[derive(serde::Deserialize)] struct A{ project: Option<String>, name: String }
                                if let Some(args) = params.get("arguments") {
                                    let a: A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let root = crate::serena_config::detect_project_root(a.project.as_deref().map(std::path::Path::new)).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let mut cfg = crate::serena_config::load_or_default(&root)
                                        .map_err(|err| {
                                            error!(error = %err, "读取项目配置失败");
                                            StatusCode::INTERNAL_SERVER_ERROR
                                        })?;
                                    cfg.active_project = Some(a.name);
                                    crate::serena_config::save(&root, &cfg).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":"ok"}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena.switch_modes" => {
                                #[derive(serde::Deserialize)] struct A{ project: Option<String>, modes: Vec<String> }
                                if let Some(args) = params.get("arguments") {
                                    let a: A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let root = crate::serena_config::detect_project_root(a.project.as_deref().map(std::path::Path::new)).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let mut cfg = crate::serena_config::load_or_default(&root)
                                        .map_err(|err| {
                                            error!(error = %err, "读取项目配置失败");
                                            StatusCode::INTERNAL_SERVER_ERROR
                                        })?;
                                    cfg.modes = a.modes;
                                    crate::serena_config::save(&root, &cfg).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":"ok"}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            "serena.check_onboarding_performed" => {
                                #[derive(serde::Deserialize)] struct A{ project: Option<String> }
                                let project = params.get("arguments").and_then(|a| serde_json::from_value::<A>(a.clone()).ok()).and_then(|a| a.project);
                                let root = crate::serena_config::detect_project_root(project.as_deref().map(std::path::Path::new)).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                let cfg = crate::serena_config::load_or_default(&root)
                                    .map_err(|err| {
                                        error!(error = %err, "读取项目配置失败");
                                        StatusCode::INTERNAL_SERVER_ERROR
                                    })?;
                                let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"json","json": {"onboarding_performed": cfg.onboarding_performed}}]}});
                                Ok(Json(response))
                            }
                            "serena.onboarding" => {
                                #[derive(serde::Deserialize)] struct A{ project: Option<String> }
                                if let Some(args) = params.get("arguments") {
                                    let a: A = serde_json::from_value(args.clone()).map_err(|_| StatusCode::BAD_REQUEST)?;
                                    let root = crate::serena_config::detect_project_root(a.project.as_deref().map(std::path::Path::new)).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let mut cfg = crate::serena_config::load_or_default(&root)
                                        .map_err(|err| {
                                            error!(error = %err, "读取项目配置失败");
                                            StatusCode::INTERNAL_SERVER_ERROR
                                        })?;
                                    cfg.onboarding_performed = true;
                                    crate::serena_config::save(&root, &cfg).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                                    let response = serde_json::json!({"jsonrpc":"2.0","id":request.get("id"),"result":{"content":[{"type":"text","text":"ok"}]}});
                                    Ok(Json(response))
                                } else { Err(StatusCode::BAD_REQUEST) }
                            }
                            // Rust Analyzer based symbols and references
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
*/
