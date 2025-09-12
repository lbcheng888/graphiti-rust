//! Graphiti MCP Server - Main Entry Point

use crate::config::{init_tracing, load_config};
use crate::types::*;
use crate::server::{build_router, start_http_server, start_https_server, initialize_services};
use crate::types::AppState;

use axum::extract::State;
use axum::response::Json;
// use chrono::DateTime; // 暂时未使用
use clap::Parser;

// use graphiti_core::graph::EntityNode; // removed duplicate import (already imported above)
use graphiti_llm::EmbeddingProvider;
// use graphiti_llm::LLMClient; // 暂时未使用
use governor::Quota;
use governor::RateLimiter;
// use nonzero_ext::nonzero; // no longer used
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
// Tower core layers for production hardening
use tracing::info;
use tracing::warn;
// Note: We keep local implementations here to avoid large refactor collisions.
// keep local handlers in this file for now to avoid refactor churn
// NonZeroU32 is not used; remove import to silence warning
use std::sync::Arc as StdArc;
use std::num::NonZeroU32;
// metrics middleware removed for compatibility; keep exporter via init_metrics
use metrics::{counter, histogram};

// Learning system imports
// use crate::handlers::serena as serena_handlers; // not used after split

// Limiter type is defined in crate::types; avoid duplicate alias here

// 使用统一的 Args 定义（crate::types::Args），删除本地重复定义

// Types moved to models module - use those instead

// GraphitiService trait moved to types module

// Request/Response types moved to models module

// All types moved to models module

// Default functions moved to utils module

#[allow(unused_assignments)]
pub fn legacy_main() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(async {
        // Parse command line arguments
        let args = Args::parse();

        // Initialize tracing
        init_tracing(&args.log_level)?;

        // Load configuration with project isolation support
        let config = load_config(&args)?;

        // Initialize services
        info!("Initializing Graphiti services...");
        // Emit startup diagnostics for embedding provider to aid debugging
        match config.embedder.provider {
            EmbeddingProvider::EmbedAnything => {
                info!("Embedding provider: embed_anything (HF model) — model={}, dim={}", config.embedder.model, config.embedder.dimension);
            }
            EmbeddingProvider::GemmaCandleApprox => {
                info!("Embedding provider: gemma_candle (native approximate) — tokenizer-only, dim={}", config.embedder.dimension);
                if std::env::var("EMBEDDING_MODEL_DIR").is_err() && config.embedder.cache_dir.is_none() {
                    warn!("EMBEDDING_MODEL_DIR not set and cache_dir not provided; ensure tokenizer.json is accessible");
                }
            }
            _ => {
                info!("Embedding provider: other ({:?})", config.embedder.provider);
            }
        }

        let (
            graphiti_service,
            storage,
            learning_detector,
            notification_manager,
            notification_receiver,
            project_scanner,
        ) = initialize_services(config.clone()).await?;

        // Create application state
        let state = AppState {
            graphiti: graphiti_service.clone(),
            storage: storage.clone(),
            learning_detector,
            notification_manager,
            notification_receiver: Arc::new(tokio::sync::RwLock::new(Some(notification_receiver))),
            project_scanner: project_scanner.clone(),
            rate_limiter: StdArc::new(RateLimiter::direct(Quota::per_second(
                NonZeroU32::new(config.server.requests_per_second).unwrap_or(NonZeroU32::new(1).unwrap()),
            ))),
            auth_token: std::env::var("GRAPHITI_AUTH_TOKEN").ok(),
            require_auth: config.server.require_auth,
            config: config.clone(),
        };

        // If authentication is required but no token provided, abort startup
        // In stdio mode, skip HTTP auth requirement checks
        if !args.stdio && state.require_auth && state.auth_token.is_none() {
            anyhow::bail!(
                "GRAPHITI_AUTH_TOKEN is required but not set (server.require_auth=true)."
            );
        }

        // Perform initial project scan only when explicitly enabled to avoid blocking stdio startup
        let enable_initial_scan = std::env::var("GRAPHITI_ENABLE_INITIAL_SCAN")
            .ok()
            .as_deref()
            == Some("1");
        if enable_initial_scan {
            if let Ok(current_dir) = std::env::current_dir() {
                if project_scanner.needs_scan(&current_dir).await {
                    info!("Performing initial project scan...");
                    tokio::spawn({
                        let scanner = project_scanner.clone();
                        let _service = graphiti_service.clone();
                        async move {
                            match scanner.scan_project(&current_dir).await {
                                Ok(result) => {
                                    info!(
                                        "Initial project scan completed: {} files, {} entities, {} memories",
                                        result.files_scanned, result.entities_added, result.memories_added
                                    );
                                    scanner.mark_scanned(&current_dir).await;
                                }
                                Err(e) => {
                                    warn!("Initial project scan failed: {}", e);
                                }
                            }
                        }
                    });
                } else {
                    info!("Project already scanned, skipping initial scan");
                }
            }
        } else {
            info!("Initial project scan disabled (set GRAPHITI_ENABLE_INITIAL_SCAN=1 to enable)");
        }

        // Build router using the server module
        let app = build_router(state.clone());

        // Initialize Prometheus exporter even in stdio mode (no /metrics endpoint there, but metrics are recorded)
        let _ = crate::server::init_metrics();

        // Check if running in stdio mode
        if args.stdio {
            info!("Starting MCP server in stdio mode");

            // Run stdio MCP server; support both LSP-style (Content-Length) and NDJSON framing
            use tokio::io::stdin;
            use tokio::io::stdout;
            use tokio::io::AsyncBufReadExt;
            use tokio::io::AsyncReadExt;
            use tokio::io::AsyncWriteExt;
            use tokio::io::BufReader;

            let stdin = stdin();
            let mut stdout = stdout();
            let mut reader = BufReader::new(stdin);
            use tokio::sync::Semaphore;

            // None => auto-detect; Some(true) => NDJSON; Some(false) => LSP (Content-Length)
            let mut use_ndjson: Option<bool> = match std::env::var("GRAPHITI_STDIN_FRAMING").ok().as_deref() {
                Some("ndjson") | Some("NDJSON") => Some(true),
                Some("lsp") | Some("LSP") => Some(false),
                _ => None,
            };

            // Concurrency guard for stdio processing
            let stdio_sema = Arc::new(Semaphore::new(config.server.max_connections));
            let stdio_timeout_secs = args
                .stdio_timeout
                .unwrap_or(config.server.request_timeout_seconds);

            loop {
                // Obtain one request either via NDJSON line or LSP headers+body
                let mut request_value: Option<serde_json::Value> = None;
                let mut original_request_id_present: bool = false;
                let mut original_method: Option<String> = None;

                // If already in NDJSON mode, read a single line and parse
                if matches!(use_ndjson, Some(true)) {
                    let mut line = String::new();
                    match reader.read_line(&mut line).await {
                        Ok(0) => return Ok(()), // EOF
                        Ok(_) => {
                            let trimmed = line.trim();
                            if trimmed.is_empty() { continue; }
                            match serde_json::from_str::<serde_json::Value>(trimmed) {
                                Ok(v) => request_value = Some(v),
                                Err(e) => {
                                    eprintln!("Parse error (NDJSON): {}", e);
                                    continue;
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Error reading NDJSON line from stdin: {}", e);
                            return Ok(());
                        }
                    }
                } else {
                    // Read headers until blank line; if the very first line looks like JSON, switch to NDJSON
                    let mut headers_raw = String::new();
                    let mut first_line_checked = false;
                    let mut ndjson_line: Option<String> = None;

                    // 1) Read headers or detect NDJSON
                    loop {
                        let mut line = String::new();
                        match reader.read_line(&mut line).await {
                            Ok(0) => {
                                // EOF
                                return Ok(());
                            }
                            Ok(_) => {
                                if !first_line_checked {
                                    first_line_checked = true;
                                    let ls = line.trim_start();
                                    if ls.starts_with('{') {
                                        // NDJSON detected
                                        use_ndjson = Some(true);
                                        ndjson_line = Some(line);
                                        break;
                                    }
                                }
                                // Header section ends with a blank line
                                if line == "\r\n" || line == "\n" {
                                    break;
                                }
                                headers_raw.push_str(&line);
                            }
                            Err(e) => {
                                eprintln!("Error reading headers from stdin: {}", e);
                                return Ok(());
                            }
                        }
                    }

                    if ndjson_line.is_some() {
                        // Parse the NDJSON line we just captured
                        let line = ndjson_line.take().unwrap();
                        let trimmed = line.trim();
                        if trimmed.is_empty() { continue; }
                        match serde_json::from_str::<serde_json::Value>(trimmed) {
                            Ok(v) => request_value = Some(v),
                            Err(e) => {
                                eprintln!("Parse error (NDJSON first line): {}", e);
                                continue;
                            }
                        }
                    } else {
                        if headers_raw.is_empty() {
                            // Spurious blank line; continue waiting for next message
                            continue;
                        }

                        // 2) Parse Content-Length header (case-insensitive)
                        let mut content_length: Option<usize> = None;
                        for line in headers_raw.lines() {
                            let lower = line.to_ascii_lowercase();
                            if let Some(rest) = lower.strip_prefix("content-length:") {
                                let len_str = rest.trim();
                                if let Ok(len) = len_str.parse::<usize>() {
                                    content_length = Some(len);
                                    break;
                                }
                            }
                        }

                        let len = match content_length {
                            Some(v) => v,
                            None => {
                                eprintln!("Missing Content-Length header; skipping message");
                                continue;
                            }
                        };

                        // 3) Read exact body bytes
                        let mut body = vec![0u8; len];
                        if let Err(e) = reader.read_exact(&mut body).await {
                            eprintln!("Error reading request body: {}", e);
                            return Ok(());
                        }

                        // 4) Parse JSON-RPC request
                        match serde_json::from_slice::<serde_json::Value>(&body) {
                            Ok(v) => {
                                request_value = Some(v);
                                // Mark we are in LSP framing mode
                                if use_ndjson.is_none() { use_ndjson = Some(false); }
                            }
                            Err(e) => {
                                eprintln!("Parse error (LSP body): {}", e);
                                continue;
                            }
                        }
                    }
                }

                // 5) Dispatch request with timeout
                let response_value = if let Some(request) = request_value {
                    original_request_id_present = request.get("id").is_some() && !request.get("id").unwrap().is_null();
                    original_method = request.get("method").and_then(|m| m.as_str()).map(|s| s.to_string());

                    // Metrics: request count
                    if let Some(ref m) = original_method {
                        counter!("mcp_requests_total", "method" => m.clone()).increment(1);
                    } else {
                        counter!("mcp_requests_total", "method" => "unknown").increment(1);
                    }

                    // Concurrency
                    let _permit = stdio_sema.acquire().await.expect("semaphore poisoned");

                    let start = std::time::Instant::now();
                    let timeout = Duration::from_secs(stdio_timeout_secs);
                    let result = tokio::time::timeout(
                        timeout,
                        crate::handlers::mcp::mcp_handler(State(state.clone()), Json(request)),
                    )
                    .await;

                    let elapsed = start.elapsed().as_secs_f64();
                    if let Some(ref m) = original_method {
                        histogram!("mcp_request_duration_seconds", "method" => m.clone()).record(elapsed);
                    } else {
                        histogram!("mcp_request_duration_seconds", "method" => "unknown").record(elapsed);
                    }

                    match result {
                        Ok(Ok(Json(response))) => response,
                        Ok(Err(e)) => serde_json::json!({
                            "jsonrpc": "2.0",
                            "error": {"code": -32000, "message": format!("Error: {:?}", e)}
                        }),
                        Err(_elapsed) => serde_json::json!({
                            "jsonrpc": "2.0",
                            "error": {"code": -32001, "message": "Request timed out"}
                        }),
                    }
                } else {
                    continue;
                };

                // 6) Write response matching the detected framing
                // Skip responding for notifications (no id) or explicit "initialized" notification
                let is_notification = !original_request_id_present
                    || matches!(original_method.as_deref(), Some("initialized"));
                if !is_notification {
                    let response_json = serde_json::to_string(&response_value)?;
                    match use_ndjson {
                        Some(true) => {
                            // NDJSON style
                            stdout.write_all(response_json.as_bytes()).await?;
                            stdout.write_all(b"\n").await?;
                            stdout.flush().await?;
                        }
                        _ => {
                            // Default to LSP-style with Content-Length header
                            let header = format!(
                                "Content-Length: {}\r\nContent-Type: application/json; charset=utf-8\r\n\r\n",
                                response_json.as_bytes().len()
                            );
                            stdout.write_all(header.as_bytes()).await?;
                            stdout.write_all(response_json.as_bytes()).await?;
                            stdout.flush().await?;
                        }
                    }
                }
            }
        } else {
            // Start HTTP/TLS server - use config values if not overridden by command line
            let host = if args.host == "0.0.0.0" {
                &config.server.host
            } else {
                &args.host
            };
            let port = if args.port == 8080 {
                config.server.port
            } else {
                args.port
            };
            let addr = SocketAddr::from((host.parse::<std::net::IpAddr>()?, port));

            // Initialize Prometheus exporter
            let _ = crate::server::init_metrics()?;
            info!("/metrics enabled");

            // TLS optional
            if let Some(_tls) = &config.server.tls {
                info!("Starting MCP server with TLS on {}", addr);
                start_https_server(addr, app, config).await?;
            } else {
                info!("Starting MCP server on {}", addr);
                start_http_server(addr, app).await?;
            }
        }

        Ok(())
    })
}
