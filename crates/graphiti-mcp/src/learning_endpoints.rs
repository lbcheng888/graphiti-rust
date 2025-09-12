//! Learning notification endpoints for MCP server

use crate::types::AppState;
use crate::handlers::mcp::mcp_handler;
use crate::types::{AddMemoryRequest, SearchMemoryRequest};
// 使用统一的类型定义，避免与 models::* 产生歧义
// （本文件并未直接使用这些请求类型，去除不必要导入）
use crate::learning::LearningNotification;
use axum::extract::Path;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::Event;
use axum::response::sse::KeepAlive;
use axum::response::Json;
use axum::response::Sse;
use futures_util::stream::Stream;

use serde_json;
use std::convert::Infallible;
use std::time::Duration;
use tokio::time::interval;

use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;
use uuid::Uuid;

/// Server-Sent Events stream for learning notifications
pub async fn learning_notifications_stream(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    info!("📡 Starting learning notifications stream");

    let stream = async_stream::stream! {
        // Get the notification receiver
        let mut receiver_opt = state.notification_receiver.write().await;
        let mut receiver = match receiver_opt.take() {
            Some(r) => r,
            None => {
                warn!("No notification receiver available");
                return;
            }
        };
        drop(receiver_opt);

        // Create a heartbeat interval
        let mut heartbeat = interval(Duration::from_secs(30));

        info!("🎧 Listening for learning notifications...");

        loop {
            tokio::select! {
                // Receive learning notifications
                result = receiver.recv() => {
                    match result {
                        Ok(notification) => {
                            debug!("📢 Broadcasting notification: {}", notification.title);

                            let event_data = match serde_json::to_string(&notification) {
                                Ok(data) => data,
                                Err(e) => {
                                    error!("Failed to serialize notification: {}", e);
                                    continue;
                                }
                            };

                            let event = Event::default()
                                .event("learning_notification")
                                .data(event_data)
                                .id(notification.id.to_string());

                            yield Ok(event);
                        },
                        Err(e) => {
                            warn!("Notification receiver error: {}", e);
                            break;
                        }
                    }
                },
                // Send periodic heartbeat
                _ = heartbeat.tick() => {
                    debug!("💓 Sending heartbeat");
                    let event = Event::default()
                        .event("heartbeat")
                        .data("ping");
                    yield Ok(event);
                }
            }
        }

        info!("📡 Learning notifications stream ended");
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Get all active notifications
pub async fn get_active_notifications(
    State(state): State<AppState>,
) -> Result<Json<ActiveNotificationsResponse>, StatusCode> {
    debug!("📋 Getting active notifications");

    let notifications = state.notification_manager.get_active_notifications().await;
    let stats = state.notification_manager.get_stats().await;

    let response = ActiveNotificationsResponse {
        notifications,
        total: stats.active_count,
        stats,
    };

    Ok(Json(response))
}

/// Dismiss a specific notification
pub async fn dismiss_notification(
    State(state): State<AppState>,
    Path(notification_id): Path<Uuid>,
) -> Result<Json<DismissNotificationResponse>, StatusCode> {
    debug!("🚫 Dismissing notification: {}", notification_id);

    match state.notification_manager.dismiss(notification_id).await {
        Ok(()) => {
            info!("✅ Notification dismissed: {}", notification_id);
            Ok(Json(DismissNotificationResponse {
                success: true,
                message: "Notification dismissed successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to dismiss notification {}: {}", notification_id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Response for active notifications endpoint
#[derive(serde::Serialize)]
pub struct ActiveNotificationsResponse {
    pub notifications: Vec<LearningNotification>,
    pub total: u32,
    pub stats: super::learning::NotificationStats,
}

/// Response for dismiss notification endpoint
#[derive(serde::Serialize)]
pub struct DismissNotificationResponse {
    pub success: bool,
    pub message: String,
}

/// Enhanced MCP handler with notification support
#[allow(dead_code)]
pub async fn enhanced_mcp_handler(
    State(state): State<AppState>,
    Json(request): Json<serde_json::Value>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    info!("MCP request: {}", request);

    // Handle standard MCP methods first
    if let Some(method) = request.get("method").and_then(|m| m.as_str()) {
        match method {
            "initialize" => {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "serverInfo": {
                            "name": "Graphiti Knowledge Graph with Learning Detection",
                            "version": "1.0.0"
                        },
                        "capabilities": {
                            "tools": {},
                            "prompts": {},
                            "resources": {},
                            "notifications": {
                                "learning_events": {
                                    "description": "Real-time learning event notifications",
                                    "stream_endpoint": "/notifications"
                                }
                            }
                        }
                    }
                });
                return Ok(Json(response));
            }
            "tools/list" => {
                let tools = vec![
                    serde_json::json!({
                        "name": "add_memory",
                        "description": "Add a new memory to the knowledge graph with learning detection",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "The content to remember"},
                                "source": {"type": "string", "description": "The source of the memory"}
                            },
                            "required": ["content"]
                        }
                    }),
                    serde_json::json!({
                        "name": "search_memory",
                        "description": "Search for memories in the knowledge graph",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "limit": {"type": "number", "default": 10, "description": "Maximum results"}
                            },
                            "required": ["query"]
                        }
                    }),
                    serde_json::json!({
                        "name": "get_learning_stats",
                        "description": "Get learning detection statistics",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }),
                    serde_json::json!({
                        "name": "get_active_notifications",
                        "description": "Get all active learning notifications",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }),
                ];

                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "tools": tools
                    }
                });
                return Ok(Json(response));
            }
            "tools/call" => {
                if let Some(params) = request.get("params") {
                    if let Some(tool_name) = params.get("name").and_then(|n| n.as_str()) {
                        return handle_enhanced_tool_call(&state, tool_name, params, &request)
                            .await;
                    }
                }
                return Err(StatusCode::BAD_REQUEST);
            }
            _ => {
                // Fallback to standard MCP handler for non-learning specific methods
                return mcp_handler(State(state), Json(request)).await;
            }
        }
    }

    Err(StatusCode::BAD_REQUEST)
}

/// Handle enhanced tool calls with learning features
async fn handle_enhanced_tool_call(
    state: &AppState,
    tool_name: &str,
    params: &serde_json::Value,
    request: &serde_json::Value,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    match tool_name {
        "add_memory" => {
            if let Some(args) = params.get("arguments") {
                let req: AddMemoryRequest = match serde_json::from_value(args.clone()) {
                    Ok(req) => req,
                    Err(_) => return Err(StatusCode::BAD_REQUEST),
                };

                // This will trigger learning detection automatically
                match state.graphiti.add_memory(req).await {
                    Ok(result) => {
                        let response = serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [{
                                    "type": "text",
                                    "text": format!("✅ Memory added successfully with ID: {}\n🧠 Learning detection is active - check notifications for new insights!", result.id)
                                }]
                            }
                        });
                        Ok(Json(response))
                    }
                    Err(e) => {
                        error!("Failed to add memory: {}", e);
                        Err(StatusCode::INTERNAL_SERVER_ERROR)
                    }
                }
            } else {
                Err(StatusCode::BAD_REQUEST)
            }
        }
        "search_memory" => {
            if let Some(args) = params.get("arguments") {
                let req: SearchMemoryRequest = match serde_json::from_value(args.clone()) {
                    Ok(req) => req,
                    Err(_) => return Err(StatusCode::BAD_REQUEST),
                };

                match state.graphiti.search_memory(req).await {
                    Ok(result) => {
                        let response = serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [{
                                    "type": "text",
                                    "text": format!("🔍 Found {} memories", result.total)
                                }]
                            }
                        });
                        Ok(Json(response))
                    }
                    Err(e) => {
                        error!("Failed to search memory: {}", e);
                        Err(StatusCode::INTERNAL_SERVER_ERROR)
                    }
                }
            } else {
                Err(StatusCode::BAD_REQUEST)
            }
        }
        "get_learning_stats" => {
            let detector_stats = match state.learning_detector.get_detector_stats().await {
                Ok(stats) => stats,
                Err(e) => {
                    error!("Failed to get detector stats: {}", e);
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
            };

            let notification_stats = state.notification_manager.get_stats().await;

            let response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [{
                        "type": "text",
                        "text": format!(
                            "📊 Learning Detection Statistics\n\
                            🔍 Total Analyses: {}\n\
                            🎯 Events Detected: {}\n\
                            📢 Notifications Sent: {}\n\
                            🎛️ Active Notifications: {}",
                            detector_stats.total_analyses,
                            detector_stats.events_detected,
                            notification_stats.total_sent,
                            notification_stats.active_count
                        )
                    }]
                }
            });
            Ok(Json(response))
        }
        "get_active_notifications" => {
            let notifications = state.notification_manager.get_active_notifications().await;
            let count = notifications.len();

            let notifications_text = if notifications.is_empty() {
                "No active notifications".to_string()
            } else {
                notifications
                    .iter()
                    .take(5) // Show first 5
                    .map(|n| format!("• {} ({})", n.title, n.level.emoji()))
                    .collect::<Vec<_>>()
                    .join("\n")
            };

            let response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [{
                        "type": "text",
                        "text": format!("📋 Active Notifications ({})\n{}", count, notifications_text)
                    }]
                }
            });
            Ok(Json(response))
        }
        _ => Err(StatusCode::NOT_FOUND),
    }
}
