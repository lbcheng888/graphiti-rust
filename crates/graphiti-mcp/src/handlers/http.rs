//! HTTP endpoint handlers

use crate::types::*;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use tracing::error;
use uuid::Uuid;

/// Health check endpoint
pub async fn health_check() -> StatusCode {
    StatusCode::OK
}

/// Readiness probe endpoint
pub async fn ready_check(State(state): State<AppState>) -> StatusCode {
    // Basic readiness check
    if state.graphiti.get_episodes(1).await.is_err() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }
    StatusCode::OK
}

/// Metrics endpoint
pub async fn metrics() -> String {
    crate::server::render_metrics()
}

/// Add memory endpoint
pub async fn add_memory(
    State(state): State<AppState>,
    Json(req): Json<AddMemoryRequest>,
) -> std::result::Result<Json<AddMemoryResponse>, StatusCode> {
    match state.graphiti.add_memory(req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => {
            error!("Failed to add memory: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Search memory endpoint
pub async fn search_memory(
    State(state): State<AppState>,
    Query(req): Query<SearchMemoryRequest>,
) -> std::result::Result<Json<SearchMemoryResponse>, StatusCode> {
    match state.graphiti.search_memory(req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => {
            error!("Failed to search memory: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Search memory endpoint (POST JSON body)
pub async fn search_memory_json(
    State(state): State<AppState>,
    Json(req): Json<SearchMemoryRequest>,
) -> std::result::Result<Json<SearchMemoryResponse>, StatusCode> {
    match state.graphiti.search_memory(req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(e) => {
            error!("Failed to search memory (json): {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Get memory by ID endpoint
pub async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> std::result::Result<Json<MemoryNode>, StatusCode> {
    match state.graphiti.get_memory(id).await {
        Ok(Some(memory)) => Ok(Json(memory)),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            error!("Failed to get memory: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Get related memories endpoint
pub async fn get_related(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> std::result::Result<Json<Vec<RelatedMemory>>, StatusCode> {
    let depth = params
        .get("depth")
        .and_then(|d| d.parse().ok())
        .unwrap_or(1);

    match state.graphiti.get_related(id, depth).await {
        Ok(related) => Ok(Json(related)),
        Err(e) => {
            error!("Failed to get related memories: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
