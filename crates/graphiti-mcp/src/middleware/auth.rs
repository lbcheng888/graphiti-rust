//! Authentication middleware

use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use axum::extract::State;
use axum::body::Body;

/// Authentication guard middleware
pub async fn auth_guard(
    State(state): State<crate::types::AppState>,
    headers: HeaderMap,
    req: axum::http::Request<Body>,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    if state.require_auth {
        let expected = match &state.auth_token {
            Some(v) => v,
            None => return Err(StatusCode::UNAUTHORIZED),
        };
        if !is_authorized(&headers, expected) {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }
    Ok(next.run(req).await)
}

/// Check if request is authorized
fn is_authorized(headers: &HeaderMap, expected: &str) -> bool {
    if let Some(val) = headers.get(axum::http::header::AUTHORIZATION) {
        if let Ok(s) = val.to_str() {
            // Support: Bearer <token> or raw token
            let bearer = format!("Bearer {}", expected);
            return s == expected || s == bearer;
        }
    }
    false
}