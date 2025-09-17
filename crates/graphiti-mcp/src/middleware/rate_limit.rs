//! Rate limiting middleware

use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;

/// Rate limit guard middleware
pub async fn rate_limit_guard(
    State(state): State<crate::types::AppState>,
    req: axum::http::Request<Body>,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    if state.rate_limiter.check().is_err() {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    Ok(next.run(req).await)
}
