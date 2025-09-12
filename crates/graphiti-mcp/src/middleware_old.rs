use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue, Request as HttpRequest},
    middleware,
    response::Response,
};
use tower_http::cors::{Any, CorsLayer};
use crate::types::{AppState, ServerSettings};
use axum::http::StatusCode;

pub fn build_cors_layer(server: &ServerSettings) -> CorsLayer {
    let mut layer = CorsLayer::new()
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
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
        layer = layer.allow_origin(Any);
    }

    layer
}

pub async fn rate_limit_guard(
    axum::extract::State(state): axum::extract::State<AppState>,
    req: HttpRequest<Body>,
    next: middleware::Next,
) -> std::result::Result<Response, StatusCode> {
    if state.rate_limiter.check().is_err() {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    Ok(next.run(req).await)
}

pub async fn auth_guard(
    axum::extract::State(state): axum::extract::State<AppState>,
    req: HttpRequest<Body>,
    next: middleware::Next,
) -> std::result::Result<Response, StatusCode> {
    if state.require_auth {
        let expected = match &state.auth_token {
            Some(v) => v,
            None => return Err(StatusCode::UNAUTHORIZED),
        };
        let headers = req.headers();
        if !is_authorized(headers, expected) {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }
    Ok(next.run(req).await)
}

fn is_authorized(headers: &HeaderMap, expected: &str) -> bool {
    if let Some(val) = headers.get(axum::http::header::AUTHORIZATION) {
        if let Ok(s) = val.to_str() {
            let bearer = format!("Bearer {}", expected);
            return s == expected || s == bearer;
        }
    }
    false
}

