//! Server startup and HTTP/TLS configuration
use crate::handlers::http::*;
use crate::handlers::mcp::mcp_handler;
use crate::learning::LearningConfig;
use crate::learning::{
    ConsoleNotificationChannel, LearningDetector, LearningNotification, MCPNotificationChannel,
    NotificationManager, SmartLearningDetector,
};
use crate::learning_endpoints::*;
use crate::middleware::auth::auth_guard;
use crate::middleware::rate_limit::rate_limit_guard;
use crate::project_scanner::ProjectScanner;
use crate::services::graphiti::RealGraphitiService;
use crate::types::AppState;
use crate::types::GraphitiService;
use crate::types::ServerConfig;
use crate::types::ServerConfig as FullServerConfig;
use crate::utils::build_cors_layer;
use axum::middleware;
use axum::routing::get;
use axum::routing::post;
use axum::Router;
use axum_server::tls_rustls::RustlsConfig;
use graphiti_llm::LLMConfig;
use metrics_exporter_prometheus::PrometheusHandle;
use once_cell::sync::OnceCell;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tower::limit::GlobalConcurrencyLimitLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::DefaultMakeSpan;
use tower_http::trace::TraceLayer;
use tracing::info;

/// Build the main application router
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/mcp", post(mcp_handler)) // Use standard MCP handler with all tools
        .route("/notifications", get(learning_notifications_stream)) // Learning notifications stream
        .route("/notifications/active", get(get_active_notifications)) // Get active notifications
        .route("/notifications/:id/dismiss", post(dismiss_notification)) // Dismiss notification
        .route("/health", get(health_check))
        .route("/ready", get(ready_check))
        // Protected write endpoints
        .route("/memory", post(add_memory))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth_guard))
        .route("/memory/search", get(search_memory))
        .route("/memory/search", post(search_memory_json))
        .route("/memory/:id", get(get_memory))
        .route("/memory/:id/related", get(get_related))
        .route("/metrics", get(metrics))
        .layer(TraceLayer::new_for_http().make_span_with(DefaultMakeSpan::default()))
        .layer(build_cors_layer(&state.config.server))
        // Core hardening stack
        .layer(GlobalConcurrencyLimitLayer::new(
            state.config.server.max_connections,
        ))
        .layer(TimeoutLayer::new(Duration::from_secs(
            state.config.server.request_timeout_seconds,
        )))
        .layer(RequestBodyLimitLayer::new(
            state.config.server.request_body_limit_bytes,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            rate_limit_guard,
        ))
        .with_state(state)
}

/// Start HTTP server (non-TLS)
pub async fn start_http_server(addr: SocketAddr, app: Router) -> anyhow::Result<()> {
    info!("Starting MCP server on {}", addr);
    let listener = TcpListener::bind(&addr).await?;
    let server = axum::serve(listener, app);
    let graceful = async move {
        tokio::select! {
            res = server => res?,
            _ = shutdown_signal() => {}
        }
        Ok::<(), anyhow::Error>(())
    };
    graceful.await
}

/// Start HTTPS server with TLS
pub async fn start_https_server(
    addr: SocketAddr,
    app: Router,
    tls_config: ServerConfig,
) -> anyhow::Result<()> {
    if let Some(tls) = &tls_config.server.tls {
        info!("Starting MCP server with TLS on {}", addr);
        let rustls_config = RustlsConfig::from_pem_file(&tls.cert_path, &tls.key_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to load TLS certs: {}", e))?;
        let server = axum_server::bind_rustls(addr, rustls_config).serve(app.into_make_service());
        let graceful = async move {
            tokio::select! {
                res = server => res?,
                _ = shutdown_signal() => {}
            }
            Ok::<(), anyhow::Error>(())
        };
        graceful.await
    } else {
        // Fallback to HTTP if no TLS config
        start_http_server(addr, app).await
    }
}

/// Graceful shutdown signal handler for production
async fn shutdown_signal() {
    // Listen for SIGINT, SIGTERM and Ctrl+C
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        sigterm.recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("Shutdown signal received. Shutting down gracefully...");
}

/// Initialize services similar to legacy main
pub async fn initialize_services(
    config: FullServerConfig,
) -> anyhow::Result<(
    Arc<dyn GraphitiService>,
    Arc<graphiti_cozo::CozoDriver>,
    Arc<dyn LearningDetector>,
    Arc<NotificationManager>,
    broadcast::Receiver<LearningNotification>,
    Arc<ProjectScanner>,
)> {
    // Initialize storage
    let cozo_config: graphiti_cozo::CozoConfig = config.cozo.clone().into();
    let storage = graphiti_cozo::CozoDriver::new(cozo_config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize CozoDB: {}", e))?;

    // LLM/Embedder config
    let _llm_config: LLMConfig = config.llm.clone().into();

    // Create base service
    let storage_arc = Arc::new(storage);
    let base_service = RealGraphitiService::new(
        storage_arc.clone(),
        config.embedder.clone(),
        config.graphiti.clone(),
    )
    .await?;

    // Initialize learning
    let learning_cfg = LearningConfig::default();
    let detector: Arc<dyn LearningDetector> =
        Arc::new(SmartLearningDetector::new(learning_cfg.detector));
    let notification_manager = Arc::new(NotificationManager::new(learning_cfg.notifications));
    let (mcp_channel, notification_receiver) = MCPNotificationChannel::new();
    notification_manager
        .add_channel(Box::new(mcp_channel))
        .await;
    notification_manager
        .add_channel(Box::new(ConsoleNotificationChannel))
        .await;

    // Wrap service with learning
    let learning_aware_service = Arc::new(
        crate::learning_integration::LearningAwareGraphitiService::new(
            Arc::new(base_service),
            detector.clone(),
            notification_manager.clone(),
        ),
    );

    // Project scanner
    let project_scanner = Arc::new(ProjectScanner::new(
        learning_aware_service.clone() as Arc<dyn GraphitiService>
    ));

    Ok((
        learning_aware_service,
        storage_arc,
        detector,
        notification_manager,
        notification_receiver,
        project_scanner,
    ))
}

/// Initialize Prometheus exporter (extracted)
static METRICS_EXPORTER: OnceCell<PrometheusHandle> = OnceCell::new();

pub fn init_metrics() -> anyhow::Result<()> {
    use metrics_exporter_prometheus::PrometheusBuilder;
    if METRICS_EXPORTER.get().is_some() {
        return Ok(());
    }
    let handle = PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let _ = METRICS_EXPORTER.set(handle);
    Ok(())
}

pub fn render_metrics() -> String {
    METRICS_EXPORTER
        .get()
        .map(|h| h.render())
        .unwrap_or_default()
}
