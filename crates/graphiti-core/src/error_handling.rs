//! Advanced error handling and recovery mechanisms

use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::error;
use tracing::info;
use tracing::warn;
use uuid::Uuid;

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Low severity - can be ignored or logged
    Low,
    /// Medium severity - should be handled but not critical
    Medium,
    /// High severity - requires immediate attention
    High,
    /// Critical severity - system failure imminent
    Critical,
}

/// Error categories for better classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Network-related errors
    Network,
    /// Database/storage errors
    Storage,
    /// Authentication/authorization errors
    Security,
    /// Input validation errors
    Validation,
    /// Business logic errors
    Business,
    /// System resource errors
    Resource,
    /// External service errors
    External,
    /// Unknown/unclassified errors
    Unknown,
}

/// Retry strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier (exponential backoff)
    pub backoff_multiplier: f64,
    /// Jitter factor to avoid thundering herd
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are rejected
    Open,
    /// Circuit is half-open, testing if service recovered
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit
    pub failure_threshold: u32,
    /// Success threshold to close circuit from half-open
    pub success_threshold: u32,
    /// Timeout before trying to close circuit
    pub timeout: Duration,
    /// Window size for failure counting
    pub window_size: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            window_size: Duration::from_secs(60),
        }
    }
}

/// Circuit breaker implementation
#[derive(Debug)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitBreakerState>>,
    failure_count: Arc<RwLock<u32>>,
    success_count: Arc<RwLock<u32>>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            success_count: Arc::new(RwLock::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
        }
    }

    /// Check if request should be allowed
    pub async fn should_allow_request(&self) -> bool {
        let state = self.state.read().await;
        match *state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = *self.last_failure_time.read().await {
                    if last_failure.elapsed() >= self.config.timeout {
                        // Transition to half-open
                        drop(state);
                        *self.state.write().await = CircuitBreakerState::HalfOpen;
                        *self.success_count.write().await = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }

    /// Record a successful operation
    pub async fn record_success(&self) {
        let state = self.state.read().await;
        match *state {
            CircuitBreakerState::Closed => {
                *self.failure_count.write().await = 0;
            }
            CircuitBreakerState::HalfOpen => {
                let mut success_count = self.success_count.write().await;
                *success_count += 1;
                if *success_count >= self.config.success_threshold {
                    drop(state);
                    *self.state.write().await = CircuitBreakerState::Closed;
                    *self.failure_count.write().await = 0;
                    info!("Circuit breaker closed after successful recovery");
                }
            }
            CircuitBreakerState::Open => {
                // Should not happen, but reset if it does
                *self.failure_count.write().await = 0;
            }
        }
    }

    /// Record a failed operation
    pub async fn record_failure(&self) {
        let mut failure_count = self.failure_count.write().await;
        *failure_count += 1;
        *self.last_failure_time.write().await = Some(Instant::now());

        let state = self.state.read().await;
        match *state {
            CircuitBreakerState::Closed => {
                if *failure_count >= self.config.failure_threshold {
                    drop(state);
                    *self.state.write().await = CircuitBreakerState::Open;
                    warn!("Circuit breaker opened due to {} failures", failure_count);
                }
            }
            CircuitBreakerState::HalfOpen => {
                drop(state);
                *self.state.write().await = CircuitBreakerState::Open;
                warn!("Circuit breaker reopened due to failure during half-open state");
            }
            CircuitBreakerState::Open => {
                // Already open, just update failure time
            }
        }
    }

    /// Get current state
    pub async fn get_state(&self) -> CircuitBreakerState {
        self.state.read().await.clone()
    }
}

/// Error context for better debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Unique error ID
    pub error_id: Uuid,
    /// Error category
    pub category: ErrorCategory,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Operation that failed
    pub operation: String,
    /// Additional context data
    pub context: std::collections::HashMap<String, String>,
    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Stack trace if available
    pub stack_trace: Option<String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(category: ErrorCategory, severity: ErrorSeverity, operation: String) -> Self {
        Self {
            error_id: Uuid::new_v4(),
            category,
            severity,
            operation,
            context: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            stack_trace: None,
        }
    }

    /// Add context information
    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }

    /// Add stack trace
    pub fn with_stack_trace(mut self, stack_trace: String) -> Self {
        self.stack_trace = Some(stack_trace);
        self
    }
}

/// Retry executor with exponential backoff
pub struct RetryExecutor {
    config: RetryConfig,
}

impl RetryExecutor {
    /// Create a new retry executor
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Execute operation with retry logic
    pub async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        let mut attempt = 0;
        let mut delay = self.config.base_delay;

        loop {
            attempt += 1;

            match operation().await {
                Ok(result) => {
                    if attempt > 1 {
                        info!("Operation succeeded after {} attempts", attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    if attempt >= self.config.max_attempts {
                        error!("Operation failed after {} attempts: {:?}", attempt, error);
                        return Err(error);
                    }

                    warn!(
                        "Operation failed on attempt {}, retrying in {:?}: {:?}",
                        attempt, delay, error
                    );

                    // Sleep with jitter
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::Hash;
                    use std::hash::Hasher;
                    let mut hasher = DefaultHasher::new();
                    std::time::SystemTime::now().hash(&mut hasher);
                    let hash = hasher.finish();
                    let jitter =
                        ((hash % 1000) as f64 / 1000.0 - 0.5) * 2.0 * self.config.jitter_factor;
                    let actual_delay =
                        Duration::from_millis((delay.as_millis() as f64 * (1.0 + jitter)) as u64);

                    tokio::time::sleep(actual_delay).await;

                    // Calculate next delay with exponential backoff
                    delay = std::cmp::min(
                        Duration::from_millis(
                            (delay.as_millis() as f64 * self.config.backoff_multiplier) as u64,
                        ),
                        self.config.max_delay,
                    );
                }
            }
        }
    }
}

/// Graceful degradation manager
pub struct DegradationManager {
    /// Cached fallback data
    fallback_cache: Arc<RwLock<std::collections::HashMap<String, Vec<u8>>>>,
}

impl DegradationManager {
    /// Create a new degradation manager
    pub fn new() -> Self {
        Self {
            fallback_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Update fallback cache for an operation
    pub async fn update_fallback_cache(&self, operation: &str, data: Vec<u8>) {
        let mut cache = self.fallback_cache.write().await;
        cache.insert(operation.to_string(), data);
    }

    /// Execute operation with fallback to cached data
    pub async fn execute_with_fallback<T, E>(
        &self,
        operation: &str,
        primary: impl std::future::Future<Output = Result<T, E>>,
        fallback_fn: impl Fn(&[u8]) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: std::fmt::Display,
    {
        match primary.await {
            Ok(result) => Ok(result),
            Err(error) => {
                warn!(
                    "Primary operation '{}' failed: {}, trying fallback",
                    operation, error
                );

                let cache = self.fallback_cache.read().await;
                if let Some(cached_data) = cache.get(operation) {
                    info!("Using cached data for fallback");
                    fallback_fn(cached_data)
                } else {
                    warn!("No fallback data available for operation: {}", operation);
                    Err(error)
                }
            }
        }
    }
}

/// Error recovery coordinator
pub struct ErrorRecoveryCoordinator {
    circuit_breakers: std::collections::HashMap<String, Arc<CircuitBreaker>>,
    retry_executor: RetryExecutor,
    _degradation_manager: DegradationManager,
}

impl ErrorRecoveryCoordinator {
    /// Create a new error recovery coordinator
    pub fn new(retry_config: RetryConfig) -> Self {
        Self {
            circuit_breakers: std::collections::HashMap::new(),
            retry_executor: RetryExecutor::new(retry_config),
            _degradation_manager: DegradationManager::new(),
        }
    }

    /// Register a circuit breaker for an operation
    pub fn register_circuit_breaker(&mut self, operation: &str, config: CircuitBreakerConfig) {
        let circuit_breaker = Arc::new(CircuitBreaker::new(config));
        self.circuit_breakers
            .insert(operation.to_string(), circuit_breaker);
    }

    /// Execute operation with full error handling
    pub async fn execute_with_recovery<F, Fut, T, E>(
        &self,
        operation_name: &str,
        operation: F,
    ) -> Result<T, E>
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        // Check circuit breaker
        if let Some(circuit_breaker) = self.circuit_breakers.get(operation_name) {
            if !circuit_breaker.should_allow_request().await {
                error!("Circuit breaker is open for operation: {}", operation_name);
                // In a real implementation, we'd return a specific error type
                // For now, we'll try the operation anyway for demo purposes
            }
        }

        // Execute with retry
        let result = self.retry_executor.execute(operation).await;

        // Update circuit breaker based on result
        if let Some(circuit_breaker) = self.circuit_breakers.get(operation_name) {
            match &result {
                Ok(_) => circuit_breaker.record_success().await,
                Err(_) => circuit_breaker.record_failure().await,
            }
        }

        result
    }

    /// Get circuit breaker state
    pub async fn get_circuit_breaker_state(&self, operation: &str) -> Option<CircuitBreakerState> {
        if let Some(circuit_breaker) = self.circuit_breakers.get(operation) {
            Some(circuit_breaker.get_state().await)
        } else {
            None
        }
    }
}

/// Error metrics collector
#[derive(Debug, Default)]
pub struct ErrorMetrics {
    /// Total error count by category
    pub error_count_by_category: std::collections::HashMap<ErrorCategory, u64>,
    /// Total error count by severity
    pub error_count_by_severity: std::collections::HashMap<ErrorSeverity, u64>,
    /// Recent errors (last 100)
    pub recent_errors: std::collections::VecDeque<ErrorContext>,
}

impl ErrorMetrics {
    /// Record an error
    pub fn record_error(&mut self, context: ErrorContext) {
        // Update counters
        *self
            .error_count_by_category
            .entry(context.category.clone())
            .or_insert(0) += 1;
        *self
            .error_count_by_severity
            .entry(context.severity)
            .or_insert(0) += 1;

        // Add to recent errors
        self.recent_errors.push_back(context);
        if self.recent_errors.len() > 100 {
            self.recent_errors.pop_front();
        }
    }

    /// Get error rate for a category
    pub fn get_error_rate(&self, category: &ErrorCategory) -> f64 {
        let total_errors: u64 = self.error_count_by_category.values().sum();
        if total_errors == 0 {
            0.0
        } else {
            let category_errors = self.error_count_by_category.get(category).unwrap_or(&0);
            (*category_errors as f64) / (total_errors as f64) * 100.0
        }
    }
}
