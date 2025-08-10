//! AI Enhancement and Adaptive Optimization System
//!
//! This module implements intelligent system optimization, adaptive parameter tuning,
//! predictive maintenance, and self-improving algorithms.

use crate::episode_processor::ProcessingStats;
use crate::error::Error;
use crate::error::Result;
use crate::storage::GraphStorage;
use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::VecDeque;
use tracing::info;
use tracing::instrument;
use uuid::Uuid;

/// AI enhancement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIEnhancementConfig {
    /// Enable adaptive parameter tuning
    pub adaptive_tuning: bool,
    /// Enable predictive maintenance
    pub predictive_maintenance: bool,
    /// Enable intelligent caching
    pub intelligent_caching: bool,
    /// Performance monitoring window (in minutes)
    pub monitoring_window_minutes: i64,
    /// Minimum samples for optimization
    pub min_samples_for_optimization: usize,
    /// Learning rate for adaptive algorithms
    pub learning_rate: f64,
    /// Performance target (processing time in ms)
    pub performance_target_ms: u64,
}

impl Default for AIEnhancementConfig {
    fn default() -> Self {
        Self {
            adaptive_tuning: true,
            predictive_maintenance: true,
            intelligent_caching: true,
            monitoring_window_minutes: 60,
            min_samples_for_optimization: 10,
            learning_rate: 0.1,
            performance_target_ms: 1000,
        }
    }
}

/// Performance metrics for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Entities processed
    pub entities_processed: usize,
    /// Edges processed
    pub edges_processed: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Adaptive parameter set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParameters {
    /// Similarity threshold for entity deduplication
    pub similarity_threshold: f64,
    /// Batch size for processing
    pub batch_size: usize,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Confidence threshold for extractions
    pub confidence_threshold: f64,
    /// Maximum iterations for algorithms
    pub max_iterations: usize,
    /// Timeout for LLM calls (in seconds)
    pub llm_timeout_seconds: u64,
}

impl Default for AdaptiveParameters {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.8,
            batch_size: 10,
            cache_size_limit: 1000,
            confidence_threshold: 0.7,
            max_iterations: 100,
            llm_timeout_seconds: 30,
        }
    }
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    pub id: Uuid,
    /// Parameter to optimize
    pub parameter: String,
    /// Current value
    pub current_value: f64,
    /// Recommended value
    pub recommended_value: f64,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Confidence in recommendation
    pub confidence: f64,
    /// Reasoning
    pub reasoning: String,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Predictive maintenance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceAlert {
    /// Alert ID
    pub id: Uuid,
    /// Alert type
    pub alert_type: AlertType,
    /// Severity level
    pub severity: AlertSeverity,
    /// Description
    pub description: String,
    /// Predicted time to failure
    pub predicted_failure_time: Option<DateTime<Utc>>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Alert types for predictive maintenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    /// Performance degradation
    PerformanceDegradation,
    /// Memory leak detected
    MemoryLeak,
    /// High error rate
    HighErrorRate,
    /// Cache inefficiency
    CacheInefficiency,
    /// Resource exhaustion
    ResourceExhaustion,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Intelligent cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// Cached value
    pub value: T,
    /// Access count
    pub access_count: usize,
    /// Last accessed time
    pub last_accessed: DateTime<Utc>,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Predicted future access probability
    pub access_probability: f64,
}

/// AI Enhancement Engine
pub struct AIEnhancementEngine<S>
where
    S: GraphStorage,
{
    _storage: S,
    config: AIEnhancementConfig,
    parameters: AdaptiveParameters,
    metrics_history: VecDeque<PerformanceMetrics>,
    intelligent_cache: HashMap<String, CacheEntry<String>>,
    optimization_history: Vec<OptimizationRecommendation>,
    maintenance_alerts: Vec<MaintenanceAlert>,
}

impl<S> AIEnhancementEngine<S>
where
    S: GraphStorage<Error = Error>,
{
    /// Create a new AI enhancement engine
    pub fn new(storage: S, config: AIEnhancementConfig) -> Self {
        Self {
            _storage: storage,
            config,
            parameters: AdaptiveParameters::default(),
            metrics_history: VecDeque::new(),
            intelligent_cache: HashMap::new(),
            optimization_history: Vec::new(),
            maintenance_alerts: Vec::new(),
        }
    }

    /// Record performance metrics
    #[instrument(skip(self))]
    pub async fn record_metrics(&mut self, stats: &ProcessingStats) -> Result<()> {
        let metrics = PerformanceMetrics {
            timestamp: Utc::now(),
            processing_time_ms: stats.processing_time_ms,
            memory_usage_bytes: self.get_memory_usage().await?,
            cpu_usage_percent: self.get_cpu_usage().await?,
            entities_processed: stats.entities_extracted,
            edges_processed: stats.edges_extracted,
            cache_hit_rate: self.calculate_cache_hit_rate(),
            error_rate: stats.errors_encountered as f64
                / (stats.entities_extracted + stats.edges_extracted) as f64,
        };

        self.metrics_history.push_back(metrics);

        // Keep only recent metrics
        let cutoff_time =
            Utc::now() - chrono::Duration::minutes(self.config.monitoring_window_minutes);
        while let Some(front) = self.metrics_history.front() {
            if front.timestamp < cutoff_time {
                self.metrics_history.pop_front();
            } else {
                break;
            }
        }

        // Trigger optimization if we have enough samples
        if self.metrics_history.len() >= self.config.min_samples_for_optimization {
            if self.config.adaptive_tuning {
                self.optimize_parameters().await?;
            }

            if self.config.predictive_maintenance {
                self.check_predictive_maintenance().await?;
            }
        }

        Ok(())
    }

    /// Get current adaptive parameters
    pub fn get_parameters(&self) -> &AdaptiveParameters {
        &self.parameters
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(&self) -> &[OptimizationRecommendation] {
        &self.optimization_history
    }

    /// Get maintenance alerts
    pub fn get_maintenance_alerts(&self) -> &[MaintenanceAlert] {
        &self.maintenance_alerts
    }

    /// Intelligent cache get with learning
    pub async fn cache_get(&mut self, key: &str) -> Option<String> {
        if let Some(entry) = self.intelligent_cache.get_mut(key) {
            entry.access_count += 1;
            entry.last_accessed = Utc::now();

            // Calculate access probability separately to avoid borrow checker issues
            let age_hours = (Utc::now() - entry.created_at).num_hours() as f64;
            let recency_hours = (Utc::now() - entry.last_accessed).num_hours() as f64;

            let frequency_score = (entry.access_count as f64).ln() / 10.0;
            let recency_score = (-recency_hours / 24.0).exp(); // Decay over days
            let age_penalty = (-age_hours / (24.0 * 7.0)).exp(); // Decay over weeks

            entry.access_probability =
                (frequency_score * 0.4 + recency_score * 0.4 + age_penalty * 0.2).min(1.0);

            Some(entry.value.clone())
        } else {
            None
        }
    }

    /// Intelligent cache put with eviction strategy
    pub async fn cache_put(&mut self, key: String, value: String) -> Result<()> {
        // Check if cache is full and needs eviction
        if self.intelligent_cache.len() >= self.parameters.cache_size_limit {
            self.evict_cache_entries().await?;
        }

        let entry = CacheEntry {
            value,
            access_count: 1,
            last_accessed: Utc::now(),
            created_at: Utc::now(),
            access_probability: 0.5, // Initial probability
        };

        self.intelligent_cache.insert(key, entry);
        Ok(())
    }

    /// Optimize parameters based on performance metrics
    async fn optimize_parameters(&mut self) -> Result<()> {
        info!("Starting adaptive parameter optimization");

        let recent_metrics: Vec<_> = self.metrics_history.iter().collect();
        let avg_processing_time = recent_metrics
            .iter()
            .map(|m| m.processing_time_ms)
            .sum::<u64>() as f64
            / recent_metrics.len() as f64;

        let avg_error_rate =
            recent_metrics.iter().map(|m| m.error_rate).sum::<f64>() / recent_metrics.len() as f64;

        // Optimize similarity threshold
        if avg_error_rate > 0.05 && self.parameters.similarity_threshold < 0.95 {
            let recommendation = OptimizationRecommendation {
                id: Uuid::new_v4(),
                parameter: "similarity_threshold".to_string(),
                current_value: self.parameters.similarity_threshold,
                recommended_value: (self.parameters.similarity_threshold + 0.05).min(0.95),
                expected_improvement: 0.1,
                confidence: 0.8,
                reasoning: "High error rate detected, increasing similarity threshold".to_string(),
                created_at: Utc::now(),
            };

            self.parameters.similarity_threshold = recommendation.recommended_value;
            self.optimization_history.push(recommendation);
        }

        // Optimize batch size based on processing time
        if avg_processing_time > self.config.performance_target_ms as f64 {
            let new_batch_size = (self.parameters.batch_size as f64 * 0.8) as usize;
            if new_batch_size > 1 {
                let recommendation = OptimizationRecommendation {
                    id: Uuid::new_v4(),
                    parameter: "batch_size".to_string(),
                    current_value: self.parameters.batch_size as f64,
                    recommended_value: new_batch_size as f64,
                    expected_improvement: 0.2,
                    confidence: 0.7,
                    reasoning: "Processing time above target, reducing batch size".to_string(),
                    created_at: Utc::now(),
                };

                self.parameters.batch_size = new_batch_size;
                self.optimization_history.push(recommendation);
            }
        }

        info!("Parameter optimization completed");
        Ok(())
    }

    /// Check for predictive maintenance issues
    async fn check_predictive_maintenance(&mut self) -> Result<()> {
        let recent_metrics: Vec<_> = self.metrics_history.iter().collect();

        // Check for performance degradation trend
        if recent_metrics.len() >= 5 {
            let recent_times: Vec<_> = recent_metrics
                .iter()
                .rev()
                .take(5)
                .map(|m| m.processing_time_ms as f64)
                .collect();

            let trend = self.calculate_trend(&recent_times);
            if trend > 0.1 {
                // 10% increase trend
                let alert = MaintenanceAlert {
                    id: Uuid::new_v4(),
                    alert_type: AlertType::PerformanceDegradation,
                    severity: AlertSeverity::Medium,
                    description: "Performance degradation trend detected".to_string(),
                    predicted_failure_time: Some(Utc::now() + chrono::Duration::hours(24)),
                    recommended_actions: vec![
                        "Review recent changes".to_string(),
                        "Check system resources".to_string(),
                        "Consider parameter optimization".to_string(),
                    ],
                    created_at: Utc::now(),
                };

                self.maintenance_alerts.push(alert);
            }
        }

        // Check for memory leak
        let memory_trend = recent_metrics
            .iter()
            .map(|m| m.memory_usage_bytes as f64)
            .collect::<Vec<_>>();

        if memory_trend.len() >= 3 {
            let trend = self.calculate_trend(&memory_trend);
            if trend > 0.05 {
                // 5% increase trend
                let alert = MaintenanceAlert {
                    id: Uuid::new_v4(),
                    alert_type: AlertType::MemoryLeak,
                    severity: AlertSeverity::High,
                    description: "Potential memory leak detected".to_string(),
                    predicted_failure_time: Some(Utc::now() + chrono::Duration::hours(12)),
                    recommended_actions: vec![
                        "Review memory allocations".to_string(),
                        "Check for resource leaks".to_string(),
                        "Restart service if necessary".to_string(),
                    ],
                    created_at: Utc::now(),
                };

                self.maintenance_alerts.push(alert);
            }
        }

        Ok(())
    }

    /// Calculate trend in a series of values
    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let sum_x = (0..values.len()).sum::<usize>() as f64;
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = values
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let sum_x2 = (0..values.len()).map(|i| (i * i) as f64).sum::<f64>();

        // Linear regression slope
        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    }

    /// Calculate cache hit rate
    fn calculate_cache_hit_rate(&self) -> f64 {
        if self.intelligent_cache.is_empty() {
            return 0.0;
        }

        let total_accesses: usize = self
            .intelligent_cache
            .values()
            .map(|entry| entry.access_count)
            .sum();

        if total_accesses == 0 {
            return 0.0;
        }

        let hits = self.intelligent_cache.len();
        hits as f64 / total_accesses as f64
    }

    /// Calculate access probability for cache entry
    #[allow(dead_code)]
    fn calculate_access_probability(&self, entry: &CacheEntry<String>) -> f64 {
        let age_hours = (Utc::now() - entry.created_at).num_hours() as f64;
        let recency_hours = (Utc::now() - entry.last_accessed).num_hours() as f64;

        let frequency_score = (entry.access_count as f64).ln() / 10.0;
        let recency_score = (-recency_hours / 24.0).exp(); // Decay over days
        let age_penalty = (-age_hours / (24.0 * 7.0)).exp(); // Decay over weeks

        (frequency_score * 0.4 + recency_score * 0.4 + age_penalty * 0.2).min(1.0)
    }

    /// Evict cache entries using intelligent strategy
    async fn evict_cache_entries(&mut self) -> Result<()> {
        let eviction_count = self.intelligent_cache.len() / 4; // Evict 25%

        let mut entries: Vec<_> = self
            .intelligent_cache
            .iter()
            .map(|(key, entry)| (key.clone(), entry.access_probability))
            .collect();

        // Sort by access probability (ascending)
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Remove entries with lowest probability
        for (key, _) in entries.into_iter().take(eviction_count) {
            self.intelligent_cache.remove(&key);
        }

        Ok(())
    }

    /// Get current memory usage from system metrics
    async fn get_memory_usage(&self) -> Result<u64> {
        // TODO: Implement real system metrics collection
        // This could use libraries like sysinfo or procfs to get actual memory usage
        // For now, return 0 to indicate metrics are not available
        Ok(0)
    }

    /// Get current CPU usage from system metrics
    async fn get_cpu_usage(&self) -> Result<f64> {
        // TODO: Implement real system metrics collection
        // This could use libraries like sysinfo to get actual CPU usage
        // For now, return 0.0 to indicate metrics are not available
        Ok(0.0)
    }
}
