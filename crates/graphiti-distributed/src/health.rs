//! Health monitoring system for distributed nodes

use crate::{DistributedConfig, DistributedResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// System health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// CPU usage percentage (0.0 - 100.0)
    pub cpu_usage: f64,
    /// Memory usage percentage (0.0 - 100.0)
    pub memory_usage: f64,
    /// Disk usage percentage (0.0 - 100.0)
    pub disk_usage: f64,
    /// Network latency in milliseconds
    pub network_latency: f64,
    /// Number of active connections
    pub active_connections: u32,
    /// Request rate per second
    pub request_rate: f64,
    /// Error rate percentage (0.0 - 100.0)
    pub error_rate: f64,
    /// Timestamp of metrics collection
    pub timestamp: DateTime<Utc>,
}

impl Default for HealthMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_usage: 0.0,
            network_latency: 0.0,
            active_connections: 0,
            request_rate: 0.0,
            error_rate: 0.0,
            timestamp: Utc::now(),
        }
    }
}

/// Health status of a node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    /// Node is healthy and operating normally
    Healthy,
    /// Node is experiencing minor issues but still functional
    Warning,
    /// Node is experiencing critical issues
    Critical,
    /// Node is unreachable or down
    Down,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Node ID
    pub node_id: Uuid,
    /// Health status
    pub status: HealthStatus,
    /// Health metrics
    pub metrics: HealthMetrics,
    /// Additional details or error messages
    pub details: Option<String>,
    /// Last check timestamp
    pub last_check: DateTime<Utc>,
}

/// Health monitoring configuration
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Health check interval in seconds
    pub check_interval: u64,
    /// CPU usage threshold for warning (percentage)
    pub cpu_warning_threshold: f64,
    /// CPU usage threshold for critical (percentage)
    pub cpu_critical_threshold: f64,
    /// Memory usage threshold for warning (percentage)
    pub memory_warning_threshold: f64,
    /// Memory usage threshold for critical (percentage)
    pub memory_critical_threshold: f64,
    /// Network latency threshold for warning (milliseconds)
    pub latency_warning_threshold: f64,
    /// Network latency threshold for critical (milliseconds)
    pub latency_critical_threshold: f64,
    /// Error rate threshold for warning (percentage)
    pub error_rate_warning_threshold: f64,
    /// Error rate threshold for critical (percentage)
    pub error_rate_critical_threshold: f64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_interval: 30, // 30 seconds
            cpu_warning_threshold: 70.0,
            cpu_critical_threshold: 90.0,
            memory_warning_threshold: 80.0,
            memory_critical_threshold: 95.0,
            latency_warning_threshold: 100.0,    // 100ms
            latency_critical_threshold: 500.0,   // 500ms
            error_rate_warning_threshold: 5.0,   // 5%
            error_rate_critical_threshold: 15.0, // 15%
        }
    }
}

/// Health monitor for tracking node health
pub struct HealthMonitor {
    /// Node configuration
    config: DistributedConfig,
    /// Health monitoring configuration
    health_config: HealthConfig,
    /// Current health status of all nodes
    node_health: Arc<RwLock<HashMap<Uuid, HealthCheck>>>,
    /// Metrics history for trend analysis
    metrics_history: Arc<RwLock<HashMap<Uuid, Vec<HealthMetrics>>>>,
    /// Maximum history size per node
    max_history_size: usize,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(config: DistributedConfig, health_config: HealthConfig) -> Self {
        Self {
            config,
            health_config,
            node_health: Arc::new(RwLock::new(HashMap::new())),
            metrics_history: Arc::new(RwLock::new(HashMap::new())),
            max_history_size: 100, // Keep last 100 metrics per node
        }
    }

    /// Start health monitoring
    pub async fn start(&self) -> DistributedResult<()> {
        info!("Starting health monitoring system");

        // Start periodic health checks
        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.run_periodic_checks().await;
        });

        Ok(())
    }

    /// Run periodic health checks
    async fn run_periodic_checks(&self) {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
            self.health_config.check_interval,
        ));

        loop {
            interval.tick().await;

            if let Err(e) = self.perform_health_check().await {
                error!("Health check failed: {}", e);
            }
        }
    }

    /// Perform a health check on the current node
    pub async fn perform_health_check(&self) -> DistributedResult<()> {
        debug!("Performing health check for node {}", self.config.node_id);

        let metrics = self.collect_metrics().await?;
        let status = self.evaluate_health_status(&metrics);

        let health_check = HealthCheck {
            node_id: self.config.node_id,
            status: status.clone(),
            metrics: metrics.clone(),
            details: None,
            last_check: Utc::now(),
        };

        // Update node health
        {
            let mut health = self.node_health.write().await;
            health.insert(self.config.node_id, health_check);
        }

        // Update metrics history
        {
            let mut history = self.metrics_history.write().await;
            let node_history = history.entry(self.config.node_id).or_insert_with(Vec::new);
            node_history.push(metrics);

            // Limit history size
            if node_history.len() > self.max_history_size {
                node_history.remove(0);
            }
        }

        match status {
            HealthStatus::Healthy => debug!("Node {} is healthy", self.config.node_id),
            HealthStatus::Warning => warn!("Node {} has warnings", self.config.node_id),
            HealthStatus::Critical => error!("Node {} is in critical state", self.config.node_id),
            HealthStatus::Down => error!("Node {} is down", self.config.node_id),
        }

        Ok(())
    }

    /// Collect current system metrics
    async fn collect_metrics(&self) -> DistributedResult<HealthMetrics> {
        // In a real implementation, this would collect actual system metrics
        // For now, we'll simulate metrics collection

        let metrics = HealthMetrics {
            cpu_usage: self.get_cpu_usage().await?,
            memory_usage: self.get_memory_usage().await?,
            disk_usage: self.get_disk_usage().await?,
            network_latency: self.get_network_latency().await?,
            active_connections: self.get_active_connections().await?,
            request_rate: self.get_request_rate().await?,
            error_rate: self.get_error_rate().await?,
            timestamp: Utc::now(),
        };

        Ok(metrics)
    }

    /// Get CPU usage percentage
    async fn get_cpu_usage(&self) -> DistributedResult<f64> {
        // Simulate CPU usage - in real implementation, use system APIs
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now().hash(&mut hasher);
        let hash = hasher.finish();
        Ok((hash % 100) as f64)
    }

    /// Get memory usage percentage
    async fn get_memory_usage(&self) -> DistributedResult<f64> {
        // Simulate memory usage - in real implementation, use system APIs
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (std::time::SystemTime::now(), "memory").hash(&mut hasher);
        let hash = hasher.finish();
        Ok((hash % 100) as f64)
    }

    /// Get disk usage percentage
    async fn get_disk_usage(&self) -> DistributedResult<f64> {
        // Simulate disk usage - in real implementation, use system APIs
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (std::time::SystemTime::now(), "disk").hash(&mut hasher);
        let hash = hasher.finish();
        Ok((hash % 100) as f64)
    }

    /// Get network latency in milliseconds
    async fn get_network_latency(&self) -> DistributedResult<f64> {
        // Simulate network latency - in real implementation, ping other nodes
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (std::time::SystemTime::now(), "latency").hash(&mut hasher);
        let hash = hasher.finish();
        Ok((hash % 200) as f64)
    }

    /// Get number of active connections
    async fn get_active_connections(&self) -> DistributedResult<u32> {
        // Simulate active connections - in real implementation, count actual connections
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (std::time::SystemTime::now(), "connections").hash(&mut hasher);
        let hash = hasher.finish();
        Ok((hash % 100) as u32)
    }

    /// Get request rate per second
    async fn get_request_rate(&self) -> DistributedResult<f64> {
        // Simulate request rate - in real implementation, track actual requests
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (std::time::SystemTime::now(), "requests").hash(&mut hasher);
        let hash = hasher.finish();
        Ok((hash % 1000) as f64)
    }

    /// Get error rate percentage
    async fn get_error_rate(&self) -> DistributedResult<f64> {
        // Simulate error rate - in real implementation, track actual errors
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (std::time::SystemTime::now(), "errors").hash(&mut hasher);
        let hash = hasher.finish();
        Ok((hash % 20) as f64)
    }

    /// Evaluate health status based on metrics
    fn evaluate_health_status(&self, metrics: &HealthMetrics) -> HealthStatus {
        // Check for critical conditions
        if metrics.cpu_usage > self.health_config.cpu_critical_threshold
            || metrics.memory_usage > self.health_config.memory_critical_threshold
            || metrics.network_latency > self.health_config.latency_critical_threshold
            || metrics.error_rate > self.health_config.error_rate_critical_threshold
        {
            return HealthStatus::Critical;
        }

        // Check for warning conditions
        if metrics.cpu_usage > self.health_config.cpu_warning_threshold
            || metrics.memory_usage > self.health_config.memory_warning_threshold
            || metrics.network_latency > self.health_config.latency_warning_threshold
            || metrics.error_rate > self.health_config.error_rate_warning_threshold
        {
            return HealthStatus::Warning;
        }

        HealthStatus::Healthy
    }

    /// Get current health status of a node
    pub async fn get_node_health(&self, node_id: Uuid) -> Option<HealthCheck> {
        let health = self.node_health.read().await;
        health.get(&node_id).cloned()
    }

    /// Get health status of all nodes
    pub async fn get_all_health(&self) -> HashMap<Uuid, HealthCheck> {
        let health = self.node_health.read().await;
        health.clone()
    }

    /// Get metrics history for a node
    pub async fn get_metrics_history(&self, node_id: Uuid) -> Vec<HealthMetrics> {
        let history = self.metrics_history.read().await;
        history.get(&node_id).cloned().unwrap_or_default()
    }

    /// Update health status for a remote node
    pub async fn update_remote_health(&self, health_check: HealthCheck) -> DistributedResult<()> {
        let mut health = self.node_health.write().await;
        health.insert(health_check.node_id, health_check);
        Ok(())
    }

    /// Check if a node is healthy
    pub async fn is_node_healthy(&self, node_id: Uuid) -> bool {
        if let Some(health) = self.get_node_health(node_id).await {
            matches!(health.status, HealthStatus::Healthy | HealthStatus::Warning)
        } else {
            false
        }
    }

    /// Get unhealthy nodes
    pub async fn get_unhealthy_nodes(&self) -> Vec<Uuid> {
        let health = self.node_health.read().await;
        health
            .iter()
            .filter(|(_, check)| {
                matches!(check.status, HealthStatus::Critical | HealthStatus::Down)
            })
            .map(|(&node_id, _)| node_id)
            .collect()
    }

    /// Calculate average metrics across all healthy nodes
    pub async fn get_cluster_average_metrics(&self) -> Option<HealthMetrics> {
        let health = self.node_health.read().await;
        let healthy_metrics: Vec<&HealthMetrics> = health
            .values()
            .filter(|check| matches!(check.status, HealthStatus::Healthy | HealthStatus::Warning))
            .map(|check| &check.metrics)
            .collect();

        if healthy_metrics.is_empty() {
            return None;
        }

        let count = healthy_metrics.len() as f64;
        Some(HealthMetrics {
            cpu_usage: healthy_metrics.iter().map(|m| m.cpu_usage).sum::<f64>() / count,
            memory_usage: healthy_metrics.iter().map(|m| m.memory_usage).sum::<f64>() / count,
            disk_usage: healthy_metrics.iter().map(|m| m.disk_usage).sum::<f64>() / count,
            network_latency: healthy_metrics
                .iter()
                .map(|m| m.network_latency)
                .sum::<f64>()
                / count,
            active_connections: (healthy_metrics
                .iter()
                .map(|m| m.active_connections as f64)
                .sum::<f64>()
                / count) as u32,
            request_rate: healthy_metrics.iter().map(|m| m.request_rate).sum::<f64>() / count,
            error_rate: healthy_metrics.iter().map(|m| m.error_rate).sum::<f64>() / count,
            timestamp: Utc::now(),
        })
    }

    /// Generate health report
    pub async fn generate_health_report(&self) -> HealthReport {
        let all_health = self.get_all_health().await;
        let total_nodes = all_health.len();
        let healthy_nodes = all_health
            .values()
            .filter(|h| matches!(h.status, HealthStatus::Healthy))
            .count();
        let warning_nodes = all_health
            .values()
            .filter(|h| matches!(h.status, HealthStatus::Warning))
            .count();
        let critical_nodes = all_health
            .values()
            .filter(|h| matches!(h.status, HealthStatus::Critical))
            .count();
        let down_nodes = all_health
            .values()
            .filter(|h| matches!(h.status, HealthStatus::Down))
            .count();

        HealthReport {
            timestamp: Utc::now(),
            total_nodes,
            healthy_nodes,
            warning_nodes,
            critical_nodes,
            down_nodes,
            cluster_average: self.get_cluster_average_metrics().await,
            node_details: all_health,
        }
    }
}

/// Health report for the entire cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Report timestamp
    pub timestamp: DateTime<Utc>,
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of healthy nodes
    pub healthy_nodes: usize,
    /// Number of nodes with warnings
    pub warning_nodes: usize,
    /// Number of critical nodes
    pub critical_nodes: usize,
    /// Number of down nodes
    pub down_nodes: usize,
    /// Cluster average metrics
    pub cluster_average: Option<HealthMetrics>,
    /// Detailed health information for each node
    pub node_details: HashMap<Uuid, HealthCheck>,
}

impl Clone for HealthMonitor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            health_config: self.health_config.clone(),
            node_health: self.node_health.clone(),
            metrics_history: self.metrics_history.clone(),
            max_history_size: self.max_history_size,
        }
    }
}
