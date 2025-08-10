//! Performance monitoring and metrics collection system

use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use tokio::sync::RwLock;
// Tracing imports will be used when logging is added

/// Metric types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Counter metric (monotonically increasing)
    Counter,
    /// Gauge metric (can go up and down)
    Gauge,
    /// Histogram metric (distribution of values)
    Histogram,
    /// Summary metric (quantiles)
    Summary,
}

/// Metric value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// Counter value
    Counter(u64),
    /// Gauge value
    Gauge(f64),
    /// Histogram buckets
    Histogram {
        /// Histogram buckets (upper_bound, count)
        buckets: Vec<(f64, u64)>,
        /// Sum of all observed values
        sum: f64,
        /// Total count of observations
        count: u64,
    },
    /// Summary quantiles
    Summary {
        /// Quantiles (quantile, value)
        quantiles: Vec<(f64, f64)>,
        /// Sum of all observed values
        sum: f64,
        /// Total count of observations
        count: u64,
    },
}

/// Metric labels
pub type MetricLabels = HashMap<String, String>;

/// A single metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Metric value
    pub value: MetricValue,
    /// Metric labels
    pub labels: MetricLabels,
    /// Timestamp
    pub timestamp: u64,
    /// Help text
    pub help: String,
}

/// Performance counter for tracking operations
#[derive(Debug)]
pub struct PerformanceCounter {
    /// Counter name
    name: String,
    /// Current value
    value: Arc<RwLock<u64>>,
    /// Labels
    labels: MetricLabels,
}

impl PerformanceCounter {
    /// Create a new performance counter
    pub fn new(name: String, labels: MetricLabels) -> Self {
        Self {
            name,
            value: Arc::new(RwLock::new(0)),
            labels,
        }
    }

    /// Increment the counter
    pub async fn increment(&self) {
        let mut value = self.value.write().await;
        *value += 1;
    }

    /// Increment the counter by a specific amount
    pub async fn increment_by(&self, amount: u64) {
        let mut value = self.value.write().await;
        *value += amount;
    }

    /// Get current value
    pub async fn get(&self) -> u64 {
        *self.value.read().await
    }

    /// Convert to metric
    pub async fn to_metric(&self, help: String) -> Metric {
        Metric {
            name: self.name.clone(),
            metric_type: MetricType::Counter,
            value: MetricValue::Counter(self.get().await),
            labels: self.labels.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            help,
        }
    }
}

/// Performance gauge for tracking current values
#[derive(Debug)]
pub struct PerformanceGauge {
    /// Gauge name
    name: String,
    /// Current value
    value: Arc<RwLock<f64>>,
    /// Labels
    labels: MetricLabels,
}

impl PerformanceGauge {
    /// Create a new performance gauge
    pub fn new(name: String, labels: MetricLabels) -> Self {
        Self {
            name,
            value: Arc::new(RwLock::new(0.0)),
            labels,
        }
    }

    /// Set the gauge value
    pub async fn set(&self, value: f64) {
        let mut current = self.value.write().await;
        *current = value;
    }

    /// Increment the gauge
    pub async fn increment(&self, amount: f64) {
        let mut current = self.value.write().await;
        *current += amount;
    }

    /// Decrement the gauge
    pub async fn decrement(&self, amount: f64) {
        let mut current = self.value.write().await;
        *current -= amount;
    }

    /// Get current value
    pub async fn get(&self) -> f64 {
        *self.value.read().await
    }

    /// Convert to metric
    pub async fn to_metric(&self, help: String) -> Metric {
        Metric {
            name: self.name.clone(),
            metric_type: MetricType::Gauge,
            value: MetricValue::Gauge(self.get().await),
            labels: self.labels.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            help,
        }
    }
}

/// Performance histogram for tracking distributions
#[derive(Debug)]
pub struct PerformanceHistogram {
    /// Histogram name
    name: String,
    /// Bucket boundaries
    buckets: Vec<f64>,
    /// Bucket counts
    bucket_counts: Arc<RwLock<Vec<u64>>>,
    /// Sum of all observed values
    sum: Arc<RwLock<f64>>,
    /// Total count of observations
    count: Arc<RwLock<u64>>,
    /// Labels
    labels: MetricLabels,
}

impl PerformanceHistogram {
    /// Create a new performance histogram
    pub fn new(name: String, buckets: Vec<f64>, labels: MetricLabels) -> Self {
        let bucket_count = buckets.len() + 1; // +1 for +Inf bucket
        Self {
            name,
            buckets,
            bucket_counts: Arc::new(RwLock::new(vec![0; bucket_count])),
            sum: Arc::new(RwLock::new(0.0)),
            count: Arc::new(RwLock::new(0)),
            labels,
        }
    }

    /// Observe a value
    pub async fn observe(&self, value: f64) {
        // Update sum and count
        {
            let mut sum = self.sum.write().await;
            *sum += value;
        }
        {
            let mut count = self.count.write().await;
            *count += 1;
        }

        // Find appropriate bucket
        let mut bucket_counts = self.bucket_counts.write().await;
        for (i, &bucket_bound) in self.buckets.iter().enumerate() {
            if value <= bucket_bound {
                bucket_counts[i] += 1;
                return;
            }
        }
        // If value is greater than all buckets, put it in the +Inf bucket
        if let Some(last) = bucket_counts.last_mut() {
            *last += 1;
        }
    }

    /// Convert to metric
    pub async fn to_metric(&self, help: String) -> Metric {
        let bucket_counts = self.bucket_counts.read().await;
        let sum = *self.sum.read().await;
        let count = *self.count.read().await;

        let mut buckets = Vec::new();
        for (i, &bound) in self.buckets.iter().enumerate() {
            buckets.push((bound, bucket_counts[i]));
        }
        // Add +Inf bucket
        buckets.push((f64::INFINITY, bucket_counts[bucket_counts.len() - 1]));

        Metric {
            name: self.name.clone(),
            metric_type: MetricType::Histogram,
            value: MetricValue::Histogram {
                buckets,
                sum,
                count,
            },
            labels: self.labels.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            help,
        }
    }
}

/// Timer for measuring operation durations
#[derive(Debug)]
pub struct Timer {
    start_time: Instant,
    histogram: Option<Arc<PerformanceHistogram>>,
}

impl Timer {
    /// Start a new timer
    pub fn start() -> Self {
        Self {
            start_time: Instant::now(),
            histogram: None,
        }
    }

    /// Start a timer with histogram tracking
    pub fn start_with_histogram(histogram: Arc<PerformanceHistogram>) -> Self {
        Self {
            start_time: Instant::now(),
            histogram: Some(histogram),
        }
    }

    /// Stop the timer and return elapsed duration
    pub async fn stop(self) -> Duration {
        let elapsed = self.start_time.elapsed();

        // Record in histogram if available
        if let Some(histogram) = self.histogram {
            histogram.observe(elapsed.as_secs_f64()).await;
        }

        elapsed
    }

    /// Get elapsed time without stopping the timer
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Metrics registry for managing all metrics
#[derive(Debug)]
pub struct MetricsRegistry {
    /// Registered counters
    counters: Arc<RwLock<HashMap<String, Arc<PerformanceCounter>>>>,
    /// Registered gauges
    gauges: Arc<RwLock<HashMap<String, Arc<PerformanceGauge>>>>,
    /// Registered histograms
    histograms: Arc<RwLock<HashMap<String, Arc<PerformanceHistogram>>>>,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a counter
    pub async fn register_counter(
        &self,
        name: String,
        labels: MetricLabels,
    ) -> Arc<PerformanceCounter> {
        let counter = Arc::new(PerformanceCounter::new(name.clone(), labels));
        let mut counters = self.counters.write().await;
        counters.insert(name, counter.clone());
        counter
    }

    /// Register a gauge
    pub async fn register_gauge(
        &self,
        name: String,
        labels: MetricLabels,
    ) -> Arc<PerformanceGauge> {
        let gauge = Arc::new(PerformanceGauge::new(name.clone(), labels));
        let mut gauges = self.gauges.write().await;
        gauges.insert(name, gauge.clone());
        gauge
    }

    /// Register a histogram
    pub async fn register_histogram(
        &self,
        name: String,
        buckets: Vec<f64>,
        labels: MetricLabels,
    ) -> Arc<PerformanceHistogram> {
        let histogram = Arc::new(PerformanceHistogram::new(name.clone(), buckets, labels));
        let mut histograms = self.histograms.write().await;
        histograms.insert(name, histogram.clone());
        histogram
    }

    /// Get all metrics
    pub async fn get_all_metrics(&self) -> Vec<Metric> {
        let mut metrics = Vec::new();

        // Collect counter metrics
        let counters = self.counters.read().await;
        for (name, counter) in counters.iter() {
            let metric = counter.to_metric(format!("Counter metric: {}", name)).await;
            metrics.push(metric);
        }

        // Collect gauge metrics
        let gauges = self.gauges.read().await;
        for (name, gauge) in gauges.iter() {
            let metric = gauge.to_metric(format!("Gauge metric: {}", name)).await;
            metrics.push(metric);
        }

        // Collect histogram metrics
        let histograms = self.histograms.read().await;
        for (name, histogram) in histograms.iter() {
            let metric = histogram
                .to_metric(format!("Histogram metric: {}", name))
                .await;
            metrics.push(metric);
        }

        metrics
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let metrics = self.get_all_metrics().await;
        let mut output = String::new();

        for metric in metrics {
            // Add help text
            output.push_str(&format!("# HELP {} {}\n", metric.name, metric.help));

            // Add type
            let type_str = match metric.metric_type {
                MetricType::Counter => "counter",
                MetricType::Gauge => "gauge",
                MetricType::Histogram => "histogram",
                MetricType::Summary => "summary",
            };
            output.push_str(&format!("# TYPE {} {}\n", metric.name, type_str));

            // Add metric value(s)
            match metric.value {
                MetricValue::Counter(value) => {
                    output.push_str(&format!(
                        "{}{} {}\n",
                        metric.name,
                        format_labels(&metric.labels),
                        value
                    ));
                }
                MetricValue::Gauge(value) => {
                    output.push_str(&format!(
                        "{}{} {}\n",
                        metric.name,
                        format_labels(&metric.labels),
                        value
                    ));
                }
                MetricValue::Histogram {
                    buckets,
                    sum,
                    count,
                } => {
                    for (bound, bucket_count) in buckets {
                        let mut labels = metric.labels.clone();
                        labels.insert(
                            "le".to_string(),
                            if bound.is_infinite() {
                                "+Inf".to_string()
                            } else {
                                bound.to_string()
                            },
                        );
                        output.push_str(&format!(
                            "{}_bucket{} {}\n",
                            metric.name,
                            format_labels(&labels),
                            bucket_count
                        ));
                    }
                    output.push_str(&format!(
                        "{}_sum{} {}\n",
                        metric.name,
                        format_labels(&metric.labels),
                        sum
                    ));
                    output.push_str(&format!(
                        "{}_count{} {}\n",
                        metric.name,
                        format_labels(&metric.labels),
                        count
                    ));
                }
                MetricValue::Summary {
                    quantiles,
                    sum,
                    count,
                } => {
                    for (quantile, value) in quantiles {
                        let mut labels = metric.labels.clone();
                        labels.insert("quantile".to_string(), quantile.to_string());
                        output.push_str(&format!(
                            "{}{} {}\n",
                            metric.name,
                            format_labels(&labels),
                            value
                        ));
                    }
                    output.push_str(&format!(
                        "{}_sum{} {}\n",
                        metric.name,
                        format_labels(&metric.labels),
                        sum
                    ));
                    output.push_str(&format!(
                        "{}_count{} {}\n",
                        metric.name,
                        format_labels(&metric.labels),
                        count
                    ));
                }
            }
            output.push('\n');
        }

        output
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Format labels for Prometheus output
fn format_labels(labels: &MetricLabels) -> String {
    if labels.is_empty() {
        return String::new();
    }

    let mut formatted = String::from("{");
    let mut first = true;
    for (key, value) in labels {
        if !first {
            formatted.push(',');
        }
        formatted.push_str(&format!("{}=\"{}\"", key, value));
        first = false;
    }
    formatted.push('}');
    formatted
}

/// Performance analyzer for detecting performance issues
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Metrics registry
    registry: Arc<MetricsRegistry>,
    /// Performance thresholds
    thresholds: PerformanceThresholds,
    /// Alert history
    alert_history: Arc<RwLock<Vec<PerformanceAlert>>>,
}

/// Performance thresholds for alerting
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable response time (seconds)
    pub max_response_time: f64,
    /// Maximum acceptable error rate (percentage)
    pub max_error_rate: f64,
    /// Maximum acceptable CPU usage (percentage)
    pub max_cpu_usage: f64,
    /// Maximum acceptable memory usage (percentage)
    pub max_memory_usage: f64,
    /// Minimum acceptable throughput (requests per second)
    pub min_throughput: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_response_time: 1.0, // 1 second
            max_error_rate: 5.0,    // 5%
            max_cpu_usage: 80.0,    // 80%
            max_memory_usage: 85.0, // 85%
            min_throughput: 10.0,   // 10 RPS
        }
    }
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    /// Alert ID
    pub id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Metric that triggered the alert
    pub metric_name: String,
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Timestamp when alert was triggered
    pub timestamp: u64,
    /// Whether the alert is resolved
    pub resolved: bool,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlertSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

impl PerformanceAnalyzer {
    /// Create a new performance analyzer
    pub fn new(registry: Arc<MetricsRegistry>, thresholds: PerformanceThresholds) -> Self {
        Self {
            registry,
            thresholds,
            alert_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Analyze current performance and generate alerts
    pub async fn analyze_performance(&self) -> Vec<PerformanceAlert> {
        let mut alerts = Vec::new();
        let metrics = self.registry.get_all_metrics().await;

        for metric in metrics {
            if let Some(alert) = self.check_metric_threshold(&metric).await {
                alerts.push(alert);
            }
        }

        // Store alerts in history
        {
            let mut history = self.alert_history.write().await;
            history.extend(alerts.clone());

            // Keep only last 1000 alerts
            if history.len() > 1000 {
                let len = history.len();
                history.drain(0..len - 1000);
            }
        }

        alerts
    }

    /// Check if a metric exceeds thresholds
    async fn check_metric_threshold(&self, metric: &Metric) -> Option<PerformanceAlert> {
        let current_value = match &metric.value {
            MetricValue::Gauge(value) => *value,
            MetricValue::Counter(value) => *value as f64,
            MetricValue::Histogram { sum, count, .. } => {
                if *count > 0 {
                    sum / (*count as f64)
                } else {
                    0.0
                }
            }
            MetricValue::Summary { sum, count, .. } => {
                if *count > 0 {
                    sum / (*count as f64)
                } else {
                    0.0
                }
            }
        };

        // Check different metric types
        match metric.name.as_str() {
            name if name.contains("response_time") => {
                if current_value > self.thresholds.max_response_time {
                    return Some(PerformanceAlert {
                        id: uuid::Uuid::new_v4().to_string(),
                        severity: if current_value > self.thresholds.max_response_time * 2.0 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::High
                        },
                        message: format!("High response time detected: {:.2}s", current_value),
                        metric_name: metric.name.clone(),
                        current_value,
                        threshold_value: self.thresholds.max_response_time,
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        resolved: false,
                    });
                }
            }
            name if name.contains("error_rate") => {
                if current_value > self.thresholds.max_error_rate {
                    return Some(PerformanceAlert {
                        id: uuid::Uuid::new_v4().to_string(),
                        severity: if current_value > self.thresholds.max_error_rate * 2.0 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::High
                        },
                        message: format!("High error rate detected: {:.2}%", current_value),
                        metric_name: metric.name.clone(),
                        current_value,
                        threshold_value: self.thresholds.max_error_rate,
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        resolved: false,
                    });
                }
            }
            name if name.contains("cpu_usage") => {
                if current_value > self.thresholds.max_cpu_usage {
                    return Some(PerformanceAlert {
                        id: uuid::Uuid::new_v4().to_string(),
                        severity: if current_value > 95.0 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Medium
                        },
                        message: format!("High CPU usage detected: {:.2}%", current_value),
                        metric_name: metric.name.clone(),
                        current_value,
                        threshold_value: self.thresholds.max_cpu_usage,
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        resolved: false,
                    });
                }
            }
            name if name.contains("memory_usage") => {
                if current_value > self.thresholds.max_memory_usage {
                    return Some(PerformanceAlert {
                        id: uuid::Uuid::new_v4().to_string(),
                        severity: if current_value > 95.0 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Medium
                        },
                        message: format!("High memory usage detected: {:.2}%", current_value),
                        metric_name: metric.name.clone(),
                        current_value,
                        threshold_value: self.thresholds.max_memory_usage,
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        resolved: false,
                    });
                }
            }
            _ => {}
        }

        None
    }

    /// Get alert history
    pub async fn get_alert_history(&self) -> Vec<PerformanceAlert> {
        self.alert_history.read().await.clone()
    }

    /// Get active alerts (unresolved)
    pub async fn get_active_alerts(&self) -> Vec<PerformanceAlert> {
        let history = self.alert_history.read().await;
        history
            .iter()
            .filter(|alert| !alert.resolved)
            .cloned()
            .collect()
    }

    /// Resolve an alert
    pub async fn resolve_alert(&self, alert_id: &str) -> bool {
        let mut history = self.alert_history.write().await;
        for alert in history.iter_mut() {
            if alert.id == alert_id {
                alert.resolved = true;
                return true;
            }
        }
        false
    }

    /// Generate performance report
    pub async fn generate_performance_report(&self) -> PerformanceReport {
        let metrics = self.registry.get_all_metrics().await;
        let alerts = self.get_active_alerts().await;

        let mut response_times = Vec::new();
        let mut error_rates = Vec::new();
        let mut throughput_values = Vec::new();

        for metric in &metrics {
            match metric.name.as_str() {
                name if name.contains("response_time") => {
                    if let MetricValue::Histogram { sum, count, .. } = &metric.value {
                        if *count > 0 {
                            response_times.push(sum / (*count as f64));
                        }
                    }
                }
                name if name.contains("error_rate") => {
                    if let MetricValue::Gauge(value) = &metric.value {
                        error_rates.push(*value);
                    }
                }
                name if name.contains("throughput") => {
                    if let MetricValue::Gauge(value) = &metric.value {
                        throughput_values.push(*value);
                    }
                }
                _ => {}
            }
        }

        PerformanceReport {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            total_metrics: metrics.len(),
            active_alerts: alerts.len(),
            average_response_time: if response_times.is_empty() {
                0.0
            } else {
                response_times.iter().sum::<f64>() / response_times.len() as f64
            },
            average_error_rate: if error_rates.is_empty() {
                0.0
            } else {
                error_rates.iter().sum::<f64>() / error_rates.len() as f64
            },
            average_throughput: if throughput_values.is_empty() {
                0.0
            } else {
                throughput_values.iter().sum::<f64>() / throughput_values.len() as f64
            },
            health_score: self.calculate_health_score(&metrics, &alerts).await,
        }
    }

    /// Calculate overall health score (0-100)
    async fn calculate_health_score(
        &self,
        _metrics: &[Metric],
        alerts: &[PerformanceAlert],
    ) -> f64 {
        let mut score: f64 = 100.0;

        // Deduct points for active alerts
        for alert in alerts {
            match alert.severity {
                AlertSeverity::Low => score -= 5.0,
                AlertSeverity::Medium => score -= 10.0,
                AlertSeverity::High => score -= 20.0,
                AlertSeverity::Critical => score -= 30.0,
            }
        }

        // Ensure score doesn't go below 0
        score.max(0.0)
    }
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report timestamp
    pub timestamp: u64,
    /// Total number of metrics
    pub total_metrics: usize,
    /// Number of active alerts
    pub active_alerts: usize,
    /// Average response time
    pub average_response_time: f64,
    /// Average error rate
    pub average_error_rate: f64,
    /// Average throughput
    pub average_throughput: f64,
    /// Overall health score (0-100)
    pub health_score: f64,
}
