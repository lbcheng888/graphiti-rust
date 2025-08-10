//! Performance optimization engine with intelligent caching and query optimization

use crate::error::Error;
use crate::error::Result;
use crate::graph::Edge;
use crate::graph::Node;
use crate::storage::GraphStorage;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::RwLock;
use uuid::Uuid;

#[cfg(test)]
mod tests;

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable intelligent caching
    pub enable_caching: bool,
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable query optimization
    pub enable_query_optimization: bool,
    /// Enable memory pool
    pub enable_memory_pool: bool,
    /// Memory pool size in MB
    pub memory_pool_size_mb: usize,
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Maximum parallel threads
    pub max_parallel_threads: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable zero-copy operations
    pub enable_zero_copy: bool,
    /// Performance monitoring interval in seconds
    pub monitoring_interval_seconds: u64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size_mb: 512,
            cache_ttl_seconds: 3600, // 1 hour
            enable_query_optimization: true,
            enable_memory_pool: true,
            memory_pool_size_mb: 256,
            enable_parallel_processing: true,
            max_parallel_threads: num_cpus::get(),
            enable_simd: true,
            enable_zero_copy: true,
            monitoring_interval_seconds: 60,
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// Cached data
    pub data: T,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last access timestamp
    pub last_accessed: Instant,
    /// Access count
    pub access_count: u64,
    /// Data size in bytes
    pub size_bytes: usize,
    /// Cache hit score
    pub hit_score: f64,
}

impl<T> CacheEntry<T> {
    /// Create a new cached item
    pub fn new(data: T, size_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size_bytes,
            hit_score: 1.0,
        }
    }

    /// Mark this item as accessed
    pub fn access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        self.update_hit_score();
    }

    fn update_hit_score(&mut self) {
        let age_factor = 1.0 / (1.0 + self.created_at.elapsed().as_secs() as f64 / 3600.0);
        let frequency_factor = (self.access_count as f64).ln();
        let recency_factor = 1.0 / (1.0 + self.last_accessed.elapsed().as_secs() as f64 / 60.0);

        self.hit_score = age_factor * frequency_factor * recency_factor;
    }

    /// Check if this item has expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

/// Intelligent cache with LRU and scoring-based eviction
pub struct IntelligentCache<K, V> {
    entries: HashMap<K, CacheEntry<V>>,
    access_order: VecDeque<K>,
    max_size_bytes: usize,
    current_size_bytes: usize,
    ttl: Duration,
    hit_count: u64,
    miss_count: u64,
}

impl<K, V> IntelligentCache<K, V>
where
    K: Clone + Eq + std::hash::Hash,
    V: Clone,
{
    /// Create a new LRU cache
    pub fn new(max_size_bytes: usize, ttl: Duration) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            max_size_bytes,
            current_size_bytes: 0,
            ttl,
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Get a value from the cache
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.get_mut(key) {
            if entry.is_expired(self.ttl) {
                self.remove(key);
                self.miss_count += 1;
                return None;
            }

            entry.access();
            self.hit_count += 1;

            // Move to end of access order
            if let Some(pos) = self.access_order.iter().position(|x| x == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push_back(key.clone());

            Some(entry.data.clone())
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// Insert a value into the cache
    pub fn insert(&mut self, key: K, value: V, size_bytes: usize) {
        // Remove existing entry if present
        if self.entries.contains_key(&key) {
            self.remove(&key);
        }

        // Ensure we have space
        while self.current_size_bytes + size_bytes > self.max_size_bytes && !self.entries.is_empty()
        {
            self.evict_one();
        }

        // Insert new entry
        let entry = CacheEntry::new(value, size_bytes);
        self.current_size_bytes += size_bytes;
        self.entries.insert(key.clone(), entry);
        self.access_order.push_back(key);
    }

    fn remove(&mut self, key: &K) -> Option<CacheEntry<V>> {
        if let Some(entry) = self.entries.remove(key) {
            self.current_size_bytes -= entry.size_bytes;

            if let Some(pos) = self.access_order.iter().position(|x| x == key) {
                self.access_order.remove(pos);
            }

            Some(entry)
        } else {
            None
        }
    }

    fn evict_one(&mut self) {
        // Find entry with lowest hit score
        let mut lowest_score = f64::INFINITY;
        let mut evict_key = None;

        for (key, entry) in &self.entries {
            if entry.hit_score < lowest_score {
                lowest_score = entry.hit_score;
                evict_key = Some(key.clone());
            }
        }

        if let Some(key) = evict_key {
            self.remove(&key);
        }
    }

    /// Get the cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }

    /// Get the total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.current_size_bytes
    }

    /// Get the number of items in the cache
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Clear all items from the cache
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.current_size_bytes = 0;
    }
}

/// Query optimization statistics
#[derive(Debug, Clone)]
pub struct QueryStats {
    /// Query pattern hash
    pub pattern_hash: u64,
    /// Execution count
    pub execution_count: u64,
    /// Average execution time in microseconds
    pub avg_execution_time_us: u64,
    /// Minimum execution time
    pub min_execution_time_us: u64,
    /// Maximum execution time
    pub max_execution_time_us: u64,
    /// Last execution time
    pub last_execution: Instant,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<String>,
}

/// Memory pool for efficient allocation
pub struct MemoryPool {
    pools: HashMap<usize, Vec<Vec<u8>>>,
    max_pool_size: usize,
    total_allocated: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: HashMap::new(),
            max_pool_size,
            total_allocated: 0,
        }
    }

    /// Allocate a buffer from the pool
    pub fn allocate(&mut self, size: usize) -> Vec<u8> {
        // Round up to next power of 2 for better pooling
        let pool_size = size.next_power_of_two();

        if let Some(pool) = self.pools.get_mut(&pool_size) {
            if let Some(mut buffer) = pool.pop() {
                buffer.resize(size, 0);
                return buffer;
            }
        }

        // Allocate new buffer
        self.total_allocated += pool_size;
        vec![0; size]
    }

    /// Return a buffer to the pool
    pub fn deallocate(&mut self, mut buffer: Vec<u8>) {
        let pool_size = buffer.capacity();

        // Only pool if we haven't exceeded max size
        if self.total_allocated <= self.max_pool_size {
            buffer.clear();
            self.pools
                .entry(pool_size)
                .or_insert_with(Vec::new)
                .push(buffer);
        } else {
            self.total_allocated -= pool_size;
        }
    }

    /// Get the total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }
}

/// Performance optimization engine
pub struct PerformanceOptimizer<S> {
    storage: Arc<S>,
    config: PerformanceConfig,
    /// Edge cache
    edge_cache: Arc<RwLock<IntelligentCache<Uuid, Edge>>>,
    /// Query result cache
    query_cache: Arc<RwLock<IntelligentCache<String, Vec<Uuid>>>>,
    /// Query statistics
    query_stats: Arc<RwLock<HashMap<u64, QueryStats>>>,
    /// Memory pool
    memory_pool: Arc<RwLock<MemoryPool>>,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Cache hit rates
    pub edge_cache_hit_rate: f64,
    /// Query cache hit rate
    pub query_cache_hit_rate: f64,
    /// Memory usage
    pub memory_usage_bytes: usize,
    /// Query performance
    pub avg_query_time_us: u64,
    /// Throughput (operations per second)
    pub throughput_ops_per_sec: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory pool efficiency
    pub memory_pool_efficiency: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            edge_cache_hit_rate: 0.0,
            query_cache_hit_rate: 0.0,
            memory_usage_bytes: 0,
            avg_query_time_us: 0,
            throughput_ops_per_sec: 0.0,
            cpu_utilization: 0.0,
            memory_pool_efficiency: 0.0,
        }
    }
}

impl<S> PerformanceOptimizer<S>
where
    S: GraphStorage<Error = Error> + Send + Sync + 'static,
{
    /// Create a new performance optimizer
    pub fn new(storage: Arc<S>, config: PerformanceConfig) -> Self {
        let cache_size_bytes = config.max_cache_size_mb * 1024 * 1024;
        let cache_ttl = Duration::from_secs(config.cache_ttl_seconds);
        let memory_pool_size = config.memory_pool_size_mb * 1024 * 1024;

        Self {
            storage,
            config,
            edge_cache: Arc::new(RwLock::new(IntelligentCache::new(
                cache_size_bytes / 2,
                cache_ttl,
            ))),
            query_cache: Arc::new(RwLock::new(IntelligentCache::new(
                cache_size_bytes / 2,
                cache_ttl,
            ))),
            query_stats: Arc::new(RwLock::new(HashMap::new())),
            memory_pool: Arc::new(RwLock::new(MemoryPool::new(memory_pool_size))),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
        }
    }

    /// Get a node (caching disabled for nodes due to trait object limitations)
    pub async fn get_node_cached(&self, id: &Uuid) -> Result<Option<Box<dyn Node>>> {
        // Direct storage access since we can't cache trait objects
        self.storage.get_node(id).await
    }

    /// Get an edge with caching (Note: Storage trait doesn't support get_edge)
    pub async fn get_edge_cached(&self, _id: &Uuid) -> Result<Option<Edge>> {
        // Storage trait doesn't have get_edge method
        // This would need to be implemented when the storage trait is extended
        Ok(None)
    }

    /// Execute optimized query
    pub async fn execute_optimized_query(&self, query: &str) -> Result<Vec<Uuid>> {
        let start_time = Instant::now();
        let query_hash = self.hash_query(query);

        // Try query cache first
        if self.config.enable_caching {
            let mut cache = self.query_cache.write().await;
            if let Some(result) = cache.get(&query.to_string()) {
                self.update_query_stats(query_hash, start_time.elapsed())
                    .await;
                return Ok(result);
            }
        }

        // Execute query with optimization
        let result = if self.config.enable_query_optimization {
            self.execute_optimized_query_internal(query).await?
        } else {
            // Fallback to basic execution
            self.execute_basic_query(query).await?
        };

        // Cache result
        if self.config.enable_caching {
            let size = result.len() * std::mem::size_of::<Uuid>();
            let mut cache = self.query_cache.write().await;
            cache.insert(query.to_string(), result.clone(), size);
        }

        self.update_query_stats(query_hash, start_time.elapsed())
            .await;
        Ok(result)
    }

    /// Hash query for statistics tracking
    fn hash_query(&self, query: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        hasher.finish()
    }

    /// Update query statistics
    async fn update_query_stats(&self, query_hash: u64, execution_time: Duration) {
        let mut stats = self.query_stats.write().await;
        let execution_time_us = execution_time.as_micros() as u64;

        match stats.get_mut(&query_hash) {
            Some(stat) => {
                stat.execution_count += 1;
                stat.avg_execution_time_us =
                    (stat.avg_execution_time_us * (stat.execution_count - 1) + execution_time_us)
                        / stat.execution_count;
                stat.min_execution_time_us = stat.min_execution_time_us.min(execution_time_us);
                stat.max_execution_time_us = stat.max_execution_time_us.max(execution_time_us);
                stat.last_execution = Instant::now();
            }
            None => {
                stats.insert(
                    query_hash,
                    QueryStats {
                        pattern_hash: query_hash,
                        execution_count: 1,
                        avg_execution_time_us: execution_time_us,
                        min_execution_time_us: execution_time_us,
                        max_execution_time_us: execution_time_us,
                        last_execution: Instant::now(),
                        optimization_suggestions: Vec::new(),
                    },
                );
            }
        }
    }

    /// Execute basic query (placeholder)
    async fn execute_basic_query(&self, _query: &str) -> Result<Vec<Uuid>> {
        // This would implement basic query execution
        // For now, return empty result
        Ok(Vec::new())
    }

    /// Execute optimized query (placeholder)
    async fn execute_optimized_query_internal(&self, _query: &str) -> Result<Vec<Uuid>> {
        // This would implement optimized query execution with:
        // - Query plan optimization
        // - Index usage optimization
        // - Parallel execution
        // - SIMD operations where applicable
        Ok(Vec::new())
    }

    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        let mut metrics = self.metrics.write().await;

        // Update cache hit rates
        {
            let edge_cache = self.edge_cache.read().await;
            metrics.edge_cache_hit_rate = edge_cache.hit_rate();
        }

        {
            let query_cache = self.query_cache.read().await;
            metrics.query_cache_hit_rate = query_cache.hit_rate();
        }

        // Update memory usage
        {
            let memory_pool = self.memory_pool.read().await;
            metrics.memory_usage_bytes = memory_pool.total_allocated();
        }

        metrics.clone()
    }

    /// Clear all caches
    pub async fn clear_caches(&self) {
        if self.config.enable_caching {
            {
                let mut cache = self.edge_cache.write().await;
                cache.clear();
            }

            {
                let mut cache = self.query_cache.write().await;
                cache.clear();
            }
        }
    }

    /// Optimize memory usage
    pub async fn optimize_memory(&self) -> Result<()> {
        // Force garbage collection of expired cache entries
        if self.config.enable_caching {
            // This would implement memory optimization strategies
            // For now, just clear caches if they're too large
            let metrics = self.get_performance_metrics().await;
            let max_memory = self.config.max_cache_size_mb * 1024 * 1024;

            if metrics.memory_usage_bytes > max_memory {
                self.clear_caches().await;
            }
        }

        Ok(())
    }

    /// Get query optimization suggestions
    pub async fn get_optimization_suggestions(&self) -> Vec<String> {
        let stats = self.query_stats.read().await;
        let mut suggestions = Vec::new();

        for stat in stats.values() {
            if stat.avg_execution_time_us > 1000000 {
                // > 1 second
                suggestions.push(format!(
                    "Query pattern {} is slow (avg: {}ms). Consider adding indexes or optimizing query structure.",
                    stat.pattern_hash,
                    stat.avg_execution_time_us / 1000
                ));
            }

            if stat.execution_count > 1000 {
                suggestions.push(format!(
                    "Query pattern {} is executed frequently ({}x). Consider caching or precomputing results.",
                    stat.pattern_hash,
                    stat.execution_count
                ));
            }
        }

        suggestions
    }
}
