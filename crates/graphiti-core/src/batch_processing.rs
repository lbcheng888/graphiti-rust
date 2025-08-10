//! Batch processing support for large-scale operations

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
use tokio::sync::RwLock;
use tokio::sync::Semaphore;
use uuid::Uuid;

#[cfg(test)]
mod tests;

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size for operations
    pub max_batch_size: usize,
    /// Maximum number of concurrent batches
    pub max_concurrent_batches: usize,
    /// Timeout for batch operations in seconds
    pub batch_timeout: u64,
    /// Enable progress reporting
    pub enable_progress_reporting: bool,
    /// Chunk size for memory-efficient processing
    pub chunk_size: usize,
    /// Enable automatic retry on failure
    pub enable_retry: bool,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Enable transaction support
    pub enable_transactions: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            max_concurrent_batches: 4,
            batch_timeout: 300, // 5 minutes
            enable_progress_reporting: true,
            chunk_size: 100,
            enable_retry: true,
            max_retries: 3,
            enable_transactions: true,
        }
    }
}

/// Batch operation types (using concrete node types for serialization)
#[derive(Debug)]
pub enum BatchOperation {
    /// Insert multiple nodes
    InsertNodes(Vec<Box<dyn Node>>),
    /// Update multiple nodes
    UpdateNodes(Vec<Box<dyn Node>>),
    /// Delete multiple nodes
    DeleteNodes(Vec<Uuid>),
    /// Insert multiple edges
    InsertEdges(Vec<Edge>),
    /// Update multiple edges
    UpdateEdges(Vec<Edge>),
    /// Delete multiple edges
    DeleteEdges(Vec<Uuid>),
    /// Complex graph transformation
    GraphTransformation {
        /// Nodes to be added to the graph
        nodes_to_add: Vec<Box<dyn Node>>,
        /// Nodes to be updated in the graph
        nodes_to_update: Vec<Box<dyn Node>>,
        /// Node IDs to be deleted from the graph
        nodes_to_delete: Vec<Uuid>,
        /// Edges to be added to the graph
        edges_to_add: Vec<Edge>,
        /// Edges to be updated in the graph
        edges_to_update: Vec<Edge>,
        /// Edge IDs to be deleted from the graph
        edges_to_delete: Vec<Uuid>,
    },
}

/// Batch processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Batch ID
    pub batch_id: Uuid,
    /// Operation type
    pub operation_type: String,
    /// Number of items processed
    pub items_processed: usize,
    /// Number of items failed
    pub items_failed: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Success rate
    pub success_rate: f64,
    /// Error messages for failed items
    pub errors: Vec<String>,
    /// Detailed results per item
    pub item_results: Vec<ItemResult>,
}

/// Result for individual item in batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemResult {
    /// Item index in batch
    pub index: usize,
    /// Whether the operation succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Processing time for this item in microseconds
    pub processing_time_us: u64,
}

/// Progress information for batch operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProgress {
    /// Batch ID
    pub batch_id: Uuid,
    /// Total items to process
    pub total_items: usize,
    /// Items processed so far
    pub processed_items: usize,
    /// Items failed so far
    pub failed_items: usize,
    /// Progress percentage (0.0 - 100.0)
    pub progress_percentage: f64,
    /// Estimated time remaining in seconds
    pub estimated_time_remaining: Option<u64>,
    /// Current processing rate (items per second)
    pub processing_rate: f64,
}

/// Batch processor for handling large-scale operations
pub struct BatchProcessor<S> {
    storage: Arc<S>,
    config: BatchConfig,
    /// Semaphore to limit concurrent batches
    batch_semaphore: Arc<Semaphore>,
    /// Progress tracking
    progress_tracker: Arc<RwLock<HashMap<Uuid, BatchProgress>>>,
    /// Active batch queue
    batch_queue: Arc<RwLock<VecDeque<BatchOperation>>>,
}

impl<S> BatchProcessor<S>
where
    S: GraphStorage<Error = Error> + Send + Sync + 'static,
{
    /// Create a new batch processor
    pub fn new(storage: Arc<S>, config: BatchConfig) -> Self {
        let batch_semaphore = Arc::new(Semaphore::new(config.max_concurrent_batches));

        Self {
            storage,
            config,
            batch_semaphore,
            progress_tracker: Arc::new(RwLock::new(HashMap::new())),
            batch_queue: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Process a batch operation
    pub async fn process_batch(&self, operation: BatchOperation) -> Result<BatchResult> {
        let batch_id = Uuid::new_v4();
        let start_time = std::time::Instant::now();

        // Acquire semaphore permit for concurrency control
        let _permit = self.batch_semaphore.acquire().await.map_err(|_| {
            Error::Processing("Failed to acquire batch processing permit".to_string())
        })?;

        // Initialize progress tracking
        if self.config.enable_progress_reporting {
            self.initialize_progress_tracking(batch_id, &operation)
                .await;
        }

        // Process the batch based on operation type
        let result = match operation {
            BatchOperation::InsertNodes(nodes) => {
                self.process_node_insertions(batch_id, nodes).await
            }
            BatchOperation::UpdateNodes(nodes) => self.process_node_updates(batch_id, nodes).await,
            BatchOperation::DeleteNodes(node_ids) => {
                self.process_node_deletions(batch_id, node_ids).await
            }
            BatchOperation::InsertEdges(edges) => {
                self.process_edge_insertions(batch_id, edges).await
            }
            BatchOperation::UpdateEdges(edges) => self.process_edge_updates(batch_id, edges).await,
            BatchOperation::DeleteEdges(edge_ids) => {
                self.process_edge_deletions(batch_id, edge_ids).await
            }
            BatchOperation::GraphTransformation {
                nodes_to_add,
                nodes_to_update,
                nodes_to_delete,
                edges_to_add,
                edges_to_update,
                edges_to_delete,
            } => {
                self.process_graph_transformation(
                    batch_id,
                    nodes_to_add,
                    nodes_to_update,
                    nodes_to_delete,
                    edges_to_add,
                    edges_to_update,
                    edges_to_delete,
                )
                .await
            }
        };

        // Clean up progress tracking
        if self.config.enable_progress_reporting {
            self.cleanup_progress_tracking(batch_id).await;
        }

        // Calculate final result
        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        let mut final_result = result?;
        final_result.processing_time_ms = processing_time_ms;
        final_result.success_rate = if final_result.items_processed > 0 {
            (final_result.items_processed - final_result.items_failed) as f64
                / final_result.items_processed as f64
                * 100.0
        } else {
            0.0
        };

        Ok(final_result)
    }

    /// Process multiple batches concurrently
    pub async fn process_batches_concurrent(
        &self,
        operations: Vec<BatchOperation>,
    ) -> Result<Vec<BatchResult>> {
        let mut handles = Vec::new();

        for operation in operations {
            let processor = self.clone();
            let handle = tokio::spawn(async move { processor.process_batch(operation).await });
            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result?),
                Err(e) => return Err(Error::Processing(format!("Batch processing failed: {}", e))),
            }
        }

        Ok(results)
    }

    /// Initialize progress tracking for a batch
    async fn initialize_progress_tracking(&self, batch_id: Uuid, operation: &BatchOperation) {
        let total_items = self.count_items_in_operation(operation);
        let progress = BatchProgress {
            batch_id,
            total_items,
            processed_items: 0,
            failed_items: 0,
            progress_percentage: 0.0,
            estimated_time_remaining: None,
            processing_rate: 0.0,
        };

        let mut tracker = self.progress_tracker.write().await;
        tracker.insert(batch_id, progress);
    }

    /// Update progress for a batch
    async fn update_progress(&self, batch_id: Uuid, processed: usize, failed: usize) {
        if !self.config.enable_progress_reporting {
            return;
        }

        let mut tracker = self.progress_tracker.write().await;
        if let Some(progress) = tracker.get_mut(&batch_id) {
            progress.processed_items = processed;
            progress.failed_items = failed;
            progress.progress_percentage = if progress.total_items > 0 {
                (processed as f64 / progress.total_items as f64) * 100.0
            } else {
                100.0
            };
        }
    }

    /// Clean up progress tracking
    async fn cleanup_progress_tracking(&self, batch_id: Uuid) {
        let mut tracker = self.progress_tracker.write().await;
        tracker.remove(&batch_id);
    }

    /// Count items in a batch operation
    fn count_items_in_operation(&self, operation: &BatchOperation) -> usize {
        match operation {
            BatchOperation::InsertNodes(nodes) => nodes.len(),
            BatchOperation::UpdateNodes(nodes) => nodes.len(),
            BatchOperation::DeleteNodes(node_ids) => node_ids.len(),
            BatchOperation::InsertEdges(edges) => edges.len(),
            BatchOperation::UpdateEdges(edges) => edges.len(),
            BatchOperation::DeleteEdges(edge_ids) => edge_ids.len(),
            BatchOperation::GraphTransformation {
                nodes_to_add,
                nodes_to_update,
                nodes_to_delete,
                edges_to_add,
                edges_to_update,
                edges_to_delete,
            } => {
                nodes_to_add.len()
                    + nodes_to_update.len()
                    + nodes_to_delete.len()
                    + edges_to_add.len()
                    + edges_to_update.len()
                    + edges_to_delete.len()
            }
        }
    }

    /// Process node insertions in batches
    async fn process_node_insertions(
        &self,
        batch_id: Uuid,
        nodes: Vec<Box<dyn Node>>,
    ) -> Result<BatchResult> {
        let mut item_results = Vec::new();
        let mut items_failed = 0;
        let chunks = nodes.chunks(self.config.chunk_size);

        for (chunk_index, chunk) in chunks.enumerate() {
            for (item_index, node) in chunk.iter().enumerate() {
                let item_start = std::time::Instant::now();
                let global_index = chunk_index * self.config.chunk_size + item_index;

                let result = if self.config.enable_retry {
                    self.retry_operation(|| async { self.storage.create_node(node.as_ref()).await })
                        .await
                } else {
                    self.storage.create_node(node.as_ref()).await
                };

                let processing_time_us = item_start.elapsed().as_micros() as u64;

                match result {
                    Ok(_) => {
                        item_results.push(ItemResult {
                            index: global_index,
                            success: true,
                            error: None,
                            processing_time_us,
                        });
                    }
                    Err(e) => {
                        items_failed += 1;
                        item_results.push(ItemResult {
                            index: global_index,
                            success: false,
                            error: Some(e.to_string()),
                            processing_time_us,
                        });
                    }
                }

                // Update progress
                self.update_progress(batch_id, global_index + 1, items_failed)
                    .await;
            }
        }

        Ok(BatchResult {
            batch_id,
            operation_type: "InsertNodes".to_string(),
            items_processed: nodes.len(),
            items_failed,
            processing_time_ms: 0, // Will be set by caller
            success_rate: 0.0,     // Will be calculated by caller
            errors: item_results
                .iter()
                .filter_map(|r| r.error.clone())
                .collect(),
            item_results,
        })
    }

    /// Retry operation with exponential backoff
    async fn retry_operation<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempt = 0;
        let mut delay = std::time::Duration::from_millis(100);

        loop {
            attempt += 1;

            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempt >= self.config.max_retries {
                        return Err(e);
                    }

                    tokio::time::sleep(delay).await;
                    delay *= 2; // Exponential backoff
                }
            }
        }
    }

    /// Process node updates in batches
    async fn process_node_updates(
        &self,
        batch_id: Uuid,
        nodes: Vec<Box<dyn Node>>,
    ) -> Result<BatchResult> {
        let mut item_results = Vec::new();
        let mut items_failed = 0;
        let chunks = nodes.chunks(self.config.chunk_size);

        for (chunk_index, chunk) in chunks.enumerate() {
            for (item_index, node) in chunk.iter().enumerate() {
                let item_start = std::time::Instant::now();
                let global_index = chunk_index * self.config.chunk_size + item_index;

                let result = if self.config.enable_retry {
                    self.retry_operation(|| async { self.storage.update_node(node.as_ref()).await })
                        .await
                } else {
                    self.storage.update_node(node.as_ref()).await
                };

                let processing_time_us = item_start.elapsed().as_micros() as u64;

                match result {
                    Ok(_) => {
                        item_results.push(ItemResult {
                            index: global_index,
                            success: true,
                            error: None,
                            processing_time_us,
                        });
                    }
                    Err(e) => {
                        items_failed += 1;
                        item_results.push(ItemResult {
                            index: global_index,
                            success: false,
                            error: Some(e.to_string()),
                            processing_time_us,
                        });
                    }
                }

                self.update_progress(batch_id, global_index + 1, items_failed)
                    .await;
            }
        }

        Ok(BatchResult {
            batch_id,
            operation_type: "UpdateNodes".to_string(),
            items_processed: nodes.len(),
            items_failed,
            processing_time_ms: 0,
            success_rate: 0.0,
            errors: item_results
                .iter()
                .filter_map(|r| r.error.clone())
                .collect(),
            item_results,
        })
    }

    /// Process node deletions in batches
    async fn process_node_deletions(
        &self,
        batch_id: Uuid,
        node_ids: Vec<Uuid>,
    ) -> Result<BatchResult> {
        let mut item_results = Vec::new();
        let mut items_failed = 0;
        let chunks = node_ids.chunks(self.config.chunk_size);

        for (chunk_index, chunk) in chunks.enumerate() {
            for (item_index, node_id) in chunk.iter().enumerate() {
                let item_start = std::time::Instant::now();
                let global_index = chunk_index * self.config.chunk_size + item_index;

                let result = if self.config.enable_retry {
                    self.retry_operation(|| async { self.storage.delete_node(node_id).await })
                        .await
                } else {
                    self.storage.delete_node(node_id).await
                };

                let processing_time_us = item_start.elapsed().as_micros() as u64;

                match result {
                    Ok(_) => {
                        item_results.push(ItemResult {
                            index: global_index,
                            success: true,
                            error: None,
                            processing_time_us,
                        });
                    }
                    Err(e) => {
                        items_failed += 1;
                        item_results.push(ItemResult {
                            index: global_index,
                            success: false,
                            error: Some(e.to_string()),
                            processing_time_us,
                        });
                    }
                }

                self.update_progress(batch_id, global_index + 1, items_failed)
                    .await;
            }
        }

        Ok(BatchResult {
            batch_id,
            operation_type: "DeleteNodes".to_string(),
            items_processed: node_ids.len(),
            items_failed,
            processing_time_ms: 0,
            success_rate: 0.0,
            errors: item_results
                .iter()
                .filter_map(|r| r.error.clone())
                .collect(),
            item_results,
        })
    }

    /// Process edge insertions in batches
    async fn process_edge_insertions(
        &self,
        batch_id: Uuid,
        edges: Vec<Edge>,
    ) -> Result<BatchResult> {
        let mut item_results = Vec::new();
        let mut items_failed = 0;
        let chunks = edges.chunks(self.config.chunk_size);

        for (chunk_index, chunk) in chunks.enumerate() {
            for (item_index, edge) in chunk.iter().enumerate() {
                let item_start = std::time::Instant::now();
                let global_index = chunk_index * self.config.chunk_size + item_index;

                let result = if self.config.enable_retry {
                    self.retry_operation(|| async { self.storage.create_edge(edge).await })
                        .await
                } else {
                    self.storage.create_edge(edge).await
                };

                let processing_time_us = item_start.elapsed().as_micros() as u64;

                match result {
                    Ok(_) => {
                        item_results.push(ItemResult {
                            index: global_index,
                            success: true,
                            error: None,
                            processing_time_us,
                        });
                    }
                    Err(e) => {
                        items_failed += 1;
                        item_results.push(ItemResult {
                            index: global_index,
                            success: false,
                            error: Some(e.to_string()),
                            processing_time_us,
                        });
                    }
                }

                self.update_progress(batch_id, global_index + 1, items_failed)
                    .await;
            }
        }

        Ok(BatchResult {
            batch_id,
            operation_type: "InsertEdges".to_string(),
            items_processed: edges.len(),
            items_failed,
            processing_time_ms: 0,
            success_rate: 0.0,
            errors: item_results
                .iter()
                .filter_map(|r| r.error.clone())
                .collect(),
            item_results,
        })
    }

    /// Process edge updates in batches (Note: Storage trait doesn't support edge updates)
    async fn process_edge_updates(&self, batch_id: Uuid, edges: Vec<Edge>) -> Result<BatchResult> {
        let mut item_results = Vec::new();
        let items_failed = edges.len(); // All fail since update is not supported

        for (index, _edge) in edges.iter().enumerate() {
            item_results.push(ItemResult {
                index,
                success: false,
                error: Some("Edge updates not supported by storage trait".to_string()),
                processing_time_us: 0,
            });
        }

        self.update_progress(batch_id, edges.len(), items_failed)
            .await;

        Ok(BatchResult {
            batch_id,
            operation_type: "UpdateEdges".to_string(),
            items_processed: edges.len(),
            items_failed,
            processing_time_ms: 0,
            success_rate: 0.0,
            errors: vec!["Edge updates not supported by storage trait".to_string()],
            item_results,
        })
    }

    /// Process edge deletions in batches (Note: Storage trait doesn't support edge deletions)
    async fn process_edge_deletions(
        &self,
        batch_id: Uuid,
        edge_ids: Vec<Uuid>,
    ) -> Result<BatchResult> {
        let mut item_results = Vec::new();
        let items_failed = edge_ids.len(); // All fail since deletion is not supported

        for (index, _edge_id) in edge_ids.iter().enumerate() {
            item_results.push(ItemResult {
                index,
                success: false,
                error: Some("Edge deletions not supported by storage trait".to_string()),
                processing_time_us: 0,
            });
        }

        self.update_progress(batch_id, edge_ids.len(), items_failed)
            .await;

        Ok(BatchResult {
            batch_id,
            operation_type: "DeleteEdges".to_string(),
            items_processed: edge_ids.len(),
            items_failed,
            processing_time_ms: 0,
            success_rate: 0.0,
            errors: vec!["Edge deletions not supported by storage trait".to_string()],
            item_results,
        })
    }

    /// Process complex graph transformation
    async fn process_graph_transformation(
        &self,
        batch_id: Uuid,
        nodes_to_add: Vec<Box<dyn Node>>,
        nodes_to_update: Vec<Box<dyn Node>>,
        nodes_to_delete: Vec<Uuid>,
        edges_to_add: Vec<Edge>,
        edges_to_update: Vec<Edge>,
        edges_to_delete: Vec<Uuid>,
    ) -> Result<BatchResult> {
        let mut all_results = Vec::new();
        let mut total_failed = 0;
        let total_items = nodes_to_add.len()
            + nodes_to_update.len()
            + nodes_to_delete.len()
            + edges_to_add.len()
            + edges_to_update.len()
            + edges_to_delete.len();

        // Process in order: deletions first, then updates, then additions

        // 1. Delete edges first
        if !edges_to_delete.is_empty() {
            let result = self
                .process_edge_deletions(batch_id, edges_to_delete)
                .await?;
            total_failed += result.items_failed;
            all_results.extend(result.item_results);
        }

        // 2. Delete nodes
        if !nodes_to_delete.is_empty() {
            let result = self
                .process_node_deletions(batch_id, nodes_to_delete)
                .await?;
            total_failed += result.items_failed;
            all_results.extend(result.item_results);
        }

        // 3. Update nodes
        if !nodes_to_update.is_empty() {
            let result = self.process_node_updates(batch_id, nodes_to_update).await?;
            total_failed += result.items_failed;
            all_results.extend(result.item_results);
        }

        // 4. Update edges
        if !edges_to_update.is_empty() {
            let result = self.process_edge_updates(batch_id, edges_to_update).await?;
            total_failed += result.items_failed;
            all_results.extend(result.item_results);
        }

        // 5. Add nodes
        if !nodes_to_add.is_empty() {
            let result = self.process_node_insertions(batch_id, nodes_to_add).await?;
            total_failed += result.items_failed;
            all_results.extend(result.item_results);
        }

        // 6. Add edges
        if !edges_to_add.is_empty() {
            let result = self.process_edge_insertions(batch_id, edges_to_add).await?;
            total_failed += result.items_failed;
            all_results.extend(result.item_results);
        }

        Ok(BatchResult {
            batch_id,
            operation_type: "GraphTransformation".to_string(),
            items_processed: total_items,
            items_failed: total_failed,
            processing_time_ms: 0,
            success_rate: 0.0,
            errors: all_results.iter().filter_map(|r| r.error.clone()).collect(),
            item_results: all_results,
        })
    }

    /// Get current progress for a batch
    pub async fn get_batch_progress(&self, batch_id: &Uuid) -> Option<BatchProgress> {
        let tracker = self.progress_tracker.read().await;
        tracker.get(batch_id).cloned()
    }

    /// Get progress for all active batches
    pub async fn get_all_batch_progress(&self) -> HashMap<Uuid, BatchProgress> {
        let tracker = self.progress_tracker.read().await;
        tracker.clone()
    }

    /// Cancel a batch operation (if possible)
    pub async fn cancel_batch(&self, batch_id: &Uuid) -> Result<()> {
        // In a real implementation, this would signal the batch to stop
        // For now, just remove from progress tracking
        let mut tracker = self.progress_tracker.write().await;
        tracker.remove(batch_id);
        Ok(())
    }
}

// Clone implementation for BatchProcessor
impl<S> Clone for BatchProcessor<S> {
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            config: self.config.clone(),
            batch_semaphore: Arc::clone(&self.batch_semaphore),
            progress_tracker: Arc::clone(&self.progress_tracker),
            batch_queue: Arc::clone(&self.batch_queue),
        }
    }
}
