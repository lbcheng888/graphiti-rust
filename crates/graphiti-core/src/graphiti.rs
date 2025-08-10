//! Main Graphiti orchestrator

use crate::code_processor::CodeContext;
use crate::code_processor::CodeProcessingResult;
use crate::code_processor::CodeProcessor;
use crate::community::CommunityConfig;
use crate::episode_processor::EmbeddingClient;
use crate::episode_processor::EntityType;
use crate::episode_processor::EpisodeProcessingResult;
use crate::episode_processor::EpisodeProcessor;
use crate::episode_processor::LLMClient;
use crate::error::Error;
use crate::error::Result;
use crate::graph::CommunityNode;
use crate::graph::Edge;
use crate::graph::EpisodeNode;
use crate::graph::EpisodeType;
use crate::graph::TemporalMetadata;

use crate::graph_traversal::TraversalConfig;
use crate::graph_traversal::TraversalResult;
use crate::storage::Direction;
use crate::storage::GraphStorage;
use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;

use tracing::info;
use tracing::instrument;
use tracing::warn;
use uuid::Uuid;

/// Configuration for the Graphiti system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphitiConfig {
    /// Name of the graph
    pub name: String,
    /// Whether to enable deduplication
    pub enable_deduplication: bool,
    /// Minimum confidence threshold for entities
    pub min_entity_confidence: f32,
    /// Minimum confidence threshold for relationships
    pub min_relationship_confidence: f32,
    /// Maximum context window for extraction
    pub max_context_window: usize,
    /// Whether to generate embeddings for all content
    pub generate_embeddings: bool,
    /// Whether to skip entity extraction (embedding-only mode)
    #[serde(default)]
    pub skip_entity_extraction: bool,
}

impl Default for GraphitiConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            enable_deduplication: true,
            min_entity_confidence: 0.7,
            min_relationship_confidence: 0.6,
            max_context_window: 4000,
            generate_embeddings: true,
            skip_entity_extraction: false,
        }
    }
}

/// Main Graphiti orchestrator
pub struct Graphiti<S, L, E>
where
    S: GraphStorage,
    L: LLMClient,
    E: EmbeddingClient,
{
    _config: GraphitiConfig,
    episode_processor: EpisodeProcessor<S, L, E>,
}

/// Trait for text search index implementations
pub trait TextSearchIndex: Send + Sync {
    // Text search methods
}

/// Trait for vector search index implementations
pub trait VectorIndex: Send + Sync {
    // Vector search methods
}

impl<S, L, E> Graphiti<S, L, E>
where
    S: GraphStorage<Error = Error> + 'static,
    L: LLMClient + 'static,
    E: EmbeddingClient + 'static,
{
    /// Create a new Graphiti instance
    pub fn new(config: GraphitiConfig, storage: S, llm: L, embedder: E) -> Result<Self> {
        // Create default entity types
        let entity_types = vec![
            EntityType {
                id: 1,
                name: "Person".to_string(),
                description: "A human being".to_string(),
            },
            EntityType {
                id: 2,
                name: "Organization".to_string(),
                description: "A company, institution, or group".to_string(),
            },
            EntityType {
                id: 3,
                name: "Location".to_string(),
                description: "A place or geographical location".to_string(),
            },
        ];

        let _excluded_types: Vec<String> = vec![];

        // Create episode processor
        let episode_processor = EpisodeProcessor::new(storage, llm, embedder, entity_types)?;

        Ok(Self {
            _config: config,
            episode_processor,
        })
    }

    /// Process a new episode (unit of information)
    #[instrument(skip(self, content, _metadata))]
    pub async fn add_episode(
        &self,
        name: String,
        content: String,
        source: String,
        episode_type: EpisodeType,
        _metadata: HashMap<String, serde_json::Value>,
        timestamp: Option<DateTime<Utc>>,
    ) -> Result<EpisodeProcessingResult> {
        info!("Processing new episode: {}", name);

        let episode_id = Uuid::new_v4();
        let now = Utc::now();
        let event_time = timestamp.unwrap_or(now);

        // Create episode node
        let episode = EpisodeNode {
            id: episode_id,
            name: name.clone(),
            episode_type,
            content: content.clone(),
            source,
            temporal: TemporalMetadata {
                created_at: now,
                valid_from: event_time,
                valid_to: None,
                expired_at: None,
            },
            embedding: None, // Will be set later if enabled
        };

        // Get previous episodes for context (simplified - in practice would query database)
        let previous_episodes = Vec::new(); // TODO: Implement proper episode retrieval

        // Process episode using the episode processor
        let result = self
            .episode_processor
            .process_episode(episode, previous_episodes)
            .await?;

        info!(
            "Successfully processed episode: {} entities, {} edges",
            result.stats.entities_extracted, result.stats.edges_extracted
        );

        Ok(result)
    }

    /// Search the graph using hybrid search
    #[instrument(skip(self))]
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>> {
        info!("Searching graph with query: {} (limit: {})", query, limit);

        // Step 1: Get all nodes from storage for text search
        let nodes = self
            .episode_processor
            .storage()
            .get_all_nodes()
            .await
            .map_err(|e| Error::Storage(format!("Failed to get nodes: {}", e)))?;

        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: Perform text-based search
        let mut results = Vec::new();
        let query_lower = query.to_lowercase();

        for node in &nodes {
            let mut score = 0.0f32;
            let mut matched_fields = Vec::new();

            // Search in node name/content
            let node_text = format!("{:?}", node).to_lowercase(); // Simplified text extraction

            // Simple text matching with scoring
            if node_text.contains(&query_lower) {
                // Exact match gets higher score
                if node_text.contains(&query) {
                    score += 1.0;
                    matched_fields.push("exact_match");
                } else {
                    score += 0.5;
                    matched_fields.push("partial_match");
                }

                // Word boundary matches get bonus
                for word in query.split_whitespace() {
                    if node_text.contains(&word.to_lowercase()) {
                        score += 0.2;
                        matched_fields.push("word_match");
                    }
                }
            }

            // Apply filters if provided
            if let Some(ref filters) = filters {
                // Apply confidence filter
                if let Some(min_conf) = filters.min_confidence {
                    if score < min_conf {
                        continue;
                    }
                }

                // Apply time range filter
                if let Some((start, end)) = filters.time_range {
                    // Check if node has temporal metadata within range
                    // This is simplified - in real implementation would check actual timestamps
                    let now = chrono::Utc::now();
                    if now < start || now > end {
                        score *= 0.5; // Reduce score for out-of-range items
                    }
                }
            }

            if score > 0.0 {
                results.push(SearchResult {
                    node: serde_json::json!({
                        "id": node.id(),
                        "type": "node",
                        "content": format!("{:?}", node)
                    }),
                    score,
                    explanation: Some(format!("Matched fields: {}", matched_fields.join(", "))),
                    path: None,
                });
            }
        }

        // Step 3: Sort by score and limit results
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        // Step 4: Add vector similarity search if embeddings are available
        let mut vector_results = Vec::new();
        // Generate embedding for the query
        match self.episode_processor.embedding_client().embed(query).await {
            Ok(query_embedding) => {
                // Perform vector similarity search
                match self.vector_search(&query_embedding, limit / 2).await {
                    Ok(vec_results) => {
                        info!("Vector search found {} results", vec_results.len());
                        // Convert vector results to SearchResult format
                        for vec_result in vec_results {
                            vector_results.push(SearchResult {
                                node: serde_json::json!({
                                    "id": vec_result.node_id,
                                    "type": "vector_match",
                                    "content": vec_result.content,
                                    "embedding_score": vec_result.similarity
                                }),
                                score: vec_result.similarity,
                                explanation: Some(format!(
                                    "Vector similarity: {:.3}",
                                    vec_result.similarity
                                )),
                                path: None,
                            });
                        }
                    }
                    Err(e) => {
                        warn!("Vector search failed: {}", e);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to generate query embedding: {}", e);
            }
        }

        // Step 5: Add graph traversal search for related entities
        let mut additional_results = Vec::new();
        if !results.is_empty() {
            // Get the top result and find related entities
            if let Some(top_result) = results.first() {
                let top_score = top_result.score;
                if let Some(node_id_value) = top_result.node.get("id") {
                    if let Some(node_id_str) = node_id_value.as_str() {
                        if let Ok(node_id) = Uuid::parse_str(node_id_str) {
                            // Find neighbors of the top result
                            match self.get_neighbors(&node_id, 2, None).await {
                                Ok(traversal_result) => {
                                    // Add neighbor nodes as additional results with lower scores
                                    for neighbor_node in traversal_result.nodes.iter().take(3) {
                                        additional_results.push(SearchResult {
                                            node: serde_json::json!({
                                                "id": neighbor_node.id,
                                                "type": "neighbor",
                                                "content": format!("Related to top result (depth: {})", neighbor_node.depth)
                                            }),
                                            score: top_score * (0.8 - neighbor_node.depth as f32 * 0.1), // Lower score based on depth
                                            explanation: Some(format!("Found through graph traversal at depth {}", neighbor_node.depth)),
                                            path: Some(neighbor_node.path.clone()),
                                        });
                                    }
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to get neighbors for graph traversal search: {}",
                                        e
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add the additional results
        results.extend(additional_results);

        // Merge vector results with text results
        results.extend(vector_results);

        // Step 6: Cross-encoder reranking (if we have results)
        if !results.is_empty() && results.len() > 1 {
            match self.rerank_search_results(query, &mut results).await {
                Ok(()) => {
                    info!("Successfully reranked {} search results", results.len());
                }
                Err(e) => {
                    warn!(
                        "Cross-encoder reranking failed, using original scores: {}",
                        e
                    );
                    // Fall back to original sorting
                    results.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        } else {
            // Sort by original scores if no reranking
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Step 7: Final limit
        results.truncate(limit);

        info!(
            "Found {} search results (including vector search and graph traversal)",
            results.len()
        );
        Ok(results)
    }

    /// Build communities from the graph
    #[instrument(skip(self))]
    pub async fn build_communities(&self) -> Result<Vec<CommunityNode>> {
        info!("Building communities");

        // Create a community detector with default configuration
        let _config = CommunityConfig::default();

        // Note: We need to clone storage for the detector
        // In a real implementation, we'd use Arc<Storage> or similar
        // For now, this is a placeholder that shows the intended structure

        // TODO: Implement actual community detection
        // This would require:
        // 1. Creating a CommunityDetector with shared storage
        // 2. Running detection algorithm
        // 3. Converting results to CommunityNode objects

        info!("Community detection not yet fully implemented");
        Ok(Vec::new())
    }

    /// Get neighbors of a node using graph traversal
    #[instrument(skip(self))]
    pub async fn get_neighbors(
        &self,
        node_id: &Uuid,
        depth: usize,
        relationship_types: Option<Vec<String>>,
    ) -> Result<TraversalResult> {
        info!("Getting neighbors for node: {}", node_id);

        // Create a traversal configuration
        let _config = TraversalConfig {
            max_depth: depth,
            max_nodes: 1000,
            relationship_types: relationship_types.unwrap_or_default(),
            direction: Direction::Both,
            include_start: false,
        };

        // TODO: Create GraphTraversal with shared storage
        // For now, return empty result
        info!("Graph traversal not yet fully implemented");
        Ok(TraversalResult {
            start_node: *node_id,
            nodes: Vec::new(),
            total_visited: 0,
            max_depth_reached: 0,
        })
    }

    /// Find shortest path between two nodes
    #[instrument(skip(self))]
    pub async fn find_path(
        &self,
        start_node: &Uuid,
        target_node: &Uuid,
        max_depth: usize,
    ) -> Result<Option<Vec<Uuid>>> {
        info!("Finding path from {} to {}", start_node, target_node);

        // Create a traversal configuration
        let _config = TraversalConfig {
            max_depth,
            max_nodes: 10000,
            relationship_types: Vec::new(),
            direction: Direction::Both,
            include_start: true,
        };

        // TODO: Create GraphTraversal with shared storage
        // For now, return None
        info!("Path finding not yet fully implemented");
        Ok(None)
    }

    /// Get graph state at a specific time
    #[instrument(skip(self))]
    pub async fn get_graph_at_time(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<(Vec<Box<dyn crate::graph::Node>>, Vec<Edge>)> {
        info!("Getting graph state at time: {}", timestamp);

        let nodes = self
            .episode_processor
            .storage()
            .get_nodes_at_time(timestamp)
            .await
            .map_err(|e| Error::Storage(format!("Failed to get nodes at time: {}", e)))?;

        let edges = self
            .episode_processor
            .storage()
            .get_edges_at_time(timestamp)
            .await
            .map_err(|e| Error::Storage(format!("Failed to get edges at time: {}", e)))?;

        info!(
            "Retrieved {} nodes and {} edges at time {}",
            nodes.len(),
            edges.len(),
            timestamp
        );
        Ok((nodes, edges))
    }

    /// Get the history of a specific node
    #[instrument(skip(self))]
    pub async fn get_node_history(
        &self,
        node_id: &Uuid,
    ) -> Result<Vec<Box<dyn crate::graph::Node>>> {
        info!("Getting history for node: {}", node_id);

        let history = self
            .episode_processor
            .storage()
            .get_node_history(node_id)
            .await
            .map_err(|e| Error::Storage(format!("Failed to get node history: {}", e)))?;

        info!("Retrieved {} versions for node {}", history.len(), node_id);
        Ok(history)
    }

    /// Get the history of a specific edge
    #[instrument(skip(self))]
    pub async fn get_edge_history(&self, edge_id: &Uuid) -> Result<Vec<Edge>> {
        info!("Getting history for edge: {}", edge_id);

        let history = self
            .episode_processor
            .storage()
            .get_edge_history(edge_id)
            .await
            .map_err(|e| Error::Storage(format!("Failed to get edge history: {}", e)))?;

        info!("Retrieved {} versions for edge {}", history.len(), edge_id);
        Ok(history)
    }

    /// Query nodes and edges within a time range
    #[instrument(skip(self))]
    pub async fn query_time_range(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<(Vec<Box<dyn crate::graph::Node>>, Vec<Edge>)> {
        info!("Querying time range: {} to {}", start_time, end_time);

        // For now, get all nodes and filter by time range
        // In a real implementation, this would be optimized with database queries
        let all_nodes = self
            .episode_processor
            .storage()
            .get_all_nodes()
            .await
            .map_err(|e| Error::Storage(format!("Failed to get all nodes: {}", e)))?;

        let all_edges = self
            .episode_processor
            .storage()
            .get_all_edges()
            .await
            .map_err(|e| Error::Storage(format!("Failed to get all edges: {}", e)))?;

        // Filter nodes by time range (simplified implementation)
        let filtered_nodes: Vec<_> = all_nodes
            .into_iter()
            .filter(|_node| {
                // This is a simplified check - in reality we'd check the temporal metadata
                true // For now, include all nodes
            })
            .collect();

        // Filter edges by time range (simplified implementation)
        let filtered_edges: Vec<_> = all_edges
            .into_iter()
            .filter(|edge| {
                // Check if edge's temporal metadata overlaps with the time range
                let edge_start = edge.temporal.valid_from;
                let edge_end = edge.temporal.valid_to.unwrap_or(end_time);

                // Check for overlap: edge_start <= end_time && edge_end >= start_time
                edge_start <= end_time && edge_end >= start_time
            })
            .collect();

        info!(
            "Found {} nodes and {} edges in time range",
            filtered_nodes.len(),
            filtered_edges.len()
        );
        Ok((filtered_nodes, filtered_edges))
    }

    /// Perform vector similarity search
    #[instrument(skip(self, query_embedding))]
    async fn vector_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>> {
        info!(
            "Performing vector search with embedding dimension: {}",
            query_embedding.len()
        );

        // Get all nodes with embeddings
        let nodes = self
            .episode_processor
            .storage()
            .get_all_nodes()
            .await
            .map_err(|e| Error::Storage(format!("Failed to get nodes for vector search: {}", e)))?;

        let mut results = Vec::new();

        for node in nodes {
            // Check if node has embedding (simplified check)
            // In a real implementation, we'd have a proper vector index
            let node_content = format!("{:?}", node); // Simplified content extraction

            // Generate embedding for similarity comparison
            match self
                .episode_processor
                .embedding_client()
                .embed(&node_content)
                .await
            {
                Ok(node_embedding) => {
                    // Calculate cosine similarity
                    let similarity = self.cosine_similarity(query_embedding, &node_embedding);

                    if similarity > 0.5 {
                        // Threshold for relevance
                        results.push(VectorSearchResult {
                            node_id: *node.id(),
                            content: node_content,
                            similarity,
                        });
                    }
                }
                Err(e) => {
                    warn!("Failed to generate embedding for node {}: {}", node.id(), e);
                }
            }
        }

        // Sort by similarity (highest first)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        results.truncate(limit);

        info!("Vector search found {} results", results.len());
        Ok(results)
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Rerank search results using cross-encoder
    #[instrument(skip(self, results))]
    async fn rerank_search_results(
        &self,
        query: &str,
        results: &mut Vec<SearchResult>,
    ) -> Result<()> {
        info!(
            "Reranking {} search results with cross-encoder",
            results.len()
        );

        // Extract document texts from results
        let documents: Vec<String> = results
            .iter()
            .map(|result| {
                // Extract content from the node JSON
                result
                    .node
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("No content")
                    .to_string()
            })
            .collect();

        // Use text-based similarity scoring for reranking
        // In a production system, this would call a cross-encoder model
        let rerank_scores = self.text_similarity_rerank(query, &documents).await?;

        // Update scores with reranked scores
        for (result, &rerank_score) in results.iter_mut().zip(rerank_scores.iter()) {
            // Combine original score with rerank score (weighted average)
            let original_weight = 0.3;
            let rerank_weight = 0.7;
            result.score = original_weight * result.score + rerank_weight * rerank_score;

            // Update explanation
            if let Some(ref mut explanation) = result.explanation {
                explanation.push_str(&format!(" | Reranked: {:.3}", rerank_score));
            } else {
                result.explanation = Some(format!("Reranked: {:.3}", rerank_score));
            }
        }

        // Sort by new combined scores
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        info!("Reranking completed");
        Ok(())
    }

    /// Text-based similarity reranking implementation
    /// Uses lexical similarity metrics for document reranking
    async fn text_similarity_rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        info!(
            "Text similarity reranking {} documents for query: {}",
            documents.len(),
            query
        );

        let mut scores = Vec::new();
        let query_lower = query.to_lowercase();

        for document in documents {
            let doc_lower = document.to_lowercase();

            // Simple relevance scoring based on:
            // 1. Exact phrase matches
            // 2. Word overlap
            // 3. Document length penalty
            let mut score = 0.0f32;

            // Exact phrase match gets high score
            if doc_lower.contains(&query_lower) {
                score += 0.8;
            }

            // Word overlap scoring
            let query_words: Vec<&str> = query_lower.split_whitespace().collect();
            let doc_words: Vec<&str> = doc_lower.split_whitespace().collect();

            let overlap_count = query_words
                .iter()
                .filter(|word| doc_words.contains(word))
                .count();

            if !query_words.is_empty() {
                score += 0.5 * (overlap_count as f32 / query_words.len() as f32);
            }

            // Length penalty (prefer shorter, more focused content)
            let length_penalty = 1.0 / (1.0 + document.len() as f32 / 1000.0);
            score *= length_penalty;

            // Normalize to 0-1 range
            score = score.clamp(0.0, 1.0);
            scores.push(score);
        }

        Ok(scores)
    }

    /// Process code-specific content with specialized handling
    #[instrument(skip(self, content))]
    pub async fn process_code_episode(
        &self,
        content: String,
        context: CodeContext,
    ) -> Result<CodeProcessingResult> {
        info!("Processing code episode with context: {:?}", context);

        // Create a code processor instance
        let code_processor = CodeProcessor::new(
            self.episode_processor.storage().clone(),
            self.episode_processor.llm_client().clone(),
            self.episode_processor.embedding_client().clone(),
        );

        // Process the code episode
        code_processor.process_code_episode(&content, context).await
    }

    /// Search for code-specific entities and patterns
    #[instrument(skip(self))]
    pub async fn search_code_knowledge(
        &self,
        query: &str,
        language: Option<String>,
        framework: Option<String>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        info!(
            "Searching code knowledge: {} (language: {:?}, framework: {:?})",
            query, language, framework
        );

        // Create enhanced filters for code search
        let filters = SearchFilters {
            entity_types: Some(vec![
                "Class".to_string(),
                "Function".to_string(),
                "API".to_string(),
                "Module".to_string(),
            ]),
            time_range: None,
            sources: None,
            min_confidence: Some(0.6),
        };

        // Perform the search with code-specific context
        let mut results = self.search(query, limit, Some(filters)).await?;

        // Post-process results to prioritize code-relevant content
        for result in &mut results {
            if let Some(content) = result.node.get("content").and_then(|v| v.as_str()) {
                let code_relevance = self.calculate_code_relevance(content, &language, &framework);
                result.score *= code_relevance;

                if let Some(ref mut explanation) = result.explanation {
                    explanation.push_str(&format!(" | Code relevance: {:.3}", code_relevance));
                }
            }
        }

        // Re-sort by updated scores
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        info!("Found {} code-specific results", results.len());
        Ok(results)
    }

    /// Calculate code relevance score
    fn calculate_code_relevance(
        &self,
        content: &str,
        language: &Option<String>,
        framework: &Option<String>,
    ) -> f32 {
        let mut relevance = 1.0f32;
        let content_lower = content.to_lowercase();

        // Language-specific keywords
        if let Some(lang) = language {
            let lang_lower = lang.to_lowercase();
            if content_lower.contains(&lang_lower) {
                relevance *= 1.5;
            }

            // Language-specific syntax patterns
            match lang_lower.as_str() {
                "rust" => {
                    if content_lower.contains("fn ")
                        || content_lower.contains("struct ")
                        || content_lower.contains("impl ")
                    {
                        relevance *= 1.3;
                    }
                }
                "python" => {
                    if content_lower.contains("def ")
                        || content_lower.contains("class ")
                        || content_lower.contains("import ")
                    {
                        relevance *= 1.3;
                    }
                }
                "javascript" | "typescript" => {
                    if content_lower.contains("function ")
                        || content_lower.contains("const ")
                        || content_lower.contains("async ")
                    {
                        relevance *= 1.3;
                    }
                }
                _ => {}
            }
        }

        // Framework-specific keywords
        if let Some(fw) = framework {
            let fw_lower = fw.to_lowercase();
            if content_lower.contains(&fw_lower) {
                relevance *= 1.4;
            }
        }

        // General code indicators
        let code_indicators = [
            "function",
            "class",
            "method",
            "api",
            "endpoint",
            "database",
            "algorithm",
            "pattern",
            "architecture",
            "design",
            "implementation",
            "bug",
            "fix",
            "test",
            "deploy",
            "refactor",
            "optimize",
        ];

        let indicator_count = code_indicators
            .iter()
            .filter(|&indicator| content_lower.contains(indicator))
            .count();

        relevance *= 1.0 + (indicator_count as f32 * 0.1);

        // Normalize to reasonable range
        relevance.min(2.0)
    }

    /// Detect and resolve temporal conflicts in edges
    #[instrument(skip(self))]
    pub async fn resolve_temporal_conflicts(&self) -> Result<Vec<TemporalConflictResolution>> {
        info!("Detecting and resolving temporal conflicts");

        let edges = self
            .episode_processor
            .storage()
            .get_all_edges()
            .await
            .map_err(|e| {
                Error::Storage(format!(
                    "Failed to get edges for conflict resolution: {}",
                    e
                ))
            })?;

        let mut resolutions = Vec::new();

        // Group edges by source-target-type combination
        let mut edge_groups: std::collections::HashMap<(Uuid, Uuid, String), Vec<&Edge>> =
            std::collections::HashMap::new();

        for edge in &edges {
            let key = (edge.source_id, edge.target_id, edge.relationship.clone());
            edge_groups.entry(key).or_default().push(edge);
        }

        // Check for temporal overlaps within each group
        for ((source_id, target_id, rel_type), group_edges) in edge_groups {
            if group_edges.len() > 1 {
                let conflicts = self.detect_temporal_overlaps(&group_edges);
                if !conflicts.is_empty() {
                    let resolution = self
                        .resolve_edge_conflicts(source_id, target_id, &rel_type, conflicts)
                        .await?;
                    resolutions.push(resolution);
                }
            }
        }

        info!("Resolved {} temporal conflicts", resolutions.len());
        Ok(resolutions)
    }

    /// Detect temporal overlaps between edges
    fn detect_temporal_overlaps(&self, edges: &[&Edge]) -> Vec<TemporalConflict> {
        let mut conflicts = Vec::new();

        for i in 0..edges.len() {
            for j in (i + 1)..edges.len() {
                let edge1 = edges[i];
                let edge2 = edges[j];

                // Check if temporal ranges overlap
                let start1 = edge1.temporal.valid_from;
                let end1 = edge1.temporal.valid_to.unwrap_or(DateTime::<Utc>::MAX_UTC);
                let start2 = edge2.temporal.valid_from;
                let end2 = edge2.temporal.valid_to.unwrap_or(DateTime::<Utc>::MAX_UTC);

                // Overlap condition: start1 <= end2 && start2 <= end1
                if start1 <= end2 && start2 <= end1 {
                    conflicts.push(TemporalConflict {
                        edge1_id: edge1.id,
                        edge2_id: edge2.id,
                        overlap_start: start1.max(start2),
                        overlap_end: end1.min(end2),
                        conflict_type: ConflictType::TemporalOverlap,
                    });
                }
            }
        }

        conflicts
    }

    /// Resolve conflicts between edges
    async fn resolve_edge_conflicts(
        &self,
        source_id: Uuid,
        target_id: Uuid,
        relationship_type: &str,
        conflicts: Vec<TemporalConflict>,
    ) -> Result<TemporalConflictResolution> {
        info!(
            "Resolving {} conflicts for edge {}->{} ({})",
            conflicts.len(),
            source_id,
            target_id,
            relationship_type
        );

        let mut _resolved_edges: Vec<Edge> = Vec::new();
        let mut actions: Vec<ResolutionAction> = Vec::new();

        for conflict in &conflicts {
            // Strategy: Keep the more recent edge, truncate the older one
            let action = ResolutionAction::TruncateOlderEdge {
                older_edge_id: conflict.edge1_id,
                new_end_time: conflict.overlap_start,
            };
            actions.push(action);
        }

        Ok(TemporalConflictResolution {
            source_id,
            target_id,
            relationship_type: relationship_type.to_string(),
            conflicts,
            actions,
            resolved_at: Utc::now(),
        })
    }

    /// Invalidate edges that are no longer valid
    #[instrument(skip(self))]
    pub async fn invalidate_expired_edges(&self, current_time: DateTime<Utc>) -> Result<Vec<Uuid>> {
        info!("Invalidating edges expired before: {}", current_time);

        let edges = self
            .episode_processor
            .storage()
            .get_all_edges()
            .await
            .map_err(|e| Error::Storage(format!("Failed to get edges for invalidation: {}", e)))?;

        let mut invalidated = Vec::new();

        for edge in edges {
            if let Some(valid_to) = edge.temporal.valid_to {
                if valid_to < current_time {
                    // Mark edge as invalid (in a real implementation, we'd update the storage)
                    invalidated.push(edge.id);
                    info!("Invalidated edge {} (expired at {})", edge.id, valid_to);
                }
            }
        }

        info!("Invalidated {} expired edges", invalidated.len());
        Ok(invalidated)
    }

    /// Get edge version history with conflict information
    #[instrument(skip(self))]
    pub async fn get_edge_version_history_with_conflicts(
        &self,
        source_id: Uuid,
        target_id: Uuid,
        relationship_type: &str,
    ) -> Result<EdgeVersionHistory> {
        info!(
            "Getting version history for edge {}->{} ({})",
            source_id, target_id, relationship_type
        );

        let all_edges = self
            .episode_processor
            .storage()
            .get_all_edges()
            .await
            .map_err(|e| Error::Storage(format!("Failed to get edges: {}", e)))?;

        // Filter edges matching the criteria
        let matching_edges: Vec<_> = all_edges
            .into_iter()
            .filter(|edge| {
                edge.source_id == source_id
                    && edge.target_id == target_id
                    && edge.relationship == relationship_type
            })
            .collect();

        // Sort by creation time
        let mut sorted_edges = matching_edges;
        sorted_edges.sort_by_key(|edge| edge.temporal.valid_from);

        // Detect conflicts in the history
        let edge_refs: Vec<&Edge> = sorted_edges.iter().collect();
        let conflicts = self.detect_temporal_overlaps(&edge_refs);

        Ok(EdgeVersionHistory {
            source_id,
            target_id,
            relationship_type: relationship_type.to_string(),
            versions: sorted_edges,
            conflicts,
            last_updated: Utc::now(),
        })
    }
}

// Note: Clone is not implemented for Graphiti because EpisodeProcessor
// contains complex state that shouldn't be cloned. Use Arc<Graphiti> instead.

/// Search filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilters {
    /// Entity types to include
    pub entity_types: Option<Vec<String>>,
    /// Time range filter
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Source filter
    pub sources: Option<Vec<String>>,
    /// Minimum confidence
    pub min_confidence: Option<f32>,
}

/// Vector search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// Node ID
    pub node_id: Uuid,
    /// Content that was matched
    pub content: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
}

/// Temporal conflict between edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConflict {
    /// First edge ID
    pub edge1_id: Uuid,
    /// Second edge ID
    pub edge2_id: Uuid,
    /// Start of overlap period
    pub overlap_start: DateTime<Utc>,
    /// End of overlap period
    pub overlap_end: DateTime<Utc>,
    /// Type of conflict
    pub conflict_type: ConflictType,
}

/// Type of temporal conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    /// Temporal ranges overlap
    TemporalOverlap,
    /// Contradictory information
    ContentConflict,
    /// Version mismatch
    VersionConflict,
}

/// Resolution action for temporal conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionAction {
    /// Truncate older edge to resolve overlap
    TruncateOlderEdge {
        /// ID of the older edge
        older_edge_id: Uuid,
        /// New end time for the older edge
        new_end_time: DateTime<Utc>,
    },
    /// Merge conflicting edges
    MergeEdges {
        /// IDs of edges to merge
        edge_ids: Vec<Uuid>,
        /// New merged edge properties
        merged_properties: serde_json::Value,
    },
    /// Mark edge as invalid
    InvalidateEdge {
        /// Edge ID to invalidate
        edge_id: Uuid,
        /// Reason for invalidation
        reason: String,
    },
}

/// Result of temporal conflict resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConflictResolution {
    /// Source node ID
    pub source_id: Uuid,
    /// Target node ID
    pub target_id: Uuid,
    /// Relationship type
    pub relationship_type: String,
    /// Detected conflicts
    pub conflicts: Vec<TemporalConflict>,
    /// Actions taken to resolve conflicts
    pub actions: Vec<ResolutionAction>,
    /// When resolution was performed
    pub resolved_at: DateTime<Utc>,
}

/// Edge version history with conflict information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeVersionHistory {
    /// Source node ID
    pub source_id: Uuid,
    /// Target node ID
    pub target_id: Uuid,
    /// Relationship type
    pub relationship_type: String,
    /// All versions of the edge
    pub versions: Vec<Edge>,
    /// Conflicts detected in the history
    pub conflicts: Vec<TemporalConflict>,
    /// Last update time
    pub last_updated: DateTime<Utc>,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The node
    pub node: serde_json::Value, // Simplified for now
    /// Relevance score
    pub score: f32,
    /// Explanation of the score
    pub explanation: Option<String>,
    /// Path from query to result
    pub path: Option<Vec<Uuid>>,
}

/// Neighbor node in graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborNode {
    /// The node
    pub node: serde_json::Value, // Simplified for now
    /// The connecting edge
    pub edge: Edge,
    /// Distance from source
    pub distance: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = GraphitiConfig::default();
        assert_eq!(config.name, "default");
        assert!(config.enable_deduplication);
        assert_eq!(config.min_entity_confidence, 0.7);
    }
}
