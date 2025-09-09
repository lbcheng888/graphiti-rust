//! Episode processing pipeline

use crate::error::Error;
use crate::error::Result;
use crate::graph::Edge;
use crate::graph::EntityNode;
use crate::graph::EpisodeNode;
use crate::graph::TemporalMetadata;
use crate::storage::GraphStorage;
use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;
use tracing::info;
use tracing::instrument;
use tracing::warn;

/// Simple entity type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityType {
    /// Type ID
    pub id: i32,
    /// Type name
    pub name: String,
    /// Type description
    pub description: String,
}

/// Trait for LLM client implementations
pub trait LLMClient: Send + Sync {
    /// Generate text completion from a prompt
    fn complete(&self, prompt: &str) -> impl std::future::Future<Output = Result<String>> + Send;
}

/// Trait for embedding client implementations
pub trait EmbeddingClient: Send + Sync {
    /// Generate embeddings for text
    fn embed(&self, text: &str) -> impl std::future::Future<Output = Result<Vec<f32>>> + Send;

    /// Generate embeddings for multiple texts
    fn embed_batch(
        &self,
        texts: &[String],
    ) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send;
}

/// Trait for cross-encoder reranking models
#[allow(async_fn_in_trait)]
pub trait CrossEncoderClient: Send + Sync {
    /// Rerank query-document pairs and return relevance scores
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>>;

    /// Rerank with batch processing for efficiency
    async fn rerank_batch(&self, queries: &[String], documents: &[String])
        -> Result<Vec<Vec<f32>>>;
}

/// Extracted entity from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Name of the extracted entity
    pub name: String,
    /// Type ID of the entity
    pub entity_type_id: i32,
    /// Confidence score for the extraction
    pub confidence: Option<f32>,
}

/// Extracted edge from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEdge {
    /// Source entity name
    pub source_entity: String,
    /// Target entity name
    pub target_entity: String,
    /// Type of relationship
    pub relationship: String,
    /// Description of the relationship
    pub description: String,
    /// Confidence score for the extraction
    pub confidence: Option<f32>,
    /// When this relationship is valid
    pub valid_at: Option<DateTime<Utc>>,
}

/// Episode processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeProcessingResult {
    /// The processed episode
    pub episode: EpisodeNode,
    /// Extracted entities
    pub entities: Vec<EntityNode>,
    /// Extracted relationships
    pub edges: Vec<Edge>,
    /// Processing statistics
    pub stats: ProcessingStats,
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Number of entities extracted
    pub entities_extracted: usize,
    /// Number of edges extracted
    pub edges_extracted: usize,
    /// Number of duplicates found
    pub duplicates_found: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Number of retries performed
    pub retries_performed: usize,
    /// Number of fallback operations used
    pub fallbacks_used: usize,
    /// Number of errors encountered
    pub errors_encountered: usize,
}

/// Processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Maximum number of retries for LLM calls
    pub max_retries: usize,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Enable fallback to rule-based extraction
    pub enable_fallback: bool,
    /// Batch size for processing multiple episodes
    pub batch_size: usize,
    /// Timeout for individual operations in seconds
    pub operation_timeout_secs: u64,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay_ms: 1000,
            enable_fallback: true,
            batch_size: 10,
            operation_timeout_secs: 30,
        }
    }
}

/// Batch processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingResult {
    /// Successfully processed episodes
    pub successful: Vec<EpisodeProcessingResult>,
    /// Failed episodes with error messages
    pub failed: Vec<(EpisodeNode, String)>,
    /// Overall batch statistics
    pub batch_stats: BatchStats,
}

/// Batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    /// Total episodes in batch
    pub total_episodes: usize,
    /// Successfully processed episodes
    pub successful_episodes: usize,
    /// Failed episodes
    pub failed_episodes: usize,
    /// Total processing time in milliseconds
    pub total_processing_time_ms: u64,
    /// Average processing time per episode
    pub avg_processing_time_ms: u64,
}

/// Episode processor handles the complete pipeline
pub struct EpisodeProcessor<S, L, E>
where
    S: GraphStorage,
    L: LLMClient,
    E: EmbeddingClient,
{
    storage: Arc<S>,
    llm_client: Arc<L>,
    embedding_client: Arc<E>,
    entity_types: Vec<EntityType>,
    config: ProcessingConfig,
}

#[allow(dead_code)]
impl<S, L, E> EpisodeProcessor<S, L, E>
where
    S: GraphStorage<Error = Error> + 'static,
    L: LLMClient + 'static,
    E: EmbeddingClient + 'static,
{
    /// Create a new episode processor
    pub fn new(
        storage: S,
        llm_client: L,
        embedding_client: E,
        entity_types: Vec<EntityType>,
    ) -> Result<Self> {
        Ok(Self {
            storage: Arc::new(storage),
            llm_client: Arc::new(llm_client),
            embedding_client: Arc::new(embedding_client),
            entity_types,
            config: ProcessingConfig::default(),
        })
    }

    /// Create a new episode processor with custom configuration
    pub fn with_config(
        storage: S,
        llm_client: L,
        embedding_client: E,
        entity_types: Vec<EntityType>,
        config: ProcessingConfig,
    ) -> Result<Self> {
        Ok(Self {
            storage: Arc::new(storage),
            llm_client: Arc::new(llm_client),
            embedding_client: Arc::new(embedding_client),
            entity_types,
            config,
        })
    }

    /// Process a complete episode with enhanced error handling and retry logic
    #[instrument(skip(self, episode))]
    pub async fn process_episode(
        &self,
        episode: EpisodeNode,
        previous_episodes: Vec<EpisodeNode>,
    ) -> Result<EpisodeProcessingResult> {
        let start_time = std::time::Instant::now();
        info!("Processing episode: {}", episode.name);

        let mut stats = ProcessingStats {
            entities_extracted: 0,
            edges_extracted: 0,
            duplicates_found: 0,
            processing_time_ms: 0,
            retries_performed: 0,
            fallbacks_used: 0,
            errors_encountered: 0,
        };

        // Step 1: Extract entities using LLM with retry logic
        let extracted_entities = self
            .extract_entities_with_retry(&episode, &previous_episodes, &mut stats)
            .await?;
        info!("Extracted {} entities", extracted_entities.len());
        stats.entities_extracted = extracted_entities.len();

        // Step 2: Extract relationships using LLM with retry logic
        let extracted_edges = self
            .extract_relationships_with_retry(
                &episode,
                &extracted_entities,
                &previous_episodes,
                &mut stats,
            )
            .await?;
        info!("Extracted {} relationships", extracted_edges.len());
        stats.edges_extracted = extracted_edges.len();

        // Step 3: Deduplicate entities and relationships
        let (deduplicated_entities, entity_duplicates) = self
            .deduplicate_entities_with_stats(&extracted_entities)
            .await?;
        let (deduplicated_edges, edge_duplicates) =
            self.deduplicate_edges_with_stats(&extracted_edges).await?;
        stats.duplicates_found = entity_duplicates + edge_duplicates;
        info!(
            "After deduplication: {} entities, {} edges, {} duplicates found",
            deduplicated_entities.len(),
            deduplicated_edges.len(),
            stats.duplicates_found
        );

        // Step 4: Convert to graph nodes and edges with error handling
        let (entities, edges) = self
            .convert_to_graph_objects(&deduplicated_entities, &deduplicated_edges, &mut stats)
            .await?;

        // Step 5: Store entities and edges with transaction-like behavior
        self.store_graph_objects(&episode, &entities, &edges, &mut stats)
            .await?;

        stats.processing_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(EpisodeProcessingResult {
            episode,
            entities,
            edges,
            stats,
        })
    }

    /// Process multiple episodes in batch
    #[instrument(skip(self, episodes))]
    pub async fn process_episodes_batch(
        &self,
        episodes: Vec<EpisodeNode>,
    ) -> Result<BatchProcessingResult> {
        let start_time = std::time::Instant::now();
        info!("Processing batch of {} episodes", episodes.len());

        let mut successful = Vec::new();
        let mut failed = Vec::new();
        let total_episodes = episodes.len();

        // Process episodes in chunks based on batch size
        for chunk in episodes.chunks(self.config.batch_size) {
            let mut chunk_tasks = Vec::new();

            for episode in chunk {
                let episode_clone = episode.clone();
                let processor = self.clone_for_concurrent_processing();

                let task = tokio::spawn(async move {
                    processor
                        .process_episode(episode_clone.clone(), Vec::new())
                        .await
                        .map_err(|e| (episode_clone, e.to_string()))
                });

                chunk_tasks.push(task);
            }

            // Wait for all tasks in this chunk to complete
            for task in chunk_tasks {
                match task.await {
                    Ok(Ok(result)) => successful.push(result),
                    Ok(Err((episode, error))) => failed.push((episode, error)),
                    Err(join_error) => {
                        warn!("Task join error: {}", join_error);
                        // We can't recover the episode here, so we'll skip it
                    }
                }
            }
        }

        let total_processing_time = start_time.elapsed().as_millis() as u64;
        let successful_count = successful.len();
        let failed_count = failed.len();

        let batch_stats = BatchStats {
            total_episodes,
            successful_episodes: successful_count,
            failed_episodes: failed_count,
            total_processing_time_ms: total_processing_time,
            avg_processing_time_ms: if total_episodes > 0 {
                total_processing_time / total_episodes as u64
            } else {
                0
            },
        };

        info!(
            "Batch processing completed: {}/{} successful",
            successful_count, total_episodes
        );

        Ok(BatchProcessingResult {
            successful,
            failed,
            batch_stats,
        })
    }

    /// Clone processor for concurrent processing (simplified)
    fn clone_for_concurrent_processing(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            llm_client: self.llm_client.clone(),
            embedding_client: self.embedding_client.clone(),
            entity_types: self.entity_types.clone(),
            config: self.config.clone(),
        }
    }

    /// Extract entities from episode using LLM
    async fn extract_entities_from_episode(
        &self,
        episode: &EpisodeNode,
    ) -> Result<Vec<ExtractedEntity>> {
        info!("Extracting entities from episode: {}", episode.name);

        // Create entity extraction prompt
        let prompt = self.create_entity_extraction_prompt(&episode.content);

        // Call LLM for entity extraction
        match self.llm_client.complete(&prompt).await {
            Ok(response) => {
                // Parse LLM response
                match self.parse_entity_response(&response) {
                    Ok(entities) => {
                        info!("Successfully extracted {} entities", entities.len());
                        Ok(entities)
                    }
                    Err(e) => {
                        warn!(
                            "Failed to parse LLM response, falling back to rule-based extraction: {}",
                            e
                        );
                        Ok(self.fallback_entity_extraction(&episode.content))
                    }
                }
            }
            Err(e) => {
                warn!(
                    "LLM call failed, falling back to rule-based extraction: {}",
                    e
                );
                Ok(self.fallback_entity_extraction(&episode.content))
            }
        }
    }

    /// Extract relationships from episode using LLM
    async fn extract_relationships_from_episode(
        &self,
        episode: &EpisodeNode,
        entities: &[ExtractedEntity],
    ) -> Result<Vec<ExtractedEdge>> {
        info!("Extracting relationships from episode: {}", episode.name);

        // Create relationship extraction prompt
        let prompt = self.create_relationship_extraction_prompt(&episode.content, entities);

        // Call LLM for relationship extraction
        match self.llm_client.complete(&prompt).await {
            Ok(response) => {
                // Parse LLM response
                match self.parse_relationship_response(&response) {
                    Ok(relationships) => {
                        info!(
                            "Successfully extracted {} relationships",
                            relationships.len()
                        );
                        Ok(relationships)
                    }
                    Err(e) => {
                        warn!(
                            "Failed to parse LLM response, falling back to rule-based extraction: {}",
                            e
                        );
                        Ok(self.fallback_relationship_extraction(entities, &episode.content))
                    }
                }
            }
            Err(e) => {
                warn!(
                    "LLM call failed, falling back to rule-based extraction: {}",
                    e
                );
                Ok(self.fallback_relationship_extraction(entities, &episode.content))
            }
        }
    }

    /// Convert extracted entity to EntityNode
    async fn convert_to_entity_node(
        &self,
        extracted_entity: &ExtractedEntity,
    ) -> Result<EntityNode> {
        // Generate embedding for the entity
        let embedding = self
            .embedding_client
            .embed(&extracted_entity.name)
            .await
            .map_err(|e| {
                crate::error::Error::Processing(format!("Embedding generation failed: {}", e))
            })?;

        // Find entity type name
        let entity_type_name = self
            .entity_types
            .iter()
            .find(|et| et.id == extracted_entity.entity_type_id)
            .map(|et| et.name.clone())
            .unwrap_or_else(|| "Entity".to_string());

        Ok(EntityNode {
            id: uuid::Uuid::new_v4(),
            name: extracted_entity.name.clone(),
            entity_type: entity_type_name.clone(),
            labels: vec![entity_type_name],
            properties: serde_json::json!({
                "confidence": extracted_entity.confidence,
                "entity_type_id": extracted_entity.entity_type_id,
            }),
            temporal: TemporalMetadata {
                created_at: chrono::Utc::now(),
                valid_from: chrono::Utc::now(),
                valid_to: None,
                expired_at: None,
            },
            embedding: Some(embedding),
        })
    }

    /// Convert extracted edge to Edge
    async fn convert_to_edge(&self, extracted_edge: &ExtractedEdge) -> Result<Edge> {
        Ok(Edge {
            id: uuid::Uuid::new_v4(),
            source_id: uuid::Uuid::new_v4(), // TODO: Map entity names to IDs
            target_id: uuid::Uuid::new_v4(), // TODO: Map entity names to IDs
            relationship: extracted_edge.relationship.clone(),
            properties: serde_json::json!({
                "description": extracted_edge.description,
                "confidence": extracted_edge.confidence,
                "source_entity": extracted_edge.source_entity,
                "target_entity": extracted_edge.target_entity,
            }),
            temporal: TemporalMetadata {
                created_at: chrono::Utc::now(),
                valid_from: extracted_edge.valid_at.unwrap_or_else(chrono::Utc::now),
                valid_to: None,
                expired_at: None,
            },
            weight: extracted_edge.confidence.unwrap_or(0.5),
        })
    }

    /// Create entity extraction prompt
    fn create_entity_extraction_prompt(&self, content: &str) -> String {
        format!(
            r#"You are an expert at extracting entities from text. Extract entities from the following text.

Entity Types Available:
{}

Text to analyze:
{}

Return a JSON object with the following structure:
{{
  "extracted_entities": [
    {{
      "name": "entity name",
      "entity_type_id": 1,
      "confidence": 0.95
    }}
  ]
}}

Extract entities now:"#,
            self.entity_types
                .iter()
                .map(|et| format!("- {} (ID: {}): {}", et.name, et.id, et.description))
                .collect::<Vec<_>>()
                .join("\n"),
            content
        )
    }

    /// Create relationship extraction prompt
    fn create_relationship_extraction_prompt(
        &self,
        content: &str,
        entities: &[ExtractedEntity],
    ) -> String {
        let entity_list = entities
            .iter()
            .map(|e| format!("- {} (Type ID: {})", e.name, e.entity_type_id))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"You are an expert at extracting relationships between entities from text.

Available Entities:
{}

Text to analyze:
{}

Available Relationship Types:
- KNOWS
- WORKS_WITH
- MANAGES
- COLLABORATES_WITH
- REPORTS_TO
- FRIENDS_WITH
- RELATED_TO

Return a JSON object with the following structure:
{{
  "extracted_edges": [
    {{
      "source_entity": "entity1 name",
      "target_entity": "entity2 name",
      "relationship": "KNOWS",
      "description": "description of relationship",
      "confidence": 0.95
    }}
  ]
}}

Extract relationships now:"#,
            entity_list, content
        )
    }

    /// Parse entity extraction response
    fn parse_entity_response(&self, response: &str) -> Result<Vec<ExtractedEntity>> {
        // Try to find JSON in the response
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];

        // Parse the JSON
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| Error::Processing(format!("Failed to parse JSON: {}", e)))?;

        let mut entities = Vec::new();

        if let Some(extracted_entities) = parsed.get("extracted_entities") {
            if let Some(entities_array) = extracted_entities.as_array() {
                for entity_value in entities_array {
                    if let (Some(name), Some(entity_type_id)) = (
                        entity_value.get("name").and_then(|v| v.as_str()),
                        entity_value.get("entity_type_id").and_then(|v| v.as_i64()),
                    ) {
                        let confidence = entity_value
                            .get("confidence")
                            .and_then(|v| v.as_f64())
                            .map(|c| c as f32);

                        entities.push(ExtractedEntity {
                            name: name.to_string(),
                            entity_type_id: entity_type_id as i32,
                            confidence,
                        });
                    }
                }
            }
        }

        Ok(entities)
    }

    /// Parse relationship extraction response
    fn parse_relationship_response(&self, response: &str) -> Result<Vec<ExtractedEdge>> {
        // Try to find JSON in the response
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];

        // Parse the JSON
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| Error::Processing(format!("Failed to parse JSON: {}", e)))?;

        let mut edges = Vec::new();

        if let Some(extracted_edges) = parsed.get("extracted_edges") {
            if let Some(edges_array) = extracted_edges.as_array() {
                for edge_value in edges_array {
                    if let (Some(source), Some(target), Some(relationship)) = (
                        edge_value.get("source_entity").and_then(|v| v.as_str()),
                        edge_value.get("target_entity").and_then(|v| v.as_str()),
                        edge_value.get("relationship").and_then(|v| v.as_str()),
                    ) {
                        let description = edge_value
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        let confidence = edge_value
                            .get("confidence")
                            .and_then(|v| v.as_f64())
                            .map(|c| c as f32);

                        edges.push(ExtractedEdge {
                            source_entity: source.to_string(),
                            target_entity: target.to_string(),
                            relationship: relationship.to_string(),
                            description,
                            confidence,
                            valid_at: Some(Utc::now()),
                        });
                    }
                }
            }
        }

        Ok(edges)
    }

    /// Fallback rule-based entity extraction
    fn fallback_entity_extraction(&self, text: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();

        // Simple rule-based extraction
        let words: Vec<&str> = text.split_whitespace().collect();

        for word in words {
            let clean_word = word.trim_end_matches(&['.', ',', '!', '?', ';', ':', '"', '\''][..]);

            // Extract capitalized words (potential names/entities)
            if clean_word.chars().next().unwrap_or('a').is_uppercase() && clean_word.len() > 2 {
                if ![
                    "The",
                    "This",
                    "That",
                    "They",
                    "There",
                    "Then",
                    "Today",
                    "Yesterday",
                    "When",
                    "Where",
                    "What",
                    "How",
                    "Why",
                ]
                .contains(&clean_word)
                {
                    entities.push(ExtractedEntity {
                        name: clean_word.to_string(),
                        entity_type_id: 0,     // Default entity type
                        confidence: Some(0.6), // Lower confidence for rule-based
                    });
                }
            }
        }

        // Deduplicate
        entities.sort_by(|a, b| a.name.cmp(&b.name));
        entities.dedup_by(|a, b| a.name == b.name);

        entities
    }

    /// Fallback rule-based relationship extraction
    fn fallback_relationship_extraction(
        &self,
        entities: &[ExtractedEntity],
        text: &str,
    ) -> Vec<ExtractedEdge> {
        let mut edges = Vec::new();

        // Simple rule-based relationship inference
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let entity1 = &entities[i];
                let entity2 = &entities[j];

                // Determine relationship type based on context
                let relationship_type = if text.to_lowercase().contains("met")
                    || text.to_lowercase().contains("meet")
                {
                    "MEETS"
                } else if text.to_lowercase().contains("work")
                    || text.to_lowercase().contains("collaborate")
                {
                    "WORKS_WITH"
                } else if text.to_lowercase().contains("friend")
                    || text.to_lowercase().contains("know")
                {
                    "KNOWS"
                } else if text.to_lowercase().contains("manage")
                    || text.to_lowercase().contains("lead")
                {
                    "MANAGES"
                } else {
                    "RELATED_TO"
                };

                edges.push(ExtractedEdge {
                    source_entity: entity1.name.clone(),
                    target_entity: entity2.name.clone(),
                    relationship: relationship_type.to_string(),
                    description: format!(
                        "Inferred relationship between {} and {}",
                        entity1.name, entity2.name
                    ),
                    confidence: Some(0.5), // Lower confidence for rule-based
                    valid_at: Some(Utc::now()),
                });
            }
        }

        edges
    }

    /// Deduplicate extracted entities using advanced similarity matching
    async fn deduplicate_entities(
        &self,
        entities: &[ExtractedEntity],
    ) -> Result<Vec<ExtractedEntity>> {
        info!(
            "Deduplicating {} entities with advanced similarity",
            entities.len()
        );

        let mut deduplicated = Vec::new();
        let mut processed = std::collections::HashSet::new();

        for (i, entity) in entities.iter().enumerate() {
            if processed.contains(&i) {
                continue;
            }

            let mut merged_entity = entity.clone();
            processed.insert(i);

            // Find similar entities using multiple similarity metrics
            for (j, other_entity) in entities.iter().enumerate().skip(i + 1) {
                if processed.contains(&j) {
                    continue;
                }

                let similarity = self
                    .calculate_advanced_entity_similarity(entity, other_entity)
                    .await?;
                if similarity > 0.75 {
                    // Merge entities with confidence weighting
                    merged_entity =
                        self.merge_entities_weighted(&merged_entity, other_entity, similarity);
                    processed.insert(j);
                    info!(
                        "Merged entities '{}' and '{}' (similarity: {:.3})",
                        entity.name, other_entity.name, similarity
                    );
                }
            }

            deduplicated.push(merged_entity);
        }

        info!("Deduplicated to {} entities", deduplicated.len());
        Ok(deduplicated)
    }

    /// Advanced deduplication with conflict resolution strategies
    async fn deduplicate_entities_advanced(
        &self,
        entities: &[ExtractedEntity],
    ) -> Result<Vec<ExtractedEntity>> {
        info!(
            "Advanced deduplication with conflict resolution for {} entities",
            entities.len()
        );

        let mut deduplicated = Vec::new();
        let mut processed = std::collections::HashSet::new();
        let mut conflict_groups = Vec::new();

        // Phase 1: Group similar entities
        for (i, entity) in entities.iter().enumerate() {
            if processed.contains(&i) {
                continue;
            }

            let mut group = vec![i];
            processed.insert(i);

            // Find all similar entities
            for (j, other_entity) in entities.iter().enumerate().skip(i + 1) {
                if processed.contains(&j) {
                    continue;
                }

                let similarity = self
                    .calculate_advanced_entity_similarity(entity, other_entity)
                    .await?;
                if similarity > 0.75 {
                    group.push(j);
                    processed.insert(j);
                }
            }

            if group.len() > 1 {
                conflict_groups.push(group);
            } else {
                deduplicated.push(entity.clone());
            }
        }

        // Phase 2: Resolve conflicts using advanced strategies
        for group in conflict_groups {
            let group_entities: Vec<&ExtractedEntity> =
                group.iter().map(|&i| &entities[i]).collect();
            let resolved_entity = self.resolve_entity_conflicts(&group_entities).await?;

            info!(
                "Resolved conflict group of {} entities into single entity: '{}'",
                group.len(),
                resolved_entity.name
            );

            deduplicated.push(resolved_entity);
        }

        info!(
            "Advanced deduplication completed: {} -> {} entities",
            entities.len(),
            deduplicated.len()
        );
        Ok(deduplicated)
    }

    /// Resolve conflicts between similar entities using multiple strategies
    async fn resolve_entity_conflicts(
        &self,
        entities: &[&ExtractedEntity],
    ) -> Result<ExtractedEntity> {
        if entities.is_empty() {
            return Err(Error::Processing(
                "Empty entity group for conflict resolution".to_string(),
            ));
        }

        if entities.len() == 1 {
            return Ok((*entities[0]).clone());
        }

        // Strategy 1: Confidence-based selection
        let highest_confidence_entity = entities
            .iter()
            .max_by(|a, b| {
                let conf_a = a.confidence.unwrap_or(0.0);
                let conf_b = b.confidence.unwrap_or(0.0);
                conf_a
                    .partial_cmp(&conf_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Strategy 2: Name length and completeness
        let most_complete_entity = entities
            .iter()
            .max_by_key(|entity| entity.name.len())
            .unwrap();

        // Strategy 3: Type consistency check
        let type_counts =
            entities
                .iter()
                .fold(std::collections::HashMap::new(), |mut acc, entity| {
                    *acc.entry(entity.entity_type_id).or_insert(0) += 1;
                    acc
                });

        let most_common_type = type_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&type_id, _)| type_id)
            .unwrap_or(highest_confidence_entity.entity_type_id);

        // Strategy 4: Semantic embedding similarity (if available)
        let best_semantic_match = self.find_best_semantic_match(entities).await?;

        // Combine strategies with weighted scoring
        let mut scores = Vec::new();

        for (idx, entity) in entities.iter().enumerate() {
            let mut score = 0.0f32;

            // Confidence weight (40%)
            if std::ptr::eq(entity as *const _, highest_confidence_entity as *const _) {
                score += 0.4;
            }

            // Completeness weight (20%)
            if std::ptr::eq(entity as *const _, most_complete_entity as *const _) {
                score += 0.2;
            }

            // Type consistency weight (20%)
            if entity.entity_type_id == most_common_type {
                score += 0.2;
            }

            // Semantic similarity weight (20%)
            if let Some(best_match_idx) = &best_semantic_match {
                if idx == *best_match_idx {
                    score += 0.2;
                }
            }

            scores.push((idx, entity, score));
        }

        // Select the entity with the highest combined score
        let best_entity = scores
            .iter()
            .max_by(|(_, _, score_a), (_, _, score_b)| {
                score_a
                    .partial_cmp(score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, entity, _)| *entity)
            .unwrap_or(highest_confidence_entity);

        // Create merged entity with enhanced confidence
        let merged_confidence = entities
            .iter()
            .map(|e| e.confidence.unwrap_or(0.5))
            .sum::<f32>()
            / entities.len() as f32;

        Ok(ExtractedEntity {
            name: best_entity.name.clone(),
            entity_type_id: most_common_type,
            confidence: Some(merged_confidence.min(1.0)),
        })
    }

    /// Find the best semantic match among entities (placeholder for embedding-based matching)
    async fn find_best_semantic_match(
        &self,
        entities: &[&ExtractedEntity],
    ) -> Result<Option<usize>> {
        // This would use embedding similarity in a full implementation
        // For now, return the index of the entity with the most "canonical" name (shortest, most common words)
        let best_match_idx = entities
            .iter()
            .enumerate()
            .min_by_key(|(_, entity)| {
                // Prefer shorter names with common words
                let word_count = entity.name.split_whitespace().count();
                let char_count = entity.name.len();
                word_count * 100 + char_count // Simple heuristic
            })
            .map(|(idx, _)| idx);

        Ok(best_match_idx)
    }

    /// Advanced edge deduplication with semantic understanding
    async fn deduplicate_edges_advanced(
        &self,
        edges: &[ExtractedEdge],
    ) -> Result<Vec<ExtractedEdge>> {
        info!("Advanced edge deduplication for {} edges", edges.len());

        let mut deduplicated = Vec::new();
        let mut processed = std::collections::HashSet::new();

        for (i, edge) in edges.iter().enumerate() {
            if processed.contains(&i) {
                continue;
            }

            let mut similar_edges = vec![i];
            processed.insert(i);

            // Find semantically similar edges
            for (j, other_edge) in edges.iter().enumerate().skip(i + 1) {
                if processed.contains(&j) {
                    continue;
                }

                if self
                    .are_edges_semantically_similar(edge, other_edge)
                    .await?
                {
                    similar_edges.push(j);
                    processed.insert(j);
                }
            }

            // Merge similar edges
            if similar_edges.len() > 1 {
                let edge_group: Vec<&ExtractedEdge> =
                    similar_edges.iter().map(|&idx| &edges[idx]).collect();
                let merged_edge = self.merge_similar_edges(&edge_group).await?;
                deduplicated.push(merged_edge);

                info!("Merged {} similar edges into one", similar_edges.len());
            } else {
                deduplicated.push(edge.clone());
            }
        }

        info!(
            "Advanced edge deduplication completed: {} -> {} edges",
            edges.len(),
            deduplicated.len()
        );
        Ok(deduplicated)
    }

    /// Check if two edges are semantically similar
    async fn are_edges_semantically_similar(
        &self,
        edge1: &ExtractedEdge,
        edge2: &ExtractedEdge,
    ) -> Result<bool> {
        // 1. Check if entities are the same (considering both directions)
        let same_entities = (edge1.source_entity == edge2.source_entity
            && edge1.target_entity == edge2.target_entity)
            || (edge1.source_entity == edge2.target_entity
                && edge1.target_entity == edge2.source_entity);

        // 2. Check if relationship types are semantically similar
        let similar_relationships =
            self.are_relationships_similar(&edge1.relationship, &edge2.relationship);

        // 3. Check temporal overlap
        let temporal_overlap = self.check_temporal_overlap(edge1, edge2);

        // Edges are similar if they have same entities and similar relationships, or significant temporal overlap
        Ok(same_entities && (similar_relationships || temporal_overlap))
    }

    /// Check if two relationship types are semantically similar
    fn are_relationships_similar(&self, rel1: &str, rel2: &str) -> bool {
        if rel1 == rel2 {
            return true;
        }

        // Define relationship similarity groups
        let similarity_groups = vec![
            vec!["WORKS_WITH", "COLLABORATES_WITH", "PARTNERS_WITH"],
            vec!["KNOWS", "FRIENDS_WITH", "ACQUAINTED_WITH"],
            vec!["MANAGES", "LEADS", "SUPERVISES"],
            vec!["USES", "UTILIZES", "EMPLOYS"],
            vec!["CREATES", "BUILDS", "DEVELOPS"],
            vec!["MEETS", "ENCOUNTERS", "VISITS"],
        ];

        for group in similarity_groups {
            if group.contains(&rel1) && group.contains(&rel2) {
                return true;
            }
        }

        false
    }

    /// Check temporal overlap between edges
    fn check_temporal_overlap(&self, edge1: &ExtractedEdge, edge2: &ExtractedEdge) -> bool {
        match (edge1.valid_at, edge2.valid_at) {
            (Some(time1), Some(time2)) => {
                // Consider edges temporally overlapping if they're within 24 hours
                let duration = if time1 > time2 {
                    time1 - time2
                } else {
                    time2 - time1
                };
                duration.num_hours() <= 24
            }
            _ => false, // If no temporal information, assume no overlap
        }
    }

    /// Merge similar edges into a single edge
    async fn merge_similar_edges(&self, edges: &[&ExtractedEdge]) -> Result<ExtractedEdge> {
        if edges.is_empty() {
            return Err(Error::Processing(
                "Empty edge group for merging".to_string(),
            ));
        }

        if edges.len() == 1 {
            return Ok((*edges[0]).clone());
        }

        // Choose the edge with highest confidence as base
        let base_edge = edges
            .iter()
            .max_by(|a, b| {
                let conf_a = a.confidence.unwrap_or(0.0);
                let conf_b = b.confidence.unwrap_or(0.0);
                conf_a
                    .partial_cmp(&conf_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Calculate merged confidence
        let total_confidence: f32 = edges.iter().map(|e| e.confidence.unwrap_or(0.5)).sum();
        let merged_confidence = (total_confidence / edges.len() as f32).min(1.0);

        // Choose the most specific relationship type
        let best_relationship = self.choose_best_relationship_type(edges);

        // Merge descriptions
        let merged_description = if edges.len() > 1 {
            format!(
                "Merged relationship: {}",
                edges
                    .iter()
                    .map(|e| e.description.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        } else {
            base_edge.description.clone()
        };

        // Use the most recent valid_at time
        let merged_valid_at = edges.iter().filter_map(|e| e.valid_at).max();

        Ok(ExtractedEdge {
            source_entity: base_edge.source_entity.clone(),
            target_entity: base_edge.target_entity.clone(),
            relationship: best_relationship,
            description: merged_description,
            confidence: Some(merged_confidence),
            valid_at: merged_valid_at,
        })
    }

    /// Choose the best relationship type from a group of similar edges
    fn choose_best_relationship_type(&self, edges: &[&ExtractedEdge]) -> String {
        // Count relationship types
        let mut type_counts = std::collections::HashMap::new();
        for edge in edges {
            *type_counts.entry(edge.relationship.clone()).or_insert(0) += 1;
        }

        // Return the most common relationship type
        type_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(rel_type, _)| rel_type)
            .unwrap_or_else(|| edges[0].relationship.clone())
    }

    /// Deduplicate extracted edges
    async fn deduplicate_edges(&self, edges: &[ExtractedEdge]) -> Result<Vec<ExtractedEdge>> {
        let mut deduplicated = Vec::new();
        let mut seen_relationships = std::collections::HashSet::new();

        for edge in edges {
            // Create a normalized key for the relationship
            let key = format!(
                "{}->{}:{}",
                edge.source_entity.to_lowercase().trim(),
                edge.target_entity.to_lowercase().trim(),
                edge.relationship.to_lowercase().trim()
            );

            if !seen_relationships.contains(&key) {
                seen_relationships.insert(key);
                deduplicated.push(edge.clone());
            } else {
                // If we've seen this relationship before, merge confidence scores
                if let Some(existing) = deduplicated.iter_mut().find(|e| {
                    let existing_key = format!(
                        "{}->{}:{}",
                        e.source_entity.to_lowercase().trim(),
                        e.target_entity.to_lowercase().trim(),
                        e.relationship.to_lowercase().trim()
                    );
                    existing_key == key
                }) {
                    // Take the higher confidence score
                    if let (Some(existing_conf), Some(new_conf)) =
                        (existing.confidence, edge.confidence)
                    {
                        existing.confidence = Some(existing_conf.max(new_conf));
                    } else if edge.confidence.is_some() {
                        existing.confidence = edge.confidence;
                    }

                    // Merge descriptions
                    if !existing.description.contains(&edge.description) {
                        existing.description =
                            format!("{}; {}", existing.description, edge.description);
                    }
                }
            }
        }

        Ok(deduplicated)
    }

    /// Get access to the storage for search operations
    pub fn storage(&self) -> &Arc<S> {
        &self.storage
    }

    /// Get access to the embedding client for vector operations
    pub fn embedding_client(&self) -> &Arc<E> {
        &self.embedding_client
    }

    /// Get access to the LLM client for text generation
    pub fn llm_client(&self) -> &Arc<L> {
        &self.llm_client
    }

    /// Extract entities with retry logic
    async fn extract_entities_with_retry(
        &self,
        episode: &EpisodeNode,
        _previous_episodes: &[EpisodeNode],
        stats: &mut ProcessingStats,
    ) -> Result<Vec<ExtractedEntity>> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match self.extract_entities_from_episode(episode).await {
                Ok(entities) => {
                    if attempt > 0 {
                        stats.retries_performed += attempt;
                    }
                    return Ok(entities);
                }
                Err(e) => {
                    last_error = Some(e);
                    stats.errors_encountered += 1;

                    if attempt < self.config.max_retries {
                        warn!(
                            "Entity extraction attempt {} failed, retrying in {}ms",
                            attempt + 1,
                            self.config.retry_delay_ms
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            self.config.retry_delay_ms,
                        ))
                        .await;
                    }
                }
            }
        }

        // If all retries failed, try fallback if enabled
        if self.config.enable_fallback {
            warn!("All entity extraction attempts failed, using fallback");
            stats.fallbacks_used += 1;
            Ok(self.fallback_entity_extraction(&episode.content))
        } else {
            Err(last_error.unwrap_or_else(|| Error::Processing("Unknown error".to_string())))
        }
    }

    /// Extract relationships with retry logic
    async fn extract_relationships_with_retry(
        &self,
        episode: &EpisodeNode,
        entities: &[ExtractedEntity],
        _previous_episodes: &[EpisodeNode],
        stats: &mut ProcessingStats,
    ) -> Result<Vec<ExtractedEdge>> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            match self
                .extract_relationships_from_episode(episode, entities)
                .await
            {
                Ok(edges) => {
                    if attempt > 0 {
                        stats.retries_performed += attempt;
                    }
                    return Ok(edges);
                }
                Err(e) => {
                    last_error = Some(e);
                    stats.errors_encountered += 1;

                    if attempt < self.config.max_retries {
                        warn!(
                            "Relationship extraction attempt {} failed, retrying in {}ms",
                            attempt + 1,
                            self.config.retry_delay_ms
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            self.config.retry_delay_ms,
                        ))
                        .await;
                    }
                }
            }
        }

        // If all retries failed, try fallback if enabled
        if self.config.enable_fallback {
            warn!("All relationship extraction attempts failed, using fallback");
            stats.fallbacks_used += 1;
            Ok(self.fallback_relationship_extraction(entities, &episode.content))
        } else {
            Err(last_error.unwrap_or_else(|| Error::Processing("Unknown error".to_string())))
        }
    }

    /// Deduplicate entities with statistics tracking
    async fn deduplicate_entities_with_stats(
        &self,
        entities: &[ExtractedEntity],
    ) -> Result<(Vec<ExtractedEntity>, usize)> {
        let original_count = entities.len();
        let deduplicated = self.deduplicate_entities(entities).await?;
        let duplicates_found = original_count - deduplicated.len();
        Ok((deduplicated, duplicates_found))
    }

    /// Deduplicate edges with statistics tracking
    async fn deduplicate_edges_with_stats(
        &self,
        edges: &[ExtractedEdge],
    ) -> Result<(Vec<ExtractedEdge>, usize)> {
        let original_count = edges.len();
        let deduplicated = self.deduplicate_edges(edges).await?;
        let duplicates_found = original_count - deduplicated.len();
        Ok((deduplicated, duplicates_found))
    }

    /// Convert extracted objects to graph objects with error handling
    async fn convert_to_graph_objects(
        &self,
        entities: &[ExtractedEntity],
        edges: &[ExtractedEdge],
        stats: &mut ProcessingStats,
    ) -> Result<(Vec<EntityNode>, Vec<Edge>)> {
        let mut graph_entities = Vec::new();
        let mut graph_edges = Vec::new();

        // Convert entities
        for extracted_entity in entities {
            match self.convert_to_entity_node(extracted_entity).await {
                Ok(entity_node) => graph_entities.push(entity_node),
                Err(e) => {
                    warn!(
                        "Failed to convert entity '{}': {}",
                        extracted_entity.name, e
                    );
                    stats.errors_encountered += 1;
                }
            }
        }

        // Convert edges
        for extracted_edge in edges {
            match self.convert_to_edge(extracted_edge).await {
                Ok(edge) => graph_edges.push(edge),
                Err(e) => {
                    warn!(
                        "Failed to convert edge '{}->{}': {}",
                        extracted_edge.source_entity, extracted_edge.target_entity, e
                    );
                    stats.errors_encountered += 1;
                }
            }
        }

        Ok((graph_entities, graph_edges))
    }

    /// Store graph objects with transaction-like behavior
    async fn store_graph_objects(
        &self,
        episode: &EpisodeNode,
        entities: &[EntityNode],
        edges: &[Edge],
        stats: &mut ProcessingStats,
    ) -> Result<()> {
        // Prefer batched writes for throughput; fallback到单条
        if !entities.is_empty() {
            let boxed: Vec<Box<dyn crate::graph::Node>> = entities
                .iter()
                .map(|e| Box::new(e.clone()) as Box<dyn crate::graph::Node>)
                .collect();
            if let Err(e) = self.storage.create_nodes_batch(&boxed).await {
                warn!("Batch create_nodes failed: {} — falling back", e);
                for entity in entities {
                    if let Err(e) = self.storage.create_node(entity).await {
                        warn!("Failed to store entity '{}': {}", entity.name, e);
                        stats.errors_encountered += 1;
                    }
                }
            }
        }

        if !edges.is_empty() {
            if let Err(e) = self.storage.create_edges_batch(edges).await {
                warn!("Batch create_edges failed: {} — falling back", e);
                for edge in edges {
                    if let Err(e) = self.storage.create_edge(edge).await {
                        warn!("Failed to store edge '{}': {}", edge.id, e);
                        stats.errors_encountered += 1;
                    }
                }
            }
        }

        // Store the episode
        if let Err(e) = self.storage.create_node(episode).await {
            warn!("Failed to store episode '{}': {}", episode.name, e);
            stats.errors_encountered += 1;
        }

        Ok(())
    }

    /// Calculate advanced entity similarity using multiple metrics
    async fn calculate_advanced_entity_similarity(
        &self,
        entity1: &ExtractedEntity,
        entity2: &ExtractedEntity,
    ) -> Result<f32> {
        let mut total_similarity = 0.0f32;
        let mut weight_sum = 0.0f32;

        // 1. Name similarity (highest weight)
        let name_similarity = self.calculate_string_similarity(&entity1.name, &entity2.name);
        total_similarity += name_similarity * 0.5;
        weight_sum += 0.5;

        // 2. Type similarity
        if entity1.entity_type_id == entity2.entity_type_id {
            total_similarity += 1.0 * 0.2;
        }
        weight_sum += 0.2;

        // 3. Semantic similarity using embeddings (if available)
        match self.calculate_semantic_similarity(entity1, entity2).await {
            Ok(semantic_sim) => {
                total_similarity += semantic_sim * 0.3;
                weight_sum += 0.3;
            }
            Err(_) => {
                // Fallback to simple text similarity
                let text_sim = self.calculate_string_similarity(&entity1.name, &entity2.name);
                total_similarity += text_sim * 0.3;
                weight_sum += 0.3;
            }
        }

        Ok(if weight_sum > 0.0 {
            total_similarity / weight_sum
        } else {
            0.0
        })
    }

    /// Calculate semantic similarity using embeddings
    async fn calculate_semantic_similarity(
        &self,
        entity1: &ExtractedEntity,
        entity2: &ExtractedEntity,
    ) -> Result<f32> {
        let text1 = format!("{} {}", entity1.name, entity1.entity_type_id);
        let text2 = format!("{} {}", entity2.name, entity2.entity_type_id);

        let embedding1 = self.embedding_client.embed(&text1).await?;
        let embedding2 = self.embedding_client.embed(&text2).await?;

        Ok(self.cosine_similarity(&embedding1, &embedding2))
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

    /// Calculate string similarity using Levenshtein distance
    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f32 {
        let s1_lower = s1.to_lowercase();
        let s2_lower = s2.to_lowercase();

        if s1_lower == s2_lower {
            return 1.0;
        }

        let distance = self.levenshtein_distance(&s1_lower, &s2_lower);
        let max_len = s1_lower.len().max(s2_lower.len());

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f32 / max_len as f32)
        }
    }

    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Merge entities with confidence weighting
    fn merge_entities_weighted(
        &self,
        entity1: &ExtractedEntity,
        entity2: &ExtractedEntity,
        similarity: f32,
    ) -> ExtractedEntity {
        let conf1 = entity1.confidence.unwrap_or(0.5);
        let conf2 = entity2.confidence.unwrap_or(0.5);

        // Choose the entity with higher confidence as base
        let (base, _other) = if conf1 >= conf2 {
            (entity1, entity2)
        } else {
            (entity2, entity1)
        };

        ExtractedEntity {
            name: base.name.clone(),
            entity_type_id: base.entity_type_id,
            confidence: Some((conf1 + conf2) / 2.0 * similarity), // Weighted average
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    // Mock implementations for testing
    struct MockStorage;
    struct MockLLMClient;
    struct MockEmbeddingClient;

    #[async_trait::async_trait]
    impl GraphStorage for MockStorage {
        type Error = Error;

        async fn create_node(&self, _node: &dyn crate::graph::Node) -> Result<()> {
            Ok(())
        }

        async fn get_node(&self, _id: &Uuid) -> Result<Option<Box<dyn crate::graph::Node>>> {
            Ok(None)
        }

        async fn update_node(&self, _node: &dyn crate::graph::Node) -> Result<()> {
            Ok(())
        }

        async fn delete_node(&self, _id: &Uuid) -> Result<()> {
            Ok(())
        }

        async fn create_edge(&self, _edge: &Edge) -> Result<()> {
            Ok(())
        }

        async fn get_edge_by_id(&self, _id: &Uuid) -> Result<Option<Edge>> {
            Ok(None)
        }

        async fn delete_edge_by_id(&self, _id: &Uuid) -> Result<bool> {
            Ok(false)
        }

        async fn get_edges(
            &self,
            _node_id: &Uuid,
            _direction: crate::storage::Direction,
        ) -> Result<Vec<Edge>> {
            Ok(Vec::new())
        }

        async fn get_all_nodes(&self) -> Result<Vec<Box<dyn crate::graph::Node>>> {
            Ok(Vec::new())
        }

        async fn get_all_edges(&self) -> Result<Vec<Edge>> {
            Ok(Vec::new())
        }

        async fn get_nodes_at_time(
            &self,
            _timestamp: chrono::DateTime<chrono::Utc>,
        ) -> Result<Vec<Box<dyn crate::graph::Node>>> {
            Ok(Vec::new())
        }

        async fn get_edges_at_time(
            &self,
            _timestamp: chrono::DateTime<chrono::Utc>,
        ) -> Result<Vec<Edge>> {
            Ok(Vec::new())
        }

        async fn get_node_history(
            &self,
            _node_id: &uuid::Uuid,
        ) -> Result<Vec<Box<dyn crate::graph::Node>>> {
            Ok(Vec::new())
        }

        async fn get_edge_history(&self, _edge_id: &uuid::Uuid) -> Result<Vec<Edge>> {
            Ok(Vec::new())
        }
    }

    impl LLMClient for MockLLMClient {
        fn complete(
            &self,
            _prompt: &str,
        ) -> impl std::future::Future<Output = Result<String>> + Send {
            async { Ok(r#"{"extracted_entities": []}"#.to_string()) }
        }
    }

    impl EmbeddingClient for MockEmbeddingClient {
        fn embed(&self, _text: &str) -> impl std::future::Future<Output = Result<Vec<f32>>> + Send {
            async { Ok(vec![0.1, 0.2, 0.3]) }
        }

        fn embed_batch(
            &self,
            texts: &[String],
        ) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>>> + Send {
            let len = texts.len();
            async move { Ok((0..len).map(|_| vec![0.1, 0.2, 0.3]).collect()) }
        }
    }

    #[tokio::test]
    async fn test_episode_processor_creation() {
        let storage = MockStorage;
        let llm_client = MockLLMClient;
        let embedding_client = MockEmbeddingClient;
        let entity_types = vec![];

        let processor = EpisodeProcessor::new(storage, llm_client, embedding_client, entity_types);

        assert!(processor.is_ok());
    }
}
