//! Graphiti service implementation

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use uuid::Uuid;
use tracing::{info, warn};

use graphiti_core::code_entities::*;
use graphiti_core::graph::*;
use graphiti_core::error::Result as GraphitiResult;
use graphiti_core::graphiti::GraphitiConfig;
use graphiti_llm::*;
use graphiti_cozo::CozoDriver;
use graphiti_core::storage::GraphStorage;
use std::collections::HashMap as StdHashMap;

use crate::types::*;

/// Real implementation using CozoDB with unified intelligence
pub struct RealGraphitiService {
    storage: Arc<CozoDriver>,
    #[allow(dead_code)]
    embedder: Arc<dyn EmbeddingClient>,
    ner_extractor: Arc<dyn graphiti_ner::EntityExtractor>,
    #[allow(dead_code)]
    config: GraphitiConfig,
    /// Simple in-memory index for episodes to support search/get operations
    memory_index: Arc<RwLock<HashMap<Uuid, EpisodeNode>>>,
    /// In-memory mapping from episode id to extracted entity names
    memory_entities: Arc<RwLock<HashMap<Uuid, Vec<String>>>>,
    /// In-memory collection of extracted relationships records
    memory_relationships: Arc<RwLock<Vec<SimpleExtractedRelationship>>>,
    /// In-memory mapping from synthetic UUID to relationship record (minimal parity)
    memory_relationships_by_id: Arc<RwLock<HashMap<Uuid, SimpleExtractedRelationship>>>,
}

impl RealGraphitiService {
    /// Create a new RealGraphitiService instance
    pub async fn new(
        storage: Arc<CozoDriver>,
        embedder_config: EmbedderConfig,
        config: GraphitiConfig,
    ) -> anyhow::Result<Self> {
        // Create embedder client based on provider
        let embedder: Arc<dyn EmbeddingClient> = match embedder_config.provider {
            EmbeddingProvider::EmbedAnything => {
                let cfg = graphiti_llm::EmbedAnythingConfig {
                    model_id: embedder_config.model.clone(),
                    batch_size: embedder_config.batch_size.max(16),
                    max_length: embedder_config.max_length.unwrap_or(8192),
                    device: embedder_config
                        .device
                        .clone()
                        .unwrap_or_else(|| "auto".to_string()),
                    cache_dir: embedder_config
                        .cache_dir
                        .clone()
                        .or_else(|| std::env::var("EMBEDDING_MODEL_DIR").ok()),
                    target_dim: Some(embedder_config.dimension),
                };
                match graphiti_llm::EmbedAnythingClient::new(cfg).await {
                    Ok(client) => Arc::new(client),
                    Err(e) => {
                        warn!("EmbedAnything initialization failed: {}. Using disabled placeholder embedder.", e);
                        struct DisabledEmbedder { dim: usize }
                        #[async_trait::async_trait]
                        impl graphiti_llm::EmbeddingClient for DisabledEmbedder {
                            async fn embed_batch(&self, texts: &[String]) -> graphiti_core::error::Result<Vec<Vec<f32>>> {
                                Ok(texts.iter().map(|_| vec![0.0; self.dim]).collect())
                            }
                        }
                        Arc::new(DisabledEmbedder { dim: embedder_config.dimension.max(128) })
                    }
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported embedder provider for server: use 'embed_anything'"
                ));
            }
        };

        // Create NER extractor
        let ner_extractor: Arc<dyn graphiti_ner::EntityExtractor> = {
            // Adapter to pass Arc<dyn EmbeddingClient> where Box<dyn EmbeddingClient> is expected
            struct ArcEmbedder(std::sync::Arc<dyn EmbeddingClient>);
            #[async_trait::async_trait]
            impl EmbeddingClient for ArcEmbedder {
                async fn embed_batch(&self, texts: &[String]) -> graphiti_core::error::Result<Vec<Vec<f32>>> {
                    self.0.embed_batch(texts).await
                }
            }

            let candle_ner =
                graphiti_ner::CandleNerExtractor::new(Box::new(ArcEmbedder(embedder.clone())))
                    .with_similarity_threshold(0.6);

            let config = graphiti_ner::HybridConfig {
                use_rust_bert: false,
                use_candle_ner: true,
                use_rule_fallback: true,
                min_confidence: 0.5,
                merge_overlapping: true,
                max_overlap_ratio: 0.7,
            };

            match graphiti_ner::HybridExtractor::new_with_candle(config, Some(candle_ner)).await {
                Ok(hybrid) => Arc::new(hybrid),
                Err(e) => {
                    warn!("Failed to initialize Candle NER extractor: {}, falling back to rule-based", e);
                    Arc::new(graphiti_ner::RuleBasedExtractor::default())
                }
            }
        };

        Ok(Self {
            storage,
            embedder,
            ner_extractor,
            config,
            memory_index: Arc::new(RwLock::new(HashMap::new())),
            memory_entities: Arc::new(RwLock::new(HashMap::new())),
            memory_relationships: Arc::new(RwLock::new(Vec::new())),
            memory_relationships_by_id: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Extract entities using NER
    async fn extract_with_ner(
        &self,
        text: &str,
    ) -> GraphitiResult<(
        Vec<ExtractedEntity>,
        Vec<ExtractedRelationship>,
        Option<String>,
    )> {
        let ner_entities = self.ner_extractor.extract(text).await?;

        // Convert NER entities to LLM-style entities
        let entities: Vec<ExtractedEntity> = ner_entities
            .into_iter()
            .map(|e| ExtractedEntity {
                name: e.text,
                entity_type: e.label.as_str().to_string(),
                confidence: e.score,
                attributes: {
                    let mut attrs = std::collections::HashMap::new();
                    attrs.insert("start".to_string(), serde_json::json!(e.start));
                    attrs.insert("end".to_string(), serde_json::json!(e.end));
                    attrs.insert(
                        "extraction_method".to_string(),
                        serde_json::json!("rule_based"),
                    );
                    attrs
                },
                span: Some((e.start, e.end)),
            })
            .collect();

        // Simple relationship inference based on proximity
        let mut relationships = Vec::new();
        let sentences: Vec<&str> = text.split(&['.', '!', '?', '。', '！', '？'][..]).collect();

        for sentence in sentences {
            let sentence_entities: Vec<_> = entities
                .iter()
                .filter(|e| {
                    e.span
                        .map(|(start, _)| text[..start].contains(sentence))
                        .unwrap_or(false)
                })
                .collect();

            // Create relationships between entities in the same sentence
            for i in 0..sentence_entities.len() {
                for j in i + 1..sentence_entities.len() {
                    relationships.push(ExtractedRelationship {
                        source: sentence_entities[i].name.clone(),
                        target: sentence_entities[j].name.clone(),
                        relationship: "RELATED_TO".to_string(),
                        confidence: 0.6,
                        attributes: {
                            let mut attrs = std::collections::HashMap::new();
                            attrs.insert("inferred".to_string(), serde_json::json!(true));
                            attrs.insert("method".to_string(), serde_json::json!("proximity"));
                            attrs
                        },
                    });
                }
            }
        }

        // Simple summary generation
        let summary = if entities.is_empty() {
            None
        } else {
            Some(format!(
                "Text containing {} entities: {}",
                entities.len(),
                entities
                    .iter()
                    .take(3)
                    .map(|e| e.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        };

        Ok((entities, relationships, summary))
    }
}

#[async_trait::async_trait]
impl GraphitiService for RealGraphitiService {
    async fn add_memory(&self, req: AddMemoryRequest) -> GraphitiResult<AddMemoryResponse> {
        info!("Adding memory: {:?}", req.name.as_deref().unwrap_or("unnamed"));

        // Create episode using the storage layer directly
        let episode_id = Uuid::new_v4();
        let now = chrono::Utc::now();
        let event_time = if let Some(ts) = &req.timestamp {
            match chrono::DateTime::parse_from_rfc3339(ts) {
                Ok(dt) => dt.with_timezone(&chrono::Utc),
                Err(_) => now,
            }
        } else {
            now
        };

        // Create episode node
        let episode = EpisodeNode {
            id: episode_id,
            name: req.name.unwrap_or_else(|| "Memory".to_string()),
            episode_type: EpisodeType::Message,
            content: req.content.clone(),
            source: req.source.unwrap_or_else(|| "user".to_string()),
            temporal: TemporalMetadata {
                created_at: now,
                valid_from: event_time,
                valid_to: None,
                expired_at: None,
            },
            embedding: None,
        };

        // Store the episode
        self.storage.create_node(&episode).await?;
        info!("Successfully added memory with episode ID: {}", episode_id);

        // Update in-memory index
        {
            let mut idx = self.memory_index.write().await;
            idx.insert(episode_id, episode.clone());
        }

        // Perform lightweight entity and relationship extraction
        let (entities, relationships) = match self.extract_with_ner(&req.content).await {
            Ok((entities, relationships, _summary)) => (entities, relationships),
            Err(e) => {
                tracing::warn!("NER extraction failed (continuing without entities): {}", e);
                (Vec::new(), Vec::new())
            }
        };

        let (simple_entities, entity_names): (Vec<SimpleExtractedEntity>, Vec<String>) = {
            let mut names = Vec::new();
            let simple = entities
                .iter()
                .map(|e| {
                    names.push(e.name.clone());
                    SimpleExtractedEntity {
                        name: e.name.clone(),
                        entity_type: e.entity_type.clone(),
                        confidence: e.confidence,
                    }
                })
                .collect();
            (simple, names)
        };

        let simple_relationships: Vec<SimpleExtractedRelationship> = relationships
            .into_iter()
            .map(|r| SimpleExtractedRelationship {
                source: r.source,
                target: r.target,
                relationship: r.relationship,
                confidence: r.confidence,
            })
            .collect();

        // Update entity/relationship in-memory indexes
        {
            let mut ents = self.memory_entities.write().await;
            ents.insert(episode_id, entity_names);
        }
        {
            let mut rels = self.memory_relationships.write().await;
            rels.extend(simple_relationships.iter().cloned());
        }
        {
            let mut rel_map = self.memory_relationships_by_id.write().await;
            for r in &simple_relationships {
                let id = Uuid::new_v4();
                rel_map.insert(id, r.clone());
            }
        }

        Ok(AddMemoryResponse {
            id: episode_id,
            entities: simple_entities,
            relationships: simple_relationships,
        })
    }

    async fn search_memory(
        &self,
        req: SearchMemoryRequest,
    ) -> GraphitiResult<SearchMemoryResponse> {
        info!("Searching memory: {}", req.query);

        let limit = req.limit.unwrap_or(10) as usize;
        let query_lower = req.query.to_lowercase();

        let index_snapshot: Vec<EpisodeNode> = {
            let idx = self.memory_index.read().await;
            idx.values().cloned().collect()
        };

        // Naive substring search with simple scoring
        let mut scored: Vec<(f32, &EpisodeNode)> = index_snapshot
            .iter()
            .filter_map(|ep| {
                let content_lower = ep.content.to_lowercase();
                if content_lower.contains(&query_lower) {
                    let score = (req.query.len() as f32) / (ep.content.len().max(1) as f32);
                    Some((score, ep))
                } else if ep.name.to_lowercase().contains(&query_lower) {
                    Some((0.5, ep))
                } else {
                    None
                }
            })
            .collect();

        // Sort by score desc, then created_at desc, then id asc
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.temporal.created_at.cmp(&a.1.temporal.created_at))
                .then_with(|| a.1.id.cmp(&b.1.id))
        });

        let total = scored.len();

        let results: Vec<crate::types::SearchResult> = scored
            .into_iter()
            .take(limit)
            .map(|(score, ep)| crate::types::SearchResult {
                id: ep.id,
                node_type: "Episode".to_string(),
                name: ep.name.clone(),
                content_preview: Some(ep.content.chars().take(200).collect()),
                score,
                timestamp: ep.temporal.valid_from.to_rfc3339(),
            })
            .collect();

        info!("Search completed, found {} results ({} returned)", total, results.len());

        Ok(crate::types::SearchMemoryResponse { results, total })
    }

    async fn get_memory(&self, id: Uuid) -> GraphitiResult<Option<MemoryNode>> {
        info!("Getting memory: {}", id);

        // Try in-memory index first
        if let Some(ep) = self.memory_index.read().await.get(&id).cloned() {
            let node = MemoryNode {
                id: ep.id,
                node_type: "Episode".to_string(),
                name: ep.name,
                content: Some(ep.content),
                created_at: ep.temporal.created_at.to_rfc3339(),
                event_time: ep.temporal.valid_from.to_rfc3339(),
                properties: serde_json::json!({
                    "source": ep.source,
                }),
            };
            return Ok(Some(node));
        }

        Ok(None)
    }

    async fn get_related(&self, id: Uuid, depth: usize) -> GraphitiResult<Vec<RelatedMemory>> {
        info!("Getting related memories for {}, depth: {}", id, depth);

        // Naive relatedness: episodes sharing at least one extracted entity
        let entities_map = self.memory_entities.read().await;
        let Some(target_entities) = entities_map.get(&id) else {
            return Ok(vec![]);
        };
        let target_set: std::collections::HashSet<&String> = target_entities.iter().collect();

        let index = self.memory_index.read().await;
        let mut related: Vec<RelatedMemory> = Vec::new();
        for (other_id, ep) in index.iter() {
            if other_id == &id {
                continue;
            }
            if let Some(ents) = entities_map.get(other_id) {
                if ents.iter().any(|e| target_set.contains(e)) {
                    let node = MemoryNode {
                        id: *other_id,
                        node_type: "Episode".to_string(),
                        name: ep.name.clone(),
                        content: Some(ep.content.clone()),
                        created_at: ep.temporal.created_at.to_rfc3339(),
                        event_time: ep.temporal.valid_from.to_rfc3339(),
                        properties: serde_json::json!({ "source": ep.source }),
                    };
                    related.push(RelatedMemory {
                        node,
                        relationship: "SHARED_ENTITY".to_string(),
                        distance: 1,
                    });
                }
            }
        }

        Ok(related)
    }

    // Other methods would be implemented here...
    async fn search_memory_facts(
        &self,
        _query: String,
        _limit: Option<usize>,
    ) -> GraphitiResult<Vec<SimpleExtractedRelationship>> {
        // Simplified implementation
        Ok(vec![])
    }

    async fn search_memory_facts_json(
        &self,
        _query: String,
        _limit: Option<usize>,
    ) -> GraphitiResult<Vec<serde_json::Value>> {
        Ok(vec![])
    }

    async fn delete_episode(&self, _id: Uuid) -> GraphitiResult<bool> {
        Ok(true)
    }

    async fn get_episodes(&self, _last_n: usize) -> GraphitiResult<Vec<EpisodeNode>> {
        Ok(vec![])
    }

    async fn clear_graph(&self) -> GraphitiResult<()> {
        Ok(())
    }

    async fn get_entity_edge_json(&self, _id: Uuid) -> GraphitiResult<Option<serde_json::Value>> {
        Ok(None)
    }

    async fn delete_entity_edge_by_uuid(&self, _id: Uuid) -> GraphitiResult<bool> {
        Ok(true)
    }

    async fn add_code_entity(
        &self,
        req: AddCodeEntityRequest,
    ) -> GraphitiResult<AddCodeEntityResponse> {
        let now = chrono::Utc::now();
        // Map string entity_type -> CodeEntityType
        let cet = match req.entity_type.to_lowercase().as_str() {
            "function" => CodeEntityType::Function,
            "class" => CodeEntityType::Class,
            "module" => CodeEntityType::Module,
            "api" => CodeEntityType::Api,
            _ => CodeEntityType::Function,
        };

        let entity_id = Uuid::new_v4();
        let mut props = serde_json::json!({
            "code_entity": true,
            "name": req.name,
            "entity_type": cet.to_string(),
            "description": req.description,
        });
        if let Some(fp) = &req.file_path { props["file_path"] = serde_json::json!(fp); }
        if let Some((s,e)) = req.line_range { props["line_range"] = serde_json::json!([s,e]); }
        if let Some(lang) = &req.language { props["language"] = serde_json::json!(lang); }
        if let Some(fw) = &req.framework { props["framework"] = serde_json::json!(fw); }
        if let Some(c) = req.complexity { props["complexity"] = serde_json::json!(c); }
        if let Some(i) = req.importance { props["importance"] = serde_json::json!(i); }

        let node = EntityNode {
            id: entity_id,
            name: format!("{}", props["name"].as_str().unwrap_or("entity")),
            entity_type: cet.to_string(),
            labels: vec!["CodeEntity".into()],
            properties: props,
            temporal: TemporalMetadata { created_at: now, valid_from: now, valid_to: None, expired_at: None },
            embedding: None,
        };
        self.storage.create_node(&node).await?;
        Ok(AddCodeEntityResponse { id: entity_id, message: "ok".into() })
    }

    async fn record_activity(
        &self,
        req: RecordActivityRequest,
    ) -> GraphitiResult<RecordActivityResponse> {
        // Persist as an Episode node for traceability
        let id = Uuid::new_v4();
        let now = chrono::Utc::now();
        let content = format!(
            "Activity [{}] {}\n{}\nDeveloper: {} | Project: {}",
            req.activity_type,
            req.title,
            req.description,
            req.developer,
            req.project
        );
        let ep = EpisodeNode {
            id,
            name: format!("{}: {}", req.activity_type, req.title),
            episode_type: EpisodeType::Event,
            content,
            source: "activity".into(),
            temporal: TemporalMetadata { created_at: now, valid_from: now, valid_to: None, expired_at: None },
            embedding: None,
        };
        self.storage.create_node(&ep).await?;
        Ok(RecordActivityResponse { id, message: "ok".into() })
    }

    async fn search_code(&self, req: SearchCodeRequest) -> GraphitiResult<SearchCodeResponse> {
        // Naive search over stored nodes labeled CodeEntity
        let nodes = self.storage.get_all_nodes().await?;
        let mut out: Vec<graphiti_core::code_entities::CodeEntity> = Vec::new();
        for n in nodes {
            let props = n.properties();
            if !props.get("code_entity").and_then(|v| v.as_bool()).unwrap_or(false) { continue; }
            if let Some(q) = &req.query.strip_prefix("") { let _ = q; }
            // Match query over name/description
            let name = props.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let desc = props.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let mut ok = true;
            if !req.query.is_empty() {
                let ql = req.query.to_lowercase();
                ok = name.to_lowercase().contains(&ql) || desc.to_lowercase().contains(&ql);
            }
            if ok {
                let id = *n.id();
                let file_path = props.get("file_path").and_then(|v| v.as_str()).map(|s| s.to_string());
                let line_range = props.get("line_range").and_then(|v| v.as_array()).and_then(|a|
                    if a.len()==2 { Some((a[0].as_u64().unwrap_or(1) as u32, a[1].as_u64().unwrap_or(1) as u32)) } else { None }
                );
                let language = props.get("language").and_then(|v| v.as_str()).map(|s| s.to_string());
                let framework = props.get("framework").and_then(|v| v.as_str()).map(|s| s.to_string());
                let complexity = props.get("complexity").and_then(|v| v.as_u64()).map(|u| u as u8);
                let importance = props.get("importance").and_then(|v| v.as_u64()).map(|u| u as u8);
                let entity_type_str = props.get("entity_type").and_then(|v| v.as_str()).unwrap_or("Function");
                let entity_type = match entity_type_str.to_lowercase().as_str() {
                    "class" => CodeEntityType::Class,
                    "module" => CodeEntityType::Module,
                    "api" => CodeEntityType::Api,
                    _ => CodeEntityType::Function,
                };
                let ce = graphiti_core::code_entities::CodeEntity {
                    id,
                    entity_type,
                    name,
                    description: desc,
                    file_path,
                    line_range,
                    language,
                    framework,
                    complexity,
                    importance,
                    created_at: n.temporal().created_at,
                    updated_at: n.temporal().valid_from,
                    metadata: StdHashMap::new(),
                };
                // Filters
                if let Some(et) = &req.entity_type { if et.to_lowercase()!=entity_type_str.to_lowercase() { continue; } }
                if let Some(langf) = &req.language { if ce.language.as_deref().unwrap_or("").to_lowercase()!=langf.to_lowercase() { continue; } }
                if let Some(fw) = &req.framework { if ce.framework.as_deref().unwrap_or("").to_lowercase()!=fw.to_lowercase() { continue; } }
                out.push(ce);
            }
        }
        let total = out.len();
        if let Some(l) = req.limit { out.truncate(l as usize); }
        Ok(SearchCodeResponse { results: out, total })
    }

    async fn batch_add_code_entities(
        &self,
        req: BatchAddCodeEntitiesRequest,
    ) -> GraphitiResult<BatchAddCodeEntitiesResponse> {
        let mut results = Vec::new();
        let mut errors = Vec::new();
        let mut ok = 0usize;
        for e in req.entities {
            match self.add_code_entity(e).await {
                Ok(r) => { results.push(r); ok+=1; }
                Err(e) => errors.push(format!("{}", e)),
            }
        }
        Ok(BatchAddCodeEntitiesResponse { results, successful_count: ok, failed_count: errors.len(), errors })
    }

    async fn batch_record_activities(
        &self,
        req: BatchRecordActivitiesRequest,
    ) -> GraphitiResult<BatchRecordActivitiesResponse> {
        let mut results = Vec::new();
        let mut errors = Vec::new();
        let mut ok = 0usize;
        for a in req.activities {
            match self.record_activity(a).await {
                Ok(r) => { results.push(r); ok+=1; }
                Err(e) => errors.push(format!("{}", e)),
            }
        }
        Ok(BatchRecordActivitiesResponse { results, successful_count: ok, failed_count: errors.len(), errors })
    }

    async fn get_context_suggestions(
        &self,
        _req: ContextSuggestionRequest,
    ) -> GraphitiResult<ContextSuggestionResponse> {
        Ok(ContextSuggestionResponse {
            suggestions: vec![],
            total: 0,
        })
    }
}
