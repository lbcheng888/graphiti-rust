use crate::{CandleNerExtractor, EntityExtractor, ExtractedEntity, RuleBasedExtractor};
use async_trait::async_trait;
use graphiti_core::error::Result;
use std::collections::HashSet;
use tracing::{debug, info, warn};

#[cfg(feature = "rust-bert-ner")]
use crate::RustBertExtractor;

/// Configuration for the hybrid extractor
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Whether to use rust-bert as the primary extractor
    pub use_rust_bert: bool,
    /// Whether to use Candle-based NER extractor
    pub use_candle_ner: bool,
    /// Whether to use rule-based extractor as fallback
    pub use_rule_fallback: bool,
    /// Minimum confidence threshold for ML-based entities
    pub min_confidence: f32,
    /// Whether to merge overlapping entities
    pub merge_overlapping: bool,
    /// Maximum overlap ratio to consider entities as duplicates (0.0 to 1.0)
    pub max_overlap_ratio: f32,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            use_rust_bert: false, // 默认不使用 rust-bert (需要 PyTorch)
            use_candle_ner: true, // 默认使用 Candle NER (纯 Rust)
            use_rule_fallback: true,
            min_confidence: 0.5,
            merge_overlapping: true,
            max_overlap_ratio: 0.7,
        }
    }
}

/// Hybrid entity extractor that combines rule-based and ML-based approaches
pub struct HybridExtractor {
    #[cfg(feature = "rust-bert-ner")]
    rust_bert_extractor: Option<RustBertExtractor>,
    #[cfg(not(feature = "rust-bert-ner"))]
    _rust_bert_placeholder: Option<()>,
    candle_ner_extractor: Option<CandleNerExtractor>,
    rule_extractor: RuleBasedExtractor,
    config: HybridConfig,
}

impl HybridExtractor {
    /// Create a new hybrid extractor
    pub async fn new(config: HybridConfig) -> Result<Self> {
        info!("Initializing hybrid NER extractor...");

        #[cfg(feature = "rust-bert-ner")]
        let rust_bert_extractor = if config.use_rust_bert {
            match RustBertExtractor::new().await {
                Ok(extractor) => {
                    info!("rust-bert extractor initialized successfully");
                    Some(extractor)
                }
                Err(e) => {
                    warn!("Failed to initialize rust-bert extractor: {}", e);
                    if !config.use_rule_fallback {
                        return Err(e);
                    }
                    None
                }
            }
        } else {
            None
        };

        #[cfg(not(feature = "rust-bert-ner"))]
        let _rust_bert_placeholder = None;

        let rule_extractor = RuleBasedExtractor::default();

        info!("Hybrid NER extractor initialized");

        Ok(Self {
            #[cfg(feature = "rust-bert-ner")]
            rust_bert_extractor,
            #[cfg(not(feature = "rust-bert-ner"))]
            _rust_bert_placeholder,
            candle_ner_extractor: None, // 在 new() 中不初始化 Candle NER
            rule_extractor,
            config,
        })
    }

    /// Create a new hybrid extractor with default configuration
    pub async fn new_default() -> Result<Self> {
        Self::new(HybridConfig::default()).await
    }

    /// Create a new hybrid extractor with Candle NER extractor
    pub async fn new_with_candle(
        config: HybridConfig,
        candle_ner: Option<CandleNerExtractor>,
    ) -> Result<Self> {
        info!("Initializing hybrid NER extractor with Candle support...");

        #[cfg(feature = "rust-bert-ner")]
        let rust_bert_extractor = if config.use_rust_bert {
            match RustBertExtractor::new().await {
                Ok(extractor) => {
                    info!("rust-bert extractor initialized successfully");
                    Some(extractor)
                }
                Err(e) => {
                    warn!("Failed to initialize rust-bert extractor: {}", e);
                    if !config.use_rule_fallback && !config.use_candle_ner {
                        return Err(e);
                    }
                    None
                }
            }
        } else {
            None
        };

        #[cfg(not(feature = "rust-bert-ner"))]
        let _rust_bert_placeholder = None;

        let rule_extractor = RuleBasedExtractor::default();

        info!("Hybrid NER extractor initialized with Candle support");

        Ok(Self {
            #[cfg(feature = "rust-bert-ner")]
            rust_bert_extractor,
            #[cfg(not(feature = "rust-bert-ner"))]
            _rust_bert_placeholder,
            candle_ner_extractor: candle_ner,
            rule_extractor,
            config,
        })
    }

    /// Merge overlapping entities, keeping the one with higher confidence
    fn merge_entities(&self, entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
        if !self.config.merge_overlapping {
            return entities;
        }

        let mut merged = Vec::new();
        let mut used_indices = HashSet::new();

        for (i, entity1) in entities.iter().enumerate() {
            if used_indices.contains(&i) {
                continue;
            }

            let mut best_entity = entity1.clone();
            let mut best_confidence = entity1.score;
            used_indices.insert(i);

            // Check for overlapping entities
            for (j, entity2) in entities.iter().enumerate().skip(i + 1) {
                if used_indices.contains(&j) {
                    continue;
                }

                let overlap = self.calculate_overlap(entity1, entity2);
                if overlap > self.config.max_overlap_ratio {
                    // Entities overlap significantly, keep the one with higher confidence
                    if entity2.score > best_confidence {
                        best_entity = entity2.clone();
                        best_confidence = entity2.score;
                    }
                    used_indices.insert(j);
                }
            }

            merged.push(best_entity);
        }

        merged
    }

    /// Calculate overlap ratio between two entities
    fn calculate_overlap(&self, entity1: &ExtractedEntity, entity2: &ExtractedEntity) -> f32 {
        let start1 = entity1.start;
        let end1 = entity1.end;
        let start2 = entity2.start;
        let end2 = entity2.end;

        let overlap_start = start1.max(start2);
        let overlap_end = end1.min(end2);

        if overlap_start >= overlap_end {
            return 0.0; // No overlap
        }

        let overlap_length = overlap_end - overlap_start;
        let entity1_length = end1 - start1;
        let entity2_length = end2 - start2;
        let min_length = entity1_length.min(entity2_length);

        if min_length == 0 {
            return 0.0;
        }

        overlap_length as f32 / min_length as f32
    }

    /// Filter entities by confidence threshold
    fn filter_by_confidence(&self, entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
        entities
            .into_iter()
            .filter(|entity| entity.score >= self.config.min_confidence)
            .collect()
    }
}

#[async_trait]
impl EntityExtractor for HybridExtractor {
    async fn extract(&self, text: &str) -> Result<Vec<ExtractedEntity>> {
        debug!("Extracting entities using hybrid approach");

        let mut all_entities = Vec::new();

        // Try rust-bert first if available
        #[cfg(feature = "rust-bert-ner")]
        if let Some(ref rust_bert) = self.rust_bert_extractor {
            match rust_bert.extract(text).await {
                Ok(entities) => {
                    debug!("rust-bert found {} entities", entities.len());
                    let filtered = self.filter_by_confidence(entities);
                    all_entities.extend(filtered);
                }
                Err(e) => {
                    warn!("rust-bert extraction failed: {}", e);
                }
            }
        }

        // Try Candle NER if available and no rust-bert results
        if self.config.use_candle_ner && (all_entities.is_empty() || !self.config.use_rust_bert) {
            if let Some(ref candle_ner) = self.candle_ner_extractor {
                match candle_ner.extract(text).await {
                    Ok(entities) => {
                        debug!("Candle NER found {} entities", entities.len());
                        let filtered = self.filter_by_confidence(entities);
                        all_entities.extend(filtered);
                    }
                    Err(e) => {
                        warn!("Candle NER extraction failed: {}", e);
                    }
                }
            }
        }

        // Use rule-based extractor as fallback or supplement
        if self.config.use_rule_fallback
            && (all_entities.is_empty()
                || (!self.config.use_rust_bert && !self.config.use_candle_ner))
        {
            match self.rule_extractor.extract(text).await {
                Ok(rule_entities) => {
                    debug!(
                        "Rule-based extractor found {} entities",
                        rule_entities.len()
                    );
                    all_entities.extend(rule_entities);
                }
                Err(e) => {
                    warn!("Rule-based extraction failed: {}", e);
                }
            }
        }

        // Merge overlapping entities
        let merged_entities = self.merge_entities(all_entities);

        debug!(
            "Final result: {} entities after merging",
            merged_entities.len()
        );
        Ok(merged_entities)
    }
}

impl Default for HybridExtractor {
    fn default() -> Self {
        // Note: This will not initialize rust-bert properly
        // Use HybridExtractor::new_default() for proper async initialization
        Self {
            #[cfg(feature = "rust-bert-ner")]
            rust_bert_extractor: None,
            #[cfg(not(feature = "rust-bert-ner"))]
            _rust_bert_placeholder: None,
            candle_ner_extractor: None,
            rule_extractor: RuleBasedExtractor::default(),
            config: HybridConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityLabel;

    #[tokio::test]
    async fn test_hybrid_extractor() {
        let config = HybridConfig {
            use_rust_bert: true,
            use_candle_ner: false,
            use_rule_fallback: true,
            min_confidence: 0.3,
            merge_overlapping: true,
            max_overlap_ratio: 0.7,
        };

        let extractor = HybridExtractor::new(config).await.unwrap();

        let text = "My name is Amy. I live in Paris. I work at Microsoft.";
        let entities = extractor.extract(text).await.unwrap();

        // Should find some entities (or at least run without error)
        // Note: With rust-bert disabled and no Candle NER, this might be empty
        println!("Found {} entities", entities.len());

        // Print entities for debugging
        for entity in &entities {
            println!(
                "Entity: {} ({:?}) - score: {:.3}",
                entity.text, entity.label, entity.score
            );
        }
    }

    #[tokio::test]
    async fn test_hybrid_extractor_fallback() {
        let config = HybridConfig {
            use_rust_bert: false, // Disable rust-bert to test fallback
            use_candle_ner: false,
            use_rule_fallback: true,
            min_confidence: 0.0,
            merge_overlapping: true,
            max_overlap_ratio: 0.7,
        };

        let extractor = HybridExtractor::new(config).await.unwrap();

        let text = "张三在北京工作。";
        let entities = extractor.extract(text).await.unwrap();

        // Should find entities using rule-based approach
        println!("Fallback entities: {:?}", entities);
    }

    #[test]
    fn test_overlap_calculation() {
        let extractor = HybridExtractor::default();

        let entity1 = ExtractedEntity {
            text: "Amy".to_string(),
            label: EntityLabel::Person,
            start: 0,
            end: 3,
            score: 0.9,
        };

        let entity2 = ExtractedEntity {
            text: "Amy Smith".to_string(),
            label: EntityLabel::Person,
            start: 0,
            end: 9,
            score: 0.8,
        };

        let overlap = extractor.calculate_overlap(&entity1, &entity2);
        assert_eq!(overlap, 1.0); // entity1 is completely contained in entity2
    }
}
