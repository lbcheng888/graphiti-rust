use crate::{EntityExtractor, EntityLabel, ExtractedEntity};
use async_trait::async_trait;
use graphiti_core::error::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

#[cfg(feature = "rust-bert-ner")]
use rust_bert::pipelines::ner::{Entity, NERModel};

/// rust-bert based entity extractor
/// Provides state-of-the-art Named Entity Recognition using BERT models
pub struct RustBertExtractor {
    #[cfg(feature = "rust-bert-ner")]
    model: Arc<Mutex<NERModel>>,
    #[cfg(not(feature = "rust-bert-ner"))]
    _phantom: std::marker::PhantomData<()>,
}

impl RustBertExtractor {
    /// Create a new rust-bert NER extractor
    pub async fn new() -> Result<Self> {
        #[cfg(feature = "rust-bert-ner")]
        {
            info!("Initializing rust-bert NER model...");

            // Initialize the NER model with default configuration
            // This will download the model if not already cached
            let model = tokio::task::spawn_blocking(|| NERModel::new(Default::default()))
                .await
                .map_err(|e| graphiti_core::error::Error::Other(anyhow::Error::new(e)))?
                .map_err(|e| graphiti_core::error::Error::Other(anyhow::Error::new(e)))?;

            info!("rust-bert NER model initialized successfully");

            Ok(Self {
                model: Arc::new(Mutex::new(model)),
            })
        }

        #[cfg(not(feature = "rust-bert-ner"))]
        {
            Err(graphiti_core::error::Error::Other(anyhow::anyhow!(
                "rust-bert feature not enabled"
            )))
        }
    }

    /// Convert rust-bert entity to our internal format
    #[cfg(feature = "rust-bert-ner")]
    fn convert_entity(entity: &Entity, _text: &str) -> ExtractedEntity {
        let label = match entity.label.as_str() {
            "I-PER" | "B-PER" => EntityLabel::Person,
            "I-ORG" | "B-ORG" => EntityLabel::Organization,
            "I-LOC" | "B-LOC" => EntityLabel::Location,
            "I-MISC" | "B-MISC" => EntityLabel::Miscellaneous,
            _ => EntityLabel::Other,
        };

        ExtractedEntity {
            text: entity.word.clone(),
            label,
            score: entity.score as f32,
            start: entity.offset.begin as usize,
            end: entity.offset.end as usize,
        }
    }
}

#[async_trait]
impl EntityExtractor for RustBertExtractor {
    async fn extract(&self, text: &str) -> Result<Vec<ExtractedEntity>> {
        #[cfg(feature = "rust-bert-ner")]
        {
            debug!("Extracting entities from text using rust-bert");

            let model = self.model.clone();
            let text_copy = text.to_string();

            let entities = tokio::task::spawn_blocking(move || {
                let model = model.blocking_lock();
                model.predict(&[text_copy.as_str()])
            })
            .await
            .map_err(|e| graphiti_core::error::Error::Other(anyhow::Error::new(e)))?;

            let mut extracted_entities = Vec::new();

            if let Some(sentence_entities) = entities.first() {
                for entity in sentence_entities {
                    let extracted = Self::convert_entity(entity, text);
                    extracted_entities.push(extracted);
                }
            }

            debug!("Extracted {} entities", extracted_entities.len());
            Ok(extracted_entities)
        }

        #[cfg(not(feature = "rust-bert-ner"))]
        {
            debug!("rust-bert feature not enabled, returning empty results");
            Ok(Vec::new())
        }
    }
}

impl Default for RustBertExtractor {
    fn default() -> Self {
        #[cfg(feature = "rust-bert-ner")]
        {
            // Note: This will panic if called from a non-async context
            // Use RustBertExtractor::new() instead for proper async initialization
            panic!("Use RustBertExtractor::new() for async initialization")
        }

        #[cfg(not(feature = "rust-bert-ner"))]
        {
            Self {
                _phantom: std::marker::PhantomData,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[cfg(feature = "rust-bert-ner")]
    async fn test_rust_bert_extractor() {
        let extractor = RustBertExtractor::new().await.unwrap();

        let text = "My name is Amy. I live in Paris.";
        let entities = extractor.extract(text).await.unwrap();

        // Should find some entities
        println!("Found {} entities", entities.len());
        for entity in &entities {
            println!(
                "Entity: {} ({:?}) - score: {:.3}",
                entity.text, entity.label, entity.score
            );
        }
    }
}
