//! Entity extraction functionality

use crate::{
    models::{ExtractedEntities, ExtractedEntity, ExtractionContext},
    PromptEngine,
};
use anyhow::Result;
use graphiti_llm::{CompletionParams, LLMClient, Message};
use serde_json;
use std::sync::Arc;
use tracing::{error, info, warn};

/// Extract entities from episode content using LLM
pub async fn extract_entities(
    prompt_engine: &PromptEngine,
    context: &ExtractionContext,
    llm_client: Arc<dyn LLMClient>,
) -> Result<ExtractedEntities> {
    info!("Extracting entities from episode content");

    // Render the prompt template
    let prompt = prompt_engine.render("extract_nodes", context)?;

    // Create messages for LLM
    let messages = vec![Message::user(&prompt)];
    let params = CompletionParams {
        max_tokens: Some(2000),
        temperature: Some(0.1), // Low temperature for consistent extraction
        ..Default::default()
    };

    // Call LLM for entity extraction
    match llm_client.complete(&messages, &params).await {
        Ok(response) => {
            // Parse JSON response
            match parse_entity_response(&response) {
                Ok(entities) => {
                    info!("Successfully extracted {} entities", entities.len());
                    Ok(ExtractedEntities {
                        extracted_entities: entities,
                    })
                }
                Err(e) => {
                    error!("Failed to parse entity extraction response: {}", e);
                    // Fallback to rule-based extraction
                    warn!("Falling back to rule-based entity extraction");
                    Ok(fallback_entity_extraction(&context.episode_content))
                }
            }
        }
        Err(e) => {
            error!("LLM call failed for entity extraction: {}", e);
            // Fallback to rule-based extraction
            warn!("Falling back to rule-based entity extraction");
            Ok(fallback_entity_extraction(&context.episode_content))
        }
    }
}

/// Parse LLM response for entity extraction
fn parse_entity_response(response: &str) -> Result<Vec<ExtractedEntity>> {
    // Try to find JSON in the response
    let json_start = response.find('{').unwrap_or(0);
    let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
    let json_str = &response[json_start..json_end];

    // Parse the JSON
    let parsed: serde_json::Value = serde_json::from_str(json_str)?;

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
                        .unwrap_or(0.8) as f32;

                    entities.push(ExtractedEntity {
                        name: name.to_string(),
                        entity_type_id: entity_type_id as i32,
                        confidence: Some(confidence),
                    });
                }
            }
        }
    }

    Ok(entities)
}

/// Fallback rule-based entity extraction
fn fallback_entity_extraction(text: &str) -> ExtractedEntities {
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

    ExtractedEntities {
        extracted_entities: entities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{EntityType, ExtractionContext};

    #[test]
    fn test_parse_entity_response() {
        let response = r#"
        {
            "extracted_entities": [
                {
                    "name": "Alice",
                    "entity_type_id": 1,
                    "confidence": 0.95
                }
            ]
        }
        "#;

        let result = parse_entity_response(response).unwrap();
        assert_eq!(result.extracted_entities.len(), 1);
        assert_eq!(result.extracted_entities[0].name, "Alice");
    }

    #[tokio::test]
    async fn test_extract_entities() {
        let prompt_engine = PromptEngine::new().unwrap();
        let context = ExtractionContext {
            episode_content: "Alice met Bob at the coffee shop.".to_string(),
            previous_episodes: vec![],
            entity_types: vec![EntityType {
                id: 1,
                name: "Person".to_string(),
                description: "A human being".to_string(),
                examples: vec!["Alice".to_string(), "Bob".to_string()],
            }],
            excluded_types: vec![],
        };

        let result = extract_entities(&prompt_engine, &context).await;
        assert!(result.is_ok());
    }
}
