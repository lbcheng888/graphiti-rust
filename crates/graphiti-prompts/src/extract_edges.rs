//! Edge/relationship extraction functionality

use crate::{
    models::{EdgeExtractionContext, ExtractedEdge, ExtractedEdges, ExtractedEntity},
    PromptEngine,
};
use anyhow::Result;
use chrono::Utc;
use graphiti_llm::{CompletionParams, LLMClient, Message};
use serde_json;
use std::sync::Arc;
use tracing::{error, info, warn};

/// Extract relationships from episode content using LLM
pub async fn extract_edges(
    prompt_engine: &PromptEngine,
    context: &EdgeExtractionContext,
    llm_client: Arc<dyn LLMClient>,
) -> Result<ExtractedEdges> {
    info!("Extracting relationships from episode content");

    // Render the prompt template
    let prompt = prompt_engine.render("extract_edges", context)?;

    // Create messages for LLM
    let messages = vec![Message::user(&prompt)];
    let params = CompletionParams {
        max_tokens: Some(2000),
        temperature: Some(0.1), // Low temperature for consistent extraction
        ..Default::default()
    };

    // Call LLM for relationship extraction
    match llm_client.complete(&messages, &params).await {
        Ok(response) => {
            // Parse JSON response
            match parse_edge_response(&response) {
                Ok(edges) => {
                    info!("Successfully extracted {} relationships", edges.len());
                    Ok(ExtractedEdges {
                        extracted_edges: edges,
                    })
                }
                Err(e) => {
                    error!("Failed to parse relationship extraction response: {}", e);
                    // Fallback to rule-based extraction
                    warn!("Falling back to rule-based relationship extraction");
                    Ok(fallback_edge_extraction(
                        &context.entities,
                        &context.episode_content,
                    ))
                }
            }
        }
        Err(e) => {
            error!("LLM call failed for relationship extraction: {}", e);
            // Fallback to rule-based extraction
            warn!("Falling back to rule-based relationship extraction");
            Ok(fallback_edge_extraction(
                &context.entities,
                &context.episode_content,
            ))
        }
    }
}

/// Parse LLM response for relationship extraction
fn parse_edge_response(response: &str) -> Result<Vec<ExtractedEdge>> {
    // Try to find JSON in the response
    let json_start = response.find('{').unwrap_or(0);
    let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
    let json_str = &response[json_start..json_end];

    // Parse the JSON
    let parsed: serde_json::Value = serde_json::from_str(json_str)?;

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

/// Fallback rule-based relationship extraction
fn fallback_edge_extraction(entities: &[ExtractedEntity], text: &str) -> ExtractedEdges {
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
            } else if text.to_lowercase().contains("friend") || text.to_lowercase().contains("know")
            {
                "KNOWS"
            } else if text.to_lowercase().contains("manage") || text.to_lowercase().contains("lead")
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

    ExtractedEdges {
        extracted_edges: edges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ExtractedEntity;

    #[test]
    fn test_parse_edge_response() {
        let response = r#"
        {
            "extracted_edges": [
                {
                    "source_entity": "Alice",
                    "target_entity": "Bob",
                    "relationship": "knows",
                    "description": "Alice knows Bob",
                    "confidence": 0.9,
                    "valid_at": null
                }
            ]
        }
        "#;

        let result = parse_edge_response(response).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].source_entity, "Alice");
    }

    #[tokio::test]
    async fn test_fallback_edge_extraction() {
        let entities = vec![
            ExtractedEntity {
                name: "Alice".to_string(),
                entity_type_id: 1,
                confidence: Some(0.95),
            },
            ExtractedEntity {
                name: "Bob".to_string(),
                entity_type_id: 1,
                confidence: Some(0.95),
            },
        ];

        let text = "Alice met Bob at the coffee shop.";
        let result = fallback_edge_extraction(&entities, text);

        assert!(!result.extracted_edges.is_empty());
        assert_eq!(result.extracted_edges[0].source_entity, "Alice");
        assert_eq!(result.extracted_edges[0].target_entity, "Bob");
    }
}
