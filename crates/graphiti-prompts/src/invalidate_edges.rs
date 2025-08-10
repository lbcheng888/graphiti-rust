//! Edge invalidation functionality

use crate::{
    models::{EdgeInvalidation, InvalidationContext},
    PromptEngine,
};
use anyhow::Result;
use serde_json;
use tracing::{info, warn};

/// Identify edges to invalidate using LLM
pub async fn invalidate_edges(
    prompt_engine: &PromptEngine,
    context: &InvalidationContext,
) -> Result<EdgeInvalidation> {
    info!("Identifying edges to invalidate");

    // Render the prompt template
    let _prompt = prompt_engine.render("invalidate_edges", context)?;

    // TODO: This would be called with actual LLM client
    // For now, return empty result
    warn!("LLM integration not yet implemented - returning no invalidations");

    Ok(EdgeInvalidation {
        invalidated_edges: Vec::new(),
        explanation: "No edges invalidated (LLM not integrated)".to_string(),
    })
}

/// Parse LLM response into structured invalidation result
pub fn parse_invalidation_response(response: &str) -> Result<EdgeInvalidation> {
    let invalidation: EdgeInvalidation = serde_json::from_str(response)?;
    Ok(invalidation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_invalidation_response() {
        let response = r#"
        {
            "invalidated_edges": ["edge_uuid1", "edge_uuid2"],
            "explanation": "These edges are contradicted by new information"
        }
        "#;

        let result = parse_invalidation_response(response).unwrap();
        assert_eq!(result.invalidated_edges.len(), 2);
        assert!(result.explanation.contains("contradicted"));
    }

    #[tokio::test]
    async fn test_invalidate_edges() {
        let prompt_engine = PromptEngine::new().unwrap();
        let context = InvalidationContext {
            new_edges: vec![serde_json::json!({"relationship": "works_at", "target": "New Corp"})],
            existing_edges: vec![
                serde_json::json!({"relationship": "works_at", "target": "Old Corp", "uuid": "edge1"}),
            ],
            episode_context: "Alice changed jobs".to_string(),
        };

        let result = invalidate_edges(&prompt_engine, &context).await;
        assert!(result.is_ok());
    }
}
