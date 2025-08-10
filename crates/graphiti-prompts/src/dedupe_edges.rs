//! Edge deduplication functionality

use crate::{
    models::{DeduplicationContext, EdgeDeduplication},
    PromptEngine,
};
use anyhow::Result;
use serde_json;
use tracing::{info, warn};

/// Deduplicate edges using LLM
pub async fn deduplicate_edges(
    prompt_engine: &PromptEngine,
    context: &DeduplicationContext,
) -> Result<EdgeDeduplication> {
    info!("Deduplicating edges");

    // Render the prompt template
    let _prompt = prompt_engine.render("dedupe_edges", context)?;

    // TODO: This would be called with actual LLM client
    // For now, return empty result
    warn!("LLM integration not yet implemented - returning no duplicates");

    Ok(EdgeDeduplication {
        duplicate_groups: Vec::new(),
        explanation: "No duplicates found (LLM not integrated)".to_string(),
    })
}

/// Parse LLM response into structured deduplication result
pub fn parse_edge_deduplication_response(response: &str) -> Result<EdgeDeduplication> {
    let deduplication: EdgeDeduplication = serde_json::from_str(response)?;
    Ok(deduplication)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_edge_deduplication_response() {
        let response = r#"
        {
            "duplicate_groups": [
                ["edge_uuid1", "edge_uuid2"]
            ],
            "explanation": "Found duplicate relationships"
        }
        "#;

        let result = parse_edge_deduplication_response(response).unwrap();
        assert_eq!(result.duplicate_groups.len(), 1);
        assert_eq!(result.duplicate_groups[0].len(), 2);
    }

    #[tokio::test]
    async fn test_deduplicate_edges() {
        let prompt_engine = PromptEngine::new().unwrap();
        let context = DeduplicationContext {
            items: vec![
                serde_json::json!({"relationship": "knows", "uuid": "edge1"}),
                serde_json::json!({"relationship": "is_familiar_with", "uuid": "edge2"}),
            ],
            context: "Test edge deduplication".to_string(),
        };

        let result = deduplicate_edges(&prompt_engine, &context).await;
        assert!(result.is_ok());
    }
}
