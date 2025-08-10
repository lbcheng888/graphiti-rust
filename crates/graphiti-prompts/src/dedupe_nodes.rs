//! Node deduplication functionality

use crate::{
    models::{DeduplicationContext, NodeDeduplication},
    PromptEngine,
};
use anyhow::Result;
use serde_json;
use tracing::{info, warn};

/// Deduplicate nodes using LLM
pub async fn deduplicate_nodes(
    prompt_engine: &PromptEngine,
    context: &DeduplicationContext,
) -> Result<NodeDeduplication> {
    info!("Deduplicating nodes");

    // Render the prompt template
    let _prompt = prompt_engine.render("dedupe_nodes", context)?;

    // TODO: This would be called with actual LLM client
    // For now, return empty result
    warn!("LLM integration not yet implemented - returning no duplicates");

    Ok(NodeDeduplication {
        duplicate_groups: Vec::new(),
        explanation: "No duplicates found (LLM not integrated)".to_string(),
    })
}

/// Parse LLM response into structured deduplication result
pub fn parse_node_deduplication_response(response: &str) -> Result<NodeDeduplication> {
    let deduplication: NodeDeduplication = serde_json::from_str(response)?;
    Ok(deduplication)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_node_deduplication_response() {
        let response = r#"
        {
            "duplicate_groups": [
                ["uuid1", "uuid2"],
                ["uuid3", "uuid4", "uuid5"]
            ],
            "explanation": "Found duplicates based on name similarity"
        }
        "#;

        let result = parse_node_deduplication_response(response).unwrap();
        assert_eq!(result.duplicate_groups.len(), 2);
        assert_eq!(result.duplicate_groups[0].len(), 2);
    }

    #[tokio::test]
    async fn test_deduplicate_nodes() {
        let prompt_engine = PromptEngine::new().unwrap();
        let context = DeduplicationContext {
            items: vec![
                serde_json::json!({"name": "Alice", "uuid": "uuid1"}),
                serde_json::json!({"name": "Alice Smith", "uuid": "uuid2"}),
            ],
            context: "Test deduplication".to_string(),
        };

        let result = deduplicate_nodes(&prompt_engine, &context).await;
        assert!(result.is_ok());
    }
}
