//! Node summarization functionality

use crate::{
    models::{NodeSummary, SummarizationContext},
    PromptEngine,
};
use anyhow::Result;
use serde_json;
use tracing::{info, warn};

/// Summarize node information using LLM
pub async fn summarize_node(
    prompt_engine: &PromptEngine,
    context: &SummarizationContext,
) -> Result<NodeSummary> {
    info!("Summarizing node information");

    // Render the prompt template
    let _prompt = prompt_engine.render("summarize_nodes", context)?;

    // TODO: This would be called with actual LLM client
    // For now, return basic summary
    warn!("LLM integration not yet implemented - returning basic summary");

    Ok(NodeSummary {
        summary: "Basic summary (LLM not integrated)".to_string(),
        attributes: serde_json::json!({}),
    })
}

/// Parse LLM response into structured summary
pub fn parse_summary_response(response: &str) -> Result<NodeSummary> {
    let summary: NodeSummary = serde_json::from_str(response)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_summary_response() {
        let response = r#"
        {
            "summary": "Alice is a software engineer who works at Tech Corp",
            "attributes": {
                "profession": "software engineer",
                "company": "Tech Corp"
            }
        }
        "#;

        let result = parse_summary_response(response).unwrap();
        assert!(result.summary.contains("Alice"));
        assert!(result.attributes.is_object());
    }

    #[tokio::test]
    async fn test_summarize_node() {
        let prompt_engine = PromptEngine::new().unwrap();
        let context = SummarizationContext {
            node_data: serde_json::json!({"name": "Alice", "type": "Person"}),
            new_information: "Alice works as a software engineer".to_string(),
            previous_summary: None,
        };

        let result = summarize_node(&prompt_engine, &context).await;
        assert!(result.is_ok());
    }
}
