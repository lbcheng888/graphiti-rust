//! OpenAI client implementation

use async_trait::async_trait;
use graphiti_core::error::Error;
use graphiti_core::error::Result;
use reqwest::Client;

use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use std::time::Duration;
use tracing::debug;
use tracing::instrument;

use crate::prompts;
use crate::CompletionParams;
use crate::ExtractedEntity;
use crate::ExtractionResult;
use crate::LLMClient;
use crate::Message;
use crate::MessageRole;

/// OpenAI API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// API key
    pub api_key: String,
    /// Base URL (defaults to https://api.openai.com/v1)
    pub base_url: String,
    /// Model to use (e.g., "gpt-4", "gpt-3.5-turbo")
    pub model: String,
    /// Request timeout in seconds
    pub timeout: u64,
    /// Maximum retries
    pub max_retries: u32,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            // Do not read env in defaults to keep tests deterministic.
            // Callers should invoke LLMConfig::from_env() explicitly when needed.
            api_key: String::new(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4".to_string(),
            timeout: 30,
            max_retries: 3,
        }
    }
}

/// OpenAI client
#[derive(Debug)]
pub struct OpenAIClient {
    client: Client,
    config: OpenAIConfig,
}

impl OpenAIClient {
    /// Create a new OpenAI client
    pub fn new(config: OpenAIConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .build()
            .map_err(|e| Error::LLMProvider(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { client, config })
    }

    /// Convert internal message format to OpenAI format
    fn convert_messages(&self, messages: &[Message]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                };
                json!({
                    "role": role,
                    "content": msg.content
                })
            })
            .collect()
    }

    /// Make a completion request to OpenAI
    async fn make_completion_request(
        &self,
        messages: &[Message],
        params: &CompletionParams,
    ) -> Result<String> {
        let mut request_body = json!({
            "model": self.config.model,
            "messages": self.convert_messages(messages),
        });

        if let Some(max_tokens) = params.max_tokens {
            request_body["max_tokens"] = json!(max_tokens);
        }
        if let Some(temperature) = params.temperature {
            request_body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = params.top_p {
            request_body["top_p"] = json!(top_p);
        }
        if let Some(stop) = &params.stop {
            request_body["stop"] = json!(stop);
        }

        let url = format!("{}/chat/completions", self.config.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| Error::LLMProvider(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::LLMProvider(format!(
                "OpenAI API error {}: {}",
                status, error_text
            )));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| Error::LLMProvider(format!("Failed to parse response: {}", e)))?;

        let content = response_json
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|content| content.as_str())
            .ok_or_else(|| Error::LLMProvider("Invalid response format".to_string()))?;

        Ok(content.to_string())
    }
}

#[async_trait]
impl LLMClient for OpenAIClient {
    #[instrument(skip(self, messages))]
    async fn complete(&self, messages: &[Message], params: &CompletionParams) -> Result<String> {
        debug!("Making OpenAI completion request");
        self.make_completion_request(messages, params).await
    }

    #[instrument(skip(self, messages, _schema))]
    async fn complete_structured(
        &self,
        messages: &[Message],
        params: &CompletionParams,
        _schema: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        // For now, just return the completion as a JSON string
        // In the future, we could use OpenAI's structured output features
        let response = self.complete(messages, params).await?;

        serde_json::from_str(&response)
            .map_err(|e| Error::LLMProvider(format!("Failed to parse structured response: {}", e)))
    }

    #[instrument(skip(self, text))]
    async fn extract(&self, text: &str, context: Option<&str>) -> Result<ExtractionResult> {
        let prompt = prompts::ENTITY_EXTRACTION
            .replace("{text}", text)
            .replace("{context}", context.unwrap_or(""));

        let messages = vec![
            Message::system(
                "You are an expert at extracting entities and relationships from text. Return your response as valid JSON.",
            ),
            Message::user(prompt),
        ];

        let params = CompletionParams {
            temperature: Some(0.1),
            ..Default::default()
        };

        let response = self.complete(&messages, &params).await?;

        // Try to parse as JSON, fallback to empty result if parsing fails
        Ok(
            serde_json::from_str(&response).unwrap_or_else(|_| ExtractionResult {
                entities: vec![],
                relationships: vec![],
                summary: Some(response),
                facts: vec![],
            }),
        )
    }

    #[instrument(skip(self, text))]
    async fn summarize(&self, text: &str, max_length: Option<usize>) -> Result<String> {
        let max_length_str = max_length
            .map(|l| l.to_string())
            .unwrap_or_else(|| "100".to_string());
        let prompt = prompts::SUMMARIZATION
            .replace("{text}", text)
            .replace("{max_length}", &max_length_str);

        let messages = vec![
            Message::system("You are an expert at summarizing text concisely."),
            Message::user(prompt),
        ];

        let params = CompletionParams {
            temperature: Some(0.3),
            max_tokens: max_length.map(|l| (l * 2) as u32), // Allow some buffer
            ..Default::default()
        };

        self.complete(&messages, &params).await
    }

    #[instrument(skip(self, entity1, entity2))]
    async fn deduplicate(
        &self,
        entity1: &ExtractedEntity,
        entity2: &ExtractedEntity,
        context: Option<&str>,
    ) -> Result<bool> {
        let prompt = prompts::DEDUPLICATION
            .replace("{name1}", &entity1.name)
            .replace("{type1}", &entity1.entity_type)
            .replace(
                "{attrs1}",
                &serde_json::to_string(&entity1.attributes).unwrap_or_default(),
            )
            .replace("{name2}", &entity2.name)
            .replace("{type2}", &entity2.entity_type)
            .replace(
                "{attrs2}",
                &serde_json::to_string(&entity2.attributes).unwrap_or_default(),
            )
            .replace("{context}", context.unwrap_or(""));

        let messages = vec![
            Message::system(
                "You are an expert at entity deduplication. Respond with only 'true' or 'false'.",
            ),
            Message::user(prompt),
        ];

        let params = CompletionParams {
            temperature: Some(0.0),
            max_tokens: Some(10),
            ..Default::default()
        };

        let response = self.complete(&messages, &params).await?;
        Ok(response.trim().to_lowercase() == "true")
    }

    #[instrument(skip(self, ambiguity, context))]
    async fn clarify(&self, ambiguity: &str, context: &str) -> Result<String> {
        let prompt = prompts::CLARIFICATION
            .replace("{ambiguity}", ambiguity)
            .replace("{context}", context);

        let messages = vec![
            Message::system(
                "You are an expert at generating clarifying questions to resolve ambiguity.",
            ),
            Message::user(prompt),
        ];

        let params = CompletionParams {
            temperature: Some(0.5),
            max_tokens: Some(100),
            ..Default::default()
        };

        self.complete(&messages, &params).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_config_default() {
        let config = OpenAIConfig::default();
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.timeout, 30);
    }

    #[test]
    fn test_message_conversion() {
        let config = OpenAIConfig::default();
        let client = OpenAIClient::new(config).unwrap();

        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let converted = client.convert_messages(&messages);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0]["role"], "system");
        assert_eq!(converted[1]["role"], "user");
        assert_eq!(converted[2]["role"], "assistant");
    }
}
