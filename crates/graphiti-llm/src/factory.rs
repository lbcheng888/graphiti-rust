//! LLM client factory for creating clients from configuration

use crate::CompletionParams;
use crate::ExtractedEntity;
use crate::ExtractionResult;
use crate::GroqConfig;
use crate::HuggingFaceConfig;
use crate::LLMClient;
use crate::LLMConfig;
use crate::LLMProvider;
use crate::Message;
use crate::OllamaClient;
use crate::OpenAIClient;
use async_trait::async_trait;
use graphiti_core::error::Error;
use graphiti_core::error::Result;

/// Multi-provider LLM client enum
pub enum MultiLLMClient {
    /// OpenAI client
    OpenAI(OpenAIClient),
    /// Ollama client
    Ollama(OllamaClient),
    /// Groq client
    Groq(GroqClient),
    /// Hugging Face client
    HuggingFace(HuggingFaceClient),
}

impl MultiLLMClient {
    /// Create an LLM client from configuration
    pub async fn new(config: &LLMConfig) -> Result<Self> {
        match config.provider {
            LLMProvider::OpenAI => {
                let client = OpenAIClient::new(config.openai.clone())?;
                Ok(Self::OpenAI(client))
            }
            LLMProvider::Ollama => {
                let client = OllamaClient::new(config.ollama.clone())?;

                // Check if Ollama is running
                if !client.health_check().await? {
                    return Err(Error::Configuration(
                        "Ollama is not running. Please start Ollama: https://ollama.ai".to_string(),
                    ));
                }

                // Check if the model is available
                let available_models = client.list_models().await?;
                if !available_models.contains(&config.ollama.model) {
                    return Err(Error::Configuration(format!(
                        "Model '{}' not found. Available models: {}. Use 'ollama pull {}' to download it.",
                        config.ollama.model,
                        available_models.join(", "),
                        config.ollama.model
                    )));
                }

                Ok(Self::Ollama(client))
            }
            LLMProvider::Groq => {
                let client = GroqClient::new(config.groq.clone())?;
                Ok(Self::Groq(client))
            }
            LLMProvider::HuggingFace => {
                let client = HuggingFaceClient::new(config.huggingface.clone())?;
                Ok(Self::HuggingFace(client))
            }
        }
    }

    /// Get the provider name
    pub fn provider_name(&self) -> &'static str {
        match self {
            Self::OpenAI(_) => "openai",
            Self::Ollama(_) => "ollama",
            Self::Groq(_) => "groq",
            Self::HuggingFace(_) => "huggingface",
        }
    }
}

#[async_trait]
impl LLMClient for MultiLLMClient {
    async fn complete(&self, messages: &[Message], params: &CompletionParams) -> Result<String> {
        match self {
            Self::OpenAI(client) => client.complete(messages, params).await,
            Self::Ollama(client) => client.complete(messages, params).await,
            Self::Groq(client) => client.complete(messages, params).await,
            Self::HuggingFace(client) => client.complete(messages, params).await,
        }
    }

    async fn complete_structured(
        &self,
        messages: &[Message],
        params: &CompletionParams,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        match self {
            Self::OpenAI(client) => client.complete_structured(messages, params, schema).await,
            Self::Ollama(client) => client.complete_structured(messages, params, schema).await,
            Self::Groq(client) => client.complete_structured(messages, params, schema).await,
            Self::HuggingFace(client) => client.complete_structured(messages, params, schema).await,
        }
    }

    async fn extract(&self, text: &str, context: Option<&str>) -> Result<ExtractionResult> {
        match self {
            Self::OpenAI(client) => client.extract(text, context).await,
            Self::Ollama(client) => client.extract(text, context).await,
            Self::Groq(client) => client.extract(text, context).await,
            Self::HuggingFace(client) => client.extract(text, context).await,
        }
    }

    async fn summarize(&self, text: &str, max_length: Option<usize>) -> Result<String> {
        match self {
            Self::OpenAI(client) => client.summarize(text, max_length).await,
            Self::Ollama(client) => client.summarize(text, max_length).await,
            Self::Groq(client) => client.summarize(text, max_length).await,
            Self::HuggingFace(client) => client.summarize(text, max_length).await,
        }
    }

    async fn deduplicate(
        &self,
        entity1: &ExtractedEntity,
        entity2: &ExtractedEntity,
        context: Option<&str>,
    ) -> Result<bool> {
        match self {
            Self::OpenAI(client) => client.deduplicate(entity1, entity2, context).await,
            Self::Ollama(client) => client.deduplicate(entity1, entity2, context).await,
            Self::Groq(client) => client.deduplicate(entity1, entity2, context).await,
            Self::HuggingFace(client) => client.deduplicate(entity1, entity2, context).await,
        }
    }

    async fn clarify(&self, ambiguity: &str, context: &str) -> Result<String> {
        match self {
            Self::OpenAI(client) => client.clarify(ambiguity, context).await,
            Self::Ollama(client) => client.clarify(ambiguity, context).await,
            Self::Groq(client) => client.clarify(ambiguity, context).await,
            Self::HuggingFace(client) => client.clarify(ambiguity, context).await,
        }
    }
}

/// Create an LLM client from configuration (convenience function)
pub async fn create_llm_client(config: &LLMConfig) -> Result<MultiLLMClient> {
    MultiLLMClient::new(config).await
}

/// Groq client implementation (OpenAI-compatible API)
pub struct GroqClient {
    inner: OpenAIClient,
}

impl GroqClient {
    /// Create a new Groq client
    pub fn new(config: GroqConfig) -> Result<Self> {
        // Convert Groq config to OpenAI config since Groq uses OpenAI-compatible API
        let openai_config = crate::OpenAIConfig {
            api_key: config.api_key,
            base_url: config.base_url,
            model: config.model,
            timeout: config.timeout.as_secs(),
            max_retries: config.max_retries,
        };

        let inner = OpenAIClient::new(openai_config)?;
        Ok(Self { inner })
    }
}

#[async_trait::async_trait]
impl LLMClient for GroqClient {
    async fn complete(
        &self,
        messages: &[crate::Message],
        params: &crate::CompletionParams,
    ) -> Result<String> {
        self.inner.complete(messages, params).await
    }

    async fn complete_structured(
        &self,
        messages: &[crate::Message],
        params: &crate::CompletionParams,
        schema: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.inner
            .complete_structured(messages, params, schema)
            .await
    }

    async fn extract(&self, text: &str, context: Option<&str>) -> Result<crate::ExtractionResult> {
        self.inner.extract(text, context).await
    }

    async fn summarize(&self, text: &str, max_length: Option<usize>) -> Result<String> {
        self.inner.summarize(text, max_length).await
    }

    async fn deduplicate(
        &self,
        entity1: &crate::ExtractedEntity,
        entity2: &crate::ExtractedEntity,
        context: Option<&str>,
    ) -> Result<bool> {
        self.inner.deduplicate(entity1, entity2, context).await
    }

    async fn clarify(&self, ambiguity: &str, context: &str) -> Result<String> {
        self.inner.clarify(ambiguity, context).await
    }
}

/// Hugging Face Inference API client
pub struct HuggingFaceClient {
    config: HuggingFaceConfig,
    client: reqwest::Client,
}

impl HuggingFaceClient {
    /// Create a new Hugging Face client
    pub fn new(config: HuggingFaceConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| Error::Configuration(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { config, client })
    }

    /// Make a request to the Hugging Face Inference API
    async fn request(&self, text: &str) -> Result<String> {
        let url = format!("{}/models/{}", self.config.base_url, self.config.model);

        let mut request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "inputs": text,
                "options": {
                    "use_cache": self.config.use_cache,
                    "wait_for_model": self.config.wait_for_model
                }
            }));

        // Add authorization header if API key is provided
        if !self.config.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        let response = request
            .send()
            .await
            .map_err(|e| Error::LLMProvider(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::LLMProvider(format!("API error: {}", error_text)));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| Error::LLMProvider(format!("Failed to parse response: {}", e)))?;

        // Handle different response formats from HF
        if let Some(text) = result.as_array().and_then(|arr| arr.first()) {
            if let Some(generated_text) = text.get("generated_text").and_then(|t| t.as_str()) {
                return Ok(generated_text.to_string());
            }
        }

        // Fallback to string representation
        Ok(result.to_string())
    }
}

#[async_trait::async_trait]
impl LLMClient for HuggingFaceClient {
    async fn complete(
        &self,
        messages: &[crate::Message],
        _params: &crate::CompletionParams,
    ) -> Result<String> {
        // Convert messages to a single prompt
        let prompt = messages
            .iter()
            .map(|m| match m.role {
                crate::MessageRole::System => format!("System: {}", m.content),
                crate::MessageRole::User => format!("Human: {}", m.content),
                crate::MessageRole::Assistant => format!("Assistant: {}", m.content),
            })
            .collect::<Vec<_>>()
            .join("\n");

        self.request(&prompt).await
    }

    async fn complete_structured(
        &self,
        messages: &[crate::Message],
        params: &crate::CompletionParams,
        _schema: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        // For HF, we can't guarantee structured output, so we try best effort
        let mut structured_messages = messages.to_vec();
        structured_messages.push(crate::Message::user(
            "Please respond with valid JSON only, no additional text.",
        ));

        let response = self.complete(&structured_messages, params).await?;

        serde_json::from_str(&response)
            .map_err(|e| Error::LLMProvider(format!("Failed to parse structured response: {}", e)))
    }

    async fn extract(&self, text: &str, context: Option<&str>) -> Result<crate::ExtractionResult> {
        let prompt = crate::prompts::ENTITY_EXTRACTION
            .replace("{text}", text)
            .replace("{context}", context.unwrap_or(""));

        let messages = vec![
            crate::Message::system(
                "You are an expert at extracting structured information from text. Respond with valid JSON only.",
            ),
            crate::Message::user(prompt),
        ];

        let schema = serde_json::json!({});
        let params = crate::CompletionParams::default();

        let json_result = self
            .complete_structured(&messages, &params, &schema)
            .await?;
        serde_json::from_value(json_result)
            .map_err(|e| Error::LLMProvider(format!("Failed to parse extraction result: {}", e)))
    }

    async fn summarize(&self, text: &str, max_length: Option<usize>) -> Result<String> {
        let prompt = crate::prompts::SUMMARIZATION
            .replace("{text}", text)
            .replace("{max_length}", &max_length.unwrap_or(100).to_string());

        let messages = vec![
            crate::Message::system("You are an expert at creating concise, informative summaries."),
            crate::Message::user(prompt),
        ];

        let params = crate::CompletionParams::default();
        self.complete(&messages, &params).await
    }

    async fn deduplicate(
        &self,
        entity1: &crate::ExtractedEntity,
        entity2: &crate::ExtractedEntity,
        context: Option<&str>,
    ) -> Result<bool> {
        let prompt = crate::prompts::DEDUPLICATION
            .replace("{name1}", &entity1.name)
            .replace("{type1}", &entity1.entity_type)
            .replace("{attrs1}", &serde_json::to_string(&entity1.attributes)?)
            .replace("{name2}", &entity2.name)
            .replace("{type2}", &entity2.entity_type)
            .replace("{attrs2}", &serde_json::to_string(&entity2.attributes)?)
            .replace("{context}", context.unwrap_or(""));

        let messages = vec![
            crate::Message::system(
                "You are an expert at entity resolution and deduplication. Respond with only 'true' or 'false'.",
            ),
            crate::Message::user(prompt),
        ];

        let params = crate::CompletionParams::default();
        let response = self.complete(&messages, &params).await?;
        Ok(response.trim().to_lowercase() == "true")
    }

    async fn clarify(&self, ambiguity: &str, context: &str) -> Result<String> {
        let prompt = crate::prompts::CLARIFICATION
            .replace("{ambiguity}", ambiguity)
            .replace("{context}", context);

        let messages = vec![
            crate::Message::system(
                "You are an expert at identifying and resolving ambiguities in information.",
            ),
            crate::Message::user(prompt),
        ];

        let params = crate::CompletionParams::default();
        self.complete(&messages, &params).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groq_client_creation() {
        let config = GroqConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        };

        let result = GroqClient::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_huggingface_client_creation() {
        let config = HuggingFaceConfig::default();
        let result = HuggingFaceClient::new(config);
        assert!(result.is_ok());
    }
}
