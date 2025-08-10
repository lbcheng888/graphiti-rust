//! Ollama client implementation for local LLM inference

use async_trait::async_trait;
use governor::clock::DefaultClock;
use governor::state::InMemoryState;
use governor::state::NotKeyed;
use governor::Quota;
use governor::RateLimiter;
use graphiti_core::error::Error;
use graphiti_core::error::Result;
use moka::future::Cache;
use reqwest::Client;
use reqwest::StatusCode;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tracing::debug;
use tracing::instrument;
use tracing::warn;

use crate::prompts;
use crate::CompletionParams;
use crate::ExtractedEntity;
use crate::ExtractionResult;
use crate::LLMClient;
use crate::Message;
use crate::MessageRole;

/// Serde helpers for Duration
mod duration_serde {
    use serde::Deserialize;
    use serde::Deserializer;
    use serde::Serialize;
    use serde::Serializer;
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

/// Ollama API configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OllamaConfig {
    /// Base URL (defaults to http://localhost:11434)
    pub base_url: String,
    /// Model to use (e.g., "llama3.2:latest", "mistral:latest")
    pub model: String,
    /// Request timeout
    #[serde(with = "duration_serde")]
    pub timeout: Duration,
    /// Maximum retries
    pub max_retries: u32,
    /// Rate limit (requests per minute) - usually higher for local inference
    pub rate_limit: u32,
    /// Keep model loaded in memory
    pub keep_alive: Duration,
    /// Additional model options
    pub options: OllamaOptions,
}

/// Ollama model options
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OllamaOptions {
    /// Temperature for sampling (0.0 to 2.0)
    pub temperature: Option<f32>,
    /// Top-p sampling
    pub top_p: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<u32>,
    /// Number of tokens to predict
    pub num_predict: Option<u32>,
    /// Stop sequences
    pub stop: Option<Vec<String>>,
    /// Number of GPU layers to use
    pub num_gpu: Option<u32>,
    /// Context window size
    pub num_ctx: Option<u32>,
}

impl Default for OllamaOptions {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(1.0),
            top_k: Some(40),
            num_predict: Some(1000),
            stop: None,
            num_gpu: None, // Auto-detect
            num_ctx: Some(4096),
        }
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "llama3.2:latest".to_string(),
            timeout: Duration::from_secs(120), // Longer timeout for local inference
            max_retries: 3,
            rate_limit: 120,                      // Higher rate limit for local
            keep_alive: Duration::from_secs(600), // Keep model loaded for 10 minutes
            options: OllamaOptions::default(),
        }
    }
}

/// Ollama API client
pub struct OllamaClient {
    config: OllamaConfig,
    client: Client,
    rate_limiter: Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>,
    cache: Cache<String, String>,
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(config: OllamaConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| Error::Configuration(format!("Failed to create HTTP client: {}", e)))?;

        // Create rate limiter
        let rate_limit = NonZeroU32::new(config.rate_limit)
            .ok_or_else(|| Error::Configuration("Rate limit must be greater than 0".to_string()))?;
        let quota = Quota::per_minute(rate_limit);
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        // Create cache with 1 hour TTL and 1000 max entries
        let cache = Cache::builder()
            .time_to_live(Duration::from_secs(3600))
            .max_capacity(1000)
            .build();

        Ok(Self {
            config,
            client,
            rate_limiter,
            cache,
        })
    }

    /// Check if Ollama is running and accessible
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.config.base_url);

        match self.client.get(&url).send().await {
            Ok(response) if response.status().is_success() => Ok(true),
            Ok(_) => Ok(false),
            Err(_) => Ok(false),
        }
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.config.base_url);

        #[derive(Deserialize)]
        struct TagsResponse {
            models: Vec<ModelInfo>,
        }

        #[derive(Deserialize)]
        struct ModelInfo {
            name: String,
        }

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::LLMProvider(format!("Failed to list models: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::LLMProvider("Failed to list models".to_string()));
        }

        let tags: TagsResponse = response
            .json()
            .await
            .map_err(|e| Error::LLMProvider(format!("Failed to parse models response: {}", e)))?;

        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }

    /// Pull a model if not already available
    pub async fn pull_model(&self, model: &str) -> Result<()> {
        let url = format!("{}/api/pull", self.config.base_url);

        let body = json!({
            "name": model
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::LLMProvider(format!("Failed to pull model: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::LLMProvider(format!(
                "Failed to pull model: {}",
                error_text
            )));
        }

        Ok(())
    }

    /// Make a request to the Ollama API with retries
    async fn request<T: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: serde_json::Value,
    ) -> Result<T> {
        let url = format!("{}/api/{}", self.config.base_url, endpoint);

        for attempt in 0..self.config.max_retries {
            // Rate limiting
            self.rate_limiter.until_ready().await;

            let response = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| Error::LLMProvider(format!("Request failed: {}", e)))?;

            match response.status() {
                StatusCode::OK => {
                    return response.json::<T>().await.map_err(|e| {
                        Error::LLMProvider(format!("Failed to parse response: {}", e))
                    });
                }
                StatusCode::TOO_MANY_REQUESTS => {
                    warn!("Rate limited, retrying after delay");
                    tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))).await;
                }
                StatusCode::INTERNAL_SERVER_ERROR
                | StatusCode::BAD_GATEWAY
                | StatusCode::SERVICE_UNAVAILABLE => {
                    warn!("Server error, retrying after delay");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                StatusCode::NOT_FOUND => {
                    let error_text = response.text().await.unwrap_or_default();
                    if error_text.contains("model") {
                        // Try to pull the model
                        warn!("Model not found, attempting to pull: {}", self.config.model);
                        if let Err(e) = self.pull_model(&self.config.model).await {
                            return Err(Error::LLMProvider(format!(
                                "Model not found and pull failed: {}",
                                e
                            )));
                        }
                        // Retry the request after pulling
                        continue;
                    }
                    return Err(Error::LLMProvider(format!("API error: {}", error_text)));
                }
                _ => {
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(Error::LLMProvider(format!("API error: {}", error_text)));
                }
            }
        }

        Err(Error::LLMProvider("Max retries exceeded".to_string()))
    }
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
    format: String,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    #[serde(default)]
    response: String,
    #[serde(default)]
    message: Option<OllamaMessage>,
    #[allow(dead_code)]
    done: bool,
}

#[async_trait]
impl LLMClient for OllamaClient {
    #[instrument(skip(self, messages))]
    async fn complete(&self, messages: &[Message], params: &CompletionParams) -> Result<String> {
        // Convert messages
        let ollama_messages: Vec<OllamaMessage> = messages
            .iter()
            .map(|m| OllamaMessage {
                role: match m.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                }
                .to_string(),
                content: m.content.clone(),
            })
            .collect();

        // Merge completion params with config options
        let mut options = self.config.options.clone();
        if let Some(temp) = params.temperature {
            options.temperature = Some(temp);
        }
        if let Some(top_p) = params.top_p {
            options.top_p = Some(top_p);
        }
        if let Some(max_tokens) = params.max_tokens {
            options.num_predict = Some(max_tokens);
        }
        if let Some(stop) = &params.stop {
            options.stop = Some(stop.clone());
        }

        let keep_alive = format!("{}s", self.config.keep_alive.as_secs());

        let request = OllamaChatRequest {
            model: self.config.model.clone(),
            messages: ollama_messages,
            options: Some(options),
            keep_alive: Some(keep_alive),
            format: None,
            stream: false,
        };

        let response: OllamaResponse = self.request("chat", serde_json::to_value(request)?).await?;

        if let Some(message) = response.message {
            Ok(message.content)
        } else {
            Ok(response.response)
        }
    }

    #[instrument(skip(self, messages))]
    async fn complete_structured(
        &self,
        messages: &[Message],
        params: &CompletionParams,
        _schema: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        // For structured output, we modify the system message to request JSON format
        let mut structured_messages = messages.to_vec();

        // Add JSON format instruction to the last user message or create a new one
        if let Some(last_msg) = structured_messages.last_mut() {
            if matches!(last_msg.role, MessageRole::User) {
                last_msg
                    .content
                    .push_str("\n\nPlease respond with valid JSON only, no additional text.");
            } else {
                structured_messages.push(Message::user(
                    "Please respond with valid JSON only, no additional text.",
                ));
            }
        } else {
            structured_messages.push(Message::user(
                "Please respond with valid JSON only, no additional text.",
            ));
        }

        // Convert messages
        let ollama_messages: Vec<OllamaMessage> = structured_messages
            .iter()
            .map(|m| OllamaMessage {
                role: match m.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                }
                .to_string(),
                content: m.content.clone(),
            })
            .collect();

        // Merge completion params with config options
        let mut options = self.config.options.clone();
        if let Some(temp) = params.temperature {
            options.temperature = Some(temp);
        }
        if let Some(top_p) = params.top_p {
            options.top_p = Some(top_p);
        }
        if let Some(max_tokens) = params.max_tokens {
            options.num_predict = Some(max_tokens);
        }
        if let Some(stop) = &params.stop {
            options.stop = Some(stop.clone());
        }

        let keep_alive = format!("{}s", self.config.keep_alive.as_secs());

        let request = OllamaChatRequest {
            model: self.config.model.clone(),
            messages: ollama_messages,
            options: Some(options),
            keep_alive: Some(keep_alive),
            format: Some("json".to_string()), // Request JSON format
            stream: false,
        };

        let response: OllamaResponse = self.request("chat", serde_json::to_value(request)?).await?;

        let content = if let Some(message) = response.message {
            message.content
        } else {
            response.response
        };

        // Try to parse the JSON response
        serde_json::from_str(&content)
            .map_err(|e| Error::LLMProvider(format!("Failed to parse structured response: {}", e)))
    }

    #[instrument(skip(self, text))]
    async fn extract(&self, text: &str, context: Option<&str>) -> Result<ExtractionResult> {
        // Check cache
        let cache_key = format!("extract:{}:{}", text, context.unwrap_or(""));
        if let Some(cached) = self.cache.get(&cache_key).await {
            if let Ok(result) = serde_json::from_str(&cached) {
                debug!("Cache hit for extraction");
                return Ok(result);
            }
        }

        let prompt = prompts::ENTITY_EXTRACTION
            .replace("{text}", text)
            .replace("{context}", context.unwrap_or(""));

        let messages = vec![
            Message::system(
                "You are an expert at extracting structured information from text. Respond with valid JSON only.",
            ),
            Message::user(prompt),
        ];

        let schema = json!({
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "entity_type": { "type": "string" },
                            "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
                            "attributes": { "type": "object" }
                        },
                        "required": ["name", "entity_type", "confidence"]
                    }
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": { "type": "string" },
                            "target": { "type": "string" },
                            "relationship": { "type": "string" },
                            "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
                            "attributes": { "type": "object" }
                        },
                        "required": ["source", "target", "relationship", "confidence"]
                    }
                },
                "summary": { "type": "string" },
                "facts": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["entities", "relationships", "facts"]
        });

        let params = CompletionParams {
            temperature: Some(0.3), // Lower temperature for extraction
            ..Default::default()
        };

        let json_result = self
            .complete_structured(&messages, &params, &schema)
            .await?;
        let result: ExtractionResult = serde_json::from_value(json_result)
            .map_err(|e| Error::LLMProvider(format!("Failed to parse extraction result: {}", e)))?;

        // Cache the result
        if let Ok(json) = serde_json::to_string(&result) {
            self.cache.insert(cache_key, json).await;
        }

        Ok(result)
    }

    #[instrument(skip(self, text))]
    async fn summarize(&self, text: &str, max_length: Option<usize>) -> Result<String> {
        let prompt = prompts::SUMMARIZATION
            .replace("{text}", text)
            .replace("{max_length}", &max_length.unwrap_or(100).to_string());

        let messages = vec![
            Message::system("You are an expert at creating concise, informative summaries."),
            Message::user(prompt),
        ];

        let params = CompletionParams {
            temperature: Some(0.5),
            max_tokens: Some(max_length.unwrap_or(100) as u32 * 2), // Rough estimate
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
            .replace("{attrs1}", &serde_json::to_string(&entity1.attributes)?)
            .replace("{name2}", &entity2.name)
            .replace("{type2}", &entity2.entity_type)
            .replace("{attrs2}", &serde_json::to_string(&entity2.attributes)?)
            .replace("{context}", context.unwrap_or(""));

        let messages = vec![
            Message::system(
                "You are an expert at entity resolution and deduplication. Respond with only 'true' or 'false'.",
            ),
            Message::user(prompt),
        ];

        let params = CompletionParams {
            temperature: Some(0.1), // Very low temperature for binary decision
            max_tokens: Some(10),
            ..Default::default()
        };

        let response = self.complete(&messages, &params).await?;
        Ok(response.trim().to_lowercase() == "true")
    }

    #[instrument(skip(self))]
    async fn clarify(&self, ambiguity: &str, context: &str) -> Result<String> {
        let prompt = prompts::CLARIFICATION
            .replace("{ambiguity}", ambiguity)
            .replace("{context}", context);

        let messages = vec![
            Message::system(
                "You are an expert at identifying and resolving ambiguities in information.",
            ),
            Message::user(prompt),
        ];

        let params = CompletionParams {
            temperature: Some(0.7),
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
    fn test_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "llama3.2:latest");
        assert_eq!(config.rate_limit, 120);
    }

    #[test]
    fn test_client_creation() {
        let config = OllamaConfig::default();
        let result = OllamaClient::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_options_serialization() {
        let options = OllamaOptions {
            temperature: Some(0.8),
            top_p: Some(0.9),
            top_k: Some(50),
            num_predict: Some(500),
            stop: Some(vec!["END".to_string()]),
            num_gpu: Some(1),
            num_ctx: Some(8192),
        };

        let json = serde_json::to_string(&options).unwrap();
        let deserialized: OllamaOptions = serde_json::from_str(&json).unwrap();

        assert_eq!(options.temperature, deserialized.temperature);
        assert_eq!(options.top_p, deserialized.top_p);
        assert_eq!(options.top_k, deserialized.top_k);
    }
}
