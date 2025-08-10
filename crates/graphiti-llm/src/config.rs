//! LLM configuration for multiple providers

use crate::{OllamaConfig, OpenAIConfig};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Primary LLM provider to use
    pub provider: LLMProvider,
    /// OpenAI configuration
    pub openai: OpenAIConfig,
    /// Ollama configuration
    pub ollama: OllamaConfig,
    /// Groq configuration
    pub groq: GroqConfig,
    /// Hugging Face configuration
    pub huggingface: HuggingFaceConfig,
}

/// Supported LLM providers
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LLMProvider {
    /// OpenAI GPT models
    OpenAI,
    /// Local Ollama models
    Ollama,
    /// Groq inference API
    Groq,
    /// Hugging Face Inference API
    HuggingFace,
}

impl Default for LLMProvider {
    fn default() -> Self {
        Self::Ollama // Default to free local option
    }
}

/// Groq API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqConfig {
    /// API key (usually from environment)
    pub api_key: String,
    /// Base URL
    pub base_url: String,
    /// Model to use
    pub model: String,
    /// Request timeout
    #[serde(with = "duration_serde")]
    pub timeout: Duration,
    /// Maximum retries
    pub max_retries: u32,
    /// Rate limit (requests per minute)
    pub rate_limit: u32,
}

impl Default for GroqConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            model: "llama-3.1-8b-instant".to_string(),
            timeout: Duration::from_secs(30),
            max_retries: 3,
            rate_limit: 30, // Groq has rate limits
        }
    }
}

/// Hugging Face Inference API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    /// API key (optional for free tier)
    pub api_key: String,
    /// Base URL
    pub base_url: String,
    /// Model to use
    pub model: String,
    /// Request timeout
    #[serde(with = "duration_serde")]
    pub timeout: Duration,
    /// Maximum retries
    pub max_retries: u32,
    /// Rate limit (requests per minute)
    pub rate_limit: u32,
    /// Use cached responses when available
    pub use_cache: bool,
    /// Wait for model to load
    pub wait_for_model: bool,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api-inference.huggingface.co".to_string(),
            model: "microsoft/DialoGPT-medium".to_string(),
            timeout: Duration::from_secs(60),
            max_retries: 3,
            rate_limit: 60,
            use_cache: true,
            wait_for_model: true,
        }
    }
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: LLMProvider::default(),
            openai: OpenAIConfig::default(),
            ollama: OllamaConfig::default(),
            groq: GroqConfig::default(),
            huggingface: HuggingFaceConfig::default(),
        }
    }
}

impl LLMConfig {
    /// Get the active provider's name as a string
    pub fn provider_name(&self) -> &'static str {
        match self.provider {
            LLMProvider::OpenAI => "openai",
            LLMProvider::Ollama => "ollama",
            LLMProvider::Groq => "groq",
            LLMProvider::HuggingFace => "huggingface",
        }
    }

    /// Check if the active provider requires an API key
    pub fn requires_api_key(&self) -> bool {
        match self.provider {
            LLMProvider::OpenAI => true,
            LLMProvider::Ollama => false,
            LLMProvider::Groq => true,
            LLMProvider::HuggingFace => false, // Free tier available
        }
    }

    /// Get API key for the active provider
    pub fn get_api_key(&self) -> Option<&str> {
        match self.provider {
            LLMProvider::OpenAI if !self.openai.api_key.is_empty() => Some(&self.openai.api_key),
            LLMProvider::Groq if !self.groq.api_key.is_empty() => Some(&self.groq.api_key),
            LLMProvider::HuggingFace if !self.huggingface.api_key.is_empty() => {
                Some(&self.huggingface.api_key)
            }
            _ => None,
        }
    }

    /// Update configuration from environment variables
    pub fn from_env(&mut self) {
        // Update API keys from environment
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            self.openai.api_key = key;
        }

        if let Ok(key) = std::env::var("GROQ_API_KEY") {
            self.groq.api_key = key;
        }

        if let Ok(key) = std::env::var("HUGGINGFACE_API_KEY") {
            self.huggingface.api_key = key;
        }

        // Update provider from environment
        if let Ok(provider) = std::env::var("LLM_PROVIDER") {
            match provider.to_lowercase().as_str() {
                "openai" => self.provider = LLMProvider::OpenAI,
                "ollama" => self.provider = LLMProvider::Ollama,
                "groq" => self.provider = LLMProvider::Groq,
                "huggingface" | "hf" => self.provider = LLMProvider::HuggingFace,
                _ => {
                    eprintln!(
                        "Warning: Unknown LLM provider '{}', using default",
                        provider
                    );
                }
            }
        }

        // Update model from environment
        if let Ok(model) = std::env::var("LLM_MODEL") {
            match self.provider {
                LLMProvider::OpenAI => self.openai.model = model,
                LLMProvider::Ollama => self.ollama.model = model,
                LLMProvider::Groq => self.groq.model = model,
                LLMProvider::HuggingFace => self.huggingface.model = model,
            }
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        match self.provider {
            LLMProvider::OpenAI => {
                if self.openai.api_key.is_empty() {
                    return Err("OpenAI API key is required".to_string());
                }
            }
            LLMProvider::Ollama => {
                // No validation needed for local Ollama
            }
            LLMProvider::Groq => {
                if self.groq.api_key.is_empty() {
                    return Err("Groq API key is required".to_string());
                }
            }
            LLMProvider::HuggingFace => {
                // API key is optional for HF free tier
            }
        }

        Ok(())
    }
}

/// Serde helpers for Duration
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_default() {
        let config = LLMConfig::default();
        assert_eq!(config.provider, LLMProvider::Ollama);
        assert_eq!(config.provider_name(), "ollama");
        assert!(!config.requires_api_key());
    }

    #[test]
    fn test_provider_serialization() {
        let provider = LLMProvider::OpenAI;
        let json = serde_json::to_string(&provider).unwrap();
        assert_eq!(json, "\"openai\"");

        let deserialized: LLMProvider = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, LLMProvider::OpenAI);
    }

    #[test]
    fn test_config_validation() {
        let mut config = LLMConfig::default();

        // Ollama should validate without API key
        assert!(config.validate().is_ok());

        // OpenAI should require API key
        config.provider = LLMProvider::OpenAI;
        assert!(config.validate().is_err());

        config.openai.api_key = "test-key".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_env_update() {
        std::env::set_var("OPENAI_API_KEY", "test-openai-key");
        std::env::set_var("LLM_PROVIDER", "openai");
        std::env::set_var("LLM_MODEL", "gpt-4");

        let mut config = LLMConfig::default();
        config.from_env();

        assert_eq!(config.provider, LLMProvider::OpenAI);
        assert_eq!(config.openai.api_key, "test-openai-key");
        assert_eq!(config.openai.model, "gpt-4");

        // Clean up
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("LLM_PROVIDER");
        std::env::remove_var("LLM_MODEL");
    }
}
