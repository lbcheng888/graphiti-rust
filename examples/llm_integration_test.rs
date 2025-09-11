//! End-to-end test for LLM and embedding service integration

use graphiti_llm::{
    CompletionParams, EmbeddingServiceConfig, LLMServiceConfig, Message, OllamaConfig,
    ServiceConfig, ServiceFactory,
};
use std::env;
use tokio;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Starting LLM and Embedding Service Integration Test");

    // Test 1: Create local services (Ollama + EmbeddingGemma)
    info!("=== Test 1: Local Services (Ollama + EmbeddingGemma) ===");

    match test_local_services().await {
        Ok(_) => info!("✅ Local services test passed"),
        Err(e) => warn!("⚠️ Local services test failed: {}", e),
    }

    // Test 2: Create hybrid services (OpenAI + EmbeddingGemma) if API key is available
    info!("=== Test 2: Hybrid Services (OpenAI + EmbeddingGemma) ===");

    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        match test_hybrid_services(api_key).await {
            Ok(_) => info!("✅ Hybrid services test passed"),
            Err(e) => warn!("⚠️ Hybrid services test failed: {}", e),
        }
    } else {
        warn!("⚠️ OPENAI_API_KEY not set, skipping hybrid services test");
    }

    // Test 3 removed (no external embedding server)

    info!("Integration test completed");
    Ok(())
}

async fn test_local_services() -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating local services...");

    let config = ServiceConfig {
        llm: LLMServiceConfig::Ollama(OllamaConfig {
            model: "llama3.2:3b".to_string(),
            base_url: "http://localhost:11434".to_string(),
            ..Default::default()
        }),
        embedding: EmbeddingServiceConfig::Generic(graphiti_llm::EmbedderConfig {
            provider: graphiti_llm::EmbeddingProvider::EmbedAnything,
            api_key: String::new(),
            model: "google/embeddinggemma-300m".to_string(),
            dimension: 768,
            batch_size: 4,
            timeout: std::time::Duration::from_secs(60),
            device: Some("auto".to_string()),
            max_length: Some(8192),
            cache_dir: None,
        }),
    };

    let (llm_client, embedding_client) = ServiceFactory::create_services(&config).await?;

    // Test LLM completion
    info!("Testing LLM completion...");
    let messages = vec![Message::user(
        "Hello! Please respond with a simple greeting.",
    )];
    let params = CompletionParams {
        max_tokens: Some(50),
        temperature: Some(0.7),
        ..Default::default()
    };

    match llm_client.complete(&messages, &params).await {
        Ok(response) => {
            info!("LLM Response: {}", response);
            if response.trim().is_empty() {
                return Err("LLM returned empty response".into());
            }
        }
        Err(e) => {
            error!("LLM completion failed: {}", e);
            return Err(e.into());
        }
    }

    // Test embedding generation
    info!("Testing embedding generation...");
    let test_texts = vec![
        "Hello world".to_string(),
        "This is a test".to_string(),
        "Embedding generation test".to_string(),
    ];

    match embedding_client.embed_batch(&test_texts).await {
        Ok(embeddings) => {
            info!("Generated {} embeddings", embeddings.len());
            if embeddings.len() != test_texts.len() {
                return Err("Embedding count mismatch".into());
            }

            for (i, embedding) in embeddings.iter().enumerate() {
                info!("Embedding {}: dimension = {}", i, embedding.len());
                if embedding.is_empty() {
                    return Err("Empty embedding generated".into());
                }
            }
        }
        Err(e) => {
            error!("Embedding generation failed: {}", e);
            return Err(e.into());
        }
    }

    // Test health check
    info!("Testing service health check...");
    let health_status = ServiceFactory::health_check(&llm_client, &embedding_client).await?;
    info!("Health status: {}", health_status.status_message());

    if !health_status.overall_healthy {
        warn!(
            "Services are not fully healthy: {}",
            health_status.status_message()
        );
    }

    Ok(())
}

async fn test_hybrid_services(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
    info!("Creating hybrid services (OpenAI + EmbeddingGemma)...");

    let config = ServiceConfig {
        llm: LLMServiceConfig::OpenAI(graphiti_llm::OpenAIConfig {
            api_key,
            ..Default::default()
        }),
        embedding: EmbeddingServiceConfig::Generic(graphiti_llm::EmbedderConfig {
            provider: graphiti_llm::EmbeddingProvider::EmbedAnything,
            api_key: String::new(),
            model: "google/embeddinggemma-300m".to_string(),
            dimension: 768,
            batch_size: 8,
            timeout: std::time::Duration::from_secs(60),
            device: Some("auto".to_string()),
            max_length: Some(8192),
            cache_dir: None,
        }),
    };

    let (llm_client, embedding_client) = ServiceFactory::create_services(&config).await?;

    // Test LLM completion with OpenAI
    info!("Testing OpenAI LLM completion...");
    let messages = vec![Message::user(
        "What is 2+2? Please respond with just the number.",
    )];
    let params = CompletionParams {
        max_tokens: Some(10),
        temperature: Some(0.1),
        ..Default::default()
    };

    match llm_client.complete(&messages, &params).await {
        Ok(response) => {
            info!("OpenAI Response: {}", response);
            if !response.contains("4") {
                warn!("Unexpected response from OpenAI: {}", response);
            }
        }
        Err(e) => {
            error!("OpenAI completion failed: {}", e);
            return Err(e.into());
        }
    }

    // Test embedding (EmbeddingGemma)
    info!("Testing EmbeddingGemma embedding with hybrid setup...");
    let test_text = "Hybrid service test".to_string();

    match embedding_client.embed(&test_text).await {
        Ok(embedding) => {
            info!("Generated embedding with dimension: {}", embedding.len());
            if embedding.is_empty() {
                return Err("Empty embedding generated".into());
            }
        }
        Err(e) => {
            error!("Embedding failed: {}", e);
            return Err(e.into());
        }
    }
    Ok(())
}
