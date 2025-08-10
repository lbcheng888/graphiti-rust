//! End-to-end test for LLM and embedding service integration

use graphiti_llm::{
    CompletionParams, EmbeddingServiceConfig, LLMServiceConfig, Message, OllamaConfig,
    QwenLocalConfig, ServiceConfig, ServiceFactory,
};
use std::env;
use tokio;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Starting LLM and Embedding Service Integration Test");

    // Test 1: Create local services (Ollama + Qwen)
    info!("=== Test 1: Local Services (Ollama + Qwen) ===");

    match test_local_services().await {
        Ok(_) => info!("✅ Local services test passed"),
        Err(e) => warn!("⚠️ Local services test failed: {}", e),
    }

    // Test 2: Create hybrid services (OpenAI + Qwen) if API key is available
    info!("=== Test 2: Hybrid Services (OpenAI + Qwen) ===");

    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        match test_hybrid_services(api_key).await {
            Ok(_) => info!("✅ Hybrid services test passed"),
            Err(e) => warn!("⚠️ Hybrid services test failed: {}", e),
        }
    } else {
        warn!("⚠️ OPENAI_API_KEY not set, skipping hybrid services test");
    }

    // Test 3: Test Qwen embedding server directly
    info!("=== Test 3: Qwen Embedding Server ===");

    match test_qwen_embedding_server().await {
        Ok(_) => info!("✅ Qwen embedding server test passed"),
        Err(e) => warn!("⚠️ Qwen embedding server test failed: {}", e),
    }

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
        embedding: EmbeddingServiceConfig::QwenLocal(QwenLocalConfig {
            model_name: "Qwen/Qwen3-0.6B-Base".to_string(),
            server_url: Some("http://localhost:8001".to_string()),
            batch_size: 4,
            ..Default::default()
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
    info!("Creating hybrid services (OpenAI + Qwen)...");

    let (llm_client, embedding_client) = ServiceFactory::create_hybrid_services(api_key).await?;

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

    // Test Qwen embedding (same as before)
    info!("Testing Qwen embedding with hybrid setup...");
    let test_text = "Hybrid service test".to_string();

    match embedding_client.embed(&test_text).await {
        Ok(embedding) => {
            info!("Generated embedding with dimension: {}", embedding.len());
            if embedding.is_empty() {
                return Err("Empty embedding generated".into());
            }
        }
        Err(e) => {
            error!("Qwen embedding failed: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

async fn test_qwen_embedding_server() -> Result<(), Box<dyn std::error::Error>> {
    info!("Testing Qwen embedding server directly...");

    let client = reqwest::Client::new();
    let server_url = "http://localhost:8001";

    // Test health endpoint
    info!("Checking server health...");
    let health_response = client.get(&format!("{}/health", server_url)).send().await?;

    if !health_response.status().is_success() {
        return Err("Qwen server health check failed".into());
    }

    let health_data: serde_json::Value = health_response.json().await?;
    info!("Server health: {:?}", health_data);

    // Test embedding endpoint
    info!("Testing embedding endpoint...");
    let request_data = serde_json::json!({
        "texts": ["Hello", "World", "Test"],
        "model": "Qwen/Qwen3-0.6B-Base",
        "normalize": true,
        "max_length": 512
    });

    let embed_response = client
        .post(&format!("{}/embed", server_url))
        .header("Content-Type", "application/json")
        .json(&request_data)
        .send()
        .await?;

    if !embed_response.status().is_success() {
        let error_text = embed_response.text().await?;
        return Err(format!("Embedding request failed: {}", error_text).into());
    }

    let embed_data: serde_json::Value = embed_response.json().await?;
    info!(
        "Embedding response: model = {}, dimension = {}",
        embed_data["model"], embed_data["dimension"]
    );

    let embeddings = embed_data["embeddings"]
        .as_array()
        .ok_or("No embeddings in response")?;

    if embeddings.len() != 3 {
        return Err("Expected 3 embeddings".into());
    }

    for (i, embedding) in embeddings.iter().enumerate() {
        let embedding_vec = embedding.as_array().ok_or("Embedding is not an array")?;
        info!("Embedding {}: dimension = {}", i, embedding_vec.len());
    }

    Ok(())
}
