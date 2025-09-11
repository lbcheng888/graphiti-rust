use graphiti_llm::{EmbedAnythingClient, EmbedAnythingConfig, EmbeddingClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Prefer EMBEDDING_MODEL_DIR if set
    let cache_dir = std::env::var("EMBEDDING_MODEL_DIR").ok();
    let cfg = EmbedAnythingConfig {
        model_id: "google/embeddinggemma-300m".to_string(),
        batch_size: 2,
        max_length: 4096,
        device: "auto".to_string(),
        cache_dir,
        target_dim: Some(768),
    };
    let client = EmbedAnythingClient::new(cfg).await?;
    let text = "EmbeddingGemma quick check".to_string();
    let v = client.embed(&text).await?;
    println!("ok, dim={} first3={:?}", v.len(), &v[..3.min(v.len())]);
    Ok(())
}
