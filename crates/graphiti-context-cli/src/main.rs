use anyhow::Result;
use clap::{Parser, Subcommand};
use graphiti_context::{Context, ContextConfig};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "graphiti-context",
    version,
    about = "Graphiti code indexer & search",
    author
)]
struct Cli {
    /// Backend: auto|local|qdrant
    #[arg(long, value_parser = ["auto", "local", "qdrant"], default_value = "auto")]
    backend: String,
    /// Qdrant URL (e.g. http://127.0.0.1:6334)
    #[arg(long)]
    qdrant_url: Option<String>,
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Index {
        path: PathBuf,
        #[arg(long)]
        force: bool,
    },
    Search {
        path: PathBuf,
        #[arg(short, long)]
        query: String,
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
    },
    Status {
        path: PathBuf,
    },
    Clear {
        path: PathBuf,
    },
    Doctor {
        #[arg(long)]
        path: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();
    let cfg = build_config(&cli)?;
    match cli.cmd {
        Commands::Index { path, force } => cmd_index_with(&path, force, cfg),
        Commands::Search { path, query, top_k } => cmd_search_with(&path, &query, top_k, cfg),
        Commands::Status { path } => cmd_status_with(&path, cfg),
        Commands::Clear { path } => cmd_clear_with(&path, cfg),
        Commands::Doctor { path } => cmd_doctor(path, cfg),
    }
}

fn build_config(cli: &Cli) -> Result<ContextConfig> {
    let mut cfg = ContextConfig::default();
    cfg.use_qdrant = match cli.backend.as_str() {
        "local" => false,
        "qdrant" => true,
        _ => std::env::var("QDRANT_URL").is_ok(),
    };
    if let Some(url) = &cli.qdrant_url {
        cfg.qdrant_url = Some(url.clone());
        cfg.use_qdrant = true;
    }
    Ok(cfg)
}

fn cmd_index_with(path: &PathBuf, force: bool, cfg: ContextConfig) -> Result<()> {
    let ctx = Context::new(Some(cfg))?;
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>3}% {msg}",
        )?
        .progress_chars("##-"),
    );

    let (files, chunks) = ctx.index_codebase(
        path,
        Some(|p: graphiti_context::types::IndexProgress| {
            use graphiti_context::types::ProgressPhase::*;
            let (msg, pct) = match p.phase {
                Preparing => ("Preparing collection...", p.percentage),
                Scanning => ("Scanning files...", p.percentage),
                Indexing => ("Indexing files...", p.percentage),
                Completed => ("Index completed", 100),
            };
            pb.set_position(pct);
            pb.set_message(msg);
        }),
        force,
    )?;
    pb.finish_with_message("Done");
    println!("Indexed {} files, {} chunks", files, chunks);
    Ok(())
}

fn cmd_search_with(path: &PathBuf, query: &str, top_k: usize, cfg: ContextConfig) -> Result<()> {
    let ctx = Context::new(Some(cfg))?;
    let results = ctx.search(path, query, top_k)?;
    if results.is_empty() {
        println!("No results");
        return Ok(());
    }
    for (i, r) in results.iter().enumerate() {
        println!(
            "{}. {}:{}-{}  score={:.2}\n{}\n---",
            i + 1,
            r.relative_path,
            r.start_line,
            r.end_line,
            r.score,
            trim(&r.content)
        );
    }
    Ok(())
}

fn cmd_status_with(path: &PathBuf, cfg: ContextConfig) -> Result<()> {
    let ctx = Context::new(Some(cfg))?;
    let name = ctx.collection_name(path);
    let has = ctx.has_index(path)?;
    let dir = ctx.collection_dir(path)?;
    println!("Collection: {}", name);
    println!("Exists: {}", has);
    println!("Location: {}", dir.display());
    Ok(())
}

fn cmd_clear_with(path: &PathBuf, cfg: ContextConfig) -> Result<()> {
    let ctx = Context::new(Some(cfg))?;
    ctx.clear_index(path)?;
    println!("Cleared index for {}", path.display());
    Ok(())
}

fn trim(s: &str) -> String {
    let s = s.trim();
    if s.len() > 500 {
        format!("{}...", &s[..500])
    } else {
        s.to_string()
    }
}

fn cmd_doctor(path: Option<PathBuf>, cfg: ContextConfig) -> Result<()> {
    let ctx = Context::new(Some(cfg))?;
    let test_path = path.unwrap_or(std::env::current_dir()?);
    println!("Embedder: {}", ctx.get_embedder_name());
    println!("Embedder dim: {}", ctx.get_embedder_dim());
    if ctx.is_qdrant_enabled() {
        let name = ctx.collection_name(&test_path);
        let ok = ctx.has_index(&test_path).unwrap_or(false);
        println!(
            "Qdrant: enabled at {:?} — collection '{}' exists: {}",
            std::env::var("QDRANT_URL").ok(),
            name,
            ok
        );
    } else {
        println!("Backend: local JSONL (fallback)");
    }
    Ok(())
}
