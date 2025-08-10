# Graphiti MCP Server

A Model Context Protocol (MCP) server for Graphiti, providing temporal knowledge graph capabilities for AI assistants.

## Features

- **Entity Extraction**: Automatically extract entities and relationships from text using LLM
- **Semantic Search**: Use embeddings for semantic similarity search
- **Temporal Awareness**: Track when events occurred and when information is valid
- **Free Stack Support**: Works with Ollama (local LLM) and HuggingFace (free embeddings)

## Prerequisites

1. **Ollama** - For local LLM inference
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull a model
   ollama pull llama3.2:latest
   ```

2. **Rust** - For building the server
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

## Configuration

Edit `config.toml` to configure the server:

```toml
[llm]
provider = "ollama"
model = "llama3.2:latest"
base_url = "http://localhost:11434"

[embedder]
provider = "huggingface"
model = "Qwen/Qwen3-Embedding-8B"
```

## Running the Server

```bash
# From the graphiti-rust directory
cd crates/graphiti-mcp

# Run the server
cargo run --bin graphiti-mcp-server

# Or with custom config
cargo run --bin graphiti-mcp-server -- --config my-config.toml
```

## API Endpoints

### Add Memory
```bash
POST /memory
{
  "content": "Alice met Bob at the conference yesterday",
  "name": "Conference Meeting",
  "source": "user",
  "memory_type": "event"
}
```

### Search Memory
```bash
GET /memory/search?query=Alice&limit=10
```

### Get Memory by ID
```bash
GET /memory/{id}
```

### Get Related Memories
```bash
GET /memory/{id}/related?depth=2
```

## Using with MCP Clients

The server is compatible with any MCP client. Configure your client to connect to:
- Host: `localhost`
- Port: `8080`

## Environment Variables

- `LLM_PROVIDER`: Override the LLM provider (ollama, openai, groq, huggingface)
- `LLM_MODEL`: Override the LLM model
- `OPENAI_API_KEY`: API key for OpenAI (if using OpenAI)
- `GROQ_API_KEY`: API key for Groq (if using Groq)
- `HUGGINGFACE_API_KEY`: API key for HuggingFace (optional for free tier)

## Development

### Running Tests
```bash
cargo test
```

### Building for Production
```bash
cargo build --release
```

## Architecture

The MCP server uses:
- **CozoDB**: Embedded graph database for storing nodes and relationships
- **LLM**: For entity extraction and relationship detection
- **Embeddings**: For semantic search capabilities
- **Axum**: High-performance web framework

## Troubleshooting

1. **Ollama not running**: Make sure Ollama is installed and running (`ollama serve`)
2. **Model not found**: Pull the required model with `ollama pull <model-name>`
3. **Port already in use**: Change the port in `config.toml` or use `--port` flag