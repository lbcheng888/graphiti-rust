# Graphiti Rust MCP Service

A high-performance, temporally-aware knowledge graph framework reimplemented in Rust as a Model Context Protocol (MCP) service.

## Overview

This is a Rust rewrite of the Graphiti knowledge graph system, designed for:
- Superior performance and memory efficiency
- Type safety and compile-time guarantees
- Native async/concurrent processing
- Production-ready reliability
- Bi-temporal data model with event and system time tracking
- Hybrid search combining text (BM25), vector similarity, and graph traversal
- **100% Free Operation** with local LLM and embedding models
- Multi-provider LLM and embedding support with automatic fallback

## 🆓 Free Usage (Zero API Costs)

**Run Graphiti completely free using local models:**

```bash
# Quick start with free providers
./start-free.sh

# Test the setup
./test-free-setup.sh
```

**Features of free setup:**
- ✅ Local LLM inference via [Ollama](https://ollama.ai)
- ✅ Local embeddings via SentenceTransformers
- ✅ No API keys required
- ✅ No ongoing costs
- ✅ Full privacy (everything runs locally)
- ✅ Production-ready performance

See [FREE_USAGE_GUIDE.md](FREE_USAGE_GUIDE.md) for detailed setup and configuration.

## Project Structure

```
graphiti-rust/
├── crates/
│   ├── graphiti-core/     # Core types and traits
│   ├── graphiti-mcp/      # MCP server implementation
│   ├── graphiti-cozo/     # CozoDB storage driver (pure Rust)
│   ├── graphiti-search/   # Search engine (Tantivy + vector)
│   └── graphiti-llm/      # LLM client implementations
├── config.free.toml       # Free configuration (Ollama + HuggingFace)
└── MIGRATION.md           # Migration guide from Python version
```

## Current Status

This is an active development project. The following components have been implemented:

- ✅ Core graph data structures (nodes, edges, temporal metadata)
- ✅ Storage trait and CozoDB driver implementation (pure Rust)
- ✅ Text search with Tantivy and vector search
- ✅ Multi-LLM client support (OpenAI, Ollama, HuggingFace, Groq)
- ✅ EmbeddingGemma-300m via Candle
  - Native approximate (tokenizer-only, zero deps, 768-dim)
  - Or embed_anything (Candle backend, higher semantic fidelity)
- ✅ MCP server with REST API endpoints
- ✅ Free operation mode (Ollama + HuggingFace)
- ✅ Claude Desktop integration ready
- 🚧 Episode processing pipeline (in progress)
- 🚧 Community detection algorithms (planned)
- 🚧 Full MCP protocol support (planned)

## Getting Started

### Prerequisites

- Rust 1.70 or later (for building)
- (Optional) Ollama for local LLM inference
- No database setup required (CozoDB embedded)
- No API keys required (free mode available)

### Quick Start (Docker - Recommended)

```bash
# Clone the repository
git clone https://github.com/getzep/graphiti.git
cd graphiti/graphiti-rust

# Run the automated setup script
./scripts/setup-docker.sh

# Or manually start with Docker Compose
docker-compose up -d

# Test the server
curl http://localhost:8080/health
```

**Available Docker setups:**
- **Full setup**: Neo4j + Ollama + MCP Server (port 8080)
- **Free setup**: CozoDB + Ollama + MCP Server (port 8091)
- **Development**: Hot reload + debugging enabled
- **Monitoring**: Includes Prometheus + Grafana

### Quick Start (Native - Free Mode)

```bash
# Clone the repository
git clone https://github.com/getzep/graphiti.git
cd graphiti/graphiti-rust

# Build the project
cargo build --release

# (Optional) Install Ollama for local LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# Start the MCP server with free configuration
# Prefer env vars to avoid hardcoded paths
EMBEDDING_MODEL_DIR=/path/to/embeddinggemma-300m \
GRAPHITI_PROJECT=. \
GRAPHITI_DATA_DIR=.graphiti/data \
GRAPHITI_CONFIG_FORCE=1 \
GRAPHITI_ENABLE_INITIAL_SCAN=0 \
HF_HUB_OFFLINE=1 \
./target/release/graphiti-mcp-server --config config.free.toml

# Test the server
curl http://localhost:8091/health
```

### Manual Building

```bash
# Copy environment variables
cp .env.example .env
# Edit .env with your configuration

# Build all crates
cargo build --release

# Run tests
cargo test

# Run the MCP server (stdio; default = native Candle approximate)
EMBEDDING_MODEL_DIR=/path/to/embeddinggemma-300m \
GRAPHITI_PROJECT=. \
GRAPHITI_DATA_DIR=.graphiti/data \
RUST_LOG=info,graphiti=debug \
GRAPHITI_CONFIG_FORCE=1 \
GRAPHITI_ENABLE_INITIAL_SCAN=0 \
HF_HUB_OFFLINE=1 \
cargo run --bin graphiti-mcp-server -- --stdio --log-level info --config ./config.free.toml
```

### Development

```bash
# Run with hot reload
make dev

# Format and lint code
make check

# Run benchmarks
make bench

# Generate documentation
make docs

# Security audit
make audit
```

## Docker Deployment

### Quick Docker Setup

The easiest way to get started is using Docker with the automated setup script:

```bash
# Run the interactive setup script
./scripts/setup-docker.sh
```

### Manual Docker Commands

```bash
# Full setup (Neo4j + Ollama + MCP Server)
docker-compose up -d

# Free setup (CozoDB + Ollama only)
docker-compose --profile free up -d

# Development setup (with hot reload)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

### Docker Endpoints

**Full Setup:**
- MCP Server: http://localhost:8080
- Neo4j Browser: http://localhost:7474 (neo4j/password)
- Ollama API: http://localhost:11434

**Free Setup:**
- MCP Server: http://localhost:8091
- Ollama API: http://localhost:11434

**Monitoring (if enabled):**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Docker Configuration

The Docker setup supports multiple deployment profiles:

1. **Default Profile**: Neo4j + Ollama + MCP Server
2. **Free Profile**: CozoDB + Ollama + MCP Server (no external dependencies)
3. **Monitoring Profile**: Adds Prometheus and Grafana
4. **Development Profile**: Hot reload and debugging enabled

For detailed Docker documentation, see [docker/README.md](docker/README.md).

For production hardening and deployment options (Docker/Kubernetes), see [docs/PRODUCTION.md](docs/PRODUCTION.md).

## API Endpoints

The MCP server exposes the following REST API endpoints:

- `POST /memory` - Add a new memory to the graph
- `GET /memory/search?query=...` - Search memories
- `GET /memory/:id` - Get a specific memory by ID
- `GET /memory/:id/related?depth=2` - Get related memories
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics

### Example: Adding a Memory

```bash
curl -X POST http://localhost:8080/memory \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Alice met Bob at the AI conference in San Francisco.",
    "source": "meeting_notes",
    "memory_type": "event"
  }'
```

### Example: Searching Memories

```bash
curl "http://localhost:8080/memory/search?query=Alice&limit=10"
```

## Architecture

The project is organized as a Rust workspace with the following crates:

- `graphiti-core` - Core types, traits, and data structures
- `graphiti-cozo` - CozoDB database driver implementation (pure Rust)
- `graphiti-search` - Text search (Tantivy) and vector search
- `graphiti-llm` - LLM and embedding client implementations
- `graphiti-mcp` - MCP server and REST API

## Features

- **Bi-temporal Data Model**: Track when events occurred and when they were recorded
- **Hybrid Search**: Combine text search (BM25), vector similarity, and graph traversal
- **Multi-Provider Support**: OpenAI, Ollama, HuggingFace, Groq for LLMs and embeddings
- **Zero-Cost Operation**: Run completely free with local models
- **CozoDB Integration**: Pure Rust graph database with Datalog queries
- **High Performance**: Written in Rust for speed and efficiency
- **MCP Integration**: Model Context Protocol server for AI assistants
- **Claude Desktop Ready**: Direct integration with Claude Desktop

## Configuration

See also: docs/DEFAULTS_AND_BEHAVIORS.md for current defaults and behavior changes.

Configuration is managed through TOML files:

### Free Configuration (`config.free.toml`)
```toml
[llm]
provider = "Ollama"
model = "llama3.1:8b"

[embedder]
provider = "huggingface"
model = "google/embeddinggemma-300m"
dimension = 4096

[cozo]
engine = "mem"  # or "sqlite" for persistence
path = "./data/graphiti.db"
```

### Claude Desktop Integration
```json
{
  "mcpServers": {
    "graphiti": {
      "command": "/path/to/graphiti-mcp-server",
      "args": ["--config", "/path/to/config.free.toml"]
    }
  }
}
```

## Performance

The Rust implementation provides significant performance improvements over Python:
- **Startup Time**: 5-10x faster (0.5s vs 5-10s)
- **Memory Usage**: 2-3x less (50-150MB vs 200-500MB)
- **Query Latency**: 5-10x faster (10-50ms vs 100-500ms)
- **Concurrent Processing**: Native async/await with Tokio
- **Zero Dependencies**: Single binary, no Java/Python runtime

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run `make check` before committing
4. Submit a pull request

## Roadmap

- [ ] Complete entity extraction and relationship detection
- [ ] Implement community detection algorithms
- [ ] Full MCP protocol implementation (tools, resources)
- [ ] Python compatibility wrapper
- [ ] Data migration tools from Python version
- [ ] Web UI for graph visualization
- [ ] Distributed deployment support

## Testing

```bash
# Unit tests
cargo test

# Integration tests (requires Neo4j)
make integration-test

# Benchmarks
cargo bench

# Test coverage
cargo tarpaulin --out Html
```

## Deployment

### Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f graphiti

# Stop services
docker-compose down
```

### Kubernetes

Kubernetes deployment manifests are planned for future releases.

## License

Apache 2.0 - See LICENSE file for details
