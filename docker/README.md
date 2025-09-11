# Graphiti Rust MCP Server - Docker Deployment

This directory contains Docker configuration for deploying the Graphiti Rust MCP Server, inspired by the [Graphiti Python MCP Server](https://github.com/getzep/graphiti/blob/main/mcp_server/README.md).

## Quick Start

### Option 1: Full Setup with CozoDB SQLite (Recommended for Production)

```bash
# Clone and navigate to the project
git clone https://github.com/getzep/graphiti.git
cd graphiti/graphiti-rust

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys (optional for Ollama setup)

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f graphiti
```

The server will be available at:
- **MCP Server**: http://localhost:8080
- **Ollama API**: http://localhost:11434
- **CozoDB**: Embedded in the MCP server (no separate UI)

### Option 2: Free Setup (Ollama + CozoDB Memory)

For a completely free setup with in-memory database:

```bash
# Start only the free services
docker-compose --profile free up -d

# The free server runs on port 8091
curl http://localhost:8091/health
```

### Option 3: With Monitoring

To include Prometheus and Grafana monitoring:

```bash
# Start with monitoring stack
docker-compose --profile monitoring up -d
```

Additional services:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Configuration

### Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Key environment variables:

```env
# Required for cloud LLM providers (optional for Ollama)
OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Optional embedding providers
VOYAGE_API_KEY=your-voyage-api-key
COHERE_API_KEY=your-cohere-api-key

# Neo4j Configuration (for full setup)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Server Configuration
RUST_LOG=info,graphiti=debug
```

### Configuration Files

The Docker setup includes two configuration profiles:

1. **`config.toml`** - Full configuration with Neo4j
2. **`config.free.toml`** - Free configuration with Ollama + CozoDB

## Architecture

### Full Setup (CozoDB SQLite + Ollama)
```
┌─────────────────┐    ┌─────────────────┐
│   Graphiti MCP  │    │     Ollama      │
│     Server      │    │   (Local LLM)   │
│   (Port 8080)   │    │  (Port 11434)   │
│   + CozoDB      │    │   + Embeddings  │
│   (SQLite)      │    │                 │
└─────────────────┘    └─────────────────┘
```

### Free Setup (CozoDB Memory + Ollama)
```
┌─────────────────┐    ┌─────────────────┐
│   Graphiti MCP  │    │     Ollama      │
│     Server      │    │   (Local LLM)   │
│   (Port 8091)   │    │  (Port 11434)   │
│   + CozoDB      │    │   + Embeddings  │
│   (Memory)      │    │                 │
└─────────────────┘    └─────────────────┘
```

## Available Endpoints

### Health Check
```bash
curl http://localhost:8080/health
```

### MCP Protocol
The server supports both stdio and SSE (Server-Sent Events) transports:

- **SSE Endpoint**: `http://localhost:8080/sse`
- **WebSocket**: `ws://localhost:8080/ws`

### REST API Examples

```bash
# Add an episode
curl -X POST http://localhost:8080/episodes \
  -H "Content-Type: application/json" \
  -d '{"content": "Alice met Bob at the conference", "source": "text"}'

# Search memories
curl "http://localhost:8080/memory/search?query=Alice&limit=10"

# Get entities
curl http://localhost:8080/entities
```

## Integration with MCP Clients

### Claude Desktop Configuration

For **stdio transport** (recommended):

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "command": "docker",
      "args": [
        "exec", "-i", "graphiti-mcp-server",
        "/usr/local/bin/graphiti-mcp-server",
        "--config", "/etc/graphiti/config.toml",
        "--stdio"
      ]
    }
  }
}
```

For **SSE transport** (requires mcp-remote):

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:8080/sse"]
    }
  }
}
```

### Cursor IDE Configuration

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "url": "http://localhost:8080/sse"
    }
  }
}
```

## Development

### Building Locally

```bash
# Build the Docker image
docker-compose build

# Run tests
docker-compose exec graphiti cargo test

# View logs
docker-compose logs -f graphiti

# Access container shell
docker-compose exec graphiti bash
```

### Hot Reload Development

For development with hot reload:

```bash
# Install cargo-watch
cargo install cargo-watch

# Run with hot reload
make dev
```

## Troubleshooting

### Common Issues

1. **Ollama models not available**:
   ```bash
   # Pull required models
   docker-compose exec ollama ollama pull llama3.1:8b
   docker-compose exec ollama ollama pull nomic-embed-text
   ```

2. **Neo4j connection issues**:
   ```bash
   # Check Neo4j status
   docker-compose logs neo4j
   
   # Verify connection
   docker-compose exec neo4j cypher-shell -u neo4j -p password "RETURN 1"
   ```

3. **Port conflicts**:
   ```bash
   # Check what's using the ports
   lsof -i :8080
   lsof -i :7474
   lsof -i :11434
   ```

### Logs and Debugging

```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f graphiti
docker-compose logs -f neo4j
docker-compose logs -f ollama

# Check service health
docker-compose ps
```

## Performance Tuning

### Neo4j Memory Settings

For production, adjust Neo4j memory in `docker-compose.yml`:

```yaml
environment:
  - NEO4J_dbms_memory_pagecache_size=2G
  - NEO4J_dbms_memory_heap_initial__size=2G
  - NEO4J_dbms_memory_heap_max__size=2G
```

### Ollama Performance

```bash
# Check available models
docker-compose exec ollama ollama list

# Monitor resource usage
docker stats
```

## Security Considerations

1. **Change default passwords** in production
2. **Use environment variables** for sensitive data
3. **Enable TLS** for production deployments
4. **Set `GRAPHITI_AUTH_TOKEN`** and keep it secret (required by default in production)

## Production Deployment

Use the production override file with hardening options:

```bash
# Ensure .env contains a strong GRAPHITI_AUTH_TOKEN
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

See also: `docs/PRODUCTION.md` for Kubernetes manifests and reverse proxy examples.
4. **Restrict network access** using Docker networks

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](../LICENSE) file for details.
