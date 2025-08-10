.PHONY: all build test clean run docker-build docker-run format lint check bench

# Default target
all: build

# Build the project
build:
	cargo build --release

# Run tests
test:
	cargo test --all-features

# Clean build artifacts
clean:
	cargo clean
	rm -rf data/

# Run the MCP server
run:
	cargo run --bin graphiti-mcp-server

# Run with hot reload
dev:
	cargo watch -x run

# Docker operations
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f graphiti

# Docker deployment profiles
docker-full:
	@echo "🚀 Starting full Graphiti stack (CozoDB SQLite + Ollama + MCP Server)..."
	docker-compose up -d
	@echo "✅ Services started:"
	@echo "   - MCP Server: http://localhost:8080"
	@echo "   - Ollama API: http://localhost:11434"
	@echo "   - CozoDB: Embedded SQLite database"

docker-free:
	@echo "🆓 Starting free Graphiti stack (CozoDB Memory + Ollama + MCP Server)..."
	docker-compose --profile free up -d
	@echo "✅ Free services started:"
	@echo "   - MCP Server: http://localhost:8091"
	@echo "   - Ollama API: http://localhost:11434"
	@echo "   - CozoDB: In-memory database"

docker-monitoring:
	@echo "📊 Starting Graphiti with monitoring stack..."
	docker-compose --profile monitoring up -d
	@echo "✅ Services with monitoring started:"
	@echo "   - MCP Server: http://localhost:8080"
	@echo "   - Prometheus: http://localhost:9090"
	@echo "   - Grafana: http://localhost:3000 (admin/admin)"

docker-status:
	@echo "📋 Docker services status:"
	docker-compose ps

docker-health:
	@echo "🏥 Health check for all services:"
	@docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

docker-clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup completed"

docker-reset:
	@echo "🔄 Resetting all Docker data..."
	docker-compose down -v
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Reset completed"

# Ollama model management
ollama-setup:
	@echo "🦙 Setting up Ollama models..."
	docker-compose exec ollama ollama pull llama3.1:8b
	docker-compose exec ollama ollama pull nomic-embed-text
	@echo "✅ Ollama models ready"

ollama-models:
	@echo "📋 Available Ollama models:"
	docker-compose exec ollama ollama list

# Development helpers
docker-shell:
	docker-compose exec graphiti bash

docker-cozo-query:
	@echo "🔍 CozoDB is embedded - use the MCP server API to query:"
	@echo "   curl http://localhost:8080/health"
	@echo "   curl http://localhost:8080/memory/search?query=test"

docker-test:
	@echo "🧪 Running tests in Docker..."
	docker-compose exec graphiti cargo test

docker-dev:
	@echo "🔧 Starting development environment..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Code quality
format:
	cargo fmt

lint:
	cargo clippy -- -D warnings

check: format lint test

# Benchmarks
bench:
	cargo bench

# Documentation
docs:
	cargo doc --no-deps --open

# Security audit
audit:
	cargo audit

# Update dependencies
update:
	cargo update

# Setup development environment
setup:
	cp .env.example .env
	mkdir -p data/search data/vectors
	@echo "Setup complete. Edit .env with your API keys."

# Integration tests (requires services)
integration-test:
	docker-compose up -d neo4j
	@echo "Waiting for Neo4j to be ready..."
	@sleep 10
	NEO4J_URI=bolt://localhost:7687 cargo test --test integration_test
	docker-compose down