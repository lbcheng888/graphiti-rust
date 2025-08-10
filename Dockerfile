# Build stage
FROM rust:1.80-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./
COPY rust-toolchain.toml ./
COPY crates/ ./crates/

# Build dependencies - this is the caching layer
RUN cargo build --release --bin graphiti-mcp-server && rm -rf target/release/deps/graphiti*

# Copy source code
COPY . .

# Build application
RUN cargo build --release --bin graphiti-mcp-server

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1001 graphiti

# Copy binary from builder
COPY --from=builder /app/target/release/graphiti-mcp-server /usr/local/bin/graphiti-mcp-server

# Copy configuration files
COPY config.toml /etc/graphiti/config.toml
COPY config.free.toml /etc/graphiti/config.free.toml

# Create data directories with proper permissions
RUN mkdir -p /var/lib/graphiti/search /var/lib/graphiti/vectors /var/lib/graphiti/data && \
    chown -R graphiti:graphiti /var/lib/graphiti

# Switch to non-root user
USER graphiti

# Expose port
EXPOSE 8080

# Set environment variables
ENV RUST_LOG=info,graphiti=debug \
    SEARCH_INDEX_PATH=/var/lib/graphiti/search \
    VECTOR_INDEX_PATH=/var/lib/graphiti/vectors

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the binary
ENTRYPOINT ["/usr/local/bin/graphiti-mcp-server"]
CMD ["--config", "/etc/graphiti/config.toml"]