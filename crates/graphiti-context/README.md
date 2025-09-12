Graphiti Context
================

This crate integrates codebase scanning, chunking, local JSONL storage, and Qdrant-backed vector search previously provided in `claude-context-rust`. It exposes a lightweight `Context` API for indexing and searching code snippets.

Key features:
- Ignore-aware project scanning with common defaults and custom overrides via env.
- Line-based chunking with overlap for better retrieval.
- Pluggable embedding backends: hashing fallback, optional EmbedAnything (Qwen3), optional Candle-based Gemma.
- Storage backends: local JSONL or Qdrant (gRPC with REST fallback).

Environment variables:
- QDRANT_URL: enable Qdrant backend when set.
- USE_QWEN3=1: enable EmbedAnything/Qwen3 (requires feature `embed-anything`).
- USE_CANDLE_GEMMA=1 and EMBEDDING_MODEL_DIR: enable Candle Gemma (requires feature `candle-gemma`).

