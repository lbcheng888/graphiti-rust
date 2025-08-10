//! Graphiti Prompts - Prompt engineering module for LLM interactions
//!
//! This crate provides structured prompts for various graph operations:
//! - Entity extraction from text
//! - Relationship extraction
//! - Node and edge deduplication
//! - Summarization and classification
//! - Edge invalidation

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod dedupe_edges;
pub mod dedupe_nodes;
pub mod extract_edges;
pub mod extract_nodes;
pub mod invalidate_edges;
pub mod models;
pub mod summarize_nodes;
pub mod templates;

use anyhow::Result;
use handlebars::Handlebars;
use serde::Serialize;
use std::collections::HashMap;

/// Prompt engine for managing and rendering templates
pub struct PromptEngine {
    handlebars: Handlebars<'static>,
}

impl PromptEngine {
    /// Create a new prompt engine with all templates registered
    pub fn new() -> Result<Self> {
        let mut handlebars = Handlebars::new();

        // Register all prompt templates
        handlebars.register_template_string(
            "extract_nodes_message",
            templates::EXTRACT_NODES_MESSAGE_TEMPLATE,
        )?;
        handlebars.register_template_string(
            "extract_nodes_json",
            templates::EXTRACT_NODES_JSON_TEMPLATE,
        )?;
        handlebars.register_template_string(
            "extract_nodes_text",
            templates::EXTRACT_NODES_TEXT_TEMPLATE,
        )?;
        handlebars
            .register_template_string("classify_nodes", templates::CLASSIFY_NODES_TEMPLATE)?;
        handlebars.register_template_string(
            "extract_attributes",
            templates::EXTRACT_ATTRIBUTES_TEMPLATE,
        )?;
        handlebars.register_template_string("extract_edges", templates::EXTRACT_EDGES_TEMPLATE)?;
        handlebars.register_template_string(
            "extract_edges_reflexion",
            templates::EXTRACT_EDGES_REFLEXION_TEMPLATE,
        )?;
        handlebars.register_template_string(
            "extract_edge_attributes",
            templates::EXTRACT_EDGE_ATTRIBUTES_TEMPLATE,
        )?;
        handlebars.register_template_string("dedupe_nodes", templates::DEDUPE_NODES_TEMPLATE)?;
        handlebars
            .register_template_string("dedupe_nodes_list", templates::DEDUPE_NODES_LIST_TEMPLATE)?;
        handlebars
            .register_template_string("dedupe_nodes_bulk", templates::DEDUPE_NODES_BULK_TEMPLATE)?;
        handlebars.register_template_string("dedupe_edges", templates::DEDUPE_EDGES_TEMPLATE)?;
        handlebars
            .register_template_string("summarize_nodes", templates::SUMMARIZE_NODES_TEMPLATE)?;
        handlebars
            .register_template_string("invalidate_edges", templates::INVALIDATE_EDGES_TEMPLATE)?;

        Ok(Self { handlebars })
    }

    /// Render a template with the given data
    pub fn render<T: Serialize>(&self, template_name: &str, data: &T) -> Result<String> {
        Ok(self.handlebars.render(template_name, data)?)
    }

    /// Generate entity extraction prompt for messages
    pub fn extract_nodes_message<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("extract_nodes_message", data)
    }

    /// Generate entity extraction prompt for JSON
    pub fn extract_nodes_json<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("extract_nodes_json", data)
    }

    /// Generate entity extraction prompt for text
    pub fn extract_nodes_text<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("extract_nodes_text", data)
    }

    /// Generate node classification prompt
    pub fn classify_nodes<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("classify_nodes", data)
    }

    /// Generate attribute extraction prompt
    pub fn extract_attributes<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("extract_attributes", data)
    }

    /// Generate edge extraction prompt
    pub fn extract_edges<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("extract_edges", data)
    }

    /// Generate edge reflexion prompt
    pub fn extract_edges_reflexion<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("extract_edges_reflexion", data)
    }

    /// Generate edge attribute extraction prompt
    pub fn extract_edge_attributes<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("extract_edge_attributes", data)
    }

    /// Generate node deduplication prompt
    pub fn dedupe_nodes<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("dedupe_nodes", data)
    }

    /// Generate node list deduplication prompt
    pub fn dedupe_nodes_list<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("dedupe_nodes_list", data)
    }

    /// Generate bulk node deduplication prompt
    pub fn dedupe_nodes_bulk<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("dedupe_nodes_bulk", data)
    }

    /// Generate edge deduplication prompt
    pub fn dedupe_edges<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("dedupe_edges", data)
    }

    /// Generate node summarization prompt
    pub fn summarize_nodes<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("summarize_nodes", data)
    }

    /// Generate edge invalidation prompt
    pub fn invalidate_edges<T: Serialize>(&self, data: &T) -> Result<String> {
        self.render("invalidate_edges", data)
    }
}

impl Default for PromptEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create prompt engine")
    }
}

/// Common prompt data structures
pub use models::*;
