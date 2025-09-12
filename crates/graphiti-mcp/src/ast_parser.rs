//! Advanced AST parsing for multiple programming languages

use crate::types::AddCodeEntityRequest;
use serde::Deserialize;
use serde::Serialize;
use std::path::Path;
use tracing::debug;

/// Advanced AST parser for extracting detailed code entities
#[allow(dead_code)]
pub struct AstParser {
    // Future: Could integrate with tree-sitter for robust parsing
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedEntity {
    pub entity_type: EntityType,
    pub name: String,
    pub full_signature: String,
    pub line_range: (u32, u32),
    pub visibility: Visibility,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<String>,
    pub docstring: Option<String>,
    pub complexity_score: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface,
    Trait,
    Module,
    Constant,
    Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Private,
    Protected,
    Internal,
    Package,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: Option<String>,
    pub default_value: Option<String>,
}

#[allow(dead_code)]
impl AstParser {
    pub fn new() -> Self {
        Self {}
    }

    /// Parse source code and extract detailed entities
    pub async fn parse_source(&self, content: &str, file_path: &Path) -> Vec<ParsedEntity> {
        let language = self.detect_language(file_path);

        match language.as_deref() {
            Some("Rust") => self.parse_rust(content),
            Some("Python") => self.parse_python(content),
            Some("JavaScript") | Some("TypeScript") => self.parse_javascript(content),
            Some("Java") => self.parse_java(content),
            Some("Go") => self.parse_go(content),
            _ => {
                debug!("No specific parser for language: {:?}", language);
                Vec::new()
            }
        }
    }

    /// Parse Rust source code
    fn parse_rust(&self, content: &str) -> Vec<ParsedEntity> {
        let mut entities = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_trimmed = line.trim();

            // Function parsing
            if line_trimmed.starts_with("pub fn ")
                || line_trimmed.starts_with("fn ")
                || line_trimmed.starts_with("async fn ")
                || line_trimmed.starts_with("pub async fn ")
            {
                if let Some(entity) = self.parse_rust_function(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }

            // Struct parsing
            if line_trimmed.starts_with("pub struct ") || line_trimmed.starts_with("struct ") {
                if let Some(entity) = self.parse_rust_struct(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }

            // Enum parsing
            if line_trimmed.starts_with("pub enum ") || line_trimmed.starts_with("enum ") {
                if let Some(entity) = self.parse_rust_enum(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }

            // Trait parsing
            if line_trimmed.starts_with("pub trait ") || line_trimmed.starts_with("trait ") {
                if let Some(entity) = self.parse_rust_trait(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }

            // Impl block methods
            if line_trimmed.starts_with("impl ") {
                let impl_entities = self.parse_rust_impl_block(&lines, line_idx);
                entities.extend(impl_entities);
            }
        }

        entities
    }

    /// Parse Python source code
    fn parse_python(&self, content: &str) -> Vec<ParsedEntity> {
        let mut entities = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_trimmed = line.trim();

            // Function parsing
            if line_trimmed.starts_with("def ") || line_trimmed.starts_with("async def ") {
                if let Some(entity) = self.parse_python_function(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }

            // Class parsing
            if line_trimmed.starts_with("class ") {
                if let Some(entity) = self.parse_python_class(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }
        }

        entities
    }

    /// Parse JavaScript/TypeScript source code
    fn parse_javascript(&self, content: &str) -> Vec<ParsedEntity> {
        let mut entities = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_trimmed = line.trim();

            // Function parsing
            if line_trimmed.starts_with("function ") || line_trimmed.contains("function ") {
                if let Some(entity) = self.parse_js_function(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }

            // Arrow function parsing
            if line_trimmed.contains(" => ") {
                if let Some(entity) = self.parse_js_arrow_function(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }

            // Class parsing
            if line_trimmed.starts_with("class ") || line_trimmed.starts_with("export class ") {
                if let Some(entity) = self.parse_js_class(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }

            // Interface parsing (TypeScript)
            if line_trimmed.starts_with("interface ")
                || line_trimmed.starts_with("export interface ")
            {
                if let Some(entity) = self.parse_ts_interface(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }
        }

        entities
    }

    /// Parse Java source code
    fn parse_java(&self, content: &str) -> Vec<ParsedEntity> {
        let mut entities = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_trimmed = line.trim();

            // Method parsing
            if self.is_java_method(line_trimmed) {
                if let Some(entity) = self.parse_java_method(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }

            // Class parsing
            if line_trimmed.contains("class ") && !line_trimmed.starts_with("//") {
                if let Some(entity) = self.parse_java_class(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }
        }

        entities
    }

    /// Parse Go source code
    fn parse_go(&self, content: &str) -> Vec<ParsedEntity> {
        let mut entities = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_trimmed = line.trim();

            // Function parsing
            if line_trimmed.starts_with("func ") {
                if let Some(entity) = self.parse_go_function(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }

            // Struct parsing
            if line_trimmed.contains("type ") && line_trimmed.contains(" struct") {
                if let Some(entity) = self.parse_go_struct(line_trimmed, &lines, line_idx) {
                    entities.push(entity);
                }
            }
        }

        entities
    }

    // Rust-specific parsing methods

    fn parse_rust_function(
        &self,
        line: &str,
        lines: &[&str],
        line_idx: usize,
    ) -> Option<ParsedEntity> {
        let visibility = if line.starts_with("pub ") {
            Visibility::Public
        } else {
            Visibility::Private
        };

        // Extract function name
        let fn_start = line.find("fn ")?;
        let name_start = fn_start + 3;
        let name_end = line[name_start..].find('(')?;
        let name = line[name_start..name_start + name_end].trim().to_string();

        // Extract parameters
        let params_start = line.find('(')?;
        let params_end = line.find(')')?;
        let params_str = &line[params_start + 1..params_end];
        let parameters = self.parse_rust_parameters(params_str);

        // Extract return type
        let return_type = if line.contains(" -> ") {
            line.split(" -> ")
                .nth(1)?
                .split_whitespace()
                .next()
                .map(|s| s.to_string())
        } else {
            None
        };

        // Find docstring (/// comments above)
        let docstring = self.find_rust_docstring(lines, line_idx);

        // Calculate end line
        let end_line = self.find_rust_block_end(lines, line_idx);

        Some(ParsedEntity {
            entity_type: EntityType::Function,
            name,
            full_signature: line.to_string(),
            line_range: (line_idx as u32 + 1, end_line),
            visibility,
            parameters,
            return_type,
            docstring,
            complexity_score: self.estimate_complexity(lines, line_idx, end_line as usize),
        })
    }

    fn parse_rust_struct(
        &self,
        line: &str,
        _lines: &[&str],
        line_idx: usize,
    ) -> Option<ParsedEntity> {
        let visibility = if line.starts_with("pub ") {
            Visibility::Public
        } else {
            Visibility::Private
        };

        let struct_start = line.find("struct ")?;
        let name_start = struct_start + 7;
        let name_end = line[name_start..]
            .find(|c: char| c.is_whitespace() || c == '<' || c == '{')
            .unwrap_or(line[name_start..].len());
        let name = line[name_start..name_start + name_end].trim().to_string();

        Some(ParsedEntity {
            entity_type: EntityType::Struct,
            name,
            full_signature: line.to_string(),
            line_range: (line_idx as u32 + 1, line_idx as u32 + 1),
            visibility,
            parameters: Vec::new(),
            return_type: None,
            docstring: None,
            complexity_score: 2,
        })
    }

    fn parse_rust_enum(
        &self,
        line: &str,
        _lines: &[&str],
        line_idx: usize,
    ) -> Option<ParsedEntity> {
        let visibility = if line.starts_with("pub ") {
            Visibility::Public
        } else {
            Visibility::Private
        };

        let enum_start = line.find("enum ")?;
        let name_start = enum_start + 5;
        let name_end = line[name_start..]
            .find(|c: char| c.is_whitespace() || c == '<' || c == '{')
            .unwrap_or(line[name_start..].len());
        let name = line[name_start..name_start + name_end].trim().to_string();

        Some(ParsedEntity {
            entity_type: EntityType::Enum,
            name,
            full_signature: line.to_string(),
            line_range: (line_idx as u32 + 1, line_idx as u32 + 1),
            visibility,
            parameters: Vec::new(),
            return_type: None,
            docstring: None,
            complexity_score: 3,
        })
    }

    fn parse_rust_trait(
        &self,
        line: &str,
        _lines: &[&str],
        line_idx: usize,
    ) -> Option<ParsedEntity> {
        let visibility = if line.starts_with("pub ") {
            Visibility::Public
        } else {
            Visibility::Private
        };

        let trait_start = line.find("trait ")?;
        let name_start = trait_start + 6;
        let name_end = line[name_start..]
            .find(|c: char| c.is_whitespace() || c == '<' || c == '{')
            .unwrap_or(line[name_start..].len());
        let name = line[name_start..name_start + name_end].trim().to_string();

        Some(ParsedEntity {
            entity_type: EntityType::Trait,
            name,
            full_signature: line.to_string(),
            line_range: (line_idx as u32 + 1, line_idx as u32 + 1),
            visibility,
            parameters: Vec::new(),
            return_type: None,
            docstring: None,
            complexity_score: 4,
        })
    }

    fn parse_rust_impl_block(&self, lines: &[&str], start_idx: usize) -> Vec<ParsedEntity> {
        let mut entities = Vec::new();
        let mut current_idx = start_idx + 1;

        while current_idx < lines.len() {
            let line = lines[current_idx].trim();

            if line == "}" {
                break;
            }

            if line.starts_with("pub fn ") || line.starts_with("fn ") {
                if let Some(entity) = self.parse_rust_function(line, lines, current_idx) {
                    let mut method_entity = entity;
                    method_entity.entity_type = EntityType::Method;
                    entities.push(method_entity);
                }
            }

            current_idx += 1;
        }

        entities
    }

    // Python-specific parsing methods

    fn parse_python_function(
        &self,
        line: &str,
        lines: &[&str],
        line_idx: usize,
    ) -> Option<ParsedEntity> {
        let name_start = line.find("def ")? + 4;
        let name_end = line[name_start..].find('(')?;
        let name = line[name_start..name_start + name_end].trim().to_string();

        // Extract parameters
        let params_start = line.find('(')?;
        let params_end = line.find(')')?;
        let params_str = &line[params_start + 1..params_end];
        let parameters = self.parse_python_parameters(params_str);

        // Extract return type annotation
        let return_type = if line.contains(" -> ") {
            line.split(" -> ")
                .nth(1)?
                .split(':')
                .next()?
                .trim()
                .to_string()
                .into()
        } else {
            None
        };

        // Find docstring
        let docstring = self.find_python_docstring(lines, line_idx);

        // Calculate end line
        let end_line = self.find_python_block_end(lines, line_idx);

        Some(ParsedEntity {
            entity_type: EntityType::Function,
            name,
            full_signature: line.to_string(),
            line_range: (line_idx as u32 + 1, end_line),
            visibility: Visibility::Public, // Python doesn't have explicit visibility
            parameters,
            return_type,
            docstring,
            complexity_score: self.estimate_complexity(lines, line_idx, end_line as usize),
        })
    }

    fn parse_python_class(
        &self,
        line: &str,
        _lines: &[&str],
        line_idx: usize,
    ) -> Option<ParsedEntity> {
        let name_start = line.find("class ")? + 6;
        let name_end = line[name_start..]
            .find(|c: char| c == ':' || c == '(' || c.is_whitespace())
            .unwrap_or(line[name_start..].len());
        let name = line[name_start..name_start + name_end].trim().to_string();

        Some(ParsedEntity {
            entity_type: EntityType::Class,
            name,
            full_signature: line.to_string(),
            line_range: (line_idx as u32 + 1, line_idx as u32 + 1),
            visibility: Visibility::Public,
            parameters: Vec::new(),
            return_type: None,
            docstring: None,
            complexity_score: 5,
        })
    }

    // Helper methods

    fn parse_rust_parameters(&self, params_str: &str) -> Vec<Parameter> {
        if params_str.trim().is_empty() {
            return Vec::new();
        }

        params_str
            .split(',')
            .filter_map(|param| {
                let param = param.trim();
                if param == "&self" || param == "self" || param == "&mut self" {
                    return None;
                }

                let parts: Vec<&str> = param.split(':').collect();
                if parts.len() >= 2 {
                    Some(Parameter {
                        name: parts[0].trim().to_string(),
                        param_type: Some(parts[1].trim().to_string()),
                        default_value: None,
                    })
                } else {
                    Some(Parameter {
                        name: param.to_string(),
                        param_type: None,
                        default_value: None,
                    })
                }
            })
            .collect()
    }

    fn parse_python_parameters(&self, params_str: &str) -> Vec<Parameter> {
        if params_str.trim().is_empty() {
            return Vec::new();
        }

        params_str
            .split(',')
            .filter_map(|param| {
                let param = param.trim();
                if param == "self" {
                    return None;
                }

                // Handle default values
                let (name_type, default) = if param.contains('=') {
                    let parts: Vec<&str> = param.split('=').collect();
                    (parts[0].trim(), Some(parts[1].trim().to_string()))
                } else {
                    (param, None)
                };

                // Handle type annotations
                let (name, param_type) = if name_type.contains(':') {
                    let parts: Vec<&str> = name_type.split(':').collect();
                    (
                        parts[0].trim().to_string(),
                        Some(parts[1].trim().to_string()),
                    )
                } else {
                    (name_type.to_string(), None)
                };

                Some(Parameter {
                    name,
                    param_type,
                    default_value: default,
                })
            })
            .collect()
    }

    fn find_rust_docstring(&self, lines: &[&str], line_idx: usize) -> Option<String> {
        let mut docstring_lines = Vec::new();
        let mut idx = line_idx;

        // Look backwards for /// comments
        while idx > 0 {
            idx -= 1;
            let line = lines[idx].trim();
            if line.starts_with("///") {
                docstring_lines.insert(0, line[3..].trim().to_string());
            } else if !line.is_empty() {
                break;
            }
        }

        if docstring_lines.is_empty() {
            None
        } else {
            Some(docstring_lines.join(" "))
        }
    }

    fn find_python_docstring(&self, lines: &[&str], line_idx: usize) -> Option<String> {
        // Look for triple-quoted docstrings after function definition
        if line_idx + 1 < lines.len() {
            let next_line = lines[line_idx + 1].trim();
            if next_line.starts_with("\"\"\"") || next_line.starts_with("'''") {
                // Simple single-line docstring
                if next_line.len() > 6 && next_line.ends_with(&next_line[..3]) {
                    return Some(next_line[3..next_line.len() - 3].trim().to_string());
                }
                // Multi-line docstring would require more complex parsing
            }
        }
        None
    }

    fn find_rust_block_end(&self, lines: &[&str], start_idx: usize) -> u32 {
        let mut brace_count = 0;
        let mut found_open_brace = false;

        for (idx, line) in lines.iter().enumerate().skip(start_idx) {
            for ch in line.chars() {
                match ch {
                    '{' => {
                        brace_count += 1;
                        found_open_brace = true;
                    }
                    '}' => {
                        brace_count -= 1;
                        if found_open_brace && brace_count == 0 {
                            return idx as u32 + 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        start_idx as u32 + 1
    }

    fn find_python_block_end(&self, lines: &[&str], start_idx: usize) -> u32 {
        let initial_indent = self.get_indent_level(lines[start_idx]);

        for (idx, line) in lines.iter().enumerate().skip(start_idx + 1) {
            if !line.trim().is_empty() {
                let current_indent = self.get_indent_level(line);
                if current_indent <= initial_indent {
                    return idx as u32;
                }
            }
        }

        lines.len() as u32
    }

    fn get_indent_level(&self, line: &str) -> usize {
        line.len() - line.trim_start().len()
    }

    fn estimate_complexity(&self, lines: &[&str], start_idx: usize, end_idx: usize) -> u8 {
        let mut complexity: u32 = 1;

        for line in lines.iter().take(end_idx).skip(start_idx) {
            let line_lower = line.to_lowercase();

            // Control flow statements increase complexity
            if line_lower.contains("if ")
                || line_lower.contains("elif ")
                || line_lower.contains("else if ")
            {
                complexity = complexity.saturating_add(1);
            }
            if line_lower.contains("for ") || line_lower.contains("while ") {
                complexity = complexity.saturating_add(2);
            }
            if line_lower.contains("match ") || line_lower.contains("switch ") {
                complexity = complexity.saturating_add(2);
            }
            if line_lower.contains("try ")
                || line_lower.contains("catch ")
                || line_lower.contains("except ")
            {
                complexity = complexity.saturating_add(1);
            }
        }

        (complexity.min(10)) as u8
    }

    // Placeholder methods for other languages
    fn parse_js_function(
        &self,
        _line: &str,
        _lines: &[&str],
        _line_idx: usize,
    ) -> Option<ParsedEntity> {
        None
    }
    fn parse_js_arrow_function(
        &self,
        _line: &str,
        _lines: &[&str],
        _line_idx: usize,
    ) -> Option<ParsedEntity> {
        None
    }
    fn parse_js_class(
        &self,
        _line: &str,
        _lines: &[&str],
        _line_idx: usize,
    ) -> Option<ParsedEntity> {
        None
    }
    fn parse_ts_interface(
        &self,
        _line: &str,
        _lines: &[&str],
        _line_idx: usize,
    ) -> Option<ParsedEntity> {
        None
    }
    fn is_java_method(&self, _line: &str) -> bool {
        false
    }
    fn parse_java_method(
        &self,
        _line: &str,
        _lines: &[&str],
        _line_idx: usize,
    ) -> Option<ParsedEntity> {
        None
    }
    fn parse_java_class(
        &self,
        _line: &str,
        _lines: &[&str],
        _line_idx: usize,
    ) -> Option<ParsedEntity> {
        None
    }
    fn parse_go_function(
        &self,
        _line: &str,
        _lines: &[&str],
        _line_idx: usize,
    ) -> Option<ParsedEntity> {
        None
    }
    fn parse_go_struct(
        &self,
        _line: &str,
        _lines: &[&str],
        _line_idx: usize,
    ) -> Option<ParsedEntity> {
        None
    }

    fn detect_language(&self, file_path: &Path) -> Option<String> {
        let ext = file_path.extension()?.to_str()?;
        match ext {
            "rs" => Some("Rust".to_string()),
            "py" => Some("Python".to_string()),
            "js" => Some("JavaScript".to_string()),
            "ts" | "tsx" => Some("TypeScript".to_string()),
            "java" => Some("Java".to_string()),
            "go" => Some("Go".to_string()),
            _ => None,
        }
    }
}

impl Default for AstParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert ParsedEntity to AddCodeEntityRequest
impl From<ParsedEntity> for AddCodeEntityRequest {
    fn from(entity: ParsedEntity) -> Self {
        AddCodeEntityRequest {
            entity_type: format!("{:?}", entity.entity_type),
            name: entity.name,
            description: entity.docstring.unwrap_or_else(|| {
                format!("Auto-detected {:?} from AST parsing", entity.entity_type)
            }),
            file_path: None, // Set by caller
            line_range: Some(entity.line_range),
            language: None, // Set by caller
            framework: None,
            complexity: Some(entity.complexity_score),
            importance: None,
        }
    }
}
