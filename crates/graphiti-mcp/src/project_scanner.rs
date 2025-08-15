//! Automatic project source code scanning and knowledge graph generation

use crate::ast_parser::AstParser;
use crate::AddCodeEntityRequest;
use crate::AddMemoryRequest;
use crate::GraphitiService;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tracing::debug;
use tracing::info;
use tracing::warn;

/// Scans project source code and generates knowledge graph on first open
#[allow(dead_code)]
pub struct ProjectScanner {
    service: Arc<dyn GraphitiService>,
    ast_parser: AstParser,
    supported_extensions: HashSet<String>,
    scan_cache: tokio::sync::RwLock<ScanCache>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScanCache {
    /// Last scan timestamp for each directory
    last_scanned: HashMap<PathBuf, std::time::SystemTime>,
    /// File modification times from last scan
    file_mtimes: HashMap<PathBuf, std::time::SystemTime>,
    /// Detected project structure
    project_info: Option<ProjectInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
    pub root_path: PathBuf,
    pub language: ProjectLanguage,
    pub framework: Option<String>,
    pub package_files: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum ProjectLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Java,
    Go,
    Mixed(Vec<String>),
}

#[derive(Debug)]
pub struct ScanResult {
    /// Number of code entities added
    pub entities_added: usize,
    /// Number of memories added
    pub memories_added: usize,
    /// Number of files scanned
    pub files_scanned: usize,
    /// Project information discovered
    pub project_info: Option<ProjectInfo>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct FileChange {
    /// Path to the changed file
    pub path: PathBuf,
    /// Type of change that occurred
    pub change_type: ChangeType,
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum ChangeType {
    Created,
    Modified,
    Deleted,
}

#[allow(dead_code)]
impl ProjectScanner {
    pub fn new(service: Arc<dyn GraphitiService>) -> Self {
        let mut supported_extensions = HashSet::new();
        supported_extensions.extend(vec![
            "rs".to_string(),   // Rust
            "py".to_string(),   // Python
            "js".to_string(),   // JavaScript
            "ts".to_string(),   // TypeScript
            "tsx".to_string(),  // TypeScript React
            "jsx".to_string(),  // JavaScript React
            "java".to_string(), // Java
            "go".to_string(),   // Go
            "cpp".to_string(),  // C++
            "cc".to_string(),   // C++
            "c".to_string(),    // C
            "h".to_string(),    // C/C++ headers
            "hpp".to_string(),  // C++ headers
        ]);

        Self {
            service,
            ast_parser: AstParser::new(),
            supported_extensions,
            scan_cache: tokio::sync::RwLock::new(ScanCache {
                last_scanned: HashMap::new(),
                file_mtimes: HashMap::new(),
                project_info: None,
            }),
        }
    }

    /// Scan entire project from root directory (first-time or full rescan)
    pub async fn scan_project(
        &self,
        root_path: &Path,
    ) -> Result<ScanResult, Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting full project scan: {}", root_path.display());

        // Detect project structure first
        let project_info = self.detect_project_structure(root_path).await?;
        info!(
            "Detected project: {} ({})",
            project_info.name,
            format!("{:?}", project_info.language)
        );

        // Update cache with project info
        {
            let mut cache = self.scan_cache.write().await;
            cache.project_info = Some(project_info.clone());
        }

        // Add project-level memory
        let project_memory = AddMemoryRequest {
            content: format!(
                "Project '{}' detected at {}. Language: {:?}, Framework: {}",
                project_info.name,
                root_path.display(),
                project_info.language,
                project_info.framework.as_deref().unwrap_or("None")
            ),
            name: Some(format!("Project: {}", project_info.name)),
            source: Some("Project Scanner".to_string()),
            memory_type: Some("project_info".to_string()),
            metadata: Some(serde_json::json!({
                "project_name": project_info.name,
                "root_path": root_path.to_string_lossy(),
                "language": format!("{:?}", project_info.language),
                "framework": project_info.framework
            })),
            group_id: None,
            timestamp: None,
        };

        self.service.add_memory(project_memory).await.map_err(|e| {
            warn!("Failed to add project memory: {}", e);
            e
        })?;

        // Scan all source files
        let mut scan_result = ScanResult {
            entities_added: 0,
            memories_added: 1, // Project memory
            files_scanned: 0,
            project_info: Some(project_info.clone()),
        };

        self.scan_directory_recursive(root_path, &mut scan_result)
            .await?;

        info!(
            "Project scan complete: {} files, {} entities, {} memories",
            scan_result.files_scanned, scan_result.entities_added, scan_result.memories_added
        );

        Ok(scan_result)
    }

    /// Incremental scan based on file changes
    pub async fn incremental_scan(
        &self,
        changes: &[FileChange],
    ) -> Result<ScanResult, Box<dyn std::error::Error + Send + Sync>> {
        debug!("🔄 Starting incremental scan for {} changes", changes.len());

        let mut scan_result = ScanResult {
            entities_added: 0,
            memories_added: 0,
            files_scanned: 0,
            project_info: self.scan_cache.read().await.project_info.clone(),
        };

        for change in changes {
            match change.change_type {
                ChangeType::Created | ChangeType::Modified => {
                    if self.should_scan_file(&change.path) {
                        self.scan_single_file(&change.path, &mut scan_result)
                            .await?;
                    }
                }
                ChangeType::Deleted => {
                    // Handle file deletion - remove related entities
                    debug!("File deleted: {}", change.path.display());
                    // In a full implementation, we'd remove entities related to this file
                }
            }
        }

        Ok(scan_result)
    }

    /// Detect project structure and metadata
    async fn detect_project_structure(
        &self,
        root_path: &Path,
    ) -> Result<ProjectInfo, Box<dyn std::error::Error + Send + Sync>> {
        let mut package_files = Vec::new();
        let mut detected_languages = HashSet::new();
        let mut framework = None;

        // Check for common project files
        let cargo_toml = root_path.join("Cargo.toml");
        let package_json = root_path.join("package.json");
        let pyproject_toml = root_path.join("pyproject.toml");
        let setup_py = root_path.join("setup.py");
        let go_mod = root_path.join("go.mod");
        let pom_xml = root_path.join("pom.xml");

        if cargo_toml.exists() {
            package_files.push(cargo_toml.clone());
            detected_languages.insert("Rust".to_string());

            // Try to read Cargo.toml for project name
            if let Ok(content) = fs::read_to_string(&cargo_toml).await {
                if let Some(name) = self.extract_cargo_name(&content) {
                    return Ok(ProjectInfo {
                        name,
                        root_path: root_path.to_path_buf(),
                        language: ProjectLanguage::Rust,
                        framework,
                        package_files,
                    });
                }
            }
        }

        if package_json.exists() {
            package_files.push(package_json.clone());
            detected_languages.insert("JavaScript".to_string());

            // Check for TypeScript
            if root_path.join("tsconfig.json").exists() {
                detected_languages.insert("TypeScript".to_string());
            }

            // Try to read package.json for project name and framework
            if let Ok(content) = fs::read_to_string(&package_json).await {
                if let Ok(pkg_data) = serde_json::from_str::<serde_json::Value>(&content) {
                    let name = pkg_data
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown-js-project")
                        .to_string();

                    // Detect framework
                    if let Some(deps) = pkg_data.get("dependencies").and_then(|d| d.as_object()) {
                        if deps.contains_key("react") {
                            framework = Some("React".to_string());
                        } else if deps.contains_key("vue") {
                            framework = Some("Vue".to_string());
                        } else if deps.contains_key("express") {
                            framework = Some("Express".to_string());
                        }
                    }

                    let language = if detected_languages.contains("TypeScript") {
                        ProjectLanguage::TypeScript
                    } else {
                        ProjectLanguage::JavaScript
                    };

                    return Ok(ProjectInfo {
                        name,
                        root_path: root_path.to_path_buf(),
                        language,
                        framework,
                        package_files,
                    });
                }
            }
        }

        if pyproject_toml.exists() || setup_py.exists() {
            if pyproject_toml.exists() {
                package_files.push(pyproject_toml);
            }
            if setup_py.exists() {
                package_files.push(setup_py);
            }
            detected_languages.insert("Python".to_string());

            // TODO: Extract project name from pyproject.toml or setup.py
            return Ok(ProjectInfo {
                name: root_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown-python-project")
                    .to_string(),
                root_path: root_path.to_path_buf(),
                language: ProjectLanguage::Python,
                framework,
                package_files,
            });
        }

        if go_mod.exists() {
            package_files.push(go_mod);
            detected_languages.insert("Go".to_string());

            return Ok(ProjectInfo {
                name: root_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown-go-project")
                    .to_string(),
                root_path: root_path.to_path_buf(),
                language: ProjectLanguage::Go,
                framework,
                package_files,
            });
        }

        if pom_xml.exists() {
            package_files.push(pom_xml);
            detected_languages.insert("Java".to_string());

            return Ok(ProjectInfo {
                name: root_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown-java-project")
                    .to_string(),
                root_path: root_path.to_path_buf(),
                language: ProjectLanguage::Java,
                framework,
                package_files,
            });
        }

        // Fallback: detect by scanning files
        let languages: Vec<String> = detected_languages.into_iter().collect();
        let language = if languages.len() == 1 {
            match languages[0].as_str() {
                "Rust" => ProjectLanguage::Rust,
                "Python" => ProjectLanguage::Python,
                "JavaScript" => ProjectLanguage::JavaScript,
                "TypeScript" => ProjectLanguage::TypeScript,
                "Java" => ProjectLanguage::Java,
                "Go" => ProjectLanguage::Go,
                _ => ProjectLanguage::Mixed(languages),
            }
        } else {
            ProjectLanguage::Mixed(languages)
        };

        Ok(ProjectInfo {
            name: root_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown-project")
                .to_string(),
            root_path: root_path.to_path_buf(),
            language,
            framework,
            package_files,
        })
    }

    /// Recursively scan directory for source files
    fn scan_directory_recursive<'a>(
        &'a self,
        dir_path: &'a Path,
        scan_result: &'a mut ScanResult,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let mut entries = fs::read_dir(dir_path).await?;

            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();

                if path.is_dir() {
                    // Skip common directories that shouldn't be scanned
                    if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                        if matches!(
                            dir_name,
                            "node_modules"
                                | "target"
                                | ".git"
                                | "build"
                                | "dist"
                                | "__pycache__"
                                | ".venv"
                                | "venv"
                        ) {
                            continue;
                        }
                    }

                    // Recursively scan subdirectory
                    self.scan_directory_recursive(&path, scan_result).await?;
                } else if self.should_scan_file(&path) {
                    self.scan_single_file(&path, scan_result).await?;
                }
            }

            Ok(())
        })
    }

    /// Scan a single source file and extract entities
    async fn scan_single_file(
        &self,
        file_path: &Path,
        scan_result: &mut ScanResult,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        debug!("📄 Scanning file: {}", file_path.display());

        let content = match fs::read_to_string(file_path).await {
            Ok(content) => content,
            Err(e) => {
                warn!("Failed to read file {}: {}", file_path.display(), e);
                return Ok(());
            }
        };

        scan_result.files_scanned += 1;

        // Extract basic file information
        let file_memory = AddMemoryRequest {
            content: format!(
                "Source file: {} ({} lines)",
                file_path.display(),
                content.lines().count()
            ),
            name: Some(format!(
                "File: {}",
                file_path.file_name().unwrap_or_default().to_string_lossy()
            )),
            source: Some("Project Scanner".to_string()),
            memory_type: Some("source_file".to_string()),
            metadata: Some(serde_json::json!({
                "file_path": file_path.to_string_lossy(),
                "line_count": content.lines().count(),
                "language": self.detect_file_language(file_path)
            })),
            group_id: None,
            timestamp: None,
        };

        self.service.add_memory(file_memory).await.map_err(|e| {
            warn!("Failed to add file memory: {}", e);
            e
        })?;
        scan_result.memories_added += 1;

        // Extract code entities using both simple patterns and AST parsing
        let mut entities = self.extract_code_entities(&content, file_path).await;

        // Enhanced extraction using AST parser
        let parsed_entities = self.ast_parser.parse_source(&content, file_path).await;
        for parsed_entity in parsed_entities {
            let mut entity_req: AddCodeEntityRequest = parsed_entity.into();
            entity_req.file_path = Some(file_path.to_string_lossy().to_string());
            entity_req.language = self.detect_file_language(file_path);
            entities.push(entity_req);
        }

        // Deduplicate entities by name and type
        entities.sort_by(|a, b| a.name.cmp(&b.name).then(a.entity_type.cmp(&b.entity_type)));
        entities.dedup_by(|a, b| a.name == b.name && a.entity_type == b.entity_type);

        for entity in entities {
            if let Err(e) = self.service.add_code_entity(entity).await {
                warn!("Failed to add code entity: {}", e);
            } else {
                scan_result.entities_added += 1;
            }
        }

        Ok(())
    }

    /// Extract code entities from file content
    async fn extract_code_entities(
        &self,
        content: &str,
        file_path: &Path,
    ) -> Vec<AddCodeEntityRequest> {
        let mut entities = Vec::new();

        // Simple pattern-based extraction (would be enhanced with proper AST parsing)
        let lines: Vec<&str> = content.lines().collect();

        for (line_no, line) in lines.iter().enumerate() {
            let line_trimmed = line.trim();

            // Function detection patterns
            if let Some(entity) = self.extract_function_entity(line_trimmed, file_path, line_no + 1)
            {
                entities.push(entity);
            }

            // Class detection patterns
            if let Some(entity) = self.extract_class_entity(line_trimmed, file_path, line_no + 1) {
                entities.push(entity);
            }

            // Struct detection (Rust, Go, C++)
            if let Some(entity) = self.extract_struct_entity(line_trimmed, file_path, line_no + 1) {
                entities.push(entity);
            }
        }

        entities
    }

    /// Extract function entity from line
    fn extract_function_entity(
        &self,
        line: &str,
        file_path: &Path,
        line_no: usize,
    ) -> Option<AddCodeEntityRequest> {
        // Rust: fn function_name
        if line.starts_with("fn ") || line.contains(" fn ") {
            if let Some(name) = self.extract_rust_function_name(line) {
                return Some(self.create_code_entity("Function", name, file_path, line_no));
            }
        }

        // Python: def function_name
        if line.starts_with("def ") {
            if let Some(name) = self.extract_python_function_name(line) {
                return Some(self.create_code_entity("Function", name, file_path, line_no));
            }
        }

        // JavaScript/TypeScript: function function_name or const function_name =
        if line.contains("function ") || (line.contains("const ") && line.contains(" = ")) {
            if let Some(name) = self.extract_js_function_name(line) {
                return Some(self.create_code_entity("Function", name, file_path, line_no));
            }
        }

        None
    }

    /// Extract class entity from line
    fn extract_class_entity(
        &self,
        line: &str,
        file_path: &Path,
        line_no: usize,
    ) -> Option<AddCodeEntityRequest> {
        if line.starts_with("class ") || line.contains(" class ") {
            // Python/JavaScript/TypeScript/Java class detection
            if let Some(name) = self.extract_class_name(line) {
                return Some(self.create_code_entity("Class", name, file_path, line_no));
            }
        }

        None
    }

    /// Extract struct entity from line (Rust, Go, C++)
    fn extract_struct_entity(
        &self,
        line: &str,
        file_path: &Path,
        line_no: usize,
    ) -> Option<AddCodeEntityRequest> {
        if line.starts_with("struct ") || line.contains(" struct ") {
            if let Some(name) = self.extract_struct_name(line) {
                return Some(self.create_code_entity("Struct", name, file_path, line_no));
            }
        }

        None
    }

    // Helper methods for name extraction
    fn extract_rust_function_name(&self, line: &str) -> Option<String> {
        let start = line.find("fn ")? + 3;
        let rest = &line[start..];
        let end = rest.find(|c: char| c == '(' || c.is_whitespace())?;
        Some(rest[..end].to_string())
    }

    fn extract_python_function_name(&self, line: &str) -> Option<String> {
        let start = line.find("def ")? + 4;
        let rest = &line[start..];
        let end = rest.find('(')?;
        Some(rest[..end].to_string())
    }

    fn extract_js_function_name(&self, line: &str) -> Option<String> {
        if line.contains("function ") {
            let start = line.find("function ")? + 9;
            let rest = &line[start..];
            let end = rest.find(|c: char| c == '(' || c.is_whitespace())?;
            Some(rest[..end].to_string())
        } else if line.contains("const ") && line.contains(" = ") {
            let start = line.find("const ")? + 6;
            let rest = &line[start..];
            let end = rest.find(' ')?;
            Some(rest[..end].to_string())
        } else {
            None
        }
    }

    fn extract_class_name(&self, line: &str) -> Option<String> {
        let start = line.find("class ")? + 6;
        let rest = &line[start..];
        let end = rest.find(|c: char| c == ':' || c == '{' || c.is_whitespace())?;
        Some(rest[..end].to_string())
    }

    fn extract_struct_name(&self, line: &str) -> Option<String> {
        let start = line.find("struct ")? + 7;
        let rest = &line[start..];
        let end = rest.find(|c: char| c == '{' || c.is_whitespace())?;
        Some(rest[..end].to_string())
    }

    fn create_code_entity(
        &self,
        entity_type: &str,
        name: String,
        file_path: &Path,
        line_no: usize,
    ) -> AddCodeEntityRequest {
        AddCodeEntityRequest {
            entity_type: entity_type.to_string(),
            name,
            description: format!(
                "Automatically detected {} from source code",
                entity_type.to_lowercase()
            ),
            file_path: Some(file_path.to_string_lossy().to_string()),
            line_range: Some((line_no as u32, line_no as u32)),
            language: self.detect_file_language(file_path),
            framework: None,
            complexity: None,
            importance: None,
        }
    }

    /// Extract project name from Cargo.toml content
    fn extract_cargo_name(&self, content: &str) -> Option<String> {
        for line in content.lines() {
            if line.trim_start().starts_with("name") && line.contains('=') {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let name = parts[1].trim().trim_matches('"').trim_matches('\'');
                    return Some(name.to_string());
                }
            }
        }
        None
    }

    /// Check if file should be scanned
    fn should_scan_file(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            self.supported_extensions.contains(ext)
        } else {
            false
        }
    }

    /// Detect programming language from file extension
    fn detect_file_language(&self, file_path: &Path) -> Option<String> {
        let ext = file_path.extension()?.to_str()?;
        match ext {
            "rs" => Some("Rust".to_string()),
            "py" => Some("Python".to_string()),
            "js" => Some("JavaScript".to_string()),
            "ts" | "tsx" => Some("TypeScript".to_string()),
            "jsx" => Some("JavaScript".to_string()),
            "java" => Some("Java".to_string()),
            "go" => Some("Go".to_string()),
            "cpp" | "cc" => Some("C++".to_string()),
            "c" => Some("C".to_string()),
            "h" => Some("C/C++ Header".to_string()),
            "hpp" => Some("C++ Header".to_string()),
            _ => None,
        }
    }

    /// Check if project directory needs scanning (based on cache)
    pub async fn needs_scan(&self, root_path: &Path) -> bool {
        let cache = self.scan_cache.read().await;

        // Always scan if never scanned before
        if !cache.last_scanned.contains_key(root_path) {
            return true;
        }

        // Check if significant time has passed
        if let Some(last_scan) = cache.last_scanned.get(root_path) {
            if let Ok(elapsed) = last_scan.elapsed() {
                // Rescan every 24 hours
                return elapsed.as_secs() > 24 * 3600;
            }
        }

        false
    }

    /// Mark directory as scanned
    pub async fn mark_scanned(&self, root_path: &Path) {
        let mut cache = self.scan_cache.write().await;
        cache
            .last_scanned
            .insert(root_path.to_path_buf(), std::time::SystemTime::now());
    }
}
