//! Context analysis for understanding development patterns

use super::*;
use crate::SearchMemoryRequest;
use std::collections::HashMap;

/// Analyzes development context to detect patterns and suggest relevant knowledge
#[allow(dead_code)]
pub struct ContextAnalyzer {
    integration_manager: Arc<IntegrationManager>,
}

#[allow(dead_code)]
impl ContextAnalyzer {
    pub fn new(integration_manager: Arc<IntegrationManager>) -> Self {
        Self {
            integration_manager,
        }
    }

    /// Analyze current context and provide suggestions
    pub async fn analyze(&self) -> ContextAnalysis {
        let context = self.integration_manager.get_context().await;
        let config = self.integration_manager.config.read().await;

        let mut analysis = ContextAnalysis {
            detected_activity: None,
            confidence: 0.0,
            relevant_memories: vec![],
            suggested_actions: vec![],
            patterns_detected: vec![],
        };

        // Detect activity type
        if let Some((activity, confidence)) = self.detect_activity_type(&context).await {
            analysis.detected_activity = Some(activity);
            analysis.confidence = confidence;
        }

        // Find relevant historical knowledge
        if config.auto_suggestions {
            analysis.relevant_memories = self.find_relevant_knowledge(&context).await;
        }

        // Detect patterns
        analysis.patterns_detected = self.detect_patterns(&context).await;

        // Generate suggested actions
        analysis.suggested_actions = self.generate_suggestions(&context, &analysis).await;

        analysis
    }

    /// Detect the type of development activity
    async fn detect_activity_type(
        &self,
        context: &DevelopmentContext,
    ) -> Option<(ActivityType, f32)> {
        let mut scores: HashMap<ActivityType, f32> = HashMap::new();

        // Analyze recent files
        for file in &context.recent_files {
            if file.contains("test") || file.ends_with("_test.py") || file.ends_with(".test.js") {
                *scores.entry(ActivityType::Testing).or_insert(0.0) += 0.3;
            }
            if file.contains("README") || file.contains("docs/") {
                *scores.entry(ActivityType::Documentation).or_insert(0.0) += 0.3;
            }
        }

        // Analyze recent searches
        for search in &context.recent_searches {
            let search_lower = search.to_lowercase();
            if search_lower.contains("error")
                || search_lower.contains("exception")
                || search_lower.contains("bug")
                || search_lower.contains("fix")
            {
                *scores.entry(ActivityType::BugFix).or_insert(0.0) += 0.4;
                *scores.entry(ActivityType::Debugging).or_insert(0.0) += 0.3;
            }
            if search_lower.contains("implement")
                || search_lower.contains("feature")
                || search_lower.contains("add")
                || search_lower.contains("create")
            {
                *scores
                    .entry(ActivityType::FeatureDevelopment)
                    .or_insert(0.0) += 0.4;
            }
            if search_lower.contains("refactor")
                || search_lower.contains("clean")
                || search_lower.contains("optimize")
            {
                *scores.entry(ActivityType::Refactoring).or_insert(0.0) += 0.4;
            }
        }

        // Analyze recent edits
        let mut total_edits = 0;
        let mut test_edits = 0;
        let mut doc_edits = 0;

        for edit in &context.recent_edits {
            total_edits += 1;
            if edit.file_path.contains("test") {
                test_edits += 1;
            }
            if edit.file_path.contains("README") || edit.file_path.contains(".md") {
                doc_edits += 1;
            }

            // Analyze edit content
            if self.is_bug_fix_edit(&edit.old_content, &edit.new_content) {
                *scores.entry(ActivityType::BugFix).or_insert(0.0) += 0.2;
            }
        }

        if total_edits > 0 {
            if test_edits as f32 / total_edits as f32 > 0.5 {
                *scores.entry(ActivityType::Testing).or_insert(0.0) += 0.5;
            }
            if doc_edits as f32 / total_edits as f32 > 0.5 {
                *scores.entry(ActivityType::Documentation).or_insert(0.0) += 0.5;
            }
        }

        // Find highest scoring activity
        scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(activity, score)| (activity, score.min(1.0)))
    }

    /// Find relevant knowledge based on current context
    async fn find_relevant_knowledge(&self, context: &DevelopmentContext) -> Vec<String> {
        let mut memories = vec![];

        // Build search query based on context
        let mut search_terms = vec![];

        if let Some(file) = &context.current_file {
            if let Some(file_name) = file.split('/').last() {
                search_terms.push(file_name.to_string());
            }
        }

        // Add recent search terms
        search_terms.extend(context.recent_searches.iter().take(3).cloned());

        if search_terms.is_empty() {
            return memories;
        }

        let query = search_terms.join(" OR ");

        // Search for relevant memories
        let search_req = SearchMemoryRequest {
            query,
            limit: Some(5),
            entity_types: None,
            start_time: None,
            end_time: None,
        };

        // This would call the actual search service
        if let Ok(results) = self
            .integration_manager
            .graphiti_service
            .search_memory(search_req)
            .await
        {
            for result in results.results {
                let content = result
                    .content_preview
                    .unwrap_or_else(|| result.name.clone());
                memories.push(format!("[{}] {}", result.score, content));
            }
        }

        memories
    }

    /// Detect patterns in development behavior
    async fn detect_patterns(&self, context: &DevelopmentContext) -> Vec<DevelopmentPattern> {
        let mut patterns = vec![];

        // Pattern: Frequent file switching (possible confusion or exploration)
        if context.recent_files.len() > 5 {
            let unique_files: std::collections::HashSet<_> = context.recent_files.iter().collect();
            if (unique_files.len() as f32 / context.recent_files.len() as f32) < 0.5 {
                patterns.push(DevelopmentPattern {
                    pattern_type: PatternType::FrequentFileSwitching,
                    description: "Switching between same files frequently".to_string(),
                    suggestion: "Consider using split view or organizing related code better"
                        .to_string(),
                });
            }
        }

        // Pattern: Multiple searches for similar terms (possible struggle)
        let search_frequency = self.calculate_search_frequency(&context.recent_searches);
        for (term, count) in search_frequency {
            if count > 3 {
                patterns.push(DevelopmentPattern {
                    pattern_type: PatternType::RepeatedSearches,
                    description: format!("Searched for '{}' {} times", term, count),
                    suggestion: format!("Consider documenting findings about '{}'", term),
                });
            }
        }

        // Pattern: Rapid edits in same file (possible experimentation)
        let file_edit_count = self.count_edits_per_file(&context.recent_edits);
        for (file, count) in file_edit_count {
            if count > 5 {
                patterns.push(DevelopmentPattern {
                    pattern_type: PatternType::RapidEditing,
                    description: format!("Made {} edits to {}", count, file),
                    suggestion: "Consider using version control or creating a test environment"
                        .to_string(),
                });
            }
        }

        patterns
    }

    /// Generate context-aware suggestions
    async fn generate_suggestions(
        &self,
        _context: &DevelopmentContext,
        analysis: &ContextAnalysis,
    ) -> Vec<String> {
        let mut suggestions = vec![];

        // Activity-based suggestions
        if let Some(activity) = &analysis.detected_activity {
            match activity {
                ActivityType::BugFix => {
                    suggestions.push("💡 Remember to add a test case for this bug fix".to_string());
                    suggestions
                        .push("💡 Consider documenting the root cause in a comment".to_string());
                }
                ActivityType::FeatureDevelopment => {
                    suggestions.push(
                        "💡 Have you updated the documentation for this feature?".to_string(),
                    );
                    suggestions.push("💡 Consider adding integration tests".to_string());
                }
                ActivityType::Refactoring => {
                    suggestions
                        .push("💡 Ensure all tests still pass after refactoring".to_string());
                    suggestions.push("💡 Document why this refactoring was necessary".to_string());
                }
                ActivityType::Testing => {
                    suggestions.push("💡 Consider edge cases and error scenarios".to_string());
                    suggestions.push(
                        "💡 Aim for meaningful test names that describe the behavior".to_string(),
                    );
                }
                _ => {}
            }
        }

        // Pattern-based suggestions
        for pattern in &analysis.patterns_detected {
            suggestions.push(format!("💡 {}", pattern.suggestion));
        }

        // Add relevant memories as suggestions
        for (i, memory) in analysis.relevant_memories.iter().take(2).enumerate() {
            suggestions.push(format!("📚 Related knowledge #{}: {}", i + 1, memory));
        }

        suggestions
    }

    // Helper methods

    fn is_bug_fix_edit(&self, old_content: &str, new_content: &str) -> bool {
        let old_lower = old_content.to_lowercase();
        let new_lower = new_content.to_lowercase();

        // Simple heuristics
        (old_lower.contains("bug") || new_lower.contains("fix"))
            || (old_lower.contains("error") && !new_lower.contains("error"))
            || (new_lower.contains("handle") && !old_lower.contains("handle"))
    }

    fn calculate_search_frequency(&self, searches: &[String]) -> HashMap<String, usize> {
        let mut frequency = HashMap::new();

        for search in searches {
            // Normalize search terms
            let normalized = search
                .to_lowercase()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");

            *frequency.entry(normalized).or_insert(0) += 1;
        }

        frequency
    }

    fn count_edits_per_file(&self, edits: &[EditContext]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for edit in edits {
            *counts.entry(edit.file_path.clone()).or_insert(0) += 1;
        }

        counts
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ContextAnalysis {
    pub detected_activity: Option<ActivityType>,
    pub confidence: f32,
    pub relevant_memories: Vec<String>,
    pub suggested_actions: Vec<String>,
    pub patterns_detected: Vec<DevelopmentPattern>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DevelopmentPattern {
    pub pattern_type: PatternType,
    pub description: String,
    pub suggestion: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum PatternType {
    FrequentFileSwitching,
    RepeatedSearches,
    RapidEditing,
    LongSession,
    ContextSwitching,
}
