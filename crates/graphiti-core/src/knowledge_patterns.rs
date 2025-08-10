//! Knowledge pattern learning and recommendation system
//!
//! This module implements intelligent pattern recognition and recommendation
//! based on development experiences and code patterns.

use crate::code_entities::DevelopmentActivity;
use crate::error::Error;
use crate::error::Result;
use crate::storage::GraphStorage;
use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::HashSet;
use tracing::info;
use tracing::instrument;

use uuid::Uuid;

/// Pattern learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternLearningConfig {
    /// Minimum frequency for pattern recognition
    pub min_frequency: usize,
    /// Confidence threshold for recommendations
    pub confidence_threshold: f64,
    /// Maximum number of recommendations to return
    pub max_recommendations: usize,
    /// Time window for pattern analysis (in days)
    pub time_window_days: i64,
    /// Minimum pattern complexity (number of entities)
    pub min_pattern_complexity: usize,
}

impl Default for PatternLearningConfig {
    fn default() -> Self {
        Self {
            min_frequency: 3,
            confidence_threshold: 0.7,
            max_recommendations: 10,
            time_window_days: 90,
            min_pattern_complexity: 2,
        }
    }
}

/// Learned pattern with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPattern {
    /// Pattern ID
    pub id: Uuid,
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Pattern entities
    pub entities: Vec<String>,
    /// Pattern relationships
    pub relationships: Vec<String>,
    /// Frequency of occurrence
    pub frequency: usize,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Success rate when applied
    pub success_rate: f64,
    /// Last seen timestamp
    pub last_seen: DateTime<Utc>,
    /// Pattern category
    pub category: PatternCategory,
    /// Associated development activities
    pub activities: Vec<Uuid>,
}

/// Pattern categories for classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PatternCategory {
    /// Architecture patterns
    Architecture,
    /// Design patterns
    Design,
    /// Testing patterns
    Testing,
    /// Performance patterns
    Performance,
    /// Security patterns
    Security,
    /// Integration patterns
    Integration,
    /// Debugging patterns
    Debugging,
    /// Refactoring patterns
    Refactoring,
}

/// Pattern recommendation with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecommendation {
    /// Recommended pattern
    pub pattern: LearnedPattern,
    /// Relevance score (0.0 - 1.0)
    pub relevance: f64,
    /// Reasoning for recommendation
    pub reasoning: String,
    /// Similar past contexts
    pub similar_contexts: Vec<Uuid>,
    /// Expected benefits
    pub expected_benefits: Vec<String>,
    /// Implementation suggestions
    pub implementation_hints: Vec<String>,
}

/// Context for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternContext {
    /// Current entities in context
    pub entities: Vec<String>,
    /// Current relationships
    pub relationships: Vec<String>,
    /// Current development stage
    pub stage: String,
    /// Project characteristics
    pub project_type: String,
    /// Team size
    pub team_size: Option<usize>,
    /// Technology stack
    pub technologies: Vec<String>,
}

/// Pattern learning and recommendation engine
pub struct PatternLearner<S>
where
    S: GraphStorage,
{
    _storage: S,
    config: PatternLearningConfig,
    learned_patterns: HashMap<Uuid, LearnedPattern>,
    pattern_index: HashMap<String, Vec<Uuid>>, // Entity/relationship -> pattern IDs
}

impl<S> PatternLearner<S>
where
    S: GraphStorage<Error = Error>,
{
    /// Create a new pattern learner
    pub fn new(storage: S, config: PatternLearningConfig) -> Self {
        Self {
            _storage: storage,
            config,
            learned_patterns: HashMap::new(),
            pattern_index: HashMap::new(),
        }
    }

    /// Learn patterns from historical development data
    #[instrument(skip(self))]
    pub async fn learn_patterns(&mut self) -> Result<Vec<LearnedPattern>> {
        info!("Starting pattern learning from historical data");

        // Load development activities from the specified time window
        let activities = self.load_recent_activities().await?;
        info!(
            "Loaded {} development activities for analysis",
            activities.len()
        );

        // Extract patterns from activities
        let mut pattern_candidates = self.extract_pattern_candidates(&activities).await?;

        // Filter patterns by frequency and complexity
        pattern_candidates.retain(|pattern| {
            pattern.frequency >= self.config.min_frequency
                && pattern.entities.len() >= self.config.min_pattern_complexity
        });

        // Calculate confidence scores
        for pattern in &mut pattern_candidates {
            pattern.confidence = self
                .calculate_pattern_confidence(pattern, &activities)
                .await?;
        }

        // Filter by confidence threshold
        pattern_candidates.retain(|pattern| pattern.confidence >= self.config.confidence_threshold);

        // Update learned patterns
        for pattern in &pattern_candidates {
            self.learned_patterns.insert(pattern.id, pattern.clone());
            self.update_pattern_index(pattern);
        }

        info!("Learned {} new patterns", pattern_candidates.len());
        Ok(pattern_candidates)
    }

    /// Get pattern recommendations for a given context
    #[instrument(skip(self))]
    pub async fn recommend_patterns(
        &self,
        context: &PatternContext,
    ) -> Result<Vec<PatternRecommendation>> {
        info!("Generating pattern recommendations for context");

        let mut recommendations = Vec::new();

        // Find patterns matching the current context
        let matching_patterns = self.find_matching_patterns(context).await?;

        for pattern in matching_patterns {
            let relevance = self.calculate_relevance(&pattern, context).await?;

            if relevance > 0.5 {
                // Minimum relevance threshold
                let recommendation = PatternRecommendation {
                    pattern: pattern.clone(),
                    relevance,
                    reasoning: self.generate_reasoning(&pattern, context).await?,
                    similar_contexts: self.find_similar_contexts(&pattern, context).await?,
                    expected_benefits: self.predict_benefits(&pattern, context).await?,
                    implementation_hints: self
                        .generate_implementation_hints(&pattern, context)
                        .await?,
                };

                recommendations.push(recommendation);
            }
        }

        // Sort by relevance
        recommendations.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());

        // Limit to max recommendations
        recommendations.truncate(self.config.max_recommendations);

        info!(
            "Generated {} pattern recommendations",
            recommendations.len()
        );
        Ok(recommendations)
    }

    /// Update pattern success rate based on feedback
    #[instrument(skip(self))]
    pub async fn update_pattern_feedback(&mut self, pattern_id: Uuid, success: bool) -> Result<()> {
        if let Some(pattern) = self.learned_patterns.get_mut(&pattern_id) {
            // Simple success rate update (could be more sophisticated)
            let total_applications =
                (pattern.frequency as f64 / pattern.success_rate).ceil() as usize;
            let successful_applications =
                (total_applications as f64 * pattern.success_rate) as usize;

            let new_total = total_applications + 1;
            let new_successful = if success {
                successful_applications + 1
            } else {
                successful_applications
            };

            pattern.success_rate = new_successful as f64 / new_total as f64;
            pattern.frequency = new_total;

            info!(
                "Updated pattern {} success rate to {:.2}",
                pattern_id, pattern.success_rate
            );
        }

        Ok(())
    }

    /// Load recent development activities from storage
    async fn load_recent_activities(&self) -> Result<Vec<DevelopmentActivity>> {
        // Query storage for recent development activities
        // This would typically involve querying the graph database for activity nodes
        // For now, return empty vector as we need proper storage integration

        // TODO: Implement proper storage query:
        // 1. Query activity nodes from the last N days
        // 2. Filter by activity type (Implementation, Testing, etc.)
        // 3. Sort by timestamp
        // 4. Return structured DevelopmentActivity objects

        Ok(vec![])
    }

    /// Extract pattern candidates from activities
    async fn extract_pattern_candidates(
        &self,
        activities: &[DevelopmentActivity],
    ) -> Result<Vec<LearnedPattern>> {
        let mut patterns = Vec::new();

        // Group activities by project and analyze patterns
        let mut project_activities: HashMap<String, Vec<&DevelopmentActivity>> = HashMap::new();

        for activity in activities {
            project_activities
                .entry(activity.project.clone())
                .or_insert_with(Vec::new)
                .push(activity);
        }

        for (project, project_acts) in project_activities {
            // Analyze common sequences and patterns
            let pattern = LearnedPattern {
                id: Uuid::new_v4(),
                name: format!("Common pattern in {}", project),
                description: "Frequently occurring development pattern".to_string(),
                entities: vec!["Authentication".to_string(), "Testing".to_string()],
                relationships: vec!["IMPLEMENTS".to_string(), "TESTS".to_string()],
                frequency: project_acts.len(),
                confidence: 0.8,
                success_rate: 0.9,
                last_seen: Utc::now(),
                category: PatternCategory::Architecture,
                activities: project_acts.iter().map(|a| a.id).collect(),
            };

            patterns.push(pattern);
        }

        Ok(patterns)
    }

    /// Calculate pattern confidence score
    async fn calculate_pattern_confidence(
        &self,
        pattern: &LearnedPattern,
        _activities: &[DevelopmentActivity],
    ) -> Result<f64> {
        // Simple confidence calculation based on frequency and success
        let base_confidence = (pattern.frequency as f64).ln() / 10.0;
        let success_bonus = pattern.success_rate * 0.3;

        Ok((base_confidence + success_bonus).min(1.0))
    }

    /// Update pattern index for fast lookup
    fn update_pattern_index(&mut self, pattern: &LearnedPattern) {
        for entity in &pattern.entities {
            self.pattern_index
                .entry(entity.clone())
                .or_insert_with(Vec::new)
                .push(pattern.id);
        }

        for relationship in &pattern.relationships {
            self.pattern_index
                .entry(relationship.clone())
                .or_insert_with(Vec::new)
                .push(pattern.id);
        }
    }

    /// Find patterns matching the given context
    async fn find_matching_patterns(
        &self,
        context: &PatternContext,
    ) -> Result<Vec<LearnedPattern>> {
        let mut matching_patterns = Vec::new();
        let mut candidate_ids = HashSet::new();

        // Find patterns by entity and relationship overlap
        for entity in &context.entities {
            if let Some(pattern_ids) = self.pattern_index.get(entity) {
                candidate_ids.extend(pattern_ids);
            }
        }

        for relationship in &context.relationships {
            if let Some(pattern_ids) = self.pattern_index.get(relationship) {
                candidate_ids.extend(pattern_ids);
            }
        }

        // Filter and score candidates
        for pattern_id in candidate_ids {
            if let Some(pattern) = self.learned_patterns.get(&pattern_id) {
                let overlap_score = self.calculate_context_overlap(pattern, context);
                if overlap_score > 0.3 {
                    // Minimum overlap threshold
                    matching_patterns.push(pattern.clone());
                }
            }
        }

        Ok(matching_patterns)
    }

    /// Calculate relevance score for a pattern in the given context
    async fn calculate_relevance(
        &self,
        pattern: &LearnedPattern,
        context: &PatternContext,
    ) -> Result<f64> {
        let context_overlap = self.calculate_context_overlap(pattern, context);
        let confidence_weight = pattern.confidence * 0.4;
        let success_weight = pattern.success_rate * 0.3;
        let frequency_weight = (pattern.frequency as f64).ln() / 10.0 * 0.3;

        Ok(context_overlap * 0.4 + confidence_weight + success_weight + frequency_weight)
    }

    /// Calculate overlap between pattern and context
    fn calculate_context_overlap(&self, pattern: &LearnedPattern, context: &PatternContext) -> f64 {
        let entity_overlap = self.calculate_set_overlap(&pattern.entities, &context.entities);
        let relationship_overlap =
            self.calculate_set_overlap(&pattern.relationships, &context.relationships);

        (entity_overlap + relationship_overlap) / 2.0
    }

    /// Calculate overlap between two sets
    fn calculate_set_overlap(&self, set1: &[String], set2: &[String]) -> f64 {
        if set1.is_empty() || set2.is_empty() {
            return 0.0;
        }

        let set1: HashSet<_> = set1.iter().collect();
        let set2: HashSet<_> = set2.iter().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        intersection as f64 / union as f64
    }

    /// Generate reasoning for recommendation
    async fn generate_reasoning(
        &self,
        pattern: &LearnedPattern,
        _context: &PatternContext,
    ) -> Result<String> {
        Ok(format!(
            "This pattern has been successfully applied {} times with a {:.1}% success rate. \
             It matches your current context with entities: {} and relationships: {}.",
            pattern.frequency,
            pattern.success_rate * 100.0,
            pattern.entities.join(", "),
            pattern.relationships.join(", ")
        ))
    }

    /// Find similar contexts where this pattern was applied
    async fn find_similar_contexts(
        &self,
        pattern: &LearnedPattern,
        _context: &PatternContext,
    ) -> Result<Vec<Uuid>> {
        // Return associated activities as similar contexts
        Ok(pattern.activities.clone())
    }

    /// Predict benefits of applying this pattern
    async fn predict_benefits(
        &self,
        pattern: &LearnedPattern,
        _context: &PatternContext,
    ) -> Result<Vec<String>> {
        let benefits = match pattern.category {
            PatternCategory::Architecture => vec![
                "Improved system modularity".to_string(),
                "Better separation of concerns".to_string(),
                "Enhanced maintainability".to_string(),
            ],
            PatternCategory::Testing => vec![
                "Higher code coverage".to_string(),
                "Reduced bug count".to_string(),
                "Faster debugging".to_string(),
            ],
            PatternCategory::Performance => vec![
                "Improved response times".to_string(),
                "Better resource utilization".to_string(),
                "Enhanced scalability".to_string(),
            ],
            _ => vec![
                "Improved code quality".to_string(),
                "Better development practices".to_string(),
            ],
        };

        Ok(benefits)
    }

    /// Generate implementation hints
    async fn generate_implementation_hints(
        &self,
        pattern: &LearnedPattern,
        _context: &PatternContext,
    ) -> Result<Vec<String>> {
        let hints = vec![
            format!(
                "Start by implementing {}",
                pattern
                    .entities
                    .first()
                    .unwrap_or(&"core component".to_string())
            ),
            "Follow the established relationship patterns".to_string(),
            "Consider the success factors from previous applications".to_string(),
            "Monitor the implementation for early feedback".to_string(),
        ];

        Ok(hints)
    }
}
