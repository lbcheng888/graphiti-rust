//! Learning event definitions and context management

use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use uuid::Uuid;

/// Types of learning events that can be detected
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LearningEventType {
    /// New entity type discovered
    NewEntityType {
        entity_type: String,
        confidence: f32,
    },

    /// New relationship pattern discovered
    NewRelationshipPattern {
        pattern: String,
        entities: Vec<String>,
        confidence: f32,
    },

    /// New code pattern identified
    NewCodePattern {
        pattern_type: String,
        language: String,
        description: String,
        confidence: f32,
    },

    /// Conceptual breakthrough - new understanding
    ConceptualBreakthrough {
        concept: String,
        related_concepts: Vec<String>,
        insight: String,
        confidence: f32,
    },

    /// Anomaly detection - unusual pattern
    AnomalyDetected {
        anomaly_type: String,
        description: String,
        severity: AnomalySeverity,
        confidence: f32,
    },

    /// Knowledge connection - linking previously separate concepts
    KnowledgeConnection {
        concept_a: String,
        concept_b: String,
        connection_type: String,
        strength: f32,
    },

    /// Domain expertise expansion
    DomainExpansion {
        domain: String,
        new_topics: Vec<String>,
        expertise_level: ExpertiseLevel,
    },
}

/// Severity levels for anomalies
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Expertise levels for domain knowledge
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Context information for learning events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningContext {
    /// Current activity when learning occurred
    pub activity: Option<String>,

    /// File or source being processed
    pub source: Option<String>,

    /// Programming language (if applicable)
    pub language: Option<String>,

    /// Project context
    pub project: Option<String>,

    /// User session information
    pub session_id: Option<Uuid>,

    /// Previous related events
    pub related_events: Vec<Uuid>,

    /// Confidence in the context accuracy
    pub context_confidence: f32,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for LearningContext {
    fn default() -> Self {
        Self {
            activity: None,
            source: None,
            language: None,
            project: None,
            session_id: None,
            related_events: Vec::new(),
            context_confidence: 0.5,
            metadata: HashMap::new(),
        }
    }
}

/// A complete learning event with context and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEvent {
    /// Unique identifier for this event
    pub id: Uuid,

    /// Type of learning that occurred
    pub event_type: LearningEventType,

    /// When the learning occurred
    pub timestamp: DateTime<Utc>,

    /// Context information
    pub context: LearningContext,

    /// Human-readable summary
    pub summary: String,

    /// Detailed description
    pub description: String,

    /// Overall confidence in this learning event
    pub confidence: f32,

    /// Impact score (how significant this learning is)
    pub impact_score: f32,

    /// Whether this event has been seen by the user
    pub acknowledged: bool,
}

#[allow(dead_code)]
impl LearningEvent {
    /// Create a new learning event
    pub fn new(
        event_type: LearningEventType,
        context: LearningContext,
        summary: String,
        description: String,
    ) -> Self {
        let confidence = match &event_type {
            LearningEventType::NewEntityType { confidence, .. } => *confidence,
            LearningEventType::NewRelationshipPattern { confidence, .. } => *confidence,
            LearningEventType::NewCodePattern { confidence, .. } => *confidence,
            LearningEventType::ConceptualBreakthrough { confidence, .. } => *confidence,
            LearningEventType::AnomalyDetected { confidence, .. } => *confidence,
            LearningEventType::KnowledgeConnection { strength, .. } => *strength,
            LearningEventType::DomainExpansion { .. } => 0.8, // Default for domain expansion
        };

        let impact_score = Self::calculate_impact_score(&event_type, confidence);

        Self {
            id: Uuid::new_v4(),
            event_type,
            timestamp: Utc::now(),
            context,
            summary,
            description,
            confidence,
            impact_score,
            acknowledged: false,
        }
    }

    /// Calculate impact score based on event type and confidence
    fn calculate_impact_score(event_type: &LearningEventType, confidence: f32) -> f32 {
        let base_score = match event_type {
            LearningEventType::ConceptualBreakthrough { .. } => 0.9,
            LearningEventType::NewEntityType { .. } => 0.7,
            LearningEventType::KnowledgeConnection { .. } => 0.8,
            LearningEventType::DomainExpansion { .. } => 0.6,
            LearningEventType::NewCodePattern { .. } => 0.5,
            LearningEventType::NewRelationshipPattern { .. } => 0.6,
            LearningEventType::AnomalyDetected { severity, .. } => match severity {
                AnomalySeverity::Critical => 0.95,
                AnomalySeverity::High => 0.8,
                AnomalySeverity::Medium => 0.6,
                AnomalySeverity::Low => 0.4,
            },
        };

        // Adjust by confidence
        base_score * confidence
    }

    /// Get a short type name for categorization
    pub fn type_name(&self) -> &'static str {
        match &self.event_type {
            LearningEventType::NewEntityType { .. } => "new_entity",
            LearningEventType::NewRelationshipPattern { .. } => "new_relationship",
            LearningEventType::NewCodePattern { .. } => "new_code_pattern",
            LearningEventType::ConceptualBreakthrough { .. } => "breakthrough",
            LearningEventType::AnomalyDetected { .. } => "anomaly",
            LearningEventType::KnowledgeConnection { .. } => "connection",
            LearningEventType::DomainExpansion { .. } => "domain_expansion",
        }
    }

    /// Check if this event should trigger a notification
    pub fn should_notify(&self, min_confidence: f32, min_impact: f32) -> bool {
        self.confidence >= min_confidence && self.impact_score >= min_impact
    }

    /// Mark this event as acknowledged by the user
    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_event_creation() {
        let event_type = LearningEventType::NewEntityType {
            entity_type: "TestEntity".to_string(),
            confidence: 0.8,
        };

        let context = LearningContext::default();
        let event = LearningEvent::new(
            event_type,
            context,
            "New entity discovered".to_string(),
            "A new type of entity was identified in the code".to_string(),
        );

        assert_eq!(event.confidence, 0.8);
        assert!(event.impact_score > 0.0);
        assert_eq!(event.type_name(), "new_entity");
        assert!(!event.acknowledged);
    }

    #[test]
    fn test_should_notify() {
        let event_type = LearningEventType::ConceptualBreakthrough {
            concept: "Design Pattern".to_string(),
            related_concepts: vec!["Observer".to_string()],
            insight: "Understanding of observer pattern".to_string(),
            confidence: 0.9,
        };

        let event = LearningEvent::new(
            event_type,
            LearningContext::default(),
            "Breakthrough in design patterns".to_string(),
            "New understanding of observer pattern".to_string(),
        );

        assert!(event.should_notify(0.5, 0.5));
        assert!(!event.should_notify(0.95, 0.95));
    }
}
