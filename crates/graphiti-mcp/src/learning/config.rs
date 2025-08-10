//! Configuration for learning detection and notifications

use super::detector::DetectorConfig;
use super::notifications::NotificationConfig;
use serde::Deserialize;
use serde::Serialize;

/// Complete configuration for the learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Enable the learning detection system
    pub enabled: bool,

    /// Detector configuration
    pub detector: DetectorConfig,

    /// Notification configuration
    pub notifications: NotificationConfig,

    /// Performance settings
    pub performance: PerformanceConfig,

    /// Language-specific settings
    pub language_settings: LanguageSettings,
}

/// Performance and resource usage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum concurrent learning analyses
    pub max_concurrent_analyses: usize,

    /// Analysis timeout in seconds
    pub analysis_timeout_seconds: u64,

    /// Enable caching of analysis results
    pub enable_caching: bool,

    /// Cache size (number of entries)
    pub cache_size: usize,

    /// Background cleanup interval in seconds
    pub cleanup_interval_seconds: u64,
}

/// Language-specific learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageSettings {
    /// Supported languages for code pattern detection
    pub supported_languages: Vec<String>,

    /// Language-specific pattern confidence thresholds
    pub language_thresholds: std::collections::HashMap<String, f32>,

    /// Enable multilingual entity detection (Chinese/English)
    pub enable_multilingual: bool,

    /// Preferred language for notifications
    pub notification_language: String,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detector: DetectorConfig::default(),
            notifications: NotificationConfig::default(),
            performance: PerformanceConfig::default(),
            language_settings: LanguageSettings::default(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_analyses: 4,
            analysis_timeout_seconds: 30,
            enable_caching: true,
            cache_size: 1000,
            cleanup_interval_seconds: 300, // 5 minutes
        }
    }
}

impl Default for LanguageSettings {
    fn default() -> Self {
        let mut language_thresholds = std::collections::HashMap::new();
        language_thresholds.insert("rust".to_string(), 0.8);
        language_thresholds.insert("python".to_string(), 0.7);
        language_thresholds.insert("javascript".to_string(), 0.7);
        language_thresholds.insert("typescript".to_string(), 0.75);
        language_thresholds.insert("java".to_string(), 0.75);
        language_thresholds.insert("cpp".to_string(), 0.8);
        language_thresholds.insert("go".to_string(), 0.75);

        Self {
            supported_languages: vec![
                "rust".to_string(),
                "python".to_string(),
                "javascript".to_string(),
                "typescript".to_string(),
                "java".to_string(),
                "cpp".to_string(),
                "c".to_string(),
                "go".to_string(),
                "swift".to_string(),
                "kotlin".to_string(),
            ],
            language_thresholds,
            enable_multilingual: true,
            notification_language: "zh-CN".to_string(), // Chinese by default
        }
    }
}

#[allow(dead_code)]
impl LearningConfig {
    /// Load configuration from TOML file
    pub fn from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get language-specific confidence threshold
    pub fn get_language_threshold(&self, language: &str) -> f32 {
        self.language_settings
            .language_thresholds
            .get(language)
            .copied()
            .unwrap_or(0.7) // Default threshold
    }

    /// Check if a language is supported
    pub fn is_language_supported(&self, language: &str) -> bool {
        self.language_settings
            .supported_languages
            .contains(&language.to_lowercase())
    }

    /// Get the default configuration as TOML string
    pub fn default_toml() -> String {
        toml::to_string_pretty(&Self::default())
            .unwrap_or_else(|_| include_str!("../../../../config.learning.toml").to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_default_config() {
        let config = LearningConfig::default();
        assert!(config.enabled);
        assert!(config.performance.enable_caching);
        assert_eq!(config.language_settings.notification_language, "zh-CN");
    }

    #[test]
    fn test_language_threshold() {
        let config = LearningConfig::default();
        assert_eq!(config.get_language_threshold("rust"), 0.8);
        assert_eq!(config.get_language_threshold("unknown"), 0.7);
    }

    #[test]
    fn test_language_support() {
        let config = LearningConfig::default();
        assert!(config.is_language_supported("rust"));
        assert!(config.is_language_supported("PYTHON")); // Case insensitive
        assert!(!config.is_language_supported("brainfuck"));
    }

    #[test]
    fn test_toml_serialization() {
        let config = LearningConfig::default();
        let toml = toml::to_string(&config).unwrap();
        let deserialized: LearningConfig = toml::from_str(&toml).unwrap();

        assert_eq!(config.enabled, deserialized.enabled);
        assert_eq!(
            config.performance.max_concurrent_analyses,
            deserialized.performance.max_concurrent_analyses
        );
    }
}
