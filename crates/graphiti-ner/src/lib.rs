//! Named Entity Recognition using multiple approaches
//!
//! This module provides NER functionality using rule-based extraction,
//! rust-bert models, and Candle-based models for pure Rust entity extraction.

#![warn(missing_docs)]

use async_trait::async_trait;
use graphiti_core::error::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, instrument};

// Module declarations
mod candle_ner_extractor;
mod hybrid_extractor;
#[cfg(feature = "rust-bert-ner")]
mod rust_bert_extractor;

// Re-exports
pub use candle_ner_extractor::CandleNerExtractor;
pub use hybrid_extractor::{HybridConfig, HybridExtractor};
#[cfg(feature = "rust-bert-ner")]
pub use rust_bert_extractor::RustBertExtractor;

/// NER label types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityLabel {
    /// Person name
    Person,
    /// Organization
    Organization,
    /// Location
    Location,
    /// Miscellaneous entity type
    Miscellaneous,
    /// Other entity type
    Other,
}

impl EntityLabel {
    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityLabel::Person => "Person",
            EntityLabel::Organization => "Organization",
            EntityLabel::Location => "Location",
            EntityLabel::Miscellaneous => "Miscellaneous",
            EntityLabel::Other => "Other",
        }
    }
}

/// Extracted entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity text
    pub text: String,
    /// Entity label
    pub label: EntityLabel,
    /// Confidence score
    pub score: f32,
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
}

/// Trait for entity extraction
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    /// Extract entities from text
    async fn extract(&self, text: &str) -> Result<Vec<ExtractedEntity>>;
}

/// Simple rule-based entity extractor
pub struct RuleBasedExtractor {
    patterns: HashMap<EntityLabel, Vec<Regex>>,
}

impl Default for RuleBasedExtractor {
    fn default() -> Self {
        let mut patterns = HashMap::new();

        // Chinese name patterns
        patterns.insert(EntityLabel::Person, vec![
            // Surname + common given name characters
            Regex::new(r"[张王李赵刘陈杨黄周吴徐孙马朱胡郭何罗高林郑梁谢唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃河汤滕殷罗毕郑詹关鲁韩杨]{1,2}[文武英华明国志强建伟晓静丽敏燕芳玲娜秀兰霞浩博涛磊洋勇军杰俊峰超群飞鹏宇阳晨辰龙云海江天翔宁安康乐欣悦思宸子轩浩然皓轩梓豪子涵雨泽宇航梓萱紫涵梦琪雅婷雨薇晨曦梓涵诗涵欣怡梦洁雅静雪儿佳怡晓萱]{1,3}").unwrap(),
            // Fallback: Surname + any single Han character (covers 张三/李四等)
            Regex::new(r"[张王李赵刘陈杨黄周吴徐孙马朱胡郭何罗高林郑梁谢唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃河汤滕殷罗毕郑詹关鲁韩杨][\p{Han}]").unwrap(),
            // English names
            Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b").unwrap(),
        ]);

        // Organization patterns
        patterns.insert(EntityLabel::Organization, vec![
            Regex::new(r"[\u4e00-\u9fa5]+(公司|集团|机构|组织|协会|委员会|部门|中心|研究院|大学|学院|医院|银行|酒店|商场|超市|餐厅|工厂|企业|政府|局|厅|处|科|股份|有限|责任|国际)").unwrap(),
            // Allow dot after Inc (e.g., Inc.) and require trailing word boundary
            // Company name composed of capitalized words followed by a known suffix
            Regex::new(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Inc\.?|Corp\.?|Corporation|Company|Ltd\.?|LLC|Group|Organization|Association|Institute|University|College|Hospital|Bank)\b").unwrap(),
        ]);

        // Location patterns (include word boundary for English to avoid partial matches)
        patterns.insert(EntityLabel::Location, vec![
            Regex::new(r"(北京|上海|广州|深圳|天津|重庆|成都|杭州|武汉|西安|苏州|南京|郑州|长沙|东莞|沈阳|青岛|合肥|佛山|山东|江苏|广东|浙江|河南|河北|湖南|湖北|福建|安徽|辽宁|陕西|内蒙古|新疆|广西|宁夏|西藏|香港|澳门|台湾|中国|美国|日本|韩国|英国|法国|德国|俄罗斯|加拿大|澳大利亚)").unwrap(),
            Regex::new(r"\b(New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|Charlotte|San Francisco|Indianapolis|Seattle|Denver|Washington|Boston|El Paso|Nashville|Detroit|Oklahoma City|Portland|Las Vegas|Memphis|Louisville|Baltimore|Milwaukee|Albuquerque|Tucson|Fresno|Mesa|Sacramento|Atlanta|Kansas City|Colorado Springs|Omaha|Raleigh|Miami|Long Beach|Virginia Beach|Oakland|Minneapolis|Tulsa|Tampa|Arlington|New Orleans|Wichita|Bakersfield|Cleveland|Aurora|Anaheim|Honolulu|Santa Ana|Riverside|Corpus Christi|Lexington|Henderson|Stockton|Saint Paul|Cincinnati|St\. Louis|Pittsburgh|Greensboro|Lincoln|Anchorage|Plano|Orlando|Irvine|Newark|Durham|Chula Vista|Toledo|Fort Wayne|St\. Petersburg|Laredo|Jersey City|Chandler|Madison|Lubbock|Scottsdale|Reno|Buffalo|Gilbert|Glendale|North Las Vegas|Winston-Salem|Chesapeake|Norfolk|Fremont|Garland|Irving|Hialeah|Richmond|Boise|Spokane|Baton Rouge|United States|China|Japan|Germany|United Kingdom|France|India|Italy|Brazil|Canada|South Korea|Spain|Australia|Russia|Mexico|Indonesia|Netherlands|Turkey|Saudi Arabia|Switzerland|Taiwan|Belgium|Sweden|Ireland|Poland|Argentina|Austria|Norway|United Arab Emirates|Israel|Egypt|Denmark|Singapore|Malaysia|Philippines|South Africa|Finland|Chile|Pakistan|Romania|Czech Republic|New Zealand|Greece|Iraq|Portugal|Algeria|Qatar|Kazakhstan|Hungary|Kuwait|Morocco|Peru|Ukraine|Ecuador|Slovakia|Venezuela|Uzbekistan|Croatia|Dominican Republic|Ethiopia|Kenya|Guatemala|Puerto Rico|Oman|Bulgaria|Luxembourg|Uruguay|Serbia|Azerbaijan|Sri Lanka|Myanmar|Slovenia|Tunisia|Belarus|Costa Rica|Lithuania|Turkmenistan|Panama|Lebanon|Jordan|Cyprus|El Salvador|Cameroon|Bolivia|Bahrain|Latvia|Trinidad and Tobago|Estonia|Paraguay|Libya|Honduras|Papua New Guinea|Senegal|Jamaica|Georgia|Gabon|Bosnia and Herzegovina|Mauritius|Armenia|Albania|Malta|Mozambique|Burkina Faso|Mongolia|Brunei|Bahamas|Macedonia|Nicaragua|Madagascar|Mali|Botswana|Benin|Guyana|Haiti|Moldova|Afghanistan|Niger|Laos|Rwanda|Kyrgyzstan|Tajikistan|Malawi|Zimbabwe|Mauritania|Barbados|Suriname|Fiji|Liberia|Sierra Leone|Togo|Eritrea|Central African Republic|Cape Verde|Djibouti|Andorra|Gambia|Burundi|Seychelles|Comoros|Solomon Islands|Guinea-Bissau|Vanuatu|Grenada|Saint Lucia|Kiribati|East Timor|Saint Vincent and the Grenadines|Tonga|Micronesia|Palau|Marshall Islands|San Marino|Liechtenstein|Monaco|Vatican City)\b").unwrap(),
        ]);

        Self { patterns }
    }
}

#[async_trait]
impl EntityExtractor for RuleBasedExtractor {
    #[instrument(skip(self, text))]
    async fn extract(&self, text: &str) -> Result<Vec<ExtractedEntity>> {
        debug!("Extracting entities from text of length {}", text.len());

        let mut entities = Vec::new();

        for (label, patterns) in &self.patterns {
            for pattern in patterns {
                for mat in pattern.find_iter(text) {
                    entities.push(ExtractedEntity {
                        text: mat.as_str().to_string(),
                        label: *label,
                        score: 0.8, // Fixed confidence for rule-based
                        start: mat.start(),
                        end: mat.end(),
                    });
                }
            }
        }

        // Deduplicate exact duplicates but allow overlaps across labels
        let mut seen: std::collections::HashSet<(usize, usize, EntityLabel, String)> =
            std::collections::HashSet::new();
        let mut unique = Vec::new();

        for e in entities.into_iter() {
            let key = (e.start, e.end, e.label, e.text.clone());
            if seen.insert(key) {
                unique.push(e);
            }
        }

        // Sort by position for stable order
        unique.sort_by_key(|e| e.start);

        debug!("Extracted {} entities", unique.len());
        Ok(unique)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chinese_extraction() {
        let extractor = RuleBasedExtractor::default();
        let text = "张三在北京大学工作，李四住在上海。";

        let entities = extractor.extract(text).await.unwrap();

        assert!(entities.len() >= 3); // At least: 张三, 北京大学, 上海

        // Check if we found person names
        assert!(entities
            .iter()
            .any(|e| e.label == EntityLabel::Person && e.text == "张三"));

        // Check if we found organizations
        assert!(entities
            .iter()
            .any(|e| e.label == EntityLabel::Organization && e.text.contains("北京大学")));

        // Check if we found locations
        assert!(entities
            .iter()
            .any(|e| e.label == EntityLabel::Location && e.text == "上海"));
    }

    #[tokio::test]
    async fn test_english_extraction() {
        let extractor = RuleBasedExtractor::default();
        let text = "John Smith works at Google Inc in San Francisco.";

        let entities = extractor.extract(text).await.unwrap();

        assert!(entities.len() >= 3);

        // Check if we found person names
        assert!(entities
            .iter()
            .any(|e| e.label == EntityLabel::Person && e.text == "John Smith"));

        // Check if we found organizations
        assert!(entities
            .iter()
            .any(|e| e.label == EntityLabel::Organization && e.text == "Google Inc"));

        // Check if we found locations
        assert!(entities
            .iter()
            .any(|e| e.label == EntityLabel::Location && e.text == "San Francisco"));
    }
}
