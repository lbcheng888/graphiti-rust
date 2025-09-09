use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

/// 代码相关的专门实体类型，用于支持agentic coding工作流
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeEntityType {
    /// 类/结构体
    Class,
    /// 函数/方法
    Function,
    /// 模块/包
    Module,
    /// API接口
    Api,
    /// 数据库表/模型
    DataModel,
    /// 配置文件
    Configuration,
    /// 测试用例
    TestCase,
    /// 文档
    Documentation,
    /// 设计决策
    DesignDecision,
    /// Bug报告
    BugReport,
    /// 性能优化
    PerformanceOptimization,
    /// 安全修复
    SecurityFix,
    /// 重构记录
    RefactoringRecord,
    /// 最佳实践
    BestPractice,
    /// 代码审查
    CodeReview,
}

impl fmt::Display for CodeEntityType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str_repr = match self {
            CodeEntityType::Class => "Class",
            CodeEntityType::Function => "Function",
            CodeEntityType::Module => "Module",
            CodeEntityType::Api => "API",
            CodeEntityType::DataModel => "DataModel",
            CodeEntityType::Configuration => "Configuration",
            CodeEntityType::TestCase => "TestCase",
            CodeEntityType::Documentation => "Documentation",
            CodeEntityType::DesignDecision => "DesignDecision",
            CodeEntityType::BugReport => "BugReport",
            CodeEntityType::PerformanceOptimization => "PerformanceOptimization",
            CodeEntityType::SecurityFix => "SecurityFix",
            CodeEntityType::RefactoringRecord => "RefactoringRecord",
            CodeEntityType::BestPractice => "BestPractice",
            CodeEntityType::CodeReview => "CodeReview",
        };
        write!(f, "{str_repr}")
    }
}

/// 代码实体节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEntity {
    /// Unique identifier for the code entity
    pub id: Uuid,
    /// Type of code entity (class, function, etc.)
    pub entity_type: CodeEntityType,
    /// Name of the code entity
    pub name: String,
    /// Description of the code entity
    pub description: String,
    /// 文件路径（如果适用）
    pub file_path: Option<String>,
    /// 行号范围（如果适用）
    pub line_range: Option<(u32, u32)>,
    /// 编程语言
    pub language: Option<String>,
    /// 代码框架/技术栈
    pub framework: Option<String>,
    /// 复杂度评级（1-10）
    pub complexity: Option<u8>,
    /// 重要程度（1-10）
    pub importance: Option<u8>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 最后修改时间
    pub updated_at: DateTime<Utc>,
    /// 扩展属性
    pub metadata: HashMap<String, serde_json::Value>,
}

/// 代码关系类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeRelationType {
    /// 依赖关系（A depends on B）
    DependsOn,
    /// 实现关系（A implements B）
    Implements,
    /// 继承关系（A extends B）
    Extends,
    /// 调用关系（A calls B）
    Calls,
    /// 包含关系（A contains B）
    Contains,
    /// 配置关系（A configures B）
    Configures,
    /// 测试关系（A tests B）
    Tests,
    /// 文档关系（A documents B）
    Documents,
    /// 修复关系（A fixes B）
    Fixes,
    /// 优化关系（A optimizes B）
    Optimizes,
    /// 重构关系（A refactors B）
    Refactors,
    /// 审查关系（A reviews B）
    Reviews,
    /// 相似关系（A is similar to B）
    SimilarTo,
    /// 影响关系（A affects B）
    Affects,
}

/// 代码关系
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRelation {
    /// Unique identifier for the relation
    pub id: Uuid,
    /// Type of code relation
    pub relation_type: CodeRelationType,
    /// Source entity ID
    pub source_id: Uuid,
    /// Target entity ID
    pub target_id: Uuid,
    /// Strength of the relation (0.0 - 1.0)
    pub strength: f32,
    /// Description of the relation
    pub description: String,
    /// When the relation was created
    pub created_at: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// 工作流阶段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkflowStage {
    /// 需求分析
    RequirementAnalysis,
    /// 架构设计
    ArchitectureDesign,
    /// 详细设计
    DetailedDesign,
    /// 编码实现
    Implementation,
    /// 单元测试
    UnitTesting,
    /// 集成测试
    IntegrationTesting,
    /// 代码审查
    CodeReview,
    /// 部署发布
    Deployment,
    /// 监控维护
    Monitoring,
    /// Bug修复
    BugFix,
    /// 性能优化
    PerformanceOptimization,
    /// 重构
    Refactoring,
}

impl fmt::Display for WorkflowStage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str_repr = match self {
            WorkflowStage::RequirementAnalysis => "RequirementAnalysis",
            WorkflowStage::ArchitectureDesign => "ArchitectureDesign",
            WorkflowStage::DetailedDesign => "DetailedDesign",
            WorkflowStage::Implementation => "Implementation",
            WorkflowStage::UnitTesting => "UnitTesting",
            WorkflowStage::IntegrationTesting => "IntegrationTesting",
            WorkflowStage::CodeReview => "CodeReview",
            WorkflowStage::Deployment => "Deployment",
            WorkflowStage::Monitoring => "Monitoring",
            WorkflowStage::BugFix => "BugFix",
            WorkflowStage::PerformanceOptimization => "PerformanceOptimization",
            WorkflowStage::Refactoring => "Refactoring",
        };
        write!(f, "{str_repr}")
    }
}

/// 开发活动记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevelopmentActivity {
    /// Unique identifier for the activity
    pub id: Uuid,
    /// Type of development activity
    pub activity_type: WorkflowStage,
    /// Title of the activity
    pub title: String,
    /// Description of the activity
    pub description: String,
    /// 相关的代码实体
    pub related_entities: Vec<Uuid>,
    /// 开发者
    pub developer: String,
    /// 项目/模块
    pub project: String,
    /// 耗时（分钟）
    pub duration_minutes: Option<u32>,
    /// 难度评级（1-10）
    pub difficulty: Option<u8>,
    /// 质量评级（1-10）
    pub quality: Option<u8>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 扩展属性
    pub metadata: HashMap<String, serde_json::Value>,
}

/// 知识模式（可复用的开发经验）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePattern {
    /// Unique identifier for the pattern
    pub id: Uuid,
    /// Name of the knowledge pattern
    pub pattern_name: String,
    /// Description of the pattern
    pub description: String,
    /// 适用场景
    pub applicable_scenarios: Vec<String>,
    /// 相关技术栈
    pub technology_stack: Vec<String>,
    /// 成功案例数量
    pub success_count: u32,
    /// 失败案例数量
    pub failure_count: u32,
    /// 平均效果评分（1-10）
    pub avg_effectiveness: f32,
    /// 复杂度（1-10）
    pub complexity: u8,
    /// 最后使用时间
    pub last_used: DateTime<Utc>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 扩展属性
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CodeEntity {
    /// Create a new code entity
    #[must_use]
    pub fn new(entity_type: CodeEntityType, name: String, description: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            entity_type,
            name,
            description,
            file_path: None,
            line_range: None,
            language: None,
            framework: None,
            complexity: None,
            importance: None,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// 设置文件位置信息
    #[must_use]
    pub fn with_location(mut self, file_path: String, line_range: Option<(u32, u32)>) -> Self {
        self.file_path = Some(file_path);
        self.line_range = line_range;
        self
    }

    /// 设置技术信息
    pub fn with_technology(mut self, language: Option<String>, framework: Option<String>) -> Self {
        self.language = language;
        self.framework = framework;
        self
    }

    /// 设置评级信息
    pub fn with_ratings(mut self, complexity: Option<u8>, importance: Option<u8>) -> Self {
        self.complexity = complexity;
        self.importance = importance;
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl CodeRelation {
    /// Create a new code relation
    pub fn new(
        relation_type: CodeRelationType,
        source_id: Uuid,
        target_id: Uuid,
        strength: f32,
        description: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            relation_type,
            source_id,
            target_id,
            strength: strength.clamp(0.0, 1.0),
            description,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl DevelopmentActivity {
    /// Create a new development activity
    pub fn new(
        activity_type: WorkflowStage,
        title: String,
        description: String,
        developer: String,
        project: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            activity_type,
            title,
            description,
            related_entities: Vec::new(),
            developer,
            project,
            duration_minutes: None,
            difficulty: None,
            quality: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// 添加相关实体
    pub fn with_entities(mut self, entity_ids: Vec<Uuid>) -> Self {
        self.related_entities = entity_ids;
        self
    }

    /// 设置评级信息
    pub fn with_ratings(
        mut self,
        duration_minutes: Option<u32>,
        difficulty: Option<u8>,
        quality: Option<u8>,
    ) -> Self {
        self.duration_minutes = duration_minutes;
        self.difficulty = difficulty;
        self.quality = quality;
        self
    }
}

impl KnowledgePattern {
    /// Create a new knowledge pattern
    pub fn new(pattern_name: String, description: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            pattern_name,
            description,
            applicable_scenarios: Vec::new(),
            technology_stack: Vec::new(),
            success_count: 0,
            failure_count: 0,
            avg_effectiveness: 0.0,
            complexity: 1,
            last_used: now,
            created_at: now,
            metadata: HashMap::new(),
        }
    }

    /// 记录使用结果
    pub fn record_usage(&mut self, successful: bool, effectiveness: f32) {
        if successful {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }

        // 更新平均效果评分
        let total_count = self.success_count + self.failure_count;
        self.avg_effectiveness = (self.avg_effectiveness * (total_count - 1) as f32
            + effectiveness)
            / total_count as f32;

        self.last_used = Utc::now();
    }

    /// 添加适用场景
    pub fn add_scenario(mut self, scenario: String) -> Self {
        self.applicable_scenarios.push(scenario);
        self
    }

    /// 添加技术栈
    pub fn add_technology(mut self, technology: String) -> Self {
        self.technology_stack.push(technology);
        self
    }
}
