use std::collections::HashMap;
use crate::core::{ModelInfo, ArchitectureType, TaskType};

/// Model registry for managing model configurations
pub struct ModelRegistry {
    models: HashMap<String, ModelProfile>,
    custom_models: HashMap<String, ModelInfo>,
}

/// Model profile containing information about a specific model
#[derive(Debug, Clone)]
pub struct ModelProfile {
    pub name: String,
    pub variants: Vec<ModelVariant>,
    pub description: String,
    pub architecture_family: String,
    pub supported_tasks: Vec<TaskType>,
    pub default_variant: String,
}

/// Model variant configuration
#[derive(Debug, Clone)]
pub struct ModelVariant {
    pub name: String,
    pub version: String,
    pub model_info: ModelInfo,
    pub performance_metrics: Option<PerformanceMetrics>,
    pub recommended_use_cases: Vec<String>,
}

/// Performance metrics for model variants
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub inference_time_ms: f32,
    pub memory_usage_mb: usize,
    pub accuracy_metrics: AccuracyMetrics,
    pub hardware_requirements: HardwareRequirements,
}

/// Accuracy metrics
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    pub map_50: Option<f32>,      // mAP@0.5
    pub map_50_95: Option<f32>,   // mAP@0.5:0.95
    pub precision: Option<f32>,
    pub recall: Option<f32>,
    pub f1_score: Option<f32>,
}

/// Hardware requirements
#[derive(Debug, Clone)]
pub struct HardwareRequirements {
    pub min_memory_gb: f32,
    pub recommended_memory_gb: f32,
    pub supports_gpu: bool,
    pub min_compute_capability: Option<String>,
}

impl ModelRegistry {
    /// Create a new model registry with built-in model profiles
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
            custom_models: HashMap::new(),
        };
        
        // Load built-in model profiles
        registry.load_builtin_profiles();
        registry
    }

    /// Load built-in model profiles
    fn load_builtin_profiles(&mut self) {
        // YOLOv8 family
        self.add_yolov8_profiles();
        
        // YOLOv5 family
        self.add_yolov5_profiles();
        
        // SSD family
        self.add_ssd_profiles();
        
        // RetinaNet family
        self.add_retinanet_profiles();
        
        // Mask R-CNN family
        self.add_mask_rcnn_profiles();
    }

    /// Add YOLOv8 model profiles
    fn add_yolov8_profiles(&mut self) {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::add_yolov8_profiles implementation")
    }

    /// Add YOLOv5 model profiles
    fn add_yolov5_profiles(&mut self) {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::add_yolov5_profiles implementation")
    }

    /// Add SSD model profiles
    fn add_ssd_profiles(&mut self) {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::add_ssd_profiles implementation")
    }

    /// Add RetinaNet model profiles
    fn add_retinanet_profiles(&mut self) {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::add_retinanet_profiles implementation")
    }

    /// Add Mask R-CNN model profiles
    fn add_mask_rcnn_profiles(&mut self) {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::add_mask_rcnn_profiles implementation")
    }

    /// Get model profile by name
    pub fn get_profile(&self, name: &str) -> Option<&ModelProfile> {
        self.models.get(name)
    }

    /// Get model variant
    pub fn get_variant(&self, model_name: &str, variant_name: &str) -> Option<&ModelVariant> {
        self.models
            .get(model_name)?
            .variants
            .iter()
            .find(|v| v.name == variant_name)
    }

    /// Get default variant for a model
    pub fn get_default_variant(&self, model_name: &str) -> Option<&ModelVariant> {
        let profile = self.models.get(model_name)?;
        self.get_variant(model_name, &profile.default_variant)
    }

    /// Register a custom model
    pub fn register_custom_model(&mut self, name: String, model_info: ModelInfo) {
        self.custom_models.insert(name, model_info);
    }

    /// Get custom model info
    pub fn get_custom_model(&self, name: &str) -> Option<&ModelInfo> {
        self.custom_models.get(name)
    }

    /// List all available models
    pub fn list_models(&self) -> Vec<String> {
        let mut models: Vec<String> = self.models.keys().cloned().collect();
        models.extend(self.custom_models.keys().cloned());
        models.sort();
        models
    }

    /// Search models by criteria
    pub fn search_models(&self, criteria: &SearchCriteria) -> Vec<&ModelProfile> {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::search_models implementation")
    }

    /// Load models from configuration file
    pub fn load_from_file(&mut self, path: &str) -> crate::error::Result<()> {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::load_from_file implementation")
    }

    /// Save models to configuration file
    pub fn save_to_file(&self, path: &str) -> crate::error::Result<()> {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::save_to_file implementation")
    }

    /// Validate model configuration
    pub fn validate_model(&self, model_info: &ModelInfo) -> crate::error::Result<()> {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::validate_model implementation")
    }

    /// Get model recommendations based on use case
    pub fn recommend_models(&self, use_case: &UseCase) -> Vec<ModelRecommendation> {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::recommend_models implementation")
    }
}

/// Search criteria for finding models
#[derive(Debug, Clone)]
pub struct SearchCriteria {
    pub architecture_type: Option<ArchitectureType>,
    pub supported_tasks: Vec<TaskType>,
    pub max_memory_mb: Option<usize>,
    pub min_accuracy: Option<f32>,
    pub max_inference_time_ms: Option<f32>,
    pub hardware_constraints: Option<HardwareRequirements>,
}

/// Use case specification for model recommendations
#[derive(Debug, Clone)]
pub struct UseCase {
    pub primary_task: TaskType,
    pub performance_priority: PerformancePriority,
    pub accuracy_requirement: AccuracyRequirement,
    pub deployment_target: DeploymentTarget,
    pub expected_throughput: Option<f32>, // images per second
}

/// Performance priority options
#[derive(Debug, Clone)]
pub enum PerformancePriority {
    Speed,
    Accuracy,
    Balanced,
    MemoryEfficiency,
}

/// Accuracy requirement levels
#[derive(Debug, Clone)]
pub enum AccuracyRequirement {
    Basic,      // Good enough for demos
    Production, // Production quality
    Research,   // State-of-the-art
}

/// Deployment target options
#[derive(Debug, Clone)]
pub enum DeploymentTarget {
    Cloud,
    Edge,
    Mobile,
    Embedded,
    Desktop,
}

/// Model recommendation
#[derive(Debug, Clone)]
pub struct ModelRecommendation {
    pub model_name: String,
    pub variant_name: String,
    pub confidence_score: f32,
    pub reasoning: String,
    pub trade_offs: String,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global model registry instance
static mut GLOBAL_REGISTRY: Option<ModelRegistry> = None;
static REGISTRY_INIT: std::sync::Once = std::sync::Once::new();

/// Get the global model registry instance
pub fn global_registry() -> &'static ModelRegistry {
    unsafe {
        REGISTRY_INIT.call_once(|| {
            GLOBAL_REGISTRY = Some(ModelRegistry::new());
        });
        GLOBAL_REGISTRY.as_ref().unwrap()
    }
}

/// Model configuration loader utilities
pub mod config_loader {
    use super::*;

    /// Load model configuration from YAML
    pub fn load_yaml_config(path: &str) -> crate::error::Result<ModelInfo> {
        // Implementation will be added in the actual build
        todo!("load_yaml_config implementation")
    }

    /// Load model configuration from JSON
    pub fn load_json_config(path: &str) -> crate::error::Result<ModelInfo> {
        // Implementation will be added in the actual build
        todo!("load_json_config implementation")
    }

    /// Auto-detect model type from ONNX file
    pub fn auto_detect_model_type(model_path: &str) -> crate::error::Result<String> {
        // Implementation will be added in the actual build
        todo!("auto_detect_model_type implementation")
    }

    /// Generate default configuration for detected model type
    pub fn generate_default_config(model_type: &str) -> crate::error::Result<ModelInfo> {
        // Implementation will be added in the actual build
        todo!("generate_default_config implementation")
    }

    /// Validate configuration against model file
    pub fn validate_config_compatibility(
        config: &ModelInfo,
        model_path: &str,
    ) -> crate::error::Result<()> {
        // Implementation will be added in the actual build
        todo!("validate_config_compatibility implementation")
    }
}