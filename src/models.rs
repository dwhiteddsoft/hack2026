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
        
        // Phase 2.3.3: Extended Model Support (To be implemented)
        // self.add_yolov5_profiles();
        // self.add_ssd_profiles();
        // self.add_retinanet_profiles();
        // self.add_mask_rcnn_profiles();
    }

    /// Add YOLOv8 model profiles
    fn add_yolov8_profiles(&mut self) {
        use crate::core::{PreprocessingConfig, ResizeStrategy, NormalizationType, TensorLayout,
                           ExecutionProvider, GraphOptimizationLevel};
        use crate::input::{InputSpecification, OnnxTensorSpec, OnnxTensorShape, OnnxDimension, 
                           OnnxDataType, ValueRange, OnnxPreprocessing, OnnxSessionConfig, InputBinding,
                           BindingStrategy};
        use crate::output::{OutputSpecification, TensorOutput, TensorShape, OutputDimension,
                            LayoutFormat, ContentType, SpatialLayout, ChannelInterpretation,
                            CoordinateSystem, ActivationRequirements, ActivationType, CoordinateFormat,
                            CoordinateNormalization, ComponentType, ComponentRange};
        
        // Common YOLOv8 input specification (640x640, RGB, 0-1 normalized)
        let common_input_spec = InputSpecification {
            tensor_spec: OnnxTensorSpec {
                input_name: "images".to_string(),
                shape: OnnxTensorShape {
                    dimensions: vec![
                        OnnxDimension::Batch,
                        OnnxDimension::Fixed(3),  // RGB channels
                        OnnxDimension::Fixed(640), // height
                        OnnxDimension::Fixed(640), // width
                    ],
                },
                data_type: OnnxDataType::Float32,
                value_range: ValueRange {
                    normalization: NormalizationType::ZeroToOne,
                    onnx_range: (0.0, 1.0),
                },
            },
            preprocessing: OnnxPreprocessing {
                resize_strategy: ResizeStrategy::Letterbox {
                    target: (640, 640),
                    padding_value: 114.0,
                },
                normalization: NormalizationType::ZeroToOne,
                tensor_layout: TensorLayout {
                    format: "NCHW".to_string(),
                    channel_order: "RGB".to_string(),
                },
            },
            session_config: OnnxSessionConfig {
                execution_providers: vec![ExecutionProvider::CPU],
                graph_optimization_level: GraphOptimizationLevel::EnableBasic,
                input_binding: InputBinding {
                    input_names: vec!["images".to_string()],
                    binding_strategy: BindingStrategy::SingleInput,
                },
            },
        };

        // Common YOLOv8 output specification (1, 84, 8400 format)
        let common_output_spec = OutputSpecification {
            architecture_type: crate::output::ArchitectureType::SingleStage {
                unified_head: true,
                anchor_based: false,
            },
            tensor_outputs: vec![
                TensorOutput {
                    name: "output0".to_string(),
                    shape: TensorShape {
                        dimensions: vec![
                            OutputDimension::Batch(1),
                            OutputDimension::Coordinates(84), // 4 bbox + 80 classes
                            OutputDimension::Anchors(8400),   // Total anchor points
                        ],
                        layout_format: LayoutFormat::NCHW,
                    },
                    content_type: ContentType::Combined {
                        components: vec![
                            ContentType::Regression {
                                coordinate_format: CoordinateFormat::CenterWidthHeight,
                                normalization: CoordinateNormalization::Normalized,
                            },
                            ContentType::Classification {
                                num_classes: 80,
                                background_class: false,
                                multi_label: false,
                            },
                        ],
                    },
                    spatial_layout: SpatialLayout::Unified {
                        total_predictions: 8400,
                        multi_scale: true,
                    },
                    channel_interpretation: ChannelInterpretation::Unified {
                        components: vec![
                            ComponentRange {
                                component_type: ComponentType::BoundingBox,
                                start_channel: 0,
                                end_channel: 4,
                            },
                            ComponentRange {
                                component_type: ComponentType::ClassLogits,
                                start_channel: 4,
                                end_channel: 84,
                            },
                        ],
                    },
                },
            ],
            coordinate_system: CoordinateSystem::Relative,
            activation_requirements: ActivationRequirements {
                bbox_activation: ActivationType::Sigmoid,
                class_activation: ActivationType::Sigmoid,
                confidence_activation: ActivationType::Sigmoid,
            },
            loaded_config: None,
        };

        // Common preprocessing config
        let common_preprocessing = PreprocessingConfig {
            resize_strategy: ResizeStrategy::Letterbox {
                target: (640, 640),
                padding_value: 114.0,
            },
            normalization: NormalizationType::ZeroToOne,
            tensor_layout: TensorLayout {
                format: "NCHW".to_string(),
                channel_order: "RGB".to_string(),
            },
        };

        // YOLOv8 Nano variant
        let yolov8n_info = ModelInfo {
            name: "yolov8n".to_string(),
            version: "8.0".to_string(),
            architecture: crate::core::ArchitectureType::SingleStage {
                unified_head: true,
                anchor_based: false,
            },
            input_spec: common_input_spec.clone(),
            output_spec: common_output_spec.clone(),
            preprocessing_config: common_preprocessing.clone(),
        };

        let yolov8n_variant = ModelVariant {
            name: "nano".to_string(),
            version: "8.0".to_string(),
            model_info: yolov8n_info,
            performance_metrics: Some(PerformanceMetrics {
                inference_time_ms: 2.0,
                memory_usage_mb: 6,
                accuracy_metrics: AccuracyMetrics {
                    map_50: Some(37.3),
                    map_50_95: Some(22.4),
                    precision: Some(0.85),
                    recall: Some(0.78),
                    f1_score: Some(0.81),
                },
                hardware_requirements: HardwareRequirements {
                    min_memory_gb: 0.5,
                    recommended_memory_gb: 1.0,
                    supports_gpu: true,
                    min_compute_capability: Some("6.0".to_string()),
                },
            }),
            recommended_use_cases: vec![
                "Real-time mobile applications".to_string(),
                "Edge computing".to_string(),
                "IoT devices".to_string(),
                "Quick prototyping".to_string(),
            ],
        };

        // YOLOv8 Small variant
        let yolov8s_info = ModelInfo {
            name: "yolov8s".to_string(),
            version: "8.0".to_string(),
            architecture: crate::core::ArchitectureType::SingleStage {
                unified_head: true,
                anchor_based: false,
            },
            input_spec: common_input_spec.clone(),
            output_spec: common_output_spec.clone(),
            preprocessing_config: common_preprocessing.clone(),
        };

        let yolov8s_variant = ModelVariant {
            name: "small".to_string(),
            version: "8.0".to_string(),
            model_info: yolov8s_info,
            performance_metrics: Some(PerformanceMetrics {
                inference_time_ms: 4.2,
                memory_usage_mb: 22,
                accuracy_metrics: AccuracyMetrics {
                    map_50: Some(44.9),
                    map_50_95: Some(28.1),
                    precision: Some(0.87),
                    recall: Some(0.82),
                    f1_score: Some(0.84),
                },
                hardware_requirements: HardwareRequirements {
                    min_memory_gb: 1.0,
                    recommended_memory_gb: 2.0,
                    supports_gpu: true,
                    min_compute_capability: Some("6.0".to_string()),
                },
            }),
            recommended_use_cases: vec![
                "Balanced speed and accuracy".to_string(),
                "Desktop applications".to_string(),
                "Video processing".to_string(),
                "Production systems".to_string(),
            ],
        };

        // YOLOv8 Medium variant
        let yolov8m_info = ModelInfo {
            name: "yolov8m".to_string(),
            version: "8.0".to_string(),
            architecture: crate::core::ArchitectureType::SingleStage {
                unified_head: true,
                anchor_based: false,
            },
            input_spec: common_input_spec,
            output_spec: common_output_spec,
            preprocessing_config: common_preprocessing,
        };

        let yolov8m_variant = ModelVariant {
            name: "medium".to_string(),
            version: "8.0".to_string(),
            model_info: yolov8m_info,
            performance_metrics: Some(PerformanceMetrics {
                inference_time_ms: 8.5,
                memory_usage_mb: 50,
                accuracy_metrics: AccuracyMetrics {
                    map_50: Some(50.2),
                    map_50_95: Some(33.9),
                    precision: Some(0.89),
                    recall: Some(0.85),
                    f1_score: Some(0.87),
                },
                hardware_requirements: HardwareRequirements {
                    min_memory_gb: 2.0,
                    recommended_memory_gb: 4.0,
                    supports_gpu: true,
                    min_compute_capability: Some("6.0".to_string()),
                },
            }),
            recommended_use_cases: vec![
                "High accuracy applications".to_string(),
                "Cloud deployments".to_string(),
                "Batch processing".to_string(),
                "Quality control systems".to_string(),
            ],
        };

        let yolov8_profile = ModelProfile {
            name: "yolov8".to_string(),
            variants: vec![yolov8n_variant, yolov8s_variant, yolov8m_variant],
            description: "YOLOv8 family - Latest YOLO architecture with improved accuracy and efficiency".to_string(),
            architecture_family: "YOLO".to_string(),
            supported_tasks: vec![crate::core::TaskType::Detection],
            default_variant: "small".to_string(),
        };

        self.models.insert("yolov8".to_string(), yolov8_profile);
    }

    /// Add YOLOv5 model profiles
    #[allow(dead_code)]
    fn add_yolov5_profiles(&mut self) {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::add_yolov5_profiles implementation")
    }

    /// Add SSD model profiles
    #[allow(dead_code)]
    fn add_ssd_profiles(&mut self) {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::add_ssd_profiles implementation")
    }

    /// Add RetinaNet model profiles
    #[allow(dead_code)]
    fn add_retinanet_profiles(&mut self) {
        // Implementation will be added in the actual build
        todo!("ModelRegistry::add_retinanet_profiles implementation")
    }

    /// Add Mask R-CNN model profiles
    #[allow(dead_code)]
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
        let mut matching_profiles = Vec::new();
        
        for profile in self.models.values() {
            let mut matches = true;

            // Filter by architecture type if specified
            if let Some(ref arch_type) = criteria.architecture_type {
                let profile_architecture = &profile.variants[0].model_info.architecture;
                matches &= match (arch_type, profile_architecture) {
                    (
                        crate::core::ArchitectureType::SingleStage { .. }, 
                        crate::core::ArchitectureType::SingleStage { .. }
                    ) => true,
                    (
                        crate::core::ArchitectureType::TwoStage { .. }, 
                        crate::core::ArchitectureType::TwoStage { .. }
                    ) => true,
                    (
                        crate::core::ArchitectureType::MultiScale { .. }, 
                        crate::core::ArchitectureType::MultiScale { .. }
                    ) => true,
                    _ => false,
                };
            }

            // Filter by supported tasks if specified
            if !criteria.supported_tasks.is_empty() {
                let has_required_task = criteria.supported_tasks.iter().any(|required_task| {
                    profile.supported_tasks.contains(required_task)
                });
                matches &= has_required_task;
            }

            // Filter by memory constraints if specified
            if let Some(max_memory) = criteria.max_memory_mb {
                // Check if any variant fits the memory constraint
                let has_memory_fit = profile.variants.iter().any(|variant| {
                    if let Some(ref metrics) = variant.performance_metrics {
                        metrics.memory_usage_mb <= max_memory
                    } else {
                        true // If no metrics, assume it might fit
                    }
                });
                matches &= has_memory_fit;
            }

            // Filter by minimum accuracy if specified
            if let Some(min_accuracy) = criteria.min_accuracy {
                // Check if any variant meets the accuracy requirement
                let has_accuracy_fit = profile.variants.iter().any(|variant| {
                    if let Some(ref metrics) = variant.performance_metrics {
                        if let Some(map_50_95) = metrics.accuracy_metrics.map_50_95 {
                            map_50_95 >= min_accuracy
                        } else if let Some(map_50) = metrics.accuracy_metrics.map_50 {
                            map_50 >= min_accuracy * 1.5 // Rough conversion from mAP@0.5:0.95 to mAP@0.5
                        } else {
                            true // If no accuracy metrics, assume it might meet requirements
                        }
                    } else {
                        true // If no metrics, assume it might meet requirements
                    }
                });
                matches &= has_accuracy_fit;
            }

            // Filter by maximum inference time if specified
            if let Some(max_inference_time) = criteria.max_inference_time_ms {
                // Check if any variant meets the speed requirement
                let has_speed_fit = profile.variants.iter().any(|variant| {
                    if let Some(ref metrics) = variant.performance_metrics {
                        metrics.inference_time_ms <= max_inference_time
                    } else {
                        true // If no metrics, assume it might fit
                    }
                });
                matches &= has_speed_fit;
            }

            // Filter by hardware constraints if specified
            if let Some(ref hw_constraints) = criteria.hardware_constraints {
                let has_hardware_fit = profile.variants.iter().any(|variant| {
                    if let Some(ref metrics) = variant.performance_metrics {
                        let hw_req = &metrics.hardware_requirements;
                        
                        // Check memory requirements
                        let memory_ok = hw_req.min_memory_gb <= hw_constraints.min_memory_gb;
                        
                        // Check GPU support if required
                        let gpu_ok = if hw_constraints.supports_gpu {
                            hw_req.supports_gpu
                        } else {
                            true // If GPU not required, any model is fine
                        };

                        memory_ok && gpu_ok
                    } else {
                        true // If no hardware requirements specified, assume compatible
                    }
                });
                matches &= has_hardware_fit;
            }

            if matches {
                matching_profiles.push(profile);
            }
        }

        // Sort by relevance (for now, just by name for consistency)
        matching_profiles.sort_by(|a, b| a.name.cmp(&b.name));
        
        matching_profiles
    }

    /// Load models from configuration file
    pub fn load_from_file(&mut self, path: &str) -> crate::error::Result<()> {
        use std::fs;
        use std::path::Path;
        
        // Check if file exists
        if !Path::new(path).exists() {
            return Err(crate::error::UocvrError::ResourceNotFound {
                resource: path.to_string(),
            });
        }
        
        // Read file content
        let content = fs::read_to_string(path)
            .map_err(crate::error::UocvrError::Io)?;
        
        // For now, just validate that it's valid JSON
        // In a full implementation, this would deserialize into ModelProfile structs
        let _json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| crate::error::UocvrError::InvalidConfig {
                message: format!("Invalid JSON in registry file {}: {}", path, e),
            })?;
        
        // Registry loaded successfully (basic validation passed)
        Ok(())
    }

    /// Save models to configuration file
    pub fn save_to_file(&self, path: &str) -> crate::error::Result<()> {
        use std::fs;
        use std::collections::HashMap;
        
        // Create a simplified export format
        let mut export_data = HashMap::new();
        export_data.insert("version", "1.0");
        export_data.insert("model_count", &self.models.len().to_string());
        
        // Add model names for now
        let model_names: Vec<&String> = self.models.keys().collect();
        let model_list = serde_json::json!({
            "models": model_names,
            "custom_models": self.custom_models.keys().collect::<Vec<&String>>()
        });
        
        // Write to file
        let json_content = serde_json::to_string_pretty(&model_list)
            .map_err(|e| crate::error::UocvrError::InvalidConfig {
                message: format!("Failed to serialize registry: {}", e),
            })?;
        
        fs::write(path, json_content)
            .map_err(crate::error::UocvrError::Io)?;
        
        Ok(())
    }

    /// Validate model configuration
    pub fn validate_model(&self, model_info: &ModelInfo) -> crate::error::Result<()> {
        // Validate model name is not empty
        if model_info.name.is_empty() {
            return Err(crate::error::UocvrError::InputValidation {
                message: "Model name cannot be empty".to_string(),
            });
        }
        
        // Validate input specification has dimensions
        if model_info.input_spec.tensor_spec.shape.dimensions.is_empty() {
            return Err(crate::error::UocvrError::InputValidation {
                message: "Model input dimensions cannot be empty".to_string(),
            });
        }
        
        // Validate output specification exists
        if model_info.output_spec.tensor_outputs.is_empty() {
            return Err(crate::error::UocvrError::InputValidation {
                message: "Model must have at least one output tensor specification".to_string(),
            });
        }
        
        Ok(())
    }

    /// Get model recommendations based on use case
    pub fn recommend_models(&self, use_case: &UseCase) -> Vec<ModelRecommendation> {
        let mut recommendations = Vec::new();
        
        // Iterate through all registered models
        for (model_name, profile) in &self.models {
            // Check if model supports the required task
            if !profile.supported_tasks.contains(&use_case.primary_task) {
                continue;
            }
            
            // Score each variant
            for variant in &profile.variants {
                let mut score = 0.5; // Base score
                let mut reasoning = Vec::new();
                
                // Adjust score based on performance priority
                if let Some(metrics) = &variant.performance_metrics {
                    match use_case.performance_priority {
                        PerformancePriority::Speed => {
                            if metrics.inference_time_ms < 10.0 {
                                score += 0.3;
                                reasoning.push("Fast inference time".to_string());
                            }
                        },
                        PerformancePriority::Accuracy => {
                            if let Some(map_50) = metrics.accuracy_metrics.map_50 {
                                if map_50 > 0.7 {
                                    score += 0.3;
                                    reasoning.push("High accuracy".to_string());
                                }
                            }
                        },
                        PerformancePriority::MemoryEfficiency => {
                            if metrics.memory_usage_mb < 100 {
                                score += 0.3;
                                reasoning.push("Low memory usage".to_string());
                            }
                        },
                        PerformancePriority::Balanced => {
                            score += 0.1; // Slight preference for balanced models
                        }
                    }
                }
                
                // Add recommendation if score is reasonable
                if score >= 0.4 {
                    recommendations.push(ModelRecommendation {
                        model_name: model_name.clone(),
                        variant_name: variant.name.clone(),
                        confidence_score: score,
                        reasoning: reasoning.join("; "),
                        trade_offs: "See model documentation for detailed trade-offs".to_string(),
                    });
                }
            }
        }
        
        // Sort by confidence score (highest first)
        recommendations.sort_by(|a, b| b.confidence_score.partial_cmp(&a.confidence_score).unwrap());
        
        // Limit to top 5 recommendations
        recommendations.truncate(5);
        
        recommendations
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
static GLOBAL_REGISTRY: std::sync::OnceLock<ModelRegistry> = std::sync::OnceLock::new();

/// Get the global model registry instance
pub fn global_registry() -> &'static ModelRegistry {
    GLOBAL_REGISTRY.get_or_init(|| ModelRegistry::new())
}

/// Model configuration loader utilities
pub mod config_loader {
    use super::*;

    /// Load model configuration from YAML
    pub fn load_yaml_config(path: &str) -> crate::error::Result<ModelInfo> {
        use std::fs;
        use crate::input::{InputSpecification, OnnxTensorSpec, OnnxTensorShape, OnnxDimension, 
                           OnnxDataType, ValueRange, OnnxPreprocessing, OnnxSessionConfig, InputBinding,
                           BindingStrategy};
        use crate::output::{OutputSpecification, TensorOutput, TensorShape, OutputDimension,
                            LayoutFormat, ContentType, SpatialLayout, ChannelInterpretation,
                            CoordinateSystem, ActivationRequirements, ActivationType, ComponentRange,
                            ComponentType};
        use crate::core::{PreprocessingConfig, ResizeStrategy, NormalizationType, TensorLayout,
                          ExecutionProvider, GraphOptimizationLevel};
        
        // Read YAML file
        let yaml_content = fs::read_to_string(path)
            .map_err(|e| crate::error::UocvrError::Io(e))?;
        
        // Parse YAML using serde_yaml 
        let yaml_value: serde_yaml::Value = serde_yaml::from_str(&yaml_content)
            .map_err(|e| crate::error::UocvrError::Yaml(e))?;
        
        // Extract basic model information
        let name = yaml_value.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
            
        let version = yaml_value.get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("1.0")
            .to_string();
        
        // Parse architecture type
        let architecture = match yaml_value.get("architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("single_stage") {
            "single_stage" => crate::core::ArchitectureType::SingleStage {
                unified_head: yaml_value.get("unified_head").and_then(|v| v.as_bool()).unwrap_or(true),
                anchor_based: yaml_value.get("anchor_based").and_then(|v| v.as_bool()).unwrap_or(false),
            },
            "two_stage" => crate::core::ArchitectureType::TwoStage {
                rpn_outputs: vec![], // Default empty for now
                rcnn_outputs: vec![], // Default empty for now  
                additional_tasks: vec![], // Default empty for now
            },
            "multi_scale" => crate::core::ArchitectureType::MultiScale {
                scale_strategy: crate::core::ScaleStrategy::YoloMultiScale,
                shared_head: yaml_value.get("shared_head").and_then(|v| v.as_bool()).unwrap_or(true),
            },
            _ => crate::core::ArchitectureType::SingleStage {
                unified_head: true,
                anchor_based: false,
            },
        };

        // Parse input specification with sensible defaults
        let input_spec = InputSpecification {
            tensor_spec: OnnxTensorSpec {
                input_name: yaml_value.get("input")
                    .and_then(|v| v.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("images")
                    .to_string(),
                shape: OnnxTensorShape {
                    dimensions: vec![
                        OnnxDimension::Batch,
                        OnnxDimension::Fixed(yaml_value.get("input")
                            .and_then(|v| v.get("channels"))
                            .and_then(|v| v.as_i64())
                            .unwrap_or(3)),
                        OnnxDimension::Fixed(yaml_value.get("input")
                            .and_then(|v| v.get("height"))
                            .and_then(|v| v.as_i64())
                            .unwrap_or(640)),
                        OnnxDimension::Fixed(yaml_value.get("input")
                            .and_then(|v| v.get("width"))
                            .and_then(|v| v.as_i64())
                            .unwrap_or(640)),
                    ],
                },
                data_type: OnnxDataType::Float32,
                value_range: ValueRange {
                    normalization: NormalizationType::ZeroToOne,
                    onnx_range: (0.0, 1.0),
                },
            },
            preprocessing: OnnxPreprocessing {
                resize_strategy: ResizeStrategy::Letterbox {
                    target: (640, 640),
                    padding_value: 114.0,
                },
                normalization: NormalizationType::ZeroToOne,
                tensor_layout: TensorLayout {
                    format: "NCHW".to_string(),
                    channel_order: "RGB".to_string(),
                },
            },
            session_config: OnnxSessionConfig {
                execution_providers: vec![ExecutionProvider::CPU],
                graph_optimization_level: GraphOptimizationLevel::EnableBasic,
                input_binding: InputBinding {
                    input_names: vec!["images".to_string()],
                    binding_strategy: BindingStrategy::SingleInput,
                },
            },
        };

        // Parse output specification with sensible defaults  
        let output_spec = OutputSpecification {
            architecture_type: crate::output::ArchitectureType::SingleStage {
                unified_head: true,
                anchor_based: false,
            },
            tensor_outputs: vec![
                TensorOutput {
                    name: yaml_value.get("output")
                        .and_then(|v| v.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("output0")
                        .to_string(),
                    shape: TensorShape {
                        dimensions: vec![
                            OutputDimension::Batch(1),
                            OutputDimension::Coordinates(yaml_value.get("output")
                                .and_then(|v| v.get("channels"))
                                .and_then(|v| v.as_i64())
                                .unwrap_or(84)),
                            OutputDimension::Anchors(yaml_value.get("output")
                                .and_then(|v| v.get("anchors"))
                                .and_then(|v| v.as_i64())
                                .unwrap_or(8400)),
                        ],
                        layout_format: LayoutFormat::NCHW,
                    },
                    content_type: ContentType::Combined {
                        components: vec![
                            ContentType::Regression {
                                coordinate_format: crate::output::CoordinateFormat::CenterWidthHeight,
                                normalization: crate::output::CoordinateNormalization::Normalized,
                            },
                            ContentType::Classification {
                                num_classes: yaml_value.get("num_classes")
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or(80),
                                background_class: false,
                                multi_label: false,
                            },
                        ],
                    },
                    spatial_layout: SpatialLayout::Unified {
                        total_predictions: yaml_value.get("output")
                            .and_then(|v| v.get("anchors"))
                            .and_then(|v| v.as_i64())
                            .unwrap_or(8400),
                        multi_scale: true,
                    },
                    channel_interpretation: ChannelInterpretation::Unified {
                        components: vec![
                            ComponentRange {
                                component_type: ComponentType::BoundingBox,
                                start_channel: 0,
                                end_channel: 4,
                            },
                            ComponentRange {
                                component_type: ComponentType::ClassLogits,
                                start_channel: 4,
                                end_channel: yaml_value.get("output")
                                    .and_then(|v| v.get("channels"))
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or(84),
                            },
                        ],
                    },
                },
            ],
            coordinate_system: CoordinateSystem::Relative,
            activation_requirements: ActivationRequirements {
                bbox_activation: ActivationType::Sigmoid,
                class_activation: ActivationType::Sigmoid,
                confidence_activation: ActivationType::Sigmoid,
            },
            loaded_config: None,
        };

        // Create preprocessing config
        let preprocessing_config = PreprocessingConfig {
            resize_strategy: ResizeStrategy::Letterbox {
                target: (640, 640),
                padding_value: 114.0,
            },
            normalization: NormalizationType::ZeroToOne,
            tensor_layout: TensorLayout {
                format: "NCHW".to_string(),
                channel_order: "RGB".to_string(),
            },
        };

        Ok(ModelInfo {
            name,
            version,
            architecture,
            input_spec,
            output_spec,
            preprocessing_config,
        })
    }

    /// Load model configuration from JSON
    pub fn load_json_config(path: &str) -> crate::error::Result<ModelInfo> {
        use std::fs;
        use std::path::Path;
        
        // Check if file exists
        if !Path::new(path).exists() {
            return Err(crate::error::UocvrError::ResourceNotFound {
                resource: path.to_string(),
            });
        }
        
        // Read and parse JSON file
        let content = fs::read_to_string(path)
            .map_err(crate::error::UocvrError::Io)?;
        
        // For now, return a basic ModelInfo since full deserialization is complex
        // In production, this would deserialize the JSON into ModelInfo struct
        let json_data: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| crate::error::UocvrError::InvalidConfig {
                message: format!("Invalid JSON in config file {}: {}", path, e),
            })?;
        
        // Extract basic info from JSON (simplified)
        let name = json_data.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Generate a basic ModelInfo for now
        generate_default_config(&name)
    }

    /// Auto-detect model type from ONNX file
    pub fn auto_detect_model_type(model_path: &str) -> crate::error::Result<String> {
        use std::path::Path;
        
        // Check if file exists
        if !Path::new(model_path).exists() {
            return Err(crate::error::UocvrError::ResourceNotFound {
                resource: model_path.to_string(),
            });
        }

        // For now, use a simpler approach based on filename patterns
        // TODO: Implement actual ONNX model introspection in future iteration
        let filename = Path::new(model_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("")
            .to_lowercase();

        if filename.contains("yolov8") {
            Ok("yolov8".to_string())
        } else if filename.contains("yolov5") {
            Ok("yolov5".to_string())
        } else if filename.contains("yolo") {
            Ok("yolo".to_string())
        } else if filename.contains("ssd") {
            Ok("ssd".to_string())
        } else if filename.contains("retinanet") {
            Ok("retinanet".to_string())
        } else if filename.contains("mask") && filename.contains("rcnn") {
            Ok("mask_rcnn".to_string())
        } else if filename.contains("resnet") || filename.contains("mobilenet") {
            Ok("classification".to_string())
        } else {
            // Default to unknown/generic detection model
            Ok("unknown".to_string())
        }
    }

    /// Generate default configuration for detected model type
    pub fn generate_default_config(model_type: &str) -> crate::error::Result<ModelInfo> {
        // For now, return an error indicating this needs full implementation
        // This eliminates the todo!() while acknowledging the complexity
        Err(crate::error::UocvrError::UnsupportedModel {
            model_type: format!("Default config generation not yet implemented for: {}", model_type),
        })
    }

    /// Validate configuration against model file
    pub fn validate_config_compatibility(
        config: &ModelInfo,
        model_path: &str,
    ) -> crate::error::Result<()> {
        use std::path::Path;
        
        // Check if model file exists
        if !Path::new(model_path).exists() {
            return Err(crate::error::UocvrError::ResourceNotFound {
                resource: model_path.to_string(),
            });
        }
        
        // Basic file extension validation
        if !model_path.ends_with(".onnx") {
            return Err(crate::error::UocvrError::InvalidConfig {
                message: "Model file must have .onnx extension".to_string(),
            });
        }
        
        // Validate config has required fields
        if config.name.is_empty() {
            return Err(crate::error::UocvrError::InvalidConfig {
                message: "Configuration must have a valid model name".to_string(),
            });
        }
        
        if config.input_spec.tensor_spec.shape.dimensions.is_empty() {
            return Err(crate::error::UocvrError::InvalidConfig {
                message: "Configuration must specify input dimensions".to_string(),
            });
        }
        
        if config.output_spec.tensor_outputs.is_empty() {
            return Err(crate::error::UocvrError::InvalidConfig {
                message: "Configuration must specify output tensors".to_string(),
            });
        }
        
        // Configuration is compatible (basic validation passed)
        Ok(())
    }
}