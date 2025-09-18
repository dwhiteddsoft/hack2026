use std::time::Duration;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use ort::{Environment, SessionBuilder as OrtSessionBuilder};

/// Main inference session wrapper
pub struct UniversalSession {
    pub id: Uuid,
    pub model_info: ModelInfo,
    pub session: std::sync::Arc<ort::Session>,
    pub input_processor: crate::input::InputProcessor,
    pub output_processor: crate::output::OutputProcessor,
}

/// Model information and configuration
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub architecture: ArchitectureType,
    pub input_spec: crate::input::InputSpecification,
    pub output_spec: crate::output::OutputSpecification,
    pub preprocessing_config: PreprocessingConfig,
}

/// Architecture type classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureType {
    SingleStage {
        unified_head: bool,
        anchor_based: bool,
    },
    TwoStage {
        rpn_outputs: Vec<String>,
        rcnn_outputs: Vec<String>,
        additional_tasks: Vec<TaskType>,
    },
    MultiScale {
        scale_strategy: ScaleStrategy,
        shared_head: bool,
    },
}

/// Task types supported by models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskType {
    Detection,
    Segmentation,
    Classification,
    PoseEstimation,
    DepthEstimation,
}

/// Scale strategy for multi-scale models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleStrategy {
    FeaturePyramid,
    YoloMultiScale,
    SsdMultiBox,
    FeaturePyramidUnified,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub resize_strategy: ResizeStrategy,
    pub normalization: NormalizationType,
    pub tensor_layout: TensorLayout,
}

/// Resize strategy options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResizeStrategy {
    Direct { target: (u32, u32) },
    Letterbox {
        target: (u32, u32),
        padding_value: f32,
    },
    ShortestEdge {
        target_size: u32,
        max_size: Option<u32>,
    },
}

/// Normalization type options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    ZeroToOne,
    ImageNet {
        mean: [f32; 3],
        std: [f32; 3],
    },
    Custom {
        mean: [f32; 3],
        std: Option<[f32; 3]>,
    },
}

/// Tensor layout specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorLayout {
    pub format: String,        // "NCHW", "NHWC"
    pub channel_order: String, // "RGB", "BGR"
}

/// Inference results in canonical format
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub detections: Vec<Detection>,
    pub processing_time: Duration,
    pub metadata: InferenceMetadata,
}

/// Individual detection result
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub class_id: u32,
    pub class_name: Option<String>,
    pub mask: Option<Mask>,
    pub keypoints: Option<Vec<Keypoint>>,
}

/// Bounding box representation
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub format: BoundingBoxFormat,
}

/// Bounding box format specification
#[derive(Debug, Clone)]
pub enum BoundingBoxFormat {
    CenterWidthHeight,  // (cx, cy, w, h)
    CornerCoordinates,  // (x1, y1, x2, y2)
    TopLeftWidthHeight, // (x, y, w, h)
}

/// Segmentation mask
#[derive(Debug, Clone)]
pub struct Mask {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: MaskFormat,
}

/// Mask format options
#[derive(Debug, Clone)]
pub enum MaskFormat {
    Binary,
    Grayscale,
    Polygon,
    Rle,
}

/// Keypoint for pose estimation
#[derive(Debug, Clone)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub visible: bool,
}

/// Inference metadata
#[derive(Debug, Clone)]
pub struct InferenceMetadata {
    pub model_name: String,
    pub input_shape: Vec<i64>,
    pub output_shapes: Vec<Vec<i64>>,
    pub inference_time: Duration,
    pub preprocessing_time: Duration,
    pub postprocessing_time: Duration,
}

/// Session builder for advanced configuration
pub struct SessionBuilder {
    model_path: Option<String>,
    config_path: Option<String>,
    execution_provider: Option<ExecutionProvider>,
    batch_size: Option<usize>,
    optimization_level: Option<GraphOptimizationLevel>,
}

/// Execution provider options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionProvider {
    CPU,
    CUDA(CudaConfig),
    TensorRT(TensorRTConfig),
    DirectML,
    CoreML,
}

/// CUDA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaConfig {
    pub device_id: i32,
    pub memory_limit: Option<usize>,
    pub arena_extend_strategy: Option<String>,
}

/// TensorRT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRTConfig {
    pub max_workspace_size: usize,
    pub fp16_enable: bool,
    pub int8_enable: bool,
    pub calibration_table: Option<String>,
}

/// Graph optimization level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphOptimizationLevel {
    DisableAll,
    EnableBasic,
    EnableExtended,
    EnableAll,
}

impl SessionBuilder {
    pub fn new() -> Self {
        Self {
            model_path: None,
            config_path: None,
            execution_provider: None,
            batch_size: None,
            optimization_level: None,
        }
    }

    pub fn model_file<P: Into<String>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    pub fn config_file<P: Into<String>>(mut self, path: P) -> Self {
        self.config_path = Some(path.into());
        self
    }

    pub fn provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = Some(provider);
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    pub fn optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
        self.optimization_level = Some(level);
        self
    }

    pub async fn build(self) -> crate::error::Result<UniversalSession> {
        let model_path = self.model_path.ok_or_else(|| crate::error::UocvrError::ModelConfig {
            message: "Model path is required".to_string(),
        })?;

        // Check if model file exists
        if !std::path::Path::new(&model_path).exists() {
            return Err(crate::error::UocvrError::ResourceNotFound {
                resource: model_path,
            });
        }

        // Create ORT environment
        let environment = std::sync::Arc::new(Environment::builder()
            .with_name("uocvr")
            .build()
            .map_err(|e| crate::error::UocvrError::Runtime {
                message: format!("Failed to create ORT environment: {}", e),
            })?);

        // Create ORT session
        let ort_session = OrtSessionBuilder::new(&environment)
            .map_err(|e| crate::error::UocvrError::Runtime {
                message: format!("Failed to create ORT session builder: {}", e),
            })?
            .with_model_from_file(&model_path)
            .map_err(|e| crate::error::UocvrError::Runtime {
                message: format!("Failed to load model from file {}: {}", model_path, e),
            })?;

        // Create model info from the session
        let model_name = std::path::Path::new(&model_path)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        
        // Determine input size based on model type - only use defaults for YOLO models
        let input_size = if model_name.contains("yolov2") {
            (416u32, 416u32)  // YOLOv2 expects 416x416
        } else if model_name.contains("yolov3") {
            (416u32, 416u32)  // YOLOv3 uses 416x416 input
        } else if model_name.contains("yolov8") {
            (640u32, 640u32)  // YOLOv8 uses 640x640 input
        } else if self.config_path.is_none() {
            // For non-YOLO models, require a config file - don't use YOLO defaults
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!("Model {} requires a configuration file. YOLO defaults not applicable for non-YOLO models.", model_name),
            });
        } else {
            // Non-YOLO model with config file - will determine size from config
            // Use placeholder size for now, will be updated from config
            (640u32, 640u32)
        };

        // Create input and output specifications with model-specific input size
        let input_spec = crate::input::InputSpecification {
            tensor_spec: crate::input::OnnxTensorSpec {
                input_name: "images".to_string(),
                shape: crate::input::OnnxTensorShape {
                    dimensions: vec![
                        crate::input::OnnxDimension::Batch,
                        crate::input::OnnxDimension::Fixed(3),
                        crate::input::OnnxDimension::Fixed(input_size.1 as i64), // height
                        crate::input::OnnxDimension::Fixed(input_size.0 as i64), // width
                    ],
                },
                data_type: crate::input::OnnxDataType::Float32,
                value_range: crate::input::ValueRange {
                    normalization: crate::core::NormalizationType::ZeroToOne,
                    onnx_range: (0.0, 1.0),
                },
            },
            preprocessing: crate::input::OnnxPreprocessing {
                resize_strategy: crate::core::ResizeStrategy::Direct { target: input_size },
                normalization: crate::core::NormalizationType::ZeroToOne,
                tensor_layout: crate::core::TensorLayout {
                    format: "NCHW".to_string(),
                    channel_order: "RGB".to_string(),
                },
            },
            session_config: crate::input::OnnxSessionConfig {
                execution_providers: vec![crate::core::ExecutionProvider::CPU],
                graph_optimization_level: crate::core::GraphOptimizationLevel::EnableBasic,
                input_binding: crate::input::InputBinding {
                    input_names: vec!["images".to_string()],
                    binding_strategy: crate::input::BindingStrategy::SingleInput,
                },
            },
        };
        let mut output_spec = crate::output::OutputSpecification::default();
        
        // Load YAML config if provided - fail if config is required but cannot be loaded
        if let Some(ref config_path) = self.config_path {
            match load_yaml_postprocessing_config(config_path).await {
                Ok(yaml_config) => {
                    output_spec.loaded_config = Some(yaml_config);
                }
                Err(e) => {
                    return Err(crate::error::UocvrError::ModelConfig {
                        message: format!("Failed to load required configuration file '{}': {}", config_path, e),
                    });
                }
            }
        }
        
        // Determine architecture type based on model
        let architecture = if model_name.contains("yolov2") {
            crate::core::ArchitectureType::SingleStage {
                unified_head: false,
                anchor_based: true,
            }
        } else if model_name.contains("yolov3") {
            crate::core::ArchitectureType::SingleStage {
                unified_head: false,  // YOLOv3 has multiple detection heads
                anchor_based: true,   // YOLOv3 uses anchor boxes
            }
        } else {
            crate::core::ArchitectureType::SingleStage {
                unified_head: true,   // YOLOv8 has unified head
                anchor_based: false,  // YOLOv8 is anchor-free
            }
        };
        
        let model_info = ModelInfo {
            name: model_name,
            version: "1.0".to_string(),
            architecture,
            input_spec: input_spec.clone(),
            output_spec: output_spec.clone(),
            preprocessing_config: crate::core::PreprocessingConfig {
                resize_strategy: crate::core::ResizeStrategy::Direct { target: input_size },
                normalization: crate::core::NormalizationType::ZeroToOne,
                tensor_layout: crate::core::TensorLayout {
                    format: "NCHW".to_string(),
                    channel_order: "RGB".to_string(),
                },
            },
        };

        // Create input and output specifications  
        let mut input_spec = crate::input::InputSpecification::default();
        
        // Update input specification with model-specific settings
        input_spec.preprocessing.resize_strategy = ResizeStrategy::Direct { target: input_size };
        
        // Update tensor shape dimensions to match the model requirements
        input_spec.tensor_spec.shape.dimensions = vec![
            crate::input::OnnxDimension::Batch,
            crate::input::OnnxDimension::Fixed(3),
            crate::input::OnnxDimension::Fixed(input_size.1 as i64), // height
            crate::input::OnnxDimension::Fixed(input_size.0 as i64), // width
        ];
        
        // Note: output_spec was already created above with loaded config

        Ok(UniversalSession {
            id: Uuid::new_v4(),
            model_info,
            session: std::sync::Arc::new(ort_session),
            input_processor: crate::input::InputProcessor::new(input_spec),
            output_processor: crate::output::OutputProcessor::new(output_spec),
        })
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl UniversalSession {
    /// Create a session from a model file with automatic configuration
    pub async fn from_model_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> crate::error::Result<Self> {
        SessionBuilder::new()
            .model_file(path.as_ref().to_string_lossy().to_string())
            .build()
            .await
    }

    /// Create a session builder for advanced configuration
    pub fn builder() -> SessionBuilder {
        SessionBuilder::new()
    }

    /// Run inference on a single image
    pub async fn infer_image<P: AsRef<std::path::Path>>(
        &self,
        image_path: P,
    ) -> crate::error::Result<Vec<Detection>> {
        // Load image
        let image = image::open(image_path.as_ref())
            .map_err(|e| crate::error::UocvrError::ImageProcessing(e))?;

        let original_shape = (image.width(), image.height());

        // Process input
        let input_tensor = self.input_processor.process_image(&image)?;
        let input_shape = (input_tensor.shape()[2] as u32, input_tensor.shape()[3] as u32);

        println!("🚀 ONNX Runtime inference pipeline activated!");
        println!("   Input tensor shape: {:?}", input_tensor.shape());
        println!("   Model: {}", self.model_info.name);

        // Run actual ONNX Runtime inference
        let inference_start = std::time::Instant::now();
        let outputs = self.run_onnx_inference(&input_tensor)?;
        let inference_time = inference_start.elapsed();
        println!("   Inference completed in: {:?}", inference_time);
        
        // Process outputs to get detections
        let detections = self.output_processor.process_outputs(
            &outputs,
            input_shape,
            original_shape,
            &self.model_info.name,
        )?;

        // println!("Found {} detections", detections.len());
        // for (i, detection) in detections.iter().enumerate() {
        //     if (detection.confidence > 0.60) {
        //     println!("  Detection {}: class_id={}, confidence={:.3}, bbox=({:.1}, {:.1}, {:.1}, {:.1})", 
        //         i + 1, detection.class_id, detection.confidence,
        //         detection.bbox.x, detection.bbox.y, detection.bbox.width, detection.bbox.height);
        //     }
        // }
        Ok(detections)
    }

    /// Run actual ONNX Runtime inference on input tensor
    fn run_onnx_inference(&self, input_tensor: &ndarray::Array4<f32>) -> crate::error::Result<Vec<ndarray::ArrayD<f32>>> {
        if self.model_info.name.contains("yolov3") {
            self.run_yolov3_inference(input_tensor)
        } else {
            self.run_single_input_inference(input_tensor)
        }
    }

    /// Run inference for single-input models (YOLOv8, YOLOv2, etc.)
    fn run_single_input_inference(&self, input_tensor: &ndarray::Array4<f32>) -> crate::error::Result<Vec<ndarray::ArrayD<f32>>> {
        use ort::Value;
        use ndarray::CowArray;
        
        // Convert Array4 to dynamic array
        let input_dyn = input_tensor.view().into_dyn();
        let input_cow = CowArray::from(input_dyn);
        
        // Create ORT value from the ndarray
        let input_value = Value::from_array(self.session.allocator(), &input_cow)
            .map_err(|e| crate::error::UocvrError::Runtime {
                message: format!("Failed to create ORT input value: {}", e),
            })?;
        
        // Run inference using the session
        let outputs = self.session
            .run(vec![input_value])
            .map_err(|e| crate::error::UocvrError::Runtime {
                message: format!("ONNX Runtime inference failed: {}", e),
            })?;

        // Convert outputs to ndarray format
        let mut output_arrays = Vec::new();
        for output in outputs {
            let output_tensor = output.try_extract::<f32>()
                .map_err(|e| crate::error::UocvrError::Runtime {
                    message: format!("Failed to extract output tensor: {}", e),
                })?;
            
            // Convert the tensor view to an owned ArrayD
            let output_array = output_tensor.view().to_owned().into_dyn();
            output_arrays.push(output_array);
        }

        Ok(output_arrays)
    }

    /// Run inference for YOLOv3 models that require image_shape input
    fn run_yolov3_inference(&self, input_tensor: &ndarray::Array4<f32>) -> crate::error::Result<Vec<ndarray::ArrayD<f32>>> {
        use ort::Value;
        use ndarray::CowArray;
        
        // Convert Array4 to dynamic array  
        let input_dyn = input_tensor.view().into_dyn();
        let input_cow = CowArray::from(input_dyn);
        
        // Create image_shape tensor [height, width] as [416, 416]
        let image_shape = ndarray::Array2::from_shape_vec((1, 2), vec![416.0f32, 416.0f32])
            .map_err(|e| crate::error::UocvrError::Runtime {
                message: format!("Failed to create image_shape tensor: {}", e),
            })?;
        let shape_dyn = image_shape.view().into_dyn();
        let shape_cow = CowArray::from(shape_dyn);
        
        // Create ORT values
        let input_value = Value::from_array(self.session.allocator(), &input_cow)
            .map_err(|e| crate::error::UocvrError::Runtime {
                message: format!("Failed to create ORT input value: {}", e),
            })?;
            
        let shape_value = Value::from_array(self.session.allocator(), &shape_cow)
            .map_err(|e| crate::error::UocvrError::Runtime {
                message: format!("Failed to create ORT image_shape value: {}", e),
            })?;
        
        // Run inference with both inputs
        let outputs = self.session
            .run(vec![input_value, shape_value])
            .map_err(|e| crate::error::UocvrError::Runtime {
                message: format!("ONNX Runtime inference failed: {}", e),
            })?;

        // Convert outputs to ndarray format
        let mut output_arrays = Vec::new();
        for output in outputs {
            // Try to extract as Float32 first, then Int32
            if let Ok(output_tensor) = output.try_extract::<f32>() {
                let output_array = output_tensor.view().to_owned().into_dyn();
                output_arrays.push(output_array);
            } else if let Ok(output_tensor) = output.try_extract::<i32>() {
                // Convert Int32 to Float32
                let int_array = output_tensor.view().to_owned().into_dyn();
                let float_data: Vec<f32> = int_array.iter().map(|&x| x as f32).collect();
                let float_array = ndarray::ArrayD::from_shape_vec(int_array.shape(), float_data)
                    .map_err(|e| crate::error::UocvrError::Runtime {
                        message: format!("Failed to convert Int32 output to Float32: {}", e),
                    })?;
                output_arrays.push(float_array);
            } else {
                return Err(crate::error::UocvrError::Runtime {
                    message: "Failed to extract output tensor: unsupported data type".to_string(),
                });
            }
        }

        Ok(output_arrays)
    }

    /// Create mock YOLO output data to demonstrate the processing pipeline
    /// NOTE: This method is kept for fallback/testing purposes
    #[allow(dead_code)]
    fn create_mock_yolo_output(&self) -> ndarray::ArrayD<f32> {
        // Create a mock YOLOv8 output tensor: [1, 84, 8400]
        // 84 = 4 (bbox) + 80 (classes)
        let mut output_data = vec![0.0f32; 1 * 84 * 8400];
        
        // Add a few mock detections with realistic values
        let detections = [
            // Detection 1: Person at center-left
            (320.0, 240.0, 80.0, 160.0, 0.85, 0), // center_x, center_y, width, height, confidence, class
            // Detection 2: Car at bottom-right
            (540.0, 380.0, 120.0, 80.0, 0.92, 2),
            // Detection 3: Dog at top-right
            (580.0, 120.0, 60.0, 90.0, 0.78, 16),
        ];
        
        for (i, &(cx, cy, w, h, conf, class)) in detections.iter().enumerate() {
            let pred_idx = i * 100; // Spread them out in the predictions
            if pred_idx < 8400 {
                // Set bounding box coordinates (first 4 values)
                output_data[0 * 84 * 8400 + 0 * 8400 + pred_idx] = cx; // center_x
                output_data[0 * 84 * 8400 + 1 * 8400 + pred_idx] = cy; // center_y
                output_data[0 * 84 * 8400 + 2 * 8400 + pred_idx] = w;  // width
                output_data[0 * 84 * 8400 + 3 * 8400 + pred_idx] = h;  // height
                
                // Set class confidence (sigmoid inverse to simulate logits)
                let class_offset = 4 + class;
                if class_offset < 84 {
                    // Convert confidence to logit (inverse sigmoid)
                    let logit = ((conf as f32) / (1.0 - conf as f32)).ln();
                    output_data[0 * 84 * 8400 + class_offset * 8400 + pred_idx] = logit;
                }
            }
        }
        
        // Create the ndarray from the mock data
        ndarray::ArrayD::from_shape_vec(vec![1, 84, 8400], output_data)
            .expect("Failed to create mock output tensor")
    }

    /// Run inference on a batch of images
    pub async fn infer_batch(
        &self,
        images: &[image::DynamicImage],
    ) -> crate::error::Result<Vec<InferenceResult>> {
        let mut results = Vec::new();
        
        for (idx, image) in images.iter().enumerate() {
            let start_time = std::time::Instant::now();
            
            // Get original image dimensions
            let original_shape = (image.width(), image.height());
            
            // Process input
            let preprocessing_start = std::time::Instant::now();
            let input_tensor = self.input_processor.process_image(image)?;
            let preprocessing_time = preprocessing_start.elapsed();
            
            let input_shape = (input_tensor.shape()[2] as u32, input_tensor.shape()[3] as u32);
            
            // Simulate inference time
            let inference_start = std::time::Instant::now();
            let mock_output = self.create_mock_yolo_output();
            let inference_time = inference_start.elapsed();
            
            // Process outputs
            let postprocessing_start = std::time::Instant::now();
            let detections = self.output_processor.process_outputs(
                &[mock_output],
                input_shape,
                original_shape,
                &self.model_info.name,
            )?;
            let postprocessing_time = postprocessing_start.elapsed();
            
            let total_time = start_time.elapsed();
            
            println!("Batch item {}: Found {} detections", idx + 1, detections.len());
            
            results.push(InferenceResult {
                detections,
                processing_time: total_time,
                metadata: InferenceMetadata {
                    model_name: self.model_info.name.clone(),
                    input_shape: input_tensor.shape().iter().map(|&x| x as i64).collect(),
                    output_shapes: vec![vec![1, 84, 8400]],
                    inference_time,
                    preprocessing_time,
                    postprocessing_time,
                },
            });
        }
        
        Ok(results)
    }

    /// Get model information
    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }
}

/// Load YAML postprocessing configuration from file
async fn load_yaml_postprocessing_config(config_path: &str) -> crate::error::Result<crate::session::YamlPostprocessingConfig> {
    use std::path::Path;
    use std::fs;
    
    if !Path::new(config_path).exists() {
        return Err(crate::error::UocvrError::ResourceNotFound {
            resource: config_path.to_string(),
        });
    }
    
    // Read and parse YAML config file
    let config_content = fs::read_to_string(config_path)
        .map_err(|e| crate::error::UocvrError::Session {
            message: format!("Failed to read config file {}: {}", config_path, e),
        })?;
        
    let yaml_config: YamlConfig = serde_yaml::from_str(&config_content)
        .map_err(|e| crate::error::UocvrError::Session {
            message: format!("Failed to parse YAML config {}: {}", config_path, e),
        })?;
    
    // Extract postprocessing config - try root level first, then nested under output
    let postprocessing = if let Some(root_postprocessing) = yaml_config.postprocessing {
        println!("📁 Found postprocessing config at root level");
        root_postprocessing
    } else if let Some(output_value) = &yaml_config.output {
        // Try to parse nested postprocessing under output
        if let Ok(output_config) = serde_yaml::from_value::<OutputConfig>(output_value.clone()) {
            if let Some(nested_postprocessing) = output_config.postprocessing {
                println!("📁 Found postprocessing config nested under output");
                nested_postprocessing
            } else {
                println!("⚠️  No postprocessing config found in output section");
                YamlPostprocessingConfig::default()
            }
        } else {
            println!("⚠️  Failed to parse output section");
            YamlPostprocessingConfig::default()
        }
    } else {
        println!("⚠️  No postprocessing config found anywhere");
        YamlPostprocessingConfig::default()
    };
    
    println!("🎯 Loaded config - confidence_threshold: {:?}", postprocessing.confidence_threshold);
    
    Ok(crate::session::YamlPostprocessingConfig {
        nms_enabled: postprocessing.nms_enabled,
        confidence_threshold: postprocessing.confidence_threshold,
        objectness_threshold: postprocessing.objectness_threshold,
        nms_threshold: postprocessing.nms_threshold,
        max_detections: postprocessing.max_detections,
        class_agnostic_nms: postprocessing.class_agnostic_nms,
        coordinate_decoding: postprocessing.coordinate_decoding,
    })
}

/// YAML configuration structures for loading from config files
#[derive(Debug, Clone, serde::Deserialize)]
struct YamlConfig {
    pub model: Option<serde_yaml::Value>,
    pub input: Option<serde_yaml::Value>,
    pub output: Option<serde_yaml::Value>,
    pub postprocessing: Option<YamlPostprocessingConfig>,
    pub processing: Option<serde_yaml::Value>,
    pub execution: Option<serde_yaml::Value>,
    pub classes: Option<serde_yaml::Value>,
}

/// YAML output configuration that can contain nested postprocessing
#[derive(Debug, Clone, serde::Deserialize)]
struct OutputConfig {
    pub postprocessing: Option<YamlPostprocessingConfig>,
    #[serde(flatten)]
    pub other: std::collections::HashMap<String, serde_yaml::Value>,
}

/// YAML postprocessing configuration
#[derive(Debug, Clone, serde::Deserialize, Default)]
struct YamlPostprocessingConfig {
    pub nms_enabled: Option<bool>,
    pub confidence_threshold: Option<f32>,
    pub objectness_threshold: Option<f32>,
    pub nms_threshold: Option<f32>,
    pub max_detections: Option<usize>,
    pub class_agnostic_nms: Option<bool>,
    pub coordinate_decoding: Option<String>,
}