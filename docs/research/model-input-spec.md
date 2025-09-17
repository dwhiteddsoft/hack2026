# Canonical ONNX Model Input Specification

## Executive Summary

After analyzing ONNX input specifications across 12 major computer vision models, I propose a **Universal ONNX Input Descriptor (UOID)** that canonically describes any ONNX model's input requirements through a structured format optimized specifically for ONNX Runtime.

## Canonical ONNX Input Model Design

### Core ONNX Tensor Structure
```rust
pub struct CanonicalOnnxInput {
    pub tensor_spec: OnnxTensorSpec,
    pub preprocessing: OnnxPreprocessing,
    pub session_config: OnnxSessionConfig,
}

pub struct OnnxTensorSpec {
    pub input_name: String,              // ONNX input tensor name (e.g., "images", "input")
    pub shape: OnnxTensorShape,
    pub data_type: OnnxDataType,
    pub value_range: ValueRange,
}

pub struct OnnxTensorShape {
    pub dimensions: Vec<OnnxDimension>,  // [N, C, H, W] format for ONNX
}

pub enum OnnxDimension {
    Fixed(i64),                          // Fixed dimension: 416
    Dynamic {                            // Dynamic dimension with constraints
        name: String,                    // ONNX symbolic name: "height", "width"
        default: i64,                    // Default value: 640
        constraints: DimensionConstraints,
    },
    Batch,                               // Batch dimension (typically dynamic)
}

pub enum OnnxDataType {
    Float32,                             // ONNX FLOAT (most common)
    Float16,                             // ONNX FLOAT16 (optimization)
    UInt8,                               // ONNX UINT8 (quantized models)
}
```

### ONNX-Specific Value Handling
```rust
pub struct ValueRange {
    pub normalization: NormalizationType,
    pub onnx_range: (f32, f32),
}

pub enum NormalizationType {
    ZeroToOne,                           // [0.0, 1.0] - YOLO family ONNX models
    ImageNet {                           // ImageNet normalization for ONNX
        mean: [f32; 3],                  // [0.485, 0.456, 0.406]
        std: [f32; 3],                   // [0.229, 0.224, 0.225]
    },
    Custom {                             // Custom ONNX model normalization
        mean: [f32; 3],
        std: Option<[f32; 3]>,
    },
}
```

### ONNX Preprocessing Pipeline
```rust
pub struct OnnxPreprocessing {
    pub resize_strategy: ResizeStrategy,
    pub normalization: NormalizationType,
    pub tensor_layout: TensorLayout,
}

pub enum ResizeStrategy {
    Direct { target: (u32, u32) },       // SSD ONNX: direct to 300x300
    Letterbox {                          // YOLO ONNX: aspect-preserving
        target: (u32, u32),
        padding_value: f32,              // Typically 114/255 = 0.447
    },
    ShortestEdge {                       // Mask R-CNN ONNX
        target_size: u32,
        max_size: Option<u32>,
    },
}

pub struct TensorLayout {
    pub format: String,                  // "NCHW" (standard for ONNX CV models)
    pub channel_order: String,           // "RGB" or "BGR"
}
```

### ONNX Session Configuration
```rust
pub struct OnnxSessionConfig {
    pub execution_providers: Vec<ExecutionProvider>,
    pub graph_optimization_level: GraphOptimizationLevel,
    pub input_binding: InputBinding,
}

pub enum ExecutionProvider {
    CPU,
    CUDA(CudaConfig),
    TensorRT(TensorRTConfig),
    DirectML,
    CoreML,
}

pub struct InputBinding {
    pub input_names: Vec<String>,        // ["images"] or ["images", "image_shape"]
    pub binding_strategy: BindingStrategy,
}

pub enum BindingStrategy {
    SingleInput,                         // Most ONNX models
    MultiInput {                         // Some YOLOv3 models require image_shape
        primary: String,                 // "images"
        auxiliary: Vec<String>,          // ["image_shape"]
    },
}
```

## ONNX Model Family Mappings

### Standard ONNX YOLO Models
```rust
let yolov8_onnx = CanonicalOnnxInput {
    tensor_spec: OnnxTensorSpec {
        input_name: "images".to_string(),
        shape: OnnxTensorShape {
            dimensions: vec![
                OnnxDimension::Batch,
                OnnxDimension::Fixed(3),     // RGB channels
                OnnxDimension::Dynamic {
                    name: "height".to_string(),
                    default: 640,
                    constraints: DimensionConstraints::MultipleOf(32),
                },
                OnnxDimension::Dynamic {
                    name: "width".to_string(), 
                    default: 640,
                    constraints: DimensionConstraints::MultipleOf(32),
                },
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
            padding_value: 0.447,  // 114/255
        },
        normalization: NormalizationType::ZeroToOne,
        tensor_layout: TensorLayout {
            format: "NCHW".to_string(),
            channel_order: "RGB".to_string(),
        },
    },
    session_config: OnnxSessionConfig {
        execution_providers: vec![ExecutionProvider::CPU],
        graph_optimization_level: GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
        input_binding: InputBinding {
            input_names: vec!["images".to_string()],
            binding_strategy: BindingStrategy::SingleInput,
        },
    },
};
```

### Fixed Input ONNX Models (SSD)
```rust
let ssd300_onnx = CanonicalOnnxInput {
    tensor_spec: OnnxTensorSpec {
        input_name: "input".to_string(),
        shape: OnnxTensorShape {
            dimensions: vec![
                OnnxDimension::Fixed(1),     // Batch size
                OnnxDimension::Fixed(3),     // RGB channels
                OnnxDimension::Fixed(300),   // Height
                OnnxDimension::Fixed(300),   // Width
            ],
        },
        data_type: OnnxDataType::Float32,
        value_range: ValueRange {
            normalization: NormalizationType::Custom {
                mean: [123.68, 116.78, 103.94],
                std: None,
            },
            onnx_range: (-128.0, 127.0),
        },
    },
    preprocessing: OnnxPreprocessing {
        resize_strategy: ResizeStrategy::Direct { target: (300, 300) },
        normalization: NormalizationType::Custom {
            mean: [123.68, 116.78, 103.94],
            std: None,
        },
        tensor_layout: TensorLayout {
            format: "NCHW".to_string(),
            channel_order: "RGB".to_string(),
        },
    },
    session_config: OnnxSessionConfig {
        execution_providers: vec![ExecutionProvider::CPU],
        graph_optimization_level: GraphOptimizationLevel::ORT_ENABLE_BASIC,
        input_binding: InputBinding {
            input_names: vec!["input".to_string()],
            binding_strategy: BindingStrategy::SingleInput,
        },
    },
};
```

### Multi-Input ONNX Models (Some YOLOv3)
```rust
let yolov3_dual_input_onnx = CanonicalOnnxInput {
    tensor_spec: OnnxTensorSpec {
        input_name: "input_1".to_string(),
        shape: OnnxTensorShape {
            dimensions: vec![
                OnnxDimension::Fixed(1),
                OnnxDimension::Fixed(3),
                OnnxDimension::Fixed(416),
                OnnxDimension::Fixed(416),
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
            target: (416, 416),
            padding_value: 0.5,
        },
        normalization: NormalizationType::ZeroToOne,
        tensor_layout: TensorLayout {
            format: "NCHW".to_string(),
            channel_order: "RGB".to_string(),
        },
    },
    session_config: OnnxSessionConfig {
        execution_providers: vec![ExecutionProvider::CPU],
        graph_optimization_level: GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
        input_binding: InputBinding {
            input_names: vec!["input_1".to_string(), "image_shape".to_string()],
            binding_strategy: BindingStrategy::MultiInput {
                primary: "input_1".to_string(),
                auxiliary: vec!["image_shape".to_string()],
            },
        },
    },
};
```

## ONNX-Specific Implementation

### 1. **ONNX Runtime Integration**
```rust
use ort::{Session, Value, CowArray};

impl CanonicalOnnxInput {
    pub fn create_session(&self, model_path: &str) -> Result<Session, OnnxError> {
        let mut session_builder = Session::builder()?;
        
        // Configure execution providers
        for provider in &self.session_config.execution_providers {
            match provider {
                ExecutionProvider::CPU => session_builder = session_builder.with_cpu_provider()?,
                ExecutionProvider::CUDA(config) => session_builder = session_builder.with_cuda_provider(config.device_id)?,
                _ => {} // Add other providers as needed
            }
        }
        
        session_builder.commit_from_file(model_path)
    }
    
    pub fn preprocess_to_onnx_tensor(&self, image: &RgbImage) -> Result<Value, PreprocessingError> {
        // Apply preprocessing pipeline
        let processed = self.apply_preprocessing(image)?;
        
        // Convert to ONNX tensor format
        let cow_array = CowArray::from(processed.view());
        Value::from_array(allocator, &cow_array)
    }
    
    pub fn run_inference(&self, session: &Session, image: &RgbImage) -> Result<Vec<Value>, InferenceError> {
        match &self.session_config.input_binding.binding_strategy {
            BindingStrategy::SingleInput => {
                let input_tensor = self.preprocess_to_onnx_tensor(image)?;
                session.run(vec![input_tensor])
            },
            BindingStrategy::MultiInput { primary, auxiliary } => {
                let primary_tensor = self.preprocess_to_onnx_tensor(image)?;
                let aux_tensors = self.create_auxiliary_tensors(image, auxiliary)?;
                
                let mut inputs = vec![primary_tensor];
                inputs.extend(aux_tensors);
                session.run(inputs)
            },
        }
    }
}
```

### 2. **ONNX Model Introspection**
```rust
impl CanonicalOnnxInput {
    pub fn from_onnx_model(model_path: &str) -> Result<Self, ModelIntrospectionError> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        
        // Extract input metadata from ONNX model
        let input_metadata = session.inputs();
        let input_info = &input_metadata[0];
        
        // Parse shape information
        let shape = Self::parse_onnx_shape(&input_info.dimensions)?;
        
        // Determine model type from shape and name patterns
        let model_type = Self::infer_model_type(&input_info.name, &shape)?;
        
        // Generate appropriate configuration
        Self::generate_config_for_model_type(model_type, input_info)
    }
    
    fn parse_onnx_shape(dimensions: &[Option<i64>]) -> Result<OnnxTensorShape, ShapeParsingError> {
        let mut parsed_dims = Vec::new();
        
        for (i, dim) in dimensions.iter().enumerate() {
            match dim {
                Some(size) => parsed_dims.push(OnnxDimension::Fixed(*size)),
                None => {
                    // Dynamic dimension - infer meaning from position
                    let name = match i {
                        0 => "batch".to_string(),
                        2 => "height".to_string(), 
                        3 => "width".to_string(),
                        _ => format!("dim_{}", i),
                    };
                    parsed_dims.push(OnnxDimension::Dynamic {
                        name,
                        default: Self::get_default_for_position(i),
                        constraints: Self::get_constraints_for_position(i),
                    });
                }
            }
        }
        
        Ok(OnnxTensorShape { dimensions: parsed_dims })
    }
}
```

### 3. **ONNX Performance Optimization**
```rust
pub struct OnnxOptimizationConfig {
    pub execution_mode: ExecutionMode,
    pub intra_op_num_threads: Option<usize>,
    pub inter_op_num_threads: Option<usize>,
    pub enable_mem_pattern: bool,
    pub enable_mem_reuse: bool,
}

impl CanonicalOnnxInput {
    pub fn optimize_for_hardware(&mut self, hardware: &HardwareInfo) {
        match hardware.device_type {
            DeviceType::CPU => {
                self.session_config.execution_providers = vec![ExecutionProvider::CPU];
                self.session_config.graph_optimization_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
            },
            DeviceType::NVIDIA_GPU => {
                self.session_config.execution_providers = vec![
                    ExecutionProvider::CUDA(CudaConfig { device_id: 0 }),
                    ExecutionProvider::CPU, // Fallback
                ];
            },
            DeviceType::Apple_Silicon => {
                self.session_config.execution_providers = vec![
                    ExecutionProvider::CoreML,
                    ExecutionProvider::CPU,
                ];
            },
        }
    }
}
```

## ONNX-Specific Benefits

### 1. **Native ONNX Integration**
- **Direct session creation** from canonical specification
- **Automatic provider selection** based on hardware
- **Built-in tensor format** handling for ONNX Runtime
- **Model introspection** for automatic configuration

### 2. **ONNX Runtime Optimization**
- **Execution provider management** (CPU, CUDA, TensorRT, etc.)
- **Graph optimization levels** appropriate for model types
- **Memory pattern optimization** for performance
- **Batch processing** configuration

### 3. **ONNX Model Validation**
```rust
impl CanonicalOnnxInput {
    pub fn validate_onnx_compatibility(&self, model_path: &str) -> Result<(), ValidationError> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        
        // Validate input names match
        let model_inputs = session.inputs();
        self.validate_input_names(&model_inputs)?;
        
        // Validate tensor shapes are compatible
        self.validate_tensor_shapes(&model_inputs)?;
        
        // Validate data types match
        self.validate_data_types(&model_inputs)?;
        
        Ok(())
    }
}
```

### 4. **Simplified Usage for ONNX**
```rust
// Load any ONNX model automatically
let model_spec = CanonicalOnnxInput::from_onnx_model("yolov8n.onnx")?;

// Create optimized session
let session = model_spec.create_session("yolov8n.onnx")?;

// Process input with automatic preprocessing
let results = model_spec.run_inference(&session, &input_image)?;
```

## Conclusion

This ONNX-focused canonical input model provides:

1. **Native ONNX Runtime integration** with direct session management
2. **Automatic model introspection** from ONNX metadata
3. **Hardware-optimized execution** provider selection
4. **Simplified preprocessing** pipeline for ONNX tensors
5. **Validation and compatibility** checking for ONNX models
6. **Performance optimization** specific to ONNX Runtime

The design is streamlined specifically for ONNX deployment scenarios while maintaining flexibility for different model architectures and hardware configurations.