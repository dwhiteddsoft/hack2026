use ndarray::Array4;
use image::{DynamicImage, GenericImageView};
use crate::core::{ResizeStrategy, NormalizationType, TensorLayout};

/// Universal input processor for ONNX computer vision models
pub struct InputProcessor {
    pub specification: InputSpecification,
    pub preprocessing_config: PreprocessingConfig,
}

/// Canonical ONNX input specification
#[derive(Debug, Clone)]
pub struct InputSpecification {
    pub tensor_spec: OnnxTensorSpec,
    pub preprocessing: OnnxPreprocessing,
    pub session_config: OnnxSessionConfig,
}

/// ONNX tensor specification
#[derive(Debug, Clone)]
pub struct OnnxTensorSpec {
    pub input_name: String,
    pub shape: OnnxTensorShape,
    pub data_type: OnnxDataType,
    pub value_range: ValueRange,
}

/// ONNX tensor shape specification
#[derive(Debug, Clone)]
pub struct OnnxTensorShape {
    pub dimensions: Vec<OnnxDimension>,
}

/// ONNX dimension types
#[derive(Debug, Clone)]
pub enum OnnxDimension {
    Fixed(i64),
    Dynamic {
        name: String,
        default: i64,
        constraints: DimensionConstraints,
    },
    Batch,
}

/// Dimension constraints
#[derive(Debug, Clone)]
pub enum DimensionConstraints {
    MultipleOf(i64),
    Range { min: i64, max: i64 },
    Fixed(i64),
    Any,
}

/// ONNX data types
#[derive(Debug, Clone)]
pub enum OnnxDataType {
    Float32,
    Float16,
    UInt8,
}

/// Value range specification
#[derive(Debug, Clone)]
pub struct ValueRange {
    pub normalization: NormalizationType,
    pub onnx_range: (f32, f32),
}

/// ONNX preprocessing configuration
#[derive(Debug, Clone)]
pub struct OnnxPreprocessing {
    pub resize_strategy: ResizeStrategy,
    pub normalization: NormalizationType,
    pub tensor_layout: TensorLayout,
}

/// ONNX session configuration
#[derive(Debug, Clone)]
pub struct OnnxSessionConfig {
    pub execution_providers: Vec<crate::core::ExecutionProvider>,
    pub graph_optimization_level: crate::core::GraphOptimizationLevel,
    pub input_binding: InputBinding,
}

/// Input binding configuration
#[derive(Debug, Clone)]
pub struct InputBinding {
    pub input_names: Vec<String>,
    pub binding_strategy: BindingStrategy,
}

/// Binding strategy options
#[derive(Debug, Clone)]
pub enum BindingStrategy {
    SingleInput,
    MultiInput {
        primary: String,
        auxiliary: Vec<String>,
    },
}

/// Preprocessing configuration
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    pub target_size: (u32, u32),
    pub maintain_aspect_ratio: bool,
    pub padding_value: f32,
    pub normalization: NormalizationType,
    pub channel_order: String,
}

impl InputProcessor {
    /// Create a new input processor with specification
    pub fn new(specification: InputSpecification) -> Self {
        let preprocessing_config = PreprocessingConfig::from_spec(&specification);
        Self {
            specification,
            preprocessing_config,
        }
    }

    /// Create an input processor from specification
    pub fn from_spec(spec: &InputSpecification) -> Self {
        Self {
            specification: spec.clone(),
            preprocessing_config: PreprocessingConfig::from_spec(spec),
        }
    }

    /// Process a single image for inference
    pub fn process_image(&self, image: &DynamicImage) -> crate::error::Result<Array4<f32>> {
        // Step 1: Preprocess (channel order conversion)
        let preprocessed = self.preprocess_image(image)?;
        
        // Step 2: Apply resize strategy
        let resized = self.apply_resize(&preprocessed)?;
        
        // Step 3: Convert to tensor format with normalization
        let tensor = self.image_to_tensor(&resized)?;
        
        // Step 4: Validate against model requirements
        self.validate_input(&tensor)?;
        
        Ok(tensor)
    }

    /// Process a batch of images for inference
    pub fn process_batch(&self, images: &[DynamicImage]) -> crate::error::Result<Array4<f32>> {
        use ndarray::{Array, Axis};
        
        if images.is_empty() {
            return Err(crate::error::UocvrError::ModelConfig {
                message: "Cannot process empty batch".to_string()
            });
        }
        
        // Process each image individually
        let mut tensors = Vec::new();
        for image in images {
            let tensor = self.process_image(image)?;
            tensors.push(tensor);
        }
        
        // Concatenate along batch dimension
        let batch_tensor = ndarray::concatenate(
            Axis(0),
            &tensors.iter().map(|t| t.view()).collect::<Vec<_>>()
        ).map_err(|e| crate::error::UocvrError::ModelConfig {
            message: format!("Failed to concatenate batch tensors: {}", e)
        })?;
        
        Ok(batch_tensor)
    }

    /// Validate input dimensions against model requirements
    pub fn validate_input(&self, input: &Array4<f32>) -> crate::error::Result<()> {
        let expected_shape = self.get_input_shape();
        let actual_shape = input.shape();
        
        // Convert to i64 for comparison
        let actual_i64: Vec<i64> = actual_shape.iter().map(|&x| x as i64).collect();
        
        // Check dimensions (allowing dynamic batch size)
        if actual_i64.len() != expected_shape.len() {
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!(
                    "Input tensor rank mismatch: expected {}, got {}",
                    expected_shape.len(),
                    actual_i64.len()
                )
            });
        }
        
        // Check non-batch dimensions (skip first dimension which is batch)
        for (i, (&actual, &expected)) in actual_i64[1..].iter().zip(expected_shape[1..].iter()).enumerate() {
            if expected > 0 && actual != expected {
                return Err(crate::error::UocvrError::ModelConfig {
                    message: format!(
                        "Input tensor dimension {} mismatch: expected {}, got {}",
                        i + 1,
                        expected,
                        actual
                    )
                });
            }
        }
        
        Ok(())
    }

    /// Get the expected input shape for the model
    pub fn get_input_shape(&self) -> Vec<i64> {
        self.specification.tensor_spec.shape.dimensions
            .iter()
            .map(|dim| match dim {
                OnnxDimension::Fixed(size) => *size,
                OnnxDimension::Dynamic { default, .. } => *default,
                OnnxDimension::Batch => -1, // Dynamic batch size
            })
            .collect()
    }

    /// Preprocess an image according to the model's requirements
    fn preprocess_image(&self, image: &DynamicImage) -> crate::error::Result<DynamicImage> {
        // Apply channel order conversion if needed
        let processed = if self.preprocessing_config.channel_order == "BGR" {
            preprocessing::convert_channel_order(image, "BGR")
        } else {
            image.clone()  // Keep as RGB
        };
        
        Ok(processed)
    }

    /// Apply resize strategy to the image
    fn apply_resize(&self, image: &DynamicImage) -> crate::error::Result<DynamicImage> {
        match &self.specification.preprocessing.resize_strategy {
            ResizeStrategy::Letterbox { target, padding_value } => {
                preprocessing::letterbox_resize(image, *target, *padding_value)
            },
            ResizeStrategy::Direct { target } => {
                preprocessing::direct_resize(image, *target)
            },
            ResizeStrategy::ShortestEdge { target_size, max_size } => {
                preprocessing::shortest_edge_resize(image, *target_size, *max_size)
            }
        }
    }

    /// Apply normalization to tensor data
    fn apply_normalization(&self, data: &mut [f32]) -> crate::error::Result<()> {
        match &self.preprocessing_config.normalization {
            NormalizationType::ZeroToOne => {
                preprocessing::zero_to_one_normalize(data);
            },
            NormalizationType::ImageNet { mean, std } => {
                preprocessing::imagenet_normalize(data, *mean, *std);
            },
            NormalizationType::Custom { mean, std } => {
                let std_values = std.unwrap_or([1.0, 1.0, 1.0]);
                preprocessing::imagenet_normalize(data, *mean, std_values);
            }
        }
        Ok(())
    }

    /// Convert image to tensor format
    fn image_to_tensor(&self, image: &DynamicImage) -> crate::error::Result<Array4<f32>> {
        use ndarray::Array;
        
        // Convert to RGB8 format
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        let raw_pixels = rgb_image.into_raw();
        
        // Convert u8 to f32 and arrange in NCHW format [1, 3, H, W]
        let mut tensor_data = vec![0.0f32; (width * height * 3) as usize];
        
        // Rearrange from HWC (interleaved) to CHW (planar)
        for y in 0..height {
            for x in 0..width {
                let pixel_idx = ((y * width + x) * 3) as usize;
                let r = raw_pixels[pixel_idx] as f32;
                let g = raw_pixels[pixel_idx + 1] as f32; 
                let b = raw_pixels[pixel_idx + 2] as f32;
                
                let flat_idx = (y * width + x) as usize;
                tensor_data[flat_idx] = r;  // R channel
                tensor_data[(width * height) as usize + flat_idx] = g;  // G channel
                tensor_data[2 * (width * height) as usize + flat_idx] = b;  // B channel
            }
        }
        
        // Apply normalization
        self.apply_normalization(&mut tensor_data)?;
        
        // Create ndarray in NCHW format [1, 3, H, W]
        let tensor = Array::from_shape_vec(
            (1, 3, height as usize, width as usize),
            tensor_data
        ).map_err(|e| crate::error::UocvrError::ModelConfig {
            message: format!("Failed to create tensor: {}", e)
        })?;
        
        Ok(tensor)
    }
}

impl PreprocessingConfig {
    /// Create preprocessing config from input specification
    pub fn from_spec(spec: &InputSpecification) -> Self {
        // Extract target size from tensor spec
        let mut target_size = (640, 640); // Default
        let mut maintain_aspect_ratio = true;
        let mut padding_value = 0.447; // 114/255 for YOLO models
        
        // Try to extract dimensions from shape
        if spec.tensor_spec.shape.dimensions.len() >= 4 {
            // Assume NCHW format: [batch, channels, height, width]
            if let (Some(OnnxDimension::Fixed(h)), Some(OnnxDimension::Fixed(w))) = 
                (spec.tensor_spec.shape.dimensions.get(2), spec.tensor_spec.shape.dimensions.get(3)) {
                target_size = (*w as u32, *h as u32);
            }
        }
        
        // Extract resize-specific parameters
        match &spec.preprocessing.resize_strategy {
            ResizeStrategy::Direct { target } => {
                target_size = *target;
                maintain_aspect_ratio = false;
            },
            ResizeStrategy::Letterbox { target, padding_value: pv } => {
                target_size = *target;
                padding_value = *pv;
                maintain_aspect_ratio = true;
            },
            ResizeStrategy::ShortestEdge { .. } => {
                maintain_aspect_ratio = true;
            }
        }
        
        Self {
            target_size,
            maintain_aspect_ratio,
            padding_value,
            normalization: spec.preprocessing.normalization.clone(),
            channel_order: "RGB".to_string(), // Default to RGB
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            target_size: (640, 640),
            maintain_aspect_ratio: true,
            padding_value: 0.447, // 114/255
            normalization: NormalizationType::ZeroToOne,
            channel_order: "RGB".to_string(),
        }
    }
}

/// Helper functions for common preprocessing operations
pub mod preprocessing {
    use super::*;

    /// Resize image with letterboxing (aspect ratio preserving)
    pub fn letterbox_resize(
        image: &DynamicImage,
        target: (u32, u32),
        padding_value: f32,
    ) -> crate::error::Result<DynamicImage> {
        use image::{ImageBuffer, Rgb, imageops::FilterType};
        
        let (target_w, target_h) = target;
        let (orig_w, orig_h) = image.dimensions();
        
        // Calculate scale factor to fit within target bounds while maintaining aspect ratio
        let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
        
        // Calculate new dimensions
        let new_w = (orig_w as f32 * scale) as u32;
        let new_h = (orig_h as f32 * scale) as u32;
        
        // Resize image maintaining aspect ratio
        let resized = image.resize(new_w, new_h, FilterType::Lanczos3);
        
        // Create target canvas with padding
        let padding_u8 = (padding_value * 255.0) as u8;
        let mut canvas = ImageBuffer::from_pixel(target_w, target_h, Rgb([padding_u8, padding_u8, padding_u8]));
        
        // Calculate position to center the resized image
        let x_offset = (target_w - new_w) / 2;
        let y_offset = (target_h - new_h) / 2;
        
        // Copy resized image to canvas
        let resized_rgb = resized.to_rgb8();
        for (x, y, pixel) in resized_rgb.enumerate_pixels() {
            canvas.put_pixel(x + x_offset, y + y_offset, *pixel);
        }
        
        Ok(DynamicImage::ImageRgb8(canvas))
    }

    /// Direct resize without maintaining aspect ratio
    pub fn direct_resize(
        image: &DynamicImage,
        target: (u32, u32),
    ) -> crate::error::Result<DynamicImage> {
        use image::imageops::FilterType;
        let resized = image.resize_exact(target.0, target.1, FilterType::Lanczos3);
        Ok(resized)
    }

    /// Shortest edge resize (commonly used in Mask R-CNN)
    pub fn shortest_edge_resize(
        image: &DynamicImage,
        target_size: u32,
        max_size: Option<u32>,
    ) -> crate::error::Result<DynamicImage> {
        use image::imageops::FilterType;
        
        let (orig_w, orig_h) = image.dimensions();
        let shortest_edge = orig_w.min(orig_h);
        let scale = target_size as f32 / shortest_edge as f32;
        
        let mut new_w = (orig_w as f32 * scale) as u32;
        let mut new_h = (orig_h as f32 * scale) as u32;
        
        // Apply max size constraint if specified
        if let Some(max_size) = max_size {
            let longest_edge = new_w.max(new_h);
            if longest_edge > max_size {
                let max_scale = max_size as f32 / longest_edge as f32;
                new_w = (new_w as f32 * max_scale) as u32;
                new_h = (new_h as f32 * max_scale) as u32;
            }
        }
        
        let resized = image.resize(new_w, new_h, FilterType::Lanczos3);
        Ok(resized)
    }

    /// Apply ImageNet normalization
    pub fn imagenet_normalize(data: &mut [f32], mean: [f32; 3], std: [f32; 3]) {
        let channels = 3;
        let pixels_per_channel = data.len() / channels;
        
        for c in 0..channels {
            let start = c * pixels_per_channel;
            let end = start + pixels_per_channel;
            for pixel in &mut data[start..end] {
                *pixel = (*pixel - mean[c]) / std[c];
            }
        }
    }

    /// Apply zero-to-one normalization
    pub fn zero_to_one_normalize(data: &mut [f32]) {
        for pixel in data.iter_mut() {
            *pixel /= 255.0;
        }
    }

    /// Convert RGB to BGR or vice versa
    pub fn convert_channel_order(image: &DynamicImage, target_order: &str) -> DynamicImage {
        use image::{ImageBuffer, Rgb};
        
        match target_order.to_uppercase().as_str() {
            "BGR" => {
                // Convert RGB to BGR by swapping R and B channels
                let rgb_image = image.to_rgb8();
                let (width, height) = rgb_image.dimensions();
                let mut bgr_data = Vec::with_capacity((width * height * 3) as usize);
                
                for pixel in rgb_image.pixels() {
                    bgr_data.push(pixel[2]); // B
                    bgr_data.push(pixel[1]); // G  
                    bgr_data.push(pixel[0]); // R
                }
                
                let bgr_image = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, bgr_data)
                    .expect("Failed to create BGR image");
                DynamicImage::ImageRgb8(bgr_image)
            },
            "RGB" | _ => {
                // Already RGB or default to RGB
                image.clone()
            }
        }
    }
}

impl Default for InputSpecification {
    fn default() -> Self {
        Self {
            tensor_spec: OnnxTensorSpec {
                input_name: "images".to_string(),
                shape: OnnxTensorShape {
                    dimensions: vec![
                        OnnxDimension::Batch,
                        OnnxDimension::Fixed(3),
                        OnnxDimension::Fixed(640),
                        OnnxDimension::Fixed(640),
                    ],
                },
                data_type: OnnxDataType::Float32,
                value_range: ValueRange { 
                    normalization: NormalizationType::ZeroToOne,
                    onnx_range: (0.0, 1.0),
                },
            },
            preprocessing: OnnxPreprocessing {
                resize_strategy: ResizeStrategy::Direct { target: (640, 640) },
                normalization: NormalizationType::ZeroToOne,
                tensor_layout: TensorLayout {
                    format: "NCHW".to_string(),
                    channel_order: "RGB".to_string(),
                },
            },
            session_config: OnnxSessionConfig {
                execution_providers: vec![crate::core::ExecutionProvider::CPU],
                graph_optimization_level: crate::core::GraphOptimizationLevel::EnableBasic,
                input_binding: InputBinding {
                    input_names: vec!["images".to_string()],
                    binding_strategy: BindingStrategy::SingleInput,
                },
            },
        }
    }
}