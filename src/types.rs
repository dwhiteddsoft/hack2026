//! Core types and configuration structures for the vision classifier.

use serde::{Deserialize, Serialize};

/// Result of a classification operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Index of the predicted class
    pub class_id: usize,
    
    /// Human-readable class name (if available)
    pub class_name: Option<String>,
    
    /// Confidence score for the predicted class
    pub confidence: f32,
    
    /// All output scores/probabilities
    pub all_scores: Vec<f32>,
    
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

impl ClassificationResult {
    /// Get the top N predictions
    pub fn top_n(&self, n: usize) -> Vec<(usize, f32)> {
        let mut indexed_scores: Vec<(usize, f32)> = self.all_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed_scores.into_iter().take(n).collect()
    }
    
    /// Get the top N predictions with class names
    pub fn top_n_with_names(&self, n: usize, class_names: &[String]) -> Vec<(String, f32)> {
        self.top_n(n)
            .into_iter()
            .map(|(idx, score)| {
                let name = class_names.get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", idx));
                (name, score)
            })
            .collect()
    }
}

/// Types of models supported by the classifier
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    /// Single frame image classification
    SingleFrame,
    
    /// Multi-frame video classification with fixed frame count
    MultiFrame { frame_count: usize },
    
    /// Variable frame count models
    Variable { min_frames: usize, max_frames: usize },
    
    /// LSTM-based sequence classification
    LSTM { sequence_length: usize },
    
    /// Two-stream models (RGB + Optical Flow)
    TwoStream,
    
    /// Object detection models that output bounding boxes
    ObjectDetection {
        /// Number of classes the model can detect
        num_classes: usize,
        /// Minimum confidence threshold for detections
        confidence_threshold: f32,
        /// Non-maximum suppression threshold
        nms_threshold: f32,
    },
}

impl ModelType {
    /// Get the required number of frames for this model type
    pub fn required_frames(&self) -> Option<usize> {
        match self {
            ModelType::SingleFrame => Some(1),
            ModelType::MultiFrame { frame_count } => Some(*frame_count),
            ModelType::LSTM { sequence_length } => Some(*sequence_length),
            ModelType::TwoStream => Some(2),
            ModelType::Variable { min_frames, .. } => Some(*min_frames),
            ModelType::ObjectDetection { .. } => Some(1),
        }
    }
    
    /// Check if this model type supports streaming
    pub fn supports_streaming(&self) -> bool {
        !matches!(self, ModelType::SingleFrame | ModelType::ObjectDetection { .. })
    }
    
    /// Check if this model type outputs bounding boxes
    pub fn outputs_bounding_boxes(&self) -> bool {
        matches!(self, ModelType::ObjectDetection { .. })
    }
}

/// Configuration for a vision classification model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Type of model
    pub model_type: ModelType,
    
    /// Input image size (width, height)
    pub input_size: (u32, u32),
    
    /// Number of input channels (typically 3 for RGB)
    pub channels: usize,
    
    /// Optional class names for human-readable output
    pub class_names: Option<Vec<String>>,
    
    /// Image normalization parameters
    pub normalization: ImageNormalization,
    
    /// Optional model-specific metadata
    pub metadata: Option<serde_json::Value>,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(model_type: ModelType) -> Self {
        Self {
            model_type,
            input_size: (224, 224),
            channels: 3,
            class_names: None,
            normalization: ImageNormalization::default(),
            metadata: None,
        }
    }
    
    /// Set input size
    pub fn with_input_size(mut self, width: u32, height: u32) -> Self {
        self.input_size = (width, height);
        self
    }
    
    /// Set class names
    pub fn with_class_names(mut self, names: Vec<String>) -> Self {
        self.class_names = Some(names);
        self
    }
    
    /// Set normalization parameters
    pub fn with_normalization(mut self, normalization: ImageNormalization) -> Self {
        self.normalization = normalization;
        self
    }
}

/// Image normalization parameters
#[derive(Debug, Clone)]
pub struct ImageNormalization {
    /// Mean values for each channel
    pub mean: [f32; 3],
    
    /// Standard deviation values for each channel
    pub std: [f32; 3],
}

impl Default for ImageNormalization {
    /// Default ImageNet normalization
    fn default() -> Self {
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

impl ImageNormalization {
    /// Create custom normalization
    pub fn custom(mean: [f32; 3], std: [f32; 3]) -> Self {
        Self { mean, std }
    }
    
    /// No normalization (identity)
    pub fn none() -> Self {
        Self {
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        }
    }
    
    /// Normalize a pixel value for a specific channel
    pub fn normalize_pixel(&self, value: f32, channel: usize) -> f32 {
        (value - self.mean[channel]) / self.std[channel]
    }
}

/// Preprocessing options
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    /// Whether to resize images to fit input size
    pub resize: bool,
    
    /// Resize filter type
    pub filter_type: image::imageops::FilterType,
    
    /// Whether to center crop after resize
    pub center_crop: bool,
    
    /// Whether to apply normalization
    pub normalize: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            resize: true,
            filter_type: image::imageops::FilterType::Lanczos3,
            center_crop: false,
            normalize: true,
        }
    }
}

/// Result of an object detection operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// List of detected objects
    pub detections: Vec<Detection>,
    
    /// Total processing time in milliseconds
    pub processing_time_ms: Option<u64>,
    
    /// Image dimensions used for detection
    pub image_width: u32,
    pub image_height: u32,
    
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

impl DetectionResult {
    /// Get detections above a confidence threshold
    pub fn filter_by_confidence(&self, min_confidence: f32) -> Vec<&Detection> {
        self.detections
            .iter()
            .filter(|d| d.confidence >= min_confidence)
            .collect()
    }
    
    /// Get detections for specific class IDs
    pub fn filter_by_classes(&self, class_ids: &[usize]) -> Vec<&Detection> {
        self.detections
            .iter()
            .filter(|d| class_ids.contains(&d.class_id))
            .collect()
    }
    
    /// Get the N most confident detections
    pub fn top_n_detections(&self, n: usize) -> Vec<&Detection> {
        let mut sorted: Vec<&Detection> = self.detections.iter().collect();
        sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }
}

/// A single object detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    /// Bounding box coordinates
    pub bbox: BoundingBox,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    
    /// Predicted class ID
    pub class_id: usize,
    
    /// Human-readable class name (if available)
    pub class_name: Option<String>,
    
    /// Optional tracking ID for video sequences
    pub track_id: Option<u32>,
}

impl Detection {
    /// Calculate the area of the bounding box
    pub fn area(&self) -> f32 {
        self.bbox.area()
    }
    
    /// Calculate Intersection over Union (IoU) with another detection
    pub fn iou(&self, other: &Detection) -> f32 {
        self.bbox.iou(&other.bbox)
    }
}

/// Bounding box coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Left x coordinate
    pub x1: f32,
    
    /// Top y coordinate  
    pub y1: f32,
    
    /// Right x coordinate
    pub x2: f32,
    
    /// Bottom y coordinate
    pub y2: f32,
}

impl BoundingBox {
    /// Create a new bounding box
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }
    
    /// Calculate the area of the bounding box
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }
    
    /// Calculate the center point
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }
    
    /// Calculate width and height
    pub fn width_height(&self) -> (f32, f32) {
        (self.x2 - self.x1, self.y2 - self.y1)
    }
    
    /// Calculate Intersection over Union (IoU) with another bounding box
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let intersection = self.intersection(other);
        if intersection.area() == 0.0 {
            return 0.0;
        }
        
        let union_area = self.area() + other.area() - intersection.area();
        if union_area == 0.0 {
            return 0.0;
        }
        
        intersection.area() / union_area
    }
    
    /// Calculate intersection with another bounding box
    pub fn intersection(&self, other: &BoundingBox) -> BoundingBox {
        let x1 = self.x1.max(other.x1);
        let y1 = self.y1.max(other.y1);
        let x2 = self.x2.min(other.x2);
        let y2 = self.y2.min(other.y2);
        
        if x1 >= x2 || y1 >= y2 {
            // No intersection
            BoundingBox::new(0.0, 0.0, 0.0, 0.0)
        } else {
            BoundingBox::new(x1, y1, x2, y2)
        }
    }
    
    /// Scale bounding box coordinates to image dimensions
    pub fn scale_to_image(&self, image_width: u32, image_height: u32) -> BoundingBox {
        BoundingBox::new(
            self.x1 * image_width as f32,
            self.y1 * image_height as f32,
            self.x2 * image_width as f32,
            self.y2 * image_height as f32,
        )
    }
    
    /// Normalize bounding box coordinates (0.0 to 1.0)
    pub fn normalize(&self, image_width: u32, image_height: u32) -> BoundingBox {
        BoundingBox::new(
            self.x1 / image_width as f32,
            self.y1 / image_height as f32,
            self.x2 / image_width as f32,
            self.y2 / image_height as f32,
        )
    }
}