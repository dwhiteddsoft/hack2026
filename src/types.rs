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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        }
    }
    
    /// Check if this model type supports streaming
    pub fn supports_streaming(&self) -> bool {
        !matches!(self, ModelType::SingleFrame)
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