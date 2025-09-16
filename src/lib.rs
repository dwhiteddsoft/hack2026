//! # ONNX Vision Classifier
//!
//! A Rust library for ONNX-based image and video classification that supports
//! both single-frame and multi-frame models.
//!
//! ## Features
//!
//! - Single-frame image classification (ResNet, EfficientNet, etc.)
//! - Multi-frame video classification (I3D, SlowFast, etc.)
//! - LSTM-based sequence classification
//! - Two-stream models (RGB + Optical Flow)
//! - Streaming frame processing with automatic buffering
//! - Flexible preprocessing pipeline
//! - Builder pattern for easy configuration
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use onnx_vision_classifier::{classifier, RgbImage};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Single frame classification
//! let classifier = classifier()
//!     .model_path("models/resnet50.onnx")
//!     .single_frame()
//!     .input_size(224, 224)
//!     .imagenet_normalization()
//!     .build()?;
//!
//! let image = image::open("test.jpg")?.to_rgb8();
//! let result = classifier.classify_single(&image)?;
//! println!("Class: {} (confidence: {:.2})", 
//!          result.class_name.unwrap_or("Unknown".to_string()), 
//!          result.confidence);
//! # Ok(())
//! # }
//! ```

mod error;
mod types;
mod buffer;
mod preprocessing;
mod classifier;
mod builder;

pub use error::{ClassificationError, Result};
pub use types::{ClassificationResult, ModelType, ModelConfig, ImageNormalization};
pub use classifier::VisionClassifier;
pub use builder::ClassifierBuilder;
pub use preprocessing::{ImagePreprocessor, DefaultPreprocessor};

#[cfg(feature = "async")]
pub mod async_classifier;

// Re-export commonly used types
pub use image::RgbImage;

/// Create a new classifier builder
///
/// This is the main entry point for creating vision classifiers.
///
/// # Example
///
/// ```rust,no_run
/// use onnx_vision_classifier::classifier;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let classifier = classifier()
///     .model_path("model.onnx")
///     .single_frame()
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub fn classifier() -> ClassifierBuilder {
    ClassifierBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let _builder = classifier();
    }

    #[test]
    fn test_model_types() {
        use crate::ModelType;
        
        let single = ModelType::SingleFrame;
        let multi = ModelType::MultiFrame { frame_count: 16 };
        let variable = ModelType::Variable { min_frames: 8, max_frames: 32 };
        let lstm = ModelType::LSTM { sequence_length: 10 };
        let two_stream = ModelType::TwoStream;
        
        // Just ensure they can be created
        assert!(matches!(single, ModelType::SingleFrame));
        assert!(matches!(multi, ModelType::MultiFrame { frame_count: 16 }));
        assert!(matches!(variable, ModelType::Variable { min_frames: 8, max_frames: 32 }));
        assert!(matches!(lstm, ModelType::LSTM { sequence_length: 10 }));
        assert!(matches!(two_stream, ModelType::TwoStream));
    }
}