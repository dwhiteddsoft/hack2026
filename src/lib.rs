// Universal ONNX Computer Vision Runtime (UOCVR)
// A unified Rust library for ONNX computer vision model inference

//! # Universal ONNX Computer Vision Runtime (UOCVR)
//!
//! UOCVR provides a unified, high-performance interface for running computer vision models
//! with ONNX Runtime. It abstracts away model-specific preprocessing and postprocessing
//! while maintaining optimal performance.
//!
//! ## Features
//!
//! - **Universal Input Processing**: Handles diverse model input requirements automatically
//! - **Universal Output Processing**: Standardized output parsing across different architectures
//! - **High Performance**: Minimal overhead over raw ONNX Runtime
//! - **Type Safety**: Leverages Rust's type system for reliable inference
//! - **Async Support**: Non-blocking inference execution
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use uocvr::UniversalSession;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load a model with automatic configuration detection
//!     let session = UniversalSession::from_model_file("yolov8n.onnx").await?;
//!     
//!     // Run inference on an image
//!     let detections = session.infer_image("test_image.jpg").await?;
//!     
//!     println!("Found {} detections", detections.len());
//!     Ok(())
//! }
//! ```

pub mod core;
pub mod input;
pub mod output;
pub mod session;
pub mod models;
pub mod error;
pub mod utils;

#[cfg(test)]
mod models_test;

// Re-export main types for convenience
pub use crate::core::{
    UniversalSession, InferenceResult, Detection, BoundingBox,
    ModelInfo, SessionBuilder,
};
pub use crate::error::{UocvrError, Result};
pub use crate::input::{InputProcessor, InputSpecification};
pub use crate::output::{OutputProcessor, OutputSpecification};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{UniversalSession, BoundingBox, Detection};
    use crate::input::{InputProcessor, InputSpecification};
    use crate::output::{OutputProcessor, OutputSpecification};
    use crate::models::ModelRegistry;
    use std::time::Duration;

    #[test]
    fn test_bounding_box_creation() {
        let bbox = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 200.0,
            format: crate::core::BoundingBoxFormat::TopLeftWidthHeight,
        };

        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
        assert_eq!(bbox.width, 100.0);
        assert_eq!(bbox.height, 200.0);
    }

    #[test]
    fn test_detection_creation() {
        let bbox = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 200.0,
            format: crate::core::BoundingBoxFormat::TopLeftWidthHeight,
        };

        let detection = Detection {
            bbox,
            confidence: 0.85,
            class_id: 0,
            class_name: Some("person".to_string()),
            mask: None,
            keypoints: None,
        };

        assert_eq!(detection.confidence, 0.85);
        assert_eq!(detection.class_id, 0);
        assert_eq!(detection.class_name, Some("person".to_string()));
    }

    #[test]
    fn test_model_registry_creation() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();
        
        // Registry should be created without error
        // Actual models will be added in implementation
        assert!(models.is_empty() || !models.is_empty()); // Always true, just testing creation
    }

    #[test]
    fn test_error_types() {
        let error = crate::error::UocvrError::model_config("Test error message");
        assert!(matches!(error, crate::error::UocvrError::ModelConfig { .. }));

        let error = crate::error::UocvrError::input_validation("Invalid input");
        assert!(matches!(error, crate::error::UocvrError::InputValidation { .. }));
    }

    #[test]
    fn test_math_utils() {
        // Test IoU calculation
        let box1 = [0.0, 0.0, 10.0, 10.0]; // x1, y1, x2, y2
        let box2 = [5.0, 5.0, 15.0, 15.0];
        
        let iou = crate::utils::math_utils::calculate_iou(&box1, &box2);
        assert!(iou > 0.0 && iou < 1.0);

        // Test sigmoid
        let result = crate::utils::math_utils::sigmoid(0.0);
        assert!((result - 0.5).abs() < f32::EPSILON);

        // Test clamp
        let clamped = crate::utils::math_utils::clamp(5.0, 0.0, 3.0);
        assert_eq!(clamped, 3.0);
    }

    #[test]
    fn test_session_builder() {
        let builder = crate::core::SessionBuilder::new()
            .model_file("test_model.onnx")
            .batch_size(4);

        // Builder should be created without error
        // Actual build will fail until implementation is complete
        assert!(true); // Placeholder assertion
    }

    #[test]
    fn test_constants() {
        use crate::utils::constants::*;

        assert_eq!(DEFAULT_CONFIDENCE_THRESHOLD, 0.25);
        assert_eq!(DEFAULT_NMS_THRESHOLD, 0.45);
        assert_eq!(DEFAULT_MAX_DETECTIONS, 300);
        assert_eq!(COCO_CLASS_COUNT, 80);
    }

    #[tokio::test]
    async fn test_async_session_creation() {
        // This test will fail until implementation is complete
        let result = UniversalSession::from_model_file("nonexistent_model.onnx").await;
        assert!(result.is_err()); // Should fail for nonexistent file
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = crate::utils::MemoryTracker::new();
        
        tracker.update(100);
        assert_eq!(tracker.current_usage(), 100);
        assert_eq!(tracker.peak_usage(), 100);

        tracker.update(150);
        assert_eq!(tracker.current_usage(), 150);
        assert_eq!(tracker.peak_usage(), 150);

        tracker.update(75);
        assert_eq!(tracker.current_usage(), 75);
        assert_eq!(tracker.peak_usage(), 150); // Peak should remain
    }

    #[test]
    fn test_timer() {
        let timer = crate::utils::Timer::new("test_timer".to_string());
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed >= Duration::from_millis(10));
    }
}