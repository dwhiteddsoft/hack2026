//! Builder pattern for easy classifier configuration.

use std::path::Path;
use crate::{VisionClassifier, ModelConfig, ModelType, ImageNormalization, preprocessing::ImagePreprocessor, Result, ClassificationError};

/// Builder for configuring and creating vision classifiers
pub struct ClassifierBuilder {
    model_path: Option<String>,
    model_type: Option<ModelType>,
    input_size: Option<(u32, u32)>,
    class_names: Option<Vec<String>>,
    normalization: Option<ImageNormalization>,
    preprocessor: Option<Box<dyn ImagePreprocessor + Send + Sync>>,
}

impl ClassifierBuilder {
    /// Create a new classifier builder
    pub fn new() -> Self {
        Self {
            model_path: None,
            model_type: None,
            input_size: None,
            class_names: None,
            normalization: None,
            preprocessor: None,
        }
    }
    
    /// Set the model file path
    pub fn model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.model_path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }
    
    /// Configure for single frame image classification
    pub fn single_frame(mut self) -> Self {
        self.model_type = Some(ModelType::SingleFrame);
        self
    }
    
    /// Configure for multi-frame video classification
    pub fn multi_frame(mut self, frame_count: usize) -> Self {
        self.model_type = Some(ModelType::MultiFrame { frame_count });
        self
    }
    
    /// Configure for variable frame count models
    pub fn variable_frames(mut self, min_frames: usize, max_frames: usize) -> Self {
        if min_frames > max_frames {
            panic!("min_frames cannot be greater than max_frames");
        }
        self.model_type = Some(ModelType::Variable { min_frames, max_frames });
        self
    }
    
    /// Configure for LSTM sequence classification
    pub fn lstm_sequence(mut self, sequence_length: usize) -> Self {
        self.model_type = Some(ModelType::LSTM { sequence_length });
        self
    }
    
    /// Configure for two-stream models (RGB + Optical Flow)
    pub fn two_stream(mut self) -> Self {
        self.model_type = Some(ModelType::TwoStream);
        self
    }
    
    /// Configure for object detection with custom thresholds
    pub fn object_detection(mut self, num_classes: usize, confidence_threshold: f32, nms_threshold: f32) -> Self {
        self.model_type = Some(ModelType::ObjectDetection {
            num_classes,
            confidence_threshold,
            nms_threshold,
        });
        self
    }
    
    /// Configure for object detection with default thresholds
    /// Uses num_classes = 80 (COCO), confidence_threshold = 0.5, nms_threshold = 0.45
    pub fn object_detection_default(self) -> Self {
        self.object_detection(80, 0.5, 0.45)
    }
    
    /// Configure for YOLO object detection
    /// Uses optimized thresholds for YOLO models (COCO dataset: 80 classes)
    pub fn yolo_detection(self) -> Self {
        self.object_detection(80, 0.25, 0.45)
    }
    
    /// Configure for strict object detection (higher confidence threshold)
    pub fn strict_object_detection(self) -> Self {
        self.object_detection(80, 0.95, 0.3)
    }
    
    /// Set input image size
    pub fn input_size(mut self, width: u32, height: u32) -> Self {
        self.input_size = Some((width, height));
        self
    }
    
    /// Set class names for human-readable output
    pub fn class_names(mut self, names: Vec<String>) -> Self {
        self.class_names = Some(names);
        self
    }
    
    /// Load class names from a file (one class per line)
    pub fn class_names_from_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ClassificationError::IoError(e))?;
        
        let names: Vec<String> = content
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();
        
        self.class_names = Some(names);
        Ok(self)
    }
    
    /// Use ImageNet normalization (default)
    pub fn imagenet_normalization(mut self) -> Self {
        self.normalization = Some(ImageNormalization::default());
        self
    }
    
    /// Use custom normalization
    pub fn custom_normalization(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.normalization = Some(ImageNormalization::custom(mean, std));
        self
    }
    
    /// No normalization (values stay in 0-1 range)
    pub fn no_normalization(mut self) -> Self {
        self.normalization = Some(ImageNormalization::none());
        self
    }
    
    /// Set a custom preprocessor
    pub fn with_preprocessor(mut self, preprocessor: Box<dyn ImagePreprocessor + Send + Sync>) -> Self {
        self.preprocessor = Some(preprocessor);
        self
    }
    
    /// Build the classifier
    pub fn build(self) -> Result<VisionClassifier> {
        let model_path = self.model_path.ok_or_else(|| {
            ClassificationError::InvalidInput("Model path is required".to_string())
        })?;
        
        let model_type = self.model_type.unwrap_or(ModelType::SingleFrame);
        
        let config = ModelConfig {
            model_type,
            input_size: self.input_size.unwrap_or((224, 224)),
            channels: 3,
            class_names: self.class_names,
            normalization: self.normalization.unwrap_or_default(),
            metadata: None,
        };
        
        let mut classifier = VisionClassifier::new(model_path, config)?;
        
        if let Some(preprocessor) = self.preprocessor {
            classifier = classifier.with_preprocessor(preprocessor);
        }
        
        Ok(classifier)
    }
}

impl Default for ClassifierBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common model configurations
impl ClassifierBuilder {
    /// Create a ResNet-style single frame classifier
    pub fn resnet_classifier<P: AsRef<Path>>(model_path: P) -> Self {
        Self::new()
            .model_path(model_path)
            .single_frame()
            .input_size(224, 224)
            .imagenet_normalization()
    }
    
    /// Create an EfficientNet-style single frame classifier
    pub fn efficientnet_classifier<P: AsRef<Path>>(model_path: P, input_size: u32) -> Self {
        Self::new()
            .model_path(model_path)
            .single_frame()
            .input_size(input_size, input_size)
            .imagenet_normalization()
    }
    
    /// Create an I3D-style video classifier
    pub fn i3d_classifier<P: AsRef<Path>>(model_path: P) -> Self {
        Self::new()
            .model_path(model_path)
            .multi_frame(64) // I3D typically uses 64 frames
            .input_size(224, 224)
            .imagenet_normalization()
    }
    
    /// Create a SlowFast-style video classifier
    pub fn slowfast_classifier<P: AsRef<Path>>(model_path: P) -> Self {
        Self::new()
            .model_path(model_path)
            .multi_frame(32) // SlowFast typically uses 32 frames for fast pathway
            .input_size(224, 224)
            .imagenet_normalization()
    }
    
    /// Create a two-stream classifier
    pub fn two_stream_classifier<P: AsRef<Path>>(model_path: P) -> Self {
        Self::new()
            .model_path(model_path)
            .two_stream()
            .input_size(224, 224)
            .imagenet_normalization()
    }
}

/// Helper functions for loading common datasets' class names
impl ClassifierBuilder {
    /// Load ImageNet class names
    pub fn with_imagenet_classes(self) -> Self {
        // In a real implementation, you might load these from a file or embed them
        let imagenet_classes = generate_imagenet_classes();
        self.class_names(imagenet_classes)
    }
    
    /// Load Kinetics-400 action classes
    pub fn with_kinetics400_classes(self) -> Self {
        let kinetics_classes = generate_kinetics400_classes();
        self.class_names(kinetics_classes)
    }
    
    /// Load COCO object detection classes
    pub fn with_coco_classes(self) -> Self {
        let coco_classes = generate_coco_classes();
        self.class_names(coco_classes)
    }
}

// Helper functions to generate class names (in a real implementation, these would load from files)
fn generate_imagenet_classes() -> Vec<String> {
    // This is a simplified version - in practice you'd load the full 1000 ImageNet classes
    vec![
        "tench".to_string(),
        "goldfish".to_string(),
        "great_white_shark".to_string(),
        "tiger_shark".to_string(),
        "hammerhead".to_string(),
        // ... would continue with all 1000 classes
    ]
}

fn generate_kinetics400_classes() -> Vec<String> {
    // This is a simplified version - in practice you'd load the full 400 Kinetics classes
    vec![
        "abseiling".to_string(),
        "air_drumming".to_string(),
        "answering_questions".to_string(),
        "applauding".to_string(),
        "applying_cream".to_string(),
        // ... would continue with all 400 classes
    ]
}

fn generate_coco_classes() -> Vec<String> {
    // COCO has 80 object classes
    vec![
        "person".to_string(),
        "bicycle".to_string(),
        "car".to_string(),
        "motorcycle".to_string(),
        "airplane".to_string(),
        "bus".to_string(),
        "train".to_string(),
        "truck".to_string(),
        "boat".to_string(),
        "traffic_light".to_string(),
        // ... would continue with all 80 COCO classes
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::DefaultPreprocessor;

    #[test]
    fn test_builder_basic() {
        let builder = ClassifierBuilder::new()
            .single_frame()
            .input_size(256, 256)
            .imagenet_normalization();
        
        // Can't actually build without a model path, but we can test the builder methods
        assert!(builder.model_type.is_some());
        assert!(builder.input_size.is_some());
        assert!(builder.normalization.is_some());
    }

    #[test]
    fn test_builder_with_class_names() {
        let builder = ClassifierBuilder::new()
            .class_names(vec!["cat".to_string(), "dog".to_string()])
            .with_imagenet_classes(); // This should override the previous class names
        
        assert!(builder.class_names.is_some());
    }

    #[test]
    fn test_convenience_constructors() {
        let _resnet_builder = ClassifierBuilder::resnet_classifier("model.onnx");
        let _efficientnet_builder = ClassifierBuilder::efficientnet_classifier("model.onnx", 380);
        let _i3d_builder = ClassifierBuilder::i3d_classifier("model.onnx");
    }

    #[test]
    #[should_panic(expected = "min_frames cannot be greater than max_frames")]
    fn test_invalid_variable_frames() {
        let _builder = ClassifierBuilder::new().variable_frames(10, 5);
    }

    #[test]
    fn test_custom_preprocessor() {
        let preprocessor = Box::new(DefaultPreprocessor::new());
        let builder = ClassifierBuilder::new().with_preprocessor(preprocessor);
        
        assert!(builder.preprocessor.is_some());
    }
}