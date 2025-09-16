//! Main classifier implementation for ONNX vision models.

use ort::{Environment, SessionBuilder, Session, Value};
use image::RgbImage;
use std::path::Path;
use std::sync::Arc;

use crate::{
    types::*,
    buffer::FrameBuffer,
    preprocessing::{ImagePreprocessor, DefaultPreprocessor},
    Result, ClassificationError,
};

/// Main vision classifier for ONNX models
pub struct VisionClassifier {
    session: Session,
    config: ModelConfig,
    preprocessor: Box<dyn ImagePreprocessor + Send + Sync>,
    frame_buffer: Option<FrameBuffer>,
}

impl VisionClassifier {
    /// Create a new vision classifier
    pub fn new(
        model_path: impl AsRef<Path>,
        config: ModelConfig,
    ) -> Result<Self> {
        // Initialize ONNX Runtime environment
        let environment = Arc::new(Environment::builder()
            .with_name("onnx_vision_classifier")
            .build()
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?);
        
        // Create session
        let session = SessionBuilder::new(&environment)
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?
            .with_model_from_file(model_path)
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        
        let frame_buffer = match config.model_type {
            ModelType::SingleFrame => None,
            ModelType::MultiFrame { frame_count } => {
                Some(FrameBuffer::new(frame_count * 2, frame_count))
            },
            ModelType::Variable { max_frames, min_frames } => {
                Some(FrameBuffer::new(max_frames * 2, min_frames))
            },
            ModelType::LSTM { sequence_length } => {
                Some(FrameBuffer::new(sequence_length * 2, sequence_length))
            },
            ModelType::TwoStream => {
                Some(FrameBuffer::new(4, 2)) // Need current + previous frame
            },
        };
        
        Ok(Self {
            session,
            config,
            preprocessor: Box::new(DefaultPreprocessor::new()),
            frame_buffer,
        })
    }
    
    /// Set a custom preprocessor
    pub fn with_preprocessor(
        mut self,
        preprocessor: Box<dyn ImagePreprocessor + Send + Sync>,
    ) -> Self {
        self.preprocessor = preprocessor;
        self
    }
    
    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    
    /// Classify a single frame (for SingleFrame models only)
    pub fn classify_single(&self, frame: &RgbImage) -> Result<ClassificationResult> {
        match self.config.model_type {
            ModelType::SingleFrame => self.classify_single_frame(frame),
            _ => Err(ClassificationError::InvalidInput(
                "Model requires multiple frames".to_string()
            )),
        }
    }
    
    /// Classify a single frame and return result as JSON string
    pub fn classify_single_json(&self, frame: &RgbImage) -> Result<String> {
        let result = self.classify_single(frame)?;
        serde_json::to_string(&result)
            .map_err(ClassificationError::SerializationError)
    }
    
    /// Classify multiple frames
    pub fn classify_frames(&self, frames: &[RgbImage]) -> Result<ClassificationResult> {
        let frame_refs: Vec<&RgbImage> = frames.iter().collect();
        self.classify_frame_refs(&frame_refs)
    }
    
    /// Classify frames by reference
    pub fn classify_frame_refs(&self, frames: &[&RgbImage]) -> Result<ClassificationResult> {
        match self.config.model_type {
            ModelType::SingleFrame => {
                if frames.len() != 1 {
                    return Err(ClassificationError::InvalidInput(
                        "Single frame model expects exactly one frame".to_string()
                    ));
                }
                self.classify_single_frame(frames[0])
            },
            ModelType::MultiFrame { frame_count } => {
                if frames.len() != frame_count {
                    return Err(ClassificationError::InsufficientFrames {
                        expected: frame_count,
                        actual: frames.len(),
                    });
                }
                self.classify_multi_frame(frames)
            },
            ModelType::Variable { min_frames, max_frames } => {
                if frames.len() < min_frames || frames.len() > max_frames {
                    return Err(ClassificationError::InvalidInput(
                        format!("Frame count must be between {} and {}", min_frames, max_frames)
                    ));
                }
                self.classify_multi_frame(frames)
            },
            ModelType::LSTM { sequence_length } => {
                if frames.len() != sequence_length {
                    return Err(ClassificationError::InsufficientFrames {
                        expected: sequence_length,
                        actual: frames.len(),
                    });
                }
                self.classify_lstm_sequence(frames)
            },
            ModelType::TwoStream => {
                if frames.len() != 2 {
                    return Err(ClassificationError::InsufficientFrames {
                        expected: 2,
                        actual: frames.len(),
                    });
                }
                self.classify_two_stream(frames)
            },
        }
    }
    
    /// Push a frame for streaming classification
    /// 
    /// Returns Some(result) when enough frames are available for classification
    pub fn push_frame(&mut self, frame: RgbImage) -> Result<Option<ClassificationResult>> {
        // First, check if we have a buffer and push the frame
        let is_ready = match &mut self.frame_buffer {
            Some(buffer) => {
                buffer.push(frame);
                buffer.is_ready()
            },
            None => return Err(ClassificationError::InvalidInput(
                "Model doesn't support streaming frames".to_string()
            )),
        };
        
        // If buffer is ready, extract frames and classify
        if is_ready {
            let model_type = self.config.model_type;
            let frames = match &self.frame_buffer {
                Some(buffer) => match model_type {
                    ModelType::MultiFrame { frame_count } => buffer.get_latest_frames(frame_count)?,
                    ModelType::LSTM { sequence_length } => buffer.get_latest_frames(sequence_length)?,
                    ModelType::TwoStream => buffer.get_latest_frames(2)?,
                    ModelType::Variable { min_frames, .. } => buffer.get_latest_frames(min_frames)?,
                    _ => return Ok(None),
                },
                None => unreachable!(),
            };
            
            // Now classify the frames
            match model_type {
                ModelType::MultiFrame { .. } => {
                    let result = self.classify_frame_refs(&frames)?;
                    Ok(Some(result))
                },
                ModelType::LSTM { .. } => {
                    let result = self.classify_lstm_sequence(&frames)?;
                    Ok(Some(result))
                },
                ModelType::TwoStream => {
                    let result = self.classify_two_stream(&frames)?;
                    Ok(Some(result))
                },
                ModelType::Variable { .. } => {
                    let result = self.classify_multi_frame(&frames)?;
                    Ok(Some(result))
                },
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }
    
    /// Clear the frame buffer
    pub fn clear_buffer(&mut self) {
        if let Some(ref mut buffer) = self.frame_buffer {
            buffer.clear();
        }
    }
    
    /// Check if the frame buffer is ready for classification
    pub fn is_buffer_ready(&self) -> bool {
        self.frame_buffer.as_ref().map_or(false, |b| b.is_ready())
    }
    
    /// Get current frame buffer length
    pub fn buffer_len(&self) -> usize {
        self.frame_buffer.as_ref().map_or(0, |b| b.len())
    }
    
    // Internal classification methods
    
    fn classify_single_frame(&self, frame: &RgbImage) -> Result<ClassificationResult> {
        // Preprocess the frame
        let processed = self.preprocessor.preprocess_single_frame(
            frame,
            self.config.input_size,
            &self.config.normalization,
        )?;
        
        // Add batch dimension: (C, H, W) -> (1, C, H, W)
        let input_array = processed.insert_axis(ndarray::Axis(0));
        
        // Create ONNX input tensor - convert to CowArray
        let input_cow = ndarray::CowArray::from(input_array.view()).into_dyn();
        let input_tensor = Value::from_array(self.session.allocator(), &input_cow)
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        
        // Run inference
        let outputs = self.session.run(vec![input_tensor])
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        
        self.parse_classification_output(&outputs)
    }
    
    fn classify_multi_frame(&self, frames: &[&RgbImage]) -> Result<ClassificationResult> {
        // Preprocess the frames
        let processed = self.preprocessor.preprocess_video_clip(
            frames,
            self.config.input_size,
            &self.config.normalization,
        )?;
        
        // Create ONNX input tensor
        let input_cow = ndarray::CowArray::from(processed.view()).into_dyn();
        let input_tensor = Value::from_array(self.session.allocator(), &input_cow)
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        
        // Run inference
        let outputs = self.session.run(vec![input_tensor])
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        
        self.parse_classification_output(&outputs)
    }
    
    fn classify_lstm_sequence(&self, frames: &[&RgbImage]) -> Result<ClassificationResult> {
        // For LSTM, we typically extract features first, then classify
        let processed = self.preprocessor.preprocess_multiple_frames(
            frames,
            self.config.input_size,
            &self.config.normalization,
        )?;
        
        // Add batch dimension: (T, C, H, W) -> (1, T, C, H, W)
        let input_array = processed.insert_axis(ndarray::Axis(0));
        
        // Create ONNX input tensor
        let input_cow = ndarray::CowArray::from(input_array.view()).into_dyn();
        let input_tensor = Value::from_array(self.session.allocator(), &input_cow)
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        
        // Run inference
        let outputs = self.session.run(vec![input_tensor])
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        
        self.parse_classification_output(&outputs)
    }
    
    fn classify_two_stream(&self, frames: &[&RgbImage]) -> Result<ClassificationResult> {
        // For two-stream, we need RGB + optical flow
        // This is a simplified implementation
        let rgb_processed = self.preprocessor.preprocess_single_frame(
            frames[1], // Current frame
            self.config.input_size,
            &self.config.normalization,
        )?;
        
        // In a real implementation, you'd compute optical flow here
        // For now, we'll use a simple frame difference as a proxy
        let flow_processed = self.compute_optical_flow(frames[0], frames[1])?;
        
        // Add batch dimensions
        let rgb_input = rgb_processed.insert_axis(ndarray::Axis(0));
        let flow_input = flow_processed.insert_axis(ndarray::Axis(0));
        
        // Create ONNX input tensors
        let rgb_cow = ndarray::CowArray::from(rgb_input.view()).into_dyn();
        let flow_cow = ndarray::CowArray::from(flow_input.view()).into_dyn();
        
        let rgb_tensor = Value::from_array(self.session.allocator(), &rgb_cow)
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        let flow_tensor = Value::from_array(self.session.allocator(), &flow_cow)
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        
        // Run inference with both inputs
        let outputs = self.session.run(vec![rgb_tensor, flow_tensor])
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        
        self.parse_classification_output(&outputs)
    }
    
    fn compute_optical_flow(&self, frame1: &RgbImage, frame2: &RgbImage) -> Result<ndarray::Array3<f32>> {
        // Simplified optical flow computation using frame difference
        // In practice, you'd use a proper optical flow algorithm like Lucas-Kanade or Farneback
        let (width, height) = self.config.input_size;
        
        // Convert images to grayscale and compute difference
        let img1 = image::imageops::grayscale(frame1);
        let img2 = image::imageops::grayscale(frame2);
        
        // Resize to target size
        let img1_resized = image::imageops::resize(&img1, width, height, image::imageops::FilterType::Lanczos3);
        let img2_resized = image::imageops::resize(&img2, width, height, image::imageops::FilterType::Lanczos3);
        
        // Create flow array (2 channels: dx, dy)
        let mut flow = ndarray::Array3::<f32>::zeros((2, height as usize, width as usize));
        
        // Simple frame difference as optical flow proxy
        for y in 0..height as usize {
            for x in 0..width as usize {
                let pixel1 = img1_resized.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
                let pixel2 = img2_resized.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
                let diff = pixel2 - pixel1;
                
                // Simple flow estimation (this is very basic)
                flow[[0, y, x]] = diff; // x-direction flow
                flow[[1, y, x]] = diff; // y-direction flow
            }
        }
        
        Ok(flow)
    }
    
    fn parse_classification_output(&self, outputs: &[Value]) -> Result<ClassificationResult> {
        if outputs.is_empty() {
            return Err(ClassificationError::InvalidInput("No model outputs".to_string()));
        }
        
        // Extract the output tensor
        let output = &outputs[0];
        let output_tensor = output.try_extract::<f32>()
            .map_err(|e| ClassificationError::OnnxError(e.to_string()))?;
        
        // Get the scores from the tensor
        let scores = output_tensor.view().iter().copied().collect::<Vec<f32>>();
        
        if scores.is_empty() {
            return Err(ClassificationError::InvalidInput("Empty prediction scores".to_string()));
        }
        
        // Find the class with highest confidence
        let (class_id, &confidence) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        
        let class_name = self.config.class_names
            .as_ref()
            .and_then(|names| names.get(class_id))
            .cloned();
        
        Ok(ClassificationResult {
            class_id,
            class_name,
            confidence,
            all_scores: scores,
            metadata: None,
        })
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for VisionClassifier {}
unsafe impl Sync for VisionClassifier {}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{RgbImage, Rgb};

    fn create_test_image(width: u32, height: u32, color: [u8; 3]) -> RgbImage {
        RgbImage::from_fn(width, height, |_, _| Rgb(color))
    }

    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig::new(ModelType::SingleFrame)
            .with_input_size(256, 256)
            .with_class_names(vec!["cat".to_string(), "dog".to_string()]);
        
        assert_eq!(config.input_size, (256, 256));
        assert_eq!(config.class_names.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_model_type_properties() {
        assert_eq!(ModelType::SingleFrame.required_frames(), Some(1));
        assert_eq!(ModelType::MultiFrame { frame_count: 16 }.required_frames(), Some(16));
        assert_eq!(ModelType::TwoStream.required_frames(), Some(2));
        
        assert!(!ModelType::SingleFrame.supports_streaming());
        assert!(ModelType::MultiFrame { frame_count: 8 }.supports_streaming());
    }

    #[test]
    fn test_classification_result_top_n() {
        let result = ClassificationResult {
            class_id: 2,
            class_name: Some("dog".to_string()),
            confidence: 0.8,
            all_scores: vec![0.1, 0.2, 0.8, 0.3, 0.7],
            metadata: None,
        };
        
        let top_2 = result.top_n(2);
        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0], (2, 0.8)); // Highest score
        assert_eq!(top_2[1], (4, 0.7)); // Second highest
        
        let class_names = vec![
            "cat".to_string(), 
            "bird".to_string(), 
            "dog".to_string(), 
            "fish".to_string(), 
            "horse".to_string()
        ];
        let top_2_with_names = result.top_n_with_names(2, &class_names);
        assert_eq!(top_2_with_names[0], ("dog".to_string(), 0.8));
        assert_eq!(top_2_with_names[1], ("horse".to_string(), 0.7));
    }

    #[test]
    #[ignore] // Ignore tests that require actual ONNX models
    fn test_single_frame_classifier() -> Result<()> {
        let config = ModelConfig::new(ModelType::SingleFrame);
        let classifier = VisionClassifier::new("mock_model.onnx", config)?;
        
        let image = create_test_image(224, 224, [255, 0, 0]);
        let result = classifier.classify_single(&image)?;
        
        assert!(result.confidence > 0.0);
        assert!(!result.all_scores.is_empty());
        
        Ok(())
    }

    #[test]
    #[ignore] // Ignore tests that require actual ONNX models
    fn test_multi_frame_classifier() -> Result<()> {
        let config = ModelConfig::new(ModelType::MultiFrame { frame_count: 4 });
        let classifier = VisionClassifier::new("mock_model.onnx", config)?;
        
        let frames = (0..4)
            .map(|i| create_test_image(112, 112, [i * 50, 0, 0]))
            .collect::<Vec<_>>();
        
        let result = classifier.classify_frames(&frames)?;
        
        assert!(result.confidence > 0.0);
        assert!(!result.all_scores.is_empty());
        
        Ok(())
    }

    #[test]
    #[ignore] // Ignore tests that require actual ONNX models
    fn test_streaming_classifier() -> Result<()> {
        let config = ModelConfig::new(ModelType::MultiFrame { frame_count: 3 });
        let mut classifier = VisionClassifier::new("mock_model.onnx", config)?;
        
        // Push frames one by one
        for i in 0..5 {
            let frame = create_test_image(224, 224, [i * 40, 0, 0]);
            let result = classifier.push_frame(frame)?;
            
            if i >= 2 { // Should have results after 3rd frame
                assert!(result.is_some());
            } else {
                assert!(result.is_none());
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_classification_result_json() {
        use crate::types::ClassificationResult;
        
        // Create a mock classification result
        let result = ClassificationResult {
            class_id: 487,
            class_name: Some("castle".to_string()),
            confidence: 0.8754,
            all_scores: vec![0.1, 0.2, 0.8754, 0.3],
            metadata: None,
        };
        
        // Test JSON serialization
        let json_str = serde_json::to_string(&result).unwrap();
        assert!(json_str.contains("\"class_id\":487"));
        assert!(json_str.contains("\"class_name\":\"castle\""));
        assert!(json_str.contains("\"confidence\":0.8754"));
        
        // Test deserialization
        let deserialized: ClassificationResult = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.class_id, 487);
        assert_eq!(deserialized.class_name, Some("castle".to_string()));
        assert_eq!(deserialized.confidence, 0.8754);
    }
}