//! Main classifier implementation for ONNX vision models.

use ort::{Environment, SessionBuilder, Session, Value};
use image::RgbImage;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::{
    types::*,
    buffer::FrameBuffer,
    preprocessing::{ImagePreprocessor, DefaultPreprocessor},
    detection::{NMS, DetectionFilter, CoordinateConverter},
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
            ModelType::ObjectDetection { .. } => None, // Single frame object detection
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
            ModelType::ObjectDetection { .. } => {
                // For object detection, we only return classification if specifically requested
                // This is mainly for models that support both tasks
                self.classify_single_frame(frame)
            },
            _ => Err(ClassificationError::InvalidInput(
                "Model requires multiple frames".to_string()
            )),
        }
    }
    
    /// Classify a single image with JSON output support
    pub fn classify_single_json(&self, image: &RgbImage) -> Result<String> {
        let result = self.classify_single(image)?;
        serde_json::to_string(&result)
            .map_err(|e| ClassificationError::SerializationError(e))
    }

    /// Detect objects in a single image
    pub fn detect_objects(&self, image: &RgbImage) -> Result<DetectionResult> {
        // Verify this is an object detection model
        match &self.config.model_type {
            ModelType::ObjectDetection { num_classes: _, confidence_threshold, nms_threshold } => {
                let start = Instant::now();
                
                // Preprocess the image
                let input_data = self.preprocessor.preprocess_single_frame(
                    image,
                    self.config.input_size,
                    &self.config.normalization,
                )?;
                
                // Try single input first (most common)
                let batched_array = input_data.clone().insert_axis(ndarray::Axis(0));
                let dynamic_array = batched_array.into_dyn();
                
                // Create owned data to avoid lifetime issues
                let image_owned = dynamic_array.to_owned();
                let image_cow = {
                    use ndarray::CowArray;
                    CowArray::from(image_owned.view())
                };
                let image_tensor = Value::from_array(self.session.allocator(), &image_cow)
                    .map_err(|e| ClassificationError::ModelInferenceError(e.to_string()))?;

                // Try single input inference first
                let outputs = match self.session.run(vec![image_tensor]) {
                    Ok(outputs) => outputs,
                    Err(e) if e.to_string().contains("image_shape") => {
                        println!("🔧 Model requires image_shape input, using dual-input mode...");
                        // This model requires dual inputs - implement proper solution
                        return self.detect_objects_dual_input(image, input_data);
                    }
                    Err(e) => return Err(ClassificationError::ModelInferenceError(e.to_string())),
                };                let inference_time = start.elapsed();
                
                // Process detection outputs
                let raw_detections = self.process_detection_outputs(&outputs, image.width(), image.height())?;
                
                // Apply NMS filtering
                let filtered_detections = NMS::apply(raw_detections, *nms_threshold);
                
                // Filter by confidence threshold
                let final_detections = DetectionFilter::by_confidence(filtered_detections, *confidence_threshold);
                
                Ok(DetectionResult {
                    detections: final_detections,
                    processing_time_ms: Some(inference_time.as_millis() as u64),
                    image_width: image.width(),
                    image_height: image.height(),
                    metadata: None,
                })
            }
            _ => Err(ClassificationError::ModelInferenceError(
                "Model is not configured for object detection".to_string()
            ))
        }
    }

    /// Detect objects in a single image with JSON output
    pub fn detect_objects_json(&self, image: &RgbImage) -> Result<String> {
        let result = self.detect_objects(image)?;
        serde_json::to_string(&result)
            .map_err(|e| ClassificationError::SerializationError(e))
    }

    /// Handle models that require dual inputs (image + image_shape)
    /// This solves the Rust lifetime issues with ONNX Runtime Value<'_> by using owned data
    fn detect_objects_dual_input(
        &self, 
        image: &RgbImage, 
        input_data: ndarray::Array3<f32>
    ) -> Result<DetectionResult> {
        let inference_time = std::time::Instant::now();
        
        // Create owned arrays to avoid lifetime conflicts
        let image_owned = input_data
            .insert_axis(ndarray::Axis(0))
            .into_dyn()
            .to_owned();
        
        let shape_owned = ndarray::Array2::from_shape_vec((1, 2), vec![
            image.height() as f32, 
            image.width() as f32
        ]).map_err(|e| ClassificationError::ModelInferenceError(e.to_string()))?
            .into_dyn().to_owned();
        
        // Create CowArrays from owned data - both will have same lifetime
        let image_cow = {
            use ndarray::CowArray;
            CowArray::from(image_owned.view())
        };
        let shape_cow = {
            use ndarray::CowArray;
            CowArray::from(shape_owned.view())
        };
        
        // Create tensors in same scope where they'll be used
        let image_tensor = Value::from_array(self.session.allocator(), &image_cow)
            .map_err(|e| ClassificationError::ModelInferenceError(e.to_string()))?;
        let shape_tensor = Value::from_array(self.session.allocator(), &shape_cow)
            .map_err(|e| ClassificationError::ModelInferenceError(e.to_string()))?;
        
        // Run inference while both tensors are in scope
        let outputs = self.session.run(vec![image_tensor, shape_tensor])
            .map_err(|e| ClassificationError::ModelInferenceError(e.to_string()))?;
        
        let inference_time = inference_time.elapsed();
        
        // Process detection outputs
        let raw_detections = self.process_detection_outputs(&outputs, image.width(), image.height())?;
        println!("🔍 Debug: Raw detections count: {}", raw_detections.len());
        
        // Apply NMS filtering based on model configuration
        if let ModelType::ObjectDetection { confidence_threshold, nms_threshold, .. } = &self.config.model_type {
            println!("🔍 Debug: Using confidence threshold: {}, NMS threshold: {}", confidence_threshold, nms_threshold);
            let filtered_detections = NMS::apply(raw_detections, *nms_threshold);
            println!("🔍 Debug: After NMS: {}", filtered_detections.len());
            let final_detections = DetectionFilter::by_confidence(filtered_detections, *confidence_threshold);
            println!("🔍 Debug: After confidence filter: {}", final_detections.len());
            
            Ok(DetectionResult {
                detections: final_detections,
                processing_time_ms: Some(inference_time.as_millis() as u64),
                image_width: image.width(),
                image_height: image.height(),
                metadata: None,
            })
        } else {
            Err(ClassificationError::ModelInferenceError("Not an object detection model".to_string()))
        }
    }

    /// Process raw detection outputs from ONNX model
    /// This handles different model formats (YOLO, SSD, etc.)
    fn process_detection_outputs(&self, outputs: &[Value], img_width: u32, img_height: u32) -> Result<Vec<Detection>> {
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        
        // Get the first output tensor (most detection models have one primary output)
        let output = &outputs[0];
        
        // Extract data from tensor
        let array_view = output.try_extract()
            .map_err(|e| ClassificationError::ModelInferenceError(format!("Failed to extract tensor: {}", e)))?;
        
        let data: Vec<f32> = array_view.view().iter().copied().collect();
        let shape: Vec<i64> = array_view.view().shape().iter().map(|&x| x as i64).collect();
        
        // Parse detections based on output format
        // This is a generic implementation - specific models may need customization
        self.parse_detection_data(&data, &shape, img_width, img_height)
    }

    /// Parse detection data based on tensor format
    /// Supports common formats like YOLO (x, y, w, h, confidence, class_scores...)
    fn parse_detection_data(&self, data: &[f32], shape: &[i64], img_width: u32, img_height: u32) -> Result<Vec<Detection>> {
        println!("🔍 Debug: Output tensor shape: {:?}", shape);
        println!("🔍 Debug: Output data length: {}", data.len());
        if shape.len() >= 2 {
            println!("🔍 Debug: Dimensions - [{}, {}]", shape[0], shape[1]);
            if shape.len() >= 3 {
                println!("🔍 Debug: 3D tensor - [{}, {}, {}]", shape[0], shape[1], shape[2]);
            }
        }
        
        let mut detections = Vec::new();
        
        // Handle different tensor shapes
        match shape.len() {
            2 => {
                // Format: [num_detections, attributes]
                let num_detections = shape[0] as usize;
                let num_attributes = shape[1] as usize;
                
                if num_attributes < 6 {
                    return Err(ClassificationError::ModelInferenceError(
                        "Insufficient attributes in detection output".to_string()
                    ));
                }
                
                for i in 0..num_detections {
                    let offset = i * num_attributes;
                    if offset + 5 < data.len() {
                        // Assuming format: [x_center, y_center, width, height, confidence, class_scores...]
                        let x_center = data[offset];
                        let y_center = data[offset + 1];
                        let width = data[offset + 2];
                        let height = data[offset + 3];
                        let confidence = data[offset + 4];
                        
                        // Find best class
                        let mut best_class_id = 0;
                        let mut best_class_score = 0.0;
                        
                        for class_idx in 5..num_attributes {
                            if offset + class_idx < data.len() {
                                let score = data[offset + class_idx];
                                if score > best_class_score {
                                    best_class_score = score;
                                    best_class_id = class_idx - 5;
                                }
                            }
                        }
                        
                        // Convert from normalized coordinates if needed
                        let bbox = if x_center <= 1.0 && y_center <= 1.0 && width <= 1.0 && height <= 1.0 {
                            // Normalized coordinates - convert to pixels
                            CoordinateConverter::yolo_to_xyxy(
                                x_center, y_center, width, height,
                                img_width, img_height
                            )
                        } else {
                            // Already in pixel coordinates
                            BoundingBox::new(
                                (x_center - width / 2.0).max(0.0),
                                (y_center - height / 2.0).max(0.0),
                                (x_center + width / 2.0).min(img_width as f32),
                                (y_center + height / 2.0).min(img_height as f32),
                            )
                        };
                        
                        detections.push(Detection {
                            bbox,
                            confidence: confidence * best_class_score, // Combined confidence
                            class_id: best_class_id,
                            class_name: None, // Will be filled by the user if they provide class names
                            track_id: None,
                        });
                    }
                }
            }
            3 => {
                // Format: [batch_size, num_detections, attributes] - YOLOv3/v5/v8 format
                if shape[0] != 1 {
                    return Err(ClassificationError::ModelInferenceError(
                        "Batch size > 1 not supported".to_string()
                    ));
                }
                
                let num_detections = shape[1] as usize;
                let num_attributes = shape[2] as usize;
                
                if num_attributes == 4 {
                    // Special case: [x, y, w, h] format without confidence/class scores
                    // This might be a coordinate-only format
                    println!("🔍 Debug: Detected 4-attribute format (coordinates only)");
                    
                    let mut valid_count = 0;
                    let mut sample_count = 0;
                    
                    for i in 0..num_detections {
                        let offset = i * num_attributes;
                        if offset + 4 <= data.len() {
                            let x_center = data[offset];
                            let y_center = data[offset + 1];
                            let width = data[offset + 2];
                            let height = data[offset + 3];
                            
                            // Debug first few samples
                            if sample_count < 5 {
                                println!("🔍 Debug sample {}: x={:.3}, y={:.3}, w={:.3}, h={:.3}", 
                                    sample_count, x_center, y_center, width, height);
                                sample_count += 1;
                            }
                            
                            // Check if coordinates seem valid (not all zeros, within reasonable range)
                            if width > 0.0 && height > 0.0 && x_center > 0.0 && y_center > 0.0 {
                                valid_count += 1;
                                
                                // These coordinates appear to be in model input space (416x416), not normalized
                                // Convert to normalized coordinates first, then to pixel coordinates
                                let model_input_size = 416.0; // YOLOv3 standard input size
                                let norm_x = x_center / model_input_size;
                                let norm_y = y_center / model_input_size;
                                let norm_w = width / model_input_size;
                                let norm_h = height / model_input_size;
                                
                                let bbox = CoordinateConverter::yolo_to_xyxy(
                                    norm_x, norm_y, norm_w, norm_h,
                                    img_width, img_height
                                );
                                
                                // Use a default confidence since it's not provided
                                detections.push(Detection {
                                    bbox,
                                    confidence: 0.5, // Default confidence for coordinate-only format
                                    class_id: 0,     // Default class
                                    class_name: None,
                                    track_id: None,
                                });
                            }
                        }
                    }
                    
                    println!("🔍 Debug: Found {} valid detections out of {} total", valid_count, num_detections);
                } else if num_attributes >= 6 {
                
                // Process YOLOv3+ format directly (skip batch dimension)
                for i in 0..num_detections {
                    let offset = i * num_attributes;
                    if offset + 5 < data.len() {
                        // YOLOv3+ format: [x_center, y_center, width, height, confidence, class_scores...]
                        let x_center = data[offset];
                        let y_center = data[offset + 1];
                        let width = data[offset + 2];
                        let height = data[offset + 3];
                        let confidence = data[offset + 4];
                        
                        // Find best class from class scores
                        let mut best_class_id = 0;
                        let mut best_class_score = 0.0;
                        
                        for class_idx in 5..num_attributes {
                            if offset + class_idx < data.len() {
                                let score = data[offset + class_idx];
                                if score > best_class_score {
                                    best_class_score = score;
                                    best_class_id = class_idx - 5;
                                }
                            }
                        }
                        
                        // YOLOv3+ uses normalized coordinates (0-1)
                        let bbox = CoordinateConverter::yolo_to_xyxy(
                            x_center, y_center, width, height,
                            img_width, img_height
                        );
                        
                        // Use combined confidence (objectness * class probability)
                        let final_confidence = confidence * best_class_score;
                        
                        if final_confidence > 0.01 { // Basic confidence filter
                            detections.push(Detection {
                                bbox,
                                confidence: final_confidence,
                                class_id: best_class_id,
                                class_name: None,
                                track_id: None,
                            });
                        }
                    }
                }
                } else {
                    return Err(ClassificationError::ModelInferenceError(
                        format!("Unsupported number of attributes: {}", num_attributes)
                    ));
                }
            }
            4 => {
                // Format: [batch_size, channels, grid_height, grid_width] - YOLOv2/v3 format
                if shape[0] != 1 {
                    return Err(ClassificationError::ModelInferenceError(
                        "Batch size > 1 not supported".to_string()
                    ));
                }
                
                let channels = shape[1] as usize;
                let grid_h = shape[2] as usize;
                let grid_w = shape[3] as usize;
                
                // YOLOv2 format: channels = num_anchors * (5 + num_classes)
                // Assuming 5 anchors and 80 classes: 5 * (5 + 80) = 425
                let num_classes = 80; // COCO classes
                let attrs_per_anchor = 5 + num_classes; // x, y, w, h, confidence + classes
                let num_anchors = channels / attrs_per_anchor;
                
                if num_anchors == 0 || channels % attrs_per_anchor != 0 {
                    return Err(ClassificationError::ModelInferenceError(
                        format!("Invalid YOLOv2 output format: channels={}, expected multiple of {}", channels, attrs_per_anchor)
                    ));
                }
                
                // Parse YOLOv2 grid format
                for grid_y in 0..grid_h {
                    for grid_x in 0..grid_w {
                        for anchor in 0..num_anchors {
                            let base_idx = ((anchor * attrs_per_anchor * grid_h + grid_y) * grid_w + grid_x) as usize;
                            
                            if base_idx + attrs_per_anchor <= data.len() {
                                // Extract box parameters (relative to grid cell)
                                let x_offset = data[base_idx];
                                let y_offset = data[base_idx + 1];
                                let w = data[base_idx + 2];
                                let h = data[base_idx + 3];
                                let confidence = data[base_idx + 4];
                                
                                if confidence > 0.01 { // Basic confidence filter
                                    // Convert to absolute coordinates
                                    let x_center = (grid_x as f32 + x_offset) / grid_w as f32 * img_width as f32;
                                    let y_center = (grid_y as f32 + y_offset) / grid_h as f32 * img_height as f32;
                                    let width = w * img_width as f32;
                                    let height = h * img_height as f32;
                                    
                                    // Find best class
                                    let mut best_class_id = 0;
                                    let mut best_class_score = 0.0;
                                    
                                    for class_idx in 0..num_classes {
                                        let score = data[base_idx + 5 + class_idx];
                                        if score > best_class_score {
                                            best_class_score = score;
                                            best_class_id = class_idx;
                                        }
                                    }
                                    
                                    // Create bounding box
                                    let bbox = BoundingBox::new(
                                        (x_center - width / 2.0).max(0.0),
                                        (y_center - height / 2.0).max(0.0),
                                        (x_center + width / 2.0).min(img_width as f32),
                                        (y_center + height / 2.0).min(img_height as f32),
                                    );
                                    
                                    detections.push(Detection {
                                        bbox,
                                        confidence: confidence * best_class_score,
                                        class_id: best_class_id,
                                        class_name: None,
                                        track_id: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(ClassificationError::ModelInferenceError(
                    format!("Unsupported output tensor shape: {:?}", shape)
                ));
            }
        }
        
        Ok(detections)
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
            ModelType::ObjectDetection { .. } => {
                if frames.len() != 1 {
                    return Err(ClassificationError::InvalidInput(
                        "Object detection model expects exactly one frame".to_string()
                    ));
                }
                self.classify_single_frame(frames[0])
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