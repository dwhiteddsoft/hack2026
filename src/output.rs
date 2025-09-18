use ndarray::ArrayD;
use crate::core::{Detection, BoundingBox, Mask, Keypoint, InferenceMetadata};

/// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Universal output processor for ONNX computer vision models
pub struct OutputProcessor {
    pub specification: OutputSpecification,
    pub postprocessing_config: PostProcessingConfig,
}

/// Canonical output specification
#[derive(Debug, Clone)]
pub struct OutputSpecification {
    pub architecture_type: ArchitectureType,
    pub tensor_outputs: Vec<TensorOutput>,
    pub coordinate_system: CoordinateSystem,
    pub activation_requirements: ActivationRequirements,
    pub loaded_config: Option<crate::session::YamlPostprocessingConfig>,
}

/// Architecture type classification
#[derive(Debug, Clone)]
pub enum ArchitectureType {
    SingleStage {
        unified_head: bool,
        anchor_based: bool,
    },
    TwoStage {
        rpn_outputs: Vec<TensorOutput>,
        rcnn_outputs: Vec<TensorOutput>,
        additional_tasks: Vec<TaskType>,
    },
    MultiScale {
        scale_strategy: ScaleStrategy,
        shared_head: bool,
    },
}

/// Task types
#[derive(Debug, Clone)]
pub enum TaskType {
    Detection,
    Segmentation,
    Classification,
    PoseEstimation,
    DepthEstimation,
}

/// Scale strategy for multi-scale architectures
#[derive(Debug, Clone)]
pub enum ScaleStrategy {
    FeaturePyramidUnified,
    YoloMultiScale,
    SsdMultiBox,
    FeaturePyramid,
}

/// Tensor output specification
#[derive(Debug, Clone)]
pub struct TensorOutput {
    pub name: String,
    pub shape: TensorShape,
    pub content_type: ContentType,
    pub spatial_layout: SpatialLayout,
    pub channel_interpretation: ChannelInterpretation,
}

/// Tensor shape specification
#[derive(Debug, Clone)]
pub struct TensorShape {
    pub dimensions: Vec<OutputDimension>,
    pub layout_format: LayoutFormat,
}

/// Output dimension types
#[derive(Debug, Clone)]
pub enum OutputDimension {
    Batch(i64),
    Classes(i64),
    Coordinates(i64),
    Anchors(i64),
    Height(i64),
    Width(i64),
    Features(i64),
    Combined(CombinedDimension),
}

/// Combined dimension specification
#[derive(Debug, Clone)]
pub struct CombinedDimension {
    pub total_size: i64,
    pub components: Vec<DimensionComponent>,
}

/// Dimension component
#[derive(Debug, Clone)]
pub struct DimensionComponent {
    pub component_type: ComponentType,
    pub size: i64,
    pub offset: i64,
}

/// Component types
#[derive(Debug, Clone)]
pub enum ComponentType {
    BoundingBox,
    Confidence,
    ClassLogits,
    Mask,
    Keypoints,
}

/// Layout format options
#[derive(Debug, Clone)]
pub enum LayoutFormat {
    NCHW,
    NHWC,
    NCL,  // N, Channels, Length (for flattened outputs)
}

/// Content type classification
#[derive(Debug, Clone)]
pub enum ContentType {
    Classification {
        num_classes: i64,
        background_class: bool,
        multi_label: bool,
    },
    Regression {
        coordinate_format: CoordinateFormat,
        normalization: CoordinateNormalization,
    },
    Objectness {
        confidence_type: ConfidenceType,
    },
    Segmentation {
        mask_format: MaskFormat,
        resolution: (i64, i64),
    },
    Combined {
        components: Vec<ContentType>,
    },
}

/// Coordinate format options
#[derive(Debug, Clone)]
pub enum CoordinateFormat {
    CenterWidthHeight,
    CornerCoordinates,
    Offsets,
}

/// Coordinate normalization
#[derive(Debug, Clone)]
pub enum CoordinateNormalization {
    Normalized,
    Pixel,
    GridRelative,
    AnchorRelative,
}

/// Confidence type
#[derive(Debug, Clone)]
pub enum ConfidenceType {
    Objectness,
    ClassConfidence,
    Combined,
}

/// Mask format options
#[derive(Debug, Clone)]
pub enum MaskFormat {
    Binary,
    Grayscale,
    Polygon,
    Rle,
}

/// Spatial layout patterns
#[derive(Debug, Clone)]
pub enum SpatialLayout {
    Grid {
        grid_size: (i64, i64),
        stride: i64,
        predictions_per_cell: i64,
    },
    Dense {
        feature_map_size: (i64, i64),
        anchor_count: i64,
    },
    Proposals {
        max_proposals: i64,
        nms_applied: bool,
    },
    Unified {
        total_predictions: i64,
        multi_scale: bool,
    },
}

/// Channel interpretation
#[derive(Debug, Clone)]
pub enum ChannelInterpretation {
    Interleaved {
        pattern: Vec<ComponentType>,
        repetitions: i64,
    },
    Separated {
        layout: SeparatedLayout,
    },
    Unified {
        components: Vec<ComponentRange>,
    },
}

/// Component range specification
#[derive(Debug, Clone)]
pub struct ComponentRange {
    pub component_type: ComponentType,
    pub start_channel: i64,
    pub end_channel: i64,
}

/// Separated layout options
#[derive(Debug, Clone)]
pub enum SeparatedLayout {
    ClassificationRegression,
    MultiTask,
    FeaturePyramid,
}

/// Coordinate system
#[derive(Debug, Clone)]
pub enum CoordinateSystem {
    Direct,
    Relative,
    AnchorBased,
}

/// Activation requirements
#[derive(Debug, Clone)]
pub struct ActivationRequirements {
    pub bbox_activation: ActivationType,
    pub class_activation: ActivationType,
    pub confidence_activation: ActivationType,
}

/// Activation types
#[derive(Debug, Clone)]
pub enum ActivationType {
    None,
    Sigmoid,
    Softmax,
    Tanh,
}

/// Post-processing configuration
#[derive(Debug, Clone)]
pub struct PostProcessingConfig {
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub max_detections: usize,
    pub per_class_nms: bool,
    pub multi_label: bool,
}

impl PostProcessingConfig {
    /// Create preprocessing config from output specification
    pub fn from_spec(spec: &OutputSpecification) -> Self {
        // Use loaded YAML config values if available, otherwise fall back to defaults
        if let Some(ref loaded_config) = spec.loaded_config {
            let confidence_threshold = loaded_config.confidence_threshold.unwrap_or(0.5);
            Self {
                confidence_threshold,
                nms_threshold: loaded_config.nms_threshold.unwrap_or(0.45),
                max_detections: loaded_config.max_detections.unwrap_or(300),
                per_class_nms: !loaded_config.class_agnostic_nms.unwrap_or(false), // Invert class_agnostic_nms
                multi_label: false, // Not typically in YAML configs, keep default
            }
        } else {
            // Fallback to hardcoded defaults when no config is loaded
            Self {
                confidence_threshold: 0.5,
                nms_threshold: 0.45,
                max_detections: 300,
                per_class_nms: true,
                multi_label: false,
            }
        }
    }
}

/// Post-processing pipeline
#[derive(Debug, Clone)]
pub struct PostProcessingPipeline {
    pub steps: Vec<PostProcessingStep>,
}

/// Post-processing steps
#[derive(Debug, Clone)]
pub enum PostProcessingStep {
    CoordinateDecoding {
        decode_strategy: DecodeStrategy,
    },
    ClassActivation {
        activation: ActivationType,
    },
    ConfidenceFiltering {
        threshold: f32,
        per_class: bool,
    },
    NonMaximumSuppression {
        iou_threshold: f32,
        strategy: NMSStrategy,
    },
    ScaleAdjustment {
        original_size: (u32, u32),
        input_size: (u32, u32),
    },
}

/// Decode strategy options
#[derive(Debug, Clone)]
pub enum DecodeStrategy {
    DirectRegression,
    AnchorBased,
    GridBased,
}

/// NMS strategy options
#[derive(Debug, Clone)]
pub enum NMSStrategy {
    Standard,
    Soft,
    DIoU,
    CIoU,
}

impl OutputProcessor {
    /// Create a new output processor with specification
    pub fn new(specification: OutputSpecification) -> Self {
        let postprocessing_config = PostProcessingConfig::from_spec(&specification);
        Self {
            specification,
            postprocessing_config,
        }
    }

    /// Create an output processor from specification
    pub fn from_spec(spec: &OutputSpecification) -> Self {
        Self {
            specification: spec.clone(),
            postprocessing_config: PostProcessingConfig::from_spec(spec),
        }
    }

    /// Process model outputs to extract detections
    pub fn process_outputs(
        &self,
        outputs: &[ArrayD<f32>],
        input_shape: (u32, u32),
        original_shape: (u32, u32),
        model_name: &str,
    ) -> crate::error::Result<Vec<Detection>> {
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        
        // Check tensor format and route to appropriate processor
        let output = &outputs[0];
        let shape = output.shape();
        
        // For YOLOv3, check if we have multiple outputs (multi-scale detection)
        if model_name.contains("yolov3") {
            self.process_yolov3_output(outputs, input_shape, original_shape)
        } else {
            match shape.len() {
                2 => {
                    // Classification format: [batch, classes] e.g., [1, 1000] for MobileNetV2
                    self.process_classification_output(outputs, input_shape, original_shape)
                },
                3 => {
                    // YOLOv8 format: [1, 84, 8400]
                    self.process_yolov8_output(outputs, input_shape, original_shape)
                },
                4 => {
                    // YOLOv2 format: [1, 425, 13, 13]
                    self.process_yolov2_output(outputs, input_shape, original_shape)
                },
                _ => {
                    Err(crate::error::UocvrError::ModelConfig {
                        message: format!("Unsupported output tensor dimension: {}D", shape.len())
                    })
                }
            }
        }
    }

    /// Process YOLOv8-style 3D output tensors
    fn process_yolov8_output(
        &self,
        outputs: &[ndarray::ArrayD<f32>],
        input_shape: (u32, u32),
        original_shape: (u32, u32),
    ) -> crate::error::Result<Vec<Detection>> {
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        
        // YOLOv8 format: [1, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
        let output = &outputs[0];
        let shape = output.shape();
        
        if shape.len() != 3 {
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!("Expected 3D output tensor for YOLOv8, got {}D", shape.len())
            });
        }
        
        let [batch_size, num_features, num_predictions] = [shape[0], shape[1], shape[2]];
        
        if batch_size != 1 {
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!("Expected batch size 1, got {}", batch_size)
            });
        }
        
        if num_features < 84 {
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!("Expected at least 84 features (4 bbox + 80 classes), got {}", num_features)
            });
        }
        
        let mut raw_detections = Vec::new();
        
        // Parse each prediction
        for pred_idx in 0..num_predictions {
            // Extract bbox coordinates (first 4 values)
            let bbox = [
                output[[0, 0, pred_idx]],
                output[[0, 1, pred_idx]],
                output[[0, 2, pred_idx]],
                output[[0, 3, pred_idx]],
            ];
            
            // Extract class logits (remaining values)
            let mut class_logits = Vec::with_capacity((num_features - 4) as usize);
            for class_idx in 4..num_features {
                class_logits.push(output[[0, class_idx, pred_idx]]);
            }
            
            // Apply sigmoid to class logits to get confidence scores
            let mut class_scores = class_logits.clone();
            self.apply_activation(&mut class_scores, &ActivationType::Sigmoid);
            
            // Find best class and confidence
            let (best_class_idx, &max_confidence) = class_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            
            // Only keep detections above confidence threshold
            if max_confidence >= self.postprocessing_config.confidence_threshold {
                // Convert center_x, center_y, width, height to absolute coordinates
                let center_x = bbox[0];
                let center_y = bbox[1];
                let width = bbox[2];
                let height = bbox[3];
                
                // Convert to corner coordinates for NMS
                let x1 = center_x - width / 2.0;
                let y1 = center_y - height / 2.0;
                let x2 = center_x + width / 2.0;
                let y2 = center_y + height / 2.0;
                
                raw_detections.push(RawDetection {
                    bbox: [x1, y1, x2, y2],
                    confidence: max_confidence,
                    class_logits: vec![best_class_idx as f32], // Store class index
                    grid_position: (0, 0), // Not used in YOLOv8
                    stride: 1.0, // Not used in YOLOv8
                });
            }
        }
        
        // Apply NMS
        let filtered_detections = self.apply_nms(raw_detections);
        
        // Convert to final Detection format and scale to original image size
        let mut final_detections = Vec::new();
        for raw_det in filtered_detections {
            let class_id = raw_det.class_logits[0] as u32;
            
            // Create BoundingBox in original image coordinates
            let bbox = self.scale_bbox_to_original(
                raw_det.bbox,
                input_shape,
                original_shape,
            );
            
            final_detections.push(Detection {
                bbox,
                confidence: raw_det.confidence,
                class_id,
                class_name: Some(format!("class_{}", class_id)), // Default name
                mask: None,
                keypoints: None,
            });
        }
        
        // Limit to max detections
        if final_detections.len() > self.postprocessing_config.max_detections {
            final_detections.truncate(self.postprocessing_config.max_detections);
        }
        
        Ok(final_detections)
    }

    /// Apply post-processing pipeline
    pub fn apply_postprocessing(
        &self,
        raw_detections: Vec<RawDetection>,
        input_shape: (u32, u32),
        original_shape: (u32, u32),
    ) -> crate::error::Result<Vec<Detection>> {
        // Step 1: Filter by confidence threshold
        let confident_detections = self.filter_by_confidence(
            raw_detections,
            self.postprocessing_config.confidence_threshold,
        );
        
        // Step 2: Apply NMS
        let nms_detections = self.apply_nms(confident_detections);
        
        // Step 3: Convert to Detection format and scale coordinates
        let mut final_detections = Vec::new();
        for raw_det in nms_detections {
            let class_id = if !raw_det.class_logits.is_empty() {
                raw_det.class_logits[0] as u32
            } else {
                0
            };
            
            // Scale bbox from input size to original size
            let bbox = self.scale_bbox_to_original(
                raw_det.bbox,
                input_shape,
                original_shape,
            );
            
            final_detections.push(Detection {
                bbox,
                confidence: raw_det.confidence,
                class_id,
                class_name: Some(format!("class_{}", class_id)),
                mask: None,
                keypoints: None,
            });
        }
        
        // Step 4: Limit to max detections
        if final_detections.len() > self.postprocessing_config.max_detections {
            final_detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
            final_detections.truncate(self.postprocessing_config.max_detections);
        }
        
        Ok(final_detections)
    }

    /// Process YOLOv2-style 4D output tensors
    fn process_yolov2_output(
        &self,
        outputs: &[ndarray::ArrayD<f32>],
        input_shape: (u32, u32),
        original_shape: (u32, u32),
    ) -> crate::error::Result<Vec<Detection>> {
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        
        // YOLOv2 format: [1, 425, 13, 13] where 425 = 5 anchors × 85 each
        let output = &outputs[0];
        let shape = output.shape();
        
        if shape.len() != 4 {
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!("Expected 4D output tensor for YOLOv2, got {}D", shape.len())
            });
        }
        
        let [batch_size, channels, grid_h, grid_w] = [shape[0], shape[1], shape[2], shape[3]];
        
        if batch_size != 1 {
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!("Expected batch size 1, got {}", batch_size)
            });
        }
        
        if channels != 425 {
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!("Expected 425 channels for YOLOv2, got {}", channels)
            });
        }
        
        if grid_h != 13 || grid_w != 13 {
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!("Expected 13x13 grid for YOLOv2, got {}x{}", grid_h, grid_w)
            });
        }
        
        // YOLOv2 anchors (width, height) in pixels for 416x416 input
        let anchors = [
            (23.83, 28.27),   // 0.57273 * 416, 0.677385 * 416
            (77.97, 85.80),   // 1.87446 * 416, 2.06253 * 416  
            (138.88, 227.69), // 3.33843 * 416, 5.47434 * 416
            (327.92, 146.76), // 7.88282 * 416, 3.52778 * 416
            (406.53, 381.46), // 9.77052 * 416, 9.16828 * 416
        ];
        
        let mut raw_detections = Vec::new();
        
        // Process each grid cell and anchor
        for i in 0..grid_h {
            for j in 0..grid_w {
                for anchor_idx in 0..5 {
                    let channel_offset = anchor_idx * 85;
                    
                    // Extract raw predictions for this anchor
                    let tx = output[[0, channel_offset + 0, i, j]];
                    let ty = output[[0, channel_offset + 1, i, j]];
                    let tw = output[[0, channel_offset + 2, i, j]];
                    let th = output[[0, channel_offset + 3, i, j]];
                    let tc = output[[0, channel_offset + 4, i, j]];
                    
                    // Apply YOLOv2 coordinate transformations
                    let x = (postprocessing::sigmoid(tx) + j as f32) / grid_w as f32;
                    let y = (postprocessing::sigmoid(ty) + i as f32) / grid_h as f32;
                    let w = anchors[anchor_idx].0 * tw.exp() / input_shape.0 as f32;
                    let h = anchors[anchor_idx].1 * th.exp() / input_shape.1 as f32;
                    let confidence = postprocessing::sigmoid(tc);
                    
                    // Skip low-confidence detections early
                    if confidence < self.postprocessing_config.confidence_threshold {
                        continue;
                    }
                    
                    // Extract class probabilities
                    let mut class_probs = Vec::with_capacity(80);
                    for class_idx in 0..80 {
                        class_probs.push(output[[0, channel_offset + 5 + class_idx, i, j]]);
                    }
                    
                    // Apply softmax to class probabilities
                    postprocessing::softmax(&mut class_probs);
                    
                    // Find best class
                    let (best_class_idx, &max_class_prob) = class_probs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or((0, &0.0));
                    
                    // Calculate final confidence
                    let final_confidence = confidence * max_class_prob;
                    
                    // Skip if final confidence is too low
                    if final_confidence < self.postprocessing_config.confidence_threshold {
                        continue;
                    }
                    
                    // Convert to absolute coordinates
                    let center_x = x * input_shape.0 as f32;
                    let center_y = y * input_shape.1 as f32;
                    let box_width = w * input_shape.0 as f32;
                    let box_height = h * input_shape.1 as f32;
                    
                    // Convert to corner coordinates
                    let x1 = center_x - box_width / 2.0;
                    let y1 = center_y - box_height / 2.0;
                    let x2 = center_x + box_width / 2.0;
                    let y2 = center_y + box_height / 2.0;
                    
                    raw_detections.push(RawDetection {
                        bbox: [x1, y1, x2, y2],
                        confidence: final_confidence,
                        class_logits: vec![best_class_idx as f32],
                        grid_position: (i, j),
                        stride: 32.0, // YOLOv2 stride
                    });
                }
            }
        }
        
        // Apply NMS
        let filtered_detections = self.apply_nms(raw_detections);
        
        // Convert to final Detection format and scale to original image size
        let mut final_detections = Vec::new();
        for raw_det in filtered_detections {
            let class_id = raw_det.class_logits[0] as u32;
            
            // Create BoundingBox in original image coordinates
            let bbox = self.scale_bbox_to_original(
                raw_det.bbox,
                input_shape,
                original_shape,
            );
            
            final_detections.push(Detection {
                bbox,
                confidence: raw_det.confidence,
                class_id,
                class_name: Some(format!("class_{}", class_id)),
                mask: None,
                keypoints: None,
            });
        }
        
        // Limit to max detections
        if final_detections.len() > self.postprocessing_config.max_detections {
            final_detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
            final_detections.truncate(self.postprocessing_config.max_detections);
        }
        
        Ok(final_detections)
    }

    /// Process YOLOv3-style multi-scale output tensors
    fn process_yolov3_output(
        &self,
        outputs: &[ArrayD<f32>],
        input_shape: (u32, u32),
        original_shape: (u32, u32),
    ) -> crate::error::Result<Vec<Detection>> {
        //println!("  YOLOv3 processing: {} output tensors", outputs.len());
        
        // for (scale_idx, output) in outputs.iter().enumerate() {
        //     let shape = output.shape();
        //     println!("  Scale {}: tensor shape {:?}", scale_idx, shape);
        // }
        
        // This YOLOv3-10 model uses a different format:
        // - outputs[0]: [1, 10647, 4] - bounding boxes (x1, y1, x2, y2)
        // - outputs[1]: [1, 80, 10647] - class probabilities  
        // - outputs[2]: [2, 3] - metadata/other
        
        if outputs.len() < 2 {
            return Ok(Vec::new());
        }
        
        let bbox_output = &outputs[0];  // [1, 10647, 4]
        let class_output = &outputs[1]; // [1, 80, 10647]
        
        let bbox_shape = bbox_output.shape();
        let class_shape = class_output.shape();
        
        // Validate shapes
        if bbox_shape.len() != 3 || class_shape.len() != 3 {
            println!("  Error: Unexpected tensor dimensions");
            return Ok(Vec::new());
        }
        
        if bbox_shape[2] != 4 {
            println!("  Error: Expected 4 bbox coordinates, got {}", bbox_shape[2]);
            return Ok(Vec::new());
        }
        
        let num_predictions = bbox_shape[1];
        let num_classes = class_shape[1];
        
        if class_shape[2] != num_predictions {
            println!("  Error: Bbox and class tensor size mismatch");
            return Ok(Vec::new());
        }
        
        //println!("  Processing {} predictions with {} classes", num_predictions, num_classes);
        
        let mut all_detections = Vec::new();
        
        // Process each prediction
        for pred_idx in 0..num_predictions {
            // Extract bounding box coordinates (assuming they're already in x1,y1,x2,y2 format)
            let x1 = bbox_output[[0, pred_idx, 0]];
            let y1 = bbox_output[[0, pred_idx, 1]];
            let x2 = bbox_output[[0, pred_idx, 2]];
            let y2 = bbox_output[[0, pred_idx, 3]];
            
            // Find best class and its probability
            let mut best_class_idx = 0;
            let mut best_class_prob = 0.0;
            
            for class_idx in 0..num_classes {
                let class_prob = class_output[[0, class_idx, pred_idx]];
                if class_prob > best_class_prob {
                    best_class_prob = class_prob;
                    best_class_idx = class_idx;
                }
            }
            
            // Apply sigmoid to class probability if needed
            let final_confidence = sigmoid(best_class_prob);
            
            // Skip if confidence is too low
            if final_confidence < self.postprocessing_config.confidence_threshold {
                continue;
            }
            
            // Skip invalid boxes
            if x1 >= x2 || y1 >= y2 {
                continue;
            }
            
            // if all_detections.len() < 10 { // Show first few detections for debugging
            //     println!("    Detection {}: class={}, conf={:.3}, bbox=({:.1},{:.1},{:.1},{:.1})", 
            //         pred_idx, best_class_idx, final_confidence, x1, y1, x2, y2);
            // }
            
            all_detections.push(RawDetection {
                bbox: [x1, y1, x2, y2],
                confidence: final_confidence,
                class_logits: vec![best_class_idx as f32],
                grid_position: (0, 0), // Not applicable for this format
                stride: 1.0,           // Not applicable for this format
            });
        }
        
        //println!("  Found {} raw detections", all_detections.len());
        
        // Apply NMS
        let filtered_detections = if self.postprocessing_config.nms_threshold > 0.0 {
            self.apply_nms(all_detections)
        } else {
            all_detections
        };
        
        // Convert to final detection format
        let mut final_detections = Vec::new();
        for raw_det in filtered_detections {
            let class_id = raw_det.class_logits[0] as u32;
            
            // Create BoundingBox in original image coordinates
            let bbox = self.scale_bbox_to_original(
                raw_det.bbox,
                input_shape,
                original_shape,
            );
            
            final_detections.push(Detection {
                bbox,
                confidence: raw_det.confidence,
                class_id,
                class_name: Some(format!("class_{}", class_id)),
                mask: None,
                keypoints: None,
            });
        }
        
        // Limit to max detections
        if final_detections.len() > self.postprocessing_config.max_detections {
            final_detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
            final_detections.truncate(self.postprocessing_config.max_detections);
        }
        
        Ok(final_detections)
    }

    /// Process 2D classification output tensors (e.g., MobileNetV2)
    fn process_classification_output(
        &self,
        outputs: &[ArrayD<f32>],
        _input_shape: (u32, u32),
        _original_shape: (u32, u32),
    ) -> crate::error::Result<Vec<Detection>> {
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        
        // Classification format: [batch, classes] e.g., [1, 1000] for MobileNetV2
        let output = &outputs[0];
        let shape = output.shape();
        
        if shape.len() != 2 {
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!("Expected 2D output tensor for classification, got {}D", shape.len())
            });
        }
        
        let [batch_size, num_classes] = [shape[0], shape[1]];
        
        if batch_size != 1 {
            return Err(crate::error::UocvrError::ModelConfig {
                message: format!("Expected batch size 1, got {}", batch_size)
            });
        }
        
        // Extract class logits and apply softmax
        let mut class_probs = Vec::with_capacity(num_classes);
        let mut max_logit = f32::NEG_INFINITY;
        
        // Find max logit for numerical stability in softmax
        for class_idx in 0..num_classes {
            let logit = output[[0, class_idx]];
            max_logit = max_logit.max(logit);
        }
        
        // Compute softmax probabilities
        let mut sum_exp = 0.0;
        for class_idx in 0..num_classes {
            let logit = output[[0, class_idx]];
            let prob = (logit - max_logit).exp();
            class_probs.push(prob);
            sum_exp += prob;
        }
        
        // Normalize probabilities
        for prob in &mut class_probs {
            *prob /= sum_exp;
        }
        
        // Create "detections" for top-k class predictions
        // For classification, we create pseudo-detections with full-image bounding boxes
        let mut detections = Vec::new();
        
        // Get top predictions above confidence threshold
        let mut class_indices_probs: Vec<(usize, f32)> = class_probs
            .iter()
            .enumerate()
            .map(|(idx, &prob)| (idx, prob))
            .filter(|(_, prob)| *prob >= self.postprocessing_config.confidence_threshold)
            .collect();
        
        // Sort by probability descending
        class_indices_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top max_detections
        let max_detections = self.postprocessing_config.max_detections;
        for (class_idx, confidence) in class_indices_probs.into_iter().take(max_detections) {
            // Create a full-image bounding box for classification results
            let bbox = BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 1.0,  // Normalized coordinates: full image
                height: 1.0,
                format: crate::core::BoundingBoxFormat::TopLeftWidthHeight,
            };
            
            detections.push(Detection {
                bbox,
                confidence,
                class_id: class_idx as u32,
                class_name: Some(format!("class_{}", class_idx)),
                mask: None,
                keypoints: None,
            });
        }
        
        Ok(detections)
    }

    /// Decode coordinates from raw model output
    fn decode_coordinates(
        &self,
        raw_coords: &[f32],
        grid_position: (usize, usize),
        stride: f32,
    ) -> crate::error::Result<BoundingBox> {
        if raw_coords.len() < 4 {
            return Err(crate::error::UocvrError::ModelConfig {
                message: "Insufficient coordinate data".to_string()
            });
        }
        
        // For YOLO format: [center_x, center_y, width, height]
        let (grid_x, grid_y) = grid_position;
        
        // Decode center coordinates relative to grid cell
        let center_x = (raw_coords[0] + grid_x as f32) * stride;
        let center_y = (raw_coords[1] + grid_y as f32) * stride;
        let width = raw_coords[2] * stride;
        let height = raw_coords[3] * stride;
        
        // Convert to corner coordinates and create BoundingBox
        let x1 = center_x - width / 2.0;
        let y1 = center_y - height / 2.0;
        
        Ok(BoundingBox {
            x: x1,
            y: y1,
            width,
            height,
            format: crate::core::BoundingBoxFormat::TopLeftWidthHeight,
        })
    }

    /// Apply activation function to raw outputs
    fn apply_activation(&self, values: &mut [f32], activation: &ActivationType) {
        match activation {
            ActivationType::Sigmoid => {
                for val in values.iter_mut() {
                    *val = postprocessing::sigmoid(*val);
                }
            },
            ActivationType::Softmax => {
                postprocessing::softmax(values);
            },
            ActivationType::Tanh => {
                for val in values.iter_mut() {
                    *val = val.tanh();
                }
            },
            ActivationType::None => {
                // No activation applied
            }
        }
    }

    /// Filter detections by confidence threshold
    fn filter_by_confidence(
        &self,
        detections: Vec<RawDetection>,
        threshold: f32,
    ) -> Vec<RawDetection> {
        detections
            .into_iter()
            .filter(|det| det.confidence >= threshold)
            .collect()
    }

    /// Apply Non-Maximum Suppression
    fn apply_nms(&self, detections: Vec<RawDetection>) -> Vec<RawDetection> {
        match self.postprocessing_config.nms_threshold {
            threshold if threshold > 0.0 => {
                postprocessing::standard_nms(detections, threshold)
            },
            _ => detections, // No NMS if threshold is 0 or negative
        }
    }

    /// Scale detections from input size to original image size
    fn scale_detections(
        &self,
        detections: Vec<Detection>,
        input_size: (u32, u32),
        original_size: (u32, u32),
    ) -> Vec<Detection> {
        let x_scale = original_size.0 as f32 / input_size.0 as f32;
        let y_scale = original_size.1 as f32 / input_size.1 as f32;
        
        detections
            .into_iter()
            .map(|mut detection| {
                // Scale bounding box coordinates
                detection.bbox.x *= x_scale;
                detection.bbox.y *= y_scale;
                detection.bbox.width *= x_scale;
                detection.bbox.height *= y_scale;
                
                detection
            })
            .collect()
    }
    
    /// Helper function to scale a single bbox from input to original coordinates
    fn scale_bbox_to_original(
        &self,
        bbox: [f32; 4], // [x1, y1, x2, y2]
        input_size: (u32, u32),
        original_size: (u32, u32),
    ) -> BoundingBox {
        let x_scale = original_size.0 as f32 / input_size.0 as f32;
        let y_scale = original_size.1 as f32 / input_size.1 as f32;
        
        let [x1, y1, x2, y2] = bbox;
        let scaled_x1 = x1 * x_scale;
        let scaled_y1 = y1 * y_scale;
        let scaled_x2 = x2 * x_scale;
        let scaled_y2 = y2 * y_scale;
        
        BoundingBox {
            x: scaled_x1,
            y: scaled_y1,
            width: scaled_x2 - scaled_x1,
            height: scaled_y2 - scaled_y1,
            format: crate::core::BoundingBoxFormat::TopLeftWidthHeight,
        }
    }
}

/// Raw detection before post-processing
#[derive(Debug, Clone)]
pub struct RawDetection {
    pub bbox: [f32; 4],
    pub confidence: f32,
    pub class_logits: Vec<f32>,
    pub grid_position: (usize, usize),
    pub stride: f32,
}

impl Default for PostProcessingConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,  // Higher default threshold
            nms_threshold: 0.45,
            max_detections: 300,
            per_class_nms: true,
            multi_label: false,
        }
    }
}

/// Helper functions for post-processing operations
pub mod postprocessing {
    use super::*;

    /// Calculate Intersection over Union (IoU)
    pub fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
        let [x1_min, y1_min, x1_max, y1_max] = *box1;
        let [x2_min, y2_min, x2_max, y2_max] = *box2;
        
        // Calculate intersection area
        let inter_x_min = x1_min.max(x2_min);
        let inter_y_min = y1_min.max(y2_min);
        let inter_x_max = x1_max.min(x2_max);
        let inter_y_max = y1_max.min(y2_max);
        
        // Check if there's any intersection
        if inter_x_min >= inter_x_max || inter_y_min >= inter_y_max {
            return 0.0;
        }
        
        let intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
        
        // Calculate union area
        let area1 = (x1_max - x1_min) * (y1_max - y1_min);
        let area2 = (x2_max - x2_min) * (y2_max - y2_min);
        let union = area1 + area2 - intersection;
        
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    /// Apply sigmoid activation
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Apply softmax activation
    pub fn softmax(values: &mut [f32]) {
        if values.is_empty() {
            return;
        }
        
        // Find max value for numerical stability
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp(x - max) and sum
        let mut sum = 0.0;
        for val in values.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }
        
        // Normalize by sum
        if sum > 0.0 {
            for val in values.iter_mut() {
                *val /= sum;
            }
        }
    }

    /// Convert center-width-height to corner coordinates
    pub fn cxcywh_to_xyxy(bbox: [f32; 4]) -> [f32; 4] {
        let [cx, cy, w, h] = bbox;
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    }

    /// Convert corner coordinates to center-width-height
    pub fn xyxy_to_cxcywh(bbox: [f32; 4]) -> [f32; 4] {
        let [x1, y1, x2, y2] = bbox;
        let w = x2 - x1;
        let h = y2 - y1;
        [x1 + w / 2.0, y1 + h / 2.0, w, h]
    }

    /// Standard Non-Maximum Suppression
    pub fn standard_nms(
        mut detections: Vec<RawDetection>,
        iou_threshold: f32,
    ) -> Vec<RawDetection> {
        if detections.is_empty() {
            return detections;
        }
        
        // Sort by confidence (highest first)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut keep = Vec::new();
        let mut suppressed = vec![false; detections.len()];
        
        for i in 0..detections.len() {
            if suppressed[i] {
                continue;
            }
            
            keep.push(detections[i].clone());
            
            // Suppress overlapping detections
            for j in (i + 1)..detections.len() {
                if suppressed[j] {
                    continue;
                }
                
                let iou = calculate_iou(&detections[i].bbox, &detections[j].bbox);
                if iou > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
        
        keep
    }

    /// Soft Non-Maximum Suppression
    pub fn soft_nms(
        mut detections: Vec<RawDetection>,
        iou_threshold: f32,
        sigma: f32,
    ) -> Vec<RawDetection> {
        if detections.is_empty() {
            return detections;
        }
        
        // Sort by confidence (highest first)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut keep = Vec::new();
        
        for i in 0..detections.len() {
            let max_detection = detections[i].clone();
            
            // Apply soft suppression
            for j in (i + 1)..detections.len() {
                let iou = calculate_iou(&max_detection.bbox, &detections[j].bbox);
                
                if iou > iou_threshold {
                    // Apply Gaussian weighting to reduce confidence
                    let weight = (-iou * iou / sigma).exp();
                    detections[j].confidence *= weight;
                }
            }
            
            keep.push(max_detection);
        }
        
        // Filter out detections with very low confidence after soft suppression
        keep.into_iter()
            .filter(|det| det.confidence > 0.001) // Minimum threshold
            .collect()
    }
}

impl Default for OutputSpecification {
    fn default() -> Self {
        Self {
            architecture_type: ArchitectureType::SingleStage {
                unified_head: true,
                anchor_based: false,
            },
            tensor_outputs: vec![TensorOutput {
                name: "output0".to_string(),
                shape: TensorShape {
                    dimensions: vec![
                        OutputDimension::Batch(1),
                        OutputDimension::Combined(CombinedDimension {
                            total_size: 84,
                            components: vec![
                                DimensionComponent {
                                    component_type: ComponentType::BoundingBox,
                                    size: 4,
                                    offset: 0,
                                },
                                DimensionComponent {
                                    component_type: ComponentType::Confidence,
                                    size: 1,
                                    offset: 4,
                                },
                                DimensionComponent {
                                    component_type: ComponentType::ClassLogits,
                                    size: 80,
                                    offset: 5,
                                },
                            ],
                        }),
                        OutputDimension::Anchors(8400),
                    ],
                    layout_format: LayoutFormat::NCL,
                },
                content_type: ContentType::Combined {
                    components: vec![
                        ContentType::Regression {
                            coordinate_format: CoordinateFormat::CenterWidthHeight,
                            normalization: CoordinateNormalization::Pixel,
                        },
                        ContentType::Objectness {
                            confidence_type: ConfidenceType::Objectness,
                        },
                        ContentType::Classification {
                            num_classes: 80,
                            background_class: false,
                            multi_label: false,
                        },
                    ],
                },
                spatial_layout: SpatialLayout::Unified {
                    total_predictions: 8400,
                    multi_scale: true,
                },
                channel_interpretation: ChannelInterpretation::Unified {
                    components: vec![
                        ComponentRange {
                            component_type: ComponentType::BoundingBox,
                            start_channel: 0,
                            end_channel: 4,
                        },
                        ComponentRange {
                            component_type: ComponentType::Confidence,
                            start_channel: 4,
                            end_channel: 5,
                        },
                        ComponentRange {
                            component_type: ComponentType::ClassLogits,
                            start_channel: 5,
                            end_channel: 85,
                        },
                    ],
                },
            }],
            coordinate_system: CoordinateSystem::Direct,
            activation_requirements: ActivationRequirements {
                bbox_activation: ActivationType::None,
                class_activation: ActivationType::Sigmoid,
                confidence_activation: ActivationType::Sigmoid,
            },
            loaded_config: None,
        }
    }
}