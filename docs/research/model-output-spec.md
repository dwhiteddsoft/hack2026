# Canonical Computer Vision Model Output Specification

## Executive Summary

After analyzing output specifications across 12 major computer vision models (YOLO family v2-v9, SSD variants, RetinaNet, Mask R-CNN), I propose a **Universal Computer Vision Output Descriptor (UCVOD)** that canonically describes any model's output requirements through a structured, extensible format.

## Canonical Output Model Design

### Core Output Structure
```rust
pub struct CanonicalOutput {
    pub output_spec: OutputSpecification,
    pub post_processing: PostProcessingPipeline,
    pub detection_format: DetectionFormat,
    pub metadata: OutputMetadata,
}

pub struct OutputSpecification {
    pub architecture_type: ArchitectureType,
    pub tensor_outputs: Vec<TensorOutput>,
    pub coordinate_system: CoordinateSystem,
    pub activation_requirements: ActivationRequirements,
}

pub enum ArchitectureType {
    SingleStage {
        unified_head: bool,              // YOLOv8: true, YOLOv6: false
        anchor_based: bool,              // YOLOv2/v3: true, YOLOv8: false
    },
    TwoStage {
        rpn_outputs: Vec<TensorOutput>,  // Mask R-CNN RPN outputs
        rcnn_outputs: Vec<TensorOutput>, // Final detection outputs
        additional_tasks: Vec<TaskType>, // Segmentation, pose, etc.
    },
    MultiScale {
        scale_strategy: ScaleStrategy,   // FPN, YOLO multi-scale, etc.
        shared_head: bool,               // Same head across scales
    },
}

pub enum TaskType {
    Detection,
    Segmentation,
    Classification,
    PoseEstimation,
    DepthEstimation,
}
```

### Tensor Output Specification
```rust
pub struct TensorOutput {
    pub name: String,                    // "output0", "cls_output", "reg_output"
    pub shape: TensorShape,
    pub content_type: ContentType,
    pub spatial_layout: SpatialLayout,
    pub channel_interpretation: ChannelInterpretation,
}

pub struct TensorShape {
    pub dimensions: Vec<OutputDimension>,
    pub layout_format: LayoutFormat,     // NCHW, NHWC, etc.
}

pub enum OutputDimension {
    Batch(i64),                          // Typically 1
    Classes(i64),                        // 80 for COCO, 1000 for ImageNet
    Coordinates(i64),                    // 4 for bbox, 8 for polygon
    Anchors(i64),                        // Number of anchors per location
    Height(i64),                         // Feature map height
    Width(i64),                          // Feature map width
    Features(i64),                       // Generic feature dimension
    Combined(CombinedDimension),         // Mixed interpretations
}

pub struct CombinedDimension {
    pub total_size: i64,
    pub components: Vec<DimensionComponent>,
}

pub struct DimensionComponent {
    pub component_type: ComponentType,
    pub size: i64,
    pub offset: i64,
}

pub enum ComponentType {
    BoundingBox,                         // 4 values: x, y, w, h or x1, y1, x2, y2
    Confidence,                          // 1 value: objectness score
    ClassLogits,                         // N values: class predictions
    Mask,                                // Segmentation mask
    Keypoints,                           // Pose estimation points
}
```

### Content Type Classification
```rust
pub enum ContentType {
    Classification {
        num_classes: i64,
        background_class: bool,          // SSD has background class
        multi_label: bool,               // Can detect multiple classes per box
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

pub enum CoordinateFormat {
    CenterWidthHeight,                   // YOLO: (cx, cy, w, h)
    CornerCoordinates,                   // SSD, RetinaNet: (x1, y1, x2, y2)
    Offsets,                             // Faster R-CNN: (dx, dy, dw, dh)
}

pub enum CoordinateNormalization {
    Normalized,                          // [0, 1] range
    Pixel,                               // Absolute pixel coordinates
    GridRelative,                        // Relative to grid cell
    AnchorRelative,                      // Relative to anchor box
}
```

### Spatial Layout Patterns
```rust
pub enum SpatialLayout {
    Grid {
        grid_size: (i64, i64),
        stride: i64,                     // Downsampling factor
        predictions_per_cell: i64,       // YOLO: 1, anchor-based: N
    },
    Dense {
        feature_map_size: (i64, i64),
        anchor_count: i64,               // RetinaNet: 9, SSD: varies
    },
    Proposals {
        max_proposals: i64,              // Mask R-CNN: ~1000
        nms_applied: bool,               // Pre or post NMS
    },
    Unified {
        total_predictions: i64,          // YOLOv8: flattened predictions
        multi_scale: bool,               // Combined scales
    },
}
```

### Channel Interpretation
```rust
pub enum ChannelInterpretation {
    Interleaved {
        pattern: Vec<ComponentType>,     // [bbox, conf, classes] repeating
        repetitions: i64,                // Number of anchors
    },
    Separated {
        layout: SeparatedLayout,         // YOLOv6: separate cls/reg heads
    },
    Unified {
        components: Vec<ComponentRange>, // YOLOv8: [bbox(4), classes(80)]
    },
}

pub struct ComponentRange {
    pub component_type: ComponentType,
    pub start_channel: i64,
    pub end_channel: i64,
}

pub enum SeparatedLayout {
    ClassificationRegression,            // YOLOv6: separate cls/reg
    MultiTask,                           // Mask R-CNN: bbox/cls/mask
    FeaturePyramid,                      // RetinaNet: separate per scale
}
```

## Model Family Mappings

### YOLOv8 (Anchor-Free, Unified)
```rust
let yolov8_output = CanonicalOutput {
    output_spec: OutputSpecification {
        architecture_type: ArchitectureType::MultiScale {
            scale_strategy: ScaleStrategy::FeaturePyramidUnified,
            shared_head: true,
        },
        tensor_outputs: vec![
            TensorOutput {
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
                                    component_type: ComponentType::ClassLogits,
                                    size: 80,
                                    offset: 4,
                                },
                            ],
                        }),
                        OutputDimension::Height(20),
                        OutputDimension::Width(20),
                    ],
                    layout_format: LayoutFormat::NCHW,
                },
                content_type: ContentType::Combined {
                    components: vec![
                        ContentType::Regression {
                            coordinate_format: CoordinateFormat::CenterWidthHeight,
                            normalization: CoordinateNormalization::GridRelative,
                        },
                        ContentType::Classification {
                            num_classes: 80,
                            background_class: false,
                            multi_label: false,
                        },
                    ],
                },
                spatial_layout: SpatialLayout::Grid {
                    grid_size: (20, 20),
                    stride: 32,
                    predictions_per_cell: 1,
                },
                channel_interpretation: ChannelInterpretation::Unified {
                    components: vec![
                        ComponentRange {
                            component_type: ComponentType::BoundingBox,
                            start_channel: 0,
                            end_channel: 4,
                        },
                        ComponentRange {
                            component_type: ComponentType::ClassLogits,
                            start_channel: 4,
                            end_channel: 84,
                        },
                    ],
                },
            },
            // Additional tensors for other scales (40x40, 80x80)
        ],
        coordinate_system: CoordinateSystem::Direct,
        activation_requirements: ActivationRequirements {
            bbox_activation: ActivationType::None, // Direct regression
            class_activation: ActivationType::Sigmoid,
            confidence_activation: ActivationType::None, // No objectness
        },
    },
    post_processing: PostProcessingPipeline {
        steps: vec![
            PostProcessingStep::CoordinateDecoding {
                decode_strategy: DecodeStrategy::DirectRegression,
            },
            PostProcessingStep::ClassActivation {
                activation: ActivationType::Sigmoid,
            },
            PostProcessingStep::ConfidenceFiltering {
                threshold: 0.25,
                per_class: true,
            },
            PostProcessingStep::NonMaximumSuppression {
                iou_threshold: 0.45,
                strategy: NMSStrategy::Standard,
            },
        ],
    },
};
```

### YOLOv2 (Anchor-Based, Unified)
```rust
let yolov2_output = CanonicalOutput {
    output_spec: OutputSpecification {
        architecture_type: ArchitectureType::SingleStage {
            unified_head: true,
            anchor_based: true,
        },
        tensor_outputs: vec![
            TensorOutput {
                name: "output".to_string(),
                shape: TensorShape {
                    dimensions: vec![
                        OutputDimension::Batch(1),
                        OutputDimension::Combined(CombinedDimension {
                            total_size: 425,
                            components: vec![
                                DimensionComponent {
                                    component_type: ComponentType::BoundingBox,
                                    size: 4,
                                    offset: 0, // Repeats every 85 channels
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
                        OutputDimension::Height(13),
                        OutputDimension::Width(13),
                    ],
                    layout_format: LayoutFormat::NCHW,
                },
                spatial_layout: SpatialLayout::Grid {
                    grid_size: (13, 13),
                    stride: 32,
                    predictions_per_cell: 5, // 5 anchor boxes
                },
                channel_interpretation: ChannelInterpretation::Interleaved {
                    pattern: vec![
                        ComponentType::BoundingBox,
                        ComponentType::Confidence,
                        ComponentType::ClassLogits,
                    ],
                    repetitions: 5,
                },
            },
        ],
        activation_requirements: ActivationRequirements {
            bbox_activation: ActivationType::SigmoidExp, // Sigmoid for x,y; exp for w,h
            class_activation: ActivationType::Softmax,
            confidence_activation: ActivationType::Sigmoid,
        },
    },
};
```

### SSD (Multi-Output, Anchor-Based)
```rust
let ssd_output = CanonicalOutput {
    output_spec: OutputSpecification {
        architecture_type: ArchitectureType::MultiScale {
            scale_strategy: ScaleStrategy::FeaturePyramidSeparated,
            shared_head: false,
        },
        tensor_outputs: vec![
            // Classification outputs
            TensorOutput {
                name: "cls_1".to_string(),
                content_type: ContentType::Classification {
                    num_classes: 81, // Including background
                    background_class: true,
                    multi_label: false,
                },
                spatial_layout: SpatialLayout::Dense {
                    feature_map_size: (38, 38),
                    anchor_count: 4,
                },
            },
            // Localization outputs  
            TensorOutput {
                name: "loc_1".to_string(),
                content_type: ContentType::Regression {
                    coordinate_format: CoordinateFormat::Offsets,
                    normalization: CoordinateNormalization::AnchorRelative,
                },
                spatial_layout: SpatialLayout::Dense {
                    feature_map_size: (38, 38),
                    anchor_count: 4,
                },
            },
            // Additional scale outputs...
        ],
    },
    post_processing: PostProcessingPipeline {
        steps: vec![
            PostProcessingStep::BackgroundFiltering,
            PostProcessingStep::CoordinateDecoding {
                decode_strategy: DecodeStrategy::AnchorBoxDecoding,
            },
            PostProcessingStep::ClassActivation {
                activation: ActivationType::Softmax,
            },
            PostProcessingStep::NonMaximumSuppression {
                iou_threshold: 0.45,
                strategy: NMSStrategy::PerClass,
            },
        ],
    },
};
```

### Mask R-CNN (Two-Stage, Multi-Task)
```rust
let mask_rcnn_output = CanonicalOutput {
    output_spec: OutputSpecification {
        architecture_type: ArchitectureType::TwoStage {
            rpn_outputs: vec![
                TensorOutput {
                    name: "rpn_cls".to_string(),
                    content_type: ContentType::Objectness {
                        confidence_type: ConfidenceType::Binary,
                    },
                },
                TensorOutput {
                    name: "rpn_reg".to_string(),
                    content_type: ContentType::Regression {
                        coordinate_format: CoordinateFormat::Offsets,
                        normalization: CoordinateNormalization::AnchorRelative,
                    },
                },
            ],
            rcnn_outputs: vec![
                TensorOutput {
                    name: "rcnn_cls".to_string(),
                    content_type: ContentType::Classification {
                        num_classes: 81,
                        background_class: true,
                        multi_label: false,
                    },
                    spatial_layout: SpatialLayout::Proposals {
                        max_proposals: 1000,
                        nms_applied: false,
                    },
                },
                TensorOutput {
                    name: "rcnn_reg".to_string(),
                    content_type: ContentType::Regression {
                        coordinate_format: CoordinateFormat::Offsets,
                        normalization: CoordinateNormalization::Pixel,
                    },
                },
                TensorOutput {
                    name: "rcnn_mask".to_string(),
                    content_type: ContentType::Segmentation {
                        mask_format: MaskFormat::Binary,
                        resolution: (28, 28),
                    },
                },
            ],
            additional_tasks: vec![TaskType::Segmentation],
        },
    },
};
```

## Post-Processing Pipeline Specification
```rust
pub struct PostProcessingPipeline {
    pub steps: Vec<PostProcessingStep>,
    pub optimization_config: OptimizationConfig,
}

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
    BackgroundFiltering,
    MultiScaleMerging {
        merge_strategy: MergeStrategy,
    },
    CoordinateNormalization {
        target_format: CoordinateFormat,
    },
}

pub enum DecodeStrategy {
    DirectRegression,                    // YOLOv8: direct coordinate prediction
    AnchorBoxDecoding,                   // YOLOv2, SSD: anchor-based decoding
    ProposalDecoding,                    // Faster R-CNN: RPN proposal decoding
    DistributionFocalLoss,               // Advanced YOLO: DFL decoding
}

pub enum NMSStrategy {
    Standard,                            // Standard IoU-based NMS
    PerClass,                            // Apply NMS per class separately
    Matrix,                              // Matrix NMS for efficiency
    Soft,                                // Soft NMS with decay
    Batched,                             // Batched NMS for multiple images
}
```

## Unified Processing Interface
```rust
impl CanonicalOutput {
    pub fn process_raw_outputs(&self, raw_outputs: Vec<Tensor>) -> Result<Vec<Detection>, ProcessingError> {
        let mut intermediate_results = Vec::new();
        
        // Stage 1: Decode tensor outputs according to specification
        for (tensor, spec) in raw_outputs.iter().zip(&self.output_spec.tensor_outputs) {
            let decoded = self.decode_tensor_output(tensor, spec)?;
            intermediate_results.push(decoded);
        }
        
        // Stage 2: Apply post-processing pipeline
        let mut detections = self.merge_multi_scale_outputs(intermediate_results)?;
        
        for step in &self.post_processing.steps {
            detections = self.apply_processing_step(detections, step)?;
        }
        
        Ok(detections)
    }
    
    pub fn validate_output_compatibility(&self, raw_outputs: &[Tensor]) -> Result<(), ValidationError> {
        // Validate tensor count, shapes, and content expectations
        self.validate_tensor_count(raw_outputs)?;
        self.validate_tensor_shapes(raw_outputs)?;
        self.validate_content_ranges(raw_outputs)?;
        Ok(())
    }
    
    pub fn optimize_for_deployment(&mut self, hardware: &HardwareInfo, performance_target: PerformanceTarget) {
        // Adjust post-processing pipeline for deployment constraints
        match performance_target {
            PerformanceTarget::Speed => self.optimize_for_speed(),
            PerformanceTarget::Accuracy => self.optimize_for_accuracy(),
            PerformanceTarget::Memory => self.optimize_for_memory(),
        }
    }
}
```

## Benefits of Canonical Output Model

### 1. **Universal Coverage**
- **All architectures**: Single-stage, two-stage, multi-task
- **All prediction formats**: Anchor-based, anchor-free, proposal-based
- **All coordinate systems**: Center-wh, corners, offsets, normalized
- **All tasks**: Detection, segmentation, classification, pose

### 2. **Automatic Post-Processing**
- **Framework agnostic**: Works with any tensor format
- **Optimized pipelines**: Hardware-specific optimizations
- **Validation built-in**: Automatic output validation
- **Performance tuning**: Speed vs accuracy trade-offs

### 3. **Extensible Design**
- **New models**: Easy to add specifications
- **New tasks**: Extend task types and content types
- **New optimizations**: Add processing steps
- **Custom formats**: Support proprietary model outputs

### 4. **Development Efficiency**
```rust
// Universal usage pattern
let model_spec = CanonicalOutput::from_model_type(ModelType::YOLOv8n)?;
let raw_outputs = run_inference(model, input_tensor)?;
let detections = model_spec.process_raw_outputs(raw_outputs)?;

// Works identically for any model
let ssd_spec = CanonicalOutput::from_model_type(ModelType::SSD300)?;
let mask_rcnn_spec = CanonicalOutput::from_model_type(ModelType::MaskRCNN)?;
```

## Conclusion

This canonical output model provides:

1. **Complete coverage** of all analyzed computer vision model outputs
2. **Structured representation** of complex output formats
3. **Automatic post-processing** pipeline generation
4. **Performance optimization** capabilities
5. **Easy extensibility** for future models and tasks
6. **Framework independence** while maintaining efficiency

The design abstracts the complexity of different output formats (single tensors, multiple tensors, anchor-based, anchor-free, multi-task) into a unified interface while preserving the ability to optimize for specific deployment scenarios and hardware constraints.