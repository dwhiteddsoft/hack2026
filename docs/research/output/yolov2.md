# YOLOv2 Output Format Research

## Overview
YOLOv2's output represents a significant evolution from YOLOv1, introducing anchor boxes for better handling of objects at different scales and aspect ratios. The output maintains YOLO's direct regression approach while improving accuracy through predefined anchor box shapes and dimension clustering.

## Output Tensor Specification

### Tensor Shape: `[batch_size, channels, grid_height, grid_width]`
- **batch_size**: Number of images processed (typically 1 for inference)
- **channels**: 425 channels total for COCO dataset (80 classes)
- **grid_height**: 13 (input_height / 32 = 416 / 32 = 13)
- **grid_width**: 13 (input_width / 32 = 416 / 32 = 13)

**Standard output shape: `[1, 425, 13, 13]`**

## Channel Structure Breakdown

### Anchor Box Configuration:
YOLOv2 uses 5 anchor boxes per grid cell, each predicting:
- **5 bounding box parameters**: (x, y, w, h, confidence)
- **80 class probabilities**: One for each COCO class

**Channel calculation:**
```
Total channels = num_anchors × (5 + num_classes)
425 = 5 × (5 + 80)
425 = 5 × 85
```

### Channel Layout:
```
Anchor 0: Channels 0-84    (85 channels)
├── Box coords: 0-3       (x, y, w, h)
├── Confidence: 4         (objectness)
└── Classes: 5-84         (80 class probabilities)

Anchor 1: Channels 85-169  (85 channels)
├── Box coords: 85-88
├── Confidence: 89
└── Classes: 90-169

Anchor 2: Channels 170-254 (85 channels)
├── Box coords: 170-173
├── Confidence: 174
└── Classes: 175-254

Anchor 3: Channels 255-339 (85 channels)
├── Box coords: 255-258
├── Confidence: 259
└── Classes: 260-339

Anchor 4: Channels 340-424 (85 channels)
├── Box coords: 340-343
├── Confidence: 344
└── Classes: 345-424
```

## Anchor Box System

### Predefined Anchor Shapes:
YOLOv2 uses 5 anchor boxes determined through k-means clustering on the training dataset:

**Common COCO anchor boxes (relative to 416×416 input):**
```
Anchor 0: width=0.57273, height=0.677385  # Medium objects
Anchor 1: width=1.87446, height=2.06253   # Large objects
Anchor 2: width=3.33843, height=5.47434   # Very large objects
Anchor 3: width=7.88282, height=3.52778   # Wide objects
Anchor 4: width=9.77052, height=9.16828   # Very large square objects
```

### Anchor Box Benefits:
- **Better aspect ratios**: Handles objects with different shapes
- **Scale awareness**: Different anchors for different object sizes
- **Improved recall**: Multiple predictions per location
- **Specialized detection**: Each anchor specializes in certain object types

## Output Interpretation

### Raw Output Structure:
For each grid cell (i, j) and anchor box k:
```python
# Channel offset for anchor k
offset = k * 85

# Extract predictions for this anchor
tx = output[0, offset + 0, i, j]     # x offset
ty = output[0, offset + 1, i, j]     # y offset
tw = output[0, offset + 2, i, j]     # width scale
th = output[0, offset + 3, i, j]     # height scale
tc = output[0, offset + 4, i, j]     # confidence score

# Class probabilities
class_probs = output[0, offset + 5:offset + 85, i, j]  # 80 classes
```

### Coordinate Transformation:
YOLOv2 uses a sophisticated coordinate system:

```python
# Apply activations
x = (sigmoid(tx) + j) / grid_size    # Relative to image width
y = (sigmoid(ty) + i) / grid_size    # Relative to image height
w = anchor_w * exp(tw) / grid_size   # Relative to image width
h = anchor_h * exp(th) / grid_size   # Relative to image height
confidence = sigmoid(tc)             # Objectness probability

# Convert to absolute coordinates (for 416x416 input)
center_x = x * 416
center_y = y * 416
box_width = w * 416
box_height = h * 416

# Convert to corner coordinates
x1 = center_x - box_width / 2
y1 = center_y - box_height / 2
x2 = center_x + box_width / 2
y2 = center_y + box_height / 2
```

### Class Probability Processing:
```python
# Apply softmax to class scores
class_scores = softmax(class_probs)

# Or interpret as individual probabilities
class_probs = sigmoid(class_probs)  # Alternative interpretation

# Final class confidence
final_scores = confidence * class_scores
```

## Post-Processing Pipeline

### 1. Activation Functions:
```python
def apply_activations(raw_output):
    """Apply appropriate activations to raw YOLOv2 output"""
    processed = raw_output.copy()
    
    for anchor in range(5):
        offset = anchor * 85
        
        # Sigmoid for x, y coordinates and confidence
        processed[:, offset:offset+2, :, :] = sigmoid(processed[:, offset:offset+2, :, :])
        processed[:, offset+4:offset+5, :, :] = sigmoid(processed[:, offset+4:offset+5, :, :])
        
        # Softmax for class probabilities
        processed[:, offset+5:offset+85, :, :] = softmax(processed[:, offset+5:offset+85, :, :], axis=1)
    
    return processed
```

### 2. Detection Extraction:
```python
def extract_detections(output, anchors, grid_size=13, input_size=416, conf_threshold=0.5):
    """Extract bounding boxes and class predictions from YOLOv2 output"""
    detections = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(5):  # 5 anchor boxes
                offset = k * 85
                
                # Extract raw predictions
                tx = output[0, offset + 0, i, j]
                ty = output[0, offset + 1, i, j]
                tw = output[0, offset + 2, i, j]
                th = output[0, offset + 3, i, j]
                tc = output[0, offset + 4, i, j]
                
                # Apply activations
                x = (sigmoid(tx) + j) / grid_size
                y = (sigmoid(ty) + i) / grid_size
                w = anchors[k][0] * np.exp(tw) / grid_size
                h = anchors[k][1] * np.exp(th) / grid_size
                confidence = sigmoid(tc)
                
                # Skip if confidence too low
                if confidence < conf_threshold:
                    continue
                
                # Get class probabilities
                class_probs = output[0, offset + 5:offset + 85, i, j]
                class_probs = softmax(class_probs)
                
                # Convert to absolute coordinates
                center_x = x * input_size
                center_y = y * input_size
                box_width = w * input_size
                box_height = h * input_size
                
                # Convert to corner format
                x1 = center_x - box_width / 2
                y1 = center_y - box_height / 2
                x2 = center_x + box_width / 2
                y2 = center_y + box_height / 2
                
                # Find best class
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id]
                final_score = confidence * class_score
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': final_score,
                    'class_id': class_id,
                    'class_score': class_score,
                    'objectness': confidence
                }
                
                detections.append(detection)
    
    return detections
```

### 3. Non-Maximum Suppression:
```python
def apply_nms(detections, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence score
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while detections:
        # Keep the highest confidence detection
        current = detections.pop(0)
        keep.append(current)
        
        # Remove overlapping detections
        detections = [
            det for det in detections 
            if calculate_iou(current['bbox'], det['bbox']) < iou_threshold
        ]
    
    return keep

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0
```

## Framework Implementation Examples

### PyTorch Implementation:
```python
import torch
import torch.nn.functional as F

def process_yolov2_output(output, anchors, conf_threshold=0.5, nms_threshold=0.45):
    """Process YOLOv2 output tensor to get final detections"""
    batch_size, channels, grid_h, grid_w = output.shape
    num_anchors = len(anchors)
    num_classes = (channels // num_anchors) - 5
    
    # Reshape output for easier processing
    output = output.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
    output = output.permute(0, 1, 3, 4, 2).contiguous()  # [batch, anchors, grid_h, grid_w, 85]
    
    # Apply sigmoid to x, y, confidence
    output[..., 0:2] = torch.sigmoid(output[..., 0:2])
    output[..., 4] = torch.sigmoid(output[..., 4])
    
    # Apply softmax to class scores
    output[..., 5:] = F.softmax(output[..., 5:], dim=-1)
    
    # Generate grid coordinates
    grid_x = torch.arange(grid_w).repeat(grid_h, 1).float()
    grid_y = torch.arange(grid_h).repeat(grid_w, 1).t().float()
    
    # Convert predictions to absolute coordinates
    pred_boxes = torch.zeros_like(output[..., :4])
    pred_boxes[..., 0] = (output[..., 0] + grid_x) / grid_w  # center x
    pred_boxes[..., 1] = (output[..., 1] + grid_y) / grid_h  # center y
    
    # Apply anchor boxes for width and height
    for i, (anchor_w, anchor_h) in enumerate(anchors):
        pred_boxes[:, i, :, :, 2] = anchor_w * torch.exp(output[:, i, :, :, 2]) / grid_w
        pred_boxes[:, i, :, :, 3] = anchor_h * torch.exp(output[:, i, :, :, 3]) / grid_h
    
    # Combine all predictions
    confidence = output[..., 4]
    class_probs = output[..., 5:]
    
    return pred_boxes, confidence, class_probs
```

### TensorFlow Implementation:
```python
import tensorflow as tf

def yolov2_decode(output, anchors, input_size=416):
    """Decode YOLOv2 output tensor"""
    grid_size = tf.shape(output)[1]  # Assuming square grid
    num_anchors = len(anchors)
    
    # Reshape output
    output = tf.reshape(output, [-1, grid_size, grid_size, num_anchors, 85])
    
    # Create grid coordinates
    grid_y = tf.range(grid_size, dtype=tf.float32)
    grid_x = tf.range(grid_size, dtype=tf.float32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    grid = tf.stack([grid_x, grid_y], axis=-1)
    grid = tf.expand_dims(tf.expand_dims(grid, 0), 3)
    
    # Extract predictions
    box_xy = tf.sigmoid(output[..., :2])
    box_wh = tf.exp(output[..., 2:4])
    box_confidence = tf.sigmoid(output[..., 4:5])
    box_class_probs = tf.nn.softmax(output[..., 5:])
    
    # Convert to absolute coordinates
    box_xy = (box_xy + grid) / tf.cast(grid_size, tf.float32)
    
    # Apply anchors
    anchors_tensor = tf.constant(anchors, dtype=tf.float32)
    anchors_tensor = tf.reshape(anchors_tensor, [1, 1, 1, num_anchors, 2])
    box_wh = box_wh * anchors_tensor / tf.cast(grid_size, tf.float32)
    
    # Convert to corner coordinates
    box_mins = box_xy - box_wh / 2.0
    box_maxs = box_xy + box_wh / 2.0
    boxes = tf.concat([box_mins, box_maxs], axis=-1)
    
    # Scale to input size
    boxes = boxes * input_size
    
    return boxes, box_confidence, box_class_probs
```

### ONNX Runtime Implementation:
```python
import onnxruntime as ort
import numpy as np

def process_onnx_yolov2_output(onnx_output, anchors, input_size=416):
    """Process ONNX YOLOv2 output"""
    # ONNX output is typically [1, 425, 13, 13]
    output = onnx_output[0]  # Remove batch dimension for processing
    
    batch_size, channels, grid_h, grid_w = output.shape
    num_anchors = len(anchors)
    num_classes = (channels // num_anchors) - 5
    
    detections = []
    
    for i in range(grid_h):
        for j in range(grid_w):
            for k in range(num_anchors):
                offset = k * (5 + num_classes)
                
                # Extract raw predictions
                tx = output[0, offset + 0, i, j]
                ty = output[0, offset + 1, i, j]
                tw = output[0, offset + 2, i, j]
                th = output[0, offset + 3, i, j]
                tc = output[0, offset + 4, i, j]
                
                # Apply activations
                x = (sigmoid(tx) + j) / grid_w
                y = (sigmoid(ty) + i) / grid_h
                w = anchors[k][0] * np.exp(tw) / grid_w
                h = anchors[k][1] * np.exp(th) / grid_h
                confidence = sigmoid(tc)
                
                # Skip low confidence predictions
                if confidence < 0.5:
                    continue
                
                # Get class scores
                class_scores = output[0, offset + 5:offset + 5 + num_classes, i, j]
                class_scores = softmax(class_scores)
                
                # Convert to pixel coordinates
                center_x = x * input_size
                center_y = y * input_size
                box_width = w * input_size
                box_height = h * input_size
                
                # Convert to corner format
                x1 = center_x - box_width / 2
                y1 = center_y - box_height / 2
                x2 = center_x + box_width / 2
                y2 = center_y + box_height / 2
                
                # Find best class
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                final_score = confidence * class_score
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(final_score),
                    'class_id': int(class_id),
                    'objectness': float(confidence),
                    'class_score': float(class_score)
                }
                
                detections.append(detection)
    
    return detections

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
```

## Comparison with YOLOv1 Output

### Key Differences:
| Aspect | YOLOv1 | YOLOv2 | Improvement |
|--------|---------|---------|-------------|
| Grid size | 7×7 | 13×13 | Higher resolution |
| Predictions per cell | 2 boxes | 5 anchor boxes | More diverse shapes |
| Coordinate system | Direct regression | Anchor-based | Better aspect ratios |
| Classes per box | Shared across boxes | Per-anchor box | More precise |
| Output channels | 30 | 425 | Richer predictions |

### Benefits of YOLOv2 Output:
1. **Better localization**: 13×13 grid provides finer spatial resolution
2. **Improved aspect ratios**: Anchor boxes handle diverse object shapes
3. **Higher recall**: More predictions per location increases detection chances
4. **Stable training**: Anchor boxes provide better gradient flow
5. **Scale awareness**: Different anchors specialize in different object sizes

## Performance Metrics

### Output Complexity:
- **Total predictions**: 13 × 13 × 5 = 845 bounding boxes
- **Memory usage**: ~2.2MB for output tensor (float32)
- **Post-processing time**: ~5-10ms on modern CPU
- **NMS complexity**: O(n²) where n is number of detections

### Detection Statistics:
- **Average detections per image**: 10-50 (after filtering)
- **Confidence threshold**: Typically 0.5-0.7
- **NMS threshold**: Typically 0.45-0.5
- **Class coverage**: 80 COCO classes

## Best Practices for YOLOv2 Output Processing

### 1. Efficient Post-Processing:
```python
def optimized_yolov2_postprocess(output, anchors, conf_thresh=0.5, nms_thresh=0.45):
    """Optimized post-processing for YOLOv2 output"""
    # Vectorized processing where possible
    # Early filtering by confidence
    # Batch NMS operations
    # Memory-efficient operations
    pass
```

### 2. Multi-Scale Testing:
```python
def multi_scale_inference(model, image, scales=[320, 416, 512]):
    """Run inference at multiple scales and combine results"""
    all_detections = []
    
    for scale in scales:
        resized_image = resize_image(image, scale)
        output = model(resized_image)
        detections = process_yolov2_output(output, anchors)
        
        # Scale detections back to original image size
        scaled_detections = scale_detections(detections, scale, image.shape)
        all_detections.extend(scaled_detections)
    
    # Apply NMS across all scales
    final_detections = apply_nms(all_detections, nms_thresh=0.3)
    
    return final_detections
```

### 3. Class-Specific Processing:
```python
def class_specific_nms(detections, class_nms_thresholds):
    """Apply different NMS thresholds for different classes"""
    results = []
    
    # Group detections by class
    by_class = defaultdict(list)
    for det in detections:
        by_class[det['class_id']].append(det)
    
    # Apply class-specific NMS
    for class_id, class_detections in by_class.items():
        nms_thresh = class_nms_thresholds.get(class_id, 0.45)
        filtered = apply_nms(class_detections, nms_thresh)
        results.extend(filtered)
    
    return results
```

## Common Issues and Solutions

### 1. **Coordinate System Confusion**:
- **Problem**: Incorrect coordinate transformation
- **Solution**: Carefully implement sigmoid for x,y and exp for w,h
- **Validation**: Test with known ground truth boxes

### 2. **Anchor Box Mismatch**:
- **Problem**: Using wrong anchor boxes for model
- **Solution**: Extract anchors from model configuration
- **Check**: Verify anchor boxes match training configuration

### 3. **Activation Function Errors**:
- **Problem**: Wrong activation functions applied
- **Solution**: Sigmoid for coordinates/confidence, softmax for classes
- **Debug**: Verify output ranges are reasonable

### 4. **NMS Parameter Tuning**:
- **Problem**: Too aggressive or too lenient NMS
- **Solution**: Tune thresholds based on application requirements
- **Balance**: Trade-off between precision and recall

YOLOv2's output format represents a significant improvement over YOLOv1, introducing anchor boxes and finer spatial resolution while maintaining the single-stage detection paradigm. The 425-channel output provides rich predictions that, when properly post-processed, deliver accurate object detection with good recall and precision characteristics.