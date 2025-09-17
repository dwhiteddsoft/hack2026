# SSD (Single Shot MultiBox Detector) Output Format Analysis

## Overview
SSD (Single Shot MultiBox Detector) is a foundational single-stage object detection model that introduced multi-scale feature map detection and default boxes (anchors). It produces separate outputs for classification and localization across multiple feature map scales, making it significantly different from YOLO's unified output approach.

## Output Tensor Structure

### Multi-Scale Feature Map Architecture
SSD produces **separate classification and localization outputs** for each feature map scale:

### Standard SSD300 Output Dimensions
For SSD with 300×300 input and 6 feature map scales:

**Classification Outputs (confidence scores):**
```
Scale 1 (38×38): [1, 16, 38, 38]    # 4 boxes × 4 classes (background + 3 foreground)
Scale 2 (19×19): [1, 24, 19, 19]    # 6 boxes × 4 classes  
Scale 3 (10×10): [1, 24, 10, 10]    # 6 boxes × 4 classes
Scale 4 (5×5):   [1, 24, 5, 5]      # 6 boxes × 4 classes
Scale 5 (3×3):   [1, 16, 3, 3]      # 4 boxes × 4 classes
Scale 6 (1×1):   [1, 16, 1, 1]      # 4 boxes × 4 classes
```

**Localization Outputs (bounding box offsets):**
```
Scale 1 (38×38): [1, 16, 38, 38]    # 4 boxes × 4 coordinates
Scale 2 (19×19): [1, 24, 19, 19]    # 6 boxes × 4 coordinates
Scale 3 (10×10): [1, 24, 10, 10]    # 6 boxes × 4 coordinates  
Scale 4 (5×5):   [1, 24, 5, 5]      # 6 boxes × 4 coordinates
Scale 5 (3×3):   [1, 16, 3, 3]      # 4 boxes × 4 coordinates
Scale 6 (1×1):   [1, 16, 1, 1]      # 4 boxes × 4 coordinates
```

### Default Box Configuration
SSD uses different numbers of default boxes per feature map location:

```python
ssd300_config = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'steps': [8, 16, 32, 64, 100, 300],      # Downsampling factors
    'sizes': [30, 60, 111, 162, 213, 264, 315],  # Default box sizes
    'aspect_ratios': [
        [2],           # 38×38: 4 boxes (1, 1/2, 2, 1_extra)
        [2, 3],        # 19×19: 6 boxes (1, 1/2, 2, 1/3, 3, 1_extra)
        [2, 3],        # 10×10: 6 boxes
        [2, 3],        # 5×5:  6 boxes
        [2],           # 3×3:  4 boxes
        [2]            # 1×1:  4 boxes
    ]
}
```

### Channel Structure Explanation
For SSD with C classes (including background):

- **Classification channels**: `num_boxes_per_location × (C + 1)`
- **Localization channels**: `num_boxes_per_location × 4`

## Default Box (Anchor) System

### Default Box Generation
SSD generates default boxes with different scales and aspect ratios:

```python
def generate_ssd_default_boxes(config):
    """Generate SSD default boxes for all feature maps"""
    default_boxes = []
    
    for k, (fmap_size, step, min_size, max_size, aspect_ratios) in enumerate(
        zip(config['feature_maps'], config['steps'], 
            config['sizes'][:-1], config['sizes'][1:], 
            config['aspect_ratios'])
    ):
        for i in range(fmap_size):
            for j in range(fmap_size):
                # Center coordinates
                cx = (j + 0.5) / fmap_size
                cy = (i + 0.5) / fmap_size
                
                # Default box with aspect ratio 1
                size = min_size / 300.0  # Normalize to [0, 1]
                default_boxes.append([cx, cy, size, size])
                
                # Additional box with scale sqrt(min_size * max_size)
                size = sqrt(min_size * max_size) / 300.0
                default_boxes.append([cx, cy, size, size])
                
                # Boxes with different aspect ratios
                for ar in aspect_ratios:
                    w = size * sqrt(ar)
                    h = size / sqrt(ar)
                    default_boxes.append([cx, cy, w, h])
                    default_boxes.append([cx, cy, h, w])  # Reciprocal
    
    return torch.tensor(default_boxes)
```

### Default Box Matching
During training and inference, default boxes are matched to ground truth:

```python
def match_default_boxes(ground_truth_boxes, default_boxes, threshold=0.5):
    """Match ground truth boxes to default boxes"""
    
    # Calculate IoU between all pairs
    ious = calculate_iou_matrix(ground_truth_boxes, default_boxes)
    
    # For each ground truth, find best matching default box
    best_default_per_gt = torch.argmax(ious, dim=1)
    
    # For each default box, find best matching ground truth
    best_gt_per_default = torch.argmax(ious, dim=0)
    best_gt_iou_per_default = torch.max(ious, dim=0)[0]
    
    # Mark matches above threshold as positive
    positive_mask = best_gt_iou_per_default > threshold
    
    # Ensure each ground truth has at least one match
    for i, best_default in enumerate(best_default_per_gt):
        positive_mask[best_default] = True
        best_gt_per_default[best_default] = i
    
    return positive_mask, best_gt_per_default
```

## Mathematical Transformations

### Localization Decoding
SSD predicts offsets to default boxes, not absolute coordinates:

```python
def decode_ssd_predictions(loc_preds, conf_preds, default_boxes, variance=[0.1, 0.2]):
    """
    Decode SSD predictions to absolute coordinates
    
    Args:
        loc_preds: [N, 4] - predicted offsets (tx, ty, tw, th)
        conf_preds: [N, C] - predicted class confidences
        default_boxes: [N, 4] - default box coordinates (cx, cy, w, h)
        variance: [2] - variance values for decoding
    """
    
    # Extract default box parameters
    default_cx = default_boxes[:, 0]
    default_cy = default_boxes[:, 1] 
    default_w = default_boxes[:, 2]
    default_h = default_boxes[:, 3]
    
    # Extract predicted offsets
    tx = loc_preds[:, 0]
    ty = loc_preds[:, 1]
    tw = loc_preds[:, 2]
    th = loc_preds[:, 3]
    
    # Decode center coordinates
    predicted_cx = tx * variance[0] * default_w + default_cx
    predicted_cy = ty * variance[0] * default_h + default_cy
    
    # Decode width and height
    predicted_w = torch.exp(tw * variance[1]) * default_w
    predicted_h = torch.exp(th * variance[1]) * default_h
    
    # Convert to corner coordinates
    x1 = predicted_cx - predicted_w / 2
    y1 = predicted_cy - predicted_h / 2
    x2 = predicted_cx + predicted_w / 2
    y2 = predicted_cy + predicted_h / 2
    
    return torch.stack([x1, y1, x2, y2], dim=1)

def encode_ssd_targets(gt_boxes, default_boxes, variance=[0.1, 0.2]):
    """
    Encode ground truth boxes relative to default boxes
    (Used during training)
    """
    # Extract coordinates
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    
    default_cx = default_boxes[:, 0]
    default_cy = default_boxes[:, 1]
    default_w = default_boxes[:, 2]
    default_h = default_boxes[:, 3]
    
    # Encode offsets
    tx = (gt_cx - default_cx) / (default_w * variance[0])
    ty = (gt_cy - default_cy) / (default_h * variance[0])
    tw = torch.log(gt_w / default_w) / variance[1]
    th = torch.log(gt_h / default_h) / variance[1]
    
    return torch.stack([tx, ty, tw, th], dim=1)
```

### Classification Processing
SSD uses softmax for multi-class classification:

```python
def process_ssd_classification(conf_preds, background_threshold=0.01):
    """
    Process SSD classification predictions
    
    Args:
        conf_preds: [N, C] - raw class logits (including background)
    """
    
    # Apply softmax to get probabilities
    class_probs = F.softmax(conf_preds, dim=1)
    
    # Extract background and foreground probabilities
    background_prob = class_probs[:, 0]  # Class 0 is background
    foreground_probs = class_probs[:, 1:]  # Classes 1+ are foreground
    
    # Get maximum foreground class
    max_fg_prob, max_fg_class = torch.max(foreground_probs, dim=1)
    
    # Filter out background predictions
    foreground_mask = background_prob < (1 - background_threshold)
    
    return {
        'foreground_mask': foreground_mask,
        'class_probs': foreground_probs,
        'max_class_prob': max_fg_prob,
        'predicted_class': max_fg_class + 1,  # +1 because we removed background
        'background_prob': background_prob
    }
```

## Post-Processing Pipeline

### Multi-Scale Detection Processing
```python
def postprocess_ssd_outputs(loc_outputs, conf_outputs, default_boxes, 
                           conf_threshold=0.01, nms_threshold=0.45, 
                           top_k=200, keep_top_k=100):
    """
    Post-process SSD outputs from all scales
    """
    batch_size = loc_outputs[0].shape[0]
    all_detections = []
    
    for batch_idx in range(batch_size):
        batch_detections = []
        
        # Process each scale
        start_idx = 0
        for scale_idx, (loc_out, conf_out) in enumerate(zip(loc_outputs, conf_outputs)):
            # Extract predictions for this batch and scale
            batch_loc = loc_out[batch_idx]    # [C_loc, H, W]
            batch_conf = conf_out[batch_idx]  # [C_conf, H, W]
            
            # Get number of default boxes for this scale
            num_boxes = batch_loc.shape[0] // 4
            end_idx = start_idx + num_boxes * batch_loc.shape[1] * batch_loc.shape[2]
            
            # Get default boxes for this scale
            scale_default_boxes = default_boxes[start_idx:end_idx]
            
            # Reshape predictions
            loc_reshaped = batch_loc.view(num_boxes, 4, -1).permute(2, 0, 1).contiguous()
            loc_reshaped = loc_reshaped.view(-1, 4)  # [H*W*num_boxes, 4]
            
            conf_reshaped = batch_conf.view(num_boxes, -1, batch_conf.shape[1], batch_conf.shape[2])
            conf_reshaped = conf_reshaped.permute(2, 3, 0, 1).contiguous()
            conf_reshaped = conf_reshaped.view(-1, conf_reshaped.shape[-1])  # [H*W*num_boxes, num_classes]
            
            # Decode predictions
            decoded_boxes = decode_ssd_predictions(
                loc_reshaped, conf_reshaped, scale_default_boxes
            )
            
            # Process classifications
            class_results = process_ssd_classification(conf_reshaped)
            
            # Filter confident predictions
            confident_mask = class_results['max_class_prob'] > conf_threshold
            if confident_mask.sum() > 0:
                confident_boxes = decoded_boxes[confident_mask]
                confident_scores = class_results['max_class_prob'][confident_mask]
                confident_classes = class_results['predicted_class'][confident_mask]
                
                # Store detections
                for box, score, cls in zip(confident_boxes, confident_scores, confident_classes):
                    detection = {
                        'bbox': box.tolist(),
                        'confidence': score.item(),
                        'class_id': cls.item(),
                        'scale': scale_idx
                    }
                    batch_detections.append(detection)
            
            start_idx = end_idx
        
        # Apply NMS to combined detections
        if len(batch_detections) > 0:
            # Sort by confidence and take top_k
            batch_detections.sort(key=lambda x: x['confidence'], reverse=True)
            batch_detections = batch_detections[:top_k]
            
            # Apply NMS
            final_detections = apply_ssd_nms(batch_detections, nms_threshold)
            final_detections = final_detections[:keep_top_k]
            
            all_detections.append(final_detections)
        else:
            all_detections.append([])
    
    return all_detections

def apply_ssd_nms(detections, nms_threshold=0.45):
    """Apply class-specific NMS for SSD detections"""
    if len(detections) == 0:
        return []
    
    # Group by class
    class_detections = {}
    for det in detections:
        class_id = det['class_id']
        if class_id not in class_detections:
            class_detections[class_id] = []
        class_detections[class_id].append(det)
    
    # Apply NMS per class
    final_detections = []
    for class_id, class_dets in class_detections.items():
        # Convert to tensors for efficient NMS
        boxes = torch.tensor([det['bbox'] for det in class_dets])
        scores = torch.tensor([det['confidence'] for det in class_dets])
        
        # Apply NMS
        keep_indices = torchvision.ops.nms(boxes, scores, nms_threshold)
        
        # Keep selected detections
        for idx in keep_indices:
            final_detections.append(class_dets[idx])
    
    return final_detections
```

## Framework Implementations

### TensorFlow/Keras Implementation
```python
import tensorflow as tf

class SSDPostProcessor:
    def __init__(self, num_classes=21, input_size=300):
        self.num_classes = num_classes
        self.input_size = input_size
        self.default_boxes = self.generate_default_boxes()
        
    def generate_default_boxes(self):
        """Generate SSD default boxes"""
        config = {
            'feature_maps': [38, 19, 10, 5, 3, 1],
            'steps': [8, 16, 32, 64, 100, 300],
            'sizes': [30, 60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        }
        
        default_boxes = []
        
        for k, (fmap, step, min_size, max_size, ars) in enumerate(zip(
            config['feature_maps'], config['steps'], 
            config['sizes'][:-1], config['sizes'][1:],
            config['aspect_ratios']
        )):
            for i in range(fmap):
                for j in range(fmap):
                    cx = (j + 0.5) * step / self.input_size
                    cy = (i + 0.5) * step / self.input_size
                    
                    # Box with aspect ratio 1
                    s = min_size / self.input_size
                    default_boxes.append([cx, cy, s, s])
                    
                    # Box with scale sqrt(min_size * max_size)
                    s = tf.sqrt(min_size * max_size) / self.input_size
                    default_boxes.append([cx, cy, s, s])
                    
                    # Boxes with other aspect ratios
                    for ar in ars:
                        s = min_size / self.input_size
                        w = s * tf.sqrt(ar)
                        h = s / tf.sqrt(ar)
                        default_boxes.append([cx, cy, w, h])
                        default_boxes.append([cx, cy, h, w])
        
        return tf.constant(default_boxes, dtype=tf.float32)
    
    @tf.function
    def decode_predictions(self, loc_preds, conf_preds):
        """Decode SSD predictions"""
        
        # Decode box coordinates
        variance = [0.1, 0.2]
        
        # Center coordinates
        predicted_cx = (loc_preds[:, 0] * variance[0] * self.default_boxes[:, 2] + 
                       self.default_boxes[:, 0])
        predicted_cy = (loc_preds[:, 1] * variance[0] * self.default_boxes[:, 3] + 
                       self.default_boxes[:, 1])
        
        # Width and height
        predicted_w = tf.exp(loc_preds[:, 2] * variance[1]) * self.default_boxes[:, 2]
        predicted_h = tf.exp(loc_preds[:, 3] * variance[1]) * self.default_boxes[:, 3]
        
        # Convert to corner coordinates
        x1 = predicted_cx - predicted_w / 2
        y1 = predicted_cy - predicted_h / 2
        x2 = predicted_cx + predicted_w / 2
        y2 = predicted_cy + predicted_h / 2
        
        decoded_boxes = tf.stack([x1, y1, x2, y2], axis=1)
        
        # Process confidences
        class_probs = tf.nn.softmax(conf_preds, axis=1)
        
        return decoded_boxes, class_probs
    
    def postprocess(self, predictions, conf_threshold=0.5, nms_threshold=0.45):
        """Post-process SSD model outputs"""
        
        loc_preds = predictions['localization']  # [batch, num_boxes, 4]
        conf_preds = predictions['classification']  # [batch, num_boxes, num_classes]
        
        batch_results = []
        
        for batch_idx in range(loc_preds.shape[0]):
            # Decode predictions for this batch
            boxes, probs = self.decode_predictions(
                loc_preds[batch_idx], 
                conf_preds[batch_idx]
            )
            
            # Filter background predictions
            foreground_probs = probs[:, 1:]  # Remove background class
            max_probs = tf.reduce_max(foreground_probs, axis=1)
            max_classes = tf.argmax(foreground_probs, axis=1) + 1  # +1 for removed background
            
            # Filter by confidence
            confident_mask = max_probs > conf_threshold
            confident_boxes = tf.boolean_mask(boxes, confident_mask)
            confident_scores = tf.boolean_mask(max_probs, confident_mask)
            confident_classes = tf.boolean_mask(max_classes, confident_mask)
            
            # Apply NMS
            selected_indices = tf.image.non_max_suppression(
                confident_boxes,
                confident_scores,
                max_output_size=100,
                iou_threshold=nms_threshold,
                score_threshold=conf_threshold
            )
            
            final_boxes = tf.gather(confident_boxes, selected_indices)
            final_scores = tf.gather(confident_scores, selected_indices)
            final_classes = tf.gather(confident_classes, selected_indices)
            
            batch_results.append({
                'boxes': final_boxes.numpy(),
                'scores': final_scores.numpy(),
                'classes': final_classes.numpy()
            })
        
        return batch_results
```

### PyTorch Implementation
```python
import torch
import torch.nn.functional as F

class SSDDetector:
    def __init__(self, num_classes=21, input_size=300, device='cuda'):
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = device
        self.default_boxes = self.generate_default_boxes().to(device)
        
    def generate_default_boxes(self):
        """Generate default boxes for SSD"""
        fmap_dims = [38, 19, 10, 5, 3, 1]
        steps = [8, 16, 32, 64, 100, 300]
        sizes = [30, 60, 111, 162, 213, 264, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        
        default_boxes = []
        
        for k, (fmap, step, min_size, max_size, ars) in enumerate(zip(
            fmap_dims, steps, sizes[:-1], sizes[1:], aspect_ratios
        )):
            for i in range(fmap):
                for j in range(fmap):
                    # Center point
                    cx = (j + 0.5) * step / self.input_size
                    cy = (i + 0.5) * step / self.input_size
                    
                    # Box with aspect ratio 1
                    s = min_size / self.input_size
                    default_boxes.append([cx, cy, s, s])
                    
                    # Additional box for aspect ratio 1
                    s = (min_size * max_size) ** 0.5 / self.input_size
                    default_boxes.append([cx, cy, s, s])
                    
                    # Boxes with other aspect ratios
                    for ar in ars:
                        s = min_size / self.input_size
                        w = s * (ar ** 0.5)
                        h = s / (ar ** 0.5)
                        default_boxes.append([cx, cy, w, h])
                        default_boxes.append([cx, cy, h, w])
        
        return torch.tensor(default_boxes, dtype=torch.float32)
    
    def decode_predictions(self, loc, conf, variance=[0.1, 0.2]):
        """Decode SSD predictions to absolute coordinates"""
        
        # Decode box coordinates
        boxes = torch.cat([
            loc[:, :2] * variance[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2],
            torch.exp(loc[:, 2:] * variance[1]) * self.default_boxes[:, 2:]
        ], dim=1)
        
        # Convert to corner format
        boxes[:, :2] -= boxes[:, 2:] / 2  # cx, cy -> x1, y1
        boxes[:, 2:] += boxes[:, :2]      # w, h -> x2, y2
        
        return boxes
    
    def detect(self, loc_preds, conf_preds, conf_threshold=0.01, nms_threshold=0.45):
        """
        Perform detection using SSD predictions
        
        Args:
            loc_preds: [batch, num_boxes, 4] - localization predictions
            conf_preds: [batch, num_boxes, num_classes] - confidence predictions
        """
        batch_size = loc_preds.size(0)
        num_classes = conf_preds.size(2)
        
        # Apply softmax to get class probabilities
        conf_preds = F.softmax(conf_preds, dim=2)
        
        batch_results = []
        
        for batch_idx in range(batch_size):
            # Decode boxes for this batch
            decoded_boxes = self.decode_predictions(loc_preds[batch_idx])
            conf_scores = conf_preds[batch_idx]
            
            batch_detections = []
            
            # Process each class (skip background class 0)
            for class_idx in range(1, num_classes):
                class_scores = conf_scores[:, class_idx]
                
                # Filter by confidence threshold
                score_mask = class_scores > conf_threshold
                if not score_mask.any():
                    continue
                
                # Get boxes and scores for this class
                class_boxes = decoded_boxes[score_mask]
                class_scores_filtered = class_scores[score_mask]
                
                # Apply NMS
                keep_indices = torchvision.ops.nms(
                    class_boxes, 
                    class_scores_filtered, 
                    nms_threshold
                )
                
                # Store detections
                for idx in keep_indices:
                    detection = {
                        'bbox': class_boxes[idx].tolist(),
                        'confidence': class_scores_filtered[idx].item(),
                        'class_id': class_idx
                    }
                    batch_detections.append(detection)
            
            batch_results.append(batch_detections)
        
        return batch_results
```

### ONNX Runtime Implementation
```python
import onnxruntime as ort
import numpy as np

class SSDONNX:
    def __init__(self, model_path, num_classes=21):
        self.session = ort.InferenceSession(model_path)
        self.num_classes = num_classes
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Generate default boxes
        self.default_boxes = self.generate_default_boxes()
        
    def generate_default_boxes(self):
        """Generate SSD default boxes"""
        # Implementation similar to PyTorch version
        # ... (code omitted for brevity)
        pass
    
    def preprocess(self, image):
        """Preprocess image for SSD"""
        # Resize to 300x300
        resized = cv2.resize(image, (300, 300))
        
        # Normalize (SSD typically uses ImageNet normalization)
        mean = np.array([123.68, 116.78, 103.94])
        normalized = resized.astype(np.float32) - mean
        
        # Transpose to CHW and add batch dimension
        preprocessed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(preprocessed, axis=0)
        
        return batched
    
    def inference(self, image):
        """Run SSD inference"""
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        return outputs
    
    def postprocess(self, outputs, conf_threshold=0.5, nms_threshold=0.45):
        """Post-process SSD outputs"""
        # outputs typically contain [localization, classification]
        loc_preds = outputs[0]  # [batch, num_boxes, 4]
        conf_preds = outputs[1]  # [batch, num_boxes, num_classes]
        
        batch_size = loc_preds.shape[0]
        results = []
        
        for batch_idx in range(batch_size):
            # Decode predictions
            boxes = self.decode_boxes(loc_preds[batch_idx])
            
            # Apply softmax to confidences
            conf_softmax = self.softmax(conf_preds[batch_idx], axis=1)
            
            detections = []
            
            # Process each class (skip background)
            for class_idx in range(1, self.num_classes):
                class_scores = conf_softmax[:, class_idx]
                
                # Filter by confidence
                valid_indices = np.where(class_scores > conf_threshold)[0]
                
                if len(valid_indices) == 0:
                    continue
                
                valid_boxes = boxes[valid_indices]
                valid_scores = class_scores[valid_indices]
                
                # Apply NMS
                keep_indices = self.nms(valid_boxes, valid_scores, nms_threshold)
                
                # Store results
                for idx in keep_indices:
                    detections.append({
                        'bbox': valid_boxes[idx].tolist(),
                        'confidence': float(valid_scores[idx]),
                        'class_id': class_idx
                    })
            
            results.append(detections)
        
        return results
    
    def decode_boxes(self, loc_preds, variance=[0.1, 0.2]):
        """Decode localization predictions"""
        # Center coordinates
        pred_cx = loc_preds[:, 0] * variance[0] * self.default_boxes[:, 2] + self.default_boxes[:, 0]
        pred_cy = loc_preds[:, 1] * variance[0] * self.default_boxes[:, 3] + self.default_boxes[:, 1]
        
        # Width and height
        pred_w = np.exp(loc_preds[:, 2] * variance[1]) * self.default_boxes[:, 2]
        pred_h = np.exp(loc_preds[:, 3] * variance[1]) * self.default_boxes[:, 3]
        
        # Convert to corner coordinates
        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2
        
        return np.stack([x1, y1, x2, y2], axis=1)
    
    def softmax(self, x, axis=1):
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def nms(self, boxes, scores, threshold):
        """Non-maximum suppression"""
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]
            
            ious = self.calculate_iou_vectorized(current_box, remaining_boxes)
            
            # Keep boxes with IoU below threshold
            below_threshold = ious < threshold
            indices = indices[1:][below_threshold]
        
        return keep
    
    def calculate_iou_vectorized(self, box, boxes):
        """Calculate IoU between one box and multiple boxes"""
        x1, y1, x2, y2 = box
        x1s, y1s, x2s, y2s = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Intersection coordinates
        xi1 = np.maximum(x1, x1s)
        yi1 = np.maximum(y1, y1s)
        xi2 = np.minimum(x2, x2s)
        yi2 = np.minimum(y2, y2s)
        
        # Intersection area
        intersection = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
        
        # Union area
        box_area = (x2 - x1) * (y2 - y1)
        boxes_area = (x2s - x1s) * (y2s - y1s)
        union = box_area + boxes_area - intersection
        
        return intersection / union
```

## Model Variants and Performance

### SSD Variants
```python
ssd_variants = {
    'SSD300': {
        'input_size': '300x300',
        'backbone': 'VGG-16',
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'total_boxes': 8732,
        'coco_map': '25.1%',
        'parameters': '26.3M'
    },
    'SSD512': {
        'input_size': '512x512',
        'backbone': 'VGG-16',
        'feature_maps': [64, 32, 16, 8, 4, 2, 1],
        'total_boxes': 24564,
        'coco_map': '28.8%',
        'parameters': '26.3M'
    },
    'SSD-MobileNet': {
        'input_size': '300x300',
        'backbone': 'MobileNet-v1',
        'feature_maps': [19, 10, 5, 3, 2, 1],
        'total_boxes': 1917,
        'coco_map': '19.3%',
        'parameters': '6.8M'
    }
}
```

### Deployment Considerations
```python
ssd_deployment = {
    'advantages': [
        'Single forward pass',
        'Multiple scale detection',
        'Good speed-accuracy tradeoff',
        'Stable training'
    ],
    'disadvantages': [
        'Complex post-processing',
        'Many hyperparameters',
        'Sensitivity to default box design',
        'Difficulty with small objects'
    ],
    'optimization_strategies': [
        'TensorRT optimization',
        'Post-processing acceleration',
        'Default box pruning',
        'Feature map reduction'
    ]
}
```

## Key Characteristics of SSD

### Architectural Advantages
- **Multi-scale detection**: Different feature map resolutions capture various object sizes
- **Default box design**: Systematic anchor generation across scales and aspect ratios
- **Single forward pass**: Efficient inference compared to two-stage detectors
- **Flexible backbone**: Can use different feature extractors (VGG, ResNet, MobileNet)

### Performance Characteristics
- **Good accuracy**: Competitive with Faster R-CNN at the time
- **Real-time capable**: Faster than two-stage methods
- **Scale-aware**: Better multi-scale object detection than earlier single-stage methods
- **Stable training**: More stable than early YOLO versions

### Implementation Complexity
- **Separate outputs**: Classification and localization handled separately
- **Complex post-processing**: Multi-scale NMS and coordinate decoding
- **Hyperparameter sensitivity**: Many parameters to tune (default box sizes, aspect ratios, thresholds)
- **Memory requirements**: Multiple feature maps and large number of default boxes