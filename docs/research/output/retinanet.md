# RetinaNet Output Format Analysis

## Overview
RetinaNet is a single-stage object detection architecture that introduced the **Focal Loss** to address the class imbalance problem between foreground and background classes. It combines a **Feature Pyramid Network (FPN)** backbone with dense prediction heads, achieving excellent accuracy while maintaining single-stage efficiency. The model uses an anchor-based approach with 9 anchors per spatial location across multiple FPN levels.

## Output Tensor Structure

### Feature Pyramid Network (FPN) Architecture
RetinaNet uses **5 FPN levels** (P3-P7) for multi-scale detection:

### Tensor Dimensions
For RetinaNet with ResNet-50 FPN backbone and 800×800 input:

**Classification Outputs (per FPN level):**
```
P3 (stride 8):  [1, A×K, H/8, W/8]    # [1, 9×80, 100, 100]
P4 (stride 16): [1, A×K, H/16, W/16]  # [1, 9×80, 50, 50]
P5 (stride 32): [1, A×K, H/32, W/32]  # [1, 9×80, 25, 25]
P6 (stride 64): [1, A×K, H/64, W/64]  # [1, 9×80, 13, 13]  (generated from P5)
P7 (stride 128):[1, A×K, H/128, W/128] # [1, 9×80, 7, 7]   (generated from P6)

Where:
- A = 9 anchors per location (3 scales × 3 aspect ratios)
- K = number of classes (80 for COCO)
- H, W = input image height, width
```

**Regression Outputs (per FPN level):**
```
P3 (stride 8):  [1, A×4, H/8, W/8]    # [1, 36, 100, 100]
P4 (stride 16): [1, A×4, H/16, W/16]  # [1, 36, 50, 50]
P5 (stride 32): [1, A×4, H/32, W/32]  # [1, 36, 25, 25]
P6 (stride 64): [1, A×4, H/64, W/64]  # [1, 36, 13, 13]
P7 (stride 128):[1, A×4, H/128, W/128] # [1, 36, 7, 7]

Where:
- A×4 = 36 coordinates (9 anchors × 4 coordinates: dx, dy, dw, dh)
```

### Complete Architecture Overview
```python
class RetinaNetOutputStructure:
    """RetinaNet output tensor structure"""
    
    def __init__(self, input_size=(800, 800), num_classes=80):
        self.input_h, self.input_w = input_size
        self.num_classes = num_classes
        self.num_anchors = 9  # 3 scales × 3 aspect ratios
        
        # FPN levels configuration
        self.fpn_config = {
            'P3': {'stride': 8,   'size': (self.input_h//8, self.input_w//8)},
            'P4': {'stride': 16,  'size': (self.input_h//16, self.input_w//16)},
            'P5': {'stride': 32,  'size': (self.input_h//32, self.input_w//32)},
            'P6': {'stride': 64,  'size': (self.input_h//64, self.input_w//64)},
            'P7': {'stride': 128, 'size': (self.input_h//128, self.input_w//128)}
        }
        
        # Anchor scales and ratios
        self.anchor_scales = [2**0, 2**(1/3), 2**(2/3)]  # [1.0, 1.26, 1.587]
        self.anchor_ratios = [0.5, 1.0, 2.0]             # [1:2, 1:1, 2:1]
        
    def get_output_shapes(self):
        """Get output tensor shapes for all FPN levels"""
        classification_shapes = {}
        regression_shapes = {}
        
        for level, config in self.fpn_config.items():
            h, w = config['size']
            
            # Classification: [batch, anchors×classes, height, width]
            classification_shapes[level] = (1, self.num_anchors * self.num_classes, h, w)
            
            # Regression: [batch, anchors×4, height, width]
            regression_shapes[level] = (1, self.num_anchors * 4, h, w)
        
        return classification_shapes, regression_shapes
    
    def total_predictions(self):
        """Calculate total number of predictions"""
        total = 0
        for level, config in self.fpn_config.items():
            h, w = config['size']
            total += h * w * self.num_anchors
        
        return total

# Example usage
retinanet = RetinaNetOutputStructure()
cls_shapes, reg_shapes = retinanet.get_output_shapes()

print("Classification shapes:")
for level, shape in cls_shapes.items():
    print(f"  {level}: {shape}")

print("\nRegression shapes:")
for level, shape in reg_shapes.items():
    print(f"  {level}: {shape}")

print(f"\nTotal predictions: {retinanet.total_predictions()}")
```

**Expected Output:**
```
Classification shapes:
  P3: (1, 720, 100, 100)    # 9×80 = 720
  P4: (1, 720, 50, 50)
  P5: (1, 720, 25, 25)
  P6: (1, 720, 13, 13)
  P7: (1, 720, 7, 7)

Regression shapes:
  P3: (1, 36, 100, 100)     # 9×4 = 36
  P4: (1, 36, 50, 50)
  P5: (1, 36, 25, 25)
  P6: (1, 36, 13, 13)
  P7: (1, 36, 7, 7)

Total predictions: 67995    # Sum of all anchor boxes across FPN levels
```

## Feature Pyramid Network (FPN) Implementation

### FPN Backbone Architecture
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for RetinaNet"""
    
    def __init__(self, backbone_channels=[256, 512, 1024, 2048], fpn_channels=256):
        super().__init__()
        self.backbone_channels = backbone_channels
        self.fpn_channels = fpn_channels
        
        # Lateral connections (1x1 convs to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, fpn_channels, 1)
            for channels in backbone_channels
        ])
        
        # Feature pyramid convs (3x3 convs for final features)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
            for _ in backbone_channels
        ])
        
        # Extra layers for P6, P7
        self.p6_conv = nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(fpn_channels, fpn_channels, 3, stride=2, padding=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize FPN weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, backbone_features):
        """
        Forward pass through FPN
        
        Args:
            backbone_features: List of feature maps from backbone
                              [C2, C3, C4, C5] from ResNet
        
        Returns:
            fpn_features: List of FPN feature maps [P3, P4, P5, P6, P7]
        """
        # Apply lateral connections
        laterals = [
            lateral_conv(backbone_features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway with lateral connections
        # Start from the highest level (C5 -> P5)
        fpn_features = [laterals[-1]]  # P5
        
        # Process P4, P3, P2 (top-down)
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample higher level feature
            upsampled = F.interpolate(
                fpn_features[0], 
                size=laterals[i].shape[-2:], 
                mode='nearest'
            )
            
            # Add lateral connection
            merged = laterals[i] + upsampled
            fpn_features.insert(0, merged)
        
        # Apply 3x3 convs to get final features
        final_features = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            final_features.append(fpn_conv(fpn_features[i]))
        
        # Generate P6 and P7
        p6 = self.p6_conv(final_features[-1])  # P6 from P5
        p7 = self.p7_conv(F.relu(p6))          # P7 from P6
        
        final_features.extend([p6, p7])
        
        return final_features  # [P3, P4, P5, P6, P7]
```

### Dense Prediction Heads
```python
class RetinaNetHead(nn.Module):
    """RetinaNet classification and regression heads"""
    
    def __init__(self, num_classes=80, num_anchors=9, fpn_channels=256, 
                 num_layers=4, prior_prob=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification subnet
        cls_layers = []
        for _ in range(num_layers):
            cls_layers.extend([
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
        self.cls_subnet = nn.Sequential(*cls_layers)
        
        # Regression subnet  
        reg_layers = []
        for _ in range(num_layers):
            reg_layers.extend([
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
        self.reg_subnet = nn.Sequential(*reg_layers)
        
        # Output layers
        self.cls_head = nn.Conv2d(
            fpn_channels, num_anchors * num_classes, 3, padding=1
        )
        self.reg_head = nn.Conv2d(
            fpn_channels, num_anchors * 4, 3, padding=1
        )
        
        # Initialize weights
        self._init_weights(prior_prob)
    
    def _init_weights(self, prior_prob):
        """Initialize head weights with focal loss bias"""
        # Standard initialization for conv layers
        for layer in [self.cls_subnet, self.reg_subnet]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, 0)
        
        # Special initialization for output layers
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.normal_(self.reg_head.weight, std=0.01)
        
        # Initialize classification bias for focal loss
        bias_value = -torch.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_value)
        nn.init.constant_(self.reg_head.bias, 0)
    
    def forward(self, fpn_features):
        """
        Forward pass through RetinaNet heads
        
        Args:
            fpn_features: List of FPN feature maps [P3, P4, P5, P6, P7]
        
        Returns:
            cls_outputs: List of classification tensors
            reg_outputs: List of regression tensors
        """
        cls_outputs = []
        reg_outputs = []
        
        for feature in fpn_features:
            # Classification subnet
            cls_feature = self.cls_subnet(feature)
            cls_output = self.cls_head(cls_feature)
            
            # Regression subnet
            reg_feature = self.reg_subnet(feature)
            reg_output = self.reg_head(reg_feature)
            
            cls_outputs.append(cls_output)
            reg_outputs.append(reg_output)
        
        return cls_outputs, reg_outputs
```

## Anchor Generation System

### RetinaNet Anchor Configuration
```python
class RetinaNetAnchorGenerator:
    """Generate anchors for RetinaNet across FPN levels"""
    
    def __init__(self, scales=[2**0, 2**(1/3), 2**(2/3)], 
                 ratios=[0.5, 1.0, 2.0], base_size=4):
        self.scales = scales      # [1.0, 1.26, 1.587]
        self.ratios = ratios      # [0.5, 1.0, 2.0]
        self.base_size = base_size
        self.num_anchors = len(scales) * len(ratios)  # 9
        
        # FPN strides and sizes
        self.fpn_strides = [8, 16, 32, 64, 128]    # P3, P4, P5, P6, P7
        self.anchor_sizes = [32, 64, 128, 256, 512]  # Base anchor sizes per level
    
    def generate_base_anchors(self, anchor_size):
        """Generate base anchors for a single FPN level"""
        anchors = []
        
        for scale in self.scales:
            for ratio in self.ratios:
                # Calculate anchor dimensions
                area = (anchor_size * scale) ** 2
                w = torch.sqrt(area / ratio)
                h = w * ratio
                
                # Create anchor (center format)
                anchor = torch.tensor([-w/2, -h/2, w/2, h/2])
                anchors.append(anchor)
        
        return torch.stack(anchors)
    
    def generate_anchors_single_level(self, height, width, stride, anchor_size):
        """Generate all anchors for a single FPN level"""
        # Generate base anchors
        base_anchors = self.generate_base_anchors(anchor_size)
        
        # Create grid of centers
        shifts_x = torch.arange(0, width) * stride + stride // 2
        shifts_y = torch.arange(0, height) * stride + stride // 2
        
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        
        # Create shifts for all anchor positions
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
        
        # Apply shifts to base anchors
        anchors = base_anchors[None, :, :] + shifts[:, None, :]
        anchors = anchors.reshape(-1, 4)
        
        return anchors
    
    def generate_all_anchors(self, feature_map_sizes):
        """
        Generate anchors for all FPN levels
        
        Args:
            feature_map_sizes: List of (height, width) for each FPN level
        
        Returns:
            all_anchors: Concatenated anchors from all levels
            anchor_level_indices: Indices indicating which level each anchor belongs to
        """
        all_anchors = []
        anchor_level_indices = []
        
        for level, (height, width) in enumerate(feature_map_sizes):
            stride = self.fpn_strides[level]
            anchor_size = self.anchor_sizes[level]
            
            # Generate anchors for this level
            level_anchors = self.generate_anchors_single_level(
                height, width, stride, anchor_size
            )
            
            all_anchors.append(level_anchors)
            
            # Track which level each anchor belongs to
            level_indices = torch.full((len(level_anchors),), level, dtype=torch.long)
            anchor_level_indices.append(level_indices)
        
        # Concatenate all anchors
        all_anchors = torch.cat(all_anchors, dim=0)
        anchor_level_indices = torch.cat(anchor_level_indices, dim=0)
        
        return all_anchors, anchor_level_indices
    
    def visualize_anchor_distribution(self, feature_map_sizes):
        """Visualize anchor distribution across FPN levels"""
        import matplotlib.pyplot as plt
        
        level_names = ['P3', 'P4', 'P5', 'P6', 'P7']
        anchor_counts = []
        anchor_areas = []
        
        for level, (height, width) in enumerate(feature_map_sizes):
            stride = self.fpn_strides[level]
            anchor_size = self.anchor_sizes[level]
            
            num_anchors = height * width * self.num_anchors
            anchor_counts.append(num_anchors)
            
            # Calculate representative anchor area
            avg_area = (anchor_size ** 2) * torch.tensor(self.scales).mean()
            anchor_areas.append(avg_area.item())
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Anchor count distribution
        ax1.bar(level_names, anchor_counts)
        ax1.set_ylabel('Number of Anchors')
        ax1.set_title('Anchor Distribution Across FPN Levels')
        
        # Anchor size distribution
        ax2.plot(level_names, anchor_areas, 'o-')
        ax2.set_ylabel('Average Anchor Area')
        ax2.set_title('Anchor Size Across FPN Levels')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        return fig

# Example usage
anchor_generator = RetinaNetAnchorGenerator()

# For 800x800 input
feature_map_sizes = [(100, 100), (50, 50), (25, 25), (13, 13), (7, 7)]
all_anchors, level_indices = anchor_generator.generate_all_anchors(feature_map_sizes)

print(f"Total anchors generated: {len(all_anchors)}")
print(f"Anchor tensor shape: {all_anchors.shape}")

# Count anchors per level
for level in range(5):
    count = (level_indices == level).sum().item()
    level_name = f'P{level + 3}'
    print(f"{level_name}: {count} anchors")
```

## Mathematical Transformations

### Coordinate Encoding/Decoding
```python
def encode_retinanet_targets(gt_boxes, anchors):
    """
    Encode ground truth boxes relative to anchors
    
    Args:
        gt_boxes: [N, 4] ground truth boxes (x1, y1, x2, y2)
        anchors: [M, 4] anchor boxes (x1, y1, x2, y2)
    
    Returns:
        encoded_boxes: [M, 4] encoded targets (dx, dy, dw, dh)
    """
    # Convert to center format
    gt_ctr_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_ctr_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    
    anchor_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    
    # Encode
    dx = (gt_ctr_x - anchor_ctr_x) / anchor_w
    dy = (gt_ctr_y - anchor_ctr_y) / anchor_h
    dw = torch.log(gt_w / anchor_w)
    dh = torch.log(gt_h / anchor_h)
    
    encoded_boxes = torch.stack([dx, dy, dw, dh], dim=1)
    return encoded_boxes

def decode_retinanet_predictions(pred_deltas, anchors):
    """
    Decode RetinaNet regression predictions
    
    Args:
        pred_deltas: [N, 4] predicted deltas (dx, dy, dw, dh)
        anchors: [N, 4] anchor boxes (x1, y1, x2, y2)
    
    Returns:
        decoded_boxes: [N, 4] decoded boxes (x1, y1, x2, y2)
    """
    # Convert anchors to center format
    anchor_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    
    # Extract deltas
    dx = pred_deltas[:, 0]
    dy = pred_deltas[:, 1]
    dw = pred_deltas[:, 2]
    dh = pred_deltas[:, 3]
    
    # Decode center coordinates
    pred_ctr_x = dx * anchor_w + anchor_ctr_x
    pred_ctr_y = dy * anchor_h + anchor_ctr_y
    
    # Decode width and height (with clipping for stability)
    pred_w = torch.exp(torch.clamp(dw, max=4.135)) * anchor_w  # exp(4.135) ≈ 62
    pred_h = torch.exp(torch.clamp(dh, max=4.135)) * anchor_h
    
    # Convert back to corner format
    x1 = pred_ctr_x - pred_w / 2
    y1 = pred_ctr_y - pred_h / 2
    x2 = pred_ctr_x + pred_w / 2
    y2 = pred_ctr_y + pred_h / 2
    
    decoded_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return decoded_boxes
```

### Focal Loss Implementation
```python
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in RetinaNet
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [N, C] class predictions (logits)
            targets: [N] target class indices
        """
        # Convert to probabilities
        p = torch.sigmoid(predictions)
        
        # Create binary targets
        ce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # Calculate p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

## Post-Processing Pipeline

### Complete RetinaNet Inference Pipeline
```python
class RetinaNetPostProcessor:
    """Complete post-processing pipeline for RetinaNet"""
    
    def __init__(self, num_classes=80, conf_threshold=0.05, 
                 nms_threshold=0.5, max_detections_per_img=100,
                 fpn_strides=[8, 16, 32, 64, 128]):
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections_per_img = max_detections_per_img
        self.fpn_strides = fpn_strides
        
        # Generate anchor generator
        self.anchor_generator = RetinaNetAnchorGenerator()
    
    def postprocess_batch(self, cls_outputs, reg_outputs, image_sizes, 
                         original_image_sizes=None):
        """
        Post-process a batch of RetinaNet outputs
        
        Args:
            cls_outputs: List of classification tensors for each FPN level
            reg_outputs: List of regression tensors for each FPN level  
            image_sizes: List of (height, width) for resized images
            original_image_sizes: List of (height, width) for original images
        
        Returns:
            batch_detections: List of detection results for each image
        """
        batch_size = cls_outputs[0].shape[0]
        batch_detections = []
        
        # Generate anchors for this batch
        feature_map_sizes = [(cls.shape[-2], cls.shape[-1]) for cls in cls_outputs]
        all_anchors, _ = self.anchor_generator.generate_all_anchors(feature_map_sizes)
        
        for batch_idx in range(batch_size):
            # Extract predictions for this image
            image_cls_outputs = [cls[batch_idx] for cls in cls_outputs]
            image_reg_outputs = [reg[batch_idx] for reg in reg_outputs]
            
            # Process single image
            detections = self.postprocess_single_image(
                image_cls_outputs, image_reg_outputs, all_anchors,
                image_sizes[batch_idx] if isinstance(image_sizes[0], tuple) else image_sizes,
                original_image_sizes[batch_idx] if original_image_sizes else None
            )
            
            batch_detections.append(detections)
        
        return batch_detections
    
    def postprocess_single_image(self, cls_outputs, reg_outputs, anchors,
                                image_size, original_image_size=None):
        """Post-process predictions for a single image"""
        
        # Flatten and concatenate all FPN level outputs
        flattened_cls, flattened_reg = self.flatten_fpn_outputs(
            cls_outputs, reg_outputs
        )
        
        # Apply sigmoid to classification outputs
        cls_probs = torch.sigmoid(flattened_cls)
        
        # Decode bounding boxes
        decoded_boxes = decode_retinanet_predictions(flattened_reg, anchors)
        
        # Clip boxes to image boundaries
        decoded_boxes = self.clip_boxes(decoded_boxes, image_size)
        
        # Apply confidence threshold and NMS
        detections = self.apply_nms(cls_probs, decoded_boxes)
        
        # Scale boxes to original image size if needed
        if original_image_size is not None:
            detections = self.scale_boxes(detections, image_size, original_image_size)
        
        return detections
    
    def flatten_fpn_outputs(self, cls_outputs, reg_outputs):
        """Flatten and concatenate FPN level outputs"""
        flattened_cls = []
        flattened_reg = []
        
        for cls_out, reg_out in zip(cls_outputs, reg_outputs):
            # cls_out: [A*C, H, W] -> [H*W*A, C]
            # reg_out: [A*4, H, W] -> [H*W*A, 4]
            
            A = 9  # number of anchors per location
            C = self.num_classes
            H, W = cls_out.shape[-2:]
            
            # Reshape classification output
            cls_reshaped = cls_out.view(A, C, H, W).permute(2, 3, 0, 1)
            cls_reshaped = cls_reshaped.contiguous().view(-1, C)
            flattened_cls.append(cls_reshaped)
            
            # Reshape regression output
            reg_reshaped = reg_out.view(A, 4, H, W).permute(2, 3, 0, 1)
            reg_reshaped = reg_reshaped.contiguous().view(-1, 4)
            flattened_reg.append(reg_reshaped)
        
        # Concatenate all levels
        flattened_cls = torch.cat(flattened_cls, dim=0)
        flattened_reg = torch.cat(flattened_reg, dim=0)
        
        return flattened_cls, flattened_reg
    
    def clip_boxes(self, boxes, image_size):
        """Clip boxes to image boundaries"""
        height, width = image_size
        
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)   # x1
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)  # y1
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)   # x2
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)  # y2
        
        return boxes
    
    def apply_nms(self, cls_probs, boxes):
        """Apply NMS per class"""
        from torchvision.ops import nms
        
        detections = []
        
        for class_id in range(self.num_classes):
            # Get scores for this class
            class_scores = cls_probs[:, class_id]
            
            # Filter by confidence threshold
            confident_indices = class_scores > self.conf_threshold
            
            if not confident_indices.any():
                continue
            
            class_boxes = boxes[confident_indices]
            class_scores = class_scores[confident_indices]
            
            # Apply NMS
            keep_indices = nms(class_boxes, class_scores, self.nms_threshold)
            
            # Create detections
            for idx in keep_indices:
                detection = {
                    'bbox': class_boxes[idx].tolist(),
                    'score': class_scores[idx].item(),
                    'class_id': class_id,
                    'class_name': f'class_{class_id}'  # Replace with actual class names
                }
                detections.append(detection)
        
        # Sort by confidence and limit detections
        detections.sort(key=lambda x: x['score'], reverse=True)
        detections = detections[:self.max_detections_per_img]
        
        return detections
    
    def scale_boxes(self, detections, resized_size, original_size):
        """Scale boxes from resized image to original image size"""
        resized_h, resized_w = resized_size
        original_h, original_w = original_size
        
        scale_x = original_w / resized_w
        scale_y = original_h / resized_h
        
        for detection in detections:
            bbox = detection['bbox']
            bbox[0] *= scale_x  # x1
            bbox[1] *= scale_y  # y1
            bbox[2] *= scale_x  # x2
            bbox[3] *= scale_y  # y2
            detection['bbox'] = bbox
        
        return detections
```

### Efficient Multi-Scale NMS
```python
def efficient_multi_scale_nms(detections, iou_threshold=0.5, score_threshold=0.05):
    """
    Efficient NMS across multiple scales and classes
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for NMS
        score_threshold: Score threshold for filtering
    
    Returns:
        filtered_detections: NMS-filtered detections
    """
    import numpy as np
    
    if len(detections) == 0:
        return []
    
    # Convert to arrays for efficient processing
    boxes = np.array([det['bbox'] for det in detections])
    scores = np.array([det['score'] for det in detections])
    classes = np.array([det['class_id'] for det in detections])
    
    # Filter by score threshold
    keep_mask = scores >= score_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    classes = classes[keep_mask]
    valid_detections = [det for i, det in enumerate(detections) if keep_mask[i]]
    
    if len(boxes) == 0:
        return []
    
    # Sort by scores
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Take highest scoring detection
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current_idx:current_idx+1]  # [1, 4]
        remaining_boxes = boxes[sorted_indices[1:]]     # [N-1, 4]
        
        ious = calculate_iou_vectorized(current_box, remaining_boxes)[0]
        
        # Filter out overlapping boxes of same class
        current_class = classes[current_idx]
        remaining_classes = classes[sorted_indices[1:]]
        
        # Keep boxes that either:
        # 1. Have IoU < threshold, OR
        # 2. Are of different class
        keep_mask = (ious < iou_threshold) | (remaining_classes != current_class)
        
        sorted_indices = sorted_indices[1:][keep_mask]
    
    # Return filtered detections
    filtered_detections = [valid_detections[i] for i in keep_indices]
    return filtered_detections

def calculate_iou_vectorized(boxes1, boxes2):
    """Vectorized IoU calculation"""
    import numpy as np
    
    # boxes1: [N, 4], boxes2: [M, 4]
    # Returns: [N, M] IoU matrix
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Broadcast for intersection calculation
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])  # [N, M]
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])  # [N, M]
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])  # [N, M]
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])  # [N, M]
    
    # Calculate intersection
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union
    union = area1[:, None] + area2[None, :] - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-8)
    
    return iou
```

## Framework Implementations

### PyTorch Implementation
```python
import torch
import torch.nn as nn
import torchvision

class RetinaNetPyTorch:
    """Complete RetinaNet implementation in PyTorch"""
    
    def __init__(self, num_classes=80, pretrained=True):
        self.num_classes = num_classes
        
        # Load pre-trained RetinaNet
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(
            pretrained=pretrained,
            num_classes=num_classes + 1  # +1 for background
        )
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Post-processor
        self.post_processor = RetinaNetPostProcessor(num_classes=num_classes)
    
    def preprocess(self, images):
        """Preprocess images for RetinaNet"""
        if not isinstance(images, list):
            images = [images]
        
        processed_images = []
        original_sizes = []
        
        for image in images:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            # Store original size
            original_sizes.append(image.shape[-2:])
            
            # Normalize (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
            
            processed_images.append(image)
        
        return processed_images, original_sizes
    
    def inference(self, images):
        """Run inference on images"""
        # Preprocess
        processed_images, original_sizes = self.preprocess(images)
        
        # Move to device
        processed_images = [img.to(self.device) for img in processed_images]
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(processed_images)
        
        return predictions, original_sizes
    
    def postprocess(self, predictions, original_sizes, 
                   conf_threshold=0.7, nms_threshold=0.5):
        """Post-process predictions"""
        detections = []
        
        for pred, orig_size in zip(predictions, original_sizes):
            # PyTorch RetinaNet returns processed detections
            boxes = pred['boxes'].cpu()
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()
            
            # Filter by confidence
            keep_mask = scores >= conf_threshold
            
            if keep_mask.sum() == 0:
                detections.append([])
                continue
            
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            labels = labels[keep_mask]
            
            # Convert to detection format
            image_detections = []
            for box, score, label in zip(boxes, scores, labels):
                detection = {
                    'bbox': box.tolist(),
                    'score': score.item(),
                    'class_id': label.item() - 1,  # Convert from 1-based to 0-based
                    'class_name': f'class_{label.item() - 1}'
                }
                image_detections.append(detection)
            
            detections.append(image_detections)
        
        return detections
    
    def predict(self, images, conf_threshold=0.7, nms_threshold=0.5):
        """Complete prediction pipeline"""
        predictions, original_sizes = self.inference(images)
        detections = self.postprocess(
            predictions, original_sizes, conf_threshold, nms_threshold
        )
        return detections

# Example usage
retinanet = RetinaNetPyTorch(num_classes=80)

# Load test image
import cv2
image = cv2.imread('test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run prediction
detections = retinanet.predict([image], conf_threshold=0.5)

print(f"Detected {len(detections[0])} objects")
for detection in detections[0][:5]:  # Show first 5 detections
    print(f"Class {detection['class_id']}: {detection['score']:.3f} at {detection['bbox']}")
```

### TensorFlow Implementation
```python
import tensorflow as tf

class RetinaNetTensorFlow:
    """RetinaNet implementation using TensorFlow"""
    
    def __init__(self, model_path, num_classes=80):
        self.num_classes = num_classes
        
        # Load model
        self.model = tf.saved_model.load(model_path)
        
        # Get model signature
        self.infer = self.model.signatures['serving_default']
        
        # Generate anchor generator
        self.anchor_generator = RetinaNetAnchorGenerator()
        
    def preprocess(self, image):
        """Preprocess image for TensorFlow model"""
        # Resize and normalize
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        
        # Resize to model input size
        resized = tf.image.resize(image, [800, 800])
        
        # Normalize to [0, 1] and then ImageNet normalization
        normalized = tf.cast(resized, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        return normalized
    
    def inference(self, image):
        """Run inference using TensorFlow model"""
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.infer(input_tensor)
        
        return outputs
    
    def postprocess_tf_outputs(self, outputs, conf_threshold=0.5):
        """Post-process TensorFlow model outputs"""
        # Assuming outputs contain detection_boxes, detection_scores, etc.
        boxes = outputs['detection_boxes'][0]       # [N, 4]
        scores = outputs['detection_scores'][0]     # [N]
        classes = outputs['detection_classes'][0]   # [N]
        num_detections = int(outputs['num_detections'][0])
        
        detections = []
        
        for i in range(num_detections):
            if scores[i] >= conf_threshold:
                # Convert normalized coordinates to pixel coordinates
                box = boxes[i] * 800  # Assuming 800x800 input
                
                detection = {
                    'bbox': [
                        float(box[1]),  # x1
                        float(box[0]),  # y1
                        float(box[3]),  # x2
                        float(box[2])   # y2
                    ],
                    'score': float(scores[i]),
                    'class_id': int(classes[i]) - 1,  # Convert to 0-based
                    'class_name': f'class_{int(classes[i]) - 1}'
                }
                detections.append(detection)
        
        return detections
```

### ONNX Runtime Implementation
```python
import onnxruntime as ort
import numpy as np

class RetinaNetONNX:
    """RetinaNet inference using ONNX Runtime"""
    
    def __init__(self, model_path, num_classes=80):
        # Create inference session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=session_options,
            providers=providers
        )
        
        self.num_classes = num_classes
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Initialize post-processor
        self.post_processor = RetinaNetPostProcessor(num_classes=num_classes)
        
    def preprocess(self, image):
        """Preprocess image for ONNX model"""
        # Resize to 800x800
        resized = cv2.resize(image, (800, 800), interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB and normalize
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Transpose to CHW and add batch dimension
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def inference(self, image):
        """Run inference using ONNX Runtime"""
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(
            self.output_names, 
            {self.input_name: input_tensor}
        )
        
        return outputs
    
    def postprocess_onnx_outputs(self, outputs, conf_threshold=0.5):
        """Post-process ONNX model outputs"""
        # Outputs format depends on the specific ONNX model
        # Common formats:
        # 1. [boxes, scores, classes] - post-processed
        # 2. [cls_outputs_P3, cls_outputs_P4, ..., reg_outputs_P3, ...] - raw
        
        if len(outputs) == 3:  # Post-processed outputs
            boxes, scores, classes = outputs
            
            detections = []
            for box, score, cls in zip(boxes[0], scores[0], classes[0]):
                if score >= conf_threshold:
                    detection = {
                        'bbox': box.tolist(),
                        'score': float(score),
                        'class_id': int(cls),
                        'class_name': f'class_{int(cls)}'
                    }
                    detections.append(detection)
            
            return detections
        
        else:  # Raw FPN outputs - need full post-processing
            # Split outputs into classification and regression
            num_levels = len(outputs) // 2
            cls_outputs = outputs[:num_levels]
            reg_outputs = outputs[num_levels:]
            
            # Convert to torch tensors for post-processing
            cls_tensors = [torch.from_numpy(cls) for cls in cls_outputs]
            reg_tensors = [torch.from_numpy(reg) for reg in reg_outputs]
            
            # Use post-processor
            detections = self.post_processor.postprocess_batch(
                cls_tensors, reg_tensors, 
                image_sizes=[(800, 800)],
                original_image_sizes=None
            )
            
            return detections[0]  # Return first (and only) image's detections
    
    def predict(self, image, conf_threshold=0.5):
        """Complete prediction pipeline"""
        outputs = self.inference(image)
        detections = self.postprocess_onnx_outputs(outputs, conf_threshold)
        return detections

# Example usage
retinanet_onnx = RetinaNetONNX('retinanet.onnx', num_classes=80)

# Load and predict
image = cv2.imread('test_image.jpg')
detections = retinanet_onnx.predict(image, conf_threshold=0.5)

print(f"ONNX model detected {len(detections)} objects")
for detection in detections[:5]:
    print(f"Class {detection['class_id']}: {detection['score']:.3f}")
```

## Performance Characteristics

### Model Specifications
```python
retinanet_specs = {
    'architecture': 'RetinaNet with ResNet-50 FPN',
    'input_size': '800×800 (variable)',
    'backbone': 'ResNet-50 + Feature Pyramid Network',
    'fpn_levels': 5,  # P3-P7
    'total_anchors': 67995,  # For 800×800 input
    'parameters': '37.7M',
    'model_size': '145MB',
    'flops': '239G',
    'coco_map': '36.5%',
    'inference_speed': {
        'v100_gpu': '~20ms',
        'rtx_3080': '~15ms',
        'cpu_i9': '~800ms'
    },
    'memory_usage': {
        'gpu_inference': '~2.5GB',
        'training': '~8GB'
    }
}
```

### Key Advantages
- **Focal Loss**: Effectively handles class imbalance
- **Feature Pyramid Network**: Excellent multi-scale detection
- **Dense Predictions**: 67K+ anchors provide comprehensive coverage
- **Single-stage Efficiency**: Faster than two-stage detectors
- **Strong Performance**: Competitive accuracy with simpler architecture

### Use Cases
- **General Object Detection**: Excellent for COCO-style datasets
- **Multi-scale Objects**: Strong performance on objects of different sizes
- **Real-time Applications**: Good balance of speed and accuracy
- **Research**: Popular baseline for detection research
- **Production Systems**: Reliable performance in deployed systems