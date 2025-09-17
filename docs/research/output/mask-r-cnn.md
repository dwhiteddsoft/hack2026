# Mask R-CNN Output Format Analysis

## Overview
Mask R-CNN is a **two-stage instance segmentation** architecture that extends Faster R-CNN by adding a **mask prediction branch**. It consists of a **Region Proposal Network (RPN)** for generating object proposals, followed by a **region-based CNN (R-CNN)** that performs classification, bounding box regression, and **pixel-level mask segmentation**. The model outputs bounding boxes, class predictions, confidence scores, and binary masks for each detected instance.

## Output Tensor Structure

### Two-Stage Architecture Overview
Mask R-CNN produces outputs at two distinct stages:

1. **Stage 1 (RPN)**: Object proposals and objectness scores
2. **Stage 2 (R-CNN)**: Final detections with masks

### Stage 1: Region Proposal Network (RPN) Outputs

**RPN Classification Outputs (Objectness):**
```python
# For each FPN level (P2, P3, P4, P5, P6)
rpn_cls_outputs = {
    'P2': [1, 3, H/4, W/4],      # [1, 3, 200, 200]  (800×800 input)
    'P3': [1, 3, H/8, W/8],      # [1, 3, 100, 100]
    'P4': [1, 3, H/16, W/16],    # [1, 3, 50, 50]
    'P5': [1, 3, H/32, W/32],    # [1, 3, 25, 25]
    'P6': [1, 3, H/64, W/64]     # [1, 3, 13, 13]
}

# Where:
# - 3 = number of anchors per spatial location
# - Values represent objectness scores (object vs background)
```

**RPN Regression Outputs (Box Deltas):**
```python
rpn_reg_outputs = {
    'P2': [1, 12, H/4, W/4],     # [1, 12, 200, 200]  (3 anchors × 4 coordinates)
    'P3': [1, 12, H/8, W/8],     # [1, 12, 100, 100]
    'P4': [1, 12, H/16, W/16],   # [1, 12, 50, 50]
    'P5': [1, 12, H/32, W/32],   # [1, 12, 25, 25]
    'P6': [1, 12, H/64, W/64]    # [1, 12, 13, 13]
}

# Where:
# - 12 = 3 anchors × 4 regression parameters (dx, dy, dw, dh)
```

**Generated Proposals:**
```python
proposals_output = {
    'proposals': [N, 4],          # [1000, 4] - Top-N proposal boxes (x1, y1, x2, y2)
    'objectness_scores': [N],     # [1000] - Objectness confidence scores
    'proposal_levels': [N]        # [1000] - FPN level assignment for each proposal
}
```

### Stage 2: R-CNN Head Outputs

**ROI Feature Extraction:**
```python
roi_features = [N, 256, 7, 7]    # [1000, 256, 7, 7] - RoIAlign extracted features
```

**Classification Head Output:**
```python
rcnn_cls_output = [N, num_classes + 1]  # [1000, 81] for COCO (80 classes + background)
```

**Regression Head Output:**
```python
rcnn_reg_output = [N, num_classes * 4]  # [1000, 320] for COCO (80 classes × 4 coordinates)
```

**Mask Head Output:**
```python
mask_output = [N, num_classes, 28, 28]  # [1000, 80, 28, 28] - Binary masks for each class
```

### Final Detection Outputs

After post-processing (NMS, score filtering), the final outputs are:

```python
final_detections = {
    'boxes': [K, 4],           # [100, 4] - Final detection boxes (x1, y1, x2, y2)
    'scores': [K],             # [100] - Classification confidence scores
    'labels': [K],             # [100] - Predicted class labels (1-based)
    'masks': [K, H, W]         # [100, 800, 800] - Binary instance masks
}

# Where K = number of final detections after NMS (typically 100-300)
```

## Complete Architecture Implementation

### Mask R-CNN Full Model Structure
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, nms

class MaskRCNN(nn.Module):
    """Complete Mask R-CNN implementation"""
    
    def __init__(self, num_classes=80, backbone='resnet50'):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone + FPN
        self.backbone = self.build_backbone_fpn(backbone)
        
        # RPN
        self.rpn = RegionProposalNetwork(
            in_channels=256,
            num_anchors=3,
            anchor_scales=[32, 64, 128, 256, 512],
            anchor_ratios=[0.5, 1.0, 2.0]
        )
        
        # R-CNN heads
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2)
        self.box_head = BoxHead(in_channels=256, num_classes=num_classes)
        self.mask_head = MaskHead(in_channels=256, num_classes=num_classes)
        
        # Post-processing parameters
        self.rpn_pre_nms_top_n_train = 2000
        self.rpn_pre_nms_top_n_test = 1000
        self.rpn_post_nms_top_n_train = 2000
        self.rpn_post_nms_top_n_test = 1000
        self.rpn_nms_thresh = 0.7
        
        self.box_score_thresh = 0.7
        self.box_nms_thresh = 0.5
        self.box_detections_per_img = 100
        
    def build_backbone_fpn(self, backbone_name):
        """Build backbone with FPN"""
        # Implementation depends on chosen backbone
        # Return FPN feature extractor
        pass
    
    def forward(self, images, targets=None):
        """
        Forward pass through Mask R-CNN
        
        Args:
            images: List of input images
            targets: Ground truth (for training)
        
        Returns:
            During training: loss dictionary
            During inference: detection results
        """
        # Extract FPN features
        features = self.backbone(images)  # Dict of {level: feature_map}
        
        # RPN forward pass
        rpn_outputs = self.rpn(features)
        proposals, proposal_losses = self.rpn.postprocess(
            rpn_outputs, images, targets
        )
        
        # R-CNN forward pass
        detections, detector_losses = self.roi_heads(
            features, proposals, images, targets
        )
        
        if self.training:
            # Return losses for training
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses
        else:
            # Return detections for inference
            return detections
    
    def roi_heads(self, features, proposals, images, targets=None):
        """R-CNN heads processing"""
        # Extract ROI features
        roi_features = self.extract_roi_features(features, proposals)
        
        # Box head
        box_cls, box_reg = self.box_head(roi_features)
        
        # Mask head
        mask_logits = self.mask_head(roi_features)
        
        if self.training:
            # Compute losses during training
            losses = self.compute_losses(
                box_cls, box_reg, mask_logits, proposals, targets
            )
            return None, losses
        else:
            # Post-process detections
            detections = self.postprocess_detections(
                box_cls, box_reg, mask_logits, proposals, images
            )
            return detections, {}
    
    def extract_roi_features(self, features, proposals):
        """Extract ROI features using RoIAlign"""
        # Determine which FPN level to use for each proposal
        proposal_levels = self.assign_fpn_levels(proposals)
        
        roi_features = []
        
        for level, feature_map in features.items():
            # Get proposals assigned to this level
            level_mask = proposal_levels == level
            if not level_mask.any():
                continue
            
            level_proposals = proposals[level_mask]
            
            # Extract features using RoIAlign
            spatial_scale = 1.0 / (2 ** (level + 2))  # FPN spatial scales
            level_roi_features = self.roi_align(
                feature_map, level_proposals, spatial_scale
            )
            
            roi_features.append(level_roi_features)
        
        # Concatenate all ROI features
        if roi_features:
            return torch.cat(roi_features, dim=0)
        else:
            return torch.empty(0, 256, 7, 7)
    
    def assign_fpn_levels(self, proposals):
        """Assign proposals to FPN levels based on scale"""
        # Calculate proposal areas
        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        areas = widths * heights
        
        # Assign to levels based on area (following FPN paper)
        levels = torch.floor(4 + torch.log2(torch.sqrt(areas) / 224))
        levels = torch.clamp(levels, min=2, max=5)  # P2-P5
        
        return levels.int()
```

### Region Proposal Network (RPN) Implementation
```python
class RegionProposalNetwork(nn.Module):
    """Region Proposal Network for generating object proposals"""
    
    def __init__(self, in_channels=256, num_anchors=3, 
                 anchor_scales=[32, 64, 128, 256, 512],
                 anchor_ratios=[0.5, 1.0, 2.0]):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        
        # Shared convolutional layer
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        # Classification head (objectness)
        self.cls_head = nn.Conv2d(in_channels, num_anchors, 1)
        
        # Regression head (box deltas)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator(
            scales=anchor_scales, ratios=anchor_ratios
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize RPN weights"""
        for layer in [self.conv, self.cls_head, self.reg_head]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features):
        """
        Forward pass through RPN
        
        Args:
            features: Dict of FPN features {level: tensor}
        
        Returns:
            rpn_cls_outputs: Classification outputs per level
            rpn_reg_outputs: Regression outputs per level
            anchors: Generated anchors for each level
        """
        rpn_cls_outputs = {}
        rpn_reg_outputs = {}
        anchors = {}
        
        for level, feature_map in features.items():
            # Shared convolution
            rpn_features = F.relu(self.conv(feature_map))
            
            # Classification (objectness)
            cls_output = self.cls_head(rpn_features)
            rpn_cls_outputs[level] = cls_output
            
            # Regression (box deltas)
            reg_output = self.reg_head(rpn_features)
            rpn_reg_outputs[level] = reg_output
            
            # Generate anchors for this level
            level_anchors = self.anchor_generator.generate_anchors(
                feature_map.shape[-2:], level
            )
            anchors[level] = level_anchors
        
        return rpn_cls_outputs, rpn_reg_outputs, anchors
    
    def postprocess(self, rpn_outputs, images, targets=None):
        """Post-process RPN outputs to generate proposals"""
        rpn_cls_outputs, rpn_reg_outputs, anchors = rpn_outputs
        
        proposals = []
        proposal_losses = {}
        
        for batch_idx in range(len(images)):
            # Process each image in the batch
            image_proposals = self.postprocess_single_image(
                rpn_cls_outputs, rpn_reg_outputs, anchors, 
                images[batch_idx], batch_idx
            )
            proposals.append(image_proposals)
        
        if self.training and targets is not None:
            # Compute RPN losses during training
            proposal_losses = self.compute_rpn_losses(
                rpn_cls_outputs, rpn_reg_outputs, anchors, targets
            )
        
        return proposals, proposal_losses
    
    def postprocess_single_image(self, cls_outputs, reg_outputs, anchors, 
                                image, batch_idx):
        """Post-process RPN outputs for a single image"""
        all_proposals = []
        all_scores = []
        
        for level in cls_outputs.keys():
            # Extract outputs for this level and batch
            cls_out = cls_outputs[level][batch_idx]  # [3, H, W]
            reg_out = reg_outputs[level][batch_idx]  # [12, H, W]
            level_anchors = anchors[level]           # [H*W*3, 4]
            
            # Reshape outputs
            H, W = cls_out.shape[-2:]
            num_anchors_per_location = 3
            
            # Reshape classification scores
            cls_scores = cls_out.view(num_anchors_per_location, H, W)
            cls_scores = cls_scores.permute(1, 2, 0).contiguous().view(-1)
            cls_scores = torch.sigmoid(cls_scores)
            
            # Reshape regression deltas
            reg_deltas = reg_out.view(num_anchors_per_location, 4, H, W)
            reg_deltas = reg_deltas.permute(2, 3, 0, 1).contiguous().view(-1, 4)
            
            # Decode proposals
            proposals = self.decode_proposals(reg_deltas, level_anchors)
            
            # Clip to image boundaries
            proposals = self.clip_proposals(proposals, image.shape[-2:])
            
            # Filter small boxes
            proposals, keep_mask = self.filter_small_boxes(proposals, min_size=1)
            cls_scores = cls_scores[keep_mask]
            
            all_proposals.append(proposals)
            all_scores.append(cls_scores)
        
        # Concatenate all levels
        all_proposals = torch.cat(all_proposals, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        # Sort by score and take top proposals
        top_k = self.rpn_pre_nms_top_n_test if not self.training else self.rpn_pre_nms_top_n_train
        
        if len(all_scores) > top_k:
            top_indices = torch.topk(all_scores, top_k)[1]
            all_proposals = all_proposals[top_indices]
            all_scores = all_scores[top_indices]
        
        # Apply NMS
        keep_indices = nms(all_proposals, all_scores, self.rpn_nms_thresh)
        
        # Limit final proposals
        post_nms_k = self.rpn_post_nms_top_n_test if not self.training else self.rpn_post_nms_top_n_train
        keep_indices = keep_indices[:post_nms_k]
        
        final_proposals = all_proposals[keep_indices]
        final_scores = all_scores[keep_indices]
        
        return {'proposals': final_proposals, 'scores': final_scores}
    
    def decode_proposals(self, reg_deltas, anchors):
        """Decode regression deltas to get proposal boxes"""
        # Convert anchors to center format
        anchor_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2
        anchor_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        
        # Extract deltas
        dx = reg_deltas[:, 0]
        dy = reg_deltas[:, 1]
        dw = reg_deltas[:, 2]
        dh = reg_deltas[:, 3]
        
        # Decode
        pred_ctr_x = dx * anchor_w + anchor_ctr_x
        pred_ctr_y = dy * anchor_h + anchor_ctr_y
        pred_w = torch.exp(torch.clamp(dw, max=4.135)) * anchor_w
        pred_h = torch.exp(torch.clamp(dh, max=4.135)) * anchor_h
        
        # Convert to corner format
        x1 = pred_ctr_x - pred_w / 2
        y1 = pred_ctr_y - pred_h / 2
        x2 = pred_ctr_x + pred_w / 2
        y2 = pred_ctr_y + pred_h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def clip_proposals(self, proposals, image_size):
        """Clip proposals to image boundaries"""
        height, width = image_size
        
        proposals[:, 0] = torch.clamp(proposals[:, 0], min=0, max=width)
        proposals[:, 1] = torch.clamp(proposals[:, 1], min=0, max=height)
        proposals[:, 2] = torch.clamp(proposals[:, 2], min=0, max=width)
        proposals[:, 3] = torch.clamp(proposals[:, 3], min=0, max=height)
        
        return proposals
    
    def filter_small_boxes(self, proposals, min_size):
        """Filter out small boxes"""
        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        
        keep_mask = (widths >= min_size) & (heights >= min_size)
        
        return proposals[keep_mask], keep_mask
```

### Box and Mask Heads Implementation
```python
class BoxHead(nn.Module):
    """Box classification and regression head"""
    
    def __init__(self, in_channels=256, num_classes=80, hidden_dim=1024):
        super().__init__()
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.fc1 = nn.Linear(in_channels * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.reg_head = nn.Linear(hidden_dim, num_classes * 4)  # Per-class regression
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize box head weights"""
        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, 0)
        
        nn.init.normal_(self.reg_head.weight, std=0.001)
        nn.init.constant_(self.reg_head.bias, 0)
    
    def forward(self, roi_features):
        """
        Forward pass through box head
        
        Args:
            roi_features: [N, 256, 7, 7] ROI features
        
        Returns:
            cls_scores: [N, num_classes + 1] classification scores
            reg_deltas: [N, num_classes * 4] regression deltas
        """
        # Flatten ROI features
        flattened = roi_features.flatten(start_dim=1)  # [N, 256*7*7]
        
        # Feature extraction
        x = F.relu(self.fc1(flattened))
        x = F.relu(self.fc2(x))
        
        # Classification and regression
        cls_scores = self.cls_head(x)
        reg_deltas = self.reg_head(x)
        
        return cls_scores, reg_deltas

class MaskHead(nn.Module):
    """Mask prediction head"""
    
    def __init__(self, in_channels=256, num_classes=80, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        
        # Convolutional layers for mask prediction
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        # Deconvolution for upsampling
        self.deconv = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2)
        
        # Final mask prediction layer
        self.mask_predictor = nn.Conv2d(hidden_dim, num_classes, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize mask head weights"""
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.deconv]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        
        nn.init.kaiming_normal_(self.mask_predictor.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.mask_predictor.bias, 0)
    
    def forward(self, roi_features):
        """
        Forward pass through mask head
        
        Args:
            roi_features: [N, 256, 7, 7] ROI features
        
        Returns:
            mask_logits: [N, num_classes, 14, 14] mask predictions
        """
        x = F.relu(self.conv1(roi_features))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Upsample to 14x14
        x = F.relu(self.deconv(x))
        
        # Predict masks
        mask_logits = self.mask_predictor(x)
        
        return mask_logits
```

## Mathematical Transformations

### RPN Target Assignment and Loss
```python
class RPNTargetAssigner:
    """Assign targets for RPN training"""
    
    def __init__(self, positive_threshold=0.7, negative_threshold=0.3,
                 batch_size=256, positive_fraction=0.5):
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction
    
    def assign_targets(self, anchors, gt_boxes):
        """
        Assign classification and regression targets to anchors
        
        Args:
            anchors: [N, 4] anchor boxes
            gt_boxes: [M, 4] ground truth boxes
        
        Returns:
            labels: [N] classification targets (-1: ignore, 0: negative, 1: positive)
            reg_targets: [N, 4] regression targets
        """
        # Calculate IoU matrix
        iou_matrix = self.calculate_iou_matrix(anchors, gt_boxes)
        
        # Find best GT for each anchor and best anchor for each GT
        max_iou_per_anchor, best_gt_per_anchor = iou_matrix.max(dim=1)
        max_iou_per_gt, best_anchor_per_gt = iou_matrix.max(dim=0)
        
        # Initialize labels (ignore by default)
        labels = torch.full((len(anchors),), -1, dtype=torch.long)
        
        # Assign positive labels
        positive_mask = max_iou_per_anchor >= self.positive_threshold
        labels[positive_mask] = 1
        
        # Ensure each GT has at least one positive anchor
        labels[best_anchor_per_gt] = 1
        
        # Assign negative labels
        negative_mask = max_iou_per_anchor < self.negative_threshold
        labels[negative_mask] = 0
        
        # Sample positive and negative anchors for training
        labels = self.sample_anchors(labels)
        
        # Compute regression targets for positive anchors
        reg_targets = torch.zeros((len(anchors), 4))
        if positive_mask.any():
            positive_anchors = anchors[positive_mask]
            positive_gt = gt_boxes[best_gt_per_anchor[positive_mask]]
            reg_targets[positive_mask] = self.encode_regression_targets(
                positive_anchors, positive_gt
            )
        
        return labels, reg_targets
    
    def sample_anchors(self, labels):
        """Sample positive and negative anchors for training"""
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        num_positive = positive_mask.sum().item()
        num_negative = negative_mask.sum().item()
        
        # Determine number of positive samples
        target_positive = int(self.batch_size * self.positive_fraction)
        num_positive_samples = min(num_positive, target_positive)
        
        # Determine number of negative samples
        num_negative_samples = min(
            num_negative, 
            self.batch_size - num_positive_samples
        )
        
        # Sample positive anchors
        if num_positive > num_positive_samples:
            positive_indices = torch.where(positive_mask)[0]
            selected_positive = torch.randperm(num_positive)[:num_positive_samples]
            disable_indices = positive_indices[selected_positive[num_positive_samples:]]
            labels[disable_indices] = -1
        
        # Sample negative anchors
        if num_negative > num_negative_samples:
            negative_indices = torch.where(negative_mask)[0]
            selected_negative = torch.randperm(num_negative)[:num_negative_samples]
            disable_indices = negative_indices[selected_negative[num_negative_samples:]]
            labels[disable_indices] = -1
        
        return labels
    
    def calculate_iou_matrix(self, anchors, gt_boxes):
        """Calculate IoU matrix between anchors and GT boxes"""
        # Implementation similar to previous examples
        # Returns [N, M] IoU matrix
        pass
    
    def encode_regression_targets(self, anchors, gt_boxes):
        """Encode regression targets"""
        # Convert to center format
        anchor_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2
        anchor_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        
        gt_ctr_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_ctr_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        
        # Encode
        dx = (gt_ctr_x - anchor_ctr_x) / anchor_w
        dy = (gt_ctr_y - anchor_ctr_y) / anchor_h
        dw = torch.log(gt_w / anchor_w)
        dh = torch.log(gt_h / anchor_h)
        
        return torch.stack([dx, dy, dw, dh], dim=1)

def rpn_loss(cls_outputs, reg_outputs, targets):
    """Compute RPN classification and regression losses"""
    # Classification loss (binary cross entropy)
    cls_targets = targets['labels']
    valid_mask = cls_targets != -1
    
    cls_loss = F.binary_cross_entropy_with_logits(
        cls_outputs[valid_mask],
        cls_targets[valid_mask].float()
    )
    
    # Regression loss (smooth L1)
    reg_targets = targets['reg_targets']
    positive_mask = cls_targets == 1
    
    if positive_mask.sum() > 0:
        reg_loss = F.smooth_l1_loss(
            reg_outputs[positive_mask],
            reg_targets[positive_mask],
            reduction='mean'
        )
    else:
        reg_loss = torch.tensor(0.0, device=cls_outputs.device)
    
    return cls_loss, reg_loss
```

### Mask Loss and Target Generation
```python
def mask_loss(mask_logits, targets, positive_proposals):
    """
    Compute mask prediction loss
    
    Args:
        mask_logits: [N, num_classes, 28, 28] predicted masks
        targets: Ground truth masks and labels
        positive_proposals: Proposals assigned to positive samples
    
    Returns:
        mask_loss: Computed mask loss
    """
    if len(positive_proposals) == 0:
        return torch.tensor(0.0, device=mask_logits.device)
    
    # Get ground truth masks and labels for positive proposals
    gt_masks = targets['masks']  # [N, H, W]
    gt_labels = targets['labels']  # [N]
    
    # Extract mask predictions for correct classes
    N = mask_logits.shape[0]
    indices = torch.arange(N, device=mask_logits.device)
    class_specific_masks = mask_logits[indices, gt_labels]  # [N, 28, 28]
    
    # Resize ground truth masks to match prediction size
    gt_masks_resized = F.interpolate(
        gt_masks.float().unsqueeze(1),  # [N, 1, H, W]
        size=(28, 28),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # [N, 28, 28]
    
    # Compute binary cross entropy loss
    mask_loss = F.binary_cross_entropy_with_logits(
        class_specific_masks,
        gt_masks_resized,
        reduction='mean'
    )
    
    return mask_loss

def generate_mask_targets(proposals, gt_boxes, gt_masks, gt_labels):
    """Generate mask targets for training"""
    # Match proposals to ground truth
    iou_matrix = calculate_iou_matrix(proposals, gt_boxes)
    max_iou, matched_gt_idx = iou_matrix.max(dim=1)
    
    # Assign labels based on IoU threshold
    labels = torch.zeros(len(proposals), dtype=torch.long)
    positive_mask = max_iou >= 0.5
    labels[positive_mask] = gt_labels[matched_gt_idx[positive_mask]]
    
    # Extract corresponding masks for positive proposals
    mask_targets = []
    for i, (proposal, is_positive, gt_idx) in enumerate(
        zip(proposals, positive_mask, matched_gt_idx)
    ):
        if is_positive:
            # Crop and resize GT mask to proposal region
            gt_mask = gt_masks[gt_idx]  # [H, W]
            
            # Extract ROI from mask
            x1, y1, x2, y2 = proposal.int()
            roi_mask = gt_mask[y1:y2+1, x1:x2+1]
            
            # Resize to standard size (28x28)
            if roi_mask.numel() > 0:
                roi_mask_resized = F.interpolate(
                    roi_mask.float().unsqueeze(0).unsqueeze(0),
                    size=(28, 28),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            else:
                roi_mask_resized = torch.zeros(28, 28)
            
            mask_targets.append(roi_mask_resized)
        else:
            # Background proposal
            mask_targets.append(torch.zeros(28, 28))
    
    return torch.stack(mask_targets), labels
```

## Post-Processing Pipeline

### Complete Mask R-CNN Post-Processing
```python
class MaskRCNNPostProcessor:
    """Post-processing for Mask R-CNN outputs"""
    
    def __init__(self, score_threshold=0.7, nms_threshold=0.5, 
                 max_detections_per_img=100, mask_threshold=0.5):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections_per_img = max_detections_per_img
        self.mask_threshold = mask_threshold
    
    def postprocess_batch(self, box_cls, box_reg, mask_logits, proposals, image_sizes):
        """Post-process a batch of Mask R-CNN outputs"""
        batch_detections = []
        
        for batch_idx in range(len(proposals)):
            # Extract outputs for this image
            image_box_cls = box_cls[batch_idx]
            image_box_reg = box_reg[batch_idx]
            image_mask_logits = mask_logits[batch_idx]
            image_proposals = proposals[batch_idx]['proposals']
            image_size = image_sizes[batch_idx]
            
            # Post-process single image
            detections = self.postprocess_single_image(
                image_box_cls, image_box_reg, image_mask_logits,
                image_proposals, image_size
            )
            
            batch_detections.append(detections)
        
        return batch_detections
    
    def postprocess_single_image(self, box_cls, box_reg, mask_logits, 
                                proposals, image_size):
        """Post-process outputs for a single image"""
        # Apply softmax to classification scores
        scores = F.softmax(box_cls, dim=1)  # [N, num_classes + 1]
        
        # Remove background scores
        scores = scores[:, 1:]  # [N, num_classes]
        
        # Decode box regression for each class
        decoded_boxes = self.decode_box_regression(box_reg, proposals)
        
        # Clip boxes to image boundaries
        decoded_boxes = self.clip_boxes(decoded_boxes, image_size)
        
        # Apply score threshold and NMS
        detections = self.apply_nms_per_class(scores, decoded_boxes, mask_logits)
        
        # Process masks for final detections
        final_detections = self.process_masks(detections, image_size)
        
        return final_detections
    
    def decode_box_regression(self, box_reg, proposals):
        """Decode per-class box regression"""
        # box_reg: [N, num_classes * 4]
        # proposals: [N, 4]
        
        num_proposals, num_classes_x4 = box_reg.shape
        num_classes = num_classes_x4 // 4
        
        # Reshape to [N, num_classes, 4]
        box_reg = box_reg.view(num_proposals, num_classes, 4)
        
        # Decode for each class
        decoded_boxes = torch.zeros_like(box_reg)
        
        for class_idx in range(num_classes):
            class_deltas = box_reg[:, class_idx, :]  # [N, 4]
            
            # Decode using proposals as anchors
            decoded_class_boxes = self.decode_single_class(class_deltas, proposals)
            decoded_boxes[:, class_idx, :] = decoded_class_boxes
        
        return decoded_boxes
    
    def decode_single_class(self, deltas, proposals):
        """Decode regression deltas for a single class"""
        # Convert proposals to center format
        prop_ctr_x = (proposals[:, 0] + proposals[:, 2]) / 2
        prop_ctr_y = (proposals[:, 1] + proposals[:, 3]) / 2
        prop_w = proposals[:, 2] - proposals[:, 0]
        prop_h = proposals[:, 3] - proposals[:, 1]
        
        # Extract deltas
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]
        
        # Decode
        pred_ctr_x = dx * prop_w + prop_ctr_x
        pred_ctr_y = dy * prop_h + prop_ctr_y
        pred_w = torch.exp(torch.clamp(dw, max=4.135)) * prop_w
        pred_h = torch.exp(torch.clamp(dh, max=4.135)) * prop_h
        
        # Convert to corner format
        x1 = pred_ctr_x - pred_w / 2
        y1 = pred_ctr_y - pred_h / 2
        x2 = pred_ctr_x + pred_w / 2
        y2 = pred_ctr_y + pred_h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def clip_boxes(self, boxes, image_size):
        """Clip boxes to image boundaries"""
        height, width = image_size
        
        # boxes: [N, num_classes, 4]
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0, max=width)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0, max=height)
        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], min=0, max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], min=0, max=height)
        
        return boxes
    
    def apply_nms_per_class(self, scores, boxes, mask_logits):
        """Apply NMS per class"""
        num_proposals, num_classes = scores.shape
        detections = []
        
        for class_idx in range(num_classes):
            # Get scores and boxes for this class
            class_scores = scores[:, class_idx]
            class_boxes = boxes[:, class_idx, :]
            
            # Filter by score threshold
            valid_mask = class_scores >= self.score_threshold
            
            if not valid_mask.any():
                continue
            
            valid_scores = class_scores[valid_mask]
            valid_boxes = class_boxes[valid_mask]
            valid_mask_logits = mask_logits[valid_mask]
            valid_indices = torch.where(valid_mask)[0]
            
            # Apply NMS
            keep_indices = nms(valid_boxes, valid_scores, self.nms_threshold)
            
            # Store detections
            for keep_idx in keep_indices:
                detection = {
                    'bbox': valid_boxes[keep_idx].tolist(),
                    'score': valid_scores[keep_idx].item(),
                    'class_id': class_idx,
                    'mask_logits': valid_mask_logits[keep_idx],  # [num_classes, 28, 28]
                    'proposal_idx': valid_indices[keep_idx].item()
                }
                detections.append(detection)
        
        # Sort by score and limit detections
        detections.sort(key=lambda x: x['score'], reverse=True)
        detections = detections[:self.max_detections_per_img]
        
        return detections
    
    def process_masks(self, detections, image_size):
        """Process and resize masks for final output"""
        final_detections = []
        
        for detection in detections:
            # Extract class-specific mask
            class_id = detection['class_id']
            mask_logits = detection['mask_logits'][class_id]  # [28, 28]
            
            # Convert to probability
            mask_prob = torch.sigmoid(mask_logits)
            
            # Resize mask to bounding box size
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Resize mask to box size
            box_w = max(x2 - x1, 1)
            box_h = max(y2 - y1, 1)
            
            resized_mask = F.interpolate(
                mask_prob.unsqueeze(0).unsqueeze(0),
                size=(box_h, box_w),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Create full image mask
            full_mask = torch.zeros(image_size, dtype=torch.float32)
            if resized_mask.numel() > 0:
                full_mask[y1:y2, x1:x2] = resized_mask
            
            # Apply threshold to get binary mask
            binary_mask = (full_mask >= self.mask_threshold).float()
            
            # Create final detection
            final_detection = {
                'bbox': bbox,
                'score': detection['score'],
                'class_id': class_id,
                'class_name': f'class_{class_id}',
                'mask': binary_mask.numpy()
            }
            
            final_detections.append(final_detection)
        
        return final_detections
```

## Framework Implementations

### PyTorch Implementation
```python
import torchvision

class MaskRCNNPyTorch:
    """Mask R-CNN implementation using PyTorch/torchvision"""
    
    def __init__(self, num_classes=80, pretrained=True):
        self.num_classes = num_classes
        
        # Load pre-trained Mask R-CNN
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=pretrained,
            num_classes=num_classes + 1  # +1 for background
        )
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # COCO class names (for visualization)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess(self, images):
        """Preprocess images for Mask R-CNN"""
        if not isinstance(images, list):
            images = [images]
        
        processed_images = []
        original_sizes = []
        
        for image in images:
            if isinstance(image, np.ndarray):
                # Convert numpy array to tensor
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
                   score_threshold=0.7, mask_threshold=0.5):
        """Post-process predictions"""
        detections = []
        
        for pred, orig_size in zip(predictions, original_sizes):
            # Extract outputs
            boxes = pred['boxes'].cpu()
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()
            masks = pred['masks'].cpu()
            
            # Filter by score threshold
            keep_mask = scores >= score_threshold
            
            if keep_mask.sum() == 0:
                detections.append([])
                continue
            
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            labels = labels[keep_mask]
            masks = masks[keep_mask]
            
            # Process masks
            binary_masks = (masks.squeeze(1) >= mask_threshold).float()
            
            # Create detection results
            image_detections = []
            for box, score, label, mask in zip(boxes, scores, labels, binary_masks):
                detection = {
                    'bbox': box.tolist(),
                    'score': score.item(),
                    'class_id': label.item() - 1,  # Convert from 1-based to 0-based
                    'class_name': self.class_names[label.item() - 1],
                    'mask': mask.numpy()
                }
                image_detections.append(detection)
            
            detections.append(image_detections)
        
        return detections
    
    def predict(self, images, score_threshold=0.7, mask_threshold=0.5):
        """Complete prediction pipeline"""
        predictions, original_sizes = self.inference(images)
        detections = self.postprocess(
            predictions, original_sizes, score_threshold, mask_threshold
        )
        return detections
    
    def visualize_results(self, image, detections, save_path=None):
        """Visualize detection and segmentation results"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Polygon
        import numpy as np
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(detections)))
        
        for detection, color in zip(detections, colors):
            # Draw bounding box
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Draw mask
            mask = detection['mask']
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
            colored_mask[:, :, :3] = color[:3]
            colored_mask[:, :, 3] = mask * 0.5  # Semi-transparent
            ax.imshow(colored_mask)
            
            # Add label
            label_text = f"{detection['class_name']}: {detection['score']:.2f}"
            ax.text(
                x1, y1 - 5, label_text,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                fontsize=10, color='black'
            )
        
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()

# Example usage
mask_rcnn = MaskRCNNPyTorch(num_classes=80)

# Load and predict
import cv2
image = cv2.imread('test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

detections = mask_rcnn.predict([image], score_threshold=0.7)

print(f"Detected {len(detections[0])} instances")
for detection in detections[0][:5]:  # Show first 5 detections
    print(f"{detection['class_name']}: {detection['score']:.3f}")
    print(f"  Bbox: {detection['bbox']}")
    print(f"  Mask shape: {detection['mask'].shape}")

# Visualize results
mask_rcnn.visualize_results(image, detections[0])
```

### TensorFlow Implementation
```python
import tensorflow as tf

class MaskRCNNTensorFlow:
    """Mask R-CNN implementation using TensorFlow"""
    
    def __init__(self, model_path, num_classes=80):
        self.num_classes = num_classes
        
        # Load saved model
        self.model = tf.saved_model.load(model_path)
        
        # Get inference function
        self.infer = self.model.signatures['serving_default']
        
        # Input tensor info
        self.input_tensor_name = list(self.infer.structured_input_signature[1].keys())[0]
        
    def preprocess(self, image):
        """Preprocess image for TensorFlow model"""
        # Convert to tensor and add batch dimension
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        
        # Convert to uint8 if necessary
        if image.dtype != tf.uint8:
            image = tf.cast(image * 255, tf.uint8)
        
        return image
    
    def inference(self, image):
        """Run inference using TensorFlow model"""
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.infer(**{self.input_tensor_name: input_tensor})
        
        return outputs
    
    def postprocess(self, outputs, score_threshold=0.7, mask_threshold=0.5):
        """Post-process TensorFlow outputs"""
        # Extract outputs
        boxes = outputs['detection_boxes'][0]        # [N, 4] normalized coordinates
        scores = outputs['detection_scores'][0]      # [N]
        classes = outputs['detection_classes'][0]    # [N]
        masks = outputs['detection_masks'][0]        # [N, H, W]
        num_detections = int(outputs['num_detections'][0])
        
        detections = []
        
        for i in range(num_detections):
            if scores[i] >= score_threshold:
                # Convert normalized coordinates to pixel coordinates
                # Note: TensorFlow format is [y1, x1, y2, x2]
                y1, x1, y2, x2 = boxes[i]
                
                # Get mask and apply threshold
                mask = masks[i]
                binary_mask = (mask >= mask_threshold).numpy().astype(np.float32)
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],  # Convert to [x1, y1, x2, y2]
                    'score': float(scores[i]),
                    'class_id': int(classes[i]) - 1,  # Convert to 0-based
                    'class_name': f'class_{int(classes[i]) - 1}',
                    'mask': binary_mask
                }
                detections.append(detection)
        
        return detections
    
    def predict(self, image, score_threshold=0.7, mask_threshold=0.5):
        """Complete prediction pipeline"""
        outputs = self.inference(image)
        detections = self.postprocess(outputs, score_threshold, mask_threshold)
        return detections

# Example usage
# mask_rcnn_tf = MaskRCNNTensorFlow('mask_rcnn_saved_model/', num_classes=80)
# detections = mask_rcnn_tf.predict(image, score_threshold=0.7)
```

### ONNX Runtime Implementation
```python
import onnxruntime as ort

class MaskRCNNONNX:
    """Mask R-CNN inference using ONNX Runtime"""
    
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
        
    def preprocess(self, image):
        """Preprocess image for ONNX model"""
        # Resize and normalize based on model requirements
        # This depends on how the ONNX model was exported
        
        if isinstance(image, np.ndarray) and image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Transpose to NCHW if needed
        if image.shape[-1] == 3:  # HWC -> CHW
            image = np.transpose(image, (0, 3, 1, 2))
        
        return image
    
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
    
    def postprocess(self, outputs, score_threshold=0.7, mask_threshold=0.5):
        """Post-process ONNX outputs"""
        # Output format depends on how the model was exported
        # Common formats:
        # 1. [boxes, scores, classes, masks] - post-processed
        # 2. Raw outputs from each head
        
        if len(outputs) == 4:  # Post-processed format
            boxes, scores, classes, masks = outputs
            
            detections = []
            for box, score, cls, mask in zip(boxes[0], scores[0], classes[0], masks[0]):
                if score >= score_threshold:
                    # Apply mask threshold
                    binary_mask = (mask >= mask_threshold).astype(np.float32)
                    
                    detection = {
                        'bbox': box.tolist(),
                        'score': float(score),
                        'class_id': int(cls),
                        'class_name': f'class_{int(cls)}',
                        'mask': binary_mask
                    }
                    detections.append(detection)
            
            return detections
        
        else:
            # Raw outputs - would need full post-processing
            # Implementation depends on specific model architecture
            raise NotImplementedError("Raw ONNX output post-processing not implemented")
    
    def predict(self, image, score_threshold=0.7, mask_threshold=0.5):
        """Complete prediction pipeline"""
        outputs = self.inference(image)
        detections = self.postprocess(outputs, score_threshold, mask_threshold)
        return detections

# Example usage
# mask_rcnn_onnx = MaskRCNNONNX('mask_rcnn.onnx', num_classes=80)
# detections = mask_rcnn_onnx.predict(image, score_threshold=0.7)
```

## Performance Characteristics

### Model Specifications
```python
mask_rcnn_specs = {
    'architecture': 'Mask R-CNN with ResNet-50 FPN',
    'input_size': '800×800 (variable)',
    'backbone': 'ResNet-50 + Feature Pyramid Network',
    'stages': 2,  # RPN + R-CNN
    'output_components': ['boxes', 'scores', 'labels', 'masks'],
    'parameters': '44.2M',
    'model_size': '170MB',
    'flops': '275G',
    'coco_map_box': '37.9%',
    'coco_map_mask': '34.6%',
    'inference_speed': {
        'v100_gpu': '~50ms',
        'rtx_3080': '~35ms',
        'cpu_i9': '~2000ms'
    },
    'memory_usage': {
        'gpu_inference': '~4GB',
        'training': '~12GB'
    }
}
```

### Key Advantages
- **Instance Segmentation**: Provides pixel-level object boundaries
- **Two-stage Accuracy**: Higher precision than single-stage detectors
- **Mature Framework**: Well-established architecture with extensive research
- **Multi-task Learning**: Joint optimization of detection and segmentation
- **Strong Performance**: State-of-the-art results on COCO dataset

### Limitations
- **Computational Cost**: Higher inference time than single-stage models
- **Memory Requirements**: Larger memory footprint due to mask branch
- **Complexity**: More complex architecture with multiple stages
- **Speed Trade-off**: Slower than detection-only models

### Use Cases
- **Instance Segmentation**: Precise object boundary detection
- **Medical Imaging**: Organ and lesion segmentation
- **Autonomous Driving**: Object detection and segmentation
- **Robotics**: Object manipulation and scene understanding
- **Content Creation**: Image editing and augmented reality
- **Industrial Inspection**: Quality control and defect detection

### Output Processing Notes
- **Mask Resolution**: Default 28×28 masks are upsampled to bounding box size
- **Class-specific Masks**: Each detection includes a mask for the predicted class
- **Binary Masks**: Final masks are thresholded to binary values
- **Coordinate System**: Bounding boxes in absolute pixel coordinates
- **Score Interpretation**: Classification confidence scores from 0-1