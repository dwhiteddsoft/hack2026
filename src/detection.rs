//! Detection utilities for post-processing object detection model outputs.

use crate::types::{Detection, BoundingBox};

/// Non-Maximum Suppression utilities
pub struct NMS;

impl NMS {
    /// Apply Non-Maximum Suppression to remove duplicate detections
    /// 
    /// # Arguments
    /// * `detections` - Vector of detections to filter
    /// * `iou_threshold` - IoU threshold for considering boxes as duplicates (typically 0.5)
    /// 
    /// # Returns
    /// Vector of filtered detections with duplicates removed
    pub fn apply(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
        if detections.is_empty() {
            return detections;
        }
        
        // Sort by confidence score (descending)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut keep = Vec::new();
        let mut suppressed = vec![false; detections.len()];
        
        for (i, detection) in detections.iter().enumerate() {
            if suppressed[i] {
                continue;
            }
            
            keep.push(detection.clone());
            
            // Suppress overlapping detections
            for (j, other) in detections.iter().enumerate().skip(i + 1) {
                if suppressed[j] {
                    continue;
                }
                
                // Only suppress if same class and high IoU
                if detection.class_id == other.class_id && detection.iou(other) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
        
        keep
    }
    
    /// Apply class-agnostic NMS (suppresses across all classes)
    pub fn apply_class_agnostic(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
        if detections.is_empty() {
            return detections;
        }
        
        // Sort by confidence score (descending)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut keep = Vec::new();
        let mut suppressed = vec![false; detections.len()];
        
        for (i, detection) in detections.iter().enumerate() {
            if suppressed[i] {
                continue;
            }
            
            keep.push(detection.clone());
            
            // Suppress overlapping detections regardless of class
            for (j, other) in detections.iter().enumerate().skip(i + 1) {
                if suppressed[j] {
                    continue;
                }
                
                if detection.iou(other) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
        
        keep
    }
}

/// Detection filtering utilities
pub struct DetectionFilter;

impl DetectionFilter {
    /// Filter detections by confidence threshold
    pub fn by_confidence(detections: Vec<Detection>, min_confidence: f32) -> Vec<Detection> {
        detections
            .into_iter()
            .filter(|d| d.confidence >= min_confidence)
            .collect()
    }
    
    /// Filter detections by specific class IDs
    pub fn by_classes(detections: Vec<Detection>, class_ids: &[usize]) -> Vec<Detection> {
        detections
            .into_iter()
            .filter(|d| class_ids.contains(&d.class_id))
            .collect()
    }
    
    /// Filter detections by minimum area
    pub fn by_min_area(detections: Vec<Detection>, min_area: f32) -> Vec<Detection> {
        detections
            .into_iter()
            .filter(|d| d.area() >= min_area)
            .collect()
    }
    
    /// Keep only the top N detections by confidence
    pub fn top_n(mut detections: Vec<Detection>, n: usize) -> Vec<Detection> {
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        detections.into_iter().take(n).collect()
    }
}

/// Coordinate conversion utilities
pub struct CoordinateConverter;

impl CoordinateConverter {
    /// Convert YOLO format (center_x, center_y, width, height) to (x1, y1, x2, y2)
    /// 
    /// # Arguments
    /// * `cx` - Center x coordinate (normalized 0-1)
    /// * `cy` - Center y coordinate (normalized 0-1)
    /// * `w` - Width (normalized 0-1)
    /// * `h` - Height (normalized 0-1)
    /// * `img_width` - Image width in pixels
    /// * `img_height` - Image height in pixels
    pub fn yolo_to_xyxy(cx: f32, cy: f32, w: f32, h: f32, img_width: u32, img_height: u32) -> BoundingBox {
        let img_w = img_width as f32;
        let img_h = img_height as f32;
        
        let x1 = (cx - w / 2.0) * img_w;
        let y1 = (cy - h / 2.0) * img_h;
        let x2 = (cx + w / 2.0) * img_w;
        let y2 = (cy + h / 2.0) * img_h;
        
        BoundingBox::new(x1, y1, x2, y2)
    }
    
    /// Convert (x1, y1, x2, y2) to YOLO format (center_x, center_y, width, height)
    pub fn xyxy_to_yolo(bbox: &BoundingBox, img_width: u32, img_height: u32) -> (f32, f32, f32, f32) {
        let img_w = img_width as f32;
        let img_h = img_height as f32;
        
        let width = bbox.x2 - bbox.x1;
        let height = bbox.y2 - bbox.y1;
        let cx = (bbox.x1 + width / 2.0) / img_w;
        let cy = (bbox.y1 + height / 2.0) / img_h;
        let w = width / img_w;
        let h = height / img_h;
        
        (cx, cy, w, h)
    }
    
    /// Clamp bounding box coordinates to image boundaries
    pub fn clamp_to_image(bbox: &BoundingBox, img_width: u32, img_height: u32) -> BoundingBox {
        BoundingBox::new(
            bbox.x1.max(0.0).min(img_width as f32),
            bbox.y1.max(0.0).min(img_height as f32),
            bbox.x2.max(0.0).min(img_width as f32),
            bbox.y2.max(0.0).min(img_height as f32),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_detection(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32, class_id: usize) -> Detection {
        Detection {
            bbox: BoundingBox::new(x1, y1, x2, y2),
            confidence,
            class_id,
            class_name: None,
            track_id: None,
        }
    }

    #[test]
    fn test_nms_basic() {
        let detections = vec![
            create_test_detection(10.0, 10.0, 50.0, 50.0, 0.9, 0),  // High confidence
            create_test_detection(15.0, 15.0, 55.0, 55.0, 0.8, 0),  // Overlapping, lower confidence
            create_test_detection(100.0, 100.0, 150.0, 150.0, 0.7, 0),  // Different location
        ];
        
        let filtered = NMS::apply(detections, 0.5);
        
        // Should keep high confidence detection and non-overlapping one
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].confidence, 0.9);
        assert_eq!(filtered[1].confidence, 0.7);
    }

    #[test]
    fn test_confidence_filtering() {
        let detections = vec![
            create_test_detection(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            create_test_detection(20.0, 20.0, 30.0, 30.0, 0.3, 1),
            create_test_detection(40.0, 40.0, 50.0, 50.0, 0.8, 2),
        ];
        
        let filtered = DetectionFilter::by_confidence(detections, 0.5);
        
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|d| d.confidence >= 0.5));
    }

    #[test]
    fn test_yolo_conversion() {
        let bbox = CoordinateConverter::yolo_to_xyxy(0.5, 0.5, 0.4, 0.6, 100, 200);
        
        // Center at (50, 100), width 40, height 120
        // So x1=30, y1=40, x2=70, y2=160
        assert!((bbox.x1 - 30.0).abs() < 0.001);
        assert!((bbox.y1 - 40.0).abs() < 0.001);
        assert!((bbox.x2 - 70.0).abs() < 0.001);
        assert!((bbox.y2 - 160.0).abs() < 0.001);
    }

    #[test]
    fn test_bbox_iou() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        
        let iou = bbox1.iou(&bbox2);
        
        // Intersection: 5x5 = 25
        // Union: 100 + 100 - 25 = 175
        // IoU: 25/175 ≈ 0.143
        assert!((iou - 0.142857).abs() < 0.001);
    }
}