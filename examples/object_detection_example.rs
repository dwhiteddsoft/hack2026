//! Object detection example using YOLOv8 or any object detection model
//! 
//! This example demonstrates:
//! - Loading an object detection ONNX model
//! - Processing images to detect objects with bounding boxes
//! - Applying Non-Maximum Suppression (NMS) to filter overlapping detections
//! - Filtering detections by confidence threshold
//! - Outputting results as both structured data and JSON

use onnx_vision_classifier::{ClassifierBuilder, Result};
use image::io::Reader as ImageReader;
use std::path::Path;

// COCO dataset class names (80 classes) - commonly used with YOLOv8
const COCO_CLASSES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

fn main() -> Result<()> {
    println!("🔍 ONNX Object Detection Example");
    println!("=================================");
    
    // Check if model and image files exist
    let model_path = "yolov2-coco-9.onnx";  // Using YOLOv2 which is fully supported
    //let model_path = "tiny-yolov3-11.onnx";  // This model also requires dual inputs
    let image_path = "test_image_640_640.jpg";
    
    if !Path::new(model_path).exists() {
        println!("❌ Model file not found: {}", model_path);
        println!("\n📥 To run this example, you need a compatible YOLO ONNX model:");
        println!("   
✅ Working Models:
   1. YOLOv2: Download yolov2-coco-9.onnx (already available in workspace)
   2. YOLOv5: Download from Ultralytics releases
   3. YOLOv8: Use 'yolo export model=yolov8n.pt format=onnx'
   
⚠️  Problematic Models (dual-input lifetime issues):
   - tiny-yolov3-11.onnx (requires image_shape input)
   - yolov3-10.onnx (requires image_shape input)
   
🔧 Quick fix: Change model_path to 'yolov2-coco-9.onnx' in the code above");
        return Ok(());
    }
    
    if !Path::new(image_path).exists() {
        println!("❌ Test image not found: {}", image_path);
        println!("   Please place a test image named 'test_image.jpg' in the project root.");
        return Ok(());
    }
    
    println!("📁 Using model: {}", model_path);
    println!("🖼️  Using image: {}", image_path);
    println!();
    
    // Create object detection classifier with YOLOv2-optimized settings
    println!("⚙️  Building object detection classifier...");
    let classifier = ClassifierBuilder::new()
        .model_path(model_path)
        //.yolo_detection()  // Uses confidence=0.25, nms=0.45, classes=80
        .strict_object_detection()  // Uses confidence=0.25, nms=0.45, classes=80
        .input_size(416, 416)  // YOLOv2 input size
        .class_names(COCO_CLASSES.iter().map(|s| s.to_string()).collect())
        .imagenet_normalization()
        .build()?;
    
    println!("✅ Classifier built successfully!");
    println!("   - Model type: Object Detection");
    println!("   - Input size: 416x416");
    println!("   - Classes: {} (COCO dataset)", COCO_CLASSES.len());
    println!("   - Confidence threshold: 0.25");
    println!("   - NMS threshold: 0.45");
    println!();
    
    // Load and process the test image
    println!("📖 Loading test image...");
    let img = ImageReader::open(image_path)?
        .decode()?
        .to_rgb8();
    
    println!("✅ Image loaded: {}x{} pixels", img.width(), img.height());
    println!();
    
    // Perform object detection
    println!("🔍 Detecting objects...");
    let start_time = std::time::Instant::now();
    
    let detection_result = classifier.detect_objects(&img)?;
    
    let inference_time = start_time.elapsed();
    println!("✅ Detection completed in {:.2}ms", inference_time.as_millis());
    println!();
    
    // Display results
    println!("📊 Detection Results");
    println!("===================");
    println!("Total detections: {}", detection_result.detections.len());
    
    if detection_result.detections.is_empty() {
        println!("🤷 No objects detected above the confidence threshold.");
        println!("   Try lowering the confidence threshold or using a different image.");
    } else {
        println!("🎯 Objects found:");
        println!();
        
        let mut sorted_detections = detection_result.detections.clone();
        sorted_detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));        
        // Debug: Show first detection for debugging
        if let Some(first_detection) = sorted_detections.first() {
            println!("🐛 Debug - First detection: {:?}", first_detection);
        }
        
        for (i, detection) in sorted_detections.iter().enumerate() {
            let class_name = match &detection.class_name {
                Some(name) => name.as_str(),
                None => {
                    if detection.class_id < COCO_CLASSES.len() {
                        COCO_CLASSES[detection.class_id]
                    } else {
                        "Unknown"
                    }
                }
            };
            
            println!("  {}. {} (confidence: {:.1}%)", 
                i + 1, 
                class_name, 
                detection.confidence// * 100.0
            );
            println!("     Bounding box: ({:.1}, {:.1}) to ({:.1}, {:.1})", 
                detection.bbox.x1, 
                detection.bbox.y1,
                detection.bbox.x2, 
                detection.bbox.y2
            );
            println!("     Size: {:.1} x {:.1} pixels (area: {:.0})",
                detection.bbox.x2 - detection.bbox.x1,
                detection.bbox.y2 - detection.bbox.y1,
                detection.area()
            );
            println!();
            if (i == 5) { break; }
        }
    }
    
    // Demonstrate filtering capabilities
    println!("🔽 Filtering Examples");
    println!("=====================");
    
    // High confidence detections only
    let high_confidence = detection_result.filter_by_confidence(0.5);
    println!("High confidence detections (>50%): {}", high_confidence.len());
    
    // Top 5 most confident detections
    let top_detections = detection_result.top_n_detections(5);
    println!("Top 5 most confident: {}", top_detections.len());
    println!();
    
    // // JSON output example
    // println!("📄 JSON Output Example");
    // println!("======================");
    
    // let json_result = classifier.detect_objects_json(&img)?;
    // println!("{}", json_result);
    // println!();
    
    // Show some statistics
    println!("📈 Performance Statistics");
    println!("=========================");
    if let Some(processing_time) = detection_result.processing_time_ms {
        println!("Processing time: {}ms", processing_time);
    }
    println!("Image dimensions: {}x{}", detection_result.image_width, detection_result.image_height);
    
    if !detection_result.detections.is_empty() {
        let avg_confidence: f32 = detection_result.detections
            .iter()
            .map(|d| d.confidence)
            .sum::<f32>() / detection_result.detections.len() as f32;
        
        let max_confidence = detection_result.detections
            .iter()
            .map(|d| d.confidence)
            .fold(0.0, f32::max);
            
        println!("Average confidence: {:.1}%", avg_confidence * 100.0);
        println!("Maximum confidence: {:.1}%", max_confidence * 100.0);
    }
    
    println!();
    println!("🎉 Object detection example completed successfully!");
    println!();
    println!("💡 Tips for better results:");
    println!("   - Use higher resolution images for small object detection");
    println!("   - Adjust confidence threshold based on your use case");
    println!("   - Lower NMS threshold to keep more overlapping detections");
    println!("   - Use model-specific input sizes for optimal performance");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_vision_classifier::{ClassifierBuilder, ModelType};
    
    #[test]
    fn test_builder_yolo_configuration() {
        let builder = ClassifierBuilder::new()
            .yolo_detection()
            .input_size(640, 640);
            
        // We can't easily test the actual build without a model file,
        // but we can test that the builder methods work
        assert!(true); // Placeholder test
    }
    
    #[test]
    fn test_coco_classes_count() {
        assert_eq!(COCO_CLASSES.len(), 80, "COCO dataset should have 80 classes");
    }
    
    #[test]
    fn test_class_names_conversion() {
        let class_names: Vec<String> = COCO_CLASSES.iter().map(|s| s.to_string()).collect();
        assert_eq!(class_names.len(), 80);
        assert_eq!(class_names[0], "person");
        assert_eq!(class_names[79], "toothbrush");
    }
}