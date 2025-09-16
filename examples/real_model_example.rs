//! Example showing how to use the classifier with a real ONNX model
//!
//! This example demonstrates:
//! 1. How to download and use real ONNX models
//! 2. How to classify images with pre-trained models
//! 3. Proper error handling for real-world scenarios

use onnx_vision_classifier::{classifier, Result};
use image::{RgbImage, open};
use std::path::Path;

fn main() -> Result<()> {
    println!("Real ONNX Model Example");
    println!("=======================");
    
    // Note: To run this example with a real model, you would:
    // 1. Download an ONNX model (e.g., from ONNX Model Zoo)
    // 2. Update the model_path below
    // 3. Provide matching class names
    // 4. Ensure input image dimensions match model requirements
    
    let model_path = "resnet50.onnx"; // Replace with actual model path
    //let model_path = "mobilenetv2-7.onnx"; // Replace with actual model path
    
    // Check if model exists
    if !Path::new(model_path).exists() {
        println!("❌ Model file '{}' not found.", model_path);
        println!();
        println!("To test with a real model:");
        println!("1. Download a model from ONNX Model Zoo:");
        println!("   https://github.com/onnx/models");
        println!("2. Place it in the project directory");
        println!("3. Update the model_path in this example");
        println!();
        println!("For example, download ResNet-50:");
        println!("curl -L -o resnet50.onnx 'https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx'");
        return Ok(());
    }
    
    // ImageNet class names (subset for demo)
    let class_names = vec![
        "tench".to_string(),
        "goldfish".to_string(), 
        "great white shark".to_string(),
        "tiger shark".to_string(),
        "hammerhead".to_string(),
        // ... add more classes as needed
        "Egyptian cat".to_string(),
        "tabby cat".to_string(),
        "tiger cat".to_string(),
    ];
    
    println!("🔧 Creating classifier...");
    let classifier = classifier()
        .model_path(model_path)
        .single_frame()
        .input_size(224, 224)  // Common size for ImageNet models
        .class_names(class_names)
        .build()?;
    
    println!("✅ Classifier created successfully!");
    
    // Example 1: Classify an image file (if it exists)
    let test_image_path = "test_image.jpg";
    //let test_image_path = "test_image_small.jpg";
    if Path::new(test_image_path).exists() {
        println!("\n📸 Classifying image from file: {}", test_image_path);
        
        let img = open(test_image_path)
            .map_err(|e| onnx_vision_classifier::ClassificationError::ImageError(e))?
            .to_rgb8();
        
        let result = classifier.classify_single(&img)?;
        
        println!("Prediction:");
        println!("  Class ID: {}", result.class_id);
        if let Some(ref class_name) = result.class_name {
            println!("  Class: {}", class_name);
        }
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
        
        // Show top 3 predictions
        let top_3 = result.top_n(3);
        println!("  Top 3 predictions:");
        for (i, (class_id, score)) in top_3.iter().enumerate() {
            println!("    {}. Class {}: {:.2}%", i + 1, class_id, score * 100.0);
        }
    } else {
        println!("\n📸 No test image found. Creating synthetic image...");
        
        // Create a test image (blue gradient)
        let test_img = RgbImage::from_fn(224, 224, |x, y| {
            let blue = ((x + y) as f32 / (224.0 + 224.0) * 255.0) as u8;
            image::Rgb([0, 0, blue])
        });
        
        let result = classifier.classify_single(&test_img)?;
        
        println!("Synthetic image classification:");
        println!("  Class ID: {}", result.class_id);
        if let Some(ref class_name) = result.class_name {
            println!("  Class: {}", class_name);
        }
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
    }
    
    println!("\n🎉 Real ONNX model integration successful!");
    println!("\nSupported model types:");
    println!("• Single frame image classification (ResNet, EfficientNet, etc.)");
    println!("• Multi-frame video classification (I3D, SlowFast, etc.)");
    println!("• LSTM sequence models");
    println!("• Two-stream RGB+optical flow models");
    println!("• Variable frame count models");
    
    Ok(())
}