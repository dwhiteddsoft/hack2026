//! Simple demonstration of the ONNX vision classifier library

use onnx_vision_classifier::{classifier, ModelType, Result};
use image::{RgbImage, Rgb};

fn main() -> Result<()> {
    println!("ONNX Vision Classifier Demo");
    println!("===========================");
    
    // Create a test image
    let test_image = RgbImage::from_fn(224, 224, |x, y| {
        // Create a simple gradient pattern
        let r = (x as f32 / 224.0 * 255.0) as u8;
        let g = (y as f32 / 224.0 * 255.0) as u8;
        let b = 128;
        Rgb([r, g, b])
    });
    
    println!("\n1. Single Frame Classification");
    println!("------------------------------");
    
    // Example 1: Single frame classification
    let single_classifier = classifier()
        .model_path("mock_model.onnx")  // Mock path for demo
        .single_frame()
        .input_size(224, 224)
        .class_names(vec![
            "cat".to_string(),
            "dog".to_string(), 
            "bird".to_string(),
            "fish".to_string(),
            "horse".to_string()
        ])
        .build()?;
    
    let result = single_classifier.classify_single(&test_image)?;
    println!("Predicted class: {} ({})", 
             result.class_id, 
             result.class_name.unwrap_or("Unknown".to_string()));
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    
    println!("\n2. Multi-Frame Video Classification");
    println!("-----------------------------------");
    
    // Example 2: Multi-frame classification
    let mut multi_classifier = classifier()
        .model_path("mock_multi_frame_model.onnx")
        .multi_frame(3)
        .input_size(112, 112)
        .class_names(vec![
            "walking".to_string(),
            "running".to_string(),
            "jumping".to_string(),
            "standing".to_string(),
        ])
        .build()?;
    
    // Simulate streaming frames
    for i in 0..5 {
        let frame = RgbImage::from_fn(112, 112, |x, y| {
            let r = ((x + i * 20) % 255) as u8;
            let g = ((y + i * 30) % 255) as u8;
            let b = (i * 50) as u8;
            Rgb([r, g, b])
        });
        
        if let Some(result) = multi_classifier.push_frame(frame)? {
            println!("Frame {}: {} (confidence: {:.2}%)", 
                     i, 
                     result.class_name.unwrap_or("Unknown".to_string()),
                     result.confidence * 100.0);
        } else {
            println!("Frame {}: Buffering...", i);
        }
    }
    
    println!("\n3. LSTM Sequence Classification");
    println!("-------------------------------");
    
    // Example 3: LSTM sequence
    let mut lstm_classifier = classifier()
        .model_path("mock_lstm_model.onnx")
        .lstm_sequence(4)
        .input_size(64, 64)
        .build()?;
    
    for i in 0..6 {
        let frame = RgbImage::from_fn(64, 64, |x, y| {
            // Create a moving pattern
            let shift = i * 10;
            let r = ((x + shift) % 255) as u8;
            let g = ((y + shift) % 255) as u8;
            let b = 200;
            Rgb([r, g, b])
        });
        
        if let Some(result) = lstm_classifier.push_frame(frame)? {
            println!("Sequence {}: Predicted class {} (confidence: {:.2}%)", 
                     i, 
                     result.class_id,
                     result.confidence * 100.0);
        } else {
            println!("Frame {}: Building sequence...", i);
        }
    }
    
    println!("\nDemo completed successfully!");
    
    Ok(())
}