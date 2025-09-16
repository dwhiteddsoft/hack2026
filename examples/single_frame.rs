//! Single frame image classification example
//!
//! This example demonstrates how to use the ONNX vision classifier for
//! single-frame image classification tasks like ImageNet models.

use onnx_vision_classifier::{classifier, ClassifierBuilder, Result};
use image::RgbImage;

fn main() -> Result<()> {
    println!("ONNX Vision Classifier - Single Frame Example");
    println!("===========================================");

    // Example 1: Using the builder pattern for a ResNet-style model
    let classifier = ClassifierBuilder::resnet_classifier("models/resnet50.onnx")
        .with_imagenet_classes()
        .build()?;

    // Load and classify an image
    let image_path = "examples/images/cat.jpg";
    if let Ok(image) = image::open(image_path) {
        let rgb_image = image.to_rgb8();
        
        match classifier.classify_single(&rgb_image) {
            Ok(result) => {
                println!("Classification result:");
                println!("  Class ID: {}", result.class_id);
                println!("  Class Name: {}", result.class_name.as_deref().unwrap_or("Unknown"));
                println!("  Confidence: {:.4}", result.confidence);
                
                // Show top 3 predictions
                println!("\nTop 3 predictions:");
                let class_names = vec!["cat".to_string(), "dog".to_string(), "bird".to_string()]; // Example classes
                let top_3 = result.top_n_with_names(3, &class_names);
                for (i, (class_name, score)) in top_3.iter().enumerate() {
                    println!("  {}. {} ({:.4})", i + 1, class_name, score);
                }
            },
            Err(e) => {
                eprintln!("Classification failed: {}", e);
            }
        }
    } else {
        eprintln!("Could not load image: {}", image_path);
        eprintln!("Note: This example expects an image file at {}", image_path);
        
        // Create a synthetic example instead
        println!("\nCreating synthetic example...");
        demonstrate_with_synthetic_image()?;
    }

    Ok(())
}

/// Demonstrate classification with a synthetic image when no real image is available
fn demonstrate_with_synthetic_image() -> Result<()> {
    println!("Creating a synthetic red image for demonstration...");
    
    // Create a simple red image
    let synthetic_image = RgbImage::from_fn(224, 224, |_x, _y| {
        image::Rgb([255, 100, 100]) // Reddish color
    });

    // Build classifier (this would fail without an actual model file)
    // For demonstration purposes, we'll show how it would be used
    println!("Classifier configuration:");
    println!("  Model type: Single Frame");
    println!("  Input size: 224x224");
    println!("  Normalization: ImageNet");
    
    // In a real scenario with a model file:
    /*
    let classifier = classifier()
        .model_path("models/resnet50.onnx")
        .single_frame()
        .input_size(224, 224)
        .imagenet_normalization()
        .class_names(load_imagenet_classes())
        .build()?;

    let result = classifier.classify_single(&synthetic_image)?;
    println!("Synthetic image classified as: {}", result.class_name.unwrap_or("Unknown".to_string()));
    */

    println!("Synthetic image created successfully (224x224 red image)");
    println!("In a real scenario, this would be classified by the ONNX model");

    Ok(())
}

/// Example of loading ImageNet class names from a file
#[allow(dead_code)]
fn load_imagenet_classes() -> Vec<String> {
    // In practice, you'd load these from a file like imagenet_classes.txt
    std::fs::read_to_string("imagenet_classes.txt")
        .unwrap_or_else(|_| {
            // Fallback to a few example classes
            "tench\ngoldfish\ngreat white shark\ntiger shark\nhammerhead".to_string()
        })
        .lines()
        .map(|line| line.trim().to_string())
        .collect()
}

/// Example of different normalization strategies
#[allow(dead_code)]
fn normalization_examples() -> Result<()> {
    // Different normalization strategies for different models
    
    // ImageNet normalization (most common)
    let _imagenet_classifier = classifier()
        .model_path("models/resnet50.onnx")
        .single_frame()
        .imagenet_normalization()
        .build()?;

    // Custom normalization
    let _custom_classifier = classifier()
        .model_path("models/custom_model.onnx")
        .single_frame()
        .custom_normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        .build()?;

    // No normalization (keep values in 0-1 range)
    let _no_norm_classifier = classifier()
        .model_path("models/raw_model.onnx")
        .single_frame()
        .no_normalization()
        .build()?;

    Ok(())
}

/// Example of different input sizes for different model architectures
#[allow(dead_code)]
fn input_size_examples() -> Result<()> {
    // ResNet: 224x224
    let _resnet = ClassifierBuilder::resnet_classifier("models/resnet50.onnx");

    // EfficientNet-B0: 224x224
    let _efficientnet_b0 = ClassifierBuilder::efficientnet_classifier("models/efficientnet_b0.onnx", 224);

    // EfficientNet-B7: 600x600
    let _efficientnet_b7 = ClassifierBuilder::efficientnet_classifier("models/efficientnet_b7.onnx", 600);

    // Vision Transformer: 384x384
    let _vit = classifier()
        .model_path("models/vit_base.onnx")
        .single_frame()
        .input_size(384, 384)
        .imagenet_normalization()
        .build()?;

    Ok(())
}