use anyhow::Result;
use uocvr::{UniversalSession, SessionBuilder};
use uocvr::core::ExecutionProvider;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    uocvr::utils::logging::init_logging()?;

    println!("UOCVR Basic Inference Example");
    println!("=============================");

    // Example 1: Simple inference with auto-detection
    println!("\n1. Simple inference with auto-detection:");
    if let Err(e) = simple_inference().await {
        eprintln!("Simple inference failed: {}", e);
    }

    // Example 2: Advanced configuration
    println!("\n2. Advanced configuration:");
    if let Err(e) = advanced_configuration().await {
        eprintln!("Advanced configuration failed: {}", e);
    }

    // Example 3: Batch processing
    println!("\n3. Batch processing:");
    if let Err(e) = batch_processing().await {
        eprintln!("Batch processing failed: {}", e);
    }

    Ok(())
}

/// Simple inference example with automatic model detection
async fn simple_inference() -> Result<()> {
    // Note: These examples use placeholder paths - replace with actual model files
    let model_path = "models/yolov8n.onnx";
    let image_path = "test_images/test_image.jpg";

    println!("Loading model: {}", model_path);
    
    // Create session with auto-detection (this will fail until implementation is complete)
    let session = match UniversalSession::from_model_file(model_path).await {
        Ok(session) => session,
        Err(e) => {
            println!("Model loading failed (expected in skeleton): {}", e);
            return Ok(());
        }
    };

    println!("Model loaded successfully!");
    println!("Model info: {:?}", session.model_info());

    // Run inference
    println!("Running inference on: {}", image_path);
    let detections = match session.infer_image(image_path).await {
        Ok(detections) => detections,
        Err(e) => {
            println!("Inference failed (expected in skeleton): {}", e);
            return Ok(());
        }
    };

    println!("Found {} detections", detections.len());
    for (i, detection) in detections.iter().enumerate() {
        println!(
            "  Detection {}: class_id={}, confidence={:.3}, bbox=({:.1}, {:.1}, {:.1}, {:.1})",
            i + 1,
            detection.class_id,
            detection.confidence,
            detection.bbox.x,
            detection.bbox.y,
            detection.bbox.width,
            detection.bbox.height
        );
    }

    Ok(())
}

/// Advanced configuration example
async fn advanced_configuration() -> Result<()> {
    let model_path = "models/yolov8s.onnx";
    let config_path = "configs/yolov8s_config.yaml";

    println!("Creating session with advanced configuration:");
    println!("  Model: {}", model_path);
    println!("  Config: {}", config_path);

    // Create session with custom configuration
    let session = match SessionBuilder::new()
        .model_file(model_path)
        .config_file(config_path)
        .provider(ExecutionProvider::CPU)
        .batch_size(4)
        .build()
        .await
    {
        Ok(session) => session,
        Err(e) => {
            println!("Advanced configuration failed (expected in skeleton): {}", e);
            return Ok(());
        }
    };

    println!("Session created with advanced configuration!");
    println!("Model info: {:?}", session.model_info());

    Ok(())
}

/// Batch processing example
async fn batch_processing() -> Result<()> {
    println!("Loading images for batch processing...");

    // Load multiple images
    let image_paths = vec![
        "test_images/test_image.jpg",
        // "test_images/image2.jpg", 
        // "test_images/image3.jpg",
        // "test_images/image4.jpg",
    ];

    let mut images = Vec::new();
    for path in &image_paths {
        match uocvr::utils::image_utils::load_image(path) {
            Ok(image) => {
                println!("  Loaded: {}", path);
                images.push(image);
            }
            Err(e) => {
                println!("  Failed to load {}: {}", path, e);
            }
        }
    }

    if images.is_empty() {
        println!("No images loaded for batch processing");
        return Ok(());
    }

    // Create session
    let model_path = "models/yolov8n.onnx";
    let session = match UniversalSession::from_model_file(model_path).await {
        Ok(session) => session,
        Err(e) => {
            println!("Model loading failed (expected in skeleton): {}", e);
            return Ok(());
        }
    };

    // Run batch inference
    println!("Running batch inference on {} images...", images.len());
    let results = match session.infer_batch(&images).await {
        Ok(results) => results,
        Err(e) => {
            println!("Batch inference failed (expected in skeleton): {}", e);
            return Ok(());
        }
    };

    println!("Batch inference completed!");
    for (i, result) in results.iter().enumerate() {
        println!(
            "  Image {}: {} detections, processing time: {:?}",
            i + 1,
            result.detections.len(),
            result.processing_time
        );
    }

    Ok(())
}

/// Utility function to print system information
#[allow(dead_code)]
fn print_system_info() {
    println!("System Information:");
    println!("  UOCVR version: {}", uocvr::VERSION);
    println!("  Available execution providers: CPU");
    #[cfg(feature = "cuda")]
    println!("  CUDA support: Enabled");
    #[cfg(not(feature = "cuda"))]
    println!("  CUDA support: Disabled");
}