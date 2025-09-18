use anyhow::Result;
use uocvr::{SessionBuilder, UniversalSession};
use uocvr::core::ExecutionProvider;

#[tokio::main]
async fn main() -> Result<()> {
    // all the models
    let models = vec![
        // "models/retinanet-9.onnx",
        // "models/MaskRCNN-12.onnx",
        // "models/mobilenetv2-7.onnx",
        ("models/ssd-10.onnx","configs/ssd-10_config.yaml"),
        //("models/yolov8n.onnx","configs/yolov8n_config.yaml"),
        //("models/yolov2-coco-9.onnx","configs/yolov2-coco-9_config.yaml"),
        //("models/yolov3-10.onnx","configs/yolov3-10_config.yaml"),
    ];
    let pics = vec![
        // "test_images/test_image_1.jpg",
        "test_images/test_image_2.jpeg",
        // "test_images/test_image_3.jpeg",
        // "test_images/test_image_4.jpeg",
        // "test_images/test_image_5.jpeg",
        // "test_images/test_image_6.jpeg",
        // "test_images/test_image_7.jpeg",
        // "test_images/test_image_8.jpeg",
        // "test_images/test_image_9.jpeg",
    ];

    // Initialize logging with ERROR level only (suppress INFO/WARN)
    // uocvr::utils::logging::init_logging()?;
    
    // Custom logging setup - only show errors
    use tracing_subscriber::filter::LevelFilter;
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::ERROR)
        .init();

    println!("UOCVR Basic Inference Example");
    println!("=============================");

    // Example 1: Simple inference with auto-detection
    //println!("\n1. Simple inference with auto-detection:");
    // for model_path in models {
    //     println!("\nTesting model: {}", model_path);
    //     for image_path in &pics {
    //         if let Err(e) = simple_inference(model_path.to_string(), image_path.to_string()).await {
    //             eprintln!("Simple inference failed: {}", e);
    //         }
    //     }
    // }

    // Example 2: Advanced configuration
    // println!("\n2. Advanced configuration:");
    for (model_path, config_path) in models {
        println!("\nTesting model: {}", model_path);
        for image_path in &pics {
            if let Err(e) = advanced_configuration(model_path.to_string(), config_path.to_string(), image_path.to_string()).await {
                eprintln!("Advanced Configuration inference failed: {}", e);
            }
        }
    }

    // Example 3: Batch processing
    // println!("\n3. Batch processing:");
    // if let Err(e) = batch_processing().await {
    //     eprintln!("Batch processing failed: {}", e);
    // }

    Ok(())
}

/// Simple inference example with automatic model detection
#[allow(dead_code)]
// async fn simple_inference(model_path: String, image_path: String) -> Result<()> {
//     //println!("Loading model: {}", model_path);
    
//     // Create session with auto-detection (this will fail until implementation is complete)
//     let session = match UniversalSession::from_model_file(model_path).await {
//         Ok(session) => session,
//         Err(e) => {
//             println!("Model loading failed (expected in skeleton): {}", e);
//             return Ok(());
//         }
//     };

//     //println!("Model loaded successfully!");
//     //println!("Model info: {:?}", session.model_info());

//     // Run inference
//     println!("Running inference on: {}", image_path);
//     let detections = match session.infer_image(image_path).await {
//         Ok(detections) => detections,
//         Err(e) => {
//             println!("Inference failed (expected in skeleton): {}", e);
//             return Ok(());
//         }
//     };

//     println!("Found {} detections", detections.len());
//     for (i, detection) in detections.iter().enumerate() {
//         println!(
//             "  Detection {}: class_id={}, confidence={:.3}, bbox=({:.1}, {:.1}, {:.1}, {:.1})",
//             i + 1,
//             detection.class_id,
//             detection.confidence,
//             detection.bbox.x,
//             detection.bbox.y,
//             detection.bbox.width,
//             detection.bbox.height
//         );
//     }

//     Ok(())
// }

/// Advanced configuration example
#[allow(dead_code)]
async fn advanced_configuration(model_path: String, config_path: String, image_path: String) -> Result<()> {
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
    //println!("Model info: {:?}", session.model_info());
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

/// Batch processing example
#[allow(dead_code)]
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