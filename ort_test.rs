// Test file to understand ORT API patterns
use ort::{Environment, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test basic ORT session creation
    println!("Testing ORT API patterns...");
    
    // Create environment
    let environment = Environment::builder()
        .with_name("test")
        .build()?;
    
    // Try to create a session (this should show us the correct API)
    let model_path = "models/yolov8n.onnx";
    
    if std::path::Path::new(model_path).exists() {
        println!("Model file exists: {}", model_path);
        
        // Test session creation
        match Session::builder(&environment)?.with_model_from_file(model_path) {
            Ok(session_builder) => {
                match session_builder.build() {
                    Ok(session) => {
                        println!("✅ Session created successfully!");
                        println!("Session inputs: {:?}", session.inputs.len());
                        println!("Session outputs: {:?}", session.outputs.len());
                    }
                    Err(e) => println!("❌ Session build failed: {}", e),
                }
            }
            Err(e) => println!("❌ Session builder failed: {}", e),
        }
    } else {
        println!("❌ Model file not found: {}", model_path);
    }
    
    Ok(())
}