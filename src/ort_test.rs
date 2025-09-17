use ort::{Environment, Session};

#[cfg(test)]
mod ort_tests {
    use super::*;

    #[test]
    fn test_ort_api_patterns() {
        // Test basic ORT API
        println!("Testing ORT API patterns...");
        
        // Create environment
        let environment = Environment::builder()
            .with_name("test")
            .build()
            .expect("Failed to create environment");
        
        // Check what methods are available on Session
        let model_path = "models/yolov8n.onnx";
        
        if std::path::Path::new(model_path).exists() {
            println!("Model file exists: {}", model_path);
            
            // Test session creation - trying different API patterns
            match Session::builder(&environment) {
                Ok(builder) => {
                    println!("✅ Session builder created");
                    match builder.with_model_from_file(model_path) {
                        Ok(configured_builder) => {
                            println!("✅ Model loaded to builder");
                            match configured_builder.build() {
                                Ok(session) => {
                                    println!("✅ Session created successfully!");
                                    println!("Session inputs: {:?}", session.inputs.len());
                                    println!("Session outputs: {:?}", session.outputs.len());
                                }
                                Err(e) => println!("❌ Session build failed: {}", e),
                            }
                        }
                        Err(e) => println!("❌ Model loading failed: {}", e),
                    }
                }
                Err(e) => println!("❌ Session builder creation failed: {}", e),
            }
        } else {
            println!("❌ Model file not found: {}", model_path);
        }
    }
}