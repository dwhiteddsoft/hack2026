#[cfg(test)]
mod ort_api_test {
    use ort::{Environment, Session, SessionBuilder};

    #[test]
    fn test_ort_api_discovery() {
        // This test will help us understand the correct API patterns
        println!("Testing ORT API patterns...");
        
        // Try to create environment
        let env_result = Environment::builder()
            .with_name("test")
            .build();
        
        match env_result {
            Ok(env) => {
                println!("✅ Environment created successfully");
                
                // Try SessionBuilder instead of Session::builder
                let model_path = "models/yolov8n.onnx";
                if std::path::Path::new(model_path).exists() {
                    println!("Model file exists: {}", model_path);
                    
                    // Test SessionBuilder pattern
                    match SessionBuilder::new(&env) {
                        Ok(session_builder) => {
                            println!("✅ SessionBuilder::new() works");
                            
                            match session_builder.with_model_from_file(model_path) {
                                Ok(session_builder_with_model) => {
                                    println!("✅ with_model_from_file() works");
                                    
                                    match session_builder_with_model.build() {
                                        Ok(session) => {
                                            println!("✅ Session created successfully!");
                                            println!("Session inputs: {:?}", session.inputs.len());
                                            println!("Session outputs: {:?}", session.outputs.len());
                                        }
                                        Err(e) => println!("❌ Session build failed: {}", e),
                                    }
                                }
                                Err(e) => println!("❌ with_model_from_file() failed: {}", e),
                            }
                        }
                        Err(e) => println!("❌ SessionBuilder::new() failed: {}", e),
                    }
                } else {
                    println!("❌ Model file not found: {}", model_path);
                    
                    // Try with a different path
                    let alt_path = "../models/yolov8n.onnx";
                    if std::path::Path::new(alt_path).exists() {
                        println!("Found model at alternative path: {}", alt_path);
                    } else {
                        // List available files
                        if let Ok(entries) = std::fs::read_dir(".") {
                            println!("Available files in current directory:");
                            for entry in entries {
                                if let Ok(entry) = entry {
                                    println!("  {:?}", entry.file_name());
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => println!("❌ Environment creation failed: {}", e),
        }
    }
}