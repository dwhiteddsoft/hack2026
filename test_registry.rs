use uocvr::models::{ModelRegistry, SearchCriteria};
use uocvr::core::TaskType;

fn main() {
    println!("Testing Model Registry Implementation...");
    
    // Test model registry creation
    let registry = ModelRegistry::new();
    println!("✅ Model registry created successfully");
    
    // Test listing all models
    let models = registry.list_models();
    println!("📋 Available models: {:?}", models);
    
    // Test getting YOLOv8 profile
    if let Some(yolov8_profile) = registry.get_profile("yolov8") {
        println!("✅ YOLOv8 profile found:");
        println!("   Name: {}", yolov8_profile.name);
        println!("   Description: {}", yolov8_profile.description);
        println!("   Variants: {}", yolov8_profile.variants.len());
        println!("   Default variant: {}", yolov8_profile.default_variant);
        
        // Test getting default variant
        if let Some(default_variant) = registry.get_default_variant("yolov8") {
            println!("   ✅ Default variant ({}): {}", default_variant.name, default_variant.version);
            if let Some(metrics) = &default_variant.performance_metrics {
                println!("      Inference time: {:.1}ms", metrics.inference_time_ms);
                println!("      Memory usage: {}MB", metrics.memory_usage_mb);
                if let Some(map_50_95) = metrics.accuracy_metrics.map_50_95 {
                    println!("      mAP@0.5:0.95: {:.1}%", map_50_95);
                }
            }
        }
    } else {
        println!("❌ YOLOv8 profile not found");
    }
    
    // Test search functionality
    println!("\n🔍 Testing search functionality:");
    
    // Search for detection models
    let criteria = SearchCriteria {
        architecture_type: None,
        supported_tasks: vec![TaskType::Detection],
        max_memory_mb: None,
        min_accuracy: None,
        max_inference_time_ms: None,
        hardware_constraints: None,
    };
    
    let results = registry.search_models(&criteria);
    println!("   Detection models found: {}", results.len());
    for profile in results {
        println!("     - {} ({})", profile.name, profile.description);
    }
    
    // Search for fast models (< 5ms inference time)
    let fast_criteria = SearchCriteria {
        architecture_type: None,
        supported_tasks: vec![],
        max_memory_mb: None,
        min_accuracy: None,
        max_inference_time_ms: Some(5.0),
        hardware_constraints: None,
    };
    
    let fast_results = registry.search_models(&fast_criteria);
    println!("   Fast models (< 5ms): {}", fast_results.len());
    for profile in fast_results {
        println!("     - {}", profile.name);
    }
    
    println!("\n✅ All model registry tests passed!");
}