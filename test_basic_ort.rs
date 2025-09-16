use ort::{Environment, SessionBuilder, Value, Session};
use ndarray::Array;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Download a simple model for testing
    println!("Testing ORT API...");
    
    // Try to create a simple session
    let environment = Arc::new(Environment::builder()
        .with_name("test")
        .build()?);
        
    println!("Environment created successfully");
    
    // Create a simple 1D array for testing
    let input_array: Array<f32, _> = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    println!("Array created: {:?}", input_array);
    
    Ok(())
}