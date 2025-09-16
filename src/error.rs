//! Error types for the ONNX vision classifier

use thiserror::Error;

/// Main error type for the classifier
#[derive(Error, Debug)]
pub enum ClassificationError {
    #[error("ONNX runtime error: {0}")]
    OnnxError(String),
    
    #[error("Image processing error: {0}")]
    ImageProcessingError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Insufficient frames: expected {expected}, got {actual}")]
    InsufficientFrames { expected: usize, actual: usize },
    
    #[error("Buffer overflow: attempted to add more frames than capacity")]
    BufferOverflow,
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("JSON serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),
    
    #[error("Array shape error: {0}")]
    ArrayShapeError(String),
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, ClassificationError>;