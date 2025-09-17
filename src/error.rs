/// Main error type for the UOCVR library
#[derive(Debug)]
pub enum UocvrError {
    OnnxRuntime(ort::OrtError),
    ImageProcessing(image::ImageError),
    Io(std::io::Error),
    Serialization(serde_json::Error),
    Yaml(serde_yaml::Error),
    ModelConfig { message: String },
    InputValidation { message: String },
    OutputProcessing { message: String },
    Session { message: String },
    UnsupportedModel { model_type: String },
    InvalidConfig { message: String },
    ResourceNotFound { resource: String },
    Runtime { message: String },
}

impl std::fmt::Display for UocvrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UocvrError::OnnxRuntime(e) => write!(f, "ONNX Runtime error: {}", e),
            UocvrError::ImageProcessing(e) => write!(f, "Image processing error: {}", e),
            UocvrError::Io(e) => write!(f, "IO error: {}", e),
            UocvrError::Serialization(e) => write!(f, "Serialization error: {}", e),
            UocvrError::Yaml(e) => write!(f, "YAML error: {}", e),
            UocvrError::ModelConfig { message } => write!(f, "Model configuration error: {}", message),
            UocvrError::InputValidation { message } => write!(f, "Input validation error: {}", message),
            UocvrError::OutputProcessing { message } => write!(f, "Output processing error: {}", message),
            UocvrError::Session { message } => write!(f, "Session error: {}", message),
            UocvrError::UnsupportedModel { model_type } => write!(f, "Unsupported model type: {}", model_type),
            UocvrError::InvalidConfig { message } => write!(f, "Invalid configuration: {}", message),
            UocvrError::ResourceNotFound { resource } => write!(f, "Resource not found: {}", resource),
            UocvrError::Runtime { message } => write!(f, "Runtime error: {}", message),
        }
    }
}

impl std::error::Error for UocvrError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            UocvrError::OnnxRuntime(e) => Some(e),
            UocvrError::ImageProcessing(e) => Some(e),
            UocvrError::Io(e) => Some(e),
            UocvrError::Serialization(e) => Some(e),
            UocvrError::Yaml(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ort::OrtError> for UocvrError {
    fn from(error: ort::OrtError) -> Self {
        UocvrError::OnnxRuntime(error)
    }
}

impl From<image::ImageError> for UocvrError {
    fn from(error: image::ImageError) -> Self {
        UocvrError::ImageProcessing(error)
    }
}

impl From<std::io::Error> for UocvrError {
    fn from(error: std::io::Error) -> Self {
        UocvrError::Io(error)
    }
}

impl From<serde_json::Error> for UocvrError {
    fn from(error: serde_json::Error) -> Self {
        UocvrError::Serialization(error)
    }
}

impl From<serde_yaml::Error> for UocvrError {
    fn from(error: serde_yaml::Error) -> Self {
        UocvrError::Yaml(error)
    }
}

/// Result type alias for UOCVR operations
pub type Result<T> = std::result::Result<T, UocvrError>;

impl UocvrError {
    pub fn model_config<S: Into<String>>(message: S) -> Self {
        Self::ModelConfig {
            message: message.into(),
        }
    }

    pub fn input_validation<S: Into<String>>(message: S) -> Self {
        Self::InputValidation {
            message: message.into(),
        }
    }

    pub fn output_processing<S: Into<String>>(message: S) -> Self {
        Self::OutputProcessing {
            message: message.into(),
        }
    }

    pub fn session<S: Into<String>>(message: S) -> Self {
        Self::Session {
            message: message.into(),
        }
    }

    pub fn unsupported_model<S: Into<String>>(model_type: S) -> Self {
        Self::UnsupportedModel {
            model_type: model_type.into(),
        }
    }

    pub fn invalid_config<S: Into<String>>(message: S) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    pub fn resource_not_found<S: Into<String>>(resource: S) -> Self {
        Self::ResourceNotFound {
            resource: resource.into(),
        }
    }

    pub fn runtime<S: Into<String>>(message: S) -> Self {
        Self::Runtime {
            message: message.into(),
        }
    }
}