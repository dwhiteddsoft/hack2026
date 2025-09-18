/// Utility functions and helper types for UOCVR
use std::time::{Duration, Instant};

/// Performance timer for measuring inference times
pub struct Timer {
    start: Instant,
    label: String,
}

/// Memory usage tracker
pub struct MemoryTracker {
    peak_usage: usize,
    current_usage: usize,
}

/// Image utilities
pub mod image_utils {
    use image::DynamicImage;

    /// Load image from file path
    pub fn load_image<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<DynamicImage> {
        image::open(path).map_err(crate::error::UocvrError::ImageProcessing)
    }

    /// Load multiple images from directory
    pub fn load_images_from_dir<P: AsRef<std::path::Path>>(
        _dir_path: P,
    ) -> crate::error::Result<Vec<DynamicImage>> {
        // Implementation will be added in the actual build
        todo!("load_images_from_dir implementation")
    }

    /// Save image to file
    pub fn save_image<P: AsRef<std::path::Path>>(
        image: &DynamicImage,
        path: P,
    ) -> crate::error::Result<()> {
        image
            .save(path)
            .map_err(crate::error::UocvrError::ImageProcessing)
    }

    /// Draw bounding boxes on image
    pub fn draw_detections(
        _image: &DynamicImage,
        _detections: &[crate::core::Detection],
    ) -> DynamicImage {
        // Implementation will be added in the actual build
        todo!("draw_detections implementation")
    }

    /// Resize image while maintaining aspect ratio
    pub fn resize_maintain_aspect(
        _image: &DynamicImage,
        _target_width: u32,
        _target_height: u32,
    ) -> DynamicImage {
        // Implementation will be added in the actual build
        todo!("resize_maintain_aspect implementation")
    }

    /// Convert image to RGB format
    pub fn ensure_rgb(image: &DynamicImage) -> DynamicImage {
        match image {
            DynamicImage::ImageRgb8(_) => image.clone(),
            _ => DynamicImage::ImageRgb8(image.to_rgb8()),
        }
    }

    /// Get image dimensions
    pub fn get_dimensions(image: &DynamicImage) -> (u32, u32) {
        (image.width(), image.height())
    }
}

/// Math utilities
pub mod math_utils {
    /// Calculate intersection over union (IoU)
    pub fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
        let [x1_1, y1_1, x2_1, y2_1] = *box1;
        let [x1_2, y1_2, x2_2, y2_2] = *box2;

        let x1 = x1_1.max(x1_2);
        let y1 = y1_1.max(y1_2);
        let x2 = x2_1.min(x2_2);
        let y2 = y2_1.min(y2_2);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        let area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        let union = area1 + area2 - intersection;

        intersection / union
    }

    /// Apply sigmoid function
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Apply softmax to a slice of values
    pub fn softmax(values: &mut [f32]) {
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;

        for val in values.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }

        for val in values.iter_mut() {
            *val /= sum;
        }
    }

    /// Clamp value to range
    pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
        value.max(min).min(max)
    }

    /// Linear interpolation
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
}

/// Logging utilities
pub mod logging {
    use tracing::{info, error};
    use tracing_subscriber::filter::LevelFilter;
    use std::time::Duration;

    /// Initialize logging system
    pub fn init_logging() -> crate::error::Result<()> {
        tracing_subscriber::fmt::init();
        Ok(())
    }

    /// Initialize logging system with custom level
    pub fn init_logging_with_level(level: LevelFilter) -> crate::error::Result<()> {
        tracing_subscriber::fmt()
            .with_max_level(level)
            .init();
        Ok(())
    }

    /// Initialize quiet logging (errors only)
    pub fn init_quiet_logging() -> crate::error::Result<()> {
        tracing_subscriber::fmt()
            .with_max_level(LevelFilter::ERROR)
            .init();
        Ok(())
    }

    /// Log inference performance
    pub fn log_inference_performance(
        model_name: &str,
        inference_time: Duration,
        preprocessing_time: Duration,
        postprocessing_time: Duration,
    ) {
        info!(
            model = model_name,
            inference_ms = inference_time.as_millis(),
            preprocessing_ms = preprocessing_time.as_millis(),
            postprocessing_ms = postprocessing_time.as_millis(),
            "Inference completed"
        );
    }

    /// Log error with context
    pub fn log_error_with_context(error: &crate::error::UocvrError, context: &str) {
        error!(error = %error, context = context, "Operation failed");
    }

    /// Log model loading
    pub fn log_model_loading(model_path: &str, model_type: &str) {
        info!(
            model_path = model_path,
            model_type = model_type,
            "Loading model"
        );
    }
}

/// Configuration utilities
pub mod config {
    use std::path::Path;
    use crate::core::ModelInfo;

    /// Load configuration from YAML file
    pub fn load_yaml_config<P: AsRef<Path>>(path: P) -> crate::error::Result<ModelInfo> {
        let _content = std::fs::read_to_string(path)?;
        // TODO: Implement YAML parsing when serde support is added
        Err(crate::error::UocvrError::runtime("YAML loading not implemented"))
    }

    /// Load configuration from JSON file
    pub fn load_json_config<P: AsRef<Path>>(path: P) -> crate::error::Result<ModelInfo> {
        let _content = std::fs::read_to_string(path)?;
        // TODO: Implement JSON parsing when serde support is added
        Err(crate::error::UocvrError::runtime("JSON loading not implemented"))
    }

    /// Save configuration to YAML file
    pub fn save_yaml_config<P: AsRef<Path>>(
        config: &ModelInfo,
        path: P,
    ) -> crate::error::Result<()> {
        let _path = path;
        let _config = config;
        // TODO: Implement YAML serialization when serde support is added
        Err(crate::error::UocvrError::runtime("YAML saving not implemented"))
    }

    /// Save configuration to JSON file
    pub fn save_json_config<P: AsRef<Path>>(
        config: &ModelInfo,
        path: P,
    ) -> crate::error::Result<()> {
        let _path = path;
        let _config = config;
        // TODO: Implement JSON serialization when serde support is added
        Err(crate::error::UocvrError::runtime("JSON saving not implemented"))
    }

    /// Validate configuration file
    pub fn validate_config_file<P: AsRef<Path>>(_path: P) -> crate::error::Result<()> {
        // Implementation will be added in the actual build
        todo!("validate_config_file implementation")
    }
}

/// Async utilities
pub mod async_utils {
    use std::future::Future;
    use std::time::Duration;

    /// Timeout wrapper for async operations
    pub async fn with_timeout<F, T>(
        future: F,
        timeout: Duration,
    ) -> crate::error::Result<T>
    where
        F: Future<Output = crate::error::Result<T>>,
    {
        tokio::time::timeout(timeout, future)
            .await
            .map_err(|_| crate::error::UocvrError::runtime("Operation timed out"))?
    }

    /// Retry wrapper for async operations
    pub async fn with_retry<F, T, Fut>(
        mut operation: F,
        max_retries: usize,
        delay: Duration,
    ) -> crate::error::Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = crate::error::Result<T>>,
    {
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(err) => {
                    last_error = Some(err);
                    if attempt < max_retries {
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }
}

impl Timer {
    /// Create a new timer with a label
    pub fn new(label: String) -> Self {
        Self {
            start: Instant::now(),
            label,
        }
    }

    /// Stop the timer and return the elapsed duration
    pub fn stop(self) -> Duration {
        let elapsed = self.start.elapsed();
        tracing::debug!(timer = %self.label, elapsed_ms = elapsed.as_millis(), "Timer stopped");
        elapsed
    }

    /// Get elapsed time without stopping the timer
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
        }
    }

    /// Update current memory usage
    pub fn update(&mut self, usage: usize) {
        self.current_usage = usage;
        if usage > self.peak_usage {
            self.peak_usage = usage;
        }
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Reset tracking
    pub fn reset(&mut self) {
        self.peak_usage = 0;
        self.current_usage = 0;
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Constants used throughout the library
pub mod constants {
    /// Default confidence threshold
    pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.70;

    /// Default NMS threshold
    pub const DEFAULT_NMS_THRESHOLD: f32 = 0.45;

    /// Default maximum detections
    pub const DEFAULT_MAX_DETECTIONS: usize = 300;

    /// Default input size for YOLO models
    pub const DEFAULT_YOLO_INPUT_SIZE: (u32, u32) = (640, 640);

    /// Default padding value for letterbox resize
    pub const DEFAULT_PADDING_VALUE: f32 = 0.447; // 114/255

    /// ImageNet normalization constants
    pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    /// COCO class count
    pub const COCO_CLASS_COUNT: usize = 80;

    /// Common execution provider names
    pub const CPU_PROVIDER: &str = "CPUExecutionProvider";
    pub const CUDA_PROVIDER: &str = "CUDAExecutionProvider";
    pub const TENSORRT_PROVIDER: &str = "TensorrtExecutionProvider";
}