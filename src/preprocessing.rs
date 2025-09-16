//! Image preprocessing utilities for vision models.

use image::{RgbImage, imageops};
use ndarray::{Array3, Array4, Array5};
use crate::{ImageNormalization, types::PreprocessingConfig, Result, ClassificationError};

/// Trait for image preprocessing operations
pub trait ImagePreprocessor: Send + Sync {
    /// Preprocess a single frame for single-frame models
    fn preprocess_single_frame(
        &self,
        frame: &RgbImage,
        target_size: (u32, u32),
        normalization: &ImageNormalization,
    ) -> Result<Array3<f32>>;
    
    /// Preprocess multiple frames for batch processing
    fn preprocess_multiple_frames(
        &self,
        frames: &[&RgbImage],
        target_size: (u32, u32),
        normalization: &ImageNormalization,
    ) -> Result<Array4<f32>>;
    
    /// Preprocess video clip for 3D CNNs
    fn preprocess_video_clip(
        &self,
        frames: &[&RgbImage],
        target_size: (u32, u32),
        normalization: &ImageNormalization,
    ) -> Result<Array5<f32>>;
}

/// Default preprocessing implementation
pub struct DefaultPreprocessor {
    config: PreprocessingConfig,
}

impl DefaultPreprocessor {
    /// Create a new default preprocessor
    pub fn new() -> Self {
        Self {
            config: PreprocessingConfig::default(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: PreprocessingConfig) -> Self {
        Self { config }
    }
    
    /// Resize an image to target size
    fn resize_image(&self, image: &RgbImage, target_size: (u32, u32)) -> RgbImage {
        if !self.config.resize {
            return image.clone();
        }
        
        let (target_width, target_height) = target_size;
        let (current_width, current_height) = image.dimensions();
        
        if current_width == target_width && current_height == target_height {
            return image.clone();
        }
        
        if self.config.center_crop {
            // Resize maintaining aspect ratio, then center crop
            let scale_w = target_width as f32 / current_width as f32;
            let scale_h = target_height as f32 / current_height as f32;
            let scale = scale_w.max(scale_h);
            
            let new_width = (current_width as f32 * scale) as u32;
            let new_height = (current_height as f32 * scale) as u32;
            
            let resized = imageops::resize(image, new_width, new_height, self.config.filter_type);
            
            // Center crop
            let crop_x = (new_width.saturating_sub(target_width)) / 2;
            let crop_y = (new_height.saturating_sub(target_height)) / 2;
            
            imageops::crop_imm(&resized, crop_x, crop_y, target_width, target_height).to_image()
        } else {
            // Simple resize without maintaining aspect ratio
            imageops::resize(image, target_width, target_height, self.config.filter_type)
        }
    }
    
    /// Convert image to normalized tensor
    fn image_to_tensor(
        &self,
        image: &RgbImage,
        target_size: (u32, u32),
        normalization: &ImageNormalization,
    ) -> Result<Array3<f32>> {
        let resized = self.resize_image(image, target_size);
        let (width, height) = resized.dimensions();
        
        let mut array = Array3::<f32>::zeros((3, height as usize, width as usize));
        
        for y in 0..height {
            for x in 0..width {
                let pixel = resized.get_pixel(x, y);
                
                if self.config.normalize {
                    array[[0, y as usize, x as usize]] = normalization.normalize_pixel(pixel[0] as f32 / 255.0, 0);
                    array[[1, y as usize, x as usize]] = normalization.normalize_pixel(pixel[1] as f32 / 255.0, 1);
                    array[[2, y as usize, x as usize]] = normalization.normalize_pixel(pixel[2] as f32 / 255.0, 2);
                } else {
                    array[[0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
                    array[[1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
                    array[[2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
                }
            }
        }
        
        Ok(array)
    }
}

impl Default for DefaultPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ImagePreprocessor for DefaultPreprocessor {
    fn preprocess_single_frame(
        &self,
        frame: &RgbImage,
        target_size: (u32, u32),
        normalization: &ImageNormalization,
    ) -> Result<Array3<f32>> {
        self.image_to_tensor(frame, target_size, normalization)
    }
    
    fn preprocess_multiple_frames(
        &self,
        frames: &[&RgbImage],
        target_size: (u32, u32),
        normalization: &ImageNormalization,
    ) -> Result<Array4<f32>> {
        if frames.is_empty() {
            return Err(ClassificationError::InvalidInput("No frames provided".to_string()));
        }
        
        let num_frames = frames.len();
        let (width, height) = target_size;
        let mut array = Array4::<f32>::zeros((num_frames, 3, height as usize, width as usize));
        
        for (i, frame) in frames.iter().enumerate() {
            let processed = self.image_to_tensor(frame, target_size, normalization)?;
            array.slice_mut(ndarray::s![i, .., .., ..]).assign(&processed);
        }
        
        Ok(array)
    }
    
    fn preprocess_video_clip(
        &self,
        frames: &[&RgbImage],
        target_size: (u32, u32),
        normalization: &ImageNormalization,
    ) -> Result<Array5<f32>> {
        if frames.is_empty() {
            return Err(ClassificationError::InvalidInput("No frames provided".to_string()));
        }
        
        let num_frames = frames.len();
        let (width, height) = target_size;
        let mut array = Array5::<f32>::zeros((1, 3, num_frames, height as usize, width as usize));
        
        for (i, frame) in frames.iter().enumerate() {
            let processed = self.image_to_tensor(frame, target_size, normalization)?;
            array.slice_mut(ndarray::s![0, .., i, .., ..]).assign(&processed);
        }
        
        Ok(array)
    }
}

/// Advanced preprocessing with additional features
pub struct AdvancedPreprocessor {
    base: DefaultPreprocessor,
    apply_augmentation: bool,
}

impl AdvancedPreprocessor {
    /// Create a new advanced preprocessor
    pub fn new() -> Self {
        Self {
            base: DefaultPreprocessor::new(),
            apply_augmentation: false,
        }
    }
    
    /// Enable data augmentation (for training scenarios)
    pub fn with_augmentation(mut self) -> Self {
        self.apply_augmentation = true;
        self
    }
    
    /// Apply simple data augmentation (horizontal flip)
    fn augment_if_enabled(&self, image: &RgbImage) -> RgbImage {
        // Simple random flip - in real usage you'd use a proper RNG
        if self.apply_augmentation && (image.width() % 2 == 0) {
            imageops::flip_horizontal(image)
        } else {
            image.clone()
        }
    }
}

impl Default for AdvancedPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ImagePreprocessor for AdvancedPreprocessor {
    fn preprocess_single_frame(
        &self,
        frame: &RgbImage,
        target_size: (u32, u32),
        normalization: &ImageNormalization,
    ) -> Result<Array3<f32>> {
        let augmented = self.augment_if_enabled(frame);
        self.base.preprocess_single_frame(&augmented, target_size, normalization)
    }
    
    fn preprocess_multiple_frames(
        &self,
        frames: &[&RgbImage],
        target_size: (u32, u32),
        normalization: &ImageNormalization,
    ) -> Result<Array4<f32>> {
        // For consistency, don't augment in multi-frame scenarios
        self.base.preprocess_multiple_frames(frames, target_size, normalization)
    }
    
    fn preprocess_video_clip(
        &self,
        frames: &[&RgbImage],
        target_size: (u32, u32),
        normalization: &ImageNormalization,
    ) -> Result<Array5<f32>> {
        // For consistency, don't augment in video scenarios
        self.base.preprocess_video_clip(frames, target_size, normalization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{RgbImage, Rgb};

    fn create_test_image(width: u32, height: u32, color: [u8; 3]) -> RgbImage {
        RgbImage::from_fn(width, height, |_, _| Rgb(color))
    }

    #[test]
    fn test_default_preprocessor() {
        let preprocessor = DefaultPreprocessor::new();
        let image = create_test_image(100, 100, [255, 0, 0]);
        let normalization = ImageNormalization::default();
        
        let result = preprocessor.preprocess_single_frame(&image, (224, 224), &normalization);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[3, 224, 224]);
    }

    #[test]
    fn test_multiple_frames_preprocessing() {
        let preprocessor = DefaultPreprocessor::new();
        let frames: Vec<RgbImage> = (0..4)
            .map(|i| create_test_image(50, 50, [i * 50, 0, 0]))
            .collect();
        let frame_refs: Vec<&RgbImage> = frames.iter().collect();
        let normalization = ImageNormalization::default();
        
        let result = preprocessor.preprocess_multiple_frames(&frame_refs, (112, 112), &normalization);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[4, 3, 112, 112]);
    }

    #[test]
    fn test_video_clip_preprocessing() {
        let preprocessor = DefaultPreprocessor::new();
        let frames: Vec<RgbImage> = (0..8)
            .map(|i| create_test_image(64, 64, [i * 30, 0, 0]))
            .collect();
        let frame_refs: Vec<&RgbImage> = frames.iter().collect();
        let normalization = ImageNormalization::default();
        
        let result = preprocessor.preprocess_video_clip(&frame_refs, (128, 128), &normalization);
        assert!(result.is_ok());
        
        let tensor = result.unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 8, 128, 128]);
    }
}