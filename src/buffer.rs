//! Frame buffer management for multi-frame models.

use std::collections::VecDeque;
use image::RgbImage;
use crate::{Result, ClassificationError};

/// A circular buffer for managing video frames
#[derive(Debug)]
pub struct FrameBuffer {
    frames: VecDeque<RgbImage>,
    capacity: usize,
    required_frames: usize,
}

impl FrameBuffer {
    /// Create a new frame buffer
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of frames to store
    /// * `required_frames` - Minimum number of frames needed for classification
    pub fn new(capacity: usize, required_frames: usize) -> Self {
        assert!(capacity >= required_frames, "Capacity must be >= required frames");
        
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
            required_frames,
        }
    }
    
    /// Add a new frame to the buffer
    ///
    /// If the buffer is at capacity, the oldest frame will be removed.
    pub fn push(&mut self, frame: RgbImage) {
        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }
    
    /// Check if the buffer has enough frames for classification
    pub fn is_ready(&self) -> bool {
        self.frames.len() >= self.required_frames
    }
    
    /// Get the latest N frames
    ///
    /// Returns the most recent `count` frames in chronological order.
    pub fn get_latest_frames(&self, count: usize) -> Result<Vec<&RgbImage>> {
        if self.frames.len() < count {
            return Err(ClassificationError::InsufficientFrames {
                expected: count,
                actual: self.frames.len(),
            });
        }
        
        Ok(self.frames.iter().rev().take(count).rev().collect())
    }
    
    /// Get all frames in the buffer
    pub fn get_all_frames(&self) -> Vec<&RgbImage> {
        self.frames.iter().collect()
    }
    
    /// Get frames at specific indices
    pub fn get_frames_at_indices(&self, indices: &[usize]) -> Result<Vec<&RgbImage>> {
        let mut result = Vec::with_capacity(indices.len());
        
        for &idx in indices {
            if idx >= self.frames.len() {
                return Err(ClassificationError::InvalidInput(
                    format!("Frame index {} out of bounds (have {} frames)", idx, self.frames.len())
                ));
            }
            result.push(&self.frames[idx]);
        }
        
        Ok(result)
    }
    
    /// Sample frames uniformly from the buffer
    pub fn sample_uniform(&self, count: usize) -> Result<Vec<&RgbImage>> {
        if self.frames.len() < count {
            return Err(ClassificationError::InsufficientFrames {
                expected: count,
                actual: self.frames.len(),
            });
        }
        
        let step = self.frames.len() as f32 / count as f32;
        let indices: Vec<usize> = (0..count)
            .map(|i| (i as f32 * step) as usize)
            .collect();
        
        self.get_frames_at_indices(&indices)
    }
    
    /// Clear all frames from the buffer
    pub fn clear(&mut self) {
        self.frames.clear();
    }
    
    /// Get current number of frames in buffer
    pub fn len(&self) -> usize {
        self.frames.len()
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
    
    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get required frame count
    pub fn required_frames(&self) -> usize {
        self.required_frames
    }
    
    /// Clone the latest frames as owned images
    pub fn clone_latest_frames(&self, count: usize) -> Result<Vec<RgbImage>> {
        let frame_refs = self.get_latest_frames(count)?;
        Ok(frame_refs.into_iter().cloned().collect())
    }
}

impl Clone for FrameBuffer {
    fn clone(&self) -> Self {
        Self {
            frames: self.frames.clone(),
            capacity: self.capacity,
            required_frames: self.required_frames,
        }
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
    fn test_frame_buffer_basic() {
        let mut buffer = FrameBuffer::new(5, 3);
        
        assert_eq!(buffer.len(), 0);
        assert!(!buffer.is_ready());
        assert!(buffer.is_empty());
        
        // Add frames
        for i in 0..3 {
            let img = create_test_image(10, 10, [i as u8, 0, 0]);
            buffer.push(img);
        }
        
        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_ready());
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_frame_buffer_overflow() {
        let mut buffer = FrameBuffer::new(3, 2);
        
        // Add more frames than capacity
        for i in 0..5 {
            let img = create_test_image(10, 10, [i as u8, 0, 0]);
            buffer.push(img);
        }
        
        // Should only have 3 frames (capacity)
        assert_eq!(buffer.len(), 3);
        
        // Should have the latest 3 frames (2, 3, 4)
        let frames = buffer.get_all_frames();
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_get_latest_frames() {
        let mut buffer = FrameBuffer::new(5, 3);
        
        for i in 0..4 {
            let img = create_test_image(10, 10, [i as u8, 0, 0]);
            buffer.push(img);
        }
        
        let latest_2 = buffer.get_latest_frames(2).unwrap();
        assert_eq!(latest_2.len(), 2);
        
        // Should fail if asking for more frames than available
        assert!(buffer.get_latest_frames(6).is_err());
    }

    #[test]
    fn test_sample_uniform() {
        let mut buffer = FrameBuffer::new(10, 5);
        
        for i in 0..8 {
            let img = create_test_image(10, 10, [i as u8, 0, 0]);
            buffer.push(img);
        }
        
        let sampled = buffer.sample_uniform(4).unwrap();
        assert_eq!(sampled.len(), 4);
        
        // Should fail if asking for more frames than available
        assert!(buffer.sample_uniform(10).is_err());
    }
}