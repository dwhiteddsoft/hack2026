//! Multi-frame video classification example
//!
//! This example demonstrates how to use the ONNX vision classifier for
//! multi-frame video classification tasks like action recognition models.

use onnx_vision_classifier::{classifier, ClassifierBuilder, Result};
use image::{RgbImage, Rgb};

fn main() -> Result<()> {
    println!("ONNX Vision Classifier - Multi-Frame Example");
    println!("==========================================");

    // Example 1: I3D-style action recognition model
    println!("1. I3D Action Recognition Example");
    i3d_example()?;
    
    // add a blank line for readability
    println!();
    
    // Example 2: SlowFast model example
    slowfast_example()?;
    
    // add a blank line for readability
    println!();
    
    // Example 3: Custom multi-frame model
    custom_multiframe_example()?;
    
    // add a blank line for readability
    println!();
    
    // Example 4: Variable frame count model

    // Example 4: Variable frame count model
    println!("4. Variable Frame Count Model Example");
    variable_frame_example()?;

    Ok(())
}

/// Example using I3D-style model (64 frames)
fn i3d_example() -> Result<()> {
    println!("Building I3D classifier (64 frames, 224x224)...");

    // I3D typically uses 64 frames sampled from a video clip
    let classifier_config = ClassifierBuilder::i3d_classifier("models/i3d_kinetics400.onnx")
        .with_kinetics400_classes();

    // Generate synthetic video frames for demonstration
    let frames = generate_synthetic_video_frames(64, 224, 224, "walking_motion")?;
    
    println!("Generated {} frames for I3D classification", frames.len());
    println!("Frame dimensions: {}x{}", frames[0].width(), frames[0].height());

    // In a real scenario with a model file:
    /*
    let classifier = classifier_config.build()?;
    let frame_refs: Vec<&RgbImage> = frames.iter().collect();
    let result = classifier.classify_frame_refs(&frame_refs)?;
    
    println!("Action classification result:");
    println!("  Action: {}", result.class_name.unwrap_or("Unknown".to_string()));
    println!("  Confidence: {:.4}", result.confidence);
    */

    println!("Configuration complete - would classify 64-frame sequence");

    Ok(())
}

/// Example using SlowFast model architecture
fn slowfast_example() -> Result<()> {
    println!("Building SlowFast classifier...");

    // SlowFast uses different temporal sampling rates
    // Fast pathway: many frames at low resolution
    // Slow pathway: few frames at high resolution
    
    let classifier_config = ClassifierBuilder::slowfast_classifier("models/slowfast_r50.onnx")
        .with_kinetics400_classes();

    // Generate frames for both pathways
    let fast_frames = generate_synthetic_video_frames(32, 224, 224, "fast_motion")?;
    let slow_frames = sample_frames_uniformly(&fast_frames, 8)?; // Subsample for slow pathway

    println!("Fast pathway: {} frames", fast_frames.len());
    println!("Slow pathway: {} frames", slow_frames.len());

    // Note: This is a simplified example. Real SlowFast models have more complex input requirements
    println!("Configuration complete - would process dual-pathway video");

    Ok(())
}

/// Example with custom multi-frame model
fn custom_multiframe_example() -> Result<()> {
    println!("Building custom multi-frame classifier (16 frames)...");

    let _classifier_config = classifier()
        .model_path("models/custom_action_model.onnx")
        .multi_frame(16)
        .input_size(112, 112) // Smaller input size for efficiency
        .imagenet_normalization()
        .class_names(vec![
            "walking".to_string(),
            "running".to_string(),
            "jumping".to_string(),
            "sitting".to_string(),
            "standing".to_string(),
        ]);

    let frames = generate_synthetic_video_frames(16, 112, 112, "jumping_motion")?;
    
    println!("Generated {} frames at {}x{}", frames.len(), 112, 112);
    println!("Configuration complete - would classify custom 16-frame sequence");

    Ok(())
}

/// Example with variable frame count model
fn variable_frame_example() -> Result<()> {
    println!("Building variable frame count classifier (8-32 frames)...");

    let _classifier_config = classifier()
        .model_path("models/variable_frame_model.onnx")
        .variable_frames(8, 32) // Can handle 8 to 32 frames
        .input_size(224, 224)
        .imagenet_normalization();

    // Test with different frame counts
    for frame_count in [8, 16, 24, 32] {
        let frames = generate_synthetic_video_frames(frame_count, 224, 224, "variable_motion")?;
        println!("  Generated sequence with {} frames", frames.len());
        
        // In real usage:
        // let result = classifier.classify_frames(&frames)?;
    }

    println!("Configuration complete - would handle variable-length sequences");

    Ok(())
}

/// Generate synthetic video frames for demonstration
fn generate_synthetic_video_frames(
    count: usize, 
    width: u32, 
    height: u32, 
    motion_type: &str
) -> Result<Vec<RgbImage>> {
    let mut frames = Vec::with_capacity(count);
    
    for i in 0..count {
        let frame = match motion_type {
            "walking_motion" => generate_walking_frame(i, width, height),
            "fast_motion" => generate_fast_motion_frame(i, width, height),
            "jumping_motion" => generate_jumping_frame(i, width, height),
            "variable_motion" => generate_variable_motion_frame(i, width, height),
            _ => generate_default_frame(i, width, height),
        };
        frames.push(frame);
    }
    
    Ok(frames)
}

/// Generate a frame simulating walking motion
fn generate_walking_frame(frame_idx: usize, width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        // Create a simple pattern that changes over time to simulate motion
        let time_factor = (frame_idx as f32 * 0.1).sin();
        let position_factor = (x as f32 / width as f32 + time_factor).sin();
        
        let r = (128.0 + 127.0 * position_factor) as u8;
        let g = (100 + frame_idx * 2) as u8 % 255;
        let b = (y as f32 / height as f32 * 255.0) as u8;
        
        Rgb([r, g, b])
    })
}

/// Generate a frame simulating fast motion
fn generate_fast_motion_frame(frame_idx: usize, width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        // Rapid changes to simulate fast motion
        let motion_blur = (frame_idx as f32 * 0.5).sin();
        let r = ((x as f32 + frame_idx as f32 * 5.0) % 255.0) as u8;
        let g = (128.0 + 127.0 * motion_blur) as u8;
        let b = ((y as f32 + frame_idx as f32 * 3.0) % 255.0) as u8;
        
        Rgb([r, g, b])
    })
}

/// Generate a frame simulating jumping motion
fn generate_jumping_frame(frame_idx: usize, width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        // Vertical motion pattern
        let jump_phase = (frame_idx as f32 * 0.2).sin();
        let vertical_shift = (jump_phase * 50.0) as i32;
        
        let adjusted_y = ((y as i32 + vertical_shift) % height as i32) as u32;
        
        let r = (x as f32 / width as f32 * 255.0) as u8;
        let g = (adjusted_y as f32 / height as f32 * 255.0) as u8;
        let b = (frame_idx * 10) as u8 % 255;
        
        Rgb([r, g, b])
    })
}

/// Generate a frame with variable motion
fn generate_variable_motion_frame(frame_idx: usize, width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        // Complex motion pattern
        let t = frame_idx as f32 * 0.1;
        let wave1 = (x as f32 * 0.1 + t).sin();
        let wave2 = (y as f32 * 0.1 + t * 1.5).cos();
        
        let r = (128.0 + 127.0 * wave1) as u8;
        let g = (128.0 + 127.0 * wave2) as u8;
        let b = (128.0 + 127.0 * (wave1 * wave2)) as u8;
        
        Rgb([r, g, b])
    })
}

/// Generate a default frame
fn generate_default_frame(frame_idx: usize, width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        let r = (x as f32 / width as f32 * 255.0) as u8;
        let g = (y as f32 / height as f32 * 255.0) as u8;
        let b = (frame_idx * 10) as u8 % 255;
        
        Rgb([r, g, b])
    })
}

/// Sample frames uniformly from a sequence
fn sample_frames_uniformly(frames: &[RgbImage], target_count: usize) -> Result<Vec<RgbImage>> {
    if frames.is_empty() {
        return Ok(Vec::new());
    }
    
    if target_count >= frames.len() {
        return Ok(frames.to_vec());
    }
    
    let step = frames.len() as f32 / target_count as f32;
    let sampled: Vec<RgbImage> = (0..target_count)
        .map(|i| {
            let index = (i as f32 * step) as usize;
            frames[index].clone()
        })
        .collect();
    
    Ok(sampled)
}

/// Example of loading video frames from actual video files
#[allow(dead_code)]
fn load_video_frames_example() -> Result<Vec<RgbImage>> {
    // In a real application, you would use a video processing library like ffmpeg
    // to extract frames from video files
    
    /*
    use ffmpeg_next as ffmpeg;
    
    ffmpeg::init().unwrap();
    
    let mut ictx = ffmpeg::format::input(&"path/to/video.mp4")?;
    let input = ictx.streams().best(ffmpeg::media::Type::Video).unwrap();
    let video_stream_index = input.index();
    
    let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
    let mut decoder = context_decoder.decoder().video()?;
    
    let mut frames = Vec::new();
    
    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;
            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                // Convert frame to RgbImage
                // frames.push(convert_frame_to_rgb_image(decoded));
            }
        }
    }
    */
    
    // For now, return synthetic frames
    generate_synthetic_video_frames(30, 224, 224, "walking_motion")
}