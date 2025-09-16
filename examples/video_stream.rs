//! Video streaming classification example
//!
//! This example demonstrates real-time video classification using frame buffers
//! and streaming input, simulating a live video feed or webcam input.

use onnx_vision_classifier::{classifier, ClassifierBuilder, Result};
use image::{RgbImage, Rgb};
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    println!("ONNX Vision Classifier - Video Streaming Example");
    println!("==============================================");

    // Example 1: Real-time action recognition
    println!("1. Real-time Action Recognition Streaming");
    realtime_action_recognition_example()?;

    println!();

    // Example 2: Sliding window classification
    println!("2. Sliding Window Classification");
    sliding_window_example()?;

    println!();

    // Example 3: Two-stream classification (RGB + Optical Flow)
    println!("3. Two-Stream Classification");
    two_stream_example()?;

    println!();

    // Example 4: Adaptive frame rate processing
    println!("4. Adaptive Frame Rate Processing");
    adaptive_framerate_example()?;

    Ok(())
}

/// Example of real-time action recognition with streaming frames
fn realtime_action_recognition_example() -> Result<()> {
    println!("Initializing real-time action recognition...");
    
    // Configure classifier for 16-frame action recognition
    let classifier_config = classifier()
        .model_path("models/action_recognition.onnx")
        .multi_frame(16)
        .input_size(224, 224)
        .imagenet_normalization()
        .class_names(vec![
            "walking".to_string(),
            "running".to_string(),
            "jumping".to_string(),
            "waving".to_string(),
            "sitting".to_string(),
        ]);

    // Simulate building the classifier
    println!("Classifier configured:");
    println!("  - Model type: Multi-frame (16 frames)");
    println!("  - Input size: 224x224");
    println!("  - Buffer size: 32 frames (2x required)");
    
    // Simulate streaming video processing
    simulate_video_stream(16, "action_recognition")?;

    Ok(())
}

/// Example of sliding window classification
fn sliding_window_example() -> Result<()> {
    println!("Initializing sliding window classification...");

    // Configure classifier with variable frame support
    let classifier_config = classifier()
        .model_path("models/sliding_window.onnx")
        .variable_frames(8, 24)
        .input_size(112, 112)
        .imagenet_normalization();

    println!("Sliding window classifier configured:");
    println!("  - Frame range: 8-24 frames");
    println!("  - Overlap: 4 frames");
    println!("  - Processing rate: Every 4 new frames");

    // Simulate sliding window processing
    simulate_sliding_window_processing()?;

    Ok(())
}

/// Example of two-stream classification (RGB + Optical Flow)
fn two_stream_example() -> Result<()> {
    println!("Initializing two-stream classification...");

    let classifier_config = classifier()
        .model_path("models/two_stream.onnx")
        .two_stream()
        .input_size(224, 224)
        .imagenet_normalization();

    println!("Two-stream classifier configured:");
    println!("  - RGB stream: Current frame");
    println!("  - Flow stream: Optical flow from previous frame");
    println!("  - Buffer size: 4 frames");

    // Simulate two-stream processing
    simulate_two_stream_processing()?;

    Ok(())
}

/// Example of adaptive frame rate processing
fn adaptive_framerate_example() -> Result<()> {
    println!("Initializing adaptive frame rate processing...");

    let classifier_config = classifier()
        .model_path("models/adaptive.onnx")
        .multi_frame(8)
        .input_size(224, 224)
        .imagenet_normalization();

    println!("Adaptive classifier configured:");
    println!("  - Base requirement: 8 frames");
    println!("  - Adaptive sampling based on motion");
    println!("  - Dynamic frame rate: 5-30 FPS");

    // Simulate adaptive processing
    simulate_adaptive_processing()?;

    Ok(())
}

/// Simulate a video stream with frame-by-frame processing
fn simulate_video_stream(required_frames: usize, stream_type: &str) -> Result<()> {
    println!("\nStarting video stream simulation...");
    
    let mut frame_count = 0;
    let start_time = Instant::now();
    let target_fps = 15.0; // 15 FPS
    let frame_duration = Duration::from_secs_f64(1.0 / target_fps);
    
    // Simulate classifier with frame buffer
    let mut buffer_ready_count = 0;
    
    for i in 0..100 { // Simulate 100 frames
        let frame = generate_stream_frame(i, 224, 224, stream_type);
        frame_count += 1;
        
        // Simulate adding frame to buffer
        let buffer_frame_count = (frame_count).min(required_frames * 2);
        let is_ready = buffer_frame_count >= required_frames;
        
        if is_ready {
            buffer_ready_count += 1;
            
            // Simulate classification every few frames
            if buffer_ready_count % 4 == 0 {
                let elapsed = start_time.elapsed();
                let actual_fps = frame_count as f64 / elapsed.as_secs_f64();
                
                println!("Frame {}: Buffer ready, classifying... (FPS: {:.1})", 
                         frame_count, actual_fps);
                
                // Simulate classification result
                match stream_type {
                    "action_recognition" => {
                        let action = match (i / 20) % 4 {
                            0 => "walking",
                            1 => "running", 
                            2 => "jumping",
                            _ => "waving",
                        };
                        println!("  -> Detected action: {} (confidence: {:.3})", 
                                action, 0.8 + (i % 10) as f32 * 0.02);
                    },
                    _ => {
                        println!("  -> Classification complete");
                    }
                }
            }
        } else {
            if frame_count % 10 == 0 {
                println!("Frame {}: Building buffer... ({}/{} frames)", 
                         frame_count, buffer_frame_count, required_frames);
            }
        }
        
        // Simulate real-time processing delay
        thread::sleep(frame_duration / 4); // Speed up for demo
    }
    
    let total_time = start_time.elapsed();
    let avg_fps = frame_count as f64 / total_time.as_secs_f64();
    
    println!("\nStream simulation complete:");
    println!("  - Total frames: {}", frame_count);
    println!("  - Average FPS: {:.1}", avg_fps);
    println!("  - Classifications: {}", buffer_ready_count / 4);

    Ok(())
}

/// Simulate sliding window processing
fn simulate_sliding_window_processing() -> Result<()> {
    println!("\nStarting sliding window simulation...");
    
    let window_size = 16;
    let step_size = 4; // Overlap of 12 frames
    let mut frames_buffer = Vec::new();
    
    for i in 0..50 {
        let frame = generate_stream_frame(i, 112, 112, "sliding_window");
        frames_buffer.push(frame);
        
        // Process when we have enough frames and at step intervals
        if frames_buffer.len() >= window_size && (i % step_size == 0) {
            let window_start = frames_buffer.len().saturating_sub(window_size);
            println!("Processing window: frames {}-{}", 
                     window_start, window_start + window_size - 1);
            
            // Simulate classification
            let confidence = 0.7 + ((i as f32 * 0.1).sin() + 1.0) * 0.15;
            println!("  -> Activity detected (confidence: {:.3})", confidence);
        }
        
        // Keep buffer from growing too large
        if frames_buffer.len() > window_size * 2 {
            frames_buffer.drain(0..step_size);
        }
        
        thread::sleep(Duration::from_millis(50)); // Simulate processing time
    }

    Ok(())
}

/// Simulate two-stream processing
fn simulate_two_stream_processing() -> Result<()> {
    println!("\nStarting two-stream simulation...");
    
    let mut previous_frame: Option<RgbImage> = None;
    
    for i in 0..30 {
        let current_frame = generate_stream_frame(i, 224, 224, "two_stream");
        
        if let Some(prev_frame) = previous_frame.as_ref() {
            println!("Processing frame pair {}-{}", i-1, i);
            
            // Simulate optical flow computation
            let flow_magnitude = compute_mock_optical_flow(&prev_frame, &current_frame);
            
            // Simulate classification
            let rgb_confidence = 0.6 + (i as f32 * 0.05).sin() * 0.2;
            let flow_confidence = 0.5 + flow_magnitude * 0.3;
            let combined_confidence = (rgb_confidence + flow_confidence) / 2.0;
            
            println!("  -> RGB confidence: {:.3}", rgb_confidence);
            println!("  -> Flow confidence: {:.3}", flow_confidence);
            println!("  -> Combined: {:.3}", combined_confidence);
            
            let action = if combined_confidence > 0.7 {
                "dynamic_action"
            } else {
                "static_pose"
            };
            println!("  -> Detected: {}", action);
        } else {
            println!("Frame {}: Initializing (need previous frame for flow)", i);
        }
        
        previous_frame = Some(current_frame);
        thread::sleep(Duration::from_millis(100));
    }

    Ok(())
}

/// Simulate adaptive frame rate processing
fn simulate_adaptive_processing() -> Result<()> {
    println!("\nStarting adaptive processing simulation...");
    
    let mut current_fps = 15.0;
    let base_frame_duration = Duration::from_secs_f64(1.0 / 30.0); // 30 FPS input
    
    for i in 0..60 {
        let frame = generate_stream_frame(i, 224, 224, "adaptive");
        
        // Simulate motion detection
        let motion_level = ((i as f32 * 0.2).sin().abs() + 0.2).min(1.0);
        
        // Adapt processing rate based on motion
        let target_fps = 5.0 + motion_level * 25.0; // 5-30 FPS range
        current_fps = current_fps * 0.8 + target_fps * 0.2; // Smooth adaptation
        
        let should_process = (i as f32) % (30.0 / current_fps) < 1.0;
        
        if should_process {
            println!("Frame {}: Processing (motion: {:.2}, fps: {:.1})", 
                     i, motion_level, current_fps);
            
            if motion_level > 0.7 {
                println!("  -> High motion detected: dynamic_scene");
            } else if motion_level > 0.3 {
                println!("  -> Moderate motion: transitioning");
            } else {
                println!("  -> Low motion: static_scene");
            }
        }
        
        thread::sleep(base_frame_duration / 2); // Speed up for demo
    }

    Ok(())
}

/// Generate a frame for streaming simulation
fn generate_stream_frame(frame_idx: usize, width: u32, height: u32, stream_type: &str) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        match stream_type {
            "action_recognition" => {
                // Simulate different actions over time
                let action_phase = (frame_idx / 20) % 4;
                let time_in_action = frame_idx % 20;
                
                match action_phase {
                    0 => { // Walking
                        let walk_cycle = (time_in_action as f32 * 0.3).sin();
                        let r = (128.0 + walk_cycle * 50.0) as u8;
                        let g = 100;
                        let b = (y as f32 / height as f32 * 255.0) as u8;
                        Rgb([r, g, b])
                    },
                    1 => { // Running
                        let run_cycle = (time_in_action as f32 * 0.8).sin();
                        let r = 150;
                        let g = (128.0 + run_cycle * 100.0) as u8;
                        let b = (x as f32 / width as f32 * 255.0) as u8;
                        Rgb([r, g, b])
                    },
                    2 => { // Jumping
                        let jump_height = (time_in_action as f32 * 0.5).sin().max(0.0);
                        let r = (jump_height * 255.0) as u8;
                        let g = 200;
                        let b = 100;
                        Rgb([r, g, b])
                    },
                    _ => { // Waving
                        let wave = (time_in_action as f32 * 0.4 + x as f32 * 0.1).sin();
                        let r = 100;
                        let g = 150;
                        let b = (128.0 + wave * 127.0) as u8;
                        Rgb([r, g, b])
                    }
                }
            },
            "two_stream" => {
                // Generate frame with motion patterns for optical flow
                let motion_x = (frame_idx as f32 * 0.1).sin() * 10.0;
                let motion_y = (frame_idx as f32 * 0.15).cos() * 5.0;
                
                let effective_x = (x as f32 + motion_x) % width as f32;
                let effective_y = (y as f32 + motion_y) % height as f32;
                
                let r = (effective_x / width as f32 * 255.0) as u8;
                let g = (effective_y / height as f32 * 255.0) as u8;
                let b = (frame_idx * 5) as u8 % 255;
                
                Rgb([r, g, b])
            },
            "adaptive" => {
                // Generate frame with varying motion levels
                let motion_intensity = ((frame_idx as f32 * 0.2).sin().abs() + 0.2).min(1.0);
                let noise_level = motion_intensity * 50.0;
                
                let base_r = (x as f32 / width as f32 * 255.0) as u8;
                let base_g = (y as f32 / height as f32 * 255.0) as u8;
                let base_b = 128;
                
                let noise_r = ((frame_idx + x as usize) as f32 * noise_level).sin() * 30.0;
                let noise_g = ((frame_idx + y as usize) as f32 * noise_level).cos() * 30.0;
                
                let r = (base_r as f32 + noise_r).clamp(0.0, 255.0) as u8;
                let g = (base_g as f32 + noise_g).clamp(0.0, 255.0) as u8;
                let b = base_b;
                
                Rgb([r, g, b])
            },
            _ => {
                // Default pattern
                let r = (x as f32 / width as f32 * 255.0) as u8;
                let g = (y as f32 / height as f32 * 255.0) as u8;
                let b = (frame_idx * 10) as u8 % 255;
                Rgb([r, g, b])
            }
        }
    })
}

/// Mock optical flow computation for demonstration
fn compute_mock_optical_flow(frame1: &RgbImage, frame2: &RgbImage) -> f32 {
    // Simple mock implementation - compute average pixel difference
    let (width, height) = frame1.dimensions();
    let mut total_diff = 0.0;
    let mut pixel_count = 0;
    
    // Sample a subset of pixels for efficiency
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            let pixel1 = frame1.get_pixel(x, y);
            let pixel2 = frame2.get_pixel(x, y);
            
            let diff = ((pixel1[0] as f32 - pixel2[0] as f32).abs() +
                       (pixel1[1] as f32 - pixel2[1] as f32).abs() +
                       (pixel1[2] as f32 - pixel2[2] as f32).abs()) / 3.0;
            
            total_diff += diff;
            pixel_count += 1;
        }
    }
    
    if pixel_count > 0 {
        (total_diff / pixel_count as f32) / 255.0 // Normalize to 0-1
    } else {
        0.0
    }
}