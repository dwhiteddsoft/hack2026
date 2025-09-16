# ONNX Vision Classifier

A comprehensive Rust library for ONNX-based image and video classification that supports both single-frame and multi-frame models. Perfect for computer vision applications, action recognition, and real-time video analysis.

[![Crates.io](https://img.shields.io/crates/v/onnx-vision-classifier)](https://crates.io/crates/onnx-vision-classifier)
[![Documentation](https://docs.rs/onnx-vision-classifier/badge.svg)](https://docs.rs/onnx-vision-classifier)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE)

## Features

- 🖼️ **Single-frame image classification** (ResNet, EfficientNet, Vision Transformers, etc.)
- 🎬 **Multi-frame video classification** (I3D, SlowFast, X3D, etc.)
- 🔄 **LSTM-based sequence classification**
- 🌊 **Two-stream models** (RGB + Optical Flow)
- 📺 **Real-time streaming** with automatic frame buffering
- 🛠️ **Flexible preprocessing** pipeline with custom preprocessors
- 🏗️ **Builder pattern** for easy configuration
- 🚀 **High performance** with optimized ONNX runtime
- 🧵 **Thread-safe** design for concurrent processing

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
onnx-vision-classifier = "0.1"
```

### Single Frame Classification

```rust
use onnx_vision_classifier::{classifier, RgbImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a ResNet classifier
    let classifier = classifier()
        .model_path("models/resnet50.onnx")
        .single_frame()
        .input_size(224, 224)
        .imagenet_normalization()
        .build()?;

    // Load and classify an image
    let image = image::open("test.jpg")?.to_rgb8();
    let result = classifier.classify_single(&image)?;

    println!("Predicted class: {} (confidence: {:.2})", 
             result.class_name.unwrap_or("Unknown".to_string()), 
             result.confidence);

    Ok(())
}
```

### Multi-Frame Video Classification

```rust
use onnx_vision_classifier::{classifier, RgbImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an I3D action recognition classifier
    let classifier = classifier()
        .model_path("models/i3d_kinetics400.onnx")
        .multi_frame(64)  // I3D uses 64 frames
        .input_size(224, 224)
        .imagenet_normalization()
        .build()?;

    // Load video frames
    let video_frames: Vec<RgbImage> = load_video_frames("video.mp4")?;
    
    // Classify the video sequence
    let result = classifier.classify_frames(&video_frames)?;
    
    println!("Detected action: {} (confidence: {:.2})", 
             result.class_name.unwrap_or("Unknown".to_string()), 
             result.confidence);

    Ok(())
}
```

### Real-time Streaming Classification

```rust
use onnx_vision_classifier::{classifier, RgbImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a streaming classifier
    let mut classifier = classifier()
        .model_path("models/action_recognition.onnx")
        .multi_frame(16)
        .input_size(224, 224)
        .imagenet_normalization()
        .build()?;

    // Process frames one by one (e.g., from a webcam)
    for frame in video_stream {
        if let Some(result) = classifier.push_frame(frame)? {
            println!("Action detected: {} (confidence: {:.2})", 
                     result.class_name.unwrap_or("Unknown".to_string()), 
                     result.confidence);
        }
    }

    Ok(())
}
```

## Model Types Supported

### Single Frame Models
Perfect for image classification tasks:
- **ResNet** family (ResNet50, ResNet101, etc.)
- **EfficientNet** (B0-B7)
- **Vision Transformers** (ViT)
- **MobileNet** for edge deployment
- Custom single-frame architectures

### Multi-Frame Models
Designed for video understanding:
- **I3D** (Inflated 3D ConvNets)
- **SlowFast** networks
- **X3D** efficient video models
- **TSN** (Temporal Segment Networks)
- **TSM** (Temporal Shift Module)

### Sequence Models
For temporal pattern recognition:
- **LSTM**-based video classifiers
- **GRU** sequence models
- Custom RNN architectures

### Two-Stream Models
Combining appearance and motion:
- **RGB + Optical Flow** fusion
- **Spatial + Temporal** streams
- Custom dual-input architectures

## Builder Pattern Examples

### Easy Configuration with Presets

```rust
use onnx_vision_classifier::ClassifierBuilder;

// ResNet classifier
let resnet = ClassifierBuilder::resnet_classifier("models/resnet50.onnx")
    .with_imagenet_classes()
    .build()?;

// EfficientNet-B4 classifier  
let efficientnet = ClassifierBuilder::efficientnet_classifier("models/efficientnet_b4.onnx", 380)
    .with_imagenet_classes()
    .build()?;

// I3D action recognition
let i3d = ClassifierBuilder::i3d_classifier("models/i3d_kinetics.onnx")
    .with_kinetics400_classes()
    .build()?;
```

### Custom Configuration

```rust
let custom_classifier = classifier()
    .model_path("models/custom_model.onnx")
    .multi_frame(32)
    .input_size(256, 256)
    .custom_normalization([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    .class_names(vec!["class1".to_string(), "class2".to_string()])
    .build()?;
```

## Advanced Features

### Custom Preprocessing

```rust
use onnx_vision_classifier::preprocessing::{DefaultPreprocessor, PreprocessingConfig};

let custom_config = PreprocessingConfig {
    resize: true,
    center_crop: true,
    normalize: true,
    filter_type: image::imageops::FilterType::Lanczos3,
};

let preprocessor = Box::new(DefaultPreprocessor::with_config(custom_config));

let classifier = classifier()
    .model_path("model.onnx")
    .single_frame()
    .with_preprocessor(preprocessor)
    .build()?;
```

### Variable Frame Count Models

```rust
let variable_classifier = classifier()
    .model_path("models/variable_model.onnx")
    .variable_frames(8, 32)  // Accepts 8 to 32 frames
    .input_size(224, 224)
    .build()?;

// Can classify with different numbers of frames
let result1 = variable_classifier.classify_frames(&frames_8)?;   // 8 frames
let result2 = variable_classifier.classify_frames(&frames_16)?;  // 16 frames
let result3 = variable_classifier.classify_frames(&frames_32)?;  // 32 frames
```

### Frame Buffer Management

```rust
let mut streaming_classifier = classifier()
    .model_path("models/streaming.onnx")
    .multi_frame(16)
    .build()?;

// Check buffer status
println!("Buffer ready: {}", streaming_classifier.is_buffer_ready());
println!("Buffer length: {}", streaming_classifier.buffer_len());

// Clear buffer when needed
streaming_classifier.clear_buffer();
```

## Performance Tips

1. **Batch Processing**: For offline processing, use `classify_frames()` with multiple frames at once.

2. **Streaming Optimization**: Use `push_frame()` for real-time processing to leverage frame buffering.

3. **Input Size**: Smaller input sizes (e.g., 112x112) provide better performance with minimal accuracy loss for many tasks.

4. **Model Selection**: Choose the right model complexity for your use case:
   - Single-frame for image classification
   - Multi-frame for action recognition
   - Two-stream for motion-sensitive tasks

## Examples

The repository includes comprehensive examples:

- [`single_frame.rs`](examples/single_frame.rs) - Image classification with various models
- [`multi_frame.rs`](examples/multi_frame.rs) - Video classification and action recognition  
- [`video_stream.rs`](examples/video_stream.rs) - Real-time streaming and buffer management

Run examples with:
```bash
cargo run --example single_frame
cargo run --example multi_frame  
cargo run --example video_stream
```

## Model Format

This library works with ONNX models. You can:

1. **Export from PyTorch**:
```python
import torch
torch.onnx.export(model, dummy_input, "model.onnx")
```

2. **Export from TensorFlow**:
```python
import tf2onnx
tf2onnx.convert.from_keras(model, output_path="model.onnx")
```

3. **Download pre-trained models** from ONNX Model Zoo

## Supported Input Formats

- **Single Frame**: `[batch, channels, height, width]` - e.g., `[1, 3, 224, 224]`
- **Multi Frame**: `[batch, channels, frames, height, width]` - e.g., `[1, 3, 64, 224, 224]`
- **LSTM**: `[batch, sequence_length, features]` - e.g., `[1, 16, 2048]`
- **Two Stream**: RGB + Flow tensors

## Error Handling

The library provides comprehensive error handling:

```rust
use onnx_vision_classifier::{ClassificationError, Result};

match classifier.classify_single(&image) {
    Ok(result) => println!("Success: {}", result.class_name.unwrap_or_default()),
    Err(ClassificationError::InsufficientFrames { expected, actual }) => {
        eprintln!("Need {} frames, got {}", expected, actual);
    },
    Err(ClassificationError::ModelConfig(msg)) => {
        eprintln!("Model configuration error: {}", msg);
    },
    Err(e) => eprintln!("Classification error: {}", e),
}
```

## Requirements

- **Rust 1.70+**
- **ONNX Runtime** (automatically handled by the `ort` crate)
- **Image processing** capabilities via the `image` crate

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- Built on top of the excellent [ort](https://github.com/pykeio/ort) ONNX runtime bindings
- Image processing powered by the [image](https://github.com/image-rs/image) crate
- Numerical computing with [ndarray](https://github.com/rust-ndarray/ndarray)

## ✅ Real ONNX Integration

The library now supports loading and running **actual ONNX models** using the ONNX Runtime! 

### Quick Test with a Real Model

Download a pre-trained model and test it:

```bash
# Download ResNet-50 from ONNX Model Zoo
curl -L -o resnet50.onnx 'https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx'
# Download Mobilenet-v2 from ONNX Model Zoo
curl -L -o mobilenetv2-7.onnx 'https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx'

# Run the real model example
cargo run --example real_model_example
```

### Supported Model Sources

- **ONNX Model Zoo**: Pre-trained models for various tasks
- **PyTorch → ONNX**: Convert your PyTorch models
- **TensorFlow → ONNX**: Convert TensorFlow models
- **Custom Models**: Any ONNX-compatible model

The library automatically handles:
- ✅ Model loading and validation
- ✅ Tensor creation and memory management  
- ✅ Input/output shape handling
- ✅ Batch processing
- ✅ Error handling and debugging

