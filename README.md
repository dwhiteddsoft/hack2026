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

# Now real world example also outputs JSON. Listed for reference below
```
```json
{"class_id":466,"class_name":null,"confidence":10.140818,"all_scores":[1.5248246,-3.8504536,-2.9580927,-6.9232135,-0.54517066,2.649688,-0.3492134,1.4262657,-0.03067708,-3.150629,-4.340497,-6.8790274,-8.977681,-6.680066,-9.218331,-2.5309577,-4.620558,-6.00918,-5.026438,-4.922855,-6.824566,-2.2366278,-4.872321,-4.04614,-6.2388964,-5.9994574,-6.016681,-5.64611,-4.898223,-3.0217025,-6.4039526,-7.278485,-5.621373,-1.8900164,-0.7111414,-2.8319983,-0.77004635,-4.6665444,-0.20355949,2.249019,-2.834589,-2.4845026,-4.162025,1.3259596,-2.681913,-2.2773256,-4.6383333,-3.1617208,-2.6461573,-1.8471086,-2.4298222,-0.4291882,0.49239397,-3.5267975,0.8579742,-0.77040094,-0.6099747,-4.856113,-0.7058864,0.44083107,2.5304055,2.9713326,1.6562905,2.775498,1.5284344,-0.09386051,-1.2175815,1.857877,-0.61199933,-4.488357,-6.127542,0.08944571,-3.7255895,-4.0893617,-6.0223727,-7.0630817,0.86413336,-4.752821,-1.863559,-0.35394073,-2.4480414,-7.5740237,-2.4891672,-2.4081948,-1.796883,-6.11872,-2.2923307,-0.9401569,-1.6423438,-4.531258,-5.4413834,1.8949983,-4.949498,-0.6218197,-7.07576,-4.6972733,-1.539204,-2.1496503,-1.9486976,1.3439889,-2.671811,-0.8638065,-2.8086479,-2.0694757,-3.3379836,-2.556376,-1.4364407,-2.8975377,-5.602714,-2.9820774,-9.702022,-3.591614,5.2960663,0.866276,-1.3979166,-7.259407,-6.527266,-1.976918,3.6051764,-1.646127,-3.4940114,3.3867862,3.5702357,-1.6705085,-0.79021907,-1.40499,-4.5573416,-7.737761,-7.1863027,-6.387836,-5.3374186,-6.504012,-5.682497,-4.2571673,-3.4974353,-3.8137484,-7.0271397,-2.9646637,-5.0313635,-7.9152293,-6.3635006,-9.587881,-5.825969,-8.035833,0.56020933,-1.3365135,-3.394205,-1.6421385,1.3032075,0.011735439,4.597702,1.3574154,-2.1013424,-0.40770718,1.2255774,-1.3466438,-0.6054831,-0.65219295,-3.576534,1.0735246,-2.138425,1.9491941,-0.23492166,1.0054641,-0.5830331,0.41643474,0.8033657,-1.8653724,5.1478305,-3.3441482,-2.7396522,0.6526919,-0.34384978,0.19890356,-0.32006717,-4.207525,1.1644126,-0.83535516,0.48178744,0.17660213,-0.44266576,-1.9839596,-0.6605679,-1.3528147,0.6231095,-0.6724701,0.2677519,-3.3157928,-4.850072,-2.3514378,-3.3868585,-1.9266372,-2.8092732,-2.9552188,-2.833581,0.7549757,-2.5136094,-2.2990766,-2.3138738,-3.0878482,-2.9673934,-1.435844,0.20523864,-1.5583525,1.6225209,-0.55313444,-0.56469214,1.1121868,-0.004853368,0.87746656,-1.8055559,2.899339,-3.7794306,0.39786062,-0.34284544,-1.0047808,-3.750548,-1.8549964,-1.664681,-2.8199973,-0.96420646,-2.0638752,-0.764336,2.2295017,0.52302307,2.0230417,-1.6200814,-2.5989575,-2.7996051,-1.9543369,0.6645297,-1.7182894,-1.948306,-2.3176942,0.983125,1.1652958,1.3652091,-2.400114,1.4835687,2.2455902,0.81683755,-2.1279337,1.1338463,2.6501145,-1.0954707,-1.5579358,1.5007632,1.5784824,0.29015303,0.21709432,-0.31405723,-2.4427404,-1.9797807,1.7662867,-1.4829057,-0.5015545,-1.3277264,0.46695817,-0.40649632,1.873367,-0.6758241,-0.92722386,0.79465544,1.486779,-0.088709734,0.37960973,-2.2400205,-1.887908,-2.2268643,-5.4823637,-3.598657,-2.267998,-5.737488,0.9414517,-2.0524883,-6.927743,-6.9359665,1.3596679,-1.5841683,-1.2110586,-2.1017542,-2.8991523,-0.88010085,-1.4880419,-0.6740669,-2.2378898,-2.0603302,-3.5996382,-4.5638804,-5.7356553,-2.6809168,-4.7714505,1.0786221,-4.975724,-2.9776142,-6.026157,-1.2087297,-3.68907,-4.2648544,-1.7893239,-10.962694,-4.5245886,-8.400192,-8.692848,-7.4164305,-5.020462,-1.6556661,-8.572701,-5.1062136,-3.986963,-5.583555,-6.300319,-2.881742,-5.1611414,0.55234224,-3.5602136,-4.381558,-5.8706512,-1.20677,-1.4003465,-3.354979,-5.9583983,-3.8104596,-3.2287886,-6.407564,-6.96218,-6.864894,0.28913152,-0.6909093,-2.6560097,-4.1313124,-4.104318,2.3334758,-2.2118618,-1.5427213,-2.5173957,-2.0614824,2.1943188,-1.5932739,2.3150923,-2.45186,0.8844237,-2.7224827,-3.512187,-2.5378695,-1.3213383,-1.5318785,-2.9311466,-5.112212,-3.3615205,-3.7569537,-3.6201305,-2.7230995,-1.3635544,-0.009608805,1.145523,2.9501534,-1.446001,6.343957,2.6922421,0.79890186,1.5833529,-0.88723755,-2.2463803,-3.3526034,1.5665808,-3.6325164,-0.77544403,1.3981074,-0.0072538853,0.58439124,0.6970004,-2.6063318,-0.49430135,-2.1475382,-2.115354,1.6483879,-0.9656606,1.8326348,-1.7959934,1.6088798,-0.05023624,1.7945077,-1.1659772,-2.577736,1.1056073,-1.3416605,3.2729933,-0.1416578,3.5391617,1.4456,3.6423306,-5.8070545,-8.179274,1.3699329,1.6915703,-5.4765406,-4.4642,-0.23245877,5.4974194,2.987291,4.8856893,5.1110554,-0.6540319,-2.3065598,-1.4248935,-4.313638,1.2972251,0.8007752,0.6470097,-1.2438128,4.5798273,2.364695,2.1326647,6.2278304,5.195902,0.9714886,0.5708339,1.14079,6.157466,3.6272984,4.478202,3.507344,1.3289431,1.244679,-3.9763668,0.06441462,4.4612837,-0.0026914477,0.6046179,1.4798745,0.27176896,1.5474266,3.5895581,4.9014416,4.8656116,-0.064544275,2.1909142,3.4743423,1.020901,3.9387174,1.2867979,1.8588994,-0.9864161,-0.9510014,7.181882,3.2630503,3.8905573,-4.2033505,2.4853475,3.2771745,1.3313973,1.9192611,0.1106118,1.3868127,-1.4630802,2.7638302,6.3554864,-1.6643547,6.264902,5.448015,-3.3668723,3.357684,2.4703705,-1.8118646,2.2420778,10.140818,5.431258,0.66431254,2.4656663,4.512905,1.4570713,0.81937194,-0.72679245,6.84331,0.6494655,-0.9151877,-1.2672446,3.5376458,-0.54907155,2.887124,1.8240898,-0.1715638,-1.6858371,0.068795204,-0.80799276,5.457072,7.709426,4.078598,0.7899038,-3.0169268,2.1153593,0.65083194,-2.8871315,0.7706176,-3.897393,0.9146547,-0.8548709,2.6467133,4.0313745,-6.7753086,3.646465,2.3575726,5.7417016,2.6265206,3.8423724,1.3641729,2.0640619,2.987656,2.7390618,3.3181052,-2.2423203,0.31590152,2.1916084,4.9891043,3.0650802,2.3337483,4.973134,0.8861412,2.6387837,2.0593832,1.1253953,-1.8487258,5.0686073,-3.9777272,5.7485085,3.8365984,0.9112318,2.8517685,5.8290024,3.1520836,2.2156434,-0.4324621,-1.5423906,4.552773,-2.233151,5.6165323,-3.8413515,1.5428424,2.3272471,-0.5801331,1.4123838,1.8093168,5.721994,0.86094546,3.035717,1.8628263,1.5719043,-1.6629814,2.6285462,1.0579427,1.5155523,0.09405398,-0.09502721,2.1381783,-3.5768542,2.8531537,-0.30315804,3.172579,-0.6640086,-1.2438229,1.3597448,1.9799381,-0.6460443,1.9348304,-1.0073031,-0.9591905,4.5023503,7.483572,-0.96684515,2.812205,2.537602,1.5732502,1.0215998,-3.6603656,0.34804034,3.1408904,1.9050798,4.077708,4.159595,6.0672226,-7.0916057,5.0520067,1.4689324,1.0474498,5.2171936,-5.2710304,5.2712193,1.2149625,4.2409515,5.1666,3.238981,0.19255829,4.8126388,4.84878,-3.260266,2.5073597,0.8810401,1.7144014,-3.1131165,3.0606441,1.0560029,1.0494069,2.6533384,2.5266004,3.6317394,5.978353,-0.3190487,7.9277363,-1.5234654,4.9212446,-2.2294168,3.6061006,0.7908737,5.426983,2.7763894,4.073283,6.1080527,3.0030513,-1.7521924,5.1462603,-1.7935004,1.3780903,0.8039402,2.1187048,-0.3742067,2.4471717,3.0195549,3.3523805,3.556098,5.9254217,2.818416,-2.0383549,2.968567,3.8076878,1.2079372,7.093935,-0.061413348,5.266981,4.01454,-4.4988804,2.9016604,1.8745879,-0.6225071,0.27233934,-4.0293994,-0.3685477,1.2782804,1.0394574,-3.285651,2.776823,3.0943909,1.4924929,2.371328,4.4047246,6.333371,2.814189,-2.7716432,0.6723906,-0.4628408,4.225253,-3.3290174,-1.6886315,-1.7885273,3.262181,0.4604719,-0.5246825,2.5580983,0.8855091,4.6007338,1.4841819,-1.37063,-1.0513182,1.9452703,0.632225,2.6144714,0.5217314,0.86685187,6.248263,-2.9256454,1.737524,6.4770203,-0.15354896,2.408608,4.271952,-7.1877413,-3.1431942,-0.57766104,-1.7072508,0.6872718,-1.9493085,4.192172,-0.291573,2.6559057,-0.04757327,1.1304836,1.1499598,3.8038301,2.2918549,2.9087474,2.0300784,-0.83706,0.9250363,4.263727,4.781153,3.148609,4.0146575,7.295848,-2.1284966,-1.1685619,1.4908645,1.275662,1.1413932,7.093824,-1.1795402,-1.7567933,1.9182303,-3.3556848,8.609678,-0.2686355,4.173652,0.35929528,2.9102588,1.854934,-0.34970054,-2.49577,-1.3204693,2.047146,2.3180757,-1.0678253,-4.880246,3.7259834,0.9656965,3.934953,0.081733584,2.3543994,1.3139805,2.3691258,0.48380923,1.6505883,3.4468007,2.4174457,1.6669676,5.004981,-1.5388532,3.5848553,-0.063154936,6.092466,-0.1313571,3.2683296,1.2050879,-2.9409308,1.3681984,0.6676495,-0.5462738,0.95145667,1.6639478,1.335414,4.6205583,3.3763454,4.346195,3.2928748,3.324314,4.8710704,2.197548,0.7316164,-0.27001274,2.1357424,-1.0337716,4.143322,3.8125057,3.6851501,-1.2190316,0.41030216,5.702166,5.184754,3.8230882,-1.2354534,6.403339,-0.703393,3.4391108,-1.3124763,2.4408255,1.7554234,-0.19572699,1.5930959,0.8092779,-1.9050783,2.1185305,1.213114,4.4378166,4.131424,1.7948315,3.6594436,3.950776,-2.4612997,1.2423996,4.276552,1.8254297,4.785384,0.39820182,-0.6523597,-1.867003,0.5876806,2.9480336,-0.36684084,2.3502793,0.5998994,4.095761,-2.8335648,1.7057605,-0.5802543,2.1485322,4.1347694,-0.29273406,-3.61402,-2.1372075,-2.139718,3.107199,2.2649093,-3.8746393,2.9027774,0.5390096,4.959779,7.240749,-1.6372435,3.2157362,-1.4775949,0.6723265,3.3750815,0.78355867,5.328334,0.62551457,1.0083151,5.8915977,1.048777,4.528241,5.0705667,2.485366,2.9606578,2.8092704,6.115181,5.8245296,4.0339046,-0.51918644,1.1286156,-0.2527903,-1.4007276,-0.55304646,-0.51039594,2.2111073,5.105545,-0.12744564,0.91559386,1.182292,-3.258165,-4.5891366,-1.6452821,0.8329529,3.0719604,0.04147899,4.4371023,2.7661057,0.23985547,-3.0522628,-0.28825378,-3.2091367,0.15041035,2.7720184,7.271402,1.5326931,0.60482466,5.205697,1.1337192,-0.8937746,-1.069916,6.823498,4.8286285,1.5999041,5.9319663,-0.19707358,3.874282,3.2133472,-0.15270388,2.1365683,1.4474496,0.21903026,0.43646854,0.2624799,5.2968388,2.085015,1.5735612,3.1765356,3.3656929,0.896971,-2.9296808,1.1269606,4.814702,2.2383237,3.5772367,-3.5058117,-1.285904,3.1186254,1.7334206,2.9477525,6.8692284,5.7981777,2.4961371,-2.8487918,0.8580121,1.6828653,-0.45920765,2.4916885,1.2836143,-0.21930075,0.59759176,6.417566,1.5391603,2.6782744,3.5189865,0.28922927,0.4847827,4.105377,-0.3990806,-3.2831488,-2.0421968,0.14958167,-2.178895,-1.1662128,4.191169,4.8486753,3.121978,5.87239,-1.1291116,6.908747,-2.449013,0.62003636,-2.8084393,-2.6085496,1.7101923,-3.7836208,-2.332204,-0.26272738,2.2107444,-3.1544738,-2.7379692,-5.343874,-1.189721,-0.46048176,-3.391368,-1.5742908,-1.7587239,-4.865179,-0.09160638,-2.9737217,1.9276339,-3.2066407,-2.9845529,-1.5589005,-3.7490048,0.18052448,2.2779818,-0.15781903,-4.1016636,-0.53354675,-3.108999,4.1620126,4.0315146,2.7498083,0.51570797,-3.027794,3.8555436,-0.5901698,-2.7706676,-1.1544678,3.328586,-1.9094872,0.41233176,1.8297627,-1.0053196,-0.35446113,0.01692319,5.194332,-1.1107261,-0.6831774,-3.8655503,-3.75406,-0.7832007,-4.1125755,-5.9436417,-2.3012714,-5.6073995,-5.8336086,-3.0910082,-5.6294823,-3.7960062,1.4911668,-3.5470746,0.40337217,2.603186],"metadata":null}
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

