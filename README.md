# mobileclip-unity

Real-time On-device Multi-modal Retrieval: MobileCLIP on Meta Quest 3 with Unity Sentis

## Overview

This repository provides a highly efficient inference implementation for **MobileCLIP2** using **Unity Sentis**. Designed specifically for standalone XR headsets, this project demonstrates **real-time** multi-modal retrieval capabilities directly on the **Meta Quest 3**.

It enables smart asset search and semantic retrieval across text and images without requiring an internet connection, leveraging the efficiency of MobileCLIP architecture in a **Extended Reality** environment.

## Features

- **⚡ Real-time Performance**: Optimized for Meta Quest 3.
- **📱 MobileCLIP2 Model**: Utilizes efficient ONNX models generated via [ml-mobileclip-onnx](https://github.com/skykim/ml-mobileclip-onnx).
- **🔍 Multi-modal Search**:
  - ✅ Text-to-Image Search
  - ✅ Image-to-Image Search
  - ✅ Text-to-Text Search
  - ✅ Image-to-Text Search

## Requirements

- **Unity**: `6000.2.10f1` (or compatible version)
- **Unity Sentis**: `2.4.1` (com.unity.ai.inference)
- **Hardware**: Meta Quest 3

## Architecture

### 1. MobileCLIP (ONNX)
The project utilizes MobileCLIP models converted to ONNX format. These models are optimized for mobile and edge devices, balancing accuracy and latency.
* **Source**: The ONNX files used in this project were generated using the [skykim/ml-mobileclip-onnx](https://github.com/skykim/ml-mobileclip-onnx) repository.

### 2. Tokenizer
Text input processing is handled by a custom **BPE (Byte Pair Encoding) Tokenizer**. This ensures that text queries are correctly tokenized and encoded to match the MobileCLIP model's expected input format.

### 3. Embedding Database
The system generates a local database (`image_embeddings.bin`) by processing images located in the StreamingAssets folder. This allows for instant similarity calculations during runtime.

## Getting Started

### 1. Project Setup
- Clone or download this repository.
- Ensure **Unity Sentis 2.4.1** is installed via the Package Manager.
- Unzip the provided [Assets.zip](https://drive.google.com/file/d/1LFHUMorgLrekllz04WRGnRmlBkp6wvxu/view?usp=drive_link) file and copy the contents into your Unity project's `Assets` folder.

### 2. Model Setup
- Verify that the MobileCLIP ONNX model files are located in the following directory:
  > **Path:** `/Assets/MobileCLIP`

### 3. Image Data Setup
- Locate the **`Images.zip`** file (included in the assets).
- **Unzip** `Images.zip` and place the image files directly into the StreamingAssets folder.
- Ensure the directory structure matches exactly:
  > **Path:** `/Assets/StreamingAssets/Images/`

### 4. Generate Database
Before running the scene on the device, you need to generate the embedding index in the Editor.

1. Open the project in the Unity Editor.
2. Select the `ImageSearchManager` object in the hierarchy.
3. Click the **"Generate And Save Embeddings (Create Index)"** button in the Inspector.
4. This process will read images from `/Assets/StreamingAssets/Images` and generate the `image_embeddings.bin` file.

### 5. Build & Run
1. Switch the build platform to **Android**.
2. Open the **Mixed Reality Scene** included in the project.
3. Build and Run on your **Meta Quest 3**.
4. You will see the MobileCLIP retrieval system working in a Mixed Reality environment.

## Links

- **ONNX Generation Repo**: [skykim/ml-mobileclip-onnx](https://github.com/skykim/ml-mobileclip-onnx)
- **MobileCLIP Paper**: [MobileCLIP2: Improving Multi-Modal Reinforced Training](https://arxiv.org/abs/2508.20691)
- **Unity Sentis Documentation**: [Unity Sentis 2.4 Manual](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.4/index.html)

## License

This project respects the licenses of the used models and libraries. Please refer to the MobileCLIP repository for specific model licensing information.
