# DeepVoiceGuard
# DeepVoiceGuard: Anti-Spoofing for ASV Systems
## Available on Hugging Face 🤗

This model is also hosted on Hugging Face for easy access and inference:

[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/spaces/Mrkomiljon/DeepVoiceGuard)

If you find this project helpful or inspiring, please consider giving it a star 🌟 on GitHub!

![cc](https://github.com/user-attachments/assets/4d244897-363d-4643-a6f8-b21e1a7c1650)


**DeepVoiceGuard** is a robust solution for detecting spoofed audio in Automatic Speaker Verification (ASV) systems. This project utilizes the **RawNet2** model, trained on the **ASVspoof 2019** dataset, and deploys the trained model using **FastAPI** for real-time inference. The repository also includes ONNX model conversion and sliding window inference for efficient processing of long audio files.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Model Training](#model-training)
5. [Model Conversion](#model-conversion)
6. [Sliding Window Inference](#sliding-window-inference)
7. [Deployment](#deployment)
8. [How to Use](#how-to-use)
9. [Results](#results)
10. [References](#references)

---

## Introduction

Automatic Speaker Verification (ASV) systems are vulnerable to spoofing attacks, such as voice conversion and speech synthesis. **DeepVoiceGuard** leverages the **RawNet2** architecture to detect and mitigate such attacks. The project also includes a FastAPI-based deployment for real-time inference, ensuring practical usability.

---

## Features

- Training the **RawNet2** model on the **ASVspoof 2019** dataset.
- Conversion of the PyTorch-trained model to **ONNX** format for efficient deployment.
- Real-time inference via **FastAPI**.
- Sliding window inference for processing long audio files.
- Supports both genuine and spoofed audio classification.

---

## Dataset

The **ASVspoof 2019** dataset is a benchmark dataset for ASV anti-spoofing tasks. It includes genuine and spoofed audio samples for training, validation, and testing.

### Preprocessing

- Audio normalization.
- Log-Mel spectrogram extraction.
- Label encoding for binary classification (genuine/spoofed).

For more details, visit the [ASVspoof 2019 official website](https://www.asvspoof.org).

---

## Model Training

### Training Configuration

- **Model**: RawNet2
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy (BCE)
- **Batch Size**: 32
- **Learning Rate Scheduler**: Cosine Annealing
- **Epochs**: 50

### Training Script
The training process is handled using `main.py`. Run the following command to train the model:
```bash
python main.py --data_path LA --epochs 100 --batch_size 32
```
### Run TensorBoard
- Run TensorBoard to visualize logs:

```bash
tensorboard --logdir=path/to/logs
```
---

## Model Conversion

After training, the PyTorch model was converted to the **ONNX** format for deployment. The conversion process is as follows:

### Conversion Script
```python
import torch
from model import RawNet

def export_to_onnx(model, onnx_path, device):
    """
    Export the RawNet model to ONNX format.
    """
    dummy_input = torch.randn(1, 64600, device=device)  # Dummy input tensor for tracing
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,               # Store the model parameters
        opset_version=11,                 # ONNX opset version
        do_constant_folding=True,         # Optimize constants
        input_names=['input'],            # Input tensor name
        output_names=['output'],          # Output tensor name
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic batch size
    )
    print(f"Model successfully exported to {onnx_path}")
```

---

## Sliding Window Inference

The project includes a **sliding window inference method** to process long audio files efficiently. This method divides the audio into smaller, fixed-length segments, processes each segment, and aggregates the results using majority voting and average probabilities. This approach ensures faster and more accurate predictions for long audio inputs.

### Features
- Processes audio in fixed-length windows (e.g., 64600 samples).
- Pads or trims audio segments as required.
- Aggregates predictions using majority voting.
- Computes average confidence probabilities.

### Script
The `predict_with_sliding_window` function in `inference_onnx.py` handles the inference.

### Example Usage
Here is how you can perform inference with the ONNX model and a sliding window:

```python
python inference_onnx.py --model_path <path_to_model.onnx> --audio_path <path_to_audio_file>
```

### Sample Output
```
Segment 1: Real, Probability: 98.34%
Segment 2: Fake, Probability: 87.42%
Segment 3: Real, Probability: 92.15%
...
Final Result: Real, Average Probability: 94.56%
```

---

## Deployment

The ONNX model was deployed using **FastAPI** for real-time inference. The FastAPI server accepts audio files as input, processes them, and returns classification results (genuine or spoofed).

### FastAPI Deployment

#### Installation
1. Install required packages:
   ```bash
   pip install fastapi uvicorn onnxruntime librosa
   ```

2. Save the following script as `app.py`:
   ```python
   from fastapi import FastAPI, File, UploadFile
   import onnxruntime as ort
   import numpy as np
   import librosa

   app = FastAPI()
   session = ort.InferenceSession("model.onnx")

   @app.post("/predict/")
   async def predict(audio_file: UploadFile = File(...)):
       audio, sr = librosa.load(audio_file.file, sr=16000)
       audio = np.expand_dims(audio[:16000], axis=0).astype(np.float32)
       input_data = {session.get_inputs()[0].name: audio}
       output = session.run(None, input_data)
       prediction = "Genuine" if np.argmax(output[0]) == 0 else "Spoofed"
       return {"prediction": prediction}
   ```

3. Run the server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

### API Endpoint

- **POST** `/predict/`
  - Input: Audio file (WAV format, 16 kHz, mono).
  - Output: JSON with prediction result (`Real` or `Fake`).

---

## How to Use

1. Train the model using the provided script.
2. Convert the trained model to ONNX format.
3. Deploy the model using FastAPI.
4. Test the deployment by sending audio files to the `/predict/` endpoint.

Example using `curl`:
```bash
curl -X POST "http://localhost:8000/predict/" -F "audio_file=@test.wav"
```

---

## Results

### Metrics
| Metric        | Development Set | Evaluation Set |
|---------------|-----------------|----------------|
| EER (%)       | 4.21            | 5.03           |
| Accuracy (%)  | 95.8            | 94.7           |
| ROC-AUC       | 0.986           | 0.975          |

---

## References

- [ASVspoof 2019 Challenge](https://www.asvspoof.org)
- [RawNet2 Paper](https://arxiv.org/abs/2006.02695)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [ONNX Runtime Documentation](https://onnxruntime.ai)

---

