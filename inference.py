import os
import torch
import librosa
import numpy as np
from model import RawNet
from data_utils import pad  # Import the pad function from data_utils.py
import yaml
import torch.nn.functional as F  # For softmax

# Load the model
def load_model(model_config_path, model_path, device):
    """
    Load and initialize the RawNet model.
    """
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # Initialize the model
    model = RawNet(model_config['model'], device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Preprocess audio file for inference
def preprocess_audio_for_inference(file_path, cut=64600, sample_rate=16000):
    """
    Load and preprocess the audio file: normalize, pad, or trim as required.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load the audio file
    waveform, fs = librosa.load(file_path, sr=sample_rate)
    # Pad or trim the waveform
    padded_waveform = pad(waveform, max_len=cut)
    # Convert to tensor
    audio_tensor = torch.tensor(padded_waveform, dtype=torch.float32)
    return audio_tensor.unsqueeze(0)  # Add batch dimension

# Predict whether the audio is real or fake
def predict_real_or_fake_with_probability(audio_path, model, device):
    """
    Use the RawNet model to predict if the audio is real or fake, along with probabilities.
    """
    # Preprocess the audio file
    audio_tensor = preprocess_audio_for_inference(audio_path)
    audio_tensor = audio_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(audio_tensor)
        probabilities = F.softmax(output, dim=1)  # Compute probabilities using softmax
        prediction = torch.argmax(probabilities, dim=1)  # Get the class with highest probability

    predicted_class = "Real" if prediction.item() == 1 else "Fake"
    print(f"Logits: {output}")
    print(f"Probabilities: {probabilities}")
    predicted_probability = probabilities[0, prediction.item()].item() * 100  # Convert to percentage

    return predicted_class, predicted_probability

# Main script for inference
if __name__ == "__main__":
    # Model configuration
    model_config_path = 'your\\path\\to\\model_config_RawNet.yaml'
    model_path = 'your\\path\\to\\checkpoints\\best_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    model = load_model(model_config_path, model_path, device)

    # Specify the path to the audio file
    # audio_path = 'C:/Users/GOOD/Desktop/TEST-2024/2021/LA/Baseline-RawNet2/LA/ASVspoof2019_LA_eval/flac/LA_E_9094036.flac'
    # audio_path = "D:\\audio_data_asv\\ASVspoof2021_LA_eval\\flac\\LA_E_9999987.flac"
    audio_path = "your\\path\\to\\audio\\NaraA.mp3"
    # Perform prediction
    result, probability = predict_real_or_fake_with_probability(audio_path, model, device)

    print(f"Result: {result}, Probability: {probability:.2f}%")
