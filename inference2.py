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

# Preprocess audio for a single segment
def preprocess_audio_segment(segment, cut=64600):
    """
    Preprocess a single audio segment: pad or trim as required.
    """
    if len(segment) < cut:
        segment = pad(segment, max_len=cut)  # Pad if shorter
    else:
        segment = segment[:cut]  # Trim if longer
    return torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Perform sliding window prediction
def predict_with_sliding_window(audio_path, model, device, window_size=64600, step_size=64600, sample_rate=16000):
    """
    Use a sliding window to predict if the audio is real or fake over the entire audio.
    """
    # Load the audio file
    waveform, _ = librosa.load(audio_path, sr=sample_rate)
    total_segments = []
    total_probabilities = []

    # Sliding window processing
    for start in range(0, len(waveform), step_size):
        end = start + window_size
        segment = waveform[start:end]

        # Preprocess the segment
        audio_tensor = preprocess_audio_segment(segment).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(audio_tensor)
            probabilities = F.softmax(output, dim=1)  # Compute probabilities
            prediction = torch.argmax(probabilities, dim=1)

        # Store the results
        predicted_class = "Real" if prediction.item() == 1 else "Fake"
        probability = probabilities[0, prediction.item()].item() * 100
        total_segments.append(predicted_class)
        total_probabilities.append(probability)

        print(f"Segment {start//step_size + 1}: {predicted_class}, Probability: {probability:.2f}%")

    # Final aggregation
    majority_class = max(set(total_segments), key=total_segments.count)  # Majority voting
    avg_probability = np.mean(total_probabilities)  # Average probability

    return majority_class, avg_probability

# Main script for inference
if __name__ == "__main__":
    # Model configuration
    model_config_path = 'your\\path\\to\\model_config_RawNet.yaml'
    model_path = 'your\\path\\to\\checkpoints\\best_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    model = load_model(model_config_path, model_path, device)

    # Specify the path to the audio file
    # audio_path = "C:\\Users\\GOOD\\Desktop\\TEST-2024\\2021\\LA\\Baseline-RawNet2\\audio\\KTA.mp3"  # Example .mp3 file
    audio_path = "your\\path\\to\\data\\real\\7.wav"

    # Perform sliding window prediction
    result, avg_probability = predict_with_sliding_window(audio_path, model, device)

    print(f"Final Result: {result}, Average Probability: {avg_probability:.2f}%")



