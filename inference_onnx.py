import os
import librosa
import numpy as np
import onnxruntime as ort
import torch
from data_utils import pad  # Import the pad function from data_utils.py

# Preprocess audio for a single segment
def preprocess_audio_segment(segment, cut=64600):
    """
    Preprocess a single audio segment: pad or trim as required.
    """
    if len(segment) < cut:
        segment = pad(segment, max_len=cut)  # Pad if shorter
    else:
        segment = segment[:cut]  # Trim if longer
    return np.expand_dims(np.array(segment, dtype=np.float32), axis=0)  # Add batch dimension

# Perform sliding window prediction
def predict_with_sliding_window(audio_path, onnx_model_path, window_size=64600, step_size=64600, sample_rate=16000):
    """
    Use a sliding window to predict if the audio is real or fake over the entire audio.
    """
    # Load the ONNX runtime session
    ort_session = ort.InferenceSession(onnx_model_path)

    # Load the audio file
    waveform, _ = librosa.load(audio_path, sr=sample_rate)
    total_segments = []
    total_probabilities = []

    # Sliding window processing
    for start in range(0, len(waveform), step_size):
        end = start + window_size
        segment = waveform[start:end]

        # Preprocess the segment
        audio_tensor = preprocess_audio_segment(segment)

        # Perform inference
        inputs = {ort_session.get_inputs()[0].name: audio_tensor}
        outputs = ort_session.run(None, inputs)
        probabilities = torch.tensor(outputs[0])  # Convert to torch tensor for processing
        probabilities = torch.nn.functional.softmax(probabilities, dim=1)  # Compute probabilities
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
    # Path to the ONNX model
    onnx_model_path = 'your\\path\\to\\checkpoints\\RawNet_model.onnx'

    # Specify the path to the audio file
    audio_path = "your\\path\\to\\audio\\R.mp3"  # Example .mp3 file

    # Perform sliding window prediction
    result, avg_probability = predict_with_sliding_window(audio_path, onnx_model_path)

    print(f"Final Result: {result}, Average Probability: {avg_probability:.2f}%")
