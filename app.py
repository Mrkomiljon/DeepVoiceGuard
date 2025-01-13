import os
import torch
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
from model import RawNet
from data_utils import pad  # Import the pad function from data_utils.py
import yaml
import torch.nn.functional as F  # For softmax
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
import uvicorn
import webbrowser

# Initialize FastAPI app
app = FastAPI()

# Load the model
model_config_path = 'your\\path\\to\\model_config_RawNet.yaml'
model_path = 'your\\path\\to\\checkpoints\\best_model.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(model_config_path, 'r') as f:
    model_config = yaml.safe_load(f)

model = RawNet(model_config['model'], device).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def preprocess_audio_segment(segment, cut=64600):
    """
    Preprocess a single audio segment: pad or trim as required.
    """
    if len(segment) < cut:
        segment = pad(segment, max_len=cut)  # Pad if shorter
    else:
        segment = segment[:cut]  # Trim if longer
    return torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


def predict_with_sliding_window(waveform, model, device, window_size=64600, step_size=64600, sample_rate=16000):
    """
    Use a sliding window to predict if the audio is real or fake over the entire audio.
    """
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
        predicted_class = "Human voice" if prediction.item() == 1 else "AI generated voice (TTS)"
        probability = probabilities[0, prediction.item()].item() * 100
        total_segments.append(predicted_class)
        total_probabilities.append(probability)

    # Final aggregation
    majority_class = max(set(total_segments), key=total_segments.count)  # Majority voting
    avg_probability = np.mean(total_probabilities)  # Average probability

    return majority_class, avg_probability


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Endpoint to process audio and predict using the RawNet model.
    """
    try:
        # Save uploaded file to a temporary file
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_filename = temp_file.name

        # Load audio file
        waveform, _ = librosa.load(temp_filename, sr=16000)

        # Perform prediction
        result, avg_probability = predict_with_sliding_window(waveform, model, device)

        # Clean up temporary file
        os.remove(temp_filename)

        return JSONResponse({
            "Your audio": result,
            "average_probability": f"{avg_probability:.2f}%"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
async def root():
    return {"message": "RawNet Sliding Window Prediction API"}

# Automatically open docs or print URL when server starts
if __name__ == "__main__":
    url = "http://127.0.0.1:8000/docs"
    print(f"API docs available at: {url}")
    webbrowser.open(url)  # Open in the default browser
    uvicorn.run(app, host="127.0.0.1", port=8000)
