import torch
import yaml
from model import RawNet

# Load the model configuration and initialize the model
def load_model(model_config_path, model_path, device):
    """
    Load and initialize the RawNet model using the provided configuration and checkpoint.
    """
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Initialize the model
    model = RawNet(model_config['model'], device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

# ONNX export function
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

if __name__ == "__main__":
    # Paths to model configuration and checkpoint
    model_config_path = 'your\\path\\to\\model_config_RawNet.yaml'
    model_path = 'your\\path\\to\\checkpoints\\best_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    model = load_model(model_config_path, model_path, device)

    # Path to save the ONNX model
    onnx_path = 'your\\path\\to\\checkpoints\\RawNet_model.onnx'

    # Export the model to ONNX
    export_to_onnx(model, onnx_path, device)
