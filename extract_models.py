import torch
import pickle
import os
import io

# Create mock class for loading
class SteganoGAN:
    def __init__(self, *args, **kwargs):
        self.encoder = None
        self.decoder = None
        self.critic = None
        self.data_depth = 1

# Define custom unpickler
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "SteganoGAN":
            return SteganoGAN
        try:
            return super().find_class(module, name)
        except:
            return getattr(torch.nn, name, None)

# Load the model using custom unpickler
model_path = 'pretrained/medical_xray.steg'
print(f"Loading model from {model_path}")

with open(model_path, 'rb') as f:
    model_data = f.read()

unpickler = CustomUnpickler(io.BytesIO(model_data))
try:
    model = unpickler.load()
    print("Model loaded successfully")
    
    # Check if encoder and decoder exist
    if hasattr(model, 'encoder') and model.encoder is not None:
        print("Encoder found")
        torch.save(model.encoder, 'encoder.pt')
        print("Encoder saved to encoder.pt")
    else:
        print("No encoder found in model")
        
    if hasattr(model, 'decoder') and model.decoder is not None:
        print("Decoder found")
        torch.save(model.decoder, 'decoder.pt')
        print("Decoder saved to decoder.pt")
    else:
        print("No decoder found in model")
    
    # Save data depth if available
    if hasattr(model, 'data_depth'):
        print(f"Data depth: {model.data_depth}")
        with open('data_depth.txt', 'w') as f:
            f.write(str(model.data_depth))
        print("Data depth saved to data_depth.txt")

except Exception as e:
    print(f"Error loading model: {e}")