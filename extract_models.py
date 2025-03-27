import torch
import pickle
import os

# Mock the necessary classes so they can be loaded
class SteganoGAN:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.critic = None

# Register the mock class with pickle
pickle.Unpickler.find_class.__globals__['__main__'] = type('MockModule', (), {'SteganoGAN': SteganoGAN})

# Load the model and extract components
model_path = 'pretrained/medical_xray.steg'
print(f"Loading model from {model_path}")

model = torch.load(model_path, map_location='cpu')
encoder = model.encoder
decoder = model.decoder

# Save just the encoder and decoder
os.makedirs('extracted', exist_ok=True)
torch.save(encoder, 'extracted/encoder.pt')
torch.save(decoder, 'extracted/decoder.pt')

print("Extracted encoder and decoder saved to 'extracted' directory.")