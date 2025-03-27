#!/usr/bin/env python3
"""
Standalone evaluation script that works with raw PyTorch models
"""
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

def encode_image(input_image, output_image, message, data_depth=1, use_cuda=False):
    """Encode a message into an image without depending on custom classes."""
    # Load the encoder
    if not os.path.exists('encoder.pt'):
        print("Encoder model not found. Run extract_models.py first.")
        return None
    
    encoder = torch.load('encoder.pt', map_location='cpu')
    encoder.eval()
    
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    print(f"Using {device} for encoding")
    
    # Load and prepare the image
    image = Image.open(input_image).convert('RGB')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    image_tensor = image_tensor.to(device)
    
    # Prepare the message as binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    print(f"Message binary length: {len(binary_message)} bits")
    
    # Create payload tensor
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    payload = torch.zeros((1, data_depth, height, width), device=device)
    
    # Fill payload with message bits (repeat if necessary)
    idx = 0
    for h in range(height):
        for w in range(width):
            for d in range(data_depth):
                if idx < len(binary_message):
                    payload[0, d, h, w] = float(binary_message[idx])
                    idx = (idx + 1) % len(binary_message)  # Cycle through message
    
    # Encode the message
    with torch.no_grad():
        try:
            generated = encoder(image_tensor, payload)
        except Exception as e:
            print(f"Error during encoding: {e}")
            # Try the forward method directly
            try:
                print("Trying alternative encoding approach...")
                generated = encoder.forward(image_tensor, payload)
            except Exception as e2:
                print(f"Alternative encoding failed: {e2}")
                return None
    
    # Save the generated image
    generated = generated.clamp(-1, 1).detach().cpu()
    generated = ((generated + 1.0) / 2.0).squeeze(0)
    generated_img = transforms.ToPILImage()(generated)
    generated_img.save(output_image)
    
    print(f"Encoded message into {output_image}")
    
    # Calculate metrics
    original = np.array(image)
    stego = np.array(generated_img)
    
    psnr_value = peak_signal_noise_ratio(original, stego)
    ssim_value = structural_similarity(
        np.mean(original, axis=2), 
        np.mean(stego, axis=2), 
        data_range=255
    )
    
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    
    return {'psnr': psnr_value, 'ssim': ssim_value}

def main():
    parser = argparse.ArgumentParser(description="Standalone SteganoGAN evaluation")
    parser.add_argument('--input', type=str, default='steganogan/data/xrays/val/0001.jpg',
                        help='Input image for encoding')
    parser.add_argument('--output', type=str, default='stego_xray.png',
                        help='Output path for steganographic image')
    parser.add_argument('--message', type=str, default='Name: YALLAPA',
                        help='Message to encode')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Try to load data_depth if available
    data_depth = 1
    if os.path.exists('data_depth.txt'):
        with open('data_depth.txt', 'r') as f:
            try:
                data_depth = int(f.read().strip())
                print(f"Using data depth: {data_depth}")
            except:
                print("Could not parse data_depth.txt, using default value 1")
    
    # Encode the message
    metrics = encode_image(args.input, args.output, args.message, data_depth, args.cuda)
    
    if metrics:
        # Display comparison
        original = Image.open(args.input).convert('RGB')
        stego = Image.open(args.output).convert('RGB')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(np.array(original))
        ax1.set_title("Original X-ray")
        ax1.axis('off')
        
        ax2.imshow(np.array(stego))
        ax2.set_title(f"Steganographic X-ray\nPSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig("comparison.png")
        print("Comparison image saved to comparison.png")
    
        try:
            plt.show()
        except:
            pass

if __name__ == "__main__":
    main()