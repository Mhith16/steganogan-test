#!/usr/bin/env python3
"""
Simple evaluation script for SteganoGAN X-ray steganography
"""
import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_model(path):
    """Load the trained model."""
    print(f"Loading model from {path}")
    model = torch.load(path, map_location='cpu')
    return model

def encode_message(model, input_image, output_image, message, use_cuda=False):
    """Encode a message into an image using the trained model."""
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    if use_cuda and torch.cuda.is_available():
        model.encoder.to(device)
        print("Using CUDA for encoding")
    else:
        print("Using CPU for encoding")

    # Load and prepare the image
    image = Image.open(input_image).convert('RGB')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    image_tensor = image_tensor.to(device)

    # Create a binary representation of the message
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    
    # Create payload tensor
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    payload = torch.zeros((1, model.data_depth, height, width), device=device)
    
    # Fill payload with message bits (and repeat if necessary)
    idx = 0
    for h in range(height):
        for w in range(width):
            for d in range(model.data_depth):
                if idx < len(binary_message):
                    payload[0, d, h, w] = float(binary_message[idx])
                    idx = (idx + 1) % len(binary_message)  # Cycle through message if needed
    
    # Encode the message
    with torch.no_grad():
        generated = model.encoder(image_tensor, payload)
    
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

def decode_message(model, stego_image, message_length, use_cuda=False):
    """Decode a message from a steganographic image."""
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    if use_cuda and torch.cuda.is_available():
        model.decoder.to(device)
        print("Using CUDA for decoding")
    else:
        print("Using CPU for decoding")

    # Load and prepare the image
    image = Image.open(stego_image).convert('RGB')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    image_tensor = image_tensor.to(device)
    
    # Decode the message
    with torch.no_grad():
        decoded = model.decoder(image_tensor)
    
    # Convert to binary
    decoded_binary = (decoded.detach().cpu() > 0).float().numpy().flatten()
    
    # Convert binary to text
    binary_message = ''.join(['1' if b > 0.5 else '0' for b in decoded_binary])
    
    # Convert each 8 bits to a character
    chars = []
    for i in range(0, len(binary_message), 8):
        if i+8 <= len(binary_message):
            byte = binary_message[i:i+8]
            chars.append(chr(int(byte, 2)))
            if len(chars) >= message_length:
                break
    
    message = ''.join(chars[:message_length])
    print(f"Decoded message: {message}")
    
    return message

def display_comparison(original_path, stego_path, metrics):
    """Display the original and steganographic images side by side."""
    original = Image.open(original_path).convert('RGB')
    stego = Image.open(stego_path).convert('RGB')
    
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
        pass  # In case we're running in an environment without display

def main():
    parser = argparse.ArgumentParser(description="Evaluate SteganoGAN X-ray model")
    parser.add_argument('--model', type=str, default='pretrained/medical_xray.steg',
                        help='Path to the trained model')
    parser.add_argument('--input', type=str, default='steganogan/data/xrays/val/0001.jpg',
                        help='Input image for encoding')
    parser.add_argument('--output', type=str, default='stego_xray.png',
                        help='Output path for steganographic image')
    parser.add_argument('--message', type=str, default='Name: YALLAPA\nAge: 17Y\nID: CDS245909',
                        help='Message to encode')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Encode the message
    metrics = encode_message(model, args.input, args.output, args.message, args.cuda)
    
    # Display the comparison
    display_comparison(args.input, args.output, metrics)
    
    # Decode the message
    decoded = decode_message(model, args.output, len(args.message), args.cuda)
    
    # Verify if the decoded message matches the original
    if decoded == args.message:
        print("✅ Success! The decoded message matches the original.")
    else:
        print("⚠️ The decoded message differs from the original:")
        print(f"  Original: {args.message}")
        print(f"  Decoded:  {decoded}")
        
        # Calculate similarity percentage
        matching = sum(1 for a, b in zip(args.message, decoded) if a == b)
        similarity = matching / len(args.message) * 100 if args.message else 0
        print(f"  Character match: {similarity:.1f}%")

if __name__ == "__main__":
    main()