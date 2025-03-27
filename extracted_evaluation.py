#!/usr/bin/env python3
"""
Simple evaluation using extracted encoder and decoder models
"""
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

def encode_message(encoder, input_image, output_image, message, use_cuda=False):
    """Encode a message into an image using the encoder model."""
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
    
    # Create payload tensor (1 channel of binary data)
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    data_depth = 1  # Assuming data_depth is 1 based on your training
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
        generated = encoder(image_tensor, payload)
    
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

def decode_message(decoder, stego_image, expected_length, use_cuda=False):
    """Decode a message from a steganographic image."""
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    decoder.to(device)
    print(f"Using {device} for decoding")
    
    # Load and prepare the image
    image = Image.open(stego_image).convert('RGB')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    image_tensor = image_tensor.to(device)
    
    # Decode the message
    with torch.no_grad():
        decoded = decoder(image_tensor)
    
    # Convert decoder output to binary
    binary_values = (decoded.detach().cpu() > 0).float().numpy().flatten()
    binary_message = ''.join(['1' if b > 0.5 else '0' for b in binary_values])
    
    # Calculate how many characters we can extract
    chars_to_extract = min(len(binary_message) // 8, expected_length * 2)
    
    # Convert each 8 bits to a character
    message = ''
    for i in range(0, chars_to_extract * 8, 8):
        byte = binary_message[i:i+8]
        try:
            char = chr(int(byte, 2))
            message += char
        except:
            pass
    
    # Try to find a repeating pattern (since we encoded by repeating the message)
    def find_pattern(text, max_length):
        for length in range(1, max_length + 1):
            pattern = text[:length]
            if pattern and pattern * (len(text) // length) == text[:len(pattern) * (len(text) // length)]:
                return pattern
        return text
    
    # Look for repeated pattern in first ~100 characters
    pattern = find_pattern(message[:100], expected_length)
    
    print(f"Raw decoded message (first 100 chars): {message[:100]}")
    print(f"Detected pattern: {pattern}")
    
    return pattern

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
        pass

def main():
    parser = argparse.ArgumentParser(description="Evaluate SteganoGAN X-ray model")
    parser.add_argument('--encoder', type=str, default='extracted/encoder.pt',
                        help='Path to the extracted encoder model')
    parser.add_argument('--decoder', type=str, default='extracted/decoder.pt',
                        help='Path to the extracted decoder model')
    parser.add_argument('--input', type=str, default='steganogan/data/xrays/val/0001.jpg',
                        help='Input image for encoding')
    parser.add_argument('--output', type=str, default='stego_xray.png',
                        help='Output path for steganographic image')
    parser.add_argument('--message', type=str, default='Name: YALLAPA',
                        help='Message to encode')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Check if extracted models exist
    if not os.path.exists(args.encoder) or not os.path.exists(args.decoder):
        print("Extracted models not found. Run extract_models.py first.")
        return
    
    # Load the encoder and decoder
    print(f"Loading encoder from {args.encoder}")
    encoder = torch.load(args.encoder, map_location='cpu')
    print(f"Loading decoder from {args.decoder}")
    decoder = torch.load(args.decoder, map_location='cpu')
    
    # Encode the message
    metrics = encode_message(encoder, args.input, args.output, args.message, args.cuda)
    
    # Display comparison
    display_comparison(args.input, args.output, metrics)
    
    # Decode the message
    decoded = decode_message(decoder, args.output, len(args.message), args.cuda)
    
    # Compare original and decoded messages
    if decoded == args.message:
        print("✅ Success! The decoded message matches the original.")
    else:
        print("⚠️ The decoded message differs from the original:")
        print(f"  Original: {args.message}")
        print(f"  Decoded:  {decoded}")
        
        # Calculate character match percentage
        matching = sum(1 for a, b in zip(args.message, decoded) if a == b)
        similarity = matching / max(len(args.message), len(decoded)) * 100
        print(f"  Character match: {similarity:.1f}%")

if __name__ == "__main__":
    main()