#!/usr/bin/env python3
"""
Evaluation script for SteganoGAN on X-ray images
"""
import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def load_model(path):
    """Load the trained SteganoGAN model."""
    print(f"Loading model from {path}")
    model = torch.load(path, map_location='cpu')
    return model

def encode_image(model, input_image, output_image, message, use_cuda=False):
    """Encode a message into an image."""
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    model.set_device(use_cuda)
    
    # Load the cover image
    cover = Image.open(input_image).convert('RGB')
    cover_array = np.array(cover)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    cover_tensor = transform(cover).unsqueeze(0)
    
    # Create payload
    _, _, height, width = cover_tensor.size()
    payload = _make_payload(model, width, height, model.data_depth, message)
    
    # Move tensors to device
    cover_tensor = cover_tensor.to(device)
    payload = payload.to(device)
    
    # Encode the message
    print(f"Encoding message into image...")
    generated = model.encoder(cover_tensor, payload)[0].clamp(-1.0, 1.0)
    
    # Convert back to image
    generated = (generated.permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5
    generated_image = Image.fromarray(generated.astype('uint8'))
    
    # Save the image
    generated_image.save(output_image)
    print(f"Steganographic image saved to {output_image}")
    
    # Calculate and return metrics
    metrics = calculate_metrics(cover_array, generated.astype('uint8'))
    return generated_image, metrics

def decode_image(model, stego_image, use_cuda=False):
    """Decode a message from a steganographic image."""
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    model.set_device(use_cuda)
    
    # Load the steganographic image
    stego = Image.open(stego_image).convert('RGB')
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    stego_tensor = transform(stego).unsqueeze(0)
    
    # Move tensor to device
    stego_tensor = stego_tensor.to(device)
    
    # Decode the message
    print(f"Decoding message from image...")
    decoded = model.decoder(stego_tensor).view(-1) > 0
    
    # Convert bits to text
    bits = decoded.data.cpu().numpy().tolist()
    message = _bits_to_text(bits)
    
    return message

def _make_payload(model, width, height, depth, text):
    """Create a payload tensor from text."""
    # Convert text to bits
    message = _text_to_bits(text) + [0] * 32
    
    # Repeat the message to fill the payload
    payload = message
    while len(payload) < width * height * depth:
        payload += message
    
    payload = payload[:width * height * depth]
    return torch.FloatTensor(payload).view(1, depth, height, width)

def _text_to_bits(text):
    """Convert text to bits."""
    # This is a simplified version - in production, use the full implementation
    byte_array = text.encode('utf-8')
    bits = []
    for byte in byte_array:
        bits_str = format(byte, '08b')
        bits.extend([int(bit) for bit in bits_str])
    return bits

def _bits_to_text(bits):
    """Convert bits to text."""
    # Group bits into bytes
    bytes_list = []
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            bytes_list.append(byte)
    
    # Try different slices to find valid UTF-8
    for end in range(len(bytes_list), 0, -1):
        try:
            text = bytes(bytes_list[:end]).decode('utf-8')
            return text
        except UnicodeDecodeError:
            continue
    
    return "Failed to decode message"

def calculate_metrics(original, stego):
    """Calculate image quality metrics."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(original, stego)
    
    # Calculate SSIM (on grayscale images)
    original_gray = np.mean(original, axis=2).astype(np.uint8)
    stego_gray = np.mean(stego, axis=2).astype(np.uint8)
    ssim = structural_similarity(original_gray, stego_gray)
    
    return {'psnr': psnr, 'ssim': ssim}

def display_images(original_path, stego_path, metrics):
    """Display the original and steganographic images side by side."""
    original = Image.open(original_path).convert('RGB')
    stego = Image.open(stego_path).convert('RGB')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(np.array(original))
    ax1.set_title('Original X-ray')
    ax1.axis('off')
    
    ax2.imshow(np.array(stego))
    ax2.set_title('Steganographic X-ray\nPSNR: {:.2f}dB, SSIM: {:.4f}'.format(
        metrics['psnr'], metrics['ssim']))
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate SteganoGAN on X-ray images')
    parser.add_argument('--model', type=str, default='pretrained/medical_xray.steg',
                        help='Path to the trained model')
    parser.add_argument('--mode', type=str, choices=['encode', 'decode', 'both'], default='both',
                        help='Operation mode: encode, decode, or both')
    parser.add_argument('--input', type=str, default='steganogan/data/xrays/val/0001.jpg',
                        help='Path to the input image (for encoding)')
    parser.add_argument('--output', type=str, default='stego_xray.png',
                        help='Path to save the steganographic image')
    parser.add_argument('--message', type=str, 
                        default='Name: YALLAPA\nAge: 17Y\nID: CDS245909',
                        help='Message to encode')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    if args.mode in ['encode', 'both']:
        # Encode the message
        stego_image, metrics = encode_image(
            model, args.input, args.output, args.message, args.cuda)
        print(f"Encoding metrics: PSNR = {metrics['psnr']:.2f}dB, SSIM = {metrics['ssim']:.4f}")
        
    if args.mode in ['decode', 'both']:
        # Decode the message
        decoded_message = decode_image(model, args.output, args.cuda)
        print(f"Decoded message: {decoded_message}")
        
        # Verify if the decoded message matches the original
        if args.mode == 'both':
            if decoded_message == args.message:
                print("✅ Success! The decoded message matches the original.")
            else:
                print("❌ The decoded message does not match the original.")
                print(f"Original: {args.message}")
                print(f"Decoded:  {decoded_message}")
    
    # Display the images
    if args.mode in ['encode', 'both']:
        display_images(args.input, args.output, metrics)

if __name__ == '__main__':
    main()