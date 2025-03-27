#!/usr/bin/env python3
"""
Basic steganography model for X-ray images
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import random
from glob import glob

# Define simple encoder-decoder architecture
class Encoder(nn.Module):
    def __init__(self, data_depth=1):
        super(Encoder, self).__init__()
        self.data_depth = data_depth
        
        # Initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        # Combined processing
        self.conv2 = nn.Sequential(
            nn.Conv2d(32 + data_depth, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        # Final image generation
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, image, data):
        x = self.conv1(image)
        x = self.conv2(torch.cat([x, data], dim=1))
        x = self.conv3(x)
        return image + x  # Residual connection

class Decoder(nn.Module):
    def __init__(self, data_depth=1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, data_depth, kernel_size=3, padding=1)
        )
    
    def forward(self, image):
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# Dataset class for X-ray images
class XrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, size=(256, 256)):
        self.root_dir = root_dir
        self.transform = transform
        self.size = size
        self.image_files = [f for f in glob(os.path.join(root_dir, "*.jpg"))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.size)
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Helper function to create random binary data
def generate_random_data(batch_size, data_depth, height, width, device):
    return torch.randint(0, 2, (batch_size, data_depth, height, width), device=device).float()

# Training function
def train_model(train_dir, val_dir, output_dir='simple_model', epochs=10, 
                batch_size=4, data_depth=1, img_size=256, use_cuda=False):
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")
    
    # Create model directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create datasets and dataloaders
    train_dataset = XrayDataset(train_dir, transform, (img_size, img_size))
    val_dataset = XrayDataset(val_dir, transform, (img_size, img_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
    
    # Initialize models
    encoder = Encoder(data_depth).to(device)
    decoder = Decoder(data_depth).to(device)
    
    # Setup optimizers
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        train_encoder_loss = 0
        train_decoder_loss = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        for batch_idx, cover in enumerate(train_loader):
            cover = cover.to(device)
            batch_size, _, height, width = cover.size()
            
            # Generate random binary data
            payload = generate_random_data(batch_size, data_depth, height, width, device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            stego = encoder(cover, payload)
            decoded = decoder(stego)
            
            # Calculate losses
            encoder_loss = nn.MSELoss()(stego, cover)
            decoder_loss = nn.BCEWithLogitsLoss()(decoded, payload)
            
            # Combined loss
            loss = encoder_loss + decoder_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_encoder_loss += encoder_loss.item()
            train_decoder_loss += decoder_loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Encoder Loss: {encoder_loss.item():.4f}, "
                      f"Decoder Loss: {decoder_loss.item():.4f}")
        
        # Calculate average losses
        train_encoder_loss /= len(train_loader)
        train_decoder_loss /= len(train_loader)
        
        # Validation
        encoder.eval()
        decoder.eval()
        
        val_encoder_loss = 0
        val_decoder_loss = 0
        val_psnr = 0
        val_ssim = 0
        val_bit_accuracy = 0
        
        with torch.no_grad():
            for cover in val_loader:
                cover = cover.to(device)
                batch_size, _, height, width = cover.size()
                
                # Generate random binary data
                payload = generate_random_data(batch_size, data_depth, height, width, device)
                
                # Forward pass
                stego = encoder(cover, payload)
                decoded = decoder(stego)
                
                # Calculate losses
                encoder_loss = nn.MSELoss()(stego, cover)
                decoder_loss = nn.BCEWithLogitsLoss()(decoded, payload)
                
                # Calculate metrics
                stego_np = ((stego + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)
                cover_np = ((cover + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)
                
                for i in range(batch_size):
                    val_psnr += peak_signal_noise_ratio(cover_np[i], stego_np[i])
                    val_ssim += structural_similarity(
                        cover_np[i], stego_np[i], multichannel=True, channel_axis=2, data_range=1.0)
                
                # Calculate bit accuracy
                bit_accuracy = torch.mean(((decoded >= 0) == (payload >= 0.5)).float())
                
                # Update metrics
                val_encoder_loss += encoder_loss.item()
                val_decoder_loss += decoder_loss.item()
                val_bit_accuracy += bit_accuracy.item()
        
        # Calculate average validation metrics
        val_encoder_loss /= len(val_loader)
        val_decoder_loss /= len(val_loader)
        val_psnr /= len(val_loader) * batch_size
        val_ssim /= len(val_loader) * batch_size
        val_bit_accuracy /= len(val_loader)
        
        print(f"Epoch {epoch+1} Results:")
        print(f"Train Encoder Loss: {train_encoder_loss:.4f}, Train Decoder Loss: {train_decoder_loss:.4f}")
        print(f"Val Encoder Loss: {val_encoder_loss:.4f}, Val Decoder Loss: {val_decoder_loss:.4f}")
        print(f"Val PSNR: {val_psnr:.2f} dB, Val SSIM: {val_ssim:.4f}")
        print(f"Val Bit Accuracy: {val_bit_accuracy:.4f}")
        
        # Save models after each epoch
        torch.save(encoder, os.path.join(output_dir, f'encoder_epoch_{epoch+1}.pt'))
        torch.save(decoder, os.path.join(output_dir, f'decoder_epoch_{epoch+1}.pt'))
    
    # Save final models
    torch.save(encoder, os.path.join(output_dir, 'encoder_final.pt'))
    torch.save(decoder, os.path.join(output_dir, 'decoder_final.pt'))
    
    print("Training completed. Models saved.")
    return encoder, decoder

# Function to encode a message into an image
def encode_message(encoder, input_image, output_image, message, data_depth=1, use_cuda=False):
    """Encode a message into an image."""
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.eval()
    
    # Load and prepare the image
    image = Image.open(input_image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
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
                    idx = (idx + 1) % len(binary_message)
    
    # Encode the message
    with torch.no_grad():
        stego = encoder(image_tensor, payload)
    
    # Save the output image
    stego = stego.clamp(-1, 1)
    stego_img = ((stego[0] + 1) / 2).cpu().permute(1, 2, 0).numpy()
    stego_img = (stego_img * 255).astype(np.uint8)
    Image.fromarray(stego_img).save(output_image)
    
    # Calculate metrics
    original_img = ((image_tensor[0] + 1) / 2).cpu().permute(1, 2, 0).numpy()
    original_img = (original_img * 255).astype(np.uint8)
    
    psnr_value = peak_signal_noise_ratio(original_img, stego_img)
    ssim_value = structural_similarity(
        np.mean(original_img, axis=2), 
        np.mean(stego_img, axis=2), 
        data_range=255
    )
    
    print(f"Message encoded and saved to {output_image}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    
    return {'psnr': psnr_value, 'ssim': ssim_value}

# Function to decode a message from an image
# Replace the decode_message function with this improved version
def decode_message(decoder, stego_image, data_depth=1, use_cuda=False):
    """Decode a message from an image without requiring prior knowledge of the message."""
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    decoder.to(device)
    decoder.eval()
    
    # Load and prepare the image
    image = Image.open(stego_image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Decode the message
    with torch.no_grad():
        decoded = decoder(image_tensor)
    
    # Extract binary message
    decoded_bits = (decoded >= 0).cpu().numpy().flatten()
    
    # Convert bits to text, looking for valid characters
    message = ""
    repeating_pattern = None
    
    # Try to decode the first few hundred characters
    max_chars_to_check = 100
    
    for i in range(0, min(max_chars_to_check * 8, len(decoded_bits)), 8):
        if i + 8 <= len(decoded_bits):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | int(decoded_bits[i + j])
            
            # Only add printable ASCII characters
            if 32 <= byte <= 126:
                message += chr(byte)
    
    # Look for repeating patterns in the decoded message
    if len(message) > 0:
        # Try to find a pattern by looking at the first half of the decoded message
        for pattern_length in range(1, len(message) // 2):
            pattern = message[:pattern_length]
            # Check if this pattern repeats at least twice
            repetitions = 0
            for i in range(0, len(message), pattern_length):
                if message[i:i+pattern_length] == pattern:
                    repetitions += 1
                else:
                    break
            
            if repetitions >= 2:
                repeating_pattern = pattern
                break
    
    print(f"Decoded raw message: {message[:50]}..." if len(message) > 50 else f"Decoded raw message: {message}")
    
    if repeating_pattern:
        print(f"Detected repeating pattern: {repeating_pattern}")
        return repeating_pattern
    else:
        return message

def main():
    parser = argparse.ArgumentParser(description="Train and test simple steganography model")
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--encode', action='store_true', help='Encode a message')
    parser.add_argument('--decode', action='store_true', help='Decode a message')
    parser.add_argument('--train_dir', type=str, default='steganogan/data/xrays/train',
                        help='Directory with training images')
    parser.add_argument('--val_dir', type=str, default='steganogan/data/xrays/val',
                        help='Directory with validation images')
    parser.add_argument('--model_dir', type=str, default='simple_model',
                        help='Directory to save/load models')
    parser.add_argument('--input', type=str, default='steganogan/data/xrays/val/0001.jpg',
                        help='Input image for encoding')
    parser.add_argument('--output', type=str, default='stego_xray.png',
                        help='Output path for steganographic image')
    parser.add_argument('--message', type=str, default='Name: YALLAPA',
                        help='Message to encode')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--data_depth', type=int, default=1, help='Data depth')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    if args.train:
        print("Training new model...")
        encoder, decoder = train_model(
            args.train_dir, 
            args.val_dir, 
            args.model_dir,
            args.epochs,
            args.batch_size,
            args.data_depth,
            use_cuda=args.cuda
        )
    else:
        # Load existing models
        encoder_path = os.path.join(args.model_dir, 'encoder_final.pt')
        decoder_path = os.path.join(args.model_dir, 'decoder_final.pt')
        
        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            print("Models not found. You need to train the model first with --train")
            return
        
        encoder = torch.load(encoder_path, map_location='cpu')
        decoder = torch.load(decoder_path, map_location='cpu')
    
    if args.encode:
        print("Encoding message...")
        metrics = encode_message(
            encoder,
            args.input,
            args.output,
            args.message,
            args.data_depth,
            args.cuda
        )
        
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
    
    if args.decode:
        print("Decoding message...")
        decoded_message = decode_message(
            decoder,
            args.output,
            args.data_depth,
            args.cuda
        )
        
        # If a message was provided, compare the results
        if args.message:
            if decoded_message == args.message:
                print("✅ Success! The decoded message matches what you provided.")
            else:
                print("⚠️ The decoded message differs from what you provided:")
                print(f"  Expected: {args.message}")
                print(f"  Decoded:  {decoded_message}")
        else:
            print(f"Final decoded message: {decoded_message}")

if __name__ == "__main__":
    main()