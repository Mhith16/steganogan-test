#!/usr/bin/env python3
"""
Enhanced steganography model for X-ray images using SteganoGAN architecture
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
import torch.backends.cudnn as cudnn
import zlib
from collections import Counter

# Try to import reedsolo, with error handling
try:
    from reedsolo import RSCodec
    rs = RSCodec(250)  # SteganoGAN uses 250 bytes for error correction
except ImportError:
    print("Warning: reedsolo module not found. Please install with 'pip install reedsolo'")
    print("Continuing without error correction...")
    rs = None

# Enable cuDNN benchmarking for faster convolutions
cudnn.benchmark = True

# Text processing utilities from SteganoGAN
def text_to_bits(text):
    """Convert text to a list of ints in {0, 1}"""
    return bytearray_to_bits(text_to_bytearray(text))

def bits_to_text(bits):
    """Convert a list of ints in {0, 1} to text"""
    return bytearray_to_text(bits_to_bytearray(bits))

def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def bits_to_bytearray(bits):
    """Convert a list of bits to a bytearray"""
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
    return bytearray(ints)

def text_to_bytearray(text):
    """Compress and add error correction"""
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    if rs is not None:
        x = rs.encode(bytearray(x))
    return x

def bytearray_to_text(x):
    """Apply error correction and decompress"""
    try:
        if rs is not None:
            x = rs.decode(x)[0]  # Get the corrected data
        text = zlib.decompress(x)
        return text.decode("utf-8")
    except Exception as e:
        print(f"Error in decoding: {e}")
        return False

# Define SteganoGAN-style encoder
class DenseEncoder(nn.Module):
    """
    The DenseEncoder module takes a cover image and data tensor and combines
    them into a steganographic image using dense connectivity.
    """
    def __init__(self, data_depth=1, hidden_size=64):
        super(DenseEncoder, self).__init__()
        self.data_depth = data_depth
        
        # First layer - feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size)
        )
        
        # Second layer - combine features with data
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size + data_depth, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size)
        )
        
        # Third layer - process combined features with dense connectivity
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2 + data_depth, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size)
        )
        
        # Output layer - generate residual
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3 + data_depth, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, image, data):
        # Dense connectivity pattern (exactly like SteganoGAN)
        x1 = self.conv1(image)
        x2 = self.conv2(torch.cat([x1, data], dim=1))
        x3 = self.conv3(torch.cat([x1, x2, data], dim=1))
        x4 = self.conv4(torch.cat([x1, x2, x3, data], dim=1))
        
        # Residual connection (crucial for image quality)
        return image + x4

# Define SteganoGAN-style decoder
class DenseDecoder(nn.Module):
    """
    The DenseDecoder module takes a steganographic image and attempts to decode
    the embedded data tensor using dense connectivity.
    """
    def __init__(self, data_depth=1, hidden_size=64):
        super(DenseDecoder, self).__init__()
        self.data_depth = data_depth
        
        # First layer - feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size)
        )
        
        # Second layer - feature processing
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size)
        )
        
        # Third layer - process with dense connectivity
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(hidden_size)
        )
        
        # Output layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_size * 3, data_depth, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # Dense connectivity pattern (exactly like SteganoGAN)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x1, x2, x3], dim=1))
        
        return x4

# Define SteganoGAN-style critic
class BasicCritic(nn.Module):
    """
    The Critic module takes an image and predicts whether it is a cover
    image or a steganographic image.
    """
    def __init__(self, hidden_size=64):
        super(BasicCritic, self).__init__()
        
        # Simple sequential model as in SteganoGAN
        self.layers = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, image):
        x = self.layers(image)
        # Global mean
        return torch.mean(x.view(x.size(0), -1), dim=1)

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

# Helper function to create random binary data for training
def generate_random_data(batch_size, data_depth, height, width, device):
    """Generate random binary data for training."""
    return torch.randint(0, 2, (batch_size, data_depth, height, width), device=device).float()

# Enhanced SteganoGAN-style training function
def train_model(train_dir, val_dir, output_dir='stegano_model', epochs=10, 
                batch_size=8, data_depth=1, img_size=256, hidden_size=64, 
                use_cuda=False):
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")
    
    # Create model directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Data transformation - SteganoGAN normalization
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create datasets and dataloaders
    train_dataset = XrayDataset(train_dir, transform, (img_size, img_size))
    val_dataset = XrayDataset(val_dir, transform, (img_size, img_size))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
    
    # Initialize models with SteganoGAN-like architecture
    encoder = DenseEncoder(data_depth, hidden_size).to(device)
    decoder = DenseDecoder(data_depth, hidden_size).to(device)
    critic = BasicCritic(hidden_size).to(device)
    
    # Setup optimizers with SteganoGAN learning rates
    _dec_list = list(decoder.parameters()) + list(encoder.parameters())
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)  # SteganoGAN rate
    decoder_optimizer = optim.Adam(_dec_list, lr=1e-4)  # SteganoGAN rate
    
    # Training loop
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        critic.train()
        
        train_encoder_loss = 0
        train_decoder_loss = 0
        train_critic_loss = 0
        train_generated_score = 0
        train_cover_score = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        for batch_idx, cover in enumerate(train_loader):
            cover = cover.to(device)
            batch_size, _, height, width = cover.size()
            
            # Generate random binary data
            payload = generate_random_data(batch_size, data_depth, height, width, device)
            
            # ===== Train Critic (exactly like SteganoGAN) =====
            critic_optimizer.zero_grad()
            
            # Generate stego images
            with torch.no_grad():
                generated = encoder(cover, payload)
            
            # Critic scores
            cover_score = torch.mean(critic(cover))
            generated_score = torch.mean(critic(generated))
            
            # Wasserstein loss for critic
            critic_loss = cover_score - generated_score
            
            # Backward and optimize
            critic_loss.backward()
            critic_optimizer.step()
            
            # Weight clipping as in SteganoGAN
            for p in critic.parameters():
                p.data.clamp_(-0.1, 0.1)
            
            # ===== Train Encoder and Decoder (exactly like SteganoGAN) =====
            decoder_optimizer.zero_grad()
            
            # Forward pass
            generated = encoder(cover, payload)
            decoded = decoder(generated)
            
            # Calculate losses
            encoder_mse = nn.MSELoss()(generated, cover)
            decoder_loss = nn.BCEWithLogitsLoss()(decoded, payload)
            generated_score = torch.mean(critic(generated))
            
            # Combined loss - SteganoGAN formula with 100x weight on MSE
            combined_loss = 100.0 * encoder_mse + decoder_loss + generated_score
            
            # Backward pass and optimize
            combined_loss.backward()
            decoder_optimizer.step()
            
            # Update metrics
            train_encoder_loss += encoder_mse.item()
            train_decoder_loss += decoder_loss.item()
            train_critic_loss += critic_loss.item()
            train_generated_score += generated_score.item()
            train_cover_score += cover_score.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, "
                     f"Encoder MSE: {encoder_mse.item():.4f}, "
                     f"Decoder Loss: {decoder_loss.item():.4f}, "
                     f"Cover Score: {cover_score.item():.4f}, "
                     f"Generated Score: {generated_score.item():.4f}")
        
        # Calculate average losses
        train_encoder_loss /= len(train_loader)
        train_decoder_loss /= len(train_loader)
        train_critic_loss /= len(train_loader)
        train_generated_score /= len(train_loader)
        train_cover_score /= len(train_loader)
        
        # Validation
        encoder.eval()
        decoder.eval()
        critic.eval()
        
        val_encoder_loss = 0
        val_decoder_loss = 0
        val_critic_loss = 0
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
                generated = encoder(cover, payload)
                decoded = decoder(generated)
                
                # Critic scores
                cover_score = torch.mean(critic(cover))
                generated_score = torch.mean(critic(generated))
                
                # Calculate losses
                encoder_mse = nn.MSELoss()(generated, cover)
                decoder_loss = nn.BCEWithLogitsLoss()(decoded, payload)
                critic_loss = cover_score - generated_score
                
                # For PSNR/SSIM calculation, convert from [-1,1] to [0,1] range
                generated_np = ((generated + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)
                cover_np = ((cover + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)
                
                for i in range(batch_size):
                    val_psnr += peak_signal_noise_ratio(cover_np[i], generated_np[i])
                    val_ssim += structural_similarity(
                        cover_np[i], generated_np[i], multichannel=True, channel_axis=2, data_range=1.0)
                
                # Calculate bit accuracy
                bit_accuracy = torch.mean(((decoded >= 0) == (payload >= 0.5)).float())
                
                # Update metrics
                val_encoder_loss += encoder_mse.item()
                val_decoder_loss += decoder_loss.item()
                val_critic_loss += critic_loss.item()
                val_bit_accuracy += bit_accuracy.item()
        
        # Calculate average validation metrics
        val_encoder_loss /= len(val_loader)
        val_decoder_loss /= len(val_loader)
        val_critic_loss /= len(val_loader)
        val_psnr /= len(val_loader) * batch_size
        val_ssim /= len(val_loader) * batch_size
        val_bit_accuracy /= len(val_loader)
        
        print(f"Epoch {epoch+1} Results:")
        print(f"Train Encoder MSE: {train_encoder_loss:.4f}, Train Decoder Loss: {train_decoder_loss:.4f}")
        print(f"Train Critic Loss: {train_critic_loss:.4f}")
        print(f"Train Cover Score: {train_cover_score:.4f}, Train Generated Score: {train_generated_score:.4f}")
        print(f"Val Encoder MSE: {val_encoder_loss:.4f}, Val Decoder Loss: {val_decoder_loss:.4f}")
        print(f"Val Critic Loss: {val_critic_loss:.4f}")
        print(f"Val PSNR: {val_psnr:.2f} dB, Val SSIM: {val_ssim:.4f}")
        print(f"Val Bit Accuracy: {val_bit_accuracy:.4f}")
        
        # Save models after each epoch
        torch.save(encoder, os.path.join(output_dir, f'encoder_epoch_{epoch+1}.pt'))
        torch.save(decoder, os.path.join(output_dir, f'decoder_epoch_{epoch+1}.pt'))
        torch.save(critic, os.path.join(output_dir, f'critic_epoch_{epoch+1}.pt'))
    
    # Save final models
    torch.save(encoder, os.path.join(output_dir, 'encoder_final.pt'))
    torch.save(decoder, os.path.join(output_dir, 'decoder_final.pt'))
    torch.save(critic, os.path.join(output_dir, 'critic_final.pt'))
    
    print("Training completed. Models saved.")
    return encoder, decoder, critic

def _make_payload(width, height, depth, text):
    """
    This takes a piece of text and encodes it into a bit vector. It then
    fills a matrix of size (width, height) with copies of the bit vector.
    """
    message = text_to_bits(text) + [0] * 32
    
    payload = message
    while len(payload) < width * height * depth:
        payload += message

    payload = payload[:width * height * depth]

    return torch.FloatTensor(payload).view(1, depth, height, width)

# Function to encode a message into an image (SteganoGAN-style)
def encode_message(encoder, input_image, output_image, message, data_depth=1, use_cuda=False):
    """Encode a message into an image using SteganoGAN approach."""
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.eval()
    
    # Load and prepare the image (SteganoGAN scaling/normalization)
    image = Image.open(input_image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Create payload using SteganoGAN's approach
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    payload = _make_payload(width, height, data_depth, message).to(device)
    
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

# Function to decode a message from an image (SteganoGAN-style)
def decode_message(decoder, stego_image, data_depth=1, use_cuda=False):
    """Decode a message from an image using SteganoGAN approach."""
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
    
    # Extract binary message using SteganoGAN approach
    image = (decoded >= 0).cpu().numpy().flatten()
    
    # split and decode messages
    candidates = Counter()
    bits = image.tolist()
    for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
        candidate = bytearray_to_text(bytearray(candidate))
        if candidate:
            candidates[candidate] += 1
    
    # choose most common message
    if len(candidates) == 0:
        # If no messages were found, try to extract printable characters
        raw_message = ""
        for i in range(0, min(1000, len(bits)), 8):
            if i + 8 <= len(bits):
                byte = 0
                for j in range(8):
                    byte = (byte << 1) | int(bits[i + j])
                
                if 32 <= byte <= 126:
                    raw_message += chr(byte)
        
        if len(raw_message) > 0:
            # Look for repeating patterns
            for pattern_length in range(1, min(50, len(raw_message) // 2)):
                pattern = raw_message[:pattern_length]
                count = 0
                for i in range(0, len(raw_message), pattern_length):
                    if raw_message[i:i+pattern_length] == pattern:
                        count += 1
                    else:
                        break
                
                if count >= 2:
                    print(f"Found repeating pattern: {pattern}")
                    return pattern
            
            return raw_message[:50]  # Return first 50 chars if no pattern found
            
        raise ValueError('Failed to find message.')
    
    candidate, count = candidates.most_common(1)[0]
    return candidate

# Function to test the critic's detection capabilities
def detect_steganography(critic, cover_image, stego_image, use_cuda=False):
    """Test the critic's ability to detect steganographic images."""
    # Set device
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    critic.to(device)
    critic.eval()
    
    # Load images
    cover = Image.open(cover_image).convert('RGB')
    stego = Image.open(stego_image).convert('RGB')
    
    # Transform images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    cover_tensor = transform(cover).unsqueeze(0).to(device)
    stego_tensor = transform(stego).unsqueeze(0).to(device)
    
    # Get critic scores
    with torch.no_grad():
        cover_score = critic(cover_tensor).item()
        stego_score = critic(stego_tensor).item()
    
    print("\nCritic Detection Results:")
    print(f"Cover Image Score: {cover_score:.4f}")
    print(f"Stego Image Score: {stego_score:.4f}")
    
    # In WGAN, lower scores indicate "real" and higher scores indicate "fake"
    score_diff = stego_score - cover_score
    
    if score_diff > 0.5:
        print("The critic STRONGLY detects steganography in the stego image.")
    elif score_diff > 0.1:
        print("The critic detects steganography in the stego image.")
    elif score_diff > -0.1:
        print("The critic slightly detects steganography in the stego image.")
    else:
        print("The critic does NOT detect steganography in the stego image (good hiding).")
    
    print(f"Detection Confidence: {min(abs(score_diff) * 100, 100):.1f}%")

# Main function
def main():
    parser = argparse.ArgumentParser(description="SteganoGAN-style steganography model")
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--encode', action='store_true', help='Encode a message')
    parser.add_argument('--decode', action='store_true', help='Decode a message')
    parser.add_argument('--train_dir', type=str, default='steganogan/data/xrays/train',
                        help='Directory with training images')
    parser.add_argument('--val_dir', type=str, default='steganogan/data/xrays/val',
                        help='Directory with validation images')
    parser.add_argument('--model_dir', type=str, default='stegano_model',
                        help='Directory to save/load models')
    parser.add_argument('--input', type=str, default='steganogan/data/xrays/val/0001.jpg',
                        help='Input image for encoding')
    parser.add_argument('--output', type=str, default='stego_image.png',
                        help='Output path for steganographic image')
    parser.add_argument('--message', type=str, default='Your secret message',
                        help='Message to encode')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--data_depth', type=int, default=1, help='Data depth')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for networks')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    if args.train:
        print("Training new model...")
        encoder, decoder, critic = train_model(
            args.train_dir, 
            args.val_dir, 
            args.model_dir,
            args.epochs,
            args.batch_size,
            args.data_depth,
            hidden_size=args.hidden_size,
            use_cuda=args.cuda
        )
    else:
        # Load existing models
        encoder_path = os.path.join(args.model_dir, 'encoder_final.pt')
        decoder_path = os.path.join(args.model_dir, 'decoder_final.pt')
        critic_path = os.path.join(args.model_dir, 'critic_final.pt')
        
        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            print("Models not found. You need to train the model first with --train")
            return
        
        encoder = torch.load(encoder_path, map_location='cpu')
        decoder = torch.load(decoder_path, map_location='cpu')
        
        if os.path.exists(critic_path):
            critic = torch.load(critic_path, map_location='cpu')
            print("Critic model loaded.")
        else:
            print("Critic model not found. Only encoder and decoder are loaded.")
            critic = None
    
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
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(np.array(stego))
        ax2.set_title(f"Steganographic Image\nPSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig("comparison.png")
        print("Comparison image saved to comparison.png")
        
        try:
            plt.show()
        except:
            pass
        
        # If critic is available, try to detect
        if critic is not None:
            detect_steganography(critic, args.input, args.output, args.cuda)
    
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