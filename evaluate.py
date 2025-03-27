#!/usr/bin/env python3
"""
Evaluation script for SteganoGAN on X-ray images
"""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import zlib
from collections import Counter

# First, define the same SteganoGAN class and helper classes
class BasicEncoder(torch.nn.Module):
    """Basic encoder network."""
    add_image = False
    
    def _conv2d(self, in_channels, out_channels):
        return torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=3, padding=1)

    def _build_models(self):
        self.features = torch.nn.Sequential(
            self._conv2d(3, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),
        )
        self.layers = torch.nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, 3),
            torch.nn.Tanh(),
        )
        return self.features, self.layers

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.version = '2'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def forward(self, image, data):
        x = self._models[0](image)
        x_list = [x]

        for layer in self._models[1:]:
            x = layer(torch.cat(x_list + [data], dim=1))
            x_list.append(x)

        if self.add_image:
            x = image + x

        return x


class DenseEncoder(BasicEncoder):
    """Dense encoder network."""
    add_image = True

    def _build_models(self):
        self.conv1 = torch.nn.Sequential(
            self._conv2d(3, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = torch.nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = torch.nn.Sequential(
            self._conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = torch.nn.Sequential(
            self._conv2d(self.hidden_size * 3 + self.data_depth, 3)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4


class BasicDecoder(torch.nn.Module):
    """Basic decoder network."""
    def _conv2d(self, in_channels, out_channels):
        return torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=3, padding=1)

    def _build_models(self):
        self.layers = torch.nn.Sequential(
            self._conv2d(3, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.data_depth)
        )

        return [self.layers]

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.version = '2'
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def forward(self, x):
        x = self._models[0](x)
        return x


class DenseDecoder(BasicDecoder):
    """Dense decoder network."""
    def _build_models(self):
        self.conv1 = torch.nn.Sequential(
            self._conv2d(3, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size)
        )

        self.conv2 = torch.nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size)
        )

        self.conv3 = torch.nn.Sequential(
            self._conv2d(self.hidden_size * 2, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size)
        )

        self.conv4 = torch.nn.Sequential(
            self._conv2d(self.hidden_size * 3, self.data_depth)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, x):
        x1 = self._models[0](x)
        x2 = self._models[1](x1)
        x3 = self._models[2](torch.cat([x1, x2], dim=1))
        x4 = self._models[3](torch.cat([x1, x2, x3], dim=1))
        return x4


class BasicCritic(torch.nn.Module):
    """Basic critic network."""
    def _conv2d(self, in_channels, out_channels):
        return torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=3)

    def _build_models(self):
        return torch.nn.Sequential(
            self._conv2d(3, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, 1)
        )

    def __init__(self, hidden_size):
        super().__init__()
        self.version = '2'
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def forward(self, x):
        x = self._models(x)
        x = torch.mean(x.view(x.size(0), -1), dim=1)
        return x


# Reed-Solomon codec for error correction (simplified version)
class RSCodec:
    def __init__(self, nsym):
        self.nsym = nsym
        
    def encode(self, data):
        # Simple version just adds padding
        return data + bytearray([0] * self.nsym)
        
    def decode(self, data):
        # Simple version just removes padding
        return data[:-self.nsym]

rs = RSCodec(250)


# Text conversion functions
def text_to_bits(text):
    """Convert text to a list of bits."""
    return bytearray_to_bits(text_to_bytearray(text))


def bits_to_text(bits):
    """Convert a list of bits to text."""
    return bytearray_to_text(bits_to_bytearray(bits))


def bytearray_to_bits(x):
    """Convert bytearray to a list of bits."""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def bits_to_bytearray(bits):
    """Convert a list of bits to a bytearray."""
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
    return bytearray(ints)


def text_to_bytearray(text):
    """Compress and add error correction."""
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))
    return x


def bytearray_to_text(x):
    """Apply error correction and decompress."""
    try:
        text = rs.decode(x)
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except Exception:
        return False


# Main SteganoGAN class
class SteganoGAN(object):
    """SteganoGAN model class."""
    
    def __init__(self, data_depth, encoder, decoder, critic,
                 cuda=False, verbose=False, log_dir=None, **kwargs):
        self.verbose = verbose
        self.data_depth = data_depth
        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic
        self.cuda = cuda
        self.device = torch.device('cpu')
        self.log_dir = log_dir
        self.history = []
        self.fit_metrics = {}
        
    def set_device(self, cuda=True):
        """Set device for computation."""
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')
            
        if self.verbose:
            print(f"Using {'CUDA' if self.cuda else 'CPU'} device")
            
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)
        
    def _random_data(self, cover):
        """Generate random data for training."""
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)
        
    def _make_payload(self, width, height, depth, text):
        """Create a payload tensor from text."""
        message = text_to_bits(text) + [0] * 32
        
        # Fill payload with repeated message
        payload = message
        while len(payload) < width * height * depth:
            payload += message
            
        payload = payload[:width * height * depth]
        return torch.FloatTensor(payload).view(1, depth, height, width)
        
    def encode(self, cover, output, text):
        """Encode a message in an image."""
        # Process cover image
        cover_img = np.array(Image.open(cover).convert('RGB'))
        cover_tensor = torch.FloatTensor(cover_img).permute(2, 0, 1) / 127.5 - 1.0
        cover_tensor = cover_tensor.unsqueeze(0)
        
        # Create payload
        height, width = cover_tensor.size()[2:]
        payload = self._make_payload(width, height, self.data_depth, text)
        
        # Move to device
        cover_tensor = cover_tensor.to(self.device)
        payload = payload.to(self.device)
        
        # Generate steganographic image
        generated = self.encoder(cover_tensor, payload)[0].clamp(-1.0, 1.0)
        
        # Save output
        generated = (generated.permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5
        Image.fromarray(generated.astype('uint8')).save(output)
        
        if self.verbose:
            print(f"Encoded message into {output}")
        
    def decode(self, image):
        """Decode a message from an image."""
        if not os.path.exists(image):
            raise ValueError(f"Image not found: {image}")
            
        # Load image
        img = np.array(Image.open(image).convert('RGB')) / 255.0
        img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        img_tensor = img_tensor.to(self.device)
        
        # Decode message
        decoded = self.decoder(img_tensor).view(-1) > 0
        
        # Process decoded bits
        candidates = Counter()
        bits = decoded.data.cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1
                
        # Return most common message
        if not candidates:
            raise ValueError("Failed to decode any message")
            
        message, _ = candidates.most_common(1)[0]
        return message
    
    def save(self, path):
        """Save model to file."""
        torch.save(self, path)


def load_model(path):
    """Load a saved SteganoGAN model."""
    print(f"Loading model from {path}")
    try:
        model = torch.load(path, map_location='cpu')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def calculate_metrics(original, stego):
    """Calculate PSNR and SSIM."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(original, stego)
    
    # Calculate SSIM
    original_gray = np.mean(original, axis=2).astype(np.uint8)
    stego_gray = np.mean(stego, axis=2).astype(np.uint8)
    ssim = structural_similarity(original_gray, stego_gray)
    
    return {'psnr': psnr, 'ssim': ssim}

def display_images(original_path, stego_path, metrics=None):
    """Display original and steganographic images."""
    original = np.array(Image.open(original_path).convert('RGB'))
    stego = np.array(Image.open(stego_path).convert('RGB'))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original)
    ax1.set_title("Original X-ray")
    ax1.axis('off')
    
    title = "Steganographic X-ray"
    if metrics:
        title += f"\nPSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.4f}"
    
    ax2.imshow(stego)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison.png")
    print("Comparison image saved to comparison.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="SteganoGAN X-ray evaluation")
    parser.add_argument('--model', type=str, default='pretrained/medical_xray.steg',
                        help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['encode', 'decode', 'both'], default='both',
                        help='Operation mode')
    parser.add_argument('--input', type=str, default='steganogan/data/xrays/val/0001.jpg',
                        help='Input image path')
    parser.add_argument('--output', type=str, default='stego_xray.png',
                        help='Output image path')
    parser.add_argument('--message', type=str, 
                        default='Name: YALLAPA\nAge: 17Y\nID: CDS245909',
                        help='Message to encode')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    model.set_device(args.cuda)
    
    # Perform operations
    if args.mode in ['encode', 'both']:
        print(f"Encoding message into {args.output}...")
        model.encode(args.input, args.output, args.message)
        
        # Calculate metrics
        original = np.array(Image.open(args.input).convert('RGB'))
        stego = np.array(Image.open(args.output).convert('RGB'))
        metrics = calculate_metrics(original, stego)
        print(f"Image quality metrics:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        
        # Display images
        display_images(args.input, args.output, metrics)
    
    if args.mode in ['decode', 'both']:
        print(f"Decoding message from {args.output}...")
        try:
            decoded_message = model.decode(args.output)
            print(f"Decoded message:\n{decoded_message}")
            
            if args.mode == 'both':
                if decoded_message == args.message:
                    print("✅ Success! The decoded message matches the original.")
                else:
                    print("⚠️ The decoded message differs from the original:")
                    print(f"  Original: {args.message}")
                    print(f"  Decoded:  {decoded_message}")
        except Exception as e:
            print(f"Error decoding message: {e}")

if __name__ == "__main__":
    main()