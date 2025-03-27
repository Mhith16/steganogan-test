#!/usr/bin/env python3
"""
Self-contained SteganoGAN training script for X-rays
"""
import argparse
import json
import os
import sys
from time import time

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms

# --- Define models directly ---

class BasicEncoder(nn.Module):
    """Basic encoder network."""
    add_image = False
    
    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=3, padding=1)

    def _build_models(self):
        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, 3),
            nn.Tanh(),
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
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3 + self.data_depth, 3)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4


class BasicDecoder(nn.Module):
    """Basic decoder network."""
    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=3, padding=1)

    def _build_models(self):
        self.layers = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

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
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )

        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )

        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )

        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3, self.data_depth)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, x):
        x1 = self._models[0](x)
        x2 = self._models[1](x1)
        x3 = self._models[2](torch.cat([x1, x2], dim=1))
        x4 = self._models[3](torch.cat([x1, x2, x3], dim=1))
        return x4


class BasicCritic(nn.Module):
    """Basic critic network."""
    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=3)

    def _build_models(self):
        return nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),

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


# --- Medical dataset loader ---

_DEFAULT_MU = [.5, .5, .5]
_DEFAULT_SIGMA = [.5, .5, .5]

MEDICAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
])

class PatientDataset(torch.utils.data.Dataset):
    """Dataset for X-ray images paired with patient data text files."""
    def __init__(self, root_dir, transform=None, limit=np.inf):
        self.root_dir = root_dir
        self.transform = transform or MEDICAL_TRANSFORM
        
        # Find all image files
        self.image_files = []
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg'):
                self.image_files.append(os.path.join(root_dir, filename))
        
        self.image_files = sorted(self.image_files)
        if limit < np.inf:
            self.image_files = self.image_files[:limit]
            
        # Check for paired text files
        self.text_files = []
        for img_path in self.image_files:
            base_name = os.path.splitext(img_path)[0]
            txt_path = base_name + ".txt"
            if os.path.exists(txt_path):
                self.text_files.append(txt_path)
            else:
                self.text_files.append(None)
                
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        txt_path = self.text_files[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Load text data if available
        if txt_path and os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                text = f.read().strip()
        else:
            text = ""
            
        return image, text

class MedicalDataLoader(torch.utils.data.DataLoader):
    """Data loader for medical X-ray images paired with patient data."""
    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=4, batch_size=4, *args, **kwargs):
        if transform is None:
            transform = MEDICAL_TRANSFORM

        super().__init__(
            PatientDataset(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )


# --- Main SteganoGAN class ---

class SteganoGAN:
    """SteganoGAN: A GAN-based steganography tool."""
    
    def __init__(self, data_depth, encoder, decoder, critic,
                 cuda=False, verbose=False, log_dir=None, **kwargs):
        
        self.verbose = verbose
        self.data_depth = data_depth
        kwargs['data_depth'] = data_depth
        
        # Initialize models
        if encoder == DenseEncoder:
            self.encoder = DenseEncoder(data_depth, kwargs.get('hidden_size', 32))
        elif encoder == BasicEncoder:
            self.encoder = BasicEncoder(data_depth, kwargs.get('hidden_size', 32))
        else:
            self.encoder = encoder
            
        if decoder == DenseDecoder:
            self.decoder = DenseDecoder(data_depth, kwargs.get('hidden_size', 32))
        elif decoder == BasicDecoder:
            self.decoder = BasicDecoder(data_depth, kwargs.get('hidden_size', 32))
        else:
            self.decoder = decoder
            
        if critic == BasicCritic:
            self.critic = BasicCritic(kwargs.get('hidden_size', 32))
        else:
            self.critic = critic
        
        # Set device
        self.set_device(cuda)
        
        # Optimizers
        self.critic_optimizer = None
        self.decoder_optimizer = None
        
        # Logging
        self.fit_metrics = None
        self.history = list()
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.samples_path = os.path.join(self.log_dir, 'samples')
            os.makedirs(self.samples_path, exist_ok=True)
    
    def set_device(self, cuda=True):
        """Set the device to use."""
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        if self.verbose:
            if not cuda:
                print('Using CPU device')
            elif not self.cuda:
                print('CUDA is not available. Defaulting to CPU device')
            else:
                print('Using CUDA device')

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)
    
    def _random_data(self, cover):
        """Generate random data to hide in the cover image."""
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)
    
    def _encode_decode(self, cover, quantize=False):
        """Encode random data and then decode it."""
        payload = self._random_data(cover)
        generated = self.encoder(cover, payload)
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0

        decoded = self.decoder(generated)
        return generated, payload, decoded
    
    def _critic_score(self, image):
        """Get the critic score for an image."""
        return torch.mean(self.critic(image))
    
    def _get_optimizers(self):
        """Get optimizers for training."""
        dec_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)
        decoder_optimizer = Adam(dec_params, lr=1e-4)
        return critic_optimizer, decoder_optimizer
    
    def _fit_critic(self, train, metrics):
        """Train the critic."""
        for cover, _ in tqdm(train, disable=not self.verbose):
            cover = cover.to(self.device)
            payload = self._random_data(cover)
            generated = self.encoder(cover, payload)
            cover_score = self._critic_score(cover)
            generated_score = self._critic_score(generated)

            self.critic_optimizer.zero_grad()
            (cover_score - generated_score).backward(retain_graph=False)
            self.critic_optimizer.step()

            # Clamp critic weights
            for p in self.critic.parameters():
                p.data.clamp_(-0.1, 0.1)

            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())
    
    def _fit_coders(self, train, metrics):
        """Train the encoder and decoder."""
        for cover, _ in tqdm(train, disable=not self.verbose):
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded)
            generated_score = self._critic_score(generated)

            self.decoder_optimizer.zero_grad()
            (100.0 * encoder_mse + decoder_loss + generated_score).backward()
            self.decoder_optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())
    
    def _coding_scores(self, cover, generated, payload, decoded):
        """Calculate scores for the encoder and decoder."""
        encoder_mse = mse_loss(generated, cover)
        decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()
        return encoder_mse, decoder_loss, decoder_acc
    
    def _validate(self, validate, metrics):
        """Validate the model."""
        for cover, _ in tqdm(validate, disable=not self.verbose):
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover, quantize=True)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded)
            generated_score = self._critic_score(generated)
            cover_score = self._critic_score(cover)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            
            # Calculate SSIM
            ssim_value = self._ssim(cover, generated).item()
            metrics['val.ssim'].append(ssim_value)
            
            # Calculate PSNR
            psnr_value = 10 * torch.log10(4 / encoder_mse).item()
            metrics['val.psnr'].append(psnr_value)
            
            # Calculate BPP
            metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))
    
    def _ssim(self, img1, img2):
        """Calculate SSIM between two images."""
        # This is a simplified SSIM implementation
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = torch.mean(img1, dim=(2, 3), keepdim=True)
        mu2 = torch.mean(img2, dim=(2, 3), keepdim=True)
        
        sigma1_sq = torch.var(img1, dim=(2, 3), keepdim=True, unbiased=False)
        sigma2_sq = torch.var(img2, dim=(2, 3), keepdim=True, unbiased=False)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=(2, 3), keepdim=True)
        
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                  ((mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return torch.mean(ssim_map)
    
    def _generate_samples(self, samples_path, cover, epoch):
        """Generate sample images for logging."""
        import imageio
        
        cover = cover.to(self.device)
        generated, payload, decoded = self._encode_decode(cover)
        samples = generated.size(0)
        for sample in range(samples):
            cover_path = os.path.join(samples_path, '{}.cover.png'.format(sample))
            sample_name = '{}.generated-{:2d}.png'.format(sample, epoch)
            sample_path = os.path.join(samples_path, sample_name)

            image = (cover[sample].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.imwrite(cover_path, (255.0 * image).astype('uint8'))

            sampled = generated[sample].clamp(-1.0, 1.0).permute(1, 2, 0)
            sampled = sampled.detach().cpu().numpy() + 1.0

            image = sampled / 2.0
            imageio.imwrite(sample_path, (255.0 * image).astype('uint8'))
    
    def fit(self, train, validate, epochs=5):
        """Train the model."""
        # Metrics to track
        METRIC_FIELDS = [
            'val.encoder_mse', 'val.decoder_loss', 'val.decoder_acc',
            'val.cover_score', 'val.generated_score', 'val.ssim',
            'val.psnr', 'val.bpp', 'train.encoder_mse', 'train.decoder_loss',
            'train.decoder_acc', 'train.cover_score', 'train.generated_score',
        ]
        
        if self.critic_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._get_optimizers()
            self.epochs = 0

        if self.log_dir:
            sample_cover = next(iter(validate))[0]

        # Start training
        total = self.epochs + epochs
        for epoch in range(1, epochs + 1):
            # Count how many epochs we have trained for this steganogan
            self.epochs += 1

            metrics = {field: list() for field in METRIC_FIELDS}

            if self.verbose:
                print('Epoch {}/{}'.format(self.epochs, total))

            self._fit_critic(train, metrics)
            self._fit_coders(train, metrics)
            self._validate(validate, metrics)

            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items() if v}
            self.fit_metrics['epoch'] = epoch

            if self.log_dir:
                self.history.append(self.fit_metrics)

                metrics_path = os.path.join(self.log_dir, 'metrics.log')
                with open(metrics_path, 'w') as metrics_file:
                    json.dump(self.history, metrics_file, indent=4)

                save_name = '{}.bpp-{:03f}.p'.format(
                    self.epochs, self.fit_metrics['val.bpp'])

                self.save(os.path.join(self.log_dir, save_name))
                self._generate_samples(self.samples_path, sample_cover, epoch)

            # Empty cuda cache (this may help for memory leaks)
            if self.cuda:
                torch.cuda.empty_cache()
    
    def save(self, path):
        """Save the model."""
        torch.save(self, path)


# --- Main training function ---

def main():
    """Train a SteganoGAN model on medical X-ray images."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    timestamp = int(time())

    parser = argparse.ArgumentParser(description='Train SteganoGAN on medical X-ray images')
    parser.add_argument('--epochs', default=4, type=int, help='Number of epochs to train')
    parser.add_argument('--encoder', default="dense", type=str, choices=['basic', 'dense'],
                        help='Encoder architecture to use')
    parser.add_argument('--data_depth', default=1, type=int, help='Data depth for message embedding')
    parser.add_argument('--hidden_size', default=32, type=int, help='Hidden size for the models')
    parser.add_argument('--dataset', default="xrays", type=str, help='Dataset folder name')
    parser.add_argument('--output', default=False, type=str, help='Output path for the model')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA for training')
    parser.add_argument('--verbose', action='store_true', help='Display verbose output')
    args = parser.parse_args()

    # Set up data loaders
    train_path = os.path.join("steganogan", "data", args.dataset, "train")
    val_path = os.path.join("steganogan", "data", args.dataset, "val")
    
    print(f"Loading data from {train_path} and {val_path}")
    train = MedicalDataLoader(train_path, shuffle=True, batch_size=args.batch_size)
    validation = MedicalDataLoader(val_path, shuffle=False, batch_size=args.batch_size)

    # Select encoder architecture
    encoder = BasicEncoder if args.encoder == 'basic' else DenseEncoder
    
    # Initialize SteganoGAN model
    log_dir = os.path.join('models', str(timestamp))
    os.makedirs(log_dir, exist_ok=True)
    
    steganogan = SteganoGAN(
        data_depth=args.data_depth,
        encoder=encoder,
        decoder=DenseDecoder,
        critic=BasicCritic,
        hidden_size=args.hidden_size,
        cuda=args.cuda,
        verbose=args.verbose,
        log_dir=log_dir
    )
    
    # Save configuration
    with open(os.path.join(log_dir, "config.json"), "wt") as fout:
        config = args.__dict__.copy()
        for key, value in config.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config[key] = str(value)
        fout.write(json.dumps(config, indent=2))

    # Train the model
    print(f"Training SteganoGAN with {args.encoder} encoder for {args.epochs} epochs")
    steganogan.fit(train, validation, epochs=args.epochs)
    
    # Save the final model
    os.makedirs('pretrained', exist_ok=True)
    final_path = os.path.join(log_dir, "medical_xray.steg")
    steganogan.save(final_path)
    print(f"Model saved to {final_path}")
    
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        steganogan.save(args.output)
        print(f"Model also saved to {args.output}")
    
    # Print final metrics
    if steganogan.fit_metrics:
        print("\nFinal Metrics:")
        print(f"PSNR: {steganogan.fit_metrics.get('val.psnr', 'N/A'):.2f} dB")
        print(f"SSIM: {steganogan.fit_metrics.get('val.ssim', 'N/A'):.4f}")
        print(f"Bits per pixel: {steganogan.fit_metrics.get('val.bpp', 'N/A'):.4f}")
        print(f"Decoder accuracy: {steganogan.fit_metrics.get('val.decoder_acc', 'N/A'):.4f}")


if __name__ == '__main__':
    main()