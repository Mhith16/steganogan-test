#!/usr/bin/env python3
"""
SteganoGAN Training Script for Medical X-ray Images

This script trains a SteganoGAN model for hiding patient data in medical X-ray images.
"""

import argparse
import json
import os
from time import time
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from steganogan import SteganoGAN
from steganogan.data import MedicalDataLoader
from steganogan.models.critics import BasicCritic
from steganogan.models.decoders import DenseDecoder
from steganogan.models.encoders import BasicEncoder, DenseEncoder, ResidualEncoder


def main():
    """Train a SteganoGAN model on medical X-ray images."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    timestamp = int(time())

    parser = argparse.ArgumentParser(description='Train SteganoGAN on medical X-ray images')
    parser.add_argument('--epochs', default=4, type=int, help='Number of epochs to train')
    parser.add_argument('--encoder', default="dense", type=str, choices=['basic', 'residual', 'dense'],
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
    train_path = os.path.join("data", args.dataset, "train")
    val_path = os.path.join("data", args.dataset, "val")
    
    # Check if the paths exist
    if not os.path.exists(train_path):
        print(f"Error: Training path {train_path} does not exist.")
        print("Please organize your X-ray images and text files in the following structure:")
        print(f"data/{args.dataset}/train/  # Training images and text files")
        print(f"data/{args.dataset}/val/    # Validation images and text files")
        return
    
    if not os.path.exists(val_path):
        print(f"Warning: Validation path {val_path} does not exist. Creating a small validation set from training data.")
        os.makedirs(val_path, exist_ok=True)
        
        # Copy a few images from training to validation as a fallback
        import shutil
        import random
        from glob import glob
        
        train_images = glob(os.path.join(train_path, "*.jpg"))
        val_samples = random.sample(train_images, min(10, len(train_images)))
        
        for img_path in val_samples:
            base_name = os.path.basename(img_path)
            txt_name = os.path.splitext(base_name)[0] + ".txt"
            txt_path = os.path.join(train_path, txt_name)
            
            # Copy image and text file if it exists
            shutil.copy(img_path, os.path.join(val_path, base_name))
            if os.path.exists(txt_path):
                shutil.copy(txt_path, os.path.join(val_path, txt_name))

    print(f"Loading data from {train_path} and {val_path}")
    train = MedicalDataLoader(train_path, shuffle=True, batch_size=args.batch_size)
    validation = MedicalDataLoader(val_path, shuffle=False, batch_size=args.batch_size)

    # Select encoder architecture
    encoder = {
        "basic": BasicEncoder,
        "residual": ResidualEncoder,
        "dense": DenseEncoder,
    }[args.encoder]
    
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
        # Convert any non-serializable objects to strings
        for key, value in config.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config[key] = str(value)
        fout.write(json.dumps(config, indent=2))

    # Train the model
    print(f"Training SteganoGAN with {args.encoder} encoder for {args.epochs} epochs")
    steganogan.fit(train, validation, epochs=args.epochs)
    
    # Save the final model
    final_path = os.path.join(log_dir, "medical_xray.steg")
    steganogan.save(final_path)
    print(f"Model saved to {final_path}")
    
    if args.output:
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