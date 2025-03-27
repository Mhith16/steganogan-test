#!/usr/bin/env python3
import argparse
import json
import os
import sys
from time import time

import torch

# Import the modules directly
from models.steganogan import SteganoGAN
from models.critics import BasicCritic
from models.decoders import DenseDecoder
from models.encoders import BasicEncoder, DenseEncoder, ResidualEncoder
from data.loader import MedicalDataLoader

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

if __name__ == '__main__':
    main()