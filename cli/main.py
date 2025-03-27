# -*- coding: utf-8 -*-

import argparse
import os
import warnings

import torch
from torch.serialization import SourceChangeWarning

from steganogan.models import SteganoGAN

warnings.filterwarnings('ignore', category=SourceChangeWarning)


def _get_steganogan(args):
    """Get a SteganoGAN instance based on the command line arguments.
    
    Args:
        args (argparse.Namespace): Command line arguments
        
    Returns:
        SteganoGAN: SteganoGAN instance
    """
    steganogan_kwargs = {
        'cuda': not args.cpu,
        'verbose': args.verbose
    }

    if args.path:
        steganogan_kwargs['path'] = args.path
    else:
        steganogan_kwargs['architecture'] = args.architecture

    return SteganoGAN.load(**steganogan_kwargs)


def _encode(args):
    """Encode a message in an image.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    steganogan = _get_steganogan(args)
    steganogan.encode(args.cover, args.output, args.message)


def _decode(args):
    """Decode a message from an image.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    try:
        steganogan = _get_steganogan(args)
        message = steganogan.decode(args.image)

        if args.verbose:
            print('Message successfully decoded:')

        print(message)

    except Exception as e:
        print('ERROR: {}'.format(e))
        if args.verbose:
            import traceback
            traceback.print_exc()


def _metrics(args):
    """Calculate image quality metrics for a steganographic image.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    try:
        from PIL import Image
        import numpy as np
        import torch
        from steganogan.utils import psnr, ssim
        
        # Load original and steganographic images
        original = Image.open(args.original).convert('RGB')
        stego = Image.open(args.stego).convert('RGB')
        
        # Convert to tensors
        original = torch.tensor(np.array(original)).permute(2, 0, 1).float() / 127.5 - 1.0
        stego = torch.tensor(np.array(stego)).permute(2, 0, 1).float() / 127.5 - 1.0
        
        # Add batch dimension
        original = original.unsqueeze(0)
        stego = stego.unsqueeze(0)
        
        # Calculate metrics
        psnr_value = psnr(original, stego).item()
        ssim_value = ssim(original, stego).item()
        
        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")
        
    except Exception as e:
        print('ERROR: {}'.format(e))
        if args.verbose:
            import traceback
            traceback.print_exc()


def _get_parser():
    """Get the command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Argument parser
    """
    # Parent Parser - Shared options
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    group = parent.add_mutually_exclusive_group()
    group.add_argument('-a', '--architecture', default='dense',
                       choices={'basic', 'dense', 'residual'},
                       help='Model architecture. Use the same one for both encoding and decoding')

    group.add_argument('-p', '--path', help='Load a pretrained model from a given path.')
    parent.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')

    parser = argparse.ArgumentParser(description='SteganoGAN Command Line Interface')

    subparsers = parser.add_subparsers(title='action', help='Action to perform')
    parser.set_defaults(action=None)

    # Encode Parser
    encode = subparsers.add_parser('encode', parents=[parent],
                                   help='Hide a message into a steganographic image')
    encode.set_defaults(action=_encode)
    encode.add_argument('-o', '--output', default='output.png',
                        help='Path and name to save the output image')
    encode.add_argument('cover', help='Path to the image to use as cover')
    encode.add_argument('message', help='Message to encode')

    # Decode Parser
    decode = subparsers.add_parser('decode', parents=[parent],
                                   help='Read a message from a steganographic image')
    decode.set_defaults(action=_decode)
    decode.add_argument('image', help='Path to the image with the hidden message')
    
    # Metrics Parser
    metrics = subparsers.add_parser('metrics', parents=[parent],
                                    help='Calculate image quality metrics')
    metrics.set_defaults(action=_metrics)
    metrics.add_argument('original', help='Path to the original (cover) image')
    metrics.add_argument('stego', help='Path to the steganographic image')

    return parser


def main():
    """Main entry point for the command line interface."""
    parser = _get_parser()
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        parser.exit()

    args.action(args)


if __name__ == '__main__':
    main()