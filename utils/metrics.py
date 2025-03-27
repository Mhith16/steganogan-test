# -*- coding: utf-8 -*-

import torch
from torch.nn.functional import conv2d

def gaussian(window_size, sigma):
    """Gaussian window for SSIM calculation.
    
    Args:
        window_size (int): Size of the window
        sigma (float): Standard deviation
        
    Returns:
        torch.Tensor: Normalized Gaussian window
    """
    _exp = [torch.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.tensor(_exp)
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """Create a 2D window for SSIM calculation.
    
    Args:
        window_size (int): Size of the window
        channel (int): Number of channels
        
    Returns:
        torch.Tensor: 2D window
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Calculate SSIM between two images.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        window (torch.Tensor): Window for SSIM calculation
        window_size (int): Size of the window
        channel (int): Number of channels
        size_average (bool, optional): Whether to average the result. Defaults to True.
        
    Returns:
        torch.Tensor: SSIM value
    """
    padding_size = window_size // 2

    mu1 = conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = conv2d(img2, window, padding=padding_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    _ssim_quotient = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    _ssim_divident = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = _ssim_quotient / _ssim_divident

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    """Calculate SSIM between two images.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        window_size (int, optional): Size of the window. Defaults to 11.
        size_average (bool, optional): Whether to average the result. Defaults to True.
        
    Returns:
        torch.Tensor: SSIM value
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def psnr(img1, img2, max_val=2.0):
    """Calculate PSNR between two images.
    
    For SteganoGAN, images are typically normalized to [-1, 1], making the range 2.0.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        max_val (float, optional): Maximum value of the images. Defaults to 2.0.
        
    Returns:
        torch.Tensor: PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    
    return 10 * torch.log10((max_val ** 2) / mse)