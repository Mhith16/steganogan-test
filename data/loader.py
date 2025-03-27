# -*- coding: utf-8 -*-

import os
from glob import glob

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_DEFAULT_MU = [.5, .5, .5]
_DEFAULT_SIGMA = [.5, .5, .5]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(360, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
])

MEDICAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),  # Standardize X-ray sizes
    transforms.ToTensor(),
    transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
])


class ImageFolder(torchvision.datasets.ImageFolder):
    """Extension of torchvision's ImageFolder with a limit on the number of images.
    
    Args:
        path (str): Path to the image folder
        transform (torchvision.transforms.Transform): Transforms to apply to the images
        limit (int or float): Maximum number of images to load
    """
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)


class PatientDataset(Dataset):
    """Dataset for X-ray images paired with patient data text files.
    
    Args:
        root_dir (str): Path to the dataset folder
        transform (torchvision.transforms.Transform): Transforms to apply to the images
        limit (int or float): Maximum number of images to load
    """
    def __init__(self, root_dir, transform=None, limit=np.inf):
        """Initialize the dataset."""
        self.root_dir = root_dir
        self.transform = transform or MEDICAL_TRANSFORM
        
        # Find all image files and their paired text files
        self.image_files = sorted(glob(os.path.join(root_dir, "*.jpg")))
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
                # If text file doesn't exist, use None
                self.text_files.append(None)
                
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_files)
        
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, text) where image is the transformed X-ray and text is the patient data
        """
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


class DataLoader(torch.utils.data.DataLoader):
    """Data loader for standard image folders.
    
    Args:
        path (str): Path to the image folder
        transform (torchvision.transforms.Transform): Transforms to apply to the images
        limit (int or float): Maximum number of images to load
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for data loading
        batch_size (int): Batch size
    """
    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=4, batch_size=4, *args, **kwargs):
        """Initialize the data loader."""
        if transform is None:
            transform = DEFAULT_TRANSFORM

        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )


class MedicalDataLoader(torch.utils.data.DataLoader):
    """Data loader for medical X-ray images paired with patient data.
    
    Args:
        path (str): Path to the dataset folder
        transform (torchvision.transforms.Transform): Transforms to apply to the images
        limit (int or float): Maximum number of images to load
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for data loading
        batch_size (int): Batch size
    """
    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=4, batch_size=4, *args, **kwargs):
        """Initialize the data loader."""
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