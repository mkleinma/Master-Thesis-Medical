import torch
import torchvision.transforms as transforms
import random
from PIL import Image

def no_augmentations():
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])


# Define Light Augmentations
def get_light_augmentations():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=2.5),  # Random affine
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),       # Random perspective
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    return transform

def heavy_augmentations_no_rotation():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, shear=4.0, scale=(0.85, 1.15)),  # No rotation
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.05 if random.random() < 0.1 else img) # low chance for gausian noise
    ])

