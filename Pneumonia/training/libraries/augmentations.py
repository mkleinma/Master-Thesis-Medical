import torch
import torchvision.transforms as transforms
import random
from PIL import Image

import numpy as np
import imgaug.augmenters as iaa

# used in GitHub Repository from 'Deep Learning for Automatic Pneumonia Detection'
def get_iaa_augmentations(): 
    return iaa.Sequential([
    iaa.Sometimes(0.1, iaa.CoarseSaltAndPepper(p=(0.01, 0.01), size_percent=(0.1, 0.2))),  # Salt & Pepper Noise
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.0))),  # Gaussian Blur
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255)))  # Additive Gaussian Noise
])

# Custom transform class to apply imgaug to add it into augmentations - imgaug allows easier transform
class HeavyImageAugmentationSupport:
    def __init__(self):
        self.iaa_augmentations = get_iaa_augmentations()

    def __call__(self, img):
        img = np.array(img)
        img = self.iaa_augmentations(image=img)  
        return Image.fromarray(img) 


# Define Light Augmentations
def get_light_augmentations_resize():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: transforms.functional.affine(
            img,
            angle=random.gauss(0, 6),  # Rotation: normal distribution (mean=0, std=6)
            translate=(0, 0),
            scale=2 ** random.gauss(0, 0.15),  # Log-normal scaling
            shear=random.gauss(0, 4)  # Shear: normal distribution (mean=0, std=4)
        )),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),  # Perspective distortion
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform

'''def get_light_augmentations_resize():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=2.5),  # Random affine
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),       # Random perspective
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    return transform'''

''' 
def get_light_augmentations_no_resize():
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=2.5),  # Random affine
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),       # Random perspective
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    return transform'''

def get_light_augmentations_no_resize():
    transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.affine(
            img,
            angle=random.gauss(0, 6),  # Rotation: normal distribution (mean=0, std=6)
            translate=(0, 0),
            scale=2 ** random.gauss(0, 0.15),  # Log-normal scaling
            shear=random.gauss(0, 4)  # Shear: normal distribution (mean=0, std=4)
        )),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),  # Perspective distortion
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform



def get_heavy_augmentations_no_rotation_no_resize():
    return transforms.Compose([
        HeavyImageAugmentationSupport(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda img: transforms.functional.affine(
            img,
            angle=random.gauss(0, 6),  # Rotation: normal distribution (mean=0, std=6)
            translate=(0, 0),
            scale=2 ** random.gauss(0, 0.15),  # Log-normal scaling
            shear=random.gauss(0, 4)  # Shear: normal distribution (mean=0, std=4)
            )),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Lambda(lambda img: transforms.functional.adjust_gamma(img, 2.0 ** random.gauss(0, 0.25))), # as in implementation from them
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    
def get_heavy_augmentations_no_rotation_resize():
    return transforms.Compose([
        HeavyImageAugmentationSupport(),
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda img: transforms.functional.affine(
            img,
            angle=random.gauss(0, 6),  # Rotation: normal distribution (mean=0, std=6)
            translate=(0, 0),
            scale=2 ** random.gauss(0, 0.15),  # Log-normal scaling
            shear=random.gauss(0, 4)  # Shear: normal distribution (mean=0, std=4)
            )),
        #transforms.RandomAffine(degrees=0, shear=4.0, scale=(0.85, 1.15)),  # No rotation
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Lambda(lambda img: transforms.functional.adjust_gamma(img, 2.0 ** random.gauss(0, 0.25))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

