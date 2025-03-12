import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
from PIL import Image

import numpy as np
import imgaug.augmenters as iaa

'''Inspired by augmentations in 'Deep Learning for Automatic Pneumonia Detection' paper
tried to convert the augmentations which consist of an uncommon pipeline for pytorch transformations
due to that some implementations look weird (e.g. the continuous call of lambda's to adjust the images in the same order with different operations)

'''
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



def get_no_augmentations_no_resize():
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    
def get_no_augmentations_resize():
    return transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    
# Define Light Augmentations
def get_light_augmentations_resize():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,  # No rotation yet
            translate=(random.uniform(-32, 32), random.uniform(-32, 32)),  # Apply first translation
            scale=1.0,  # No scaling yet
            shear=0  # No shear yet
        )),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,  # No rotation yet
            translate=(0, 0),
            scale=1.0 / (2 ** random.gauss(0, 0.1)),  # Apply correct inverse scaling
            shear=0  # No shear yet
        )),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=random.gauss(0, 5),  # Apply rotation
            translate=(0, 0),
            scale=1.0,  # No additional scaling
            shear=random.gauss(0, 2.5)  # Apply shear after rotation
        )),

        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),  # Perspective distortion
        transforms.Lambda(lambda img: TF.adjust_gamma(img, 2.0 ** random.gauss(0, 0.20))), # as in implementation from them

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform

# 3 transforms based on order of source (detection_dataset.py)
def get_light_augmentations_no_resize():
    transform = transforms.Compose([
        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,  # No rotation yet
            translate=(random.uniform(-32, 32), random.uniform(-32, 32)),  # Apply first translation
            scale=1.0,  # No scaling yet
            shear=0  # No shear yet
        )),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,  # No rotation yet
            translate=(0, 0),
            scale=1.0 / (2 ** random.gauss(0, 0.1)),  # Apply correct inverse scaling
            shear=0  # No shear yet
        )),

        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=random.gauss(0, 5),  # Apply rotation
            translate=(0, 0),
            scale=1.0,  # No additional scaling
            shear=random.gauss(0, 2.5)  # Apply shear after rotation
        )),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),  # Perspective distortion
        transforms.Lambda(lambda img: TF.adjust_gamma(img, 2.0 ** random.gauss(0, 0.20))), # as in implementation from them

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform



def get_heavy_augmentations_no_rotation_no_resize():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,  # No rotation yet
            translate=(random.uniform(-32, 32), random.uniform(-32, 32)),  # Apply first translation
            scale=1.0,  # No scaling yet
            shear=0  # No shear yet
        )),
        
        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,  # Rotation: normal distribution (mean=0, std=6)
            translate=(0,0),
            scale=1.0 / (2 ** random.gauss(0, 0.15)),  # Log-normal scaling
            shear=0
        )),

        transforms.Lambda(lambda img: transforms.functional.affine(
            img,
            angle=random.gauss(0, 6),  # Rotation: normal distribution (mean=0, std=6)
            translate=(0,0),
            scale=1.0,  # Log-normal scaling
            shear=random.gauss(0, 4)  # Shear: normal distribution (mean=0, std=4)
        )),
        
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.Lambda(lambda img: TF.adjust_gamma(img, 2.0 ** random.gauss(0, 0.25))), # as in implementation from them
        HeavyImageAugmentationSupport(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    
def get_heavy_augmentations_no_rotation_resize():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,  # No rotation yet
            translate=(random.uniform(-32, 32), random.uniform(-32, 32)),  # Apply first translation
            scale=1.0,  # No scaling yet
            shear=0  # No shear yet
        )),
        
        transforms.Lambda(lambda img: TF.affine(
            img,
            angle=0,  # Rotation: normal distribution (mean=0, std=6)
            translate=(0,0),
            scale=1.0 / (2 ** random.gauss(0, 0.15)),  # Log-normal scaling
            shear=0
        )),

        transforms.Lambda(lambda img: transforms.functional.affine(
            img,
            angle=random.gauss(0, 6),  # Rotation: normal distribution (mean=0, std=6)
            translate=(0,0),
            scale=1.0,  # Log-normal scaling
            shear=random.gauss(0, 4)  # Shear: normal distribution (mean=0, std=4)
        )),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.Lambda(lambda img: TF.adjust_gamma(img, 2.0 ** random.gauss(0, 0.25))),
        HeavyImageAugmentationSupport(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

