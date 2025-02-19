import json
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import random
import pickle 
import os
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image
import numpy as np
import csv
import warnings
import numpy as np
import ConfigSpace
from typing import Dict, Union, List

warnings.filterwarnings('ignore')

from libraries.bcosconv2d import NormedConv2d

## assisting script
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    return fig


def save_checkpoint(model, optimizer, scheduler, epoch, fold, path, best_f1):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'fold': fold,
        'best_f1': best_f1

    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")
    
    
def load_checkpoint(path, model, optimizer, scheduler):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        fold = checkpoint['fold']  # Load the last completed fold
        best_f1 = checkpoint['best_f1']
        print(f"Checkpoint loaded from {path}")
        return start_epoch, fold, best_f1, True
    return 0, 0, 0.0, False

def find_latest_checkpoint(model_output_dir):
    """
    Find the latest available checkpoint in the directory.
    Looks for checkpoints named in the format 'checkpoint_fold_X.pth' where X is the fold number.
    """
    for fold in range(5, 0, -1):  # Check from fold_5 to fold_1
        checkpoint_path = os.path.join(model_output_dir, f"checkpoint_fold_{fold}.pth")
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint: {checkpoint_path}")
            return checkpoint_path, fold
    print("No checkpoint found. Starting training from scratch.")
    return None, None


# Set random seeds for reproducibility
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


# Paths
csv_path = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
image_folder = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/rsna-pneumonia-detection-challenge/stage_2_train_images"
splits_path = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/training_splits/splits_balanced.pkl"
cm_output_dir = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/trained_models/30_epochs_bcos_resnet50_224/seed_0/confusion_matrix"
model_output_dir = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/trained_models/30_epochs_bcos_resnet50_224/seed_0/"

# Load data and splits
data = pd.read_csv(csv_path)
with open(splits_path, 'rb') as f:
    splits = pickle.load(f)


# Dataset class for Pneumonia
class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.data = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_folder, f"{row['patientId']}.dcm")
        label = row['Target']

        # Load DICOM file and process it into RGB format
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array
        image = Image.fromarray(image).convert("RGB")
        
        if self.transform: # does not work atm but its okay
            image = self.transform(image)

        tensor_image = TF.to_tensor(image)
        
        return tensor_image, torch.tensor(label, dtype=torch.long)


# Define transformations for the datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

fold = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from dehb import DEHB
import ConfigSpace as CS
import numpy as np

cs = CS.ConfigurationSpace()
cs.add([CS.UniformFloatHyperparameter("lr", lower=1e-6, upper=1e-3, default_value=1e-4, log=True), 
        CS.UniformIntegerHyperparameter("batch_size", lower=8, upper=32, default_value=16, log=True),
        CS.UniformFloatHyperparameter("weight_decay", lower=0, upper=1e-4, default_value=0, log=True),
        CS.UniformIntegerHyperparameter("patience", lower=3, upper=10, default_value=5, log=True)])



def train_model(config: Union[ConfigSpace.Configuration, List, np.array], fidelity: Union[int, float] = None, **kwargs) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
    model.fc.linear = NormedConv2d(2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=False) # code from B-cos paper reused to adjust network
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config['patience'], verbose=True)
    
    
    total_f1 = 0
    total_val_loss = 0
    total_val_accuracy = 0

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_dataset = PneumoniaDataset(data.iloc[train_idx], image_folder, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
        val_dataset = PneumoniaDataset(data.iloc[val_idx], image_folder, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
        
        train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']), shuffle=False)
        
        for epoch in range(int(fidelity)):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().flatten())  
                    all_preds.extend(preds.cpu().numpy().flatten())  
                    all_labels.extend(labels.cpu().numpy().flatten())

                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            f1 = f1_score(all_labels, all_preds)
            
            total_f1 += f1
            total_val_loss += val_loss
            total_val_accuracy += val_correct / val_total

            
    avg_f1 = total_f1 / len(splits)
    avg_val_loss = total_val_loss / len(splits)
    avg_val_accuracy = total_val_accuracy / len(splits)
        
    return {
        "fitness": -avg_f1,  # DEHB minimizes this value
        "cost": fidelity * len(splits),  # Total number of epochs across all folds
        "info": {"val_accuracy": avg_val_accuracy}
    }



de = DEHB(f=train_model,
    dimensions=4,
    cs=cs,
    min_fidelity=1,
    max_fidelity=7, # number of epochs to run it for
    output_path="/pfs/work7/workspace/scratch/ma_mkleinma-thesis/dehb_results",
    n_workers=1)
incumbent = de.run(fevals=10)

print(incumbent)


