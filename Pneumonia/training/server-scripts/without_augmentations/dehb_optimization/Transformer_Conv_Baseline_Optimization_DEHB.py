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


# Set random seeds for reproducibility
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# Paths
csv_path = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/training_splits/grouped_data.csv"
image_folder = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/rsna-pneumonia-detection-challenge/stage_2_train_images"
splits_path = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/training_splits/splits_balanced_fix.pkl"

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
        
        return image, torch.tensor(label, dtype=torch.long)

fold = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from dehb import DEHB
import ConfigSpace as CS
import numpy as np
import logging

logging.basicConfig(filename="train_debug_dehb_trans_base.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

cs = CS.ConfigurationSpace()
cs.add([CS.CategoricalHyperparameter("lr", [1e-6, 1e-5, 1e-4]),
        CS.CategoricalHyperparameter("weight_decay", [0, 1e-6, 1e-5, 1e-4, 1e-3]),
        CS.CategoricalHyperparameter("patience", [3, 5, 10])])


def train_model(config: Union[ConfigSpace.Configuration, List, np.array], fidelity: Union[int, float] = None, **kwargs) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.hub.load('B-cos/B-cos-v2', 'standard_vitc_b_patch1_14', pretrained=True)        
    model.linear_head.linear = torch.nn.Linear(in_features=768, out_features=2, bias=True)    
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config['patience'], verbose=True)
    print(f"Running train_model with config: {config}, fidelity: {fidelity}", flush=True)

    
    best_f1 = 0.0
    val_loss = 0.0
    val_accuracy = 0.0

    train_idx, val_idx = splits[0]
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
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    for epoch in range(int(fidelity)):
        print(f"Starting epoch {epoch+1}", flush=True)
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            if torch.isnan(outputs).any():
                print("Warning: NaN detected in model outputs during training!", flush=True)
                return {"fitness": -1, "cost": fidelity, "info": {"status": "NaN detected"}}
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
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
                if torch.isnan(outputs).any():
                    print("Warning: NaN detected in model outputs during validation!", flush=True)
                    return {"fitness": -1, "cost": fidelity, "info": {"status": "NaN detected"}}

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
        
        if best_f1 < f1:
            best_f1 = f1

    print(f"Final best f1: {best_f1} before potentially returning default values", flush=True)

    return {
        "fitness": 1 - best_f1,  # DEHB minimizes this value
        "cost": fidelity,
        "info": {"status": "Successful Execution"}
    }

de = DEHB(f=train_model,
    dimensions=3,
    cs=cs,
    min_fidelity=1,
    max_fidelity=10, # number of epochs to run it for
    output_path="/pfs/work7/workspace/scratch/ma_mkleinma-thesis/dehb_results/trans_base/",
    n_workers=1)
incumbent = de.run(fevals=10, runtime=160000)

print(incumbent)


