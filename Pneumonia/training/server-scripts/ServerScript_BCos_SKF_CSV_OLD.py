import cv2
import csv
import os
import pickle
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as Variable
import torchvision.transforms.functional as TF
from libraries.data_transforms import AddInverse
from libraries.bcosconv2d import NormedConv2d
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np



csv_path = r"/home/mkleinma/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
data = pd.read_csv(csv_path)
image_folder = r"/home/mkleinma/rsna-pneumonia-detection-challenge/stage_2_train_images"
splits_path = r"/home/mkleinma/training_splits/splits.pkl"

with open(splits_path, 'rb') as f:
    splits = pickle.load(f)



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
        tensor_image = TF.to_tensor(image)
        six_channel_image = torch.cat([tensor_image, 1-tensor_image], dim=0)
                
        return six_channel_image, torch.tensor(label, dtype=torch.float32)


# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

fold = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# list all available models
torch.hub.list('B-cos/B-cos-v2')

# load a pretrained model
model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
model.to("cuda")


# Make sure no layer is frozen
for param in model.parameters():
    param.requires_grad = False
    
num_features = model.layer4[-1].conv3.linear.out_channels

model.fc.linear = NormedConv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1), bias=False) # code from B-cos paper reused to adjust network
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()  # For binary classification
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)   #Only train the final layer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


# Training loop
with open('results_bcos.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write header only once
    writer.writerow(['Fold', 'Epoch', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])

    for train_idx, val_idx in splits:
        fold += 1
        print(f"Training fold {fold}...")

        # Split data for the current fold
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
            
        train_dataset = PneumoniaDataset(train_data, image_folder, transform=transform)
        val_dataset = PneumoniaDataset(val_data, image_folder, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

        # Training and validation loop for each fold
        num_epochs = 30
        epochCount = 0
        for epoch in range(num_epochs):
            epochCount += 1
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
                
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate training loss and accuracy
                running_loss += loss.item() * images.size(0)
                preds = torch.round(torch.sigmoid(outputs))
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            train_loss = running_loss / len(train_loader.dataset)
            train_accuracy = correct / total

            # Evaluation - get metrics to understand model performance
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
                
            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).unsqueeze(1)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                        
                    val_loss += loss.item() * images.size(0)
                    preds = torch.round(torch.sigmoid(outputs))
                        
                    all_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())  # Probabilities for AUC
                    all_preds.extend(preds.cpu().numpy().flatten())  # Binary predictions for precision/recall/F1
                    all_labels.extend(labels.cpu().numpy().flatten())  # True labels
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                val_loss /= len(val_loader.dataset)
                val_accuracy = val_correct / val_total
                scheduler.step(val_loss)
                
                precision = precision_score(all_labels, all_preds)
                recall = recall_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds)
                auc = roc_auc_score(all_labels, all_probs)
                
                
                writer.writerow([str(fold), str(epochCount), str(val_accuracy), str(precision), str(recall), str(f1), str(auc)])
                print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, "
                    f"Val Acc: {val_accuracy:.4f}, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
                
            print(f"Finished training fold {fold}.\n")
        
        


model_path = "pneumonia_detection_model_bcos_csv.pth"
torch.save(model.state_dict(), model_path)

# %%



