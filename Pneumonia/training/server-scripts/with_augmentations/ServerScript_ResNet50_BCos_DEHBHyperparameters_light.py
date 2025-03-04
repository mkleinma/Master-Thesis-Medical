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

from libraries.bcosconv2d import NormedConv2d
from libraries import augmentations


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
cm_output_dir = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/trained_models/30_epochs_bcos_resnet50_dehb_l/seed_0/confusion_matrix"
model_output_dir = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/trained_models/30_epochs_bcos_resnet50_dehb_l/seed_0/"

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
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# Define transformations for the datasets
transform = augmentations.get_light_augmentations_no_resize()

fold = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
with open('results_resnet_bcos_allLayers_noAug.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Fold', 'Epoch', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
    
        
    model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
    model.fc.linear = NormedConv2d(2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=False) # code from B-cos paper reused to adjust network
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    start_epoch, start_fold, best_f1, checkpoint_exists = 0, 0, 0.0, False
    
    # we check for latest checkpoint and if it exists, then we load the checkpoint and start from there - else we start from 0 and it does not exist
    latest_checkpoint_path, latest_fold = find_latest_checkpoint(model_output_dir)
    if latest_checkpoint_path:
        start_epoch, start_fold, best_f1, checkpoint_exists = load_checkpoint(
            latest_checkpoint_path, model, optimizer, scheduler)
    
    for current_fold, (train_idx, val_idx) in enumerate(splits):
        fold = current_fold + 1
        print(f"Training fold {fold}...")
        
        if checkpoint_exists and fold < start_fold:
            print(f"Skipping fold {current_fold} as it's already completed.")
            continue  # Skip completed folds
                
        # if we dont start from checkpoint: initialize new model to train
        if not checkpoint_exists or current_fold != start_fold:
            model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
            model.fc.linear = NormedConv2d(2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=False) # code from B-cos paper reused to adjust network
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-05)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
            model = model.to(device)
            checkpoint_exists = False

    
        log_dir = os.path.join(model_output_dir, f"tensorboard_logs_fold_{fold}")
        log_writer = SummaryWriter(log_dir=log_dir)

        # Prepare datasets and dataloaders
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        train_dataset = PneumoniaDataset(train_data, image_folder, transform=transform)
        val_dataset = PneumoniaDataset(val_data, image_folder, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
                
        # Train for the current fold
        num_epochs = 30
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            if (epoch < start_epoch and start_epoch < num_epochs):
                print(f"Skipping epoch {epoch}, resuming from checkpoint at epoch {start_epoch}.")
                continue

            if fold == start_fold:
                checkpoint_exists = False 
                start_epoch = 0

            for images, labels in train_loader:
                labels = labels.to(device)
                
                six_channel_images = []
                # create model.transform images
                for img_tensor in images:
                  numpy_image = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                  pil_image = Image.fromarray(numpy_image)
                  transformed_image = model.transform(pil_image) # to 224x224 resolution
                  six_channel_images.append(transformed_image)
                
                six_channel_images = torch.stack(six_channel_images).to(device)
                
                # Forward pass
                outputs = model(six_channel_images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            train_accuracy = correct / total
            
            log_writer.add_scalar('Loss/Train', train_loss, epoch)
            log_writer.add_scalar('Accuracy/Train', train_accuracy, epoch)


            print(f"Training Accuracy: {train_accuracy:.4f}")

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for images, labels in val_loader:
                    labels = labels.to(device)
                    six_channel_images = []
                    for img_tensor in images:
                      numpy_image = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                      pil_image = Image.fromarray(numpy_image)
                      transformed_image = model.transform(pil_image)
                      six_channel_images.append(transformed_image)
                      
                    six_channel_images = torch.stack(six_channel_images).to(device)
                    outputs = model(six_channel_images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    preds = torch.argmax(outputs, dim=1)

                    all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().flatten())
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
                    
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(val_loader.dataset)
            val_accuracy = val_correct / val_total
            scheduler.step(val_loss)
            
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_probs)
            
            log_writer.add_scalar('Loss/Validation', val_loss, epoch)
            log_writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
            log_writer.add_scalar('Metrics/Precision', precision, epoch)
            log_writer.add_scalar('Metrics/Recall', recall, epoch)
            log_writer.add_scalar('Metrics/F1', f1, epoch)
            log_writer.add_scalar('Metrics/AUC', auc, epoch)
            
            current_lr = optimizer.param_groups[0]['lr']
            log_writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            cm = confusion_matrix(all_labels, all_preds)
            class_names = ['No Pneumonia', 'Pneumonia']
            cm_figure = plot_confusion_matrix(cm, class_names)
            log_writer.add_figure('Confusion_Matrix', cm_figure, epoch)

            
            if (f1 > best_f1):
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(model_output_dir, f"pneumonia_detection_model_resnet_bcos_bestf1_{fold}_{epoch}.pth"))
                cm_file_path = os.path.join(cm_output_dir, f"confusion_matrix_best_f1_{fold}_{epoch}.json")
                with open(cm_file_path, 'w') as cm_file:
                    json.dump({'confusion_matrix': cm.tolist()}, cm_file, indent=4)
                print(f"Confusion Matrix for Fold {fold} saved at {cm_file_path}")

            if epoch == num_epochs - 1:
                cm = confusion_matrix(all_labels, all_preds)
                cm_file_path = os.path.join(cm_output_dir, f"confusion_matrix_fold_{fold}.json")
                with open(cm_file_path, 'w') as cm_file:
                    json.dump({'confusion_matrix': cm.tolist()}, cm_file, indent=4)
                print(f"Confusion Matrix for Fold {fold} saved at {cm_file_path}")
                
            save_checkpoint_path = os.path.join(model_output_dir, f"checkpoint_fold_{fold}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, fold, save_checkpoint_path, best_f1)

            writer.writerow([fold, epoch + 1, val_accuracy, precision, recall, f1, auc])
            print(f"Fold {fold}, Epoch {epoch + 1}/{num_epochs}, "
                  f"Val Acc: {val_accuracy:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        print(f"Finished training fold {fold}.\n")
        
        # Save the final model
        model_path = f"pneumonia_detection_model_fold_{fold}_resnet_bcos.pth"
        torch.save(model.state_dict(), os.path.join(model_output_dir, model_path))
        log_writer.close()



