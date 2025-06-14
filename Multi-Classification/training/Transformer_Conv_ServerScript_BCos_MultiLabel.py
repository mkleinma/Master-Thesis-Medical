import argparse
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
from torch.utils.data import WeightedRandomSampler

from libraries_multilabel.bcosconv2d import NormedConv2d
from libraries_multilabel.bcoslinear import BcosLinear
from libraries_multilabel import augmentations
from libraries_multilabel.MultiLabelDatasets import MultiLabelDataset


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Random seed for training")
parser.add_argument("--augmentation", type=str, choices=["no", "light", "heavy"], required=True, help="Augmentation Type")
parser.add_argument("--sampling", type=lambda x: x.lower() == "true", default=False, help="Enable sampling (True/False)")


args = parser.parse_args()

## assisting script
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    return fig


def save_checkpoint(model, optimizer, scheduler, epoch, fold, path, best_f1, best_recall):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'fold': fold,
        'best_f1': best_f1,
        'best_recall' : best_recall
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
        best_recall = checkpoint['best_recall']
        print(f"Checkpoint loaded from {path}")
        return start_epoch, fold, best_f1, best_recall, True
    return 0, 0, 0.0, 0.0, False

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
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
samp_text = "nosamp"
if args.sampling:
    samp_text = "oversamp"
    
# Paths
csv_path = r"/pfs/work9/workspace/scratch/ma_mkleinma-thesis/training_splits/multilabel_dataset.csv"
image_folder = r"/pfs/work9/workspace/scratch/ma_mkleinma-thesis/vinbigdata-chest-xray-abnormalities-detection/train_png_224"
splits_path = r"/pfs/work9/workspace/scratch/ma_mkleinma-thesis/training_splits/vinbigdata_5fold_splits.pkl"
model_output_dir = f"/pfs/work9/workspace/scratch/ma_mkleinma-thesis/trained_models_multilabel/30_bcos_transformer_{args.augmentation}_{samp_text}/seed_{args.seed}"
cm_output_dir = os.path.join(model_output_dir, "confusion_matrix")


# to make sure there is no issue when paths dont exist
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(cm_output_dir, exist_ok=True)

# Load data and splits
data = pd.read_csv(csv_path)
with open(splits_path, 'rb') as f:
    splits = pickle.load(f)

# Define transformations for the datasets --- resize for baselines as model.transform else resizes
if args.augmentation == "no":
    transform = augmentations.get_no_augmentations_no_resize()
elif args.augmentation == "light":
    transform = augmentations.get_light_augmentations_no_resize()
elif args.augmentation == "heavy":
    transform = augmentations.get_heavy_augmentations_no_rotation_resize()

transform_val = augmentations.get_no_augmentations_no_resize()

class_names = [
    "Aortic Enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural Effusion", "Pleural Thickening",
    "Pneumothorax", "Pulmonary Fibrosis"]


fold = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
model = torch.hub.load('B-cos/B-cos-v2', 'vitc_b_patch1_14', pretrained=True)
model[0].linear_head.linear = BcosLinear(in_features=768, out_features=14, bias=False, b=2)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
start_epoch, start_fold, best_f1, best_recall, checkpoint_exists = 0, 0, 0.0, 0.0, False

# we check for latest checkpoint and if it exists, then we load the checkpoint and start from there - else we start from 0 and it does not exist
latest_checkpoint_path, latest_fold = find_latest_checkpoint(model_output_dir)
if latest_checkpoint_path:
    start_epoch, start_fold, best_f1, best_recall, checkpoint_exists = load_checkpoint(
        latest_checkpoint_path, model, optimizer, scheduler)

for current_fold, (train_idx, val_idx) in enumerate(splits):
    fold = current_fold + 1
    print(f"Training fold {fold}...")
    
    if checkpoint_exists and fold < start_fold:
        print(f"Skipping fold {current_fold} as it's already completed.")
        continue  # Skip completed folds
            
    # if we dont start from checkpoint: initialize new model to train
    if not checkpoint_exists or current_fold != start_fold:
        best_f1 = 0.0
        best_recall = 0.0
        model = torch.hub.load('B-cos/B-cos-v2', 'vitc_b_patch1_14', pretrained=True)
        model[0].linear_head.linear = BcosLinear(in_features=768, out_features=14, bias=False, b=2)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        model = model.to(device)
        checkpoint_exists = False


    log_dir = os.path.join(model_output_dir, f"tensorboard_logs_fold_{fold}")
    log_writer = SummaryWriter(log_dir=log_dir)

    # Prepare datasets and dataloaders
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]

    train_dataset = MultiLabelDataset(train_data, image_folder, transform=transform)
    val_dataset = MultiLabelDataset(val_data, image_folder, transform=transform_val)
    
    
    if args.sampling:
        label_counts = train_data.iloc[:, 1:].sum(axis=0).values # how many of each label
        class_weights = 1. / (label_counts + 1e-6) # smoothing factor 0.1
        class_weights = np.clip(class_weights, a_min=1/3, a_max=3.0)
        sample_weights = train_data.iloc[:, 1:].dot(class_weights)
        sample_weights /= sample_weights.max()
        sampler = WeightedRandomSampler(
            weights=sample_weights.to_numpy(),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
    # Train for the current fold
    num_epochs = 10
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
            probs = torch.sigmoid(outputs)  # Convert logits to probabilities
            preds = (probs > 0.5).int()  # Multi-label thresholding at 0.5
            
            # per label accuracy
            correct += (preds == labels).sum().item()  # Count correct predictions across all labels
            total += labels.numel()  # Total number of labels

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
                # create model.transform images
                for img_tensor in images:
                    numpy_image = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    pil_image = Image.fromarray(numpy_image)
                    transformed_image = model.transform(pil_image) # to 224x224 resolution
                    six_channel_images.append(transformed_image)
                
                six_channel_images = torch.stack(six_channel_images).to(device)

                outputs = model(six_channel_images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)  # Convert to probabilities
                preds = (probs > 0.5).int()

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                
                val_correct += (preds == labels).sum().item()  # Total correct predictions across all labels
                val_total += labels.numel()  # Total number of labels (samples * num_labels)


        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        scheduler.step(val_loss)
        

        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        auc = roc_auc_score(all_labels, all_probs, average='macro')
                                    
        log_writer.add_scalar('Loss/Validation', val_loss, epoch+1)
        log_writer.add_scalar('Accuracy/Validation', val_accuracy, epoch+1)
        log_writer.add_scalar('Metrics/Precision', precision, epoch+1)
        log_writer.add_scalar('Metrics/Recall', recall, epoch+1)
        log_writer.add_scalar('Metrics/F1', f1, epoch+1)
        log_writer.add_scalar('Metrics/AUC', auc, epoch+1)
        
        current_lr = optimizer.param_groups[0]['lr']
        log_writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        num_classes = 14 
        cm_per_class = np.zeros((num_classes, 2, 2), dtype=int)  # 14 confusion matrices of size 2x2

        for class_idx in range(num_classes):
            true_labels = all_labels[:, class_idx].astype(int)
            pred_labels = all_preds[:, class_idx].astype(int)

            cm_per_class[class_idx] = confusion_matrix(
                true_labels, pred_labels, labels=[0, 1]
            )

        # Save per-class confusion matrices
        cm_file_path = os.path.join(cm_output_dir, f"confusion_matrices.json")
        with open(cm_file_path, 'w') as cm_file:
            json.dump({
                'confusion_matrices': cm_per_class.tolist(),
                'shape': cm_per_class.shape  
            }, cm_file, indent=4)
        print(f"Per-class confusion matrices saved at {cm_file_path}")
        
        for class_idx in range(num_classes):
            cm = cm_per_class[class_idx]
            class_name = class_names[class_idx]
            print(f"Logging confusion matrix for {class_name}...")

            # Plot confusion matrix
            cm_figure = plot_confusion_matrix(cm, ['Class 0', 'Class 1'])

            # Log the plot to TensorBoard
            log_writer.add_figure(f'Confusion_Matrix/{class_name}', cm_figure, epoch)

        if (recall > best_recall):
            best_recall = recall
            torch.save(model.state_dict(), os.path.join(model_output_dir, f"pneumonia_detection_model_transformer_bcos_bestrecall_{fold}.pth"))
            cm_file_path = os.path.join(cm_output_dir, f"confusion_matrix_best_recall_{fold}.json")
            with open(cm_file_path, 'w') as cm_file:
                json.dump({
                    'confusion_matrices': cm_per_class.tolist(),
                    'shape': cm_per_class.shape  # Preserve array dimensions for later use
                }, cm_file, indent=4)
            print(f"Confusion Matrix for Fold {fold} saved at {cm_file_path}")

        
        if (f1 > best_f1):
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(model_output_dir, f"pneumonia_detection_model_transformer_bcos_bestf1_{fold}.pth"))
            cm_file_path = os.path.join(cm_output_dir, f"confusion_matrix_best_f1_{fold}.json")
            with open(cm_file_path, 'w') as cm_file:
                json.dump({
                    'confusion_matrices': cm_per_class.tolist(),
                    'shape': cm_per_class.shape  # Preserve array dimensions for later use
                }, cm_file, indent=4)            
            print(f"Confusion Matrix for Fold {fold} saved at {cm_file_path}")
            
        save_checkpoint_path = os.path.join(model_output_dir, f"checkpoint_fold_{fold}.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, fold, save_checkpoint_path, best_f1, best_recall)

        print(f"Fold {fold}, Epoch {epoch + 1}/{num_epochs}, "
                f"Val Acc: {val_accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    print(f"Finished training fold {fold}.\n")
    
    # Save the final model
    model_path = f"pneumonia_detection_model_fold_{fold}_transformer_bcos.pth"
    torch.save(model.state_dict(), os.path.join(model_output_dir, model_path))
    log_writer.close()