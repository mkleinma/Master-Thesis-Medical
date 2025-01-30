# %%
## Supportive Methods

def pred_count(list):
    zeroCount = 0
    oneCount = 0
    
    for element in list:
        if element == 0:
            zeroCount += 1
        elif element == 1:
            oneCount += 1
    
    return zeroCount, oneCount


def overlap_lists(list1, list2):
    count = 0
    for index in range(len(list1)):
        if (list1[index] == list2[index]):
            count += 1
    
    percentage = count/32
    #print(f"Overlap percentage: {percentage}")
    
        
        

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[7:] 
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# %%
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
from torchvision.transforms import functional as TF
from PIL import Image
from bcosconv2d import NormedConv2d
import pydicom 
import random
import matplotlib.pyplot as plt


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

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
        
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array
        
        image = Image.fromarray(image).convert("RGB")
        tensor_image = TF.to_tensor(image)
        six_channel_image = torch.cat([tensor_image, 1-tensor_image], dim=0) 
                
        return six_channel_image, torch.tensor(label, dtype=torch.float32)
    
# Paths
csv_path = r"/home/mkleinma/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
image_folder = r"/home/mkleinma/rsna-pneumonia-detection-challenge/stage_2_train_images"
splits_path = r"/home/mkleinma/training_splits/splits_balanced.pkl"
model_path = r"/home/mkleinma/pneumonia_detection_model_fold_1.pth" # old: bcos_csv

data = pd.read_csv(csv_path)
with open(splits_path, 'rb') as f:
    splits = pickle.load(f)
    

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
model.fc.linear = NormedConv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
state_dict = torch.load(model_path, map_location=device)
#fixed_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




# Evaluate model on the 5th split
def evaluate_model_on_fifth_split(model, data, split, image_folder, transform, device):
    val_idx = split[1]  # Only use the validation indices from the 5th split
    val_data = data.iloc[val_idx]
    val_dataset = PneumoniaDataset(val_data, image_folder, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []
    
    print("Counting probs/preds")
    expl_count = 0
        
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            # Model predictions
            outputs = model(images)  # Logits
            probs = torch.sigmoid(outputs)  # Probabilities
            preds = torch.round(torch.sigmoid(outputs))  # Binary predictions
              
            #zeroCountPred, oneCountPred = pred_count(preds)
            #zeroCountLabel, oneCountLabel = pred_count(labels)
            ## Debug
            #print(f"Prediction counts: {zeroCountPred} and {oneCountPred}")
            #print(f"Label counts: {zeroCountLabel} and {oneCountLabel}")
            
            #overlap_lists(preds, labels)
            

            # Collect predictions and labels
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            
            del outputs, preds, probs
            torch.cuda.empty_cache()

            
        print("Calculating metrics")
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_preds)                
        return cm, precision, recall, f1, auc, accuracy

fifth_split = splits[0]  # Index 0 corresponds to the 1st fold

# Run evaluation
cm, precision, recall, f1, auc, accuracy = evaluate_model_on_fifth_split(
    model=model,
    data=data,
    split=fifth_split,
    image_folder=image_folder,
    transform=transform,
    device=device
)

# Print results
print("Evaluation Metrics for the 5th Split:")
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")




