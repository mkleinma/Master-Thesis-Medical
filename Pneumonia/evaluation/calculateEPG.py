from libraries.energyPointGame import energy_point_game
from libraries.bcosconv2d import NormedConv2d


import random
import numpy as np
import torch
import pandas as pd
import pydicom
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from libraries.bcosconv2d import NormedConv2d
#from blurpool.blur_bcosconv2d import ModifiedBcosConv2d
from pooling.flc_bcosconv2d import ModifiedFLCBcosConv2d
from libraries.bcoslinear import BcosLinear
import argparse


parser = argparse.ArgumentParser(description='Calculate Energy-based Pointing Game')
parser.add_argument('--model_subpath', type=str, required=True, help='Path to the model')
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
args = parser.parse_args()

BASE_PATH = "/pfs/work7/workspace/scratch/ma_mkleinma-thesis/trained_models/"

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

original_width, original_height = 1024, 1024
explanation_width, explanation_height = 224, 224

image_folder = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/rsna-pneumonia-detection-challenge/stage_2_train_images"
csv_path_splits = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/training_splits/grouped_data.csv"
csv_path = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
splits_path = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/training_splits/splits_balanced_fix.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv(csv_path)
data_splits = pd.read_csv(csv_path_splits)

with open(splits_path, 'rb') as f:
    splits = pickle.load(f)

# Loop over whole validation set of first fold 

### alternative in new models
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
        patient_id = row['patientId']

        # Load DICOM file and process it into RGB format
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array
        image = Image.fromarray(image).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), patient_id


transform = transforms.Compose([
    transforms.ToTensor()  # Normalize with ImageNet stats
])


''' 
transform = no_augmentations() 
val_dataset = PneumoniaDataset(val_data, image_folder, transform=transform)

'''

scale_x = explanation_width / original_width
scale_y = explanation_height / original_height

avg_proportions = []
avg_proportions_incorrect = []
avg_proportions_correct = []

for fold in range(1,6):
    
    split = splits[fold-1] # fold selection
    val_idx = split[1]  # Only use the validation indices from the first fold
    val_data = data_splits.iloc[val_idx]
    val_dataset = PneumoniaDataset(val_data, image_folder, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


    proportions = []
    proportions_correct = []
    proportions_incorrect = []
    count_correct = 0
    count_incorrect = 0
    print(f"Original model subpath from args: {args.model_subpath}")

    model_subpath = args.model_subpath.replace("{fold}", str(fold))


    model_path = os.path.join(BASE_PATH, model_subpath)
    
    print(f"Formatted path for fold {fold}: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for fold {fold} at: {model_path}")

    model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
    model.fc.linear = NormedConv2d(2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=False) # code from B-cos paper reused to adjust network
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    with torch.no_grad():
        for images, labels, patient_ids in val_loader:
            #images, labels = images.to(device), labels.to(device)
            labels = labels.to(device)
            six_channel_images = []
            for img_tensor in images:
                numpy_image = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(numpy_image)
                transformed_image = model.transform(pil_image)
                six_channel_images.append(transformed_image)
                
            six_channel_images = torch.stack(six_channel_images).to(device)
            
            for image, label, patient_id in zip(six_channel_images, labels, patient_ids):
                filtered_rows = data[(data['patientId'] == patient_id) & (data['Target'] == 1)]
                if not filtered_rows.empty: 
                    image = image[None]
                    output = model(image)
                    #prediction = torch.argmax(output, dim=1)
                    expl = model.explain(image)
                    prediction = expl['prediction']
                    contribution_map = expl['contribution_map'].squeeze(0).cpu()
                    contribution_map[contribution_map<0] = 0  
                    proportion = 0.0
                    for _, row in filtered_rows.iterrows():
                        x, y, width, height = round(row["x"] * scale_x), round(row["y"] * scale_y), round(row["width"] * scale_x), round(row["height"] * scale_y)
                        coordinates_list = [x, y, x + width, y + height]
                        coordinates_tensor = torch.tensor(coordinates_list, dtype=torch.int32)
                        ebpg_result = energy_point_game(coordinates_tensor, contribution_map)
                        proportion += ebpg_result
                        
                    proportions.append(proportion)
                    if prediction == 1:
                        proportions_correct.append(proportion)
                        count_correct = count_correct + 1
                    else:
                        proportions_incorrect.append(proportion)
                        count_incorrect = count_incorrect + 1
    if proportions:
        avg_proportion = sum(proportions) / len(proportions)
        avg_proportion_incorrect = sum(proportions_incorrect) / len(proportions_incorrect)
        avg_proportion_correct = sum(proportions_correct) / len(proportions_correct)
        avg_proportions.append(avg_proportion)
        avg_proportions_incorrect.append(avg_proportion_incorrect)
        avg_proportions_correct.append(avg_proportion_correct)

    avg_proportion = round(avg_proportion.item(), 4)
    avg_proportion_incorrect = round(avg_proportion_incorrect.item(), 4)
    avg_proportion_correct = round(avg_proportion_correct.item(), 4)

    print(f"Average Energy-Based Pointing Game Proportion (Positive): {avg_proportion}")
    print(f"Average Energy-Based Pointing Game Proportion (Positive) of Incorrectly Classified Images: {avg_proportion_incorrect}, Count: {count_incorrect}", flush=True)
    print(f"Average Energy-Based Pointing Game Proportion (Positive) of Correctly Classified Images: {avg_proportion_correct}, Count: {count_correct}", flush=True)


final_avg_prop = sum(avg_proportions) / len(avg_proportions)
final_avg_prop_incorrect = sum(avg_proportions_incorrect) / len(avg_proportions_incorrect)
final_avg_prop_correct = sum(avg_proportions_correct) / len(avg_proportions_correct)

final_avg_prop = round(final_avg_prop.item(), 4)
final_avg_prop_incorrect = round(final_avg_prop_incorrect.item(), 4)
final_avg_prop_correct = round(final_avg_prop_correct.item(), 4)

print(f"Average Energy-Based Pointing Game Proportion (Positive) over all folds: {final_avg_prop}", flush=True)
print(f"Average Energy-Based Pointing Game Proportion (Positive) of Incorrectly Classified Images over all folds: {final_avg_prop_incorrect}", flush=True)
print(f"Average Energy-Based Pointing Game Proportion (Positive) of Correctly Classified Images over all folds: {final_avg_prop_correct}", flush=True)
    
    
    
avg_proportions = []
avg_proportions_incorrect = []
avg_proportions_correct = []

for fold in range(1,6):
    proportions = []
    proportions_correct = []
    proportions_incorrect = []
    count_correct = 0
    count_incorrect = 0
    model_subpath = args.model_subpath.replace("{fold}", str(fold))

    model_path = os.path.join(BASE_PATH, model_subpath)
    model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
    model.fc.linear = NormedConv2d(2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=False) # code from B-cos paper reused to adjust network
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    with torch.no_grad():
        for images, labels, patient_ids in val_loader:
            #images, labels = images.to(device), labels.to(device)
            labels = labels.to(device)
            six_channel_images = []
            for img_tensor in images:
                numpy_image = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(numpy_image)
                transformed_image = model.transform(pil_image)
                six_channel_images.append(transformed_image)
                
            six_channel_images = torch.stack(six_channel_images).to(device)
            
            for image, label, patient_id in zip(six_channel_images, labels, patient_ids):
                filtered_rows = data[(data['patientId'] == patient_id) & (data['Target'] == 1)]
                if not filtered_rows.empty: 
                    image = image[None]
                    output = model(image)
                    prediction = torch.argmax(output, dim=1)
                    expl = model.explain(image)
                    #prediction = expl['prediction']
                    contribution_map = expl['contribution_map'].squeeze(0).cpu()
                    contribution_map[contribution_map>0] = 0  
                    proportion = 1.0
                    for _, row in filtered_rows.iterrows():
                        x, y, width, height = round(row["x"] * scale_x), round(row["y"] * scale_y), round(row["width"] * scale_x), round(row["height"] * scale_y)
                        coordinates_list = [x, y, x + width, y + height]
                        coordinates_tensor = torch.tensor(coordinates_list, dtype=torch.int32)
                        ebpg_result = energy_point_game(coordinates_tensor, contribution_map)
                        proportion = proportion - ebpg_result
                        
                    proportions.append(proportion)
    if proportions:
        avg_proportion = sum(proportions) / len(proportions)
        avg_proportions.append(avg_proportion)

    avg_proportion = round(avg_proportion.item(), 4)

    print(f"Average Energy-Based Pointing Game Proportion (Negative): {avg_proportion}", flush=True)


final_avg_prop = sum(avg_proportions) / len(avg_proportions)
final_avg_prop = round(final_avg_prop.item(), 4)

print(f"Average Energy-Based Pointing Game Proportion (Negative) over all folds: {final_avg_prop}", flush=True)