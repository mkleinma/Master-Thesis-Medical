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
from libraries.bcosconv2d import NormedConv2d
import pydicom 
import random
import matplotlib.pyplot as plt

from collections import OrderedDict

from libraries.bcosconv2d import NormedConv2d
from pooling.flc_bcosconv2d import ModifiedFLCBcosConv2d


np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# Paths
csv_path = r"C:\Users\Admin\Documents\rsna-pneumonia-detection-challenge\stage_2_train_labels.csv"
image_folder = r"C:\Users\Admin\Documents\rsna-pneumonia-detection-challenge\stage_2_train_images"
splits_path = r"G:\Meine Ablage\Universität\Master Thesis\Pneumonia\training\splits\splits_balanced.pkl"
model_path_flc = r"C:\Users\Admin\Documents\MasterThesis\results\ResNet50_BCos_FLC_HammingWindow_Fix\seed_0\pneumonia_detection_model_bcos_trans_bestf1_1_25.pth"
model_path_normal = r"C:\Users\Admin\Documents\MasterThesis\results\ResNet_BCos\seed_0\pneumonia_detection_model_resnet_bcos_bestf1_1_26.pth"
data = pd.read_csv(csv_path)
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
        patient_id = row['patientId']
        
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array
        
        image = Image.fromarray(image).convert("RGB")
        tensor_image = TF.to_tensor(image)
                
        return tensor_image, torch.tensor(label, dtype=torch.long), patient_id
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_normal = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
model_normal.fc.linear = NormedConv2d(2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
state_dict = torch.load(model_path_normal, map_location=device)

model_normal.load_state_dict(state_dict)
model_normal = model_normal.to(device)
model_normal.eval()

# Load model
model_flc = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
model_flc.layer2[0].conv2 = ModifiedFLCBcosConv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), b=2, transpose=True)
model_flc.layer2[0].downsample[0] = ModifiedFLCBcosConv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), b=2, transpose=False)

model_flc.layer3[0].conv2 = ModifiedFLCBcosConv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), b=2, transpose=True)
model_flc.layer3[0].downsample[0] = ModifiedFLCBcosConv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), b=2, transpose=False)

model_flc.layer4[0].conv2 = ModifiedFLCBcosConv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), b=2, transpose=True)
model_flc.layer4[0].downsample[0] = ModifiedFLCBcosConv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), b=2, transpose=False)    
model_flc.fc.linear = NormedConv2d(2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=False) # code from B-cos paper reused to adjust network

state_dict = torch.load(model_path_flc, map_location=device)
model_flc.load_state_dict(state_dict)
model_flc = model_flc.to(device)
model_flc.eval()


# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data = pd.read_csv(csv_path)
first_split = splits[0]
val_idx = first_split[1]  # Only use the validation indices from the 5th split
val_data = data.iloc[val_idx]
val_dataset = PneumoniaDataset(val_data, image_folder, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

i = 0
with torch.no_grad():
    for images, labels, patient_ids in val_loader:
        images, labels = images.to(device), labels.to(device)
        six_channel_images = []
        for img_tensor in images:
            numpy_image = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(numpy_image)
            transformed_image = model_normal.transform(pil_image)
            six_channel_images.append(transformed_image)
        six_channel_images = torch.stack(six_channel_images).to(device)
        outputs = model_normal(six_channel_images)  # Logits
        probs = torch.softmax(outputs, dim=1)  # Probabilities
        preds = torch.argmax(probs, dim=1)  # Binary predictions
        for image, patient_id in zip(six_channel_images, patient_ids):
          i += 1
          if i > 50:
            break
          image = image[None]
          expl_normal = model_normal.explain(image)
          filename = f"{patient_id}_normal_explanation.png"
          plt.imshow(expl_normal["explanation"])
          plt.axis('off')
          plt.show()
          image_path = os.path.join(r"C:\Users\Admin\Documents\MasterThesis\comparison_images\FLCWithHammingWindow_fix2", filename)
          plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
          plt.close()
          
          expl_flc = model_flc.explain(image)
          filename = f"{patient_id}_flc_explanation2.png"
          plt.imshow(expl_flc["explanation"])
          plt.axis('off')
          plt.show()
          image_path = os.path.join(r"C:\Users\Admin\Documents\MasterThesis\comparison_images\FLCWithHammingWindow_fix2", filename)
          plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
          plt.close()



