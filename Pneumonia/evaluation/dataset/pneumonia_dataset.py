import os
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image


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
        tensor_image = TF.to_tensor(image)
        
        if self.transform:
            image = self.transform(image)

        return tensor_image, torch.tensor(label, dtype=torch.long), patient_id
    