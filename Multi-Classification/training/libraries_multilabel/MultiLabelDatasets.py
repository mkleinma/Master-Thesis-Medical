from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.data = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.data.iloc[idx, 0]
        labels = self.data.iloc[idx, 1:].values.astype('float32')
        
        image_path = os.path.join(self.image_folder, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)


class MultiLabelDatasetID(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.data = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.data.iloc[idx, 0]
        labels = self.data.iloc[idx, 1:].values.astype('float32')
        
        image_path = os.path.join(self.image_folder, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels), image_id
