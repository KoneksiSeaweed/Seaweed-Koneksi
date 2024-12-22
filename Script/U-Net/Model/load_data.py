import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from torchvision import transforms

class CustomGeoDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.file_paths = image_paths  # List of file paths for geospatial data
        self.labels = label_paths
        self.transform = transform  # Data augmentation/transformations

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = rasterio.open(self.labels[idx])
        label = label.read()
        #label = np.moveaxis(label, 0, 2)
        label = np.asarray(label)
        label = label.astype(np.float32)
        label_tensor = torch.from_numpy(label)

        # Open the geospatial file using Rasterio
        image = rasterio.open(self.file_paths[idx])
        image = image.read()
        #image = np.moveaxis(image, 0, 2)
        image = np.asarray(image)/10000
        image = image.astype(np.float32)
        image_tensor = torch.from_numpy(image)

        return image_tensor, label_tensor

    
