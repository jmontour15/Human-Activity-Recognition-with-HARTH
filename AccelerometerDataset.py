import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset

class AccelerometerDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe['features'].tolist()  # Convert DataFrame column to list of arrays
        self.labels = dataframe['label'].values     # Store labels as a numpy array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = torch.tensor(self.data[idx], dtype=torch.float32)
        window = window.T # Transpose windows from 50x6 to 6x50 for 6 conv channels
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return window, label