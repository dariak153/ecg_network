import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, sig_file, mask_file):
        self.signals = np.load(sig_file, allow_pickle=True)
        self.masks = np.load(mask_file, allow_pickle=True)

    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32)
        mask = np.expand_dims((mask > 0).astype(np.float32), axis=-1)
        if signal.ndim == 1:
            signal = np.expand_dims(signal, axis=-1)
        return torch.tensor(signal), torch.tensor(mask)
