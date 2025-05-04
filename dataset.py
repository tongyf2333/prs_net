import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class MyDataset(Dataset):
    def __init__(self, preprocessed_dir):
        nearests= np.load(os.path.join(preprocessed_dir, 'nearests.npy'))
        voxels = np.load(os.path.join(preprocessed_dir, 'voxels.npy'))
        sampled = np.load(os.path.join(preprocessed_dir, 'sampled.npy'))
        self.nearests = torch.from_numpy(nearests).float()
        self.voxels = torch.from_numpy(voxels).float()
        self.sampled = torch.from_numpy(sampled).float()
        self.data_len=nearests.shape[0]
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.nearests[idx], self.voxels[idx], self.sampled[idx]

if __name__ == "__main__":
    # Example usage
    dataset = MyDataset('preprocessed/chair')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("Dataset loaded successfully.")