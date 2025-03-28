"""
this file creates a dataset that will fetch embeddings from a curated dataset
"""

import numpy as np
from torch.utils.data import Dataset
import torch
import pickle


class HierarchicalSampleDataset(Dataset):
    """Hierarchical Sample"""
    def __init__(self, 
        indices_file="/ourdisk/hpc/ai2es/luketerry/debug_config/curated_datasets/2r_mul1_100_balanced_selection.npy", 
        embedding_file="/ourdisk/hpc/ai2es/luketerry/npy_laion_embeddings/laion_embeddings_shard_0.npy",
        key_path="/ourdisk/hpc/ai2es/luketerry/npy_laion_embeddings/filenames_shard_0.pkl"
        ):
        # load the array of indices
        self.indices = np.load(indices_file)

        # store memmap of the embeddings
        self.embeddings = np.load(embedding_file, mmap_mode='r')
        self.embeddings_dim = self.embeddings[0].shape[0]

        # load the filepaths
        with open(key_path, "rb") as fp:
            self.filepaths = pickle.load(fp)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get index of sample
        index = self.indices[idx]

        # fetch the sample and return it
        return {
            'embedding': torch.tensor(self.embeddings[index]), 
            'filepath': self.filepaths[index],
            'index': index
            }

if __name__ == "__main__":
    test = HierarchicalSampleDataset()
    print(test[0])