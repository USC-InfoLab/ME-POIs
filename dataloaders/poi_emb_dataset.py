import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union
import numpy as np

class POIEmbeddingDataset(Dataset):
    def __init__(
        self,
        poi_ids: np.ndarray,
        labels: Union[np.ndarray, List],
        poi_embedding_dict: Dict[str, np.ndarray],
        text_embedding_dict: Dict[str, np.ndarray]
    ):
        """
        Dataset for POI embeddings and labels.

        Args:
            poi_ids (np.ndarray): Array of POI IDs.
            labels (np.ndarray): Array of labels (scalar or vector).
            embedding_dict (dict): Dict mapping POI ID to embedding vector.
        """
        self.poi_ids = poi_ids
        self.labels = labels
        self.poi_embedding_dict = poi_embedding_dict
        self.text_embedding_dict = text_embedding_dict

    def __len__(self):
        return len(self.poi_ids)

    def __getitem__(self, idx):
        poi_id = self.poi_ids[idx]
        # clone, detach, and convert to tensor
        poi_emb = self.poi_embedding_dict[poi_id.item()].clone().detach()
        text_emb = self.text_embedding_dict[poi_id.item()].clone().detach()

        label = self.labels[idx]
        return (poi_emb, text_emb), label