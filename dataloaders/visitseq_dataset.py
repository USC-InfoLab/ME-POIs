import torch
import pandas as pd
import numpy as np
from typing import List, Dict
from torch.utils.data import Dataset
from utils.modeling import generate_mlm_mask
from utils.constants import PAD
from torch.nn.utils.rnn import pad_sequence

class VisitSequencesDataset(Dataset):
    """Reads the visits dataset and creates sequences of visits."""
    def __init__(
        self, 
        data: pd.DataFrame, 
        window_size: int,
        dim_embed: int = 768,
        text_emb_path: str = None
    ):
        """
        Initialize the dataset.
        Args:
            data (pd.DataFrame): The input data containing visit sequences.
            window_size (int): The size of sequence window.
            text_emb_path (str, optional): Path to precomputed text embeddings for POIs. Defaults to None.
        """
        self.data = data
        self.window_size = window_size
        if text_emb_path:
            self.text_embeddings = torch.load(text_emb_path)
            # add a dummy embedding for PAD token
            self.text_embeddings[PAD] = torch.zeros(dim_embed)
        else:
            self.text_embeddings = None

        # Store user indices instead of full sequences
        self.user_sequences = []
        self.data = self.data.sort_values(['user_id', 'arrival_time'])
        for _, user_data in self.data.groupby('user_id'):
            visit_indices = user_data.index.tolist()
            total_visits = len(visit_indices)
            for start in range(0, total_visits - self.window_size + 1, self.window_size): # TODO: Large window size tends to skip some POIs
                end = start + self.window_size
                window = visit_indices[start:end]
                # if len(window) < 5:  # ensure at least 5 visits in the window
                #     continue
                self.user_sequences.append(window)

    def __len__(self) -> int:
        return len(self.user_sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        indices = self.user_sequences[idx]
        sequence = self.data.iloc[indices]
        if self.text_embeddings is not None:
            # extract text embeddings based on place_id
            text_emb = torch.stack([self.text_embeddings[pid] for pid in sequence['place_id'].values])
        else:
            text_emb = torch.zeros((len(sequence), 1))  # dummy tensor if no text embeddings
        
        user_ids = torch.tensor(sequence['user_id'].values, dtype=torch.long)

        return {
            'place_id': torch.tensor(sequence['place_id'].values, dtype=torch.long),
            'user_id': user_ids,
            'location': torch.tensor(sequence[['lat', 'lon']].values, dtype=torch.float32),
            'category': torch.tensor(sequence['category_id'].values, dtype=torch.long),
            'arrival_time_w': torch.tensor(sequence['arrival_time_day_of_week'].values, dtype=torch.float32),
            'arrival_time_h': torch.tensor(sequence['arrival_time_hour_of_day'].values, dtype=torch.float32),
            'departure_time_w': torch.tensor(sequence['departure_time_day_of_week'].values, dtype=torch.float32),
            'departure_time_h': torch.tensor(sequence['departure_time_hour_of_day'].values, dtype=torch.float32),
            'travel_time': torch.tensor(sequence['travel_time'].values, dtype=torch.float32),
            'duration': torch.tensor(sequence['duration'].values, dtype=torch.float32),
            'attention_mask': torch.ones(len(sequence), dtype=torch.bool),
            'open_hours': torch.stack(
                [torch.tensor(x, dtype=torch.long) for x in sequence['open_hours'].values]
            ),
            'is_closed': torch.tensor(sequence['is_closed'].values, dtype=torch.long),
            'text_emb': text_emb
        }

def collate_visit_sequences(batch: List[Dict]) -> Dict:
    """Pads each field in a batch of visit sequences to the same length (pad=0 for all fields)."""
    fields = batch[0].keys()
    return {
        field: pad_sequence([item[field] for item in batch], batch_first=True, padding_value=PAD)
        for field in fields
    }