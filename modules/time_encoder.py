import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """
    Implementation of https://arxiv.org/pdf/1907.05321.pdf
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(1, embedding_dim)
        self.act = torch.sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (*)
        o: (*, embedding_dim)
        """
        x = x.unsqueeze(-1)  # (*, 1)
        h = self.linear(x)  # (*, embedding_dim)
        # Inplace operation causes RuntimeError for gradient computation. Concat is necessary.
        encoding = torch.concat([h[..., :1], self.act(h[..., 1:])], axis=-1)
        return encoding