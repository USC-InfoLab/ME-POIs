import torch
import torch.nn as nn

class EmbedTuner(nn.Module):
    def __init__(
        self,
        poi_embed_dim: int,
        text_embed_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout_rate: float = 0.0
    ):
        """
        Fine Tuning class for mobility poi embeddings and text embeddings.
        Args:
            poi_embed_dim: Dimension of the input poi embeddings.
            text_embed_dim: Dimension of the input text embeddings.
            num_classes: Number of output classes.
            hidden_dim: Dimension of the hidden layers.
            dropout_rate: Dropout rate for regularization.
        """
        super(EmbedTuner, self).__init__()

        self.text_ft_model = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.poi_ft_model = nn.Sequential(
            nn.Linear(poi_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.poi_single_emb = nn.Sequential(
            nn.Linear(poi_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

        self.text_single_emb = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        self.final_emb = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        poi_emb: torch.Tensor,
        text_emb: torch.Tensor,
        modality: str = "poi"
    ) -> torch.Tensor:
        """
        Forward pass for fine-tuning. 
        Args:
            x1: Input tensor for POI embeddings.
            x2: Input tensor for text embeddings (optional).
            modality: Modality type ("poi", "text", or "both").
        """
        if modality == "poi":
            x = self.poi_single_emb(poi_emb)
        elif modality == "text":
            x = self.text_single_emb(text_emb)
        elif modality == "both":
            x_poi = self.poi_ft_model(poi_emb)
            x_text = self.text_ft_model(text_emb)
            x = self.final_emb(torch.cat((x_poi, x_text), dim=1))
        return x
