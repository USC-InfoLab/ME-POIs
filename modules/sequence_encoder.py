import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch

class SequenceEncoder(nn.Module):
    def __init__(self, dim_model, num_heads, dim_feedforward, dropout, num_layers):
        super(SequenceEncoder, self).__init__()
        
        transformer_layer = TransformerEncoderLayer(
            dim_model, num_heads, dim_feedforward, dropout, batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(
            transformer_layer, num_layers
        )

    def forward(self, src, attn_mask):
        """
        Encode the input sequence.
        """
        if attn_mask is None:
            attn_mask = torch.zeros(src.shape[0], src.shape[1], dtype=torch.bool, device=src.device)
        output = self.transformer_encoder(src, src_key_padding_mask=~attn_mask.bool())
        # check if output contains NaN values
        if torch.isnan(output).any():
            raise ValueError("Output contains NaN values")
            
        return output
