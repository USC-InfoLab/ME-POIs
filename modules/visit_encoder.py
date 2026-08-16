import torch
import torch.nn as nn
from modules.input_encoder import InputEncoder
from modules.sequence_encoder import SequenceEncoder
from modules.positional_encoder import PositionalEncoding
from utils.constants import PAD

class VisitEncoder(nn.Module):
    """
    VisitEncoder encodes visit sequences to generate POI embeddings.    
    """
    def __init__(self, 
        dim_embed: int, 
        num_heads: int, 
        dim_feedforward: int, 
        dropout: float, 
        num_layers: int, 
        num_pois: int, 
        num_categories: int,
        loc_encoder_type: str = "theory", 
        strategy: str = "CL",
        init_embeds = None,
        args = None,
    ):
        super(VisitEncoder, self).__init__()
        """
        Args:
            dim_embed (int): Dimension of each encoded attribute.
            num_heads (int): Number of attention heads in the transformer.
            dim_feedforward (int): Dimension of the feedforward network in the transformer.
            dropout (float): Dropout rate.
            num_layers (int): Number of transformer layers.
            num_pois (int): Number of unique POIs.
            num_categories (int): Number of unique POI categories.
            loc_encoder_type (str): Type of location encoder to use. Options: "theory
            strategy (str): Pretraining strategy. Options: "CL" for contrastive learning, "MLM" for masked language modeling.
            init_embeds: Pretrained POI embeddings to initialize the POI embedding layer.
            args (dict): Additional arguments for location encoder.
        """

        # Initialize input encoders including Location Encoder, Time Encoder and POI Embedder
        self.input_encoder = InputEncoder(
            dim_embed=dim_embed,
            num_pois=num_pois,
            num_categories=num_categories,
            loc_encoder_type=loc_encoder_type,
            strategy=strategy,
            init_embeds=init_embeds,
            args=args
        )
        
        self.dim_model = dim_embed

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.dim_model, dropout=0.1
        )

        # Transformer sequence encoder
        self.sequence_encoder = SequenceEncoder(
            self.dim_model, num_heads, dim_feedforward, dropout, num_layers
        )
        
        self.W = nn.Linear(dim_embed, dim_embed, bias=False)
        
        self.poi_id_head = nn.Linear(dim_embed, num_pois)

        self.text_tuner = nn.Sequential(
            nn.Linear(768, dim_embed),
            nn.LayerNorm(dim_embed),
            nn.GELU(),
            nn.Linear(dim_embed, dim_embed)
        )

        ## TODO(Maria): Remove
        self.location_head = nn.Sequential(
            nn.Linear(self.dim_model, self.dim_model),
            nn.Linear(self.dim_model, 2)
        )
        self.travel_time_head = nn.Sequential(
            nn.Linear(2*self.dim_model, self.dim_model), # concatenation of POI location and sequence output
            nn.Linear(self.dim_model, 1),
            nn.Softplus(),
        )
        self.duration_head = nn.Sequential(
            nn.Linear(4*self.dim_model, self.dim_model), # concatenation of POI location, arrival time (2 features) and sequence output
            nn.Linear(self.dim_model, 1),
            nn.Softplus(),
        )
                

    def forward(self, batch: dict, use_text_embed: bool = False, mlm_mask: torch.Tensor = None):
        """
        Forward pass through the model.
        """
        if mlm_mask is not None:
            input_embeds = self.input_encoder.mlm_encode(
                batch['place_id'],
                batch['category'],
                batch['location'],
                (batch['arrival_time_h'], batch['arrival_time_w']),
                (batch['departure_time_h'], batch['departure_time_w']),
                mlm_mask
            )
        else:
            input_embeds = self.input_encoder(
                batch['category'],
                batch['location'],
                (batch['arrival_time_h'], batch['arrival_time_w']),
                (batch['departure_time_h'], batch['departure_time_w']),
                batch['text_emb'] if use_text_embed else None
            )

        input_embeds = self.positional_encoding(input_embeds)

        sequence_output = self.sequence_encoder(
            input_embeds,
            batch['attention_mask']
        )
        
        return sequence_output
    
    def poi_emb_sim_logits(self, sequence_output: torch.Tensor, unique_poi_ids: torch.Tensor):
        """
        Pulls visits closer to their POI embeddings.
        
        Returns:
            logits: Tensor of shape (batch_size, window_size, num_unique_pois)
            where each logit corresponds to the similarity score between
            each visit embedding in the sequence and the POI embeddings.
        """
        # Get POI embeddings for the unique POI ids in the batch
        poi_embeddings = self.input_encoder.poi_embedder.get_poi_embedding(unique_poi_ids)
        # Linear projection and transpose for dot product
        poi_embeddings = self.W(poi_embeddings)
        poi_embeddings = poi_embeddings.T

        # Compute similarity logits
        logits = torch.matmul(sequence_output, poi_embeddings)  # (batch_size, window_size, num_unique_pois)
        
        return logits

    def text_poi_emb_align(self, batch: dict, unique_poi_ids: torch.Tensor):
        """
        Align text embeddings with POI embeddings.
        """
        # Get POI embeddings for the unique POI ids in the batch
        poi_embeddings = self.input_encoder.poi_embedder.get_poi_embedding(unique_poi_ids)
        flat_place_ids = batch['place_id'].reshape(-1)
        flat_text_emb = batch['text_emb'].reshape(-1, batch['text_emb'].shape[-1])
        # For each unique_poi_id, find the first occurrence in flat_place_ids
        idx = torch.stack([
            (flat_place_ids == pid).nonzero(as_tuple=True)[0][0]
            for pid in unique_poi_ids
        ])
        text_embeddings = self.text_tuner(flat_text_emb[idx])
        
        return text_embeddings, poi_embeddings

    def poi_usage_predict(self, unique_poi_ids: torch.Tensor):
        """
        Predicts the usage distribution for each POI in the batch.
        """
        return self.input_encoder.poi_embedder.usage_distr_predict(unique_poi_ids)

    def mlm_predict(self, sequence_output: torch.Tensor, batch: dict, only_poi_id: bool = True):
        """
        Masked Language Modeling prediction: location, arrival, departure times.
        """
        # Predict POI IDs
        poi_id_logits = self.poi_id_head(sequence_output)
        
        if only_poi_id:
            return poi_id_logits
        
        # Get ground truth values from the batch
        location = batch['location']
        arrival_time_h = batch['arrival_time_h']
        arrival_time_w = batch['arrival_time_w']

        # Preduict location given the sequence output
        location_pred = self.location_head(sequence_output)
        loc_encoding_true = self.input_encoder.poi_embedder.location_encoder(location)

        # Predict travel time given the POI location and sequence output
        loc_context_concat = torch.cat([loc_encoding_true, sequence_output], dim=-1)
        travel_time_pred = self.travel_time_head(loc_context_concat)

        arrival_encoding_true = torch.concat([
            self.input_encoder.arrival_encoder_h(arrival_time_h),
            self.input_encoder.arrival_encoder_w(arrival_time_w)
        ], dim=-1)
        
        # Predict duration given the POI location and arrival time and sequence output
        all_inputs = torch.cat([arrival_encoding_true, loc_encoding_true, sequence_output], dim=-1)
        duration_pred = self.duration_head(all_inputs)
        
        return poi_id_logits, location_pred, travel_time_pred, duration_pred
    
    def get_poi_embeddings(self):
        """
        Get the learned POI embeddings.
        """
        return self.input_encoder.poi_embedder.get_all_poi_embeddings_dict()
    
    def freeze_encoder(self):
        """
        Freezes the encoder components to prevent gradient updates.
        """
        for module in [self.input_encoder, self.positional_encoding, self.sequence_encoder]:
            for param in module.parameters():
                param.requires_grad = False