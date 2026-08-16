import torch
import torch.nn as nn

from modules.location_encoders import get_location_encoder
from modules.time_encoder import Time2Vec
from utils.modeling import mask_sequence
from modules.poi_embed import POIEmbedderCL, POIEmbedderMLM


class InputEncoder(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        num_pois: int,
        num_categories: int,
        loc_encoder_type: str = "theory",
        strategy: str = "CL",
        init_embeds = None,
        args = None,
    ):
        super(InputEncoder, self).__init__()
        """
        InputEncoder encodes the input features of a visit sequence.
        
        Args:
            dim_embed (int): Dimension of each encoded attribute.
            num_pois (int): Number of unique POIs.
            num_categories (int): Number of unique POI categories.
            loc_encoder_type (str): Type of location encoder to use. Options: "theory", "geoclip", "poly2vec".
            strategy (str): Pretraining strategy. Options: "CL" for contrastive learning, "MLM" for masked language modeling.
            args (dict): Additional arguments for location encoder.
        
        """ 
        # We use seperate Time2Vec encoders (4 towers) to make sure
        # that each encoder is optimized for its specific feature.
        self.arrival_encoder_h = Time2Vec(dim_embed)
        self.arrival_encoder_w = Time2Vec(dim_embed)
        self.depart_encoder_h = Time2Vec(dim_embed)
        self.depart_encoder_w = Time2Vec(dim_embed)
        
        # Location encoder for POI locations
        location_encoder = get_location_encoder(loc_encoder_type, dim_embed, args)
        
        # POI embedder combines category, and location embeddings in a wrapper class
        if strategy == "CL":
            self.poi_embedder = POIEmbedderCL(
                dim_embed, num_pois, num_categories, location_encoder, init_embeds
            )
            # Fully connected layer to combine all encodings. 
            # We have 4 encodings: departure time, arrival time, 
            # POI category, and location embedding.
            self.fc = nn.Linear(dim_embed * 4, dim_embed)
        elif strategy == "MLM":
            self.poi_embedder = POIEmbedderMLM(
                dim_embed, num_pois, num_categories, location_encoder
            )
            # Fully connected layer to combine all encodings. 
            # We have 5 encodings: departure time, arrival time, 
            # POI category, location and POI id embeddings.
            self.fc = nn.Linear(dim_embed * 5, dim_embed)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'CL' or 'MLM'.")


        self.mask_embedding = nn.Embedding(1, dim_embed)  # for [<MASK>] token

    def forward(self,
        category: torch.Tensor, 
        location: torch.Tensor, 
        arrival_time: torch.Tensor, 
        departure_time: torch.Tensor,
        text_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Encodes the input sequence. This version is used for contrastive learning (CL).
        """
        # Encode each feature
        poi_embedding = self.poi_embedder(
            category, location, text_embeddings
        ) # (batch_size, seq_len, dim_embed * 3)
        
        # Encode time features
        arr_time_h, arr_time_w = arrival_time
        dep_time_h, dep_time_w = departure_time
        
        # Encode arrival time (batch_size, seq_len, dim_embed)
        arrival_encoding = self.arrival_encoder_h(arr_time_h) + self.arrival_encoder_w(arr_time_w)
        
        # Encode departure time (batch_size, seq_len, dim_embed)
        departure_encoding = self.depart_encoder_h(dep_time_h) + self.depart_encoder_w(dep_time_w)
        
        time_encoding = torch.cat([arrival_encoding, departure_encoding], dim=-1) # (batch_size, seq_len, dim_embed * 2)
        
        # Concatenate encodings for a full contextual visit embedding
        visit_encoding = torch.cat([poi_embedding, time_encoding], dim=-1) # (batch_size, seq_len, dim_embed * 4)
        visit_encoding = self.fc(visit_encoding) # (batch_size, seq_len, dim_embed)

        return visit_encoding
    
    def mlm_encode(self, 
        poi_id: torch.Tensor,
        category: torch.Tensor,
        location: torch.Tensor,
        arrival_time: torch.Tensor,
        departure_time: torch.Tensor,
        mlm_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode the input sequence. This version is used for masked language modeling (MLM).
        """
        # Encode each feature
        poi_embedding = self.poi_embedder(
            poi_id, category, location
        )  # (batch_size, seq_len, dim_embed * 3)
        
        # Encode time features
        arr_time_h, arr_time_w = arrival_time
        dep_time_h, dep_time_w = departure_time
        
        # Encode arrival time (batch_size, seq_len, dim_embed)
        arrival_encoding = self.arrival_encoder_h(arr_time_h) + self.arrival_encoder_w(arr_time_w)
        
        # Encode departure time (batch_size, seq_len, dim_embed)
        departure_encoding = self.depart_encoder_h(dep_time_h) + self.depart_encoder_w(dep_time_w)
        
        time_encoding = torch.cat([arrival_encoding, departure_encoding], dim=-1) # (batch_size, seq_len, dim_embed * 2)
        
        # Concatenate encodings for a full contextual visit embedding
        visit_encoding = torch.cat([poi_embedding, time_encoding], dim=-1) # (batch_size, seq_len, dim_embed * 5)
        visit_encoding = self.fc(visit_encoding) # (batch_size, seq_len, dim_embed)

        # Erase encodings of masked items
        mask_encoding = self.mask_embedding(torch.zeros(1).long().to(location.device))
        visit_encoding = mask_sequence(visit_encoding, mask_encoding, mlm_mask)

        return visit_encoding
