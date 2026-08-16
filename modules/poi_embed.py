import torch
import torch.nn as nn
from tqdm import tqdm

class POIEmbedderCL(nn.Module):
    def __init__(self, 
        embedding_dim: int, 
        num_pois: int, 
        num_categories: int, 
        location_encoder: nn.Module,
        init_embeds = None,
        num_distr_scales: int = 3
    ):
        """
        POIEmbedder is a class representation of a POI. 
        Each POI is represented by its category, and location. POI id is not included to avoid trivial matches.
        
        Args:
            embedding_dim (int): Dimension of each encoded attribute.
            num_pois (int): Number of unique POIs.
            num_categories (int): Number of unique POI categories.
            location_encoder (nn.Module): Location encoder module to encode POI locations.
        """
        super(POIEmbedderCL, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_pois = num_pois
        self.num_categories = num_categories
        
        # Initilize location encoder
        self.location_encoder = location_encoder
        # Initialize POI embedding
        if init_embeds is None:
            print("Initializing POI embeddings randomly.")
            self.poi_embedding = nn.Embedding(num_pois, embedding_dim)
        else:
            print("Initializing POI embeddings from provided embeddings.")
            embedding_matrix = torch.zeros((num_pois, embedding_dim), dtype=torch.float32)
            for poi_id in range(num_pois):
                if poi_id in init_embeds:
                    emb = init_embeds[poi_id]
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb, dtype=torch.float32)
                    else:
                        emb = emb.clone().detach().to(dtype=torch.float32)
                    embedding_matrix[poi_id] = emb
                else:
                    embedding_matrix[poi_id] = torch.randn(embedding_dim)
            self.poi_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # Initialize category embedding
        self.cat_embedding = nn.Embedding(num_categories, embedding_dim)
        # Text embedding linear layer to match dimensions
        self.text_embedding_fc = nn.Linear(768, embedding_dim)
        
        # Used to predict the usage distribution for each POI
        self.poi_usage_distribution_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 100),
            nn.Softmax(dim=-1)
        )
        
        # Parameters to learn the alpha values for the mixture of distributions
        self.alpha_param = nn.Embedding(num_pois, num_distr_scales)
        nn.init.zeros_(self.alpha_param.weight)

    def forward(self, poi_cats: torch.Tensor, poi_locs: torch.Tensor, text_emb: torch.Tensor = None) -> torch.Tensor:
        # Encode categories and locations
        cat_embeds = self.cat_embedding(poi_cats) # (batch_size, seq_len, embedding_dim)
        loc_embeds = self.location_encoder(poi_locs) # (batch_size, seq_len, embedding_dim)

        if text_emb is not None:
            # If text embeddings are provided, use them instead of simple category embeddings
            #text_emb = self.text_embedding_fc(text_emb) # (batch_size, seq_len, embedding_dim)
            combined_embeds = torch.cat([text_emb, loc_embeds], dim=-1)
        else:
            combined_embeds = torch.cat([cat_embeds, loc_embeds], dim=-1)

        return combined_embeds # (batch_size, seq_len, embedding_dim * 2)
    
    def compute_distr_prior_mixer(self, poi_ids: torch.Tensor, transfer_distr: torch.Tensor) -> torch.Tensor:
        """
        Compute the prior distribution for the given POI IDs using a mixture of transfer distributions.
        Args:
            poi_ids (torch.Tensor): Tensor of POI IDs, shape (num_sparse_pois,).
            transfer_distr (torch.Tensor): Tensor of transfer distributions, shape (num_sparse_pois, num_distr_scales, distr_shape).
        """
        # Get the alpha values for the given POI ids
        alpha = self.alpha_param(poi_ids)   # (num_sparse_pois, num_distr_scales)
        # Pass through softmax to get the weights
        alpha = torch.softmax(alpha, dim=-1) # (num_sparse_pois, num_distr_scales)
        
        # Compute the prior distribution as a weighted sum of the transfer distributions
        p_prior = torch.einsum('us,ust->ut', alpha, transfer_distr)  # (num_sparse_pois, distr_shape)
        p_prior = p_prior / (p_prior.sum(dim=-1, keepdim=True) + 1e-12)
        
        return p_prior
    
    def get_poi_embedding(self, poi_ids: torch.Tensor) -> torch.Tensor:
        """
        Get POI embeddings for the given POI IDs.
        """
        return self.poi_embedding(poi_ids)
    
    def usage_distr_predict(self, poi_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict usage distribution for the given POI IDs.
        """
        poi_embeds = self.get_poi_embedding(poi_ids)
        return self.poi_usage_distribution_head(poi_embeds)
    
    def get_all_poi_embeddings_dict(self) -> dict:
        """
        Get all POI embeddings as a dict: {poi_id: embedding}.
        """
        weights = self.poi_embedding.weight.detach().cpu()
        return {poi_id: weights[poi_id] for poi_id in tqdm(range(weights.size(0)), desc="Extracting POI embeddings")}



class POIEmbedderMLM(nn.Module):
    def __init__(self,
                embedding_dim: int, 
                num_pois: int, 
                num_categories: int, 
                location_encoder: nn.Module):
        """
        POIEmbedder is a class representation of a POI. 
        Each POI is represented by its category, and location. POI id is not included to avoid trivial matches.
        
        Args:
            embedding_dim (int): Dimension of each encoded attribute.
            num_pois (int): Number of unique POIs.
            num_categories (int): Number of unique POI categories.
            location_encoder (nn.Module): Location encoder module to encode POI locations.
        """
        super(POIEmbedderMLM, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_pois = num_pois
        self.num_categories = num_categories
        
        self.location_encoder = location_encoder
        self.poi_embedding = nn.Embedding(num_pois, embedding_dim)
        self.cat_embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, poi_ids, poi_cats, poi_locs):
        # Encode categories, locations and POI ids
        poi_embeds = self.poi_embedding(poi_ids) # (batch_size, seq_len, embedding_dim)
        cat_embeds = self.cat_embedding(poi_cats) # (batch_size, seq_len, embedding_dim)
        loc_embeds = self.location_encoder(poi_locs) # (batch_size, seq_len, embedding_dim)
        
        # Combine the encodings
        combined_embeds = torch.cat([cat_embeds, loc_embeds, poi_embeds], dim=-1)
        return combined_embeds # (batch_size, seq_len, embedding_dim * 3)
    
    def get_poi_embedding(self, poi_ids: torch.Tensor) -> torch.Tensor:
        """
        Get POI embeddings for the given POI IDs.
        """
        return self.poi_embedding(poi_ids) # (num_pois, embedding_dim)
    
    def get_all_poi_embeddings_dict(self) -> dict:
        """
        Get all POI embeddings as a dict: {poi_id: embedding}.
        """
        weights = self.poi_embedding.weight.detach().cpu()
        return {poi_id: weights[poi_id] for poi_id in tqdm(range(weights.size(0)), desc="Extracting POI embeddings")}