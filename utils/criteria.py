import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import PAD

class Criterion(nn.Module):
    def __init__(self, num_categories, pad_token=PAD, ignore_index=-1):
        """
        Criterion class for computing losses.
        
        Args:
            num_categories (int): Number of unique POI categories.
            pad_token (int): Token ID for padding, used in loss calculations. Default is PAD=0.
            ignore_index (int): Index to ignore in loss calculations, typically used for padding. Default is -1.
        """
        super().__init__()
        self.num_categories = num_categories
        self.spacetime_loss = nn.MSELoss()
        self.category_loss = nn.CrossEntropyLoss(ignore_index=pad_token)
        self.poi_id_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.cos_sim = nn.CosineEmbeddingLoss(margin=0.0)
        self.temperature = 0.1  # For contrastive loss scaling
    
    def mlm_loss(self, pred_vals: tuple, true_vals: tuple, mlm_mask: torch.Tensor):
        """
        Computes MLM loss over masked positions.
        """
        poi_id_pred, location_pred, travel_pred, duration_pred = pred_vals
        poi_id_true, location_true, travel_true, duration_true = true_vals

        mlm_mask = mlm_mask.bool()
        # Loss terms
        loc_loss = self.spacetime_loss(location_pred[mlm_mask], location_true[mlm_mask])
        travel_loss = self.spacetime_loss(travel_pred[mlm_mask], travel_true[mlm_mask].unsqueeze(-1))
        duration_loss = self.spacetime_loss(duration_pred[mlm_mask], duration_true[mlm_mask].unsqueeze(-1))
        poi_id_loss = self.poi_id_loss(poi_id_pred[mlm_mask], poi_id_true[mlm_mask])

        return poi_id_loss + loc_loss + travel_loss + duration_loss
    
    def in_batch_CL(self, poi_logits: torch.Tensor, true_poi: torch.Tensor):
        """
        Loss for POI id prediction, with temperature scaling.
        """
        batch_size, window_size, num_pois = poi_logits.shape
        logits_scaled = (poi_logits / self.temperature).reshape(-1, num_pois)
        true_poi = true_poi.reshape(-1)
        
        assert logits_scaled.shape[0] == true_poi.shape[0], \
            f"Shape mismatch: {logits_scaled.shape} vs {true_poi.shape}"

        return self.poi_id_loss(logits_scaled, true_poi)

    
    def mlm_poi_id_loss(self, poi_id_logits: torch.Tensor, poi_ids_true: torch.Tensor, mlm_mask: torch.Tensor):
        """
        Classification loss for POI ID.
        """
        mlm_mask = mlm_mask.bool()
        return self.poi_id_loss(poi_id_logits[mlm_mask], poi_ids_true[mlm_mask])
    
    def kl_usage_loss(
        self,
        q_theta: torch.Tensor,
        p_prior: torch.Tensor,
        mask: torch.Tensor = None,
        eps: float = 1e-8,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Computes KL divergence loss between predicted and true usage distributions for anchor POIs only.

        Args:
            q_theta: Predicted usage distributions (num_unique_pois, distr_shape)
            anchor_distr: Ground truth distributions for anchor POIs (num_unique_pois, distr_shape)
            anchor_mask: Boolean tensor mask indicating anchor POIs (num_unique_pois,)
            reduction: Reduction method ('mean' or 'sum')

        Returns:
            Scalar KL divergence loss for anchors.
        """
        q_theta = q_theta.clamp(min=eps)
        p_prior = p_prior.clamp(min=eps)

        # Filter only anchor POIs
        if mask is not None:
            q_theta = q_theta[mask]
            p_prior = p_prior[mask]

        if q_theta.numel() == 0:
            return torch.tensor(0.0, device=q_theta.device)

        # KL divergence: D_KL(anchor_true || anchor_pred)
        # = sum_i p_i * log(p_i / q_i) = kl_div(log(q), p)
        kl_loss = F.kl_div(q_theta.log(), p_prior, reduction="batchmean" if reduction == 'mean' else reduction)

        return kl_loss
    
    def cosine_loss(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine embedding loss between two sets of embeddings.
        
        Args:
            e1: First set of embeddings (batch_size, embed_dim)
            e2: Second set of embeddings (batch_size, embed_dim)
        
        Returns:
            Scalar cosine embedding loss.
        """
        target = torch.ones(e1.size(0), device=e1.device)
        return self.cos_sim(e1, e2, target)

