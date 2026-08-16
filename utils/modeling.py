import torch
import pandas as pd
import numpy as np
from utils.constants import PAD

def generate_mlm_mask(shape, device, mask_prob):
    """
    Generate a mask for masked language modeling.
    """
    mask = (torch.rand(shape, device=device) < mask_prob)
    
    # Ensure at least one position is masked: Avoids empty tensors during MLM pretraining
    if mask.sum() == 0:
        rand_idx = torch.randint(0, shape, (1,), device=device)
        mask[rand_idx] = 1
    
    return mask.long()


def mask_sequence(encoding, mask_encoding, mask):
    """
    Replace encoding with mask_encoding at masked positions.
    """
    mask = mask.unsqueeze(-1)
    ones = torch.ones_like(mask)
    encoding = encoding * (ones - mask) + mask_encoding * mask
    
    return encoding


def get_unique_poi_ids(poi_ids: torch.Tensor, pad_token=PAD) -> torch.Tensor:
    """
    Get unique POI IDs from the batch POI IDs.
    
    Args:
        poi_ids (torch.Tensor): Tensor of POI IDs, shape (batch_size, seq_len).
        pad_token (int): Token ID for padding, defaults to PAD (0).
    """
    unique_poi_ids = torch.unique(poi_ids)
    unique_poi_ids = unique_poi_ids[unique_poi_ids != pad_token]

    return unique_poi_ids


def reindex_true_poi_ids(true_poi_ids, unique_poi_ids, pad_token=0, ignore_index=-1):
    """
    Map each global POI ID to its index in unique_poi_ids (local batch vocab).
    - Any POI ID that is not in unique_poi_ids is mapped to ignore_index.
    """
    id_to_idx = {pid.item(): idx for idx, pid in enumerate(unique_poi_ids)}
    flat = true_poi_ids.view(-1)
    out = []
    for pid in flat:
        p = pid.item()
        if p == pad_token:
            out.append(ignore_index)
        else:
            out.append(id_to_idx.get(p, ignore_index))

    return torch.tensor(out, dtype=torch.long, device=true_poi_ids.device).view_as(true_poi_ids)

def transfer_usage_distributions(
    sparse_pois_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    sigma: float
) -> pd.DataFrame:
    """
    Transfers usage distributions from anchor POIs to sparse POIs using a single Gaussian kernel.
    
    Parameters:
        sparse_pois_df (pd.DataFrame): Sparse POIs with columns ['place_id', 'lat', 'lon']
        anchors_df (pd.DataFrame): Anchor POIs with columns ['safegraph_place_id', 'lat', 'lon', 'daily', 'weekly']
        sigma (float): Normalized standard deviation of Gaussian kernel (in normalized coord space)
    
    Returns:
        pd.DataFrame: Result with columns ['place_id', 'pred_daily', 'pred_weekly']
    """

    # Extract arrays
    sparse_coords = sparse_pois_df[['lat', 'lon']].to_numpy()  # (N_sparse, 2)
    anchor_coords = anchors_df[['lat', 'lon']].to_numpy()      # (N_anchors, 2)

    # Compute pairwise Euclidean distances (N_sparse, N_anchors)
    dists = np.linalg.norm(sparse_coords[:, None, :] - anchor_coords[None, :, :], axis=-1)

    # Gaussian kernel weights: w_ij = exp(-d_ij^2 / (2σ^2))
    weights = np.exp(-dists**2 / (2 * sigma**2))
    weights_sum = weights.sum(axis=1, keepdims=True)
    
    weights = np.divide(
        weights, weights_sum,
        out=np.zeros_like(weights),  # fill with 0 where denom = 0
        where=weights_sum != 0
    ) # Normalize weights to sum to 1 for each sparse POI

    # Stack anchor distributions (N_anchors, 100)
    daily_stack = np.stack(anchors_df['daily'].to_numpy())  # shape: (N_anchors, 100)
    weekly_stack = np.stack(anchors_df['weekly'].to_numpy())  # shape: (N_anchors, 100)

    # Weighted sum (N_sparse, 100)
    pred_daily = weights @ daily_stack
    pred_weekly = weights @ weekly_stack

    # Construct output
    gauss_w_df = pd.DataFrame({
        'place_id': sparse_pois_df['place_id'].values,
        'pred_daily': list(pred_daily),
        'pred_weekly': list(pred_weekly),
    })
    
    num_skipped = (weights.sum(axis=1) == 0).sum()
    print(f"Skipped {num_skipped} sparse POIs (no close anchors) for σ = {sigma}")

    return gauss_w_df

def get_usage_distr_targets(
    poi_ids: torch.Tensor,
    anchor_map: dict[int, torch.Tensor],
    sparse_multiscale_map: dict[float, dict[int, torch.Tensor]],
    sparse_text_sim_map: dict[int, torch.Tensor],
    sigmas: list[float],
    distr_shape: int = 100,
    device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the ground truth usage distributions and masks for a batch of POI ids.
    Args:
        poi_ids: Tensor of POI ids (num_unique_pois)
        anchor_map: {place_id: usage distribution}
        sparse_multiscale_map: {sigma: {place_id: usage distribution}}
        sparse_text_sim_map: {place_id: usage distribution based on text similarity}
        sigmas: list of float sigmas
        distr_shape: number of bins in the distribution (e.g., 100)
    Returns:
        - anchor_distr: Tensor (num_unique_pois, distr_shape)
        - anchor_mask: Bool Tensor (num_unique_pois,)
        - sparse_distr: Tensor (num_unique_pois, num_sigmas, distr_shape)
        - sparse_mask: Bool Tensor (num_unique_pois,)
        - sparse_text_sim_distr: Tensor (num_unique_pois, distr_shape)
    """    
    # Include only valid POI ids (not PAD)
    valid_poi_ids = poi_ids[poi_ids != PAD].cpu().numpy()
    
    anchor_distr = []
    anchor_mask = []
    sparse_distr = [[] for _ in sigmas]
    sparse_text_sim_distr = []
    sparse_mask = []
    for pid in valid_poi_ids:
        is_anchor = pid in anchor_map
        anchor_mask.append(is_anchor)
        sparse_mask.append(not is_anchor)

        # Get anchor precomputed usage distribution
        if is_anchor:
            anchor_distr.append(anchor_map[pid])
        else:
            anchor_distr.append(torch.zeros(distr_shape))

        # Get transfer distribution from trusted anchors
        # Transfer distribution is multiscale: one per sigma
        for i, sigma in enumerate(sigmas):
            scale_map = sparse_multiscale_map[sigma]
            if not is_anchor and pid in scale_map:
                sparse_distr[i].append(scale_map[pid])
            else:
                sparse_distr[i].append(torch.zeros(distr_shape))
        if not is_anchor and pid in sparse_text_sim_map:
            sparse_text_sim_distr.append(sparse_text_sim_map[pid])
        else:
            sparse_text_sim_distr.append(torch.zeros(distr_shape))

    # Stack into tensors
    anchor_distr = torch.stack(anchor_distr).to(device)  # (num_unique_pois, distr_shape)
    anchor_mask = torch.tensor(anchor_mask, dtype=torch.bool).to(device)  # (num_unique_pois,)
    sparse_mask = torch.tensor(sparse_mask, dtype=torch.bool).to(device)  # (num_unique_pois,)
    sparse_distr = torch.stack([torch.stack(d) for d in sparse_distr], dim=1).to(device)  # (num_unique_pois, num_sigmas, distr_shape)
    sparse_text_sim_distr = torch.stack(sparse_text_sim_distr).to(device)  # (num_unique_pois, distr_shape)

    return anchor_distr, anchor_mask, sparse_distr, sparse_mask, sparse_text_sim_distr

def transfer_usage_by_text_similarity(
    sparse_pois_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    text_embeds: dict,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Transfer usage distributions from anchor POIs to each sparse POI using text embedding similarity.
    Returns a DataFrame with columns ['place_id', 'pred_daily', 'pred_weekly']
    """
    anchor_ids = [pid for pid in anchors_df['place_id'].values if pid != PAD]
    anchor_embeds = np.stack([text_embeds[pid].cpu().numpy() for pid in anchor_ids])
    anchor_daily = [anchors_df[anchors_df['place_id'] == pid]['daily'].values[0] for pid in anchor_ids]
    anchor_weekly = [anchors_df[anchors_df['place_id'] == pid]['weekly'].values[0] for pid in anchor_ids]
    anchor_daily = np.stack(anchor_daily)
    anchor_weekly = np.stack(anchor_weekly)

    results = []
    for pid in sparse_pois_df['place_id'].values:
        if pid == PAD or pid not in text_embeds:
            continue
        sparse_emb = text_embeds[pid].cpu().numpy()
        # Cosine similarity
        sim = anchor_embeds @ sparse_emb / (np.linalg.norm(anchor_embeds, axis=1) * np.linalg.norm(sparse_emb) + 1e-8)
        top_idx = np.argsort(sim)[-top_k:]
        top_sims = sim[top_idx]
        weights = top_sims / (np.sum(top_sims) + 1e-8)
        pred_daily = (weights[:, None] * anchor_daily[top_idx]).sum(axis=0)
        pred_weekly = (weights[:, None] * anchor_weekly[top_idx]).sum(axis=0)
        results.append({'place_id': pid, 'pred_daily': pred_daily, 'pred_weekly': pred_weekly})

    return pd.DataFrame(results)
