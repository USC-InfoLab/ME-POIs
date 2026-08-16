import torch
import torch.nn as nn
from tqdm import tqdm
from utils.modeling import reindex_true_poi_ids, get_unique_poi_ids, get_usage_distr_targets

class CL_Pretrainer:
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module,
        anchor_distr: dict,
        sparse_distr: dict,
        sparse_text_sim_distr: dict,
        sigmas: list[float],
        device: torch.device,
        log_vars: torch.nn.Parameter = None,
        use_text_emb: bool = False,
        wandb_logger: any = None, 
        checkpoint_saver: any = None,
    ):
        """
        Contrastive Learning Pretrainer for POI embeddings.
        Args:
            model (nn.Module): The VisitEncoder model to be pretrained.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss function.
            anchor_distr (dict): Anchor POI distribution for usage prediction.
            sparse_distr (dict): Sparse POI distribution for usage prediction.
            sparse_text_sim_distr (dict): Sparse POI distribution based on text similarity for usage prediction.
            sigmas (list[float]): List of Gaussian sigmas for sparse POIs.
            device (torch.device): Device to run the training on.
            log_vars (torch.nn.Parameter): Learnable log variance parameters for loss weighting.
            use_text_emb (bool): Whether to use text embeddings as input.
            wandb_logger: Weights & Biases logger for experiment tracking.
            checkpoint_saver: Utility to save model checkpoints.
        """
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.sparse_distr = sparse_distr
        self.sparse_text_sim_distr = sparse_text_sim_distr
        self.anchor_distr = anchor_distr
        self.sigmas = sigmas
        self.device = device
        self.use_text_emb = use_text_emb
        self.wandb_logger = wandb_logger
        self.checkpoint_saver = checkpoint_saver
        self.log_vars = log_vars  # Learnable log variance parameters for loss weighting
        if self.log_vars is None:
            print("Losses will not be scaled with learned weights.")
            self.loss_scaling = False
        else:
            print("Losses will be scaled with learned weights.")
            self.loss_scaling = True

        # Early stopping trackers
        self.best_loss = float('inf')
        self.patience_counter = 0

    def move_to_device(self, batch: torch.Tensor):
        return {k: v.to(self.device) for k, v in batch.items()}

    def compute_weighted_loss(
        self, 
        poi_loss: torch.Tensor, 
        kl_anchor_loss: torch.Tensor, 
        kl_sparse_loss: torch.Tensor
    ) -> torch.Tensor:
        # Clamp log_vars to avoid instability
        log_vars_clamped = torch.clamp(self.log_vars, min=-1.0, max=1.0)

        log_var_poi = log_vars_clamped[0]
        log_var_kl_anchor = log_vars_clamped[1]
        log_var_kl_sparse = log_vars_clamped[2]

        loss = (
            torch.exp(-log_var_poi) * poi_loss + log_var_poi +
            torch.exp(-log_var_kl_anchor) * kl_anchor_loss + log_var_kl_anchor +
            torch.exp(-log_var_kl_sparse) * kl_sparse_loss + log_var_kl_sparse
        )

        # For logging
        weights = torch.exp(-log_vars_clamped).detach().cpu().numpy()
        return loss, weights

    def log(self, metrics: dict[str, float], epoch: int):
        if self.wandb_logger:
            for key, value in metrics.items():
                if value is not None:
                    self.wandb_logger.log(key, value, epoch)

    def print(self, metrics: dict[str, float], epoch: int):
        msg = f"Epoch {epoch}: " + " ".join([f"{k}={v:.4f}" for k, v in metrics.items() if v is not None])
        print(msg)
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def early_stop_check(self, val_loss: float, epoch: int) -> bool:
        """
        Update best loss and patience counter.
        """
        if val_loss is None:
            return False  # Nothing to check
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            # Save the model checkpoint
            self.checkpoint_saver.save(epoch, self.model, self.optimizer, val_loss)
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience_limit

    def run(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader = None,
            epochs: int = 100,
            verbose: bool = True,
            patience: int = 10
    ):
        """
        Main pretraining loop.
        """
        self.patience_limit = patience
        if self.wandb_logger:
            self.wandb_logger.watch_model(self.model)

        for epoch in range(epochs):
            tloss = self.run_epoch(train_loader, train=True)
            if val_loader:
                print("Validation...")
                eloss = self.run_epoch(val_loader, train=False)

            # Log metrics
            metrics = {
                "Pretrain/Loss": tloss['total'],
                "Pretrain/POI_Loss": tloss['poi_loss'],
                "Pretrain/KL_Anchor": tloss['kl_anchor'],
                "Pretrain/KL_Sparse": tloss['kl_sparse'],
                #"Pretrain/KL_Txt_Sparse": tloss['kl_txt_sparse'],
                "Pretrain/Align_Loss": tloss['align_loss'],
                "Pretrain/LR": self.optimizer.param_groups[0]["lr"],
            }
            
            if val_loader:
                metrics.update({
                    "Eval/Loss": eloss['total'],
                    "Eval/POI_Loss": eloss['poi_loss'],
                    "Eval/KL_Anchor": eloss['kl_anchor'],
                    "Eval/KL_Sparse": eloss['kl_sparse'],
                    #"Eval/KL_Txt_Sparse": eloss['kl_txt_sparse'],
                    "Eval/Align_Loss": eloss['align_loss'],
                })
            
            self.log(metrics, epoch)
            
            # Print metrics
            if verbose:
                self.print(metrics, epoch)

            # Early stopping check
            if self.early_stop_check(tloss['total'], epoch):
                print(f"Early stopping triggered at epoch {epoch}")
                break

    def run_epoch(self, visit_seq_loader: torch.utils.data.DataLoader, train: bool = True):
        total_loss = 0
        total_poi_loss = 0
        total_kl_anchor_loss = 0
        total_kl_sparse_loss = 0
        total_align_loss = 0
        self.model.train() if train else self.model.eval()

        context = torch.no_grad() if not train else torch.enable_grad()
        with context:
            for batch in tqdm(visit_seq_loader, desc="Training" if train else "Evaluating"):
                batch = self.move_to_device(batch)
                
                if train:
                    self.optimizer.zero_grad()

                # Encode visit sequence
                seq_enc = self.model(batch, self.use_text_emb, mlm_mask=None) # (batch_size, seq_len, dim_model)
                # Contrastive Learning between visits and POI representations
                unique_poi_ids = get_unique_poi_ids(batch['place_id']) # (num_unique_pois,)
                # We compute the logits only for the unique POIs in the batch
                logits = self.model.poi_emb_sim_logits(seq_enc, unique_poi_ids) # (batch_size, seq_len, num_unique_pois)
                # Re-index poi ids to match the logits shape for loss calculation
                reindexed_poi_ids = reindex_true_poi_ids(batch['place_id'], unique_poi_ids) # (batch_size, seq_len)
                # Compute contrastive learning loss
                poi_loss = self.criterion.in_batch_CL(logits, reindexed_poi_ids)
                
                # Predict usage distribution for each POI in the batch (q_theta)
                q_theta = self.model.poi_usage_predict(unique_poi_ids) # (num_unique_pois, dist_shape)
                # Get target usage distributions (p) and masks for anchors and sparse POIs
                anchors_p_prior, anchors_mask, sparse_p_prior, sparse_mask, sparse_text_sim_p_prior = ( 
                    get_usage_distr_targets(
                        unique_poi_ids, 
                        self.anchor_distr, 
                        self.sparse_distr, 
                        self.sparse_text_sim_distr, 
                        self.sigmas, 
                        device=self.device
                    )
                )
                # For anchor POIs, compute prior using the precomputed anchor distributions
                anchors_kl_loss = self.criterion.kl_usage_loss(q_theta, anchors_p_prior, anchors_mask) 
                # For sparse POIs, compute prior using the mixture of transfer distributions
                sparse_p_prior = self.model.input_encoder.poi_embedder.compute_distr_prior_mixer(
                    unique_poi_ids[sparse_mask], sparse_p_prior[sparse_mask]
                ) if sparse_mask.any() else torch.zeros((0, q_theta.size(1)), device=self.device)
                 
                sparse_kl_loss = self.criterion.kl_usage_loss(q_theta[sparse_mask], sparse_p_prior)
                # For sparse POIs, compute prior using the text-similarity based distribution
                #sparse_txt_kl_loss = self.criterion.kl_usage_loss(q_theta, sparse_text_sim_p_prior, sparse_mask)

                # # Compute cosine similarity between text and poi embeddings to monitor alignment
                e_text, e_poi = self.model.text_poi_emb_align(batch, unique_poi_ids)
                align_loss = self.criterion.cosine_loss(e_text, e_poi)

                loss = poi_loss + sparse_kl_loss #+ anchors_kl_loss + sparse_kl_loss + align_loss

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                total_poi_loss += poi_loss.item()
                total_kl_anchor_loss += anchors_kl_loss.item()
                total_kl_sparse_loss += sparse_kl_loss.item()
                #total_kl_txt_sparse_loss += sparse_txt_kl_loss.item()
                total_align_loss += align_loss.item()

        return {
            "total": total_loss / len(visit_seq_loader),
            "poi_loss": total_poi_loss / len(visit_seq_loader),
            "kl_anchor": total_kl_anchor_loss / len(visit_seq_loader),
            "kl_sparse": total_kl_sparse_loss / len(visit_seq_loader),
            # "kl_txt_sparse": total_kl_txt_sparse_loss / len(visit_seq_loader),
            "align_loss": total_align_loss / len(visit_seq_loader)
        }

    def save_embeddings(self, path: str, save_text_emb: bool = False, loader = torch.utils.data.DataLoader):
        """
        Store the learned POI embeddings to a file.
        """
        self.model.eval()
        embeddings_dict = self.model.get_poi_embeddings()
        # Save it as pt
        torch.save(embeddings_dict, path)
        print(f"Saved POI embeddings to {path}")