import torch
import numpy as np
from utils.log import thresh_max_f1

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score, roc_auc_score, average_precision_score, 
    mean_absolute_error
)

class FineTuning:
    def __init__(self, args, model, optimizer, criterion, device, task, wandb_logger=None, checkpoint_saver=None, mode="poi"):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.task = task
        self.wandb_logger = wandb_logger
        self.checkpoint_saver = checkpoint_saver
        self.threshold = 0.5
        self.mode = mode  # "poi", "text", or "both"

        # Early stopping trackers
        self.best_loss = float('inf')
        self.patience_counter = 0

    def move_to_device(self, batch: torch.Tensor):
        """
        Move a batch of data to the specified device.
        """
        return (batch[0].to(self.device), 
                batch[1].to(self.device),
                batch[2].to(self.device))

    def early_stop_check(self, val_loss: int, epoch: int):
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
    
    def log(self, metrics: dict, epoch: int):
        """
        Log metrics to WandB.
        """
        if self.wandb_logger:
            for key, value in metrics.items():
                if value is not None:
                    self.wandb_logger.log(key, value, epoch)

    def print(self, metrics: dict, epoch: int):
        """
        Print metrics to the console in a formatted string.
        """
        msg = f"Epoch {epoch}: " + " ".join([f"{k}={v:.4f}" for k, v in metrics.items() if v is not None])
        print(msg)

    def run(self, train_loader, val_loader, epochs: int = 100, verbose: bool = True, patience: int = 10):
        """
        Main fine-tuning loop.
        """
        self.patience_limit = patience
        if self.wandb_logger:
            self.wandb_logger.watch_model(self.model)

        for epoch in range(epochs):
            tloss = self.train_epoch(train_loader)
            eloss = self.eval_epoch(val_loader)

            metrics = {
                "Fine Tuning/Train Loss": tloss['loss'],
                "Fine Tuning/Eval Loss": eloss['loss'],
                "Pretrain/LR_MLM": self.optimizer.param_groups[0]["lr"],
            }

            if self.task == "open_hours":
                metrics.update({
                    "Eval/AUROC": eloss.get("auroc"),
                    "Eval/Precision": eloss.get("precision"),
                    "Eval/Recall": eloss.get("recall"),
                    "Eval/F1": eloss.get("f1_score"),
                })
            elif self.task == "busyness":
                metrics.update({
                    "Eval/MAE": eloss.get("mae"),
                    "Eval/CosineSim": eloss.get("cosine_sim"),
                })
            else:
                metrics.update({
                    "Eval/Accuracy": eloss.get("accuracy"),
                    "Eval/Precision": eloss.get("precision"),
                    "Eval/Recall": eloss.get("recall"),
                    "Eval/F1": eloss.get("f1_score"),
                    "Eval/AUROC": eloss.get("auroc"),
                    "Eval/AUPRC": eloss.get("auprc"),
                })

            self.log(metrics, epoch)
            
            if verbose:
                self.print(metrics, epoch)
            
            if self.early_stop_check(eloss['loss'], epoch):
                print(f"Early stopping triggered at epoch {epoch}")
                break

    def train_epoch(self, loader):
        """
        Runs a single training epoch.
        """
        self.model.train()
        total_loss = 0.0

        for (poi_emb, text_emb), labels in loader:
            poi_emb, text_emb, labels = self.move_to_device((poi_emb, text_emb, labels))
            self.optimizer.zero_grad()

            logits = self.model(poi_emb, text_emb, modality=self.mode)
            loss = self.criterion(logits, labels.float())
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return {"loss": total_loss / len(loader)}

    def eval_epoch(self, loader, best_model=None):
        """
        Runs a single evaluation epoch and computes eval metrics.
        """
        if best_model:
            model = best_model
        else:
            model = self.model
            
        model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for (poi_emb, text_emb), labels in loader:
                # Move to device
                poi_emb, text_emb, labels = self.move_to_device((poi_emb, text_emb, labels))

                # Forward pass
                logits = model(poi_emb, text_emb, modality=self.mode)

                # Compute loss
                loss = self.criterion(logits, labels.float())
                total_loss += loss.item()
                
                # Store logits and labels for thresholding
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(labels.cpu())

        # Concatenate all logits and labels
        y_prob = torch.cat(all_probs)
        y_true = torch.cat(all_labels)
        
        result = {"loss": total_loss / len(loader)}
        
        # Busyness is a regression task, exit before classification threshold calcs
        if self.task == "busyness":
            mae = mean_absolute_error(y_true, y_prob)
            cos = torch.nn.CosineSimilarity(dim=1)(y_prob, y_true).mean().item()

            result.update({
                "mae": mae,
                "cosine_sim": cos
            })
            return result
        
        # Continue for classification tasks
        # Update threshold
        new_thresh = thresh_max_f1(
            y_true=np.array(y_true), y_prob=np.array(y_prob), n_classes=self.args.d_out
        )

        if isinstance(new_thresh, np.ndarray):
            if new_thresh.size == 1:
                self.threshold = float(new_thresh[0])
            else:
                self.threshold = torch.tensor(new_thresh, dtype=y_prob.dtype, device=y_prob.device)
        else:
            self.threshold = float(new_thresh)

        # Compute predictions
        y_pred = (y_prob > self.threshold).int()

        if self.task == "open_hours":
            result.update({
                "hamming_loss": hamming_loss(y_true, y_pred),
                "jaccard_score": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
                "auroc": roc_auc_score(y_true, y_prob, average='macro'),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
            })
        else:
            result.update({
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
                "auroc": roc_auc_score(y_true, y_prob),
                "auprc": average_precision_score(y_true, y_prob),
            })

        return result